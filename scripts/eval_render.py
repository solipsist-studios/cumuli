#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""eval_render.py - rendered-quality evaluation for .sogst assets.

Decodes a .sogst archive (or a 4D interchange PLY) exactly as the shipping
viewer would, evaluates the temporal model at each test camera's timestamp

    mean(t)  = xyz + v * (t - t_center)                         (degree 1)
    mean(t)  = xyz + v * (t - t_center) + a * (t - t_center)^2  (degree 2)
    alpha(t) = sigmoid(opacity) * exp(-0.5 * ((t - t_center) / t_sigma)^2)

rasterizes with gsplat, and scores PSNR / SSIM / LPIPS against ground-truth
frames.  Cameras come from a nerfstudio/blender-style transforms_test.json
(OpenGL c2w, per-frame `time` in seconds).  Ground truth comes from a
directory of frames named after each entry's file_path basename.

A view whose entry has a `time_offset` (a camera's sub-frame sync residual)
is rendered at `time + time_offset`, the moment that camera took the photo
(--shutter captured, the default and the only setting comparable to the
photo); --shutter instant renders every view at the shared nominal time.

Run under an environment with gsplat + lpips + torchmetrics + CUDA:

    python scripts/eval_render.py \
        --model splat_4d.sogst \
        --transforms <run>/dataset_4dgs/transforms_test.json \
        --gt-dir <run>/dataset_4dgs/eval_gt_flat --every 10

--downscale defaults to 1, for build_flipbook_4dgs_dataset.py output, whose
transforms already carry output-resolution intrinsics. n3v-style datasets
(for example coffee_martini with eval_gt_half) carry full-resolution
intrinsics beside half-resolution ground truth and need `--downscale 2`,
which matches the trainer's `resolution: 2`. Numbers are directly
comparable to cumuli-trainer's own eval on the same test cameras.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cumuli_core_path  # noqa: E402,F401  (adds deps/cumuli-core/src if needed)
from cumuli_core.cameras import load_transforms  # noqa: E402
# decode_sogst_fields and load_model are re-exported: merge_sogst_segments,
# compare_sogst and the tests import them from here.
from cumuli_core.sogst import decode_sogst_fields, load_model  # noqa: E402,F401


def build_report(views, config):
    """Assemble the --report_json payload from per-view metric rows.

    `views` is a list of {"name", "time", "psnr_db", "ssim", "lpips"} dicts
    in evaluation order. Kept free of torch/gsplat imports so the schema is
    unit-testable in a CPU-only environment."""
    if views:
        mean = {
            "psnr_db": float(sum(v["psnr_db"] for v in views) / len(views)),
            "ssim": float(sum(v["ssim"] for v in views) / len(views)),
            "lpips": float(sum(v["lpips"] for v in views) / len(views)),
        }
    else:
        mean = {"psnr_db": None, "ssim": None, "lpips": None}
    return {"mean": mean, "views": list(views), "config": dict(config)}


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------

def load_cameras(transforms_path, downscale, every):
    """Cameras as dicts (w2c, K, w, h, time, name), for older callers.
    main() uses cumuli_core.cameras.load_transforms directly."""
    return [{'w2c': c.w2c, 'K': c.K, 'w': c.width, 'h': c.height,
             'time': c.time, 'name': c.name}
            for c in load_transforms(transforms_path, downscale, every)]


#: Photo blurs tried, in pixels (Gaussian sigma), and the largest photo shift tried, in pixels each way.
BLUR_SIGMAS = tuple(i * 0.25 for i in range(17))  # 0 .. 4 px
BLUR_MAX_SHIFT = 2


def _gaussian_blur(img, sigma):
    """Separable Gaussian blur of an [H, W, C] tensor, reflect-padded. sigma 0 returns img."""
    import torch
    import torch.nn.functional as F

    if sigma <= 0:
        return img
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    c = img.shape[-1]
    t = img.permute(2, 0, 1)[None]  # [1, C, H, W]
    t = F.pad(t, (radius, radius, 0, 0), mode='reflect')
    t = F.conv2d(t, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    t = F.pad(t, (0, 0, radius, radius), mode='reflect')
    t = F.conv2d(t, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return t[0].permute(1, 2, 0)


def blur_and_shift(photo, render, sigmas=BLUR_SIGMAS, max_shift=BLUR_MAX_SHIFT, threshold=0.02):
    """How blurred a render is against its photo, and how much of that is misregistration.

    Over the subject's box (pixels either image shows, on black), the photo is blurred by each sigma and
    compared with the render (mean absolute difference); the sigma whose blurred photo matches best is the
    render's blur in pixels. A render that is sharp but displaced also prefers a blurred photo, so the photo
    is also shifted by up to max_shift pixels each way: when a shift explains the match with much less blur,
    the cause is misregistration (wrong pose, or a camera rendered at the wrong time), not blur.

    Returns {"blur_px", "aligned_blur_px", "shift_px": [dx, dy], "error", "aligned_error"} or None when the
    box is empty. photo and render are [H, W, 3] tensors in [0, 1] on the same device."""
    import torch

    seen = (photo.max(dim=-1).values > threshold) | (render.max(dim=-1).values > threshold)
    rows, cols = torch.nonzero(seen.any(dim=1)).flatten(), torch.nonzero(seen.any(dim=0)).flatten()
    if rows.numel() == 0:
        return None
    h, w = seen.shape
    m = int(max_shift)
    y0, y1 = max(int(rows[0]), m), min(int(rows[-1]) + 1, h - m)
    x0, x1 = max(int(cols[0]), m), min(int(cols[-1]) + 1, w - m)
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None
    target = render[y0:y1, x0:x1]
    errors = {}  # (sigma, dx, dy) -> error
    for s in sigmas:
        blurred = _gaussian_blur(photo, float(s))
        for dy in range(-m, m + 1):
            for dx in range(-m, m + 1):
                # photo shifted by (dx, dy): render pixel (x, y) is compared with photo pixel (x + dx, y + dy)
                crop = blurred[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
                errors[(float(s), dx, dy)] = float((crop - target).abs().mean())
    still = {s: e for (s, dx, dy), e in errors.items() if dx == 0 and dy == 0}
    blur = min(still, key=still.get)
    best = min(errors, key=errors.get)
    return {"blur_px": blur, "error": still[blur], "aligned_blur_px": best[0], "shift_px": [best[1], best[2]],
            "aligned_error": errors[best]}


def summarize_blur(rows):
    """Means of the per-view blur figures, and how many views a shift explained better than blur alone."""
    rows = [r for r in rows if r]
    if not rows:
        return None
    mean = lambda k: float(sum(r[k] for r in rows) / len(rows))  # noqa: E731
    shifted = sum(1 for r in rows if r["shift_px"] != [0, 0])
    return {"views": len(rows), "blur_px": mean("blur_px"), "aligned_blur_px": mean("aligned_blur_px"),
            "shift_px": float(sum((r["shift_px"][0] ** 2 + r["shift_px"][1] ** 2) ** 0.5 for r in rows) / len(rows)),
            "shifted_views": shifted}


def view_time(cam, shutter='captured'):
    """Seconds to render a view at: its camera's capture time (nominal time plus
    the sub-frame offset; older cumuli_core has no offsets) or the nominal time."""
    if shutter == 'instant':
        return float(cam.time)
    return float(getattr(cam, 'capture_time', cam.time))


# ---------------------------------------------------------------------------
# render + metrics
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', required=True, help='.sogst archive, or a 4D interchange .ply')
    ap.add_argument('--transforms', required=True, help='transforms_test.json (OpenGL c2w + time)')
    ap.add_argument('--gt-dir', required=True,
                    help='directory of ground-truth frames named <file_path basename>.png')
    ap.add_argument('--downscale', type=float, default=1.0,
                    # 1 for build_flipbook_4dgs_dataset.py output, the
                    # standard dataset, whose intrinsics are already at the
                    # output resolution. The n3v-style datasets this was
                    # first written against carry FULL-resolution intrinsics
                    # beside half-resolution ground truth and need 2.
                    help='intrinsics downscale (default 1, for flipbook-built '
                         'datasets; 2 for n3v-style datasets, matching the '
                         'trainer resolution: 2)')
    ap.add_argument('--every', type=int, default=10, help='evaluate every Nth test frame')
    ap.add_argument('--time-scale', type=float, default=None,
                    help='multiply camera times by this to reach model time units '
                         '(default: auto from model duration / max camera time)')
    ap.add_argument('--shutter', choices=('captured', 'instant'), default='captured',
                    help="captured: render each view at its camera's capture time (time + time_offset); "
                         'instant: at the nominal time, as if every camera fired together')
    ap.add_argument('--no-blur', action='store_true',
                    help='skip the blur/misregistration measure (per view: the photo blur and shift that best '
                         'match the render)')
    ap.add_argument('--dump-dir', default=None, help='optionally write rendered PNGs here')
    ap.add_argument('--report_json', default=None,
                    help='write mean + per-view PSNR/SSIM/LPIPS as JSON here')
    args = ap.parse_args()

    import torch
    from PIL import Image

    from cumuli_core.metrics import LPIPS, psnr as psnr_fn, ssim as ssim_fn
    from cumuli_core.render import render
    from cumuli_core.spacetime import from_fields

    header, fields = load_model(args.model)
    cams = load_transforms(args.transforms, args.downscale, args.every)
    n = header['count']
    print(f'model: {args.model}  splats: {n}  time: [{header["time_min"]:.3f}, '
          f'{header["time_max"]:.3f}]  cams: {len(cams)}')

    times = [c.time for c in cams]
    cam_span = max(times) - min(times)
    duration = header['time_max'] - header['time_min']
    tscale = args.time_scale
    if tscale is None:
        # Camera times and model times usually share units. Only rescale when
        # the ranges clearly disagree (for example normalized training time).
        # Compare SPANS, not maxima: with --every large enough that every
        # sampled view lands on the same instant, the maximum carries no
        # information about the units and the old test invented a scale from
        # it.
        if cam_span <= 0.0 or duration <= 0.0:
            tscale = 1.0
        else:
            ratio = duration / cam_span
            tscale = ratio if not (0.8 < ratio < 1.25) else 1.0
        if tscale != 1.0:
            print(f'note: rescaling camera time by {tscale:.4f} to match model range')

    dev = torch.device('cuda')
    # degree = bands: 9, 24 or 45 f_rest columns are degree 1, 2 or 3.
    # from_fields floors |t_sigma| at 1e-6 and lays f_rest out channel-major.
    st = from_fields(fields, dev)
    scales = st.scales
    lpips_metric = LPIPS(verbose=True)

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    psnrs, ssims, lpipss = [], [], []
    view_rows = []
    for cam in cams:
        t = header['time_min'] + view_time(cam, args.shutter) * tscale
        means, alpha = st.slice(t)

        gt_path = os.path.join(args.gt_dir, cam.name + '.png')
        if not os.path.exists(gt_path):
            print(f'  missing GT {gt_path}, skipping')
            continue
        gt = torch.tensor(np.asarray(Image.open(gt_path), dtype=np.float32) / 255.0,
                          device=dev)[..., :3]
        if cam.width is None:
            cam.height, cam.width = int(gt.shape[0]), int(gt.shape[1])
        if gt.shape[:2] != (cam.height, cam.width):
            raise SystemExit(
                f'GT size {tuple(gt.shape[:2])} != render '
                f'{(cam.height, cam.width)} for {cam.name}. --downscale is '
                f'{args.downscale}. A dataset whose transforms already carry '
                'output-resolution intrinsics (anything from '
                'build_flipbook_4dgs_dataset.py) needs --downscale 1; an '
                'n3v-style dataset with full-resolution intrinsics beside '
                'half-resolution ground truth needs --downscale 2.')

        with torch.no_grad():
            # packed=True is gsplat's own default, which this script always
            # used; it keeps scores bit-identical to earlier reports.
            img, _, _ = render(means, st.quats_wxyz, scales, alpha, st.sh, [cam],
                               cam.width, cam.height, sh_degree=st.sh_degree,
                               packed=True)
            img = img[0].clamp(0, 1)
            psnr = float(psnr_fn(img, gt))
            ssim = float(ssim_fn(img, gt))
            lp = float(lpips_metric(img, gt))
            blur = None if args.no_blur else blur_and_shift(gt, img)
        psnrs.append(psnr)
        ssims.append(ssim)
        lpipss.append(lp)
        row = {'name': cam.name, 'time': float(cam.time), 'capture_time': view_time(cam, 'captured'),
               'psnr_db': psnr, 'ssim': ssim, 'lpips': lp}
        if blur:
            row['blur'] = blur
        view_rows.append(row)
        extra = (f'  blur {blur["blur_px"]:.2f} px, {blur["aligned_blur_px"]:.2f} px at shift '
                 f'{blur["shift_px"][0]:+d},{blur["shift_px"][1]:+d}') if blur else ''
        print(f'  {cam.name}  t={cam.time:.3f}  PSNR {psnr:6.3f}  SSIM {ssim:.4f}  LPIPS {lp:.4f}{extra}')

        if args.dump_dir:
            Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(
                os.path.join(args.dump_dir, cam.name + '.png'))

    blur_summary = summarize_blur([r.get('blur') for r in view_rows])
    if psnrs:
        print(f'MEAN over {len(psnrs)} views:  PSNR {np.mean(psnrs):.3f}  '
              f'SSIM {np.mean(ssims):.4f}  LPIPS {np.mean(lpipss):.4f}')
    if blur_summary:
        print(f'BLUR over {blur_summary["views"]} views:  {blur_summary["blur_px"]:.2f} px; '
              f'{blur_summary["aligned_blur_px"]:.2f} px when the photo may shift (mean shift '
              f'{blur_summary["shift_px"]:.2f} px, {blur_summary["shifted_views"]} views shifted)')

    if args.report_json:
        report = build_report(view_rows, {
            'model': args.model, 'transforms': args.transforms,
            'gt_dir': args.gt_dir, 'every': args.every,
            'downscale': args.downscale, 'shutter': args.shutter})
        if blur_summary:
            report['blur'] = blur_summary
        with open(args.report_json, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
