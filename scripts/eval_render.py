#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""eval_render.py - rendered-quality evaluation for .sogst assets.

Decodes a .sogst archive (or a 4D interchange PLY) exactly as the shipping
viewer would, evaluates the temporal model at each test camera's timestamp

    mean(t)  = xyz + v * (t - t_center)
    alpha(t) = sigmoid(opacity) * exp(-0.5 * ((t - t_center) / t_sigma)^2)

rasterizes with gsplat, and scores PSNR / SSIM / LPIPS against ground-truth
frames.  Cameras come from a nerfstudio/blender-style transforms_test.json
(OpenGL c2w, per-frame `time` in seconds).  Ground truth comes from a
directory of frames named after each entry's file_path basename.

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
comparable to the OMG4 trainer's eval on the same test cameras.
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
        t = header['time_min'] + cam.time * tscale
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
        psnrs.append(psnr)
        ssims.append(ssim)
        lpipss.append(lp)
        view_rows.append({'name': cam.name, 'time': float(cam.time),
                          'psnr_db': psnr, 'ssim': ssim, 'lpips': lp})
        print(f'  {cam.name}  t={cam.time:.3f}  PSNR {psnr:6.3f}  SSIM {ssim:.4f}  LPIPS {lp:.4f}')

        if args.dump_dir:
            Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(
                os.path.join(args.dump_dir, cam.name + '.png'))

    if psnrs:
        print(f'MEAN over {len(psnrs)} views:  PSNR {np.mean(psnrs):.3f}  '
              f'SSIM {np.mean(ssims):.4f}  LPIPS {np.mean(lpipss):.4f}')

    if args.report_json:
        report = build_report(view_rows, {
            'model': args.model, 'transforms': args.transforms,
            'gt_dir': args.gt_dir, 'every': args.every,
            'downscale': args.downscale})
        with open(args.report_json, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
