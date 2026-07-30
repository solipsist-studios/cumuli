#!/usr/bin/env python3
"""eval_render.py - rendered-quality evaluation for .omg4 files.

Decodes a v2 or v3 .omg4 exactly as the shipping viewer would, evaluates
the temporal model at each test camera's timestamp

    mean(t)  = xyz + v * (t - t_center)
    alpha(t) = sigmoid(opacity) * exp(-0.5 * ((t - t_center) / t_sigma)^2)

rasterizes with gsplat, and scores PSNR / SSIM / LPIPS against ground-truth
frames.  Cameras come from a nerfstudio/blender-style transforms_test.json
(OpenGL c2w, per-frame `time` in seconds); ground truth comes from a
directory of frames named after each entry's file_path basename.

Run under the `omg4` conda env (gsplat + lpips + torchmetrics + CUDA):

    ~/miniconda3/envs/omg4/bin/python scripts/eval_render.py \
        --model coffee_martini_v3.omg4 \
        --transforms ~/Dev/datasets/n3v/coffee_martini/transforms_test.json \
        --gt-dir ~/Dev/datasets/n3v/coffee_martini/eval_gt_half \
        --downscale 2 --every 10

Numbers are directly comparable to the OMG4 trainer's eval (same test
cameras, same resolution convention: `--downscale 2` matches the trainer's
`resolution: 2`).
"""

import argparse
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sog_pack import decode_webp, fields_from_v2  # noqa: E402
from splat4d_io import OMG4_V2_FIELDS  # noqa: E402


# ---------------------------------------------------------------------------
# v3 archive -> field arrays (mirrors the engine decoder: sog_pack.verify_v3
# covers the same math for scalar attributes; this adds quats and shN)
# ---------------------------------------------------------------------------

def decode_v3_fields(v3_path):
    """Decode a v3 archive into the same field dict fields_from_v2 returns,
    plus a (time_min, time_max, fps) header tuple."""
    zf = zipfile.ZipFile(v3_path)
    meta = json.loads(zf.read('meta.json'))
    n = meta['count']

    if meta.get('streams'):
        groups = []
        if meta['streams']['persistent']:
            groups.append((meta['streams']['persistent'], meta['segments']['persistent']))
        for prefix, seg in zip(meta['streams']['segments'], meta['segments']['list']):
            if prefix:
                groups.append((prefix, seg['range']))
        names = {name.split('/', 1)[1] for name in zf.namelist() if '/' in name}
        tex = {name: np.zeros((n, 4), dtype=np.uint8) for name in names}
        for prefix, (a, b) in groups:
            for name in names:
                tex[name][a:b] = decode_webp(zf.read(f'{prefix}/{name}')).reshape(-1, 4)[:b - a]
        cent_raw = zf.read('shN_centroids.webp') if 'shN_centroids.webp' in zf.namelist() else None
    else:
        tex = {name: decode_webp(zf.read(name)).reshape(-1, 4)[:n]
               for name in zf.namelist() if name.endswith('.webp') and name != 'shN_centroids.webp'}
        cent_raw = zf.read('shN_centroids.webp') if 'shN_centroids.webp' in zf.namelist() else None

    def unsplit16(l, u, mins, maxs):
        q = (u.astype(np.float64) * 256 + l) / 65535.0
        t = np.asarray(mins)[None, :] + q * (np.asarray(maxs)[None, :] - np.asarray(mins)[None, :])
        return np.sign(t) * (np.exp(np.abs(t)) - 1.0)

    fields = {}
    xyz = unsplit16(tex['means_l.webp'][:, :3], tex['means_u.webp'][:, :3],
                    meta['means']['mins'], meta['means']['maxs'])
    fields['x'], fields['y'], fields['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # smallest-three quats: byte planes are the three kept components in
    # [-1/sqrt(2), 1/sqrt(2)], alpha = 252 + index of the dropped (largest)
    # component in wxyz order
    qb = (tex['quats.webp'][:, :3].astype(np.float64) / 255.0 - 0.5) * np.sqrt(2.0)
    mode = tex['quats.webp'][:, 3].astype(np.int64) - 252
    d = np.sqrt(np.clip(1.0 - (qb * qb).sum(axis=1), 0.0, None))
    quat = np.empty((n, 4))
    # engine mapping (GSplatSogIterator): mode 0 -> (a,b,c,d) as x,y,z,w ...
    # expressed in w-first storage below (rot_0 = w)
    a, b, c = qb[:, 0], qb[:, 1], qb[:, 2]
    for m, (w_, x_, y_, z_) in enumerate((
            (d, a, b, c), (a, d, b, c), (a, b, d, c), (a, b, c, d))):
        sel = mode == m
        quat[sel, 0] = w_[sel]
        quat[sel, 1] = x_[sel]
        quat[sel, 2] = y_[sel]
        quat[sel, 3] = z_[sel]
    for i in range(4):
        fields[f'rot_{i}'] = quat[:, i]

    scales_cb = np.asarray(meta['scales']['codebook'])
    for c_ in range(3):
        fields[f'scale_{c_}'] = scales_cb[tex['scales.webp'][:, c_]]

    sh0_cb = np.asarray(meta['sh0']['codebook'])
    for c_ in range(3):
        fields[f'f_dc_{c_}'] = sh0_cb[tex['sh0.webp'][:, c_]]
    alpha = tex['sh0.webp'][:, 3].astype(np.float64) / 255.0
    alpha = np.clip(alpha, 1e-5, 1.0 - 1e-5)
    fields['opacity'] = np.log(alpha / (1.0 - alpha))

    vel = unsplit16(tex['motion_l.webp'][:, :3], tex['motion_u.webp'][:, :3],
                    meta['motion']['mins'], meta['motion']['maxs'])
    fields['vx'], fields['vy'], fields['vz'] = vel[:, 0], vel[:, 1], vel[:, 2]

    fields['t_center'] = np.asarray(meta['trbf']['center']['codebook'])[tex['trbf.webp'][:, 0]]
    fields['t_sigma'] = np.asarray(meta['trbf']['sigma']['codebook'])[tex['trbf.webp'][:, 1]]

    if meta.get('shN') and cent_raw is not None:
        cent = decode_webp(cent_raw)                       # [H, W, 4] RGBA bytes
        h, w = cent.shape[0], cent.shape[1]
        flat = cent.reshape(h * w, 4)
        labels = (tex['shN_labels.webp'][:, 0].astype(np.int64) +
                  (tex['shN_labels.webp'][:, 1].astype(np.int64) << 8))
        codebook = np.asarray(meta['shN']['codebook'])
        coeffs = {1: 3, 2: 8, 3: 15}[meta['shN']['bands']]
        # engine layout: palette entry n occupies texels
        # [(n % 64) * coeffs, (n % 64 + 1) * coeffs) on row n // 64;
        # sh[j*15 + k] = codebook[centroid_bytes[(u + k)*4 + j + v*W*4]]
        u = (labels % 64) * coeffs
        v = labels // 64
        base = v * w + u                                    # texel index of coeff 0
        f_rest = np.zeros((n, 45), dtype=np.float64)
        for k in range(coeffs):
            texel = flat[base + k]                          # [N, 4] bytes
            for j in range(3):
                f_rest[:, j * 15 + k] = codebook[texel[:, j]]
        fields['f_rest'] = f_rest

    time = meta.get('time', {})
    header = {'time_min': time.get('min', 0.0), 'time_max': time.get('max', 0.0),
              'fps': time.get('fps', 30.0), 'count': n}
    return header, fields


def load_model(path):
    if zipfile.is_zipfile(path):
        return decode_v3_fields(path)
    header, fields = fields_from_v2(path)
    return {'time_min': header['time_min'], 'time_max': header['time_max'],
            'fps': header.get('fps', 30.0), 'count': len(fields['x'])}, fields


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------

def load_cameras(transforms_path, downscale, every):
    t = json.load(open(transforms_path))
    cams = []
    for i, f in enumerate(t['frames']):
        if i % every:
            continue
        # intrinsics are global (n3v-style) or per-frame (custom rigs)
        fx = f.get('fl_x', t.get('fl_x')) / downscale
        fy = f.get('fl_y', t.get('fl_y')) / downscale
        cx = f.get('cx', t.get('cx')) / downscale
        cy = f.get('cy', t.get('cy')) / downscale
        w = t.get('w') and int(round(t['w'] / downscale))
        h = t.get('h') and int(round(t['h'] / downscale))
        c2w = np.asarray(f['transform_matrix'], dtype=np.float64)
        # OpenGL c2w (nerfstudio/blender) -> OpenCV: flip the y/z axes
        c2w = c2w.copy()
        c2w[:3, 1:3] *= -1.0
        w2c = np.linalg.inv(c2w)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        name = os.path.basename(f['file_path'])
        cams.append({'w2c': w2c, 'K': K, 'w': w, 'h': h,
                     'time': float(f.get('time', 0.0)), 'name': name})
    return cams


# ---------------------------------------------------------------------------
# render + metrics
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', required=True, help='.omg4 v2 or v3 file')
    ap.add_argument('--transforms', required=True, help='transforms_test.json (OpenGL c2w + time)')
    ap.add_argument('--gt-dir', required=True,
                    help='directory of ground-truth frames named <file_path basename>.png')
    ap.add_argument('--downscale', type=float, default=2.0,
                    help='intrinsics downscale (2 matches the trainer resolution: 2)')
    ap.add_argument('--every', type=int, default=10, help='evaluate every Nth test frame')
    ap.add_argument('--time-scale', type=float, default=None,
                    help='multiply camera times by this to reach model time units '
                         '(default: auto from model duration / max camera time)')
    ap.add_argument('--dump-dir', default=None, help='optionally write rendered PNGs here')
    args = ap.parse_args()

    import torch
    from gsplat import rasterization
    from PIL import Image

    header, fields = load_model(args.model)
    cams = load_cameras(args.transforms, args.downscale, args.every)
    n = header['count']
    print(f'model: {args.model}  splats: {n}  time: [{header["time_min"]:.3f}, '
          f'{header["time_max"]:.3f}]  cams: {len(cams)}')

    max_cam_t = max(c['time'] for c in cams) or 1.0
    duration = header['time_max'] - header['time_min']
    tscale = args.time_scale
    if tscale is None:
        # camera times and model times usually share units; only rescale when
        # the ranges clearly disagree (e.g. normalized training time)
        ratio = duration / max_cam_t if max_cam_t > 0 else 1.0
        tscale = ratio if not (0.8 < ratio < 1.25) else 1.0
        if tscale != 1.0:
            print(f'note: rescaling camera time by {tscale:.4f} to match model range')

    dev = torch.device('cuda')
    to = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    xyz = to(np.stack([fields['x'], fields['y'], fields['z']], axis=1))
    vel = to(np.stack([fields['vx'], fields['vy'], fields['vz']], axis=1))
    quats = to(np.stack([fields[f'rot_{i}'] for i in range(4)], axis=1))       # wxyz
    scales = torch.exp(to(np.stack([fields[f'scale_{i}'] for i in range(3)], axis=1)))
    op_logit = to(fields['opacity'])
    t_center = to(fields['t_center'])
    t_sigma = to(np.maximum(np.abs(fields['t_sigma']), 1e-6))

    sh_degree = 3 if 'f_rest' in fields else 0
    n_sh = (sh_degree + 1) ** 2
    shs = torch.zeros((n, n_sh, 3), dtype=torch.float32, device=dev)
    shs[:, 0, 0] = to(fields['f_dc_0'])
    shs[:, 0, 1] = to(fields['f_dc_1'])
    shs[:, 0, 2] = to(fields['f_dc_2'])
    if sh_degree:
        f_rest = to(fields['f_rest'])                        # [N, 45] channel-major
        shs[:, 1:, :] = f_rest.reshape(n, 3, 15).permute(0, 2, 1)

    import lpips as lpips_mod
    lpips_model = lpips_mod.LPIPS(net='alex').to(dev)
    from torchmetrics.functional import structural_similarity_index_measure as tm_ssim

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    psnrs, ssims, lpipss = [], [], []
    for cam in cams:
        t = header['time_min'] + cam['time'] * tscale
        dt = t - t_center
        means = xyz + vel * dt[:, None]
        alpha = torch.sigmoid(op_logit) * torch.exp(-0.5 * (dt / t_sigma) ** 2)

        gt_path = os.path.join(args.gt_dir, cam['name'] + '.png')
        if not os.path.exists(gt_path):
            print(f'  missing GT {gt_path}, skipping')
            continue
        gt = torch.tensor(np.asarray(Image.open(gt_path), dtype=np.float32) / 255.0,
                          device=dev)[..., :3]
        if cam['w'] is None:
            cam['h'], cam['w'] = int(gt.shape[0]), int(gt.shape[1])
        if gt.shape[:2] != (cam['h'], cam['w']):
            raise SystemExit(f'GT size {tuple(gt.shape[:2])} != render {(cam["h"], cam["w"])}')

        vm = to(cam['w2c'])[None]
        K = to(cam['K'])[None]
        with torch.no_grad():
            img, _, _ = rasterization(
                means, quats, scales, alpha, shs, vm, K, cam['w'], cam['h'],
                sh_degree=sh_degree, render_mode='RGB')
        img = img[0].clamp(0, 1)

        mse = torch.mean((img - gt) ** 2)
        psnr = float(-10.0 * torch.log10(mse))
        chw = img.permute(2, 0, 1)[None]
        gt_chw = gt.permute(2, 0, 1)[None]
        ssim = float(tm_ssim(chw, gt_chw, data_range=1.0))
        lp = float(lpips_model(chw * 2 - 1, gt_chw * 2 - 1))
        psnrs.append(psnr)
        ssims.append(ssim)
        lpipss.append(lp)
        print(f'  {cam["name"]}  t={cam["time"]:.3f}  PSNR {psnr:6.3f}  SSIM {ssim:.4f}  LPIPS {lp:.4f}')

        if args.dump_dir:
            Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(
                os.path.join(args.dump_dir, cam['name'] + '.png'))

    if psnrs:
        print(f'MEAN over {len(psnrs)} views:  PSNR {np.mean(psnrs):.3f}  '
              f'SSIM {np.mean(ssims):.4f}  LPIPS {np.mean(lpipss):.4f}')


if __name__ == '__main__':
    main()
