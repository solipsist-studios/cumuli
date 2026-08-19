#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
build_4dgs_dataset.py  –  Assemble a D-NeRF-style dataset for native 4D
Gaussian Splatting training (fudan-zvg/4d-gaussian-splatting "Blender"
loader) from this pipeline's per-camera frame layout.

Input (a dataset root laid out as follows):
    images/<camera_label>/<pattern % frame>     RGB frames, static cameras
    fmasks/<camera_label>/<pattern % frame>     subject masks (white = subject)
    transforms_undistorted_intrinsics.json      nerfstudio-style: ONE shared
        pinhole intrinsics block (fl_x fl_y cx cy w h, zero distortion) +
        frames[] with per-camera OpenGL/Blender c2w transform_matrix

Output (--out):
    images/<camera_label>/<frame>.png    RGBA (mask in alpha), downscaled
    transforms_train.json                D-NeRF format: shared intrinsics
    transforms_test.json                 (scaled to output res) + one entry
                                         per camera x frame with `time` in
                                         seconds; test = --test_cameras
    points3d.ply                         per-frame visual-hull init points
                                         with colour and a `time` property
                                         (the 4DGS loader initialises each
                                         Gaussian's temporal centre from it)

The visual hull is carved per frame by rejection sampling: points must
project inside the subject mask in at least --hull_min_views cameras.
Colours are averaged from a few spread cameras (no occlusion reasoning —
it's an initialisation, densification takes it from there).

Usage:
    python build_4dgs_dataset.py \
        --dataset_root <dataset_root> \
        --out <dataset_root>_4dgs \
        --pattern 'undistorted_Frame{frame:04d}.png' \
        --start 100 --end 187 --fps 24 --downscale 2 \
        --test_cameras Camera_0016
"""

import argparse
import concurrent.futures
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_rig(transforms_path):
    """Shared intrinsics + per-camera COLMAP-convention w2c/c2w matrices."""
    t = json.load(open(transforms_path))
    for key in ('fl_x', 'fl_y', 'cx', 'cy', 'w', 'h'):
        if key not in t:
            sys.exit(f'ERROR: {transforms_path} has no shared {key} — this script '
                     'assumes one pinhole intrinsics block for all cameras')
    cams = {}
    for fr in t['frames']:
        c2w = np.array(fr['transform_matrix'], dtype=np.float64)
        c2w[:3, 1:3] *= -1                    # OpenGL/Blender axes -> COLMAP
        cams[fr['camera_label']] = {
            'c2w_gl': np.array(fr['transform_matrix'], dtype=np.float64),
            'w2c': np.linalg.inv(c2w),
        }
    return t, cams


def project(w2c, fl_x, fl_y, cx, cy, pts):
    """World points [N,3] -> (u, v, in-front mask) in full-res pixels."""
    cam = (w2c[:3, :3] @ pts.T + w2c[:3, 3:4]).T
    z = cam[:, 2]
    front = z > 0.05
    zs = np.where(front, z, 1.0)
    return fl_x * cam[:, 0] / zs + cx, fl_y * cam[:, 1] / zs + cy, front


def in_mask(mask, u, v, front):
    h, w = mask.shape
    ui = np.clip(u, 0, w - 1).astype(np.int32)
    vi = np.clip(v, 0, h - 1).astype(np.int32)
    inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return inside & (mask[vi, ui] > 127)


def load_masks(root, cams, pattern, frame):
    return {label: np.array(Image.open(
        root / 'fmasks' / label / pattern.format(frame=frame)).convert('L'))
        for label in cams}


def hull_votes(cams, intr, masks, pts):
    votes = np.zeros(len(pts), dtype=np.int32)
    for label, cam in cams.items():
        u, v, front = project(cam['w2c'], *intr, pts)
        votes += in_mask(masks[label], u, v, front)
    return votes


def carve_frame(root, cams, intr, pattern, frame, bbox, n_target, min_views,
                color_cams, rng):
    """Visual-hull points + colours for one frame: [K,3] pts, [K,3] rgb."""
    masks = load_masks(root, cams, pattern, frame)
    lo, hi = bbox
    kept = []
    for _ in range(8):                       # rejection-sample in batches
        cand = rng.uniform(lo, hi, size=(n_target * 4, 3))
        good = hull_votes(cams, intr, masks, cand) >= min_views
        kept.append(cand[good])
        if sum(len(k) for k in kept) >= n_target:
            break
    pts = np.concatenate(kept, axis=0)[:n_target]
    if len(pts) == 0:
        return pts.astype(np.float32), np.zeros((0, 3), dtype=np.uint8)

    rgb = np.zeros((len(pts), 3), dtype=np.float64)
    n_seen = np.zeros(len(pts), dtype=np.int32)
    for label in color_cams:
        img = np.asarray(Image.open(
            root / 'images' / label / pattern.format(frame=frame)).convert('RGB'))
        u, v, front = project(cams[label]['w2c'], *intr, pts)
        ok = in_mask(masks[label], u, v, front)
        ui = np.clip(u, 0, img.shape[1] - 1).astype(np.int32)
        vi = np.clip(v, 0, img.shape[0] - 1).astype(np.int32)
        rgb[ok] += img[vi[ok], ui[ok]]
        n_seen[ok] += 1
    rgb /= np.maximum(n_seen, 1)[:, None]
    rgb[n_seen == 0] = 127.0
    return pts.astype(np.float32), rgb.astype(np.uint8)


def write_ply_with_time(path, pts, rgb, times):
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {len(pts)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'property float time\nend_header\n')
    rec = struct.Struct('<fffBBBf')
    with open(path, 'wb') as fp:
        fp.write(header.encode('ascii'))
        for p, c, t in zip(pts, rgb, times):
            fp.write(rec.pack(p[0], p[1], p[2], c[0], c[1], c[2], t))


def convert_image(src_img, src_mask, dst, downscale):
    with Image.open(src_img) as im:
        rgb = im.convert('RGB')
    with Image.open(src_mask) as mm:
        alpha = mm.convert('L')
    if downscale != 1:
        size = (rgb.width // downscale, rgb.height // downscale)
        rgb = rgb.resize(size, Image.LANCZOS)
        alpha = alpha.resize(size, Image.LANCZOS)
    rgb.putalpha(alpha)
    rgb.save(dst, optimize=False)
    return dst


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--transforms', default='transforms_undistorted_intrinsics.json',
                        help='transforms file (relative to dataset root) with shared '
                             'intrinsics + per-camera GT poses')
    parser.add_argument('--pattern', default='undistorted_Frame{frame:04d}.png',
                        help="frame filename pattern under images/<cam>/ and fmasks/<cam>/")
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--fps', type=float, default=24.0,
                        help='maps frame i to time (i-start)/fps seconds (default: 24)')
    parser.add_argument('--downscale', type=int, default=2,
                        help='integer image downscale factor (default: 2)')
    parser.add_argument('--test_cameras', nargs='*', default=[],
                        help='camera labels held out into transforms_test.json')
    parser.add_argument('--hull_points', type=int, default=300_000,
                        help='total visual-hull init points across all frames (default: 300k)')
    parser.add_argument('--hull_min_views', type=int, default=14,
                        help='min cameras whose mask must contain a hull point (default: 14)')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out = Path(args.out)
    frames = list(range(args.start, args.end + 1))
    t, cams = load_rig(root / args.transforms)
    labels = sorted(cams)
    missing = [c for c in args.test_cameras if c not in cams]
    if missing:
        sys.exit(f'ERROR: unknown test camera(s): {missing}')
    print(f'{len(labels)} cameras x {len(frames)} frames | '
          f'test cameras: {args.test_cameras or "none"}')

    intr = (t['fl_x'], t['fl_y'], t['cx'], t['cy'])

    # ── visual-hull bbox discovery on the middle frame ─────────────────────
    rng = np.random.default_rng(0)
    mid = frames[len(frames) // 2]
    masks = load_masks(root, cams, args.pattern, mid)
    centroid = np.mean([np.linalg.inv(c['w2c'])[:3, 3] for c in cams.values()], axis=0)
    span = max(np.ptp([np.linalg.inv(c['w2c'])[:3, 3] for c in cams.values()], axis=0)) or 4.0
    lo0, hi0 = centroid - span, centroid + span
    cand = rng.uniform(lo0, hi0, size=(500_000, 3))
    good = cand[hull_votes(cams, intr, masks, cand) >= args.hull_min_views]
    if len(good) < 100:
        sys.exit(f'ERROR: bbox discovery found only {len(good)} hull points — '
                 'check mask/pose consistency')
    pad = 0.15 * (good.max(0) - good.min(0)) + 0.05
    bbox = (good.min(0) - pad, good.max(0) + pad)
    print(f'  hull bbox (frame {mid}): {np.round(bbox[0], 2)} .. {np.round(bbox[1], 2)} '
          f'({len(good):,} seed points)')

    # ── per-frame hull carving (parallel) ──────────────────────────────────
    per_frame = max(args.hull_points // len(frames), 200)
    color_cams = labels[:: max(len(labels) // 4, 1)][:4]
    print(f'  carving {per_frame:,} points/frame, colours from {color_cams}')
    hull = [None] * len(frames)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(carve_frame, root, cams, intr, args.pattern, f, bbox,
                            per_frame, args.hull_min_views, color_cams,
                            np.random.default_rng(1000 + f)): i
                for i, f in enumerate(frames)}
        for n_done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            hull[futs[fut]] = fut.result()
            if n_done % 20 == 0 or n_done == len(frames):
                print(f'  hull: {n_done}/{len(frames)} frames', flush=True)

    times = np.concatenate([
        np.full(len(p), (f - args.start) / args.fps, dtype=np.float32)
        for (p, _), f in zip(hull, frames)])
    pts = np.concatenate([p for p, _ in hull])
    rgb = np.concatenate([c for _, c in hull])
    out.mkdir(parents=True, exist_ok=True)
    write_ply_with_time(out / 'points3d.ply', pts, rgb, times)
    print(f'  points3d.ply: {len(pts):,} points over [{times.min():.3f}, {times.max():.3f}]s')

    # ── RGBA images (parallel) ─────────────────────────────────────────────
    jobs = []
    for label in labels:
        (out / 'images' / label).mkdir(parents=True, exist_ok=True)
        for f in frames:
            jobs.append((root / 'images' / label / args.pattern.format(frame=f),
                         root / 'fmasks' / label / args.pattern.format(frame=f),
                         out / 'images' / label / f'{f:04d}.png'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(convert_image, s, m, d, args.downscale) for s, m, d in jobs]
        for n_done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            fut.result()
            if n_done % 200 == 0 or n_done == len(jobs):
                print(f'  images: {n_done}/{len(jobs)}', flush=True)

    # ── transforms jsons ───────────────────────────────────────────────────
    ds = args.downscale
    base = {
        'camera_model': 'OPENCV',
        'fl_x': t['fl_x'] / ds, 'fl_y': t['fl_y'] / ds,
        'cx': t['cx'] / ds, 'cy': t['cy'] / ds,
        'w': t['w'] // ds, 'h': t['h'] // ds,
    }
    def entries(cam_labels):
        return [{
            'file_path': f'images/{label}/{f:04d}',
            'time': (f - args.start) / args.fps,
            'transform_matrix': cams[label]['c2w_gl'].tolist(),
        } for label in cam_labels for f in frames]

    train_labels = [c for c in labels if c not in args.test_cameras]
    json.dump({**base, 'frames': entries(train_labels)},
              open(out / 'transforms_train.json', 'w'), indent=1)
    json.dump({**base, 'frames': entries(args.test_cameras or train_labels[:1])},
              open(out / 'transforms_test.json', 'w'), indent=1)
    print(f'  transforms: {len(train_labels)} train cams, '
          f'{len(args.test_cameras) or 1} test cams, done -> {out}')


if __name__ == '__main__':
    main()
