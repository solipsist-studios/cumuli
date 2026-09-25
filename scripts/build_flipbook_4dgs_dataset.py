#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
build_flipbook_4dgs_dataset.py – Assemble a D-NeRF-style 4D training
dataset from a *flipbook* capture layout (frame-major, as produced by the
Diffuman4D preprocessing flow) instead of build_4dgs_dataset.py's
camera-major layout.

Input (--flipbook_root, for example capture_flipbook/):
    frame_NNNN/transforms.json      per-frame nerfstudio transforms: one
        entry per camera with 2-digit camera_label, OpenGL c2w, and
        PER-CAMERA undistorted pinhole intrinsics (zero distortion).
        The rig is static: every frame carries the same cameras. This
        script verifies that and reads the rig from the first frame.
    frame_NNNN/images_flat/<label>.png   undistorted RGB per camera
    frame_NNNN/fmasks_clean/<label>.png  subject masks (white = subject)

Output (--out) mirrors the per-frame refit dataset layout:
    realcams/cam<label>/frame_NNNNN.png  RGBA (mask in alpha), downscaled
    transforms_train.json / transforms_test.json
        per-view entries with per-frame intrinsics (the rotor trainer's
        Blender loader and eval_render.py both support these. There is
        deliberately NO global intrinsics block: the cameras differ)
    points3d.ply                     per-frame visual-hull init points
                                     with colour and per-point `time`
    eval_gt_flat/<name>.png          scored frames composited to RGB over
        black at output resolution, byte-consistent with what the masked
        trainer renders, so eval_render.py scores are honest (an earlier
        eval GT was mis-framed against the renders. This writes GT and
        transforms from the same pixels + numbers)
    evalcams/cam<label>/...          only with --eval_root, below

Two ways to obtain scored views:

  --test_cameras   holds a rig camera out of training and scores it. The
      viewpoint is wherever that camera happens to sit, so scores from two
      different rigs measure different things and cannot be compared.
  --eval_root      a second flipbook of cameras that never train. Every rig
      can be scored against the same fixed novel views, which is what makes
      a camera-configuration comparison meaningful. Rendered rigs use this
      (see run_synthetic_pipeline.py); real captures cannot, since the views
      would have to be physically shot without joining the reconstruction.

Usage:
    python build_flipbook_4dgs_dataset.py \
        --flipbook_root <flipbook_root> \
        --out <out> \
        --fps 30 --downscale 4 --test_cameras 05

    python build_flipbook_4dgs_dataset.py \
        --flipbook_root <run>/flipbook_src --eval_root <run>/eval_src \
        --out <run>/dataset_4dgs --fps 24 --downscale 2 \
        --init_bbox "-0.5,0.0,-0.3,0.5,1.8,0.4"
"""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_4dgs_dataset import (
    project, in_mask, write_ply_with_time, convert_image,
)


def load_flipbook_rig(frame_dirs):
    """Static rig from the first frame's transforms.json, verified against
    the last frame. Returns {label: {c2w_gl, w2c, intr=(fl_x,fl_y,cx,cy),
    w, h}}."""
    def read(frame_dir):
        with open(frame_dir / 'transforms.json') as f:
            t = json.load(f)
        cams = {}
        for fr in t['frames']:
            for k in ('k1', 'k2', 'p1', 'p2'):
                if abs(fr.get(k, 0.0)) > 1e-9:
                    sys.exit(f'ERROR: camera {fr["camera_label"]} has nonzero '
                             f'distortion {k}={fr[k]} — this script expects '
                             'undistorted flipbook images')
            c2w = np.array(fr['transform_matrix'], dtype=np.float64)
            colmap = c2w.copy()
            colmap[:3, 1:3] *= -1                  # OpenGL/Blender -> COLMAP
            cams[fr['camera_label']] = {
                'c2w_gl': c2w,
                'w2c': np.linalg.inv(colmap),
                'intr': (fr['fl_x'], fr['fl_y'], fr['cx'], fr['cy']),
                'w': fr['w'], 'h': fr['h'],
            }
        return cams

    rig = read(frame_dirs[0])
    check = read(frame_dirs[-1])
    if sorted(rig) != sorted(check):
        sys.exit('ERROR: camera set differs between first and last frame')
    for label in rig:
        if not np.allclose(rig[label]['c2w_gl'], check[label]['c2w_gl']) or \
           not np.allclose(rig[label]['intr'], check[label]['intr']):
            sys.exit(f'ERROR: camera {label} moves between frames — rig is not static')
    return rig


def drop_duplicate_frames(frame_dirs, ref_label):
    """Drop frames whose reference-camera image is byte-identical to the
    previous kept frame's, returning (kept_dirs, kept_indices).

    Flipbook captures can carry held frames where the source video dropped
    one (one capture measured 39 unique images across 61 frame dirs).
    Keeping them is worse than redundant supervision. A held pose at two
    timestamps asks the 4D model to render identical output at t and
    t+1/fps while moving on either side, and it can only satisfy that by
    inflating temporal sigma (smearing). Dropping the holds and leaving
    survivors at their ORIGINAL timestamps lets the model interpolate
    smoothly across the gap, which is what physically happened.
    """
    kept_dirs, kept_idx = [], []
    last_hash = None
    for i, d in enumerate(frame_dirs):
        h = hashlib.md5((d / 'images_flat' / f'{ref_label}.png').read_bytes()).hexdigest()
        if h != last_hash:
            kept_dirs.append(d)
            kept_idx.append(i)
        last_hash = h
    return kept_dirs, kept_idx


def load_frame_masks(frame_dir, labels, masks_dir):
    return {label: np.array(Image.open(
        frame_dir / masks_dir / f'{label}.png').convert('L'))
        for label in labels}


def hull_votes(rig, masks, pts):
    votes = np.zeros(len(pts), dtype=np.int32)
    for label, cam in rig.items():
        u, v, front = project(cam['w2c'], *cam['intr'], pts)
        votes += in_mask(masks[label], u, v, front)
    return votes


def carve_frame(frame_dir, rig, masks_dir, bbox, n_target, min_views,
                color_cams, rng):
    """Visual-hull points + colours for one frame: [K,3] pts, [K,3] rgb."""
    labels = sorted(rig)
    masks = load_frame_masks(frame_dir, labels, masks_dir)
    lo, hi = bbox
    kept = []
    for _ in range(8):
        cand = rng.uniform(lo, hi, size=(n_target * 4, 3))
        good = hull_votes(rig, masks, cand) >= min_views
        kept.append(cand[good])
        if sum(len(k) for k in kept) >= n_target:
            break
    pts = np.concatenate(kept, axis=0)[:n_target]
    if len(pts) == 0:
        return pts.astype(np.float32), np.zeros((0, 3), dtype=np.uint8)

    rgb = np.zeros((len(pts), 3), dtype=np.float64)
    n_seen = np.zeros(len(pts), dtype=np.int32)
    for label in color_cams:
        img = np.asarray(Image.open(frame_dir / 'images_flat' / f'{label}.png').convert('RGB'))
        u, v, front = project(rig[label]['w2c'], *rig[label]['intr'], pts)
        ok = in_mask(masks[label], u, v, front)
        ui = np.clip(u, 0, img.shape[1] - 1).astype(np.int32)
        vi = np.clip(v, 0, img.shape[0] - 1).astype(np.int32)
        rgb[ok] += img[vi[ok], ui[ok]]
        n_seen[ok] += 1
    rgb /= np.maximum(n_seen, 1)[:, None]
    rgb[n_seen == 0] = 127.0
    return pts.astype(np.float32), rgb.astype(np.uint8)


def eval_basename(label, frame_index):
    """Name for one eval-camera view.

    eval_render.py finds ground truth by the basename of an entry's
    file_path, so the name has to be unique across cameras as well as
    frames. Held-out rig cameras keep their historical frame_NNNNN naming
    (only one is ever scored at a time); eval cameras carry their label."""
    return f'cam{label}_frame_{frame_index + 1:05d}'


def load_eval_root(eval_root, n_source_frames, frame_idx):
    """Rig and frame dirs for the separate eval cameras, or ({}, []).

    The eval root is a flipbook laid out exactly like the training one, so
    the same loader validates it: static rig, undistorted, per-camera
    intrinsics. It is indexed by the ORIGINAL frame indices the training
    side kept, so --dedupe drops the same instants from both and eval views
    stay aligned with the timestamps they are scored at."""
    if not eval_root:
        return {}, []
    root = Path(eval_root).expanduser()
    dirs = sorted(d for d in root.iterdir()
                  if d.is_dir() and d.name.startswith('frame_'))
    if not dirs:
        sys.exit(f'ERROR: no frame_* directories under {root}')
    if len(dirs) != n_source_frames:
        sys.exit(f'ERROR: --eval_root has {len(dirs)} frame directories but '
                 f'the training flipbook has {n_source_frames}. The two are '
                 'rendered from the same clip and must line up frame for '
                 'frame.')
    kept = [dirs[i] for i in frame_idx]
    return load_flipbook_rig(kept), kept


def flatten_gt(rgba_path, dst):
    """RGBA -> RGB composited over black (what a masked trainer renders)."""
    with Image.open(rgba_path) as im:
        rgba = np.asarray(im.convert('RGBA'), dtype=np.float32)
    rgb = rgba[..., :3] * (rgba[..., 3:4] / 255.0)
    Image.fromarray(rgb.astype(np.uint8)).save(dst, optimize=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--flipbook_root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fps', type=float, default=30.0,
                        help='maps frame i to time i/fps seconds (default: 30)')
    parser.add_argument('--downscale', type=int, default=4,
                        help='integer image downscale factor (default: 4)')
    parser.add_argument('--test_cameras', nargs='*', default=[],
                        help='camera labels held out into transforms_test.json '
                             '(also written flat into eval_gt_flat/). Use ONE for '
                             'eval_render scoring: basenames collide otherwise.')
    parser.add_argument('--holdout_cameras', nargs='*', default=[],
                        help='additional camera labels excluded from training but NOT '
                             'scored. Rigs with stereo pairs need this: holding out one '
                             'camera whose pair-mate is 1-2 degrees away and still in '
                             'training is not a novel-view test at all (pairs on the '
                             'capture this was built against: 00/10, 01/02, 04/05, '
                             '06/07, 08/09). Hold out the mate too.')
    parser.add_argument('--init_bbox',
                        help='"x0,y0,z0,x1,y1,z1" subject bounds in the '
                             'dataset world frame, used instead of searching '
                             'for them. Rejection sampling over the whole rig '
                             'extent finds very few hull points when the '
                             'subject is small next to the camera spread.')
    parser.add_argument('--eval_root',
                        help='a SECOND flipbook root whose cameras are scored '
                             'but never trained on. Written to evalcams/ and '
                             'appended to transforms_test.json + eval_gt_flat '
                             'with per-camera basenames. Use it to score every '
                             'camera configuration against one fixed set of '
                             'novel views: holding out a rig camera instead '
                             'moves the test viewpoint with the rig, so two '
                             'rigs cannot be compared.')
    parser.add_argument('--masks_dir', default='fmasks_clean',
                        help='per-frame mask subdirectory (default: fmasks_clean)')
    parser.add_argument('--dedupe', action='store_true',
                        help='drop frames whose images are byte-identical to the previous '
                             'kept frame (held frames from dropped source video frames). '
                             'Survivors keep their ORIGINAL timestamps, so the model '
                             'interpolates across the gap instead of fitting a hold. '
                             'Holds otherwise force inflated temporal sigma (smearing).')
    parser.add_argument('--hull_points', type=int, default=300_000,
                        help='total visual-hull init points across all frames')
    parser.add_argument('--hull_min_views', type=int, default=9,
                        help='min cameras whose mask must contain a hull point '
                             '(default: 9)')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()

    root = Path(args.flipbook_root).expanduser()
    out = Path(args.out).expanduser()
    frame_dirs = sorted(d for d in root.iterdir()
                        if d.is_dir() and d.name.startswith('frame_'))
    if not frame_dirs:
        sys.exit(f'ERROR: no frame_* directories under {root}')
    rig = load_flipbook_rig(frame_dirs)
    labels = sorted(rig)
    # frame_idx[i] is the ORIGINAL frame_NNNN index of kept frame i. It
    # drives both the timestamp and the output filename, so names stay
    # traceable to source frames and times stay physically correct when
    # --dedupe removes holds.
    frame_idx = list(range(len(frame_dirs)))
    n_source_frames = len(frame_dirs)
    if args.dedupe:
        n_before = len(frame_dirs)
        frame_dirs, frame_idx = drop_duplicate_frames(frame_dirs, labels[0])
        print(f'  dedupe: {n_before} frame dirs -> {len(frame_dirs)} unique '
              f'({n_before - len(frame_dirs)} held frames dropped; survivors keep '
              'their original timestamps)')
    missing = [c for c in args.test_cameras + args.holdout_cameras if c not in rig]
    if missing:
        sys.exit(f'ERROR: unknown test/holdout camera(s): {missing}')
    if len(args.test_cameras) > 1:
        print('WARNING: eval_gt_flat basenames collide across multiple test '
              'cameras; only the last written survives. Use one test camera '
              'for eval_render scoring.')
    print(f'{len(labels)} cameras x {len(frame_dirs)} frames | '
          f'test cameras: {args.test_cameras or "none"}'
          f'{" | holdout: " + ",".join(args.holdout_cameras) if args.holdout_cameras else ""} | '
          f'duration {frame_idx[-1] / args.fps:.3f}s @ {args.fps} fps')

    # ── visual-hull bbox: given, or discovered on the middle frame ─────────
    rng = np.random.default_rng(0)
    mid_dir = frame_dirs[len(frame_dirs) // 2]
    masks = load_frame_masks(mid_dir, labels, args.masks_dir)
    if args.init_bbox:
        vals = [float(v) for v in args.init_bbox.replace(',', ' ').split()]
        if len(vals) != 6:
            sys.exit('ERROR: --init_bbox needs 6 numbers '
                     '(x0 y0 z0 x1 y1 z1) in the dataset world frame')
        lo, hi = np.array(vals[:3]), np.array(vals[3:])
        pad = 0.15 * (hi - lo) + 0.05
        bbox = (lo - pad, hi + pad)
        print(f'  hull bbox (given): {np.round(bbox[0], 2)} .. '
              f'{np.round(bbox[1], 2)}')
    else:
        # Rejection sampling over the rig's own extent. A subject that is
        # small next to the camera spread makes hits rare, so widen the
        # sample count before giving up rather than failing on a dataset
        # that is perfectly fine.
        positions = [np.linalg.inv(rig[c]['w2c'])[:3, 3] for c in labels]
        centroid = np.mean(positions, axis=0)
        span = max(np.ptp(positions, axis=0)) or 4.0
        good = np.zeros((0, 3))
        n_cand = 500_000
        for attempt in range(4):
            cand = rng.uniform(centroid - span, centroid + span, size=(n_cand, 3))
            good = cand[hull_votes(rig, masks, cand) >= args.hull_min_views]
            if len(good) >= 100:
                break
            print(f'  bbox discovery: {len(good)} points from {n_cand:,} '
                  'candidates, widening the search')
            n_cand *= 4
        if len(good) < 100:
            sys.exit(
                f'ERROR: bbox discovery found only {len(good)} hull points '
                f'from {n_cand // 4:,} candidates. Check mask/pose '
                'consistency, lower --hull_min_views, or pass --init_bbox '
                'when the subject extent is already known (the synthetic '
                'pipeline passes it from the scene manifest).')
        pad = 0.15 * (good.max(0) - good.min(0)) + 0.05
        bbox = (good.min(0) - pad, good.max(0) + pad)
        print(f'  hull bbox ({mid_dir.name}): {np.round(bbox[0], 2)} .. '
              f'{np.round(bbox[1], 2)} ({len(good):,} seed points)')

    # ── per-frame hull carving (parallel) ──────────────────────────────────
    per_frame = max(args.hull_points // len(frame_dirs), 200)
    color_cams = labels[:: max(len(labels) // 4, 1)][:4]
    print(f'  carving {per_frame:,} points/frame, colours from {color_cams}')
    hull = [None] * len(frame_dirs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(carve_frame, d, rig, args.masks_dir, bbox,
                            per_frame, args.hull_min_views, color_cams,
                            np.random.default_rng(1000 + i)): i
                for i, d in enumerate(frame_dirs)}
        for n_done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            hull[futs[fut]] = fut.result()
            if n_done % 20 == 0 or n_done == len(frame_dirs):
                print(f'  hull: {n_done}/{len(frame_dirs)} frames', flush=True)

    times = np.concatenate([
        np.full(len(p), frame_idx[i] / args.fps, dtype=np.float32)
        for i, (p, _) in enumerate(hull)])
    pts = np.concatenate([p for p, _ in hull])
    rgb = np.concatenate([c for _, c in hull])
    out.mkdir(parents=True, exist_ok=True)
    write_ply_with_time(out / 'points3d.ply', pts, rgb, times)
    print(f'  points3d.ply: {len(pts):,} points over [{times.min():.3f}, {times.max():.3f}]s')

    # ── RGBA images (parallel) ─────────────────────────────────────────────
    jobs = []
    for label in labels:
        (out / 'realcams' / f'cam{label}').mkdir(parents=True, exist_ok=True)
        for i, d in enumerate(frame_dirs):
            jobs.append((d / 'images_flat' / f'{label}.png',
                         d / args.masks_dir / f'{label}.png',
                         out / 'realcams' / f'cam{label}' / f'frame_{frame_idx[i] + 1:05d}.png'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(convert_image, s, m, d, args.downscale) for s, m, d in jobs]
        for n_done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            fut.result()
            if n_done % 100 == 0 or n_done == len(jobs):
                print(f'  images: {n_done}/{len(jobs)}', flush=True)

    # ── separate eval-camera views (synthetic rigs) ────────────────────────
    # Cameras that never train and exist only to be scored. Holding out a rig
    # camera instead scores a different viewpoint for every rig, so numbers
    # from two camera configurations are not comparable; a fixed eval set is.
    eval_rig, eval_frame_dirs = load_eval_root(
        args.eval_root, n_source_frames, frame_idx)
    # Scored views from an earlier build into the same --out must not
    # survive into this one. The orchestrators turn scoring on when
    # eval_gt_flat holds any image, and a training camera standing in for
    # the test views reuses the frame_NNNNN names a held-out camera wrote,
    # so a stale file would be scored against the wrong camera's render.
    for stale in ('evalcams', 'eval_gt_flat'):
        shutil.rmtree(out / stale, ignore_errors=True)
    if eval_rig:
        eval_labels = sorted(eval_rig)
        jobs = []
        for label in eval_labels:
            (out / 'evalcams' / f'cam{label}').mkdir(parents=True, exist_ok=True)
            for i, d in enumerate(eval_frame_dirs):
                jobs.append((d / 'images_flat' / f'{label}.png',
                             d / args.masks_dir / f'{label}.png',
                             out / 'evalcams' / f'cam{label}' /
                             f'{eval_basename(label, frame_idx[i])}.png'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = [pool.submit(convert_image, s, m, d, args.downscale)
                    for s, m, d in jobs]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
        print(f'  eval cameras: {len(eval_labels)} x {len(eval_frame_dirs)} '
              f'frames from {args.eval_root}')

    # ── transforms jsons (per-frame intrinsics, cameras differ) ────────────
    ds = args.downscale

    def entries(cam_labels, source_rig=None, subdir='realcams', namer=None):
        source_rig = rig if source_rig is None else source_rig
        rows = []
        for label in cam_labels:
            cam = source_rig[label]
            fl_x, fl_y, cx, cy = cam['intr']
            for i in frame_idx:
                name = namer(label, i) if namer else f'frame_{i + 1:05d}'
                rows.append({
                    'file_path': f'{subdir}/cam{label}/{name}',
                    'camera_label': label,
                    'time': i / args.fps,
                    'fl_x': fl_x / ds, 'fl_y': fl_y / ds,
                    'cx': cx / ds, 'cy': cy / ds,
                    'w': cam['w'] // ds, 'h': cam['h'] // ds,
                    'transform_matrix': cam['c2w_gl'].tolist(),
                })
        return rows

    excluded = set(args.test_cameras) | set(args.holdout_cameras)
    train_labels = [c for c in labels if c not in excluded]
    with open(out / 'transforms_train.json', 'w') as f:
        json.dump({'camera_model': 'OPENCV', 'frames': entries(train_labels)},
                  f, indent=1)

    test_rows = []
    if eval_rig:
        test_rows += entries(sorted(eval_rig), source_rig=eval_rig,
                             subdir='evalcams', namer=eval_basename)
    if args.test_cameras:
        test_rows += entries(args.test_cameras)
    if not test_rows:
        # The trainer's eval loop needs at least one view. A training camera
        # stands in, and its score is a training-view monitor, not a
        # held-out measurement.
        test_rows = entries(train_labels[:1])
    with open(out / 'transforms_test.json', 'w') as f:
        json.dump({'camera_model': 'OPENCV', 'frames': test_rows}, f, indent=1)

    # ── flat GT for eval_render (scored cameras only) ──────────────────────
    # eval_render.py looks ground truth up by the basename of file_path, so
    # every scored view needs a name unique across cameras as well as frames.
    if args.test_cameras or eval_rig:
        (out / 'eval_gt_flat').mkdir(exist_ok=True)
        n_gt = 0
        for label in args.test_cameras:
            for i in frame_idx:
                flatten_gt(out / 'realcams' / f'cam{label}' / f'frame_{i + 1:05d}.png',
                           out / 'eval_gt_flat' / f'frame_{i + 1:05d}.png')
                n_gt += 1
        for label in sorted(eval_rig):
            for i in frame_idx:
                name = f'{eval_basename(label, i)}.png'
                flatten_gt(out / 'evalcams' / f'cam{label}' / name,
                           out / 'eval_gt_flat' / name)
                n_gt += 1
        print(f'  eval_gt_flat: {n_gt} black-composited GT frames')

    print(f'  transforms: {len(train_labels)} train cams, '
          f'{len(args.test_cameras) + len(eval_rig)} scored cams '
          f'({len(test_rows)} test views), done -> {out}')


if __name__ == '__main__':
    main()
