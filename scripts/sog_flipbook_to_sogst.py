#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
sog_flipbook_to_sogst.py  –  Convert a per-frame .sog flipbook sequence into
a single .sogst archive.

A flipbook stores an independent splat set per frame. .sogst has no
per-frame data — every splat carries a temporal Gaussian — so each frame's
splats are emitted with:

    t_center = frame time
    t_sigma  = t_sigma_factor * frame_interval
    velocity = 0

With the default t_sigma_factor of 0.5 the per-frame temporal Gaussians sum
to a near-constant coverage over the clip (a theta function): playback
crossfades adjacent frames (~14% ghost of each neighbour at a frame's own
centre) instead of hard-cutting, which reads as mild motion blur.

Note what this discards: velocity = 0 and a constant sigma throw away the
temporal modelling the format exists to carry, so a flipbook asset is
much larger than a trained 4D one for the same clip. This path is for
content that only exists as per-frame splats; prefer bake_sogst.py where
a 4D model is available.

Size/feasibility: a player evaluates EVERY splat every rendered frame, so
summing whole frames is untenable (a 481-frame flipbook holds ~68M splats).
--max-splats sets a total budget divided evenly across frames; each frame
keeps its top-k splats by visual importance (alpha x mean-scale^2).

Frame decoding shells out to the `splat-transform` CLI (npm i -g
@playcanvas/splat-transform) for .sog -> .ply, so this script needs no webp
codec of its own.

Usage:
    python sog_flipbook_to_sogst.py \
        --input-dir path/to/frames --pattern '{frame:04d}.sog' \
        --start 100 --end 187 --fps 24 \
        --max-splats 2500000 \
        --output ariana.sogst
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sogst_io import SOGST_FIELDS, report_output


def read_ply_float32(path):
    """Minimal reader for splat-transform's all-float32 binary PLY output."""
    with open(path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            line = f.readline()
            if not line:
                raise ValueError(f'{path}: truncated PLY header')
            header += line
        n = 0
        fields = []
        for ln in header.decode('ascii').splitlines():
            if ln.startswith('element vertex'):
                n = int(ln.split()[-1])
            elif ln.startswith('property float'):
                fields.append(ln.split()[-1])
            elif ln.startswith('property'):
                raise ValueError(f'{path}: non-float property: {ln}')
        if 'binary_little_endian' not in header.decode('ascii'):
            raise ValueError(f'{path}: expected binary_little_endian PLY')
        data = np.fromfile(f, dtype=np.float32, count=n * len(fields))
    data = data.reshape(n, len(fields))
    return {name: data[:, i] for i, name in enumerate(fields)}, n


def load_frame(sog_path, tmp_dir, keep, prune_alpha):
    """Decode one .sog and return its top-`keep` splats as a [K,14] array:
    x y z rot0..3 scale0..2 opacity fdc0..2 (v2 field order minus temporal)."""
    with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix='.ply', delete=False) as tf:
        ply_path = tf.name
    try:
        subprocess.run(
            ['splat-transform', '-w', str(sog_path), ply_path],
            check=True, capture_output=True, text=True)
        d, n = read_ply_float32(ply_path)
    finally:
        os.unlink(ply_path)

    cols = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
            'scale_0', 'scale_1', 'scale_2', 'opacity',
            'f_dc_0', 'f_dc_1', 'f_dc_2']
    m = np.stack([d[c] for c in cols], axis=1)

    # normalise quaternions (SOG quantisation can leave them slightly off-unit)
    qn = np.linalg.norm(m[:, 3:7], axis=1, keepdims=True)
    m[:, 3:7] /= np.maximum(qn, 1e-12)

    alpha = 1.0 / (1.0 + np.exp(-m[:, 10]))
    visible = alpha >= prune_alpha
    m = m[visible]
    alpha = alpha[visible]

    if keep and len(m) > keep:
        mean_scale = np.exp(m[:, 7:10].mean(axis=1))
        importance = alpha * mean_scale ** 2
        top = np.argpartition(-importance, keep - 1)[:keep]
        m = m[top]
    return np.ascontiguousarray(m, dtype=np.float32), n


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', required=True, help='Directory of per-frame .sog files')
    parser.add_argument('--pattern', default='{frame:04d}.sog',
                        help="Filename pattern with a {frame:...} field (default: '{frame:04d}.sog')")
    parser.add_argument('--start', type=int, required=True, help='First frame number (inclusive)')
    parser.add_argument('--end', type=int, required=True, help='Last frame number (inclusive)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Take every Nth frame (default: 1). Halves splat count per doubling; '
                             'playback stays real-time (frame interval widens and t_sigma with it).')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Source flipbook playback rate (default: 30). Frame i maps to '
                             't = i*stride/fps, so the clip duration matches the original.')
    parser.add_argument('--max-splats', type=int, default=2_500_000,
                        help='Total splat budget divided evenly across frames; each frame keeps its '
                             'top share by importance (default: 2500000; 0 = keep everything, which '
                             'is usually untenable — see module docstring)')
    parser.add_argument('--t-sigma-factor', type=float, default=0.5,
                        help='t_sigma as a fraction of the frame interval (default: 0.5 — near-flat '
                             'total coverage, mild crossfade; smaller = crisper cuts but pulsing '
                             'between frame centres)')
    parser.add_argument('--prune-alpha', type=float, default=1.0 / 255,
                        help='Drop splats whose peak alpha is below this before ranking (default: 1/255)')
    parser.add_argument('--jobs', type=int, default=8, help='Parallel frame decodes (default: 8)')
    parser.add_argument('--segment-duration', type=float, default=0.1,
                        help='Temporal segment length in seconds for per-segment culling '
                             '(default 0.1; 0 disables segmentation). A flipbook segments '
                             'especially well: every splat is short-lived by construction.')
    parser.add_argument('--output', required=True, help='Destination .sogst archive')
    args = parser.parse_args()

    frames = list(range(args.start, args.end + 1, args.stride))
    n_frames = len(frames)
    dt = args.stride / args.fps
    keep = args.max_splats // n_frames if args.max_splats else 0
    print(f'{n_frames} frames (stride {args.stride}) | dt={dt * 1000:.1f}ms | '
          f'budget {args.max_splats:,} -> keep {keep:,}/frame' if keep else
          f'{n_frames} frames (stride {args.stride}) | dt={dt * 1000:.1f}ms | keeping all splats')

    in_dir = Path(args.input_dir)
    paths = [in_dir / args.pattern.format(frame=f) for f in frames]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        sys.exit(f'ERROR: missing frames, e.g. {missing[0]}')

    tmp_dir = tempfile.mkdtemp(prefix='sog2sogst_')
    results = [None] * n_frames
    total_in = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(load_frame, p, tmp_dir, keep, args.prune_alpha): i
                for i, p in enumerate(paths)}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            results[i], n_in = fut.result()
            total_in += n_in
            done += 1
            if done % 25 == 0 or done == n_frames:
                print(f'  decoded {done}/{n_frames} frames', flush=True)
    os.rmdir(tmp_dir)

    counts = np.array([len(r) for r in results])
    total = int(counts.sum())
    print(f'  kept {total:,} / {total_in:,} splats '
          f'({total / total_in * 100:.1f}%) | per-frame min/median/max: '
          f'{counts.min():,}/{int(np.median(counts)):,}/{counts.max():,}')

    base = np.concatenate(results, axis=0)          # [N,14]
    t_center = np.concatenate([
        np.full(len(r), i * dt, dtype=np.float32) for i, r in enumerate(results)])
    del results
    time_min = 0.0
    time_max = (n_frames - 1) * dt
    t_sigma = np.full(total, args.t_sigma_factor * dt, dtype=np.float32)
    zeros = np.zeros(total, dtype=np.float32)

    # base[:, :14] is the v2 field order minus the temporal columns, i.e.
    # SOGST_FIELDS[:14]; velocity is identically zero for a flipbook.
    fields = {name: np.ascontiguousarray(base[:, i])
              for i, name in enumerate(SOGST_FIELDS[:14])}
    fields.update({'vx': zeros, 'vy': zeros, 'vz': zeros,
                   't_center': t_center, 't_sigma': t_sigma})

    print(f'  clip: [{time_min}, {time_max:.3f}]s @ {args.fps / args.stride:g}fps advisory, '
          f't_sigma={args.t_sigma_factor * dt * 1000:.1f}ms')

    from sogst_pack import pack_sogst
    pack_sogst(args.output, fields, time_min, time_max, args.fps / args.stride,
               shn_count=0,          # a flipbook decode carries DC colour only
               segment_duration=args.segment_duration,
               generator='volumetric-capture-pipeline sog_flipbook_to_sogst')
    report_output(args.output)


if __name__ == '__main__':
    main()
