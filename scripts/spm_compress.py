#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""spm_compress.py - OMG4 sampling->pruning->merging as a pipeline stage.

Drives the OMG4 reference implementation (MinShirley/OMG4) to compress a
pretrained Real-Time4DGS checkpoint by reducing its Gaussian count with
staged fine-tuning, then converts the result to a streamable .sogst.
The whole stage is upstream of the container: the output is the same
representation with fewer primitives, so the viewer needs no changes.

Stages (each skipped when its artifact already exists, so reruns resume):

  1. compute_gradient.py  -> <model_path>/{view,t}_grad.npy   (SD score)
  2. train.py             -> <model_path>/comp.xz             (S->P->M->SVQ,
                             ~12k fine-tune iterations from iter 30000)
  3. bake_sogst.py        -> interchange PLY (baked MLPs -> explicit SH)
  4. sogst_pack.py        -> .sogst (--shn-count auto-scales
                             with the post-SPM splat count: the fixed-size
                             centroid table dominates small scenes)

The scene config must be an OMG4-style yaml whose OptimizationParams carry
the SPM block (tau_GS, tau_GP, merge + SVQ settings) with iterations
42_001 — see configs/custom/perframe90_omg4.yaml in the OMG4 repo for the
template. Hyperparameters worth knowing: tau_GS is the sampling keep
quantile pressure (paper headline uses 0.2 = keep ~20%; our subject scenes
have shipped with the gentler 0.8), tau_GP the pruning quantile.

Example:

    python scripts/spm_compress.py \
        --omg4-repo ~/Dev/github/OMG4 \
        --config configs/custom/perframe90_omg4.yaml \
        --checkpoint output/perframe90_refit/chkpnt30000.pth \
        --fps 29.97 \
        --output /tmp/tatum_spm.sogst
"""

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile


def sh_count_for(n_splats: int) -> int:
    """SH VQ centroid count scaled with scene size (~N/16, power of two,
    clamped to [1024, 65536])."""
    if n_splats <= 0:
        return 65536
    k = 2 ** int(math.ceil(math.log2(max(1024, n_splats / 16))))
    return int(min(65536, max(1024, k)))


def run(cmd, cwd, log_path):
    print(f'  $ {" ".join(cmd)}')
    print(f'    (log: {log_path})')
    with open(log_path, 'a') as log:
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f'stage failed (exit {proc.returncode}); see {log_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--omg4-repo', required=True, help='path to the MinShirley/OMG4 clone')
    ap.add_argument('--config', required=True,
                    help='scene yaml with the SPM OptimizationParams block (repo-relative or absolute)')
    ap.add_argument('--checkpoint', required=True,
                    help='pretrained Real-Time4DGS chkpntNNNNN.pth (repo-relative or absolute)')
    ap.add_argument('--output', required=True, help='destination .sogst archive')
    ap.add_argument('--fps', type=float, default=30.0, help='playback fps stored in the export')
    ap.add_argument('--shn-count', type=int, default=0,
                    help='SH VQ centroid count (0 = auto from post-SPM splat count)')
    ap.add_argument('--segment-duration', type=float, default=0.1)
    ap.add_argument('--python', default=os.path.expanduser('~/miniconda3/envs/omg4/bin/python'),
                    help='python with torch+CUDA+diff_gaussian_rasterization (omg4 env)')
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.omg4_repo))
    cfg = args.config if os.path.isabs(args.config) else os.path.join(repo, args.config)
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(repo, args.checkpoint)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    # model_path from the yaml decides where OMG4 writes its artifacts
    model_path = None
    time_duration = None
    for line in open(cfg):
        m = re.match(r'\s*model_path:\s*"([^"]+)"', line)
        if m:
            model_path = m.group(1)
        m = re.match(r'\s*time_duration:\s*\[([^\]]+)\]', line)
        if m:
            time_duration = [float(v) for v in m.group(1).split(',')]
    if not model_path:
        raise SystemExit(f'no model_path in {cfg}')
    model_dir = model_path if os.path.isabs(model_path) else os.path.join(repo, model_path)
    os.makedirs(model_dir, exist_ok=True)

    # -- 1. SD-score gradients ---------------------------------------------
    grads = [os.path.join(model_dir, f'{k}_grad.npy') for k in ('view', 't')]
    if all(os.path.exists(g) for g in grads):
        print('[1/4] SD-score gradients: already present, skipping')
    else:
        print('[1/4] SD-score gradients (compute_gradient.py)')
        run([args.python, 'compute_gradient.py', '--config', cfg, '--start_checkpoint', ckpt],
            repo, os.path.join(model_dir, 'gradient.log'))

    # -- 2. sampling -> pruning -> merging -> SVQ ---------------------------
    comp = os.path.join(model_dir, 'comp.xz')
    if os.path.exists(comp):
        print('[2/4] SPM compression: comp.xz already present, skipping')
    else:
        print('[2/4] SPM compression (train.py, ~12k fine-tune iterations)')
        run([args.python, 'train.py', '--config', cfg, '--start_checkpoint', ckpt,
             '--grad', model_dir], repo, os.path.join(model_dir, 'train.log'))
        if not os.path.exists(comp):
            raise SystemExit(f'train.py finished but {comp} is missing')

    # -- 3. bake to the interchange PLY --------------------------------------
    print('[3/4] bake to interchange PLY (bake_sogst.py)')
    ply_path = os.path.join(tempfile.gettempdir(), os.path.basename(args.output) + '.ply')
    # The legacy corruption filters (bad_color/garbage, calibrated for the old
    # SVQ pipeline's catastrophic MLP extrapolation) are replaced here by the
    # black-floater filter: bad_color's binary out-of-range test deleted the
    # subject's own dark clothing along with the junk (transparency), while
    # keeping everything left a near-black floater cloud around the subject.
    # black_floater_mask keeps legitimate dark surface and drops only the
    # detached murk. sh_clamp 3.0 instead of the 1.5 default because kept
    # splats with an out-of-range DC rely on their higher SH bands to land in
    # range, and 1.5 zeroes exactly those bands (the actual washout cause in
    # the first SPM exports).
    cmd = [args.python, os.path.join(scripts_dir, 'bake_sogst.py'),
           '--input', comp, '--emit_ply', ply_path, '--fps', str(args.fps),
           '--no_filter_corrupted', '--filter_black_floaters', '--sh_clamp', '3.0']
    if time_duration:
        cmd += ['--time_min', str(time_duration[0]), '--time_max', str(time_duration[1])]
    run(cmd, repo, os.path.join(model_dir, 'export.log'))

    # -- 4. pack .sogst ------------------------------------------------------
    shn = args.shn_count
    if not shn:
        sys.path.insert(0, scripts_dir)
        from sogst_ply import ply_vertex_count
        n_splats = ply_vertex_count(ply_path)
        shn = sh_count_for(n_splats)
        print(f'[4/4] pack .sogst (sogst_pack.py, auto --shn-count {shn} for {n_splats} splats)')
    else:
        print(f'[4/4] pack .sogst (sogst_pack.py, --shn-count {shn})')
    run([args.python, os.path.join(scripts_dir, 'sogst_pack.py'),
         '--input', ply_path, '--output', args.output,
         '--shn-count', str(shn), '--segment-duration', str(args.segment_duration),
         '--verify'],
        repo, os.path.join(model_dir, 'export.log'))

    os.remove(ply_path)
    size = os.path.getsize(args.output) / 1e6
    print(f'done: {args.output} ({size:.1f} MB)')
    print('next: score it with scripts/eval_render.py against held-out views')


if __name__ == '__main__':
    main()
