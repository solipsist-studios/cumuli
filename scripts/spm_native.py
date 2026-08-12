#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""spm_native.py - OMG4 sampling->pruning->merging with explicit SH kept.

The SPM-native counterpart to spm_compress.py. Both drive the OMG4
reference implementation's SD-score compression schedule; the difference
is what happens after the merge rounds:

  spm_compress.py  -> OMG4's full pipeline: appearance is distilled into
                      6-dim latents + MLPs + SVQ, then baked back to SH
                      by the exporter. The count reduction is excellent
                      but the appearance round-trip reads soft/washed on
                      subject captures (measured on tatum: the loss is
                      count-independent).
  spm_native.py    -> train.py --spm_native_out: after S->P->M the
                      surviving Gaussians keep their explicit per-splat
                      SH and fine-tune a few thousand more iterations,
                      then a rotor-style checkpoint is saved. The bake
                      takes xz_to_omg4.py's proven checkpoint path (the
                      same one the full-quality pretrain exports use), so
                      there is no appearance round-trip at all.

Stages (each skipped when its artifact already exists, so reruns resume):

  1. compute_gradient.py  -> <model_path>/{view,t}_grad.npy   (SD score)
  2. train.py --spm_native_out
                          -> <model_path>/chkpnt_spm_native.pth
  3. xz_to_omg4.py        -> v2 (bake-only: no MLPs to evaluate)
  4. sog_pack.py          -> v3 (--shn-count auto-scales with splat count)

The scene config must be an OMG4-style yaml whose OptimizationParams
carry the SPM block (tau_GS, tau_GP, merge settings) with iterations
42_001 — see configs/custom/heidi_spm_native.yaml in the OMG4 repo.
Requires the OMG4 clone's feature/ftgs-degree2 branch (or later), which
carries the --spm_native_out patch.

Example:

    python scripts/spm_native.py \
        --omg4-repo ~/Dev/github/OMG4 \
        --config configs/custom/heidi_spm_native.yaml \
        --checkpoint output/heidi_pretrain/chkpnt30000.pth \
        --fps 30 --output-v3 /tmp/heidi_spm_native_v3.omg4
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

from spm_compress import run, sh_count_for


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--omg4-repo', required=True, help='path to the MinShirley/OMG4 clone (with the spm_native patch)')
    ap.add_argument('--config', required=True,
                    help='scene yaml with the SPM OptimizationParams block (repo-relative or absolute)')
    ap.add_argument('--checkpoint', required=True,
                    help='pretrained Real-Time4DGS chkpntNNNNN.pth (repo-relative or absolute)')
    ap.add_argument('--output-v3', required=True, help='destination .omg4 v3 archive')
    ap.add_argument('--fps', type=float, default=30.0, help='playback fps stored in the export')
    ap.add_argument('--extra-iter', type=int, default=3000,
                    help='explicit-SH fine-tune iterations after the last merge (default 3000)')
    ap.add_argument('--shn-count', type=int, default=0,
                    help='SH VQ centroid count (0 = auto from post-SPM splat count)')
    ap.add_argument('--sh-clamp', type=float, default=3.0,
                    help='per-splat SH excursion clamp during bake (default 3.0)')
    ap.add_argument('--keep-main-cluster', action='store_true',
                    help='drop splats outside the largest spatial cluster during bake')
    ap.add_argument('--freeze-temporal', action='store_true',
                    help='freeze t / scaling_t / rotation_r during the recovery fine-tune. '
                         'OMG4s update_learning_rate() decays only the xyz group, so temporal '
                         'parameters keep their initial lr for the whole run and drift during '
                         'recovery. MEASURED: this makes no difference (22.56 vs 22.32 dB) — a '
                         'parameter diff showed opacity drifting most (24.5%% relative), not the '
                         'temporal group. Kept as plumbing for further experiments; do not '
                         'expect it to recover quality.')
    ap.add_argument('--filter-black-floaters', action='store_true',
                    help='drop the detached near-black streaks a merge can leave behind. '
                         'Not needed on every scene (heidi renders clean without it); '
                         'check a white-background render before deciding — black floaters '
                         'are invisible over the eval renders black background.')
    ap.add_argument('--segment-duration', type=float, default=0.1)
    ap.add_argument('--python', default=os.path.expanduser('~/miniconda3/envs/omg4/bin/python'),
                    help='python with torch+CUDA+diff_gaussian_rasterization (omg4 env)')
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.omg4_repo))
    cfg = args.config if os.path.isabs(args.config) else os.path.join(repo, args.config)
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(repo, args.checkpoint)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

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

    # -- 2. sampling -> pruning -> merging, explicit SH kept ----------------
    native_ckpt = os.path.join(model_dir, 'chkpnt_spm_native.pth')
    if os.path.exists(native_ckpt):
        print('[2/4] SPM-native fine-tune: checkpoint already present, skipping')
    else:
        print(f'[2/4] SPM-native fine-tune (train.py --spm_native_out, +{args.extra_iter} recovery iters)')
        train_cmd = [args.python, 'train.py', '--config', cfg, '--start_checkpoint', ckpt,
                     '--grad', model_dir, '--spm_native_out', native_ckpt,
                     '--spm_native_extra_iter', str(args.extra_iter)]
        if args.freeze_temporal:
            train_cmd.append('--spm_native_freeze_temporal')
        run(train_cmd, repo, os.path.join(model_dir, 'train.log'))
        if not os.path.exists(native_ckpt):
            raise SystemExit(f'train.py finished but {native_ckpt} is missing')

    # -- 3. bake to .omg4 v2 (checkpoint path: explicit SH, no MLPs) --------
    print('[3/4] bake to v2 (xz_to_omg4.py, checkpoint path)')
    v2_path = os.path.join(tempfile.gettempdir(), os.path.basename(args.output_v3) + '.v2.omg4')
    cmd = [args.python, os.path.join(scripts_dir, 'xz_to_omg4.py'),
           '--input', native_ckpt, '--output', v2_path, '--fps', str(args.fps),
           '--sh_clamp', str(args.sh_clamp)]
    if args.keep_main_cluster:
        cmd.append('--keep_main_cluster')
    if args.filter_black_floaters:
        cmd.append('--filter_black_floaters')
    if time_duration:
        cmd += ['--time_min', str(time_duration[0]), '--time_max', str(time_duration[1])]
    run(cmd, repo, os.path.join(model_dir, 'export.log'))

    # -- 4. pack v3 ----------------------------------------------------------
    shn = args.shn_count
    if not shn:
        sys.path.insert(0, scripts_dir)
        from sog_pack import fields_from_v2
        header, _ = fields_from_v2(v2_path)
        shn = sh_count_for(header['num_splats'])
        print(f'[4/4] pack v3 (sog_pack.py, auto --shn-count {shn} for {header["num_splats"]} splats)')
    else:
        print(f'[4/4] pack v3 (sog_pack.py, --shn-count {shn})')
    run([args.python, os.path.join(scripts_dir, 'sog_pack.py'),
         '--input', v2_path, '--output', args.output_v3,
         '--shn-count', str(shn), '--segment-duration', str(args.segment_duration),
         '--verify'],
        repo, os.path.join(model_dir, 'export.log'))

    os.remove(v2_path)
    size = os.path.getsize(args.output_v3) / 1e6
    print(f'done: {args.output_v3} ({size:.1f} MB)')
    print('next: score it with scripts/eval_render.py against held-out views')


if __name__ == '__main__':
    main()
