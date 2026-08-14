#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""make_sogst_fixture.py - the synthetic conformance fixture for .sogst.

Emits a small, fully deterministic 4D scene as an interchange PLY (and
optionally packs it into a .sogst archive).  Its purpose is to let a second
implementation of the format -- an encoder or a player -- check itself
against something whose correct output is known analytically, without
needing a real capture and without the subject-release question that real
sample assets carry.

The scene is a ballistic parabola: `--count` splats launched from a common
origin with a spread of initial velocities under a constant downward
acceleration, each fading in and out around its own temporal centre.  It is
chosen because it exercises the parts of the format that are easy to get
wrong and hard to notice:

  - Motion is genuinely second-order, so `accel` is meaningful rather than
    synthesised.  The degree-1 variant (`--degree 1`) drops the accel
    columns and keeps the same velocities, which makes the two files a
    direct test of the gating rule: a player that ignores `accel` must
    render the degree-2 file, just along straight lines.
  - t_center is spread across the clip with a small t_sigma, so the
    encoder produces many populated temporal segments, plus a deliberate
    minority of long-lived splats that land in the persistent range.
  - Quaternions are constructed to hit all four smallest-three dropped-
    component cases (mode bytes 252..255), which a random rotation
    distribution will do eventually but a fixture should do by
    construction.
  - Scales, opacities and DC colours span wide enough ranges to make
    codebook quantization error visible if a codebook is built wrongly.

The ground truth is the PLY itself: pack it, decode the archive with
eval_render.decode_sogst_fields(), and compare per field.  See
docs/sogst-format.md section 10 -- compare decoded fields, not rendered
PSNR, because codebook initialisation is implementation-defined.

    python make_sogst_fixture.py --output parabola.ply
    python make_sogst_fixture.py --output parabola_d1.ply --degree 1 --no-sh
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sogst_ply import write_sogst_ply, write_sogst_sidecar  # noqa: E402

FIXTURE_SEED = 20260813
GRAVITY = -9.81           # scene units / s^2, the fixture's known constant

# --blocks places each temporal segment's splats in its own spatial cluster, so
# a viewer can COUNT which segments are being drawn.  That requires knowing the
# segment boundaries the packer will choose, so this must match
# sogst_pack.compute_sogst_order's `segment_duration` default; --blocks writes
# the value it assumed into the sidecar so a mismatch is detectable rather than
# silent.  Pass the same --segment-duration to the packer if you change it.
SEGMENT_DURATION = 0.1
BLOCK_SPACING = 1.0       # centre-to-centre; >> BLOCK_RADIUS so blocks never touch
BLOCK_RADIUS = 0.12


def build_fixture(count=8192, time_min=0.0, time_max=2.0, degree=2,
                  include_sh=True, persistent_fraction=0.08, seed=FIXTURE_SEED,
                  gap=None, blocks=False):
    """Build the parabola fixture's field arrays.

    `gap` is an optional (start, end) window in seconds that no splat's
    t_center falls into, which forces the encoder to emit empty segments
    (`first == last`).  Those are spec-legal (section 5) and no real capture
    produces them, so the branches that handle them -- a player's culling
    scan, an encoder's zero-splat group -- are otherwise only ever tested by
    reading the code.

    `blocks` replaces the parabola with static, spatially separated clusters,
    one per temporal segment plus one for the persistent group.  It exists
    because "nothing broke" is not the same claim as "the right segments were
    drawn", and the parabola cannot tell them apart: it is an expanding cloud,
    so a player that culls nothing at all looks the same as one that culls
    correctly.  With `blocks` the drawn population is countable by eye -- at
    any instant you should see the persistent block plus only the block(s)
    whose segment spans that instant, and a culling bug shows up as a block
    that is present when it should not be (or missing when it should).
    Combined with `gap` it is the strong form: an empty RUN whose surviving
    neighbours are in visibly different places.

    Motion is deliberately zero here (degree 1, velocity 0): a moving block
    would reintroduce the ambiguity the mode exists to remove.  Section 4.8
    forbids synthesising `accel`, so block mode never emits it.

    Returns (fields, meta) where `meta` records the analytic constants a
    verifier may want (the gravity term, the persistent split, the block
    layout) alongside the clip scalars.
    """
    rng = np.random.default_rng(seed)
    n = int(count)

    # Temporal layout: most splats are short-lived and spread evenly across
    # the clip, so segmentation has something to segment.  A minority get a
    # long sigma and must end up in the persistent range.
    # The short sigmas are capped so that a short-lived splat's active span
    # (2 * k_sigma * sigma, k_sigma = 3.8) stays under the default
    # persistence threshold of 3 * 0.1 s -- otherwise the encoder promotes
    # most of them to persistent and the fixture stops exercising
    # per-segment culling, which is the thing it is there to exercise.
    t_center = rng.uniform(time_min, time_max, n)
    if gap is not None:
        # Fold anything inside the window out to its edges, keeping the
        # splat count exact.
        lo, hi = float(gap[0]), float(gap[1])
        mid = 0.5 * (lo + hi)
        inside = (t_center > lo) & (t_center < hi)
        t_center[inside & (t_center <= mid)] = lo
        t_center[inside & (t_center > mid)] = hi
    t_sigma = rng.uniform(0.01, 0.03, n)
    n_persistent = int(round(persistent_fraction * n))
    persistent_idx = rng.choice(n, size=n_persistent, replace=False)
    t_sigma[persistent_idx] = rng.uniform(0.4, 0.7, n_persistent)

    # Ballistic motion.  Each splat is a particle launched at t = 0 from
    # near the origin; its PLY columns describe that trajectory re-anchored
    # to its own t_center, which is exactly the re-anchoring a producer has
    # to get right (position, velocity and accel move together).
    launch = rng.normal(0.0, 0.05, (n, 3))
    speed = rng.uniform(2.0, 6.0, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    elevation = rng.uniform(np.pi / 6, np.pi / 3, n)
    v0 = np.stack([
        speed * np.cos(elevation) * np.cos(theta),
        speed * np.sin(elevation),
        speed * np.cos(elevation) * np.sin(theta),
    ], axis=1)
    accel_true = np.zeros((n, 3))
    accel_true[:, 1] = 0.5 * GRAVITY        # raw dt^2 coefficient, NOT a/2 again

    # Re-anchor to t_center:  p(t) = p0 + v0*t + a*t^2  =>  at tc,
    #   xyz = p(tc),  v = v0 + 2*a*tc,  a unchanged.
    tc = t_center[:, None]
    xyz = launch + v0 * tc + accel_true * tc * tc
    velocity = v0 + 2.0 * accel_true * tc

    block_layout = None
    if blocks:
        # One cluster per segment the packer will bucket into, laid out along
        # x in segment order, plus the persistent group lifted clear on y.
        # The bucket expression mirrors compute_sogst_order exactly -- if the
        # two ever drift, the blocks stop lining up with the segments and the
        # fixture silently stops testing what it claims to.
        n_seg = max(1, int(np.ceil((time_max - time_min) / SEGMENT_DURATION)))
        bucket = np.clip(((t_center - time_min) / SEGMENT_DURATION).astype(np.int64),
                         0, n_seg - 1)
        centres = np.zeros((n, 3))
        centres[:, 0] = (bucket - 0.5 * (n_seg - 1)) * BLOCK_SPACING
        is_persistent = np.zeros(n, dtype=bool)
        is_persistent[persistent_idx] = True
        centres[is_persistent, 0] = 0.0
        centres[is_persistent, 1] = BLOCK_SPACING
        xyz = centres + rng.normal(0.0, BLOCK_RADIUS / 3.0, (n, 3))
        velocity = np.zeros((n, 3))
        accel_true = np.zeros((n, 3))
        block_layout = {
            'segment_duration': float(SEGMENT_DURATION),
            'segments': int(n_seg),
            'spacing': float(BLOCK_SPACING),
            'radius': float(BLOCK_RADIUS),
            'segment_block_x': [float((k - 0.5 * (n_seg - 1)) * BLOCK_SPACING)
                                for k in range(n_seg)],
            'persistent_block': [0.0, float(BLOCK_SPACING), 0.0],
            'splats_per_segment': [int((bucket[~is_persistent] == k).sum())
                                   for k in range(n_seg)],
        }

    # Quaternions hitting every smallest-three case by construction: cycle
    # the largest-magnitude component through w, x, y, z.
    quat = rng.normal(0.0, 1.0, (n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    dominant = np.arange(n) % 4
    boost = np.zeros((n, 4))
    boost[np.arange(n), dominant] = 2.0
    quat = quat + boost
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    log_scales = rng.uniform(-6.0, -2.0, (n, 3))
    opacity_logit = rng.uniform(-1.0, 4.0, n)
    f_dc = rng.uniform(-1.5, 1.5, (n, 3))

    fields = {
        'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2],
        'rot_0': quat[:, 0], 'rot_1': quat[:, 1],
        'rot_2': quat[:, 2], 'rot_3': quat[:, 3],
        'scale_0': log_scales[:, 0], 'scale_1': log_scales[:, 1],
        'scale_2': log_scales[:, 2],
        'opacity': opacity_logit,
        'f_dc_0': f_dc[:, 0], 'f_dc_1': f_dc[:, 1], 'f_dc_2': f_dc[:, 2],
        'vx': velocity[:, 0], 'vy': velocity[:, 1], 'vz': velocity[:, 2],
        't_center': t_center,
        't_sigma': t_sigma,
    }
    if degree == 2 and not blocks:
        fields['ax'] = accel_true[:, 0]
        fields['ay'] = accel_true[:, 1]
        fields['az'] = accel_true[:, 2]
    if include_sh:
        # Channel-major, index j*15 + k, decaying by band so the values look
        # like a real SH tail.  Crucially it is also spatially coherent --
        # a smooth function of launch direction plus a little noise --
        # because real SH is, and because pure per-splat noise is the
        # worst possible case for the vector quantizer: it makes the
        # encoder's VQ error look like a layout bug in a field comparison.
        band_scale = np.tile(np.repeat([0.4, 0.15, 0.05], [3, 5, 7]), 3)[None, :]
        direction = v0 / np.linalg.norm(v0, axis=1, keepdims=True)
        basis = np.stack([direction[:, 0], direction[:, 1], direction[:, 2],
                          direction[:, 0] * direction[:, 1]], axis=1)   # [n, 4]
        mixing = rng.normal(0.0, 1.0, (4, 45))
        fields['f_rest'] = (basis @ mixing + rng.normal(0.0, 0.15, (n, 45))) * band_scale

    meta = {
        'time_min': float(time_min), 'time_max': float(time_max),
        'fps': 30.0, 'count': n,
        'motion_degree': 1 if blocks else int(degree),
        'gap': (None if gap is None else [float(gap[0]), float(gap[1])]),
        'gravity_dt2_coefficient': (None if blocks else float(0.5 * GRAVITY)),
        'long_lived_splats': int(n_persistent),
        'seed': int(seed),
        'blocks': block_layout,
    }
    return fields, meta


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True, help='Destination .ply path')
    parser.add_argument('--count', type=int, default=8192, help='Splat count (default 8192)')
    parser.add_argument('--degree', type=int, default=2, choices=(1, 2),
                        help='Motion degree: 2 emits ax/ay/az (default), 1 omits them')
    parser.add_argument('--no-sh', action='store_true',
                        help='Omit the 45 f_rest_* columns')
    parser.add_argument('--time-max', type=float, default=2.0, help='Clip length in seconds')
    parser.add_argument('--seed', type=int, default=FIXTURE_SEED)
    parser.add_argument('--gap', type=str, default=None, metavar='START,END',
                        help='Leave a window (seconds) with no splats, so the encoder '
                             'emits EMPTY segments (first == last). Spec-legal and never '
                             'produced by a real capture, so it is the only way to '
                             'execute a player culling scan or an encoder group writer '
                             'against a zero-splat segment.')
    parser.add_argument('--blocks', action='store_true',
                        help='Static spatially-separated clusters, one per temporal '
                             'segment plus one for the persistent group, instead of '
                             'the parabola. Makes the drawn population countable by '
                             'eye, so "the right segments were drawn" can be checked '
                             'rather than only "nothing broke" -- the expanding cloud '
                             'cannot distinguish correct culling from no culling. '
                             'Forces degree 1 and zero motion. Pair with --gap.')
    parser.add_argument('--sidecar', action='store_true',
                        help='Also write the optional <name>.sogst.json sidecar')
    parser.add_argument('--pack', type=str, default=None,
                        help='Also pack the fixture into this .sogst path')
    args = parser.parse_args()

    gap = tuple(float(v) for v in args.gap.split(',')) if args.gap else None
    fields, meta = build_fixture(count=args.count, time_max=args.time_max,
                                 degree=args.degree, include_sh=not args.no_sh,
                                 seed=args.seed, gap=gap, blocks=args.blocks)
    n = write_sogst_ply(args.output, fields, meta['time_min'], meta['time_max'],
                        meta['fps'], generator='volumetric-capture-pipeline '
                                               'make_sogst_fixture '
                                               + ('blocks' if args.blocks else 'parabola'))
    if args.sidecar:
        write_sogst_sidecar(args.output, meta['time_min'], meta['time_max'],
                            meta['fps'], motion_degree=meta['motion_degree'])
    print(f"Wrote {args.output}: {n:,} splats, degree {meta['motion_degree']}, "
          f"SH={'no' if args.no_sh else 'yes'}, "
          f"{meta['time_min']}..{meta['time_max']}s @ {meta['fps']} fps")
    print(f"  {meta['long_lived_splats']:,} long-lived splats (expected persistent range)")
    if meta['gravity_dt2_coefficient'] is not None:
        print(f"  dt^2 coefficient on y: {meta['gravity_dt2_coefficient']:.4f} "
              '(raw coefficient, not half-acceleration)')
    if meta['blocks']:
        b = meta['blocks']
        populated = [k for k, c in enumerate(b['splats_per_segment']) if c]
        empty = [k for k, c in enumerate(b['splats_per_segment']) if not c]
        print(f"  block layout: {b['segments']} segment blocks along x, spacing "
              f"{b['spacing']} (cluster radius {b['radius']}), persistent block at "
              f"{b['persistent_block']}")
        print(f"    populated blocks {populated}")
        print(f"    empty blocks     {empty}"
              if empty else '    empty blocks     none (pass --gap to create some)')
        print('    at time t the persistent block plus only the block(s) whose '
              'segment spans t should be drawn')

    if args.pack:
        from sogst_pack import pack_sogst
        pack_meta = pack_sogst(args.pack, fields, meta['time_min'], meta['time_max'],
                            meta['fps'], shn_count=0 if args.no_sh else 4096,
                            generator='volumetric-capture-pipeline make_sogst_fixture')
        seg = pack_meta.get('segments')
        print(f"Packed {args.pack}: {os.path.getsize(args.pack) / 1024:.0f} KB, "
              f"persistent {pack_meta['segments']['persistent'][1]:,}" if seg else
              f"Packed {args.pack}")


if __name__ == '__main__':
    main()
