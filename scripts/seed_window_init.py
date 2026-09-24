#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
seed_window_init.py - start a window from the previous window's model.

Windows of a long clip are trained independently, so each one relearns the
subject's surface from a visual hull. That is most of what a window costs:
splat count fits 181k per window plus 1150k times its share of clip motion,
so the per-window term is paid whatever the window covers. It is also why
seams show, because two independently fitted models disagree about geometry
at the instant where one hands over to the other.

ATGS (SIGGRAPH 2026) makes the same observation about long sequences and
answers it with time-conditioned anchors: persistent handles whose features
vary with time, rather than primitives that each track long-range motion.
Its architecture does not fit here, because it decodes Gaussians through a
hash grid and an MLP at render time and nothing bakes to .sogst. The idea
does fit, in the form the trainer already accepts.

The obvious route, resuming from the previous window's checkpoint, is a trap.
train_scratch.py restores `first_iter` from the checkpoint (line 124) while
densification is gated on `iteration < densify_until_iter` (line 354), so a
resume at 30k lands past the densification window: the model inherits a full
splat set, never re-densifies, and saturates the cap immediately.

Seeding the point cloud avoids all of that. readNerfSyntheticInfo reads
points3d.ply and, when it holds more points than --num_pts, randomly
subsamples to that budget (scene/dataset_readers.py:311-330). Hand it a cloud
drawn from the previous window's trained splats and the next window starts
sparse but structurally correct, with the densification schedule running from
iteration 0 exactly as it does now. No trainer patch, no checkpoint surgery,
no cap.

The cloud is a mix, because the two halves know different things. The hull
carving is coarse but it is carved from THIS window's own masks and carries a
time per point spanning the window, so it covers the whole span. The seeded
half is an accurate surface, but only at the instant it was evaluated. Half
and half by default; --seed_fraction 1.0 and 0.0 give the pure variants.

Usage:
    python3 scripts/seed_window_init.py \\
        --model win_0/splat_4d.sogst --at_seconds 1.2917 \\
        --hull win_1/dataset_4dgs/points3d.ply \\
        --out win_1/dataset_4dgs/points3d.ply --window_seconds 1.2083
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_4dgs_dataset import write_ply_with_time  # noqa: E402
from eval_render import decode_sogst_fields  # noqa: E402

# Degree-0 spherical harmonic constant. A splat's base colour is stored as
# f_dc, and rgb = 0.5 + C0 * f_dc recovers it, which is the inverse of the
# RGB2SH the trainer applies to an init cloud.
SH_C0 = 0.28209479177387814


def temporal_alpha(fields, t):
    """A splat's opacity at time `t`, from its temporal Gaussian."""
    opacity = np.asarray(fields["opacity"], dtype=np.float64)
    centre = np.asarray(fields["t_center"], dtype=np.float64)
    sigma = np.abs(np.asarray(fields["t_sigma"], dtype=np.float64))
    peak = 1.0 / (1.0 + np.exp(-opacity))
    return peak * np.exp(-0.5 * ((t - centre) / np.maximum(sigma, 1e-9)) ** 2)


def advance(fields, t):
    """Where each splat sits at time `t`: xyz + v * (t - t_center).

    The inverse of the shift merge_sogst_segments.load_segment applies when
    it moves a window into global time, and the reason a splat evaluated far
    outside its own window streaks across the room."""
    centre = np.asarray(fields["t_center"], dtype=np.float64)
    delta = t - centre
    return np.stack([
        np.asarray(fields[axis], dtype=np.float64)
        + np.asarray(fields[vel], dtype=np.float64) * delta
        for axis, vel in (("x", "vx"), ("y", "vy"), ("z", "vz"))
    ], axis=1)


def splat_colours(fields):
    dc = np.stack([np.asarray(fields[f"f_dc_{i}"], dtype=np.float64)
                   for i in range(3)], axis=1)
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    return np.rint(rgb * 255.0).astype(np.uint8)


def seed_points(model, at_seconds, count, alpha_min=0.01, seed=0):
    """Draw `count` points from a trained window, at the handover instant.

    Sampling is weighted by each splat's opacity there, so the cloud follows
    what the model actually renders rather than counting invisible splats the
    densifier left behind."""
    _, fields = decode_sogst_fields(str(model))
    alpha = temporal_alpha(fields, at_seconds)
    live = alpha >= alpha_min
    if not live.any():
        raise SystemExit(
            f"no splat in {model} has opacity {alpha_min} or more at "
            f"t={at_seconds:.4f}s, so it renders nothing there. Check the "
            "handover time is inside the window's own span.")
    points = advance(fields, at_seconds)[live]
    colours = splat_colours(fields)[live]
    weights = alpha[live]
    weights = weights / weights.sum()

    rng = np.random.default_rng(seed)
    take = min(count, len(points))
    idx = rng.choice(len(points), size=take, replace=False, p=weights)
    return points[idx], colours[idx], int(live.sum())


def read_hull(path):
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    colours = np.stack([vertex["red"], vertex["green"], vertex["blue"]],
                       axis=1).astype(np.uint8)
    times = np.asarray(vertex["time"], dtype=np.float64) if "time" in \
        vertex.data.dtype.names else np.zeros(len(points))
    return points, colours, times


def build_seeded_cloud(model, hull_path, at_seconds, window_seconds, total,
                       seed_fraction=0.5, alpha_min=0.01, seed=0):
    """Mix a seeded half and a hull half into one init cloud.

    The seeded half can come up short: only the splats that actually render
    at the handover are eligible, and on the 5 s take that was 107k of one
    window's 575k. The hull makes up the difference rather than the cloud
    shrinking, because the count is what keeps the trainer on the branch that
    subsamples."""
    hull_points, hull_colours, hull_times = read_hull(hull_path)
    rng = np.random.default_rng(seed)

    wanted = int(round(total * seed_fraction))
    seeded = None
    live = 0
    if wanted > 0:
        points, colours, live = seed_points(model, at_seconds, wanted,
                                            alpha_min, seed)
        # Spread the seeded points across the new window rather than piling
        # them at its first instant: the trainer places each splat's t_center
        # from its point's `time`, so a cloud with one timestamp starts every
        # seeded splat centred on the handover and leaves the rest of the
        # window to densification alone.
        times = rng.uniform(0.0, window_seconds, size=len(points))
        seeded = (points, colours, times)

    n_seed = len(seeded[0]) if seeded else 0
    n_hull = total - n_seed
    parts = []
    if n_hull > 0:
        if len(hull_points) >= n_hull:
            idx = rng.choice(len(hull_points), size=n_hull, replace=False)
        else:                       # a short hull is topped up with repeats
            idx = rng.choice(len(hull_points), size=n_hull, replace=True)
        parts.append((hull_points[idx], hull_colours[idx], hull_times[idx]))
    if seeded:
        parts.append(seeded)

    points = np.concatenate([p for p, _, _ in parts], axis=0)
    colours = np.concatenate([c for _, c, _ in parts], axis=0)
    times = np.concatenate([t for _, _, t in parts], axis=0)
    return points, colours, times, {"seeded": n_seed, "hull": max(n_hull, 0),
                                    "requested_seed": wanted,
                                    "live_at_handover": live,
                                    "hull_available": len(hull_points)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="the previous window's splat_4d.sogst")
    ap.add_argument("--at_seconds", type=float, required=True,
                    help="handover instant, in the PREVIOUS window's local time")
    ap.add_argument("--window_seconds", type=float, required=True,
                    help="the new window's duration, over which seeded points "
                         "are spread in time")
    ap.add_argument("--hull", required=True,
                    help="the new window's existing points3d.ply")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=300_000,
                    help="points to write. Keep this ABOVE the trainer's "
                         "--num_pts: readNerfSyntheticInfo subsamples a longer "
                         "cloud and mishandles a shorter one (default 300000)")
    ap.add_argument("--seed_fraction", type=float, default=0.5,
                    help="share taken from the previous model rather than the "
                         "hull; 1.0 and 0.0 give the pure variants")
    ap.add_argument("--alpha_min", type=float, default=0.01,
                    help="ignore splats fainter than this at the handover")
    ap.add_argument("--num_pts", type=int, default=None,
                    help="the trainer's --num_pts, checked against --total")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not 0.0 <= args.seed_fraction <= 1.0:
        raise SystemExit("--seed_fraction must be between 0 and 1")
    if args.num_pts and args.total <= args.num_pts:
        raise SystemExit(
            f"--total {args.total} is not above --num_pts {args.num_pts}. The "
            "trainer only subsamples clouds LONGER than its budget; a shorter "
            "one takes the branch that mishandles it.")

    points, colours, times, report = build_seeded_cloud(
        args.model, args.hull, args.at_seconds, args.window_seconds,
        args.total, args.seed_fraction, args.alpha_min, args.seed)

    if args.num_pts and len(points) <= args.num_pts:
        raise SystemExit(
            f"the cloud came out at {len(points):,} points, not above "
            f"--num_pts {args.num_pts}. Raise --total or lower --num_pts.")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply_with_time(out, points, colours, times.astype(np.float32))

    print(f"seeded {out}")
    short = ("" if report["seeded"] >= report["requested_seed"]
             else f", short of {report['requested_seed']:,} asked for")
    print(f"  {report['seeded']:,} points from {Path(args.model).parent.name} at "
          f"t={args.at_seconds:.4f}s, where {report['live_at_handover']:,} "
          f"splats render{short}")
    print(f"  {report['hull']:,} points from the hull carving "
          f"({report['hull_available']:,} available)")
    print(f"  {len(points):,} total, spread over [0, {args.window_seconds:.4f}] s")


if __name__ == "__main__":
    main()
