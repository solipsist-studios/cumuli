#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
stitch_window_plys.py - stitch the raw window PLYs of a windowed training run
into one clip PLY, ready to seed a whole-clip fine-tune.

This is merge_sogst_segments.py's stitch (shift into global time, partition at
the seams, fade, gate the temporal tails) applied to the trainer's own
splat_4d.ply files instead of baked .sogst archives, and without repacking.

Usage:
    python3 stitch_window_plys.py \\
        --window win_00/train4d_output/splat_4d.ply 0.0 \\
        --window win_01/train4d_output/splat_4d.ply 1.2012 \\
        --out init_stitch.ply [--fade 0.35] [--gate_k 2.0] [--gate_floor 0.4] \\
        [--report_json stitch.json]

Each --window is a PLY and the time, in seconds, at which that window starts in
the clip. The output spans from the earliest start to the latest end.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import merge_sogst_segments as ms  # noqa: E402
from cumuli_core.ply import read_sogst_ply, write_sogst_ply  # noqa: E402


def load_window(path, offset, default_fps):
    header, fields = read_sogst_ply(str(path))
    fields = {k: np.asarray(v, np.float32).copy() for k, v in fields.items()}
    fields["t_center"] = fields["t_center"] + np.float32(offset)
    return {
        "path": str(path), "offset": offset, "fields": fields, "count": len(fields["x"]),
        "t_lo": float(header["time_min"]) + offset, "t_hi": float(header["time_max"]) + offset,
        "fps": float(header.get("fps") or default_fps),
    }


def stitch(segments, fade, gate_k, gate_floor):
    """The merged fields and the per-window counts kept."""
    seams = ms.default_seams(segments)
    kept, counts = [], []
    for segment, (lo, hi) in zip(segments, ms.slot_bounds(segments, seams)):
        mask, weight = ms.select_fade(segment, lo, hi, fade)
        fields = {k: (v[mask] if v.ndim == 1 else v[mask, :]) for k, v in segment["fields"].items()}
        fields = ms.apply_weight(fields, weight[mask])
        fields = ms.gate_tails(fields, lo, hi, gate_k, gate_floor, 0.0)
        kept.append(fields)
        counts.append(len(fields["x"]))
    merged = {k: np.concatenate([f[k] for f in kept], axis=0).astype(np.float32) for k in kept[0]}
    return merged, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", nargs=2, action="append", metavar=("PLY", "START_S"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fade", type=float, default=0.35, help="Seam fade width, seconds (default 0.35)")
    parser.add_argument("--gate_k", type=float, default=2.0)
    parser.add_argument("--gate_floor", type=float, default=0.4)
    parser.add_argument("--fps", type=float, default=30.0, help="Used only when a window's PLY carries no fps")
    parser.add_argument("--report_json", type=Path, default=None)
    args = parser.parse_args()

    windows = sorted(((Path(p), float(s)) for p, s in args.window), key=lambda w: w[1])
    for path, _ in windows:
        if not path.is_file():
            sys.exit(f"Error: {path} not found")
    segments = [load_window(p, s, args.fps) for p, s in windows]
    merged, counts = stitch(segments, args.fade, args.gate_k, args.gate_floor)
    t0, t1 = min(s["t_lo"] for s in segments), max(s["t_hi"] for s in segments)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = write_sogst_ply(str(args.out), merged, t0, t1, segments[0]["fps"], generator="stitch_window_plys")
    print(f"stitched {len(segments)} windows into {total} splats over {t0:.3f}..{t1:.3f} s -> {args.out}")
    if args.report_json:
        args.report_json.write_text(json.dumps({
            "windows": [{"path": s["path"], "start_s": s["offset"], "splats_in": s["count"], "splats_kept": n}
                        for s, n in zip(segments, counts)],
            "splats": int(total), "time_min": t0, "time_max": t1, "fps": segments[0]["fps"],
        }, indent=2))


if __name__ == "__main__":
    main()
