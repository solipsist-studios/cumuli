#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
run_window_plan.py - train a clip's windows, stitch them, score the result.

plan_temporal_windows.py decides where a long clip is cut;
merge_sogst_segments.py puts the pieces back together. This runs everything
between: it points each window at its slice of an already-rendered master
run, builds its dataset, trains it, then merges and evaluates.

Nothing is re-rendered. A window's flipbook is symlinks into the master's,
renumbered from zero, which is the layout run_synthetic_pipeline.py's
dataset4d stage expects (`<out_dir>/flipbook_src/frame_NNNN`). Rendering a
121-frame 12-camera take costs hours; slicing it costs nothing, so a window
plan can be re-cut and retrained as often as the question needs.

With --seed, each window after the first starts from its predecessor's
trained model through seed_window_init.py instead of from the visual hull
alone. That needs the dataset built before training rather than in one pass,
so those windows run dataset4d and train4d as two calls with the seeding in
between.

Usage:
    python3 scripts/run_window_plan.py --plan window_plan.json \\
        --master ~/runs/ring12_5s --blend ~/assets/ariana_packed.blend \\
        --rig_spec configs/rigs/ring12.json --merge_out ~/runs/planned.sogst
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PYTHON = sys.executable


def info(message):
    print(f"[window-plan] {message}", flush=True)


def run(command, label):
    printable = " ".join(str(c) for c in command)
    info(f"{label}: {printable}")
    started = time.time()
    result = subprocess.run([str(c) for c in command])
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")
    info(f"{label} finished in {(time.time() - started) / 60:.1f} min")


def link_frames(master, out_dir, first, count, subdirs=("flipbook_src", "eval_src")):
    """Point a window's frame directories at the master's, renumbered from 0.

    Symlinks rather than copies: the images are the expensive artifact and
    every window plan over the same take shares them."""
    for sub in subdirs:
        source_root = master / sub
        if not source_root.is_dir():
            continue
        target_root = out_dir / sub
        if target_root.exists():
            shutil.rmtree(target_root) if not target_root.is_symlink() \
                else target_root.unlink()
        target_root.mkdir(parents=True)
        for i in range(count):
            source = source_root / f"frame_{first + i:04d}"
            if not source.exists():
                raise SystemExit(
                    f"{source} is missing: the plan asks for frames "
                    f"{first}..{first + count - 1} but the master run does not "
                    "hold them all")
            (target_root / f"frame_{i:04d}").symlink_to(source.resolve())
    info(f"linked frames {first}..{first + count - 1} into {out_dir.name}")


def pipeline_command(args, out_dir, frame_start, frame_count, start, stop):
    command = [
        PYTHON, SCRIPT_DIR / "run_synthetic_pipeline.py",
        "--blend", args.blend, "--rig_spec", args.rig_spec,
        "--out_dir", out_dir,
        "--frame_start", str(frame_start), "--frame_count", str(frame_count),
        "--dataset_downscale", str(args.dataset_downscale),
        "--total_train_iters", str(args.total_train_iters),
        "--densify_until_iter", str(args.densify_until_iter),
        "--num_pts", str(args.num_pts),
        "--eval_every", str(args.eval_every),
        "--start_from_stage", start,
    ]
    if stop:
        command += ["--stop_after_stage", stop]
    if args.extra:
        command += args.extra
    return command


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True,
                    help="window_plan.json from plan_temporal_windows.py")
    ap.add_argument("--master", required=True,
                    help="the run holding the rendered frames and the eval GT")
    ap.add_argument("--blend", required=True)
    ap.add_argument("--rig_spec", required=True)
    ap.add_argument("--seed", action="store_true",
                    help="start each window after the first from its "
                         "predecessor's trained model")
    ap.add_argument("--seed_fraction", type=float, default=0.5)
    ap.add_argument("--total_train_iters", type=int, default=30000)
    ap.add_argument("--densify_until_iter", type=int, default=25000)
    ap.add_argument("--num_pts", type=int, default=200000)
    ap.add_argument("--dataset_downscale", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--merge_out", default=None,
                    help="stitch the windows here when they are all trained")
    ap.add_argument("--merge_args", nargs="*", default=None,
                    help="extra flags for merge_sogst_segments.py")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--skip_trained", action="store_true",
                    help="leave windows that already hold a splat_4d.sogst")
    ap.add_argument("--extra", nargs="*", default=None,
                    help="extra flags passed through to every pipeline call")
    args = ap.parse_args()

    plan_path = Path(args.plan).expanduser()
    plan = json.loads(plan_path.read_text())
    master = Path(args.master).expanduser()
    windows = plan["windows"]
    if any(not w.get("out_dir") for w in windows):
        raise SystemExit(f"{plan_path} has windows with no out_dir; re-plan "
                         "with --out_dir_template")

    info(f"{len(windows)} windows from {plan_path.name}, master {master.name}")
    for w in windows:
        info(f"  window {w['index']}: frames {w['local_frame_start']}.."
             f"{w['local_frame_start'] + w['frame_count'] - 1} "
             f"({w['frame_count']}), offset {w['offset_seconds']:.4f}s "
             f"-> {w['out_dir']}")

    fps = float(plan.get("fps") or 24.0)
    started = time.time()
    for i, window in enumerate(windows):
        out_dir = Path(window["out_dir"]).expanduser()
        model = out_dir / "splat_4d.sogst"
        if args.skip_trained and model.is_file():
            info(f"window {i}: already trained, skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        link_frames(master, out_dir, window["local_frame_start"],
                    window["frame_count"])

        seeding = args.seed and i > 0
        stop = "dataset4d" if seeding else None
        run(pipeline_command(args, out_dir, window["frame_start"],
                             window["frame_count"], "dataset4d", stop),
            f"window {i} dataset4d" if seeding else f"window {i}")

        if seeding:
            previous = Path(windows[i - 1]["out_dir"]).expanduser()
            # The handover is this window's first instant, expressed in the
            # PREVIOUS window's local time, and clamped into the span that
            # window actually covers: seeding from an instant it never saw
            # extrapolates every splat and reads back as noise.
            previous_span = (windows[i - 1]["frame_count"] - 1) / fps
            handover = min(window["offset_seconds"]
                           - windows[i - 1]["offset_seconds"], previous_span)
            hull = out_dir / "dataset_4dgs" / "points3d.ply"
            run([PYTHON, SCRIPT_DIR / "seed_window_init.py",
                 "--model", previous / "splat_4d.sogst",
                 "--at_seconds", f"{handover:.6f}",
                 "--window_seconds", f"{(window['frame_count'] - 1) / fps:.6f}",
                 "--hull", hull, "--out", hull,
                 "--seed_fraction", str(args.seed_fraction),
                 "--num_pts", str(args.num_pts)],
                f"window {i} seed from window {i - 1}")
            run(pipeline_command(args, out_dir, window["frame_start"],
                                 window["frame_count"], "train4d", None),
                f"window {i} train4d")

    info(f"all windows done in {(time.time() - started) / 60:.0f} min")

    if args.merge_out:
        merge_out = Path(args.merge_out).expanduser()
        command = [PYTHON, SCRIPT_DIR / "merge_sogst_segments.py",
                   "--plan", plan_path, "--out", merge_out,
                   "--report_json", merge_out.with_suffix(".stitch.json")]
        if args.merge_args:
            command += args.merge_args
        run(command, "merge")

        if not args.skip_eval:
            dataset = master / "dataset_4dgs"
            run([PYTHON, SCRIPT_DIR / "eval_render.py",
                 "--model", merge_out,
                 "--transforms", dataset / "transforms_test.json",
                 "--gt-dir", dataset / "eval_gt_flat",
                 "--downscale", "1", "--every", "1",
                 "--report_json", merge_out.with_suffix(".eval.json")],
                "eval stitched")


if __name__ == "__main__":
    main()
