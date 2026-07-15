#!/usr/bin/env python3
"""
render_frame_sequence.py

Renders a SEQUENCE of trained Brush splats across multiple target_times from
the same static multi-camera rig, for stop-motion/flipbook-style 4D
playback -- independently-trained splats per frame, not a single
temporally-aware 4D representation.

Camera sync correction and pose calibration (transforms_refined.json) don't
depend on which moment in time is being rendered -- they describe the FIXED
physical rig, calibrated once from a representative time window. So unlike
run_unified_pipeline.py (one target_time, full sync+production+poses+
masks+branch pipeline every time), this script:
  1. Reuses an ALREADY-COMPLETED single-frame run_unified_pipeline.py
     run's sync offsets and transforms_refined.json (--calib_run_dir)
     instead of recomputing them.
  2. Loops ONLY the per-frame-dependent stages (production undistortion,
     masks, subject triangulation, training) once per resolved target time
     (--start_time/--stop_time/--fps), into out_dir/frame_<NNNN>/.

Reuses run_unified_pipeline.py's stage_production/stage_masks/
stage_branch_direct functions directly (imported as a regular module)
rather than reimplementing that orchestration: those functions are
already tested against this exact rig.

Before calling stage_masks(), manually applies run_hloc.py's
restructure_flat_to_percam() to this frame's production_undist dir --
build_flat_dataset.py expects that Camera_<id>/0000.ext layout, which
normally comes from run_hloc()'s side effect, but we're deliberately not
re-running HLOC here (the whole point is that poses are already fixed).

Training steps: pick --total_train_iters much lower than a hero-shot run
(script default is 30000, meant for one splat) -- training that many steps
per frame across even a short sequence multiplies out fast. A few thousand
is usually enough to validate the sequence looks right.

Usage:
    python3 render_frame_sequence.py \\
        --calib_run_dir /path/to/completed/single_frame_run \\
        --video_dir /path/to/movies \\
        --calib_dir /path/to/calibration_pkls \\
        --out_dir /path/to/flipbook_run \\
        --start_time 1.4s --stop_time 1.6s --fps 30 \\
        --total_train_iters 3000

Output:
    out_dir/frame_0000/brush_output/..., out_dir/frame_0001/brush_output/...
    one trained splat per resolved target time, in temporal order.
"""

import argparse
import sys
from pathlib import Path

import run_hloc as hloc_mod
import run_unified_pipeline as unified


def resolve_sync_json(calib_run_dir: Path) -> Path:
    """Same resolution order run_unified_pipeline.py's own resume logic uses."""
    resolved_path = calib_run_dir / "resolved_sync_json.txt"
    if resolved_path.exists():
        return Path(resolved_path.read_text().strip())
    return calib_run_dir / "sync_offsets.json"


def resolve_target_times(args):
    """Uniform --start_time/--stop_time/--fps range."""
    start = unified.parse_target_time(args.start_time)
    stop = unified.parse_target_time(args.stop_time)
    if args.fps <= 0:
        unified.fail("--fps must be positive")
        sys.exit(1)
    if stop < start:
        unified.fail(f"--stop_time ({args.stop_time}) is before --start_time ({args.start_time})")
        sys.exit(1)

    step = 1.0 / args.fps
    times = []
    t = start
    while t <= stop + step / 2:  # epsilon guards against float accumulation dropping the last frame
        times.append(round(t, 6))
        t += step
    return times


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib_run_dir", required=True, type=Path,
                        help="A completed run_unified_pipeline.py --out_dir to reuse sync + "
                             "transforms_refined.json from (camera poses don't change per frame)")
    parser.add_argument("--video_dir", required=True, type=Path)
    parser.add_argument("--calib_dir", required=True, type=Path, help="Native fisheye calibration PKLs")
    parser.add_argument("--target_pkl_dir", type=Path, default=None)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--start_time", required=True,
                        help="Start of the uniform time range (requires --stop_time and --fps), e.g. '1.4s'.")
    parser.add_argument("--stop_time", required=True,
                        help="End (inclusive) of the uniform time range -- same form as --start_time.")
    parser.add_argument("--fps", type=float, required=True,
                        help="Frame rate for the --start_time/--stop_time range, e.g. 30.")
    parser.add_argument("--pp3_dir", type=Path, default=None)

    parser.add_argument("--sapiens_env", default="sapiens2")
    parser.add_argument("--keypoint_model", choices=["goliath308", "coco_wholebody133"], default="goliath308")
    parser.add_argument("--sapiens_checkpoint_root", type=Path, default=None)
    parser.add_argument("--triangulate_env", default="queen")
    parser.add_argument("--generic_env", default="queen")
    parser.add_argument("--brush_app", type=Path, default=Path.home() / "brush-app-x86_64-unknown-linux-gnu" / "brush_app")
    parser.add_argument("--total_train_iters", type=int, default=3000,
                        help="Per-frame step count -- keep well below a single hero-shot run's "
                             "30000, this multiplies by the number of frames (default 3000)")
    parser.add_argument("--export_every", type=int, default=5000)
    parser.add_argument("--with_viewer", dest="with_viewer", action="store_true", default=True,
                        help="See run_unified_pipeline.py's --with_viewer help. On by default "
                             "for the same reason. See --no_viewer.")
    parser.add_argument("--no_viewer", dest="with_viewer", action="store_false",
                        help="See run_unified_pipeline.py's --no_viewer help.")
    parser.add_argument("--display", default=":2",
                        help="See run_unified_pipeline.py's --display help.")
    parser.add_argument("--run_name", default=None, help="Prefix for output filenames (default: out_dir's name)")
    return parser


def main():
    args = build_parser().parse_args()
    args.run_name = args.run_name or args.out_dir.name
    image_ext = ".png" if args.pp3_dir else ".jpg"

    times = resolve_target_times(args)
    if not times:
        unified.fail("Resolved to zero target times")
        sys.exit(1)

    transforms_refined = args.calib_run_dir / "transforms_refined.json"
    if not transforms_refined.is_file():
        unified.fail(f"{transforms_refined} not found -- is --calib_run_dir a completed "
                     "run_unified_pipeline.py run?")
        sys.exit(1)

    sync_json = resolve_sync_json(args.calib_run_dir)
    if not sync_json.is_file():
        unified.fail(f"Could not find resolved sync offsets under {args.calib_run_dir}")
        sys.exit(1)

    if not args.video_dir.is_dir():
        unified.fail(f"--video_dir {args.video_dir} is not a directory")
        sys.exit(1)

    videos = unified.discover_cameras(args.video_dir)
    n_real = len(videos)

    unified.banner(f"FRAME SEQUENCE RENDER -- {len(times)} frames, {n_real} cameras")
    unified.info(f"Reusing calibration from {args.calib_run_dir}")
    unified.info(f"  transforms_refined: {transforms_refined}")
    unified.info(f"  sync offsets: {sync_json}")

    base_run_name = args.run_name
    for i, t in enumerate(times):
        frame_tag = f"frame_{i:04d}"
        unified.banner(f"{frame_tag} (t={t:.3f}s) -- {i + 1}/{len(times)}")

        frame_dir = args.out_dir / frame_tag
        frame_dir.mkdir(parents=True, exist_ok=True)
        L = unified.build_layout(frame_dir)
        L["transforms_refined"] = transforms_refined  # fixed calibration, not per-frame

        args.target_time_s = t
        args.run_name = f"{base_run_name}_{frame_tag}"

        try:
            unified.stage_production(args, L, image_ext, sync_json)
            hloc_mod.restructure_flat_to_percam(L["production_undist"], image_ext)
            unified.stage_masks(args, L, image_ext, n_real)
            unified.stage_branch_direct(args, L)
        except unified.StageError as e:
            unified.fail(str(e))
            unified.fail(f"{frame_tag} failed -- stopping sequence ({i}/{len(times)} frames "
                         "completed before this one). Re-run with a narrower --start_time/--stop_time "
                         "range to resume from here.")
            sys.exit(1)

    unified.banner("FRAME SEQUENCE COMPLETE")
    unified.ok(f"{len(times)} trained splat(s) under {args.out_dir}/frame_*/brush_output/")


if __name__ == "__main__":
    main()
