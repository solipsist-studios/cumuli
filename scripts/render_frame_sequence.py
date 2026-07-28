#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

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
  1. Computes sync + HLOC pose estimation (run_unified_pipeline.py's
     stage_sync/stage_production/stage_poses) only ONCE, at --calib_time
     (defaults to --start_time), writing transforms_refined.json/
     sync_offsets.json directly under --out_dir -- reused automatically
     on any later invocation with the same --out_dir. If you already have
     a completed run_unified_pipeline.py run for this rig, pass its
     --out_dir as --calib_run_dir instead to reuse that one directly
     rather than recomputing.
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

Usage (no prior calibration run -- computes sync + HLOC pose estimation once):
    python3 render_frame_sequence.py \\
        --video_dir /path/to/movies \\
        --calib_dir /path/to/calibration_pkls \\
        --out_dir /path/to/flipbook_run \\
        --start_time 1.4s --stop_time 1.6s --fps 30 \\
        --total_train_iters 3000 \\
        [--calib_time 1.5s]  # defaults to --start_time \\
        [--initial_sync_json /path/to/sync_offsets_v5.json]  # skip audio sync \\
        [--config /path/to/rig.json]  # same per-rig file run_unified_pipeline.py takes

Usage (reuse an already-completed run_unified_pipeline.py run instead):
    python3 render_frame_sequence.py \\
        --calib_run_dir /path/to/completed/single_frame_run \\
        ... (same flags as above, minus --calib_time/--initial_sync_json)

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


def compute_or_reuse_calibration(args, image_ext):
    """When --calib_run_dir isn't given, compute sync + HLOC pose estimation
    once directly under --out_dir instead of requiring a separate
    already-completed run_unified_pipeline.py run. Reuses
    run_unified_pipeline.py's own stage_sync/stage_production/stage_poses
    (and their internal on-disk resume checks) rather than reimplementing
    that orchestration, exactly like the per-frame loop below does for
    stage_masks/stage_branch_direct.

    Safe to call again with the same --out_dir: if transforms_refined.json
    is already there from a previous invocation, it's reused as-is."""
    calib_L = unified.build_layout(args.out_dir)

    if calib_L["transforms_refined"].is_file():
        unified.info(f"{calib_L['transforms_refined']} already present under --out_dir -- "
                     "reusing previously computed calibration.")
        return calib_L["transforms_refined"], resolve_sync_json(args.out_dir)

    unified.banner("STAGE: CALIBRATION (sync + HLOC pose estimation, once for this rig)")
    args.target_time_s = unified.parse_target_time(
        args.calib_time if args.calib_time is not None else args.start_time)

    sync_json = unified.stage_sync(args, calib_L, image_ext)
    with open(args.out_dir / "resolved_sync_json.txt", "w") as f:
        f.write(str(sync_json))

    unified.stage_production(args, calib_L, image_ext, sync_json)
    unified.stage_poses(args, calib_L, image_ext, sync_json)

    return calib_L["transforms_refined"], sync_json


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
    parser.add_argument("--config", type=Path, default=None,
                        help="Same per-rig JSON run_unified_pipeline.py takes (conda env names, "
                             "--brush_app, --display, SAPIENS_CHECKPOINT_ROOT -- see "
                             "configs/example_rig.json). Explicit CLI flags still override the config.")
    parser.add_argument("--calib_run_dir", type=Path, default=None,
                        help="A completed run_unified_pipeline.py --out_dir to reuse sync + "
                             "transforms_refined.json from (camera poses don't change per frame). "
                             "If omitted, sync + HLOC pose estimation are computed once under "
                             "--out_dir instead -- see --calib_time.")
    parser.add_argument("--calib_time", default=None,
                        help="Representative timestamp for the one-time sync + HLOC pose estimation "
                             "(only used when --calib_run_dir is omitted), e.g. '1.5s'. Defaults to "
                             "--start_time.")
    parser.add_argument("--initial_sync_json", type=Path, default=None,
                        help="Pre-existing sync_offsets.json to seed calibration with instead of "
                             "recomputing audio cross-correlation (only used when --calib_run_dir is "
                             "omitted -- see run_unified_pipeline.py's flag of the same name).")
    parser.add_argument("--sync_window", type=int, default=5,
                        help="Number of candidate frames for pose refinement (only used when "
                             "--calib_run_dir is omitted -- see run_unified_pipeline.py's flag of "
                             "the same name, default 5).")
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

    parser.add_argument("--multiframe_sfm_script", type=Path, default=unified.DEFAULT_MULTIFRAME_SFM_SCRIPT,
                        help="Path to multiframe_sfm.py (only used when --calib_run_dir is omitted).")
    parser.add_argument("--hloc_feature_type", default="superpoint", choices=["superpoint", "aliked"],
                        help="See run_unified_pipeline.py's flag of the same name (only used when "
                             "--calib_run_dir is omitted).")
    parser.add_argument("--hloc_resize_max", type=int, default=4096,
                        help="See run_unified_pipeline.py's flag of the same name (only used when "
                             "--calib_run_dir is omitted).")
    parser.add_argument("--hloc_max_keypoints", type=int, default=8192,
                        help="See run_unified_pipeline.py's flag of the same name (only used when "
                             "--calib_run_dir is omitted).")

    parser.add_argument("--sapiens_env", default="sapiens2")
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
    parser = build_parser()
    unified.apply_config_defaults(parser)
    args = parser.parse_args()
    args.run_name = args.run_name or args.out_dir.name
    image_ext = ".png" if args.pp3_dir else ".jpg"

    times = resolve_target_times(args)
    if not times:
        unified.fail("Resolved to zero target times")
        sys.exit(1)

    if not args.video_dir.is_dir():
        unified.fail(f"--video_dir {args.video_dir} is not a directory")
        sys.exit(1)
    if not args.calib_dir.is_dir():
        unified.fail(f"--calib_dir {args.calib_dir} is not a directory")
        sys.exit(1)
    if args.target_pkl_dir and not args.target_pkl_dir.is_dir():
        unified.fail(f"--target_pkl_dir {args.target_pkl_dir} is not a directory")
        sys.exit(1)

    if args.calib_run_dir is not None:
        transforms_refined = args.calib_run_dir / "transforms_refined.json"
        if not transforms_refined.is_file():
            unified.fail(f"{transforms_refined} not found -- is --calib_run_dir a completed "
                         "run_unified_pipeline.py run?")
            sys.exit(1)
        sync_json = resolve_sync_json(args.calib_run_dir)
        if not sync_json.is_file():
            unified.fail(f"Could not find resolved sync offsets under {args.calib_run_dir}")
            sys.exit(1)
    else:
        if not args.multiframe_sfm_script.is_file():
            unified.fail(f"--multiframe_sfm_script {args.multiframe_sfm_script} not found "
                         "(pass --multiframe_sfm_script to point at a different copy)")
            sys.exit(1)
        if args.initial_sync_json and not args.initial_sync_json.is_file():
            unified.fail(f"--initial_sync_json {args.initial_sync_json} not found")
            sys.exit(1)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            transforms_refined, sync_json = compute_or_reuse_calibration(args, image_ext)
        except unified.StageError as e:
            unified.fail(str(e))
            unified.fail("Calibration failed -- re-run once that's fixed (completed calibration "
                         "sub-stages are reusable via their own on-disk resume checks).")
            sys.exit(1)

    try:
        videos = unified.discover_cameras(args.video_dir)
    except unified.StageError as e:
        unified.fail(str(e))
        sys.exit(1)
    n_real = len(videos)

    unified.banner(f"FRAME SEQUENCE RENDER -- {len(times)} frames, {n_real} cameras")
    if args.calib_run_dir is not None:
        unified.info(f"Reusing calibration from {args.calib_run_dir}")
    else:
        unified.info(f"Using calibration computed under {args.out_dir}")
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
            unified.stage_masks(args, L, image_ext)
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
