#!/usr/bin/env python3
"""
20_refine_poses_with_keypoints.py

Wraps ~/4dgs-utils/refine_poses_with_keypoints.py ("human as calibration
wand"): bundle-adjusts camera poses against 2D human keypoints across
several time instants, anchoring accuracy at the subject where background
SfM features (HLOC, stage 05) don't reach. Background-only poses can look
great on distant walls/windows yet disagree by tens of pixels at the
capture volume center -- exactly where the subject moves.

IMPORTANT: use keypoints from MANY time instants (10+), not one -- with a
single instant the optimizer just absorbs sync error into the poses
instead of correcting real pose error. Produce candidate instants with
02_extract_synced_frames.py --window N (ideally already sync-corrected via
19_skeleton_sync_search.py), then run stages 04/08/09 on each f{k}/ dir.

This wrapper merges those per-instant poses_2d/<camera_label>/000000.json
directories (each stage 09 run defaults to the same "000000" temporal
label) into one directory with distinct temporal labels per instant, then
invokes the underlying refine script.

conda env: none beyond numpy/scipy (same as the underlying script).

Usage:
    python3 20_refine_poses_with_keypoints.py \\
        --transforms /path/to/solipsist_out/transforms_multiframe.json \\
        --kp2d_dirs /path/f0/poses_2d,/path/f1/poses_2d,/path/f2/poses_2d,... \\
        --out_transforms /path/to/transforms_refined.json \\
        [--refine_script ~/4dgs-utils/refine_poses_with_keypoints.py] \\
        [--report_only]

Output:
    out_transforms (refined transforms.json, same frames/intrinsics,
    updated transform_matrix). Point every later stage at this file
    instead of the un-refined transforms_multiframe.json.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_REFINE_SCRIPT = Path.home() / "4dgs-utils" / "refine_poses_with_keypoints.py"


def merge_instants(kp2d_dirs, merged_dir: Path):
    """poses_2d/<cam>/000000.json (per instant dir) -> merged_dir/<cam>/{tem}.json,
    tem = f'{instant_idx:06d}'. Returns set of camera labels found."""
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True)

    cam_labels = set()
    for k, kp2d_dir in enumerate(kp2d_dirs):
        cam_dirs = sorted(p for p in kp2d_dir.iterdir() if p.is_dir())
        if not cam_dirs:
            print(f"  WARNING: no per-camera subdirs found in {kp2d_dir}")
            continue
        for cam_dir in cam_dirs:
            jsons = sorted(cam_dir.glob("*.json"))
            if not jsons:
                continue
            out_cam_dir = merged_dir / cam_dir.name
            out_cam_dir.mkdir(exist_ok=True)
            shutil.copy(jsons[0], out_cam_dir / f"{k:06d}.json")
            cam_labels.add(cam_dir.name)
    return cam_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transforms", required=True, type=Path)
    parser.add_argument("--kp2d_dirs", required=True,
                        help="Comma-separated poses_2d dirs, one per time instant (stage 09 output layout)")
    parser.add_argument("--out_transforms", required=True, type=Path)
    parser.add_argument("--merged_dir", type=Path, default=None,
                        help="Where to write the merged multi-instant keypoint dir (default: alongside out_transforms)")
    parser.add_argument("--refine_script", type=Path, default=DEFAULT_REFINE_SCRIPT)
    parser.add_argument("--score_thr", type=float, default=0.5)
    parser.add_argument("--min_views", type=int, default=3)
    parser.add_argument("--huber_px", type=float, default=8.0)
    parser.add_argument("--outlier_px", type=float, default=150.0,
                        help="Drop observations above this initial reprojection error (sync glitches, misdetections)")
    parser.add_argument("--max_iters", type=int, default=300)
    parser.add_argument("--report_only", action="store_true",
                        help="Only measure subject-space reprojection error, don't optimize (run this first)")
    args = parser.parse_args()

    if not args.refine_script.is_file():
        print(f"Error: {args.refine_script} not found. Pass --refine_script or clone 4dgs-utils to ~/4dgs-utils.")
        sys.exit(1)

    kp2d_dirs = [Path(p) for p in args.kp2d_dirs.split(",")]
    for d in kp2d_dirs:
        if not d.is_dir():
            print(f"Error: {d} is not a directory")
            sys.exit(1)
    if len(kp2d_dirs) < 5:
        print(f"WARNING: only {len(kp2d_dirs)} time instant(s) given -- with too few instants the "
              "optimizer absorbs sync error into the poses instead of correcting real pose error. "
              "10+ instants (sync-corrected via 19_skeleton_sync_search.py) is recommended.")

    merged_dir = args.merged_dir or (args.out_transforms.parent / "poses_2d_merged")
    print(f"Merging {len(kp2d_dirs)} instant(s) into {merged_dir} ...")
    cam_labels = merge_instants(kp2d_dirs, merged_dir)
    print(f"  {len(cam_labels)} cameras: {sorted(cam_labels)}")

    args.out_transforms.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(args.refine_script),
        "--transforms", str(args.transforms),
        "--kp2d", str(merged_dir),
        "--out_transforms", str(args.out_transforms),
        "--score_thr", str(args.score_thr),
        "--min_views", str(args.min_views),
        "--huber_px", str(args.huber_px),
        "--outlier_px", str(args.outlier_px),
        "--max_iters", str(args.max_iters),
    ]
    if args.report_only:
        cmd.append("--report_only")

    print("\nRunning:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
