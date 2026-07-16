#!/usr/bin/env python3
"""
run_unified_pipeline.py

Unified entry point for the volumetric capture pipeline: sync -> undistort
-> pose solve/refine -> mask -> direct 4K masked training -> train Brush.
Orchestrates the other scripts in this directory via subprocess; it does
not reimplement any of their logic. (A Diffuman4D 48-camera dense-ring
branch is planned but not wired in yet.)

--------------------------------------------------------------------------
STAGE ORDERING
--------------------------------------------------------------------------
Pose refinement needs a multi-instant keypoint set RE-EXTRACTED at the
verified sync (the historically-validated approach), not computed
alongside sync itself. So the actual flow is:

  1. SYNC: compute_sync_offsets.py (rough audio offsets, or
     --initial_sync_json to skip it) -> extract_synced_frames.py (extract
     one frame) -> make_sync_grid.py (QA grid) -> sync_offsets.json as-is.
     NOT auto-corrected -- inspect the grid, hand-tune any camera's
     frame_offset that's visibly off, and pass the result back in via
     --initial_sync_json on the next run.
  2. PRODUCTION: re-extract the single production frame at target_time
     using that sync -> undistort_frames.py single-warp undistort (native
     fisheye straight to the 4K pinhole target, calibration-scale
     validated)
  3. POSES: extract + process a candidate window at that same sync
     (undistort_frames.py/generate_masks.py/predict_keypoints_2d.py/
     split_keypoints_per_camera.py) -> run_hloc.py (final HLOC on the
     production frame) -> run_pose_refinement.py (bundle-adjust against
     the multi-instant keypoints) -> transforms_refined.json
  4. MASKS: build_flat_dataset.py (flatten labels to 2-digit) ->
     generate_masks.py (BiRefNet) -> predict_keypoints_2d.py/
     split_keypoints_per_camera.py (keypoints on the flat images, needed
     by mask cleanup/triangulation) -> clean_masks.py (skeleton-guided
     cleanup, retry fallback, no dilation) ->
     triangulate_and_project_keypoints.py (triangulate subject point
     cloud against the real cameras only)
  5. BRANCH: build_colmap_sparse.py (COLMAP sparse + --masks_dir RGBA
     bake) -> train_brush.py (train, 4096 res; Brush auto-detects the
     alpha channel baked in by build_colmap_sparse.py, no special flag
     needed -- brush_app has no --alpha-mode flag, verified against
     `brush_app --help`). This is the only branch this build supports;
     the Diffuman4D 48-camera dense-ring branch isn't wired in yet.

Usage (see --help for every flag):
    python3 run_unified_pipeline.py \\
        --video_dir /media/ai/datasets/CAPTURE/movies \\
        --calib_dir /path/to/calibration_pkls \\
        --out_dir ~/capture_run \\
        --target_time 1500ms \\
        [--target_pkl_dir /path/to/4k_undistorted]  # omit to derive undistorted intrinsics on the fly \\
        [--initial_sync_json /path/to/sync_offsets_v5.json]  # skip audio sync, seed from a known-good file
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from image_formats import SUPPORTED_VIDEO_EXTS

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_MULTIFRAME_SFM_SCRIPT = SCRIPTS_DIR / "multiframe_sfm.py"

STAGE_KEYS = ["sync", "production", "poses", "masks", "branch"]

CONDA_ENV_HLOC = "hloc"
CONDA_ENV_DIFFUMAN4D = "diffuman4d"

_CONDA_CANDIDATES = [
    Path.home() / "miniforge3" / "condabin" / "conda",
    Path.home() / "miniforge3" / "bin" / "conda",
    Path.home() / "miniconda3" / "condabin" / "conda",
    Path.home() / "anaconda3" / "condabin" / "conda",
    Path("/opt/conda/condabin/conda"),
]


def resolve_conda():
    """`conda` is frequently a shell function set up by `conda init`, not a
    plain PATH executable -- shutil.which can miss it even in a shell where
    `conda ...` works interactively, and subprocess (no shell) can't see
    shell functions at all. Fall back to well-known install locations."""
    found = shutil.which("conda")
    if found:
        return found
    for candidate in _CONDA_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    raise StageError(
        "Could not locate the 'conda' executable (checked PATH and "
        f"{[str(c) for c in _CONDA_CANDIDATES]}). Pass its path via --conda_bin.")


# --------------------------------------------------------------- console UI
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"


def banner(title):
    line = f" {title} "
    print(f"\n{C.BOLD}{C.CYAN}{line.center(78, '=')}{C.RESET}")


def info(msg):
    print(f"{C.BLUE}[info]{C.RESET} {msg}")


def ok(msg):
    print(f"{C.GREEN}[ok]{C.RESET} {msg}")


def warn(msg):
    print(f"{C.YELLOW}[warn]{C.RESET} {msg}")


def fail(msg):
    print(f"{C.RED}{C.BOLD}[FAIL]{C.RESET} {msg}")


class StageError(RuntimeError):
    pass


def run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
    """Invoke one pipeline script as a subprocess. Always passes a FRESH copy
    of os.environ (plus extra_env) to the child -- the orchestrator's own
    process environment is never mutated, so conda envs never leak between
    stages or back into this process."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.is_file():
        raise StageError(f"{script_name} not found at {script_path}")

    str_args = [str(a) for a in args]
    if conda_env:
        cmd = [resolve_conda(), "run", "-n", conda_env, "--no-capture-output",
               "python3", str(script_path)] + str_args
    else:
        cmd = [sys.executable, str(script_path)] + str_args

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    tag = label or script_name
    env_suffix = f"  {C.YELLOW}[conda: {conda_env}]{C.RESET}" if conda_env else ""
    info(f"$ {' '.join(cmd)}{env_suffix}")

    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if result.returncode != 0:
        fail(f"{tag} exited with code {result.returncode} -- stopping pipeline.")
        raise StageError(f"{tag} failed (exit {result.returncode})")
    ok(f"{tag} complete")


def check_expected(path: Path, stage_key: str):
    if not path.exists():
        warn(f"Resuming at --start_from_stage {stage_key!r} but expected output "
             f"{path} does not exist. This stage may fail.")


# --------------------------------------------------------------- misc helpers
def parse_target_time(value: str) -> float:
    """'1.5' / '1.5s' / '1500ms' -> seconds as float."""
    v = value.strip().lower()
    if v.endswith("ms"):
        return float(v[:-2]) / 1000.0
    if v.endswith("s"):
        return float(v[:-1])
    return float(v)


def discover_cameras(video_dir: Path):
    videos = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in SUPPORTED_VIDEO_EXTS)
    if not videos:
        raise StageError(f"No videos found in {video_dir}")
    return videos


def build_layout(out_dir: Path):
    L = {
        "sync_offsets": out_dir / "sync_offsets.json",

        "cand_raw": out_dir / "sync_candidates_raw",
        "sync_grid": out_dir / "sync_grid.jpg",

        "cand_corr": out_dir / "sync_candidates_corrected",
        "cand_corr_undist": out_dir / "sync_candidates_corrected_undist",
        "cand_corr_pkls": out_dir / "sync_candidates_corrected_pkls",
        "cand_corr_fmasks": out_dir / "sync_candidates_corrected_fmasks",
        "cand_corr_kp2d": out_dir / "sync_candidates_corrected_kp2d",
        "cand_corr_poses2d": out_dir / "sync_candidates_corrected_poses2d",

        "production_raw": out_dir / "production_raw",
        "production_undist": out_dir / "production_undistorted",
        "production_pkls": out_dir / "production_undistorted_pkls",

        "hloc_final": out_dir / "hloc_final",
        "transforms_refined": out_dir / "transforms_refined.json",

        "flat_images": out_dir / "images_flat",
        "flat_transforms": out_dir / "transforms.json",
        "flat_label_map": out_dir / "camera_label_map.json",
        "flat_fmasks": out_dir / "fmasks_flat",
        "flat_fmasks_clean": out_dir / "fmasks_clean",
        "flat_kp2d": out_dir / "poses_2d_flat",
        "flat_poses2d": out_dir / "poses_2d",
        "poses_pcd_fullres": out_dir / "poses_pcd_fullres",
        "poses_3d_fullres": out_dir / "poses_3d_fullres",

        "train_set": out_dir / "train_set",
        "brush_output": out_dir / "brush_output",

    }
    return L


# ------------------------------------------------------- candidate window prep
def prepare_candidate_window(args, L, sync_json: Path, window: int, image_ext: str,
                              raw_dir, undist_dir, pkl_dir, fmasks_dir, kp2d_dir, poses2d_dir,
                              tag: str, start_time_s: float = None):
    """Extract a --window-frame candidate instant set starting at start_time_s
    (defaults to target_time_s) using sync_json, then run
    undistort/masks/keypoints/split on each instant. Returns the list of
    per-instant poses_2d dirs, in order."""
    if start_time_s is None:
        start_time_s = args.target_time_s
    extract_args = [str(args.video_dir), str(sync_json), str(raw_dir), str(start_time_s),
                    "--window", str(window)]
    if args.pp3_dir:
        extract_args += ["--pp3_dir", str(args.pp3_dir)]
    run_script("extract_synced_frames.py", extract_args, label=f"extract_synced_frames.py (candidates, {tag})")

    poses2d_dirs = []
    for k in range(window):
        f_raw, f_undist, f_pkl = raw_dir / f"f{k}", undist_dir / f"f{k}", pkl_dir / f"f{k}"
        f_fmask, f_kp2d, f_poses2d = fmasks_dir / f"f{k}", kp2d_dir / f"f{k}", poses2d_dir / f"f{k}"

        if f_poses2d.is_dir() and any(f_poses2d.rglob("*.json")):
            info(f"f{k} ({tag}) already fully processed on disk -- reusing "
                 f"(delete {f_poses2d} to force a redo)")
            poses2d_dirs.append(f_poses2d)
            continue

        run_script("undistort_frames.py", undistort_args(args, f_raw, f_undist, f_pkl, image_ext),
                   conda_env=args.generic_env, label=f"undistort_frames.py (candidates f{k}, {tag})")

        run_script("generate_masks.py", [
            "--images_dir", f_undist, "--out_fmasks_dir", f_fmask, "--image_ext", image_ext,
        ], conda_env=CONDA_ENV_DIFFUMAN4D, label=f"generate_masks.py (candidates f{k}, {tag})")

        run_script("predict_keypoints_2d.py", keypoint_args(args, f_undist, f_kp2d, f_fmask),
                   conda_env=args.sapiens_env, label=f"predict_keypoints_2d.py (candidates f{k}, {tag})")

        run_script("split_keypoints_per_camera.py", [
            "--kp2d_flat_dir", f_kp2d, "--out_dir", f_poses2d,
        ], label=f"split_keypoints_per_camera.py (candidates f{k}, {tag})")

        poses2d_dirs.append(f_poses2d)

    return poses2d_dirs


def undistort_args(args, frames_dir, out_dir, out_pkl_dir, image_ext):
    """Common undistort_frames.py args. Single-warp mode (--target_pkl_dir)
    when a pre-made target pinhole calibration set was given; otherwise falls
    back to undistort_frames.py's default offline_undistort.py wrapper mode,
    which derives new (undistorted) intrinsics on the fly per camera -- no
    target set required."""
    a = ["--frames_dir", frames_dir, "--calib_dir", args.calib_dir,
         "--out_dir", out_dir, "--out_pkl_dir", out_pkl_dir, "--image_ext", image_ext]
    if args.target_pkl_dir:
        a += ["--target_pkl_dir", args.target_pkl_dir]
    return a


def keypoint_args(args, images_dir, out_kp2d_dir, fmasks_dir):
    a = ["--images_dir", images_dir, "--out_kp2d_dir", out_kp2d_dir, "--fmasks_dir", fmasks_dir,
         "--model", args.keypoint_model]
    if args.sapiens_checkpoint_root:
        a += ["--sapiens_checkpoint_root", args.sapiens_checkpoint_root]
    return a


def run_hloc(args, undistorted_dir, undistorted_pkl_dir, outputs_dir, image_ext, label):
    run_script("run_hloc.py", [
        "--undistorted_dir", undistorted_dir,
        "--undistorted_pkl_dir", undistorted_pkl_dir,
        "--outputs_dir", outputs_dir,
        "--multiframe_sfm_script", args.multiframe_sfm_script,
        "--num_timestamps", "1",
        "--image_ext", image_ext,
        "--feature_type", args.hloc_feature_type,
        "--resize_max", str(args.hloc_resize_max),
        "--max_keypoints", str(args.hloc_max_keypoints),
    ], conda_env=CONDA_ENV_HLOC, label=label)
    return outputs_dir / "transforms_multiframe.json"


# --------------------------------------------------------------------- stages
def stage_sync(args, L, image_ext):
    banner("STAGE: SYNC")

    if args.initial_sync_json:
        info(f"Using pre-existing sync offsets from {args.initial_sync_json} "
             "(skipping compute_sync_offsets.py)")
        shutil.copy(args.initial_sync_json, L["sync_offsets"])
        raw_sync_json = L["sync_offsets"]
    else:
        run_script("compute_sync_offsets.py",
                   [args.video_dir, args.out_dir], conda_env=args.generic_env, label="compute_sync_offsets.py (rough audio sync)")
        # compute_sync_offsets.py always names its output sync_offsets.json inside the given out dir
        raw_sync_json = L["sync_offsets"]

    extract_args = [str(args.video_dir), str(raw_sync_json), str(L["cand_raw"]), str(args.target_time_s)]
    run_script("extract_synced_frames.py", extract_args, label="extract_synced_frames.py (extract frame for sync QA grid)")
    run_script("make_sync_grid.py",
               [L["cand_raw"], L["sync_grid"]], conda_env=args.generic_env, label="make_sync_grid.py (sync QA grid)")
    info(f"Inspect {L['sync_grid']} -- every camera should show the same instant of action. "
         "This offset is NOT auto-corrected (automatic skeleton-based correction isn't part "
         "of this build yet) -- hand-tune any camera's frame_offset in sync_offsets.json that's "
         "visibly off, save the result, and pass it back in via --initial_sync_json next time.")

    return raw_sync_json


def stage_production(args, L, image_ext, sync_json: Path):
    banner("STAGE: PRODUCTION (direct single-warp undistortion)")

    extract_args = [str(args.video_dir), str(sync_json), str(L["production_raw"]),
                    str(args.target_time_s)]
    if args.pp3_dir:
        extract_args += ["--pp3_dir", str(args.pp3_dir)]
    run_script("extract_synced_frames.py", extract_args, label="extract_synced_frames.py (production frame)")

    run_script("undistort_frames.py",
               undistort_args(args, L["production_raw"], L["production_undist"], L["production_pkls"], image_ext),
               conda_env=args.generic_env, label="undistort_frames.py (production)")


def stage_poses(args, L, image_ext, sync_json: Path):
    banner("STAGE: POSES (background estimation & keypoint refine)")

    poses2d_dirs = [L["cand_corr_poses2d"] / f"f{k}" for k in range(args.sync_window)]
    if all(d.is_dir() and any(d.rglob("*.json")) for d in poses2d_dirs):
        info("Corrected-sync candidate window already complete on disk -- reusing it "
             "(delete sync_candidates_corrected*/ under --out_dir to force a redo).")
    else:
        poses2d_dirs = prepare_candidate_window(
            args, L, sync_json, args.sync_window, image_ext,
            L["cand_corr"], L["cand_corr_undist"], L["cand_corr_pkls"],
            L["cand_corr_fmasks"], L["cand_corr_kp2d"], L["cand_corr_poses2d"],
            tag="corrected sync",
        )

    final_transforms_path = L["hloc_final"] / "transforms_multiframe.json"
    if final_transforms_path.is_file():
        info("Final HLOC on the production frame already complete on disk -- reusing it "
             "(delete hloc_final/ under --out_dir to force a redo).")
        final_transforms = final_transforms_path
    else:
        final_transforms = run_hloc(
            args, L["production_undist"], L["production_pkls"],
            L["hloc_final"], image_ext, label="run_hloc.py (final HLOC on production frame)",
        )

    run_script("run_pose_refinement.py", [
        "--transforms", final_transforms,
        "--kp2d_dirs", ",".join(str(p) for p in poses2d_dirs),
        "--out_transforms", L["transforms_refined"],
        "--report_only",
    ], conda_env=args.generic_env, label="run_pose_refinement.py (pose refinement, report only -- pre-optimization residuals)")

    run_script("run_pose_refinement.py", [
        "--transforms", final_transforms,
        "--kp2d_dirs", ",".join(str(p) for p in poses2d_dirs),
        "--out_transforms", L["transforms_refined"],
    ], conda_env=args.generic_env, label="run_pose_refinement.py (pose refinement)")


def stage_masks(args, L, image_ext):
    banner("STAGE: MASKS (generation & cleaning)")

    # build_flat_dataset.py expects the Camera_XXXX/0000.ext layout, not the
    # flat one -- L["production_undist"] only has that structure because
    # stage_poses()'s run_hloc() call mutated it in place (run_hloc.py's
    # restructure_flat_to_percam side effect). This relies on stage_poses
    # having already run; if resuming with --start_from_stage masks, that
    # restructuring must already be done on disk.
    if L["flat_transforms"].is_file() and any(L["flat_images"].glob("*")):
        info("build_flat_dataset.py already complete on disk -- reusing flat images/transforms.")
    else:
        run_script("build_flat_dataset.py", [
            "--transforms", L["transforms_refined"],
            "--undistorted_dir", L["production_undist"],
            "--out_images_flat", L["flat_images"],
            "--out_transforms", L["flat_transforms"],
            "--image_ext", image_ext, "--out_image_ext", ".png",
        ], conda_env=args.generic_env, label="build_flat_dataset.py (flatten to 2-digit labels)")

    if L["flat_fmasks"].is_dir() and any(L["flat_fmasks"].glob("*.png")):
        info("generate_masks.py already complete on disk -- reusing flat masks.")
    else:
        run_script("generate_masks.py", [
            "--images_dir", L["flat_images"], "--out_fmasks_dir", L["flat_fmasks"], "--image_ext", ".png",
        ], conda_env=CONDA_ENV_DIFFUMAN4D, label="generate_masks.py (masks, flat production frame)")

    if L["flat_poses2d"].is_dir() and any(L["flat_poses2d"].rglob("*.json")):
        info("predict_keypoints_2d.py/split_keypoints_per_camera.py already complete on disk -- reusing flat keypoints.")
    else:
        run_script("predict_keypoints_2d.py", keypoint_args(args, L["flat_images"], L["flat_kp2d"], L["flat_fmasks"]),
                   conda_env=args.sapiens_env, label="predict_keypoints_2d.py (keypoints, flat production frame)")

        run_script("split_keypoints_per_camera.py", [
            "--kp2d_flat_dir", L["flat_kp2d"], "--out_dir", L["flat_poses2d"],
        ], label="split_keypoints_per_camera.py (split, flat production frame)")

    if L["flat_fmasks_clean"].is_dir() and any(L["flat_fmasks_clean"].glob("*.png")):
        info("clean_masks.py already complete on disk -- reusing cleaned masks.")
    else:
        run_script("clean_masks.py", [
            "--fmasks_dir", L["flat_fmasks"], "--kp2d_dir", L["flat_poses2d"],
            "--out_dir", L["flat_fmasks_clean"], "--images_dir", L["flat_images"], "--retry",
        ], conda_env=CONDA_ENV_DIFFUMAN4D, label="clean_masks.py (mask cleanup, skeleton-guided, retry fallback)")

    run_script("triangulate_and_project_keypoints.py", [
        "--camera_path", L["flat_transforms"], "--kp2d_dir", L["flat_poses2d"],
        "--out_kp3d_dir", L["poses_3d_fullres"], "--out_pcd_dir", L["poses_pcd_fullres"],
    ], conda_env=args.triangulate_env, label="triangulate_and_project_keypoints.py (triangulate subject point cloud, real cameras)")


def stage_branch_direct(args, L):
    banner("STAGE: BRANCH (direct 4K masked pipeline)")

    run_script("build_colmap_sparse.py", [
        "--transforms", L["flat_transforms"],
        "--points_ply", L["poses_pcd_fullres"] / "000000.ply",
        "--out_dir", L["train_set"],
        "--image_subdir", "images_flat",
        "--images_dir", L["flat_images"],
        "--masks_dir", L["flat_fmasks_clean"],
    ], conda_env=args.generic_env, label="build_colmap_sparse.py (COLMAP sparse + RGBA mask bake)")

    run_script("train_brush.py", [
        "--data", L["train_set"],
        "--total_steps", str(args.total_train_iters), "--max_resolution", "4096",
        "--export_every", str(args.export_every),
        "--brush_app", args.brush_app, "--export_path", L["brush_output"],
        "--export_name", f"{args.run_name}_4k_{{iter}}.ply",
        "--display", args.display,
        "--with_viewer" if args.with_viewer else "--no_viewer",
    ], label="train_brush.py (train Brush, 4K masked)")


# ------------------------------------------------------------------------ CLI
CONFIGURABLE_DEFAULTS = {
    "sapiens_env", "sapiens_checkpoint_root", "triangulate_env", "generic_env",
    "multiframe_sfm_script", "hloc_feature_type", "hloc_resize_max", "hloc_max_keypoints",
    "brush_app", "display",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON file of per-rig defaults (conda env names, --brush_app, --display, "
                             "SAPIENS_CHECKPOINT_ROOT, HLOC settings -- see configs/example_rig.json). "
                             "Explicit CLI flags always override the config. Per-run flags like --video_dir, "
                             "--calib_dir, --out_dir aren't configurable here -- those change every run.")
    parser.add_argument("--video_dir", required=True, type=Path, help="Root dir of raw multi-camera GoPro footage")
    parser.add_argument("--calib_dir", required=True, type=Path,
                        help="Native fisheye calibration PKLs, e.g. Dev/calibration/5k")
    parser.add_argument("--target_pkl_dir", type=Path, default=None,
                        help="Optional target pinhole calibration PKLs, e.g. Dev/calibration/4k_undistorted. "
                             "When given, undistort_frames.py runs single-warp mode (native fisheye "
                             "straight to this target geometry). When omitted, falls back to "
                             "undistort_frames.py's default offline_undistort.py wrapper mode, which "
                             "derives undistorted intrinsics on the fly per camera -- use this when no "
                             "pre-made target set exists yet.")
    parser.add_argument("--out_dir", required=True, type=Path, help="Master working directory for this run")
    parser.add_argument("--target_time", required=True,
                        help="Central target timestamp, e.g. '1.5', '1.5s', or '1500ms'")
    parser.add_argument("--initial_sync_json", type=Path, default=None,
                        help="Pre-existing sync_offsets.json (e.g. a hand-tuned sync_offsets_v5.json) "
                             "to seed the sync stage with, instead of recomputing audio cross-correlation "
                             "from scratch via compute_sync_offsets.py")
    parser.add_argument("--pp3_dir", type=Path, default=None,
                        help="Optional per-camera RawTherapee .pp3 profile directory (e.g. thumbs/)")

    parser.add_argument("--start_from_stage", choices=STAGE_KEYS, default=STAGE_KEYS[0],
                        help=f"Resume at one of {STAGE_KEYS} (assumes earlier stages' outputs already "
                             "exist under --out_dir)")
    parser.add_argument("--stop_after_stage", choices=STAGE_KEYS, default=None,
                        help="Stop once this stage completes, instead of running through 'branch'. "
                             "Useful to checkpoint and inspect (e.g. the sync grid) before committing "
                             "to the rest of the pipeline.")
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip validate_stage_output.py's on-disk sanity checks after each stage "
                             "(camera counts, non-degenerate masks/poses, non-empty COLMAP/splat output). "
                             "On by default so a stage that exits 0 but produced garbage doesn't silently "
                             "waste GPU time on the next (often more expensive) stage.")

    parser.add_argument("--sync_window", type=int, default=5,
                        help="Number of candidate frames for pose refinement, stage 'poses' (default 5)")
    parser.add_argument("--sapiens_env", default="sapiens2", help="Conda env for predict_keypoints_2d.py (sapiens2 or sapiens_lite)")
    parser.add_argument("--keypoint_model", choices=["goliath308", "coco_wholebody133"], default="goliath308",
                        help="308kp Sapiens2 Goliath (default, needs the Diffuman4D fork's Sapiens2 support) "
                             "or the legacy 133kp COCO-WholeBody model (needs mmdet, --sapiens_env sapiens_lite)")
    parser.add_argument("--sapiens_checkpoint_root", type=Path, default=None,
                        help="Overrides SAPIENS_CHECKPOINT_ROOT for predict_keypoints_2d.py. If omitted, "
                             "falls back to whatever SAPIENS_CHECKPOINT_ROOT is set to in this shell.")
    parser.add_argument("--triangulate_env", default="queen",
                        help="Conda env for triangulate_and_project_keypoints.py (needs easyvolcap/fire/open3d; "
                             "'queen' historically had these)")
    parser.add_argument("--generic_env", default="queen",
                        help="Conda env for scripts needing numpy/scipy/Pillow/plyfile/cv2 but no special "
                             "framework (make_sync_grid.py, undistort_frames.py, build_flat_dataset.py, "
                             "build_colmap_sparse.py, run_pose_refinement.py). The orchestrator's own "
                             "launching python may not have these installed, so these are NOT run bare "
                             "with sys.executable. 'queen' is confirmed to have the full set on this machine.")
    parser.add_argument("--multiframe_sfm_script", type=Path, default=DEFAULT_MULTIFRAME_SFM_SCRIPT,
                        help="Path to multiframe_sfm.py (override to test local changes to it)")
    parser.add_argument("--hloc_feature_type", default="superpoint", choices=["superpoint", "aliked"],
                        help="Feature detector for HLOC's SfM pose solve (passed through to "
                             "multiframe_sfm.py, matched with LightGlue either way).")
    parser.add_argument("--hloc_resize_max", type=int, default=4096,
                        help="Long-edge resize before HLOC feature extraction. multiframe_sfm.py's own "
                             "default is 2048, which is low relative to this rig's ~5312px native media "
                             "-- raised to 4096 by default here to preserve more detail for keypoint "
                             "localization / camera pose accuracy (higher GPU memory + runtime cost "
                             "during the HLOC stage; has headroom to coexist with a concurrent Brush "
                             "training job on a 24GB GPU, but not with two).")
    parser.add_argument("--hloc_max_keypoints", type=int, default=8192,
                        help="Max keypoints per image for HLOC feature extraction.")
    parser.add_argument("--brush_app", type=Path, default=Path.home() / "brush-app-x86_64-unknown-linux-gnu" / "brush_app")
    parser.add_argument("--total_train_iters", type=int, default=30000)
    parser.add_argument("--export_every", type=int, default=5000,
                        help="Brush checkpoint export interval in steps (also always exports once at completion)")
    parser.add_argument("--with_viewer", dest="with_viewer", action="store_true", default=True,
                        help="Open Brush's live viewer during training. On by default -- this is "
                             "the tested, working configuration; --no_viewer hasn't been "
                             "separately verified (here or elsewhere, e.g. WSL). See --no_viewer.")
    parser.add_argument("--no_viewer", dest="with_viewer", action="store_false",
                        help="Disable Brush's viewer. Use this if you've confirmed headless training "
                             "works in your environment.")
    parser.add_argument("--display", default=":2",
                        help="X DISPLAY brush_app connects to (default ':2').")
    parser.add_argument("--run_name", default=None, help="Label used in output filenames (default: out_dir's name)")
    return parser


def apply_config_defaults(parser):
    """Pre-scan argv for --config (without triggering the real parser's required-arg
    checks) and, if given, use its contents to override defaults for the flags in
    CONFIGURABLE_DEFAULTS. Explicit CLI flags still win -- set_defaults() only changes
    what's used when a flag isn't passed at all."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config is None:
        return

    if not pre_args.config.is_file():
        fail(f"--config {pre_args.config} not found")
        sys.exit(1)
    with open(pre_args.config) as f:
        config = json.load(f)

    unknown = set(config) - CONFIGURABLE_DEFAULTS
    if unknown:
        fail(f"--config {pre_args.config} has unrecognized key(s): {sorted(unknown)} "
             f"-- configurable keys are: {sorted(CONFIGURABLE_DEFAULTS)}")
        sys.exit(1)
    parser.set_defaults(**config)


def main():
    parser = build_parser()
    apply_config_defaults(parser)
    args = parser.parse_args()
    args.target_time_s = parse_target_time(args.target_time)
    args.run_name = args.run_name or args.out_dir.name
    image_ext = ".png" if args.pp3_dir else ".jpg"

    if not args.video_dir.is_dir():
        fail(f"--video_dir {args.video_dir} is not a directory")
        sys.exit(1)
    if not args.calib_dir.is_dir():
        fail(f"--calib_dir {args.calib_dir} is not a directory")
        sys.exit(1)
    if args.target_pkl_dir and not args.target_pkl_dir.is_dir():
        fail(f"--target_pkl_dir {args.target_pkl_dir} is not a directory")
        sys.exit(1)
    if args.initial_sync_json and not args.initial_sync_json.is_file():
        fail(f"--initial_sync_json {args.initial_sync_json} not found")
        sys.exit(1)
    if not args.multiframe_sfm_script.is_file():
        fail(f"--multiframe_sfm_script {args.multiframe_sfm_script} not found "
             "(pass --multiframe_sfm_script to point at a different copy)")
        sys.exit(1)
    if not args.sapiens_checkpoint_root and not os.environ.get("SAPIENS_CHECKPOINT_ROOT"):
        warn("SAPIENS_CHECKPOINT_ROOT is not set in this shell and --sapiens_checkpoint_root "
             "was not given -- predict_keypoints_2d.py (stages 'sync', 'poses', 'masks') "
             "will fail. Pass --sapiens_checkpoint_root, or export it before running, e.g. "
             "export SAPIENS_CHECKPOINT_ROOT=~/sapiens")

    try:
        videos = discover_cameras(args.video_dir)
    except StageError as e:
        fail(str(e))
        sys.exit(1)
    n_real = len(videos)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    L = build_layout(args.out_dir)

    banner(f"UNIFIED VOLUMETRIC PIPELINE -- {n_real} cameras, t={args.target_time_s:.3f}s")
    info(f"Working directory: {args.out_dir}")

    start_idx = STAGE_KEYS.index(args.start_from_stage)
    stop_idx = STAGE_KEYS.index(args.stop_after_stage) if args.stop_after_stage else None
    real_cameras_arg = ",".join(sorted(v.stem for v in videos))

    def should_run(key):
        return STAGE_KEYS.index(key) >= start_idx

    def hit_stop(key):
        if stop_idx is not None and STAGE_KEYS.index(key) == stop_idx:
            banner(f"STOPPED after stage {key!r} (--stop_after_stage)")
            return True
        return False

    def validate_stage(key):
        if args.no_validate:
            return
        run_script("validate_stage_output.py", [
            "--stage", key, "--out_dir", args.out_dir, "--real_cameras", real_cameras_arg,
        ], conda_env=args.generic_env, label=f"validate_stage_output.py ({key})")

    try:
        if should_run("sync"):
            sync_json = stage_sync(args, L, image_ext)
            with open(args.out_dir / "resolved_sync_json.txt", "w") as f:
                f.write(str(sync_json))
            validate_stage("sync")
        else:
            info("Skipping stage 'sync' (--start_from_stage)")
            resolved_path = args.out_dir / "resolved_sync_json.txt"
            check_expected(resolved_path, "sync")
            sync_json = Path(resolved_path.read_text().strip()) if resolved_path.exists() \
                else L["sync_offsets"]
        if hit_stop("sync"):
            return

        if should_run("production"):
            stage_production(args, L, image_ext, sync_json)
            validate_stage("production")
        else:
            info("Skipping stage 'production' (--start_from_stage)")
            check_expected(L["production_undist"], "production")
        if hit_stop("production"):
            return

        if should_run("poses"):
            stage_poses(args, L, image_ext, sync_json)
            validate_stage("poses")
        else:
            info("Skipping stage 'poses' (--start_from_stage)")
            check_expected(L["transforms_refined"], "poses")
        if hit_stop("poses"):
            return

        if should_run("masks"):
            stage_masks(args, L, image_ext)
            validate_stage("masks")
        else:
            info("Skipping stage 'masks' (--start_from_stage)")
            check_expected(L["flat_fmasks_clean"], "masks")
        if hit_stop("masks"):
            return

        if should_run("branch"):
            stage_branch_direct(args, L)
            validate_stage("branch")
        else:
            info("Skipping stage 'branch' (--start_from_stage)")

    except StageError as e:
        fail(str(e))
        fail(f"Pipeline stopped. Re-run with --start_from_stage <stage> to resume "
             f"(completed stages up to and including the one before the failure are reusable).")
        sys.exit(1)

    banner("PIPELINE COMPLETE")
    ok(f"Trained splat(s) exported to {L['brush_output']}")


if __name__ == "__main__":
    main()
