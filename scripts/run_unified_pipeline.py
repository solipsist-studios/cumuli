#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
run_unified_pipeline.py

Unified entry point for the volumetric capture pipeline: sync -> undistort
-> pose solve/refine -> mask -> 4D dataset build -> rotor 4DGS training,
baked to a streamable .sogst asset.
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
  5. DATASET4D: extract a --train_window frame sequence at the verified
     sync -> per-frame undistort/masks/keypoints/clean (the same chain
     the production instant gets) -> assemble_flipbook_frame.py (frame-
     major layout; the rig is static, so the production instant's flat
     transforms apply to every frame) -> build_flipbook_4dgs_dataset.py
     (D-NeRF dataset with per-view intrinsics, mate-aware eval holdouts).
     CPU-capable end to end.
  6. TRAIN4D (CUDA only): generate the trainer config from
     configs/gs4d_pretrain_template.yaml -> train_scratch.py in the
     vendored deps/OMG4 fork -> bake_sogst.py (lifetime mask-consistency
     filter on) -> eval_render.py --report_json on the held-out camera.
     The Diffuman4D 48-camera dense-ring branch is still not wired in.

Usage (see --help for every flag):
    python3 run_unified_pipeline.py \\
        --video_dir <capture>/movies \\
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

STAGE_KEYS = ["sync", "production", "poses", "masks", "dataset4d", "train4d"]

# Every stage runs in the one pipeline env (envs/cumuli.yml,
# scripts/setup_cumuli_env.sh). The historical per-stage env parameters
# were removed once the merged env was verified by full GPU and CPU
# pipeline runs; a machine that genuinely needs a different env name can
# change this constant.
CONDA_ENV = "cumuli"

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


_ENV_PYTHON = None


def resolve_env_python():
    """The CONDA_ENV interpreter, resolved once per process.

    Every stage runs in the same env, so paying `conda run`'s activation
    overhead (its own Python boot plus env resolution, seconds per call)
    on every one of the potentially hundreds of stage subprocesses bought
    nothing. Scripts are invoked with the env's python directly instead,
    which is how the pipeline's historical manual runs always worked."""
    global _ENV_PYTHON
    if _ENV_PYTHON is not None:
        return _ENV_PYTHON
    result = subprocess.run([resolve_conda(), "info", "--base"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise StageError(f"conda info --base failed: {result.stderr.strip()}")
    candidate = Path(result.stdout.strip()) / "envs" / CONDA_ENV / "bin" / "python3"
    if not candidate.is_file():
        raise StageError(
            f"conda env {CONDA_ENV!r} not found at {candidate}. Provision it "
            f"with scripts/setup_cumuli_env.sh.")
    _ENV_PYTHON = str(candidate)
    return _ENV_PYTHON


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
    process environment is never mutated, so nothing leaks between stages
    or back into this process.

    conda_env selects the interpreter: truthy runs the script with the
    CONDA_ENV env's python (resolved once, no per-call `conda run`
    activation), None runs it with the orchestrator's own interpreter
    (stdlib-only helpers)."""
    script_path = Path(script_name)
    if not script_path.is_absolute():
        script_path = SCRIPTS_DIR / script_name
    if not script_path.is_file():
        raise StageError(f"{script_name} not found at {script_path}")

    str_args = [str(a) for a in args]
    if conda_env:
        cmd = [resolve_env_python(), str(script_path)] + str_args
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


def gpu_available() -> bool:
    """CUDA presence probe, stdlib-only so the orchestrator itself never
    imports torch. nvidia-smi exists and lists at least one device."""
    import shutil as _shutil
    exe = _shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        result = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


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

        # 4D training path (stage_dataset4d / stage_train4d)
        "seq_raw": out_dir / "train_seq_raw",
        "seq_undist": out_dir / "train_seq_undistorted",
        "seq_pkls": out_dir / "train_seq_pkls",
        "flipbook_src": out_dir / "flipbook_src",
        "dataset4d": out_dir / "dataset_4dgs",
        "train4d_config": out_dir / "gs4d_config.yaml",
        "train4d_model": out_dir / "train4d_output",
        "sogst_out": out_dir / "splat_4d.sogst",
        "eval4d_report": out_dir / "eval_4d.json",
    }
    return L


# ------------------------------------------------------- candidate window prep
def _instant_dirs(base_dirs: tuple[Path, ...], window: int, k: int) -> tuple[Path, ...]:
    """The k-th candidate instant's set of per-purpose dirs, matching
    whichever output layout extract_synced_frames.py actually produces for
    this window size: window==1 writes flat (directly into each base dir,
    no subdir at all -- see that script's own --window==1 special case),
    any window>1 writes per-instant f0/../f{N-1}/ subdirs. Real bug found
    running this locally (2026-07-30): --sync_window 1 through this
    orchestrator always assumed the subdir layout unconditionally, so it
    looked for an f0/ dir extract_synced_frames.py never created, and
    crashed immediately -- window==1 is genuinely a different code path in
    that script, not just "fewer frames"."""
    if window == 1:
        return base_dirs
    return tuple(d / f"f{k}" for d in base_dirs)


def prepare_candidate_window(args, L, sync_json: Path, window: int, image_ext: str,
                              raw_dir, undist_dir, pkl_dir, fmasks_dir, kp2d_dir, poses2d_dir,
                              tag: str, start_time_s: float | None = None):
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
        f_raw, f_undist, f_pkl, f_fmask, f_kp2d, f_poses2d = _instant_dirs(
            (raw_dir, undist_dir, pkl_dir, fmasks_dir, kp2d_dir, poses2d_dir), window, k
        )

        if f_poses2d.is_dir() and any(f_poses2d.rglob("*.json")):
            info(f"f{k} ({tag}) already fully processed on disk -- reusing "
                 f"(delete {f_poses2d} to force a redo)")
            poses2d_dirs.append(f_poses2d)
            continue

        run_script("undistort_frames.py", undistort_args(args, f_raw, f_undist, f_pkl, image_ext),
                   conda_env=CONDA_ENV, label=f"undistort_frames.py (candidates f{k}, {tag})")

        run_script("generate_masks.py", [
            "--images_dir", f_undist, "--out_fmasks_dir", f_fmask, "--image_ext", image_ext,
        ], conda_env=CONDA_ENV, label=f"generate_masks.py (candidates f{k}, {tag})")

        run_script("predict_keypoints_2d.py", keypoint_args(args, f_undist, f_kp2d, f_fmask),
                   conda_env=CONDA_ENV, label=f"predict_keypoints_2d.py (candidates f{k}, {tag})")

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
    a = ["--images_dir", images_dir, "--out_kp2d_dir", out_kp2d_dir, "--fmasks_dir", fmasks_dir]
    if args.sapiens_checkpoint_root:
        a += ["--sapiens_checkpoint_root", args.sapiens_checkpoint_root]
    if args.sapiens_model_size != "1b":
        a += ["--sapiens_model_size", args.sapiens_model_size]
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
    ], conda_env=CONDA_ENV, label=label)
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
                   [args.video_dir, args.out_dir], conda_env=CONDA_ENV, label="compute_sync_offsets.py (rough audio sync)")
        # compute_sync_offsets.py always names its output sync_offsets.json inside the given out dir
        raw_sync_json = L["sync_offsets"]

    extract_args = [str(args.video_dir), str(raw_sync_json), str(L["cand_raw"]), str(args.target_time_s)]
    run_script("extract_synced_frames.py", extract_args, label="extract_synced_frames.py (extract frame for sync QA grid)")
    run_script("make_sync_grid.py",
               [L["cand_raw"], L["sync_grid"]], conda_env=CONDA_ENV, label="make_sync_grid.py (sync QA grid)")
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
               conda_env=CONDA_ENV, label="undistort_frames.py (production)")


def stage_poses(args, L, image_ext, sync_json: Path):
    banner("STAGE: POSES (background estimation & keypoint refine)")

    poses2d_dirs = [_instant_dirs((L["cand_corr_poses2d"],), args.sync_window, k)[0]
                     for k in range(args.sync_window)]
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
    ], conda_env=CONDA_ENV, label="run_pose_refinement.py (pose refinement, report only -- pre-optimization residuals)")

    run_script("run_pose_refinement.py", [
        "--transforms", final_transforms,
        "--kp2d_dirs", ",".join(str(p) for p in poses2d_dirs),
        "--out_transforms", L["transforms_refined"],
    ], conda_env=CONDA_ENV, label="run_pose_refinement.py (pose refinement)")


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
        ], conda_env=CONDA_ENV, label="build_flat_dataset.py (flatten to 2-digit labels)")

    if L["flat_fmasks"].is_dir() and any(L["flat_fmasks"].glob("*.png")):
        info("generate_masks.py already complete on disk -- reusing flat masks.")
    else:
        run_script("generate_masks.py", [
            "--images_dir", L["flat_images"], "--out_fmasks_dir", L["flat_fmasks"], "--image_ext", ".png",
        ], conda_env=CONDA_ENV, label="generate_masks.py (masks, flat production frame)")

    if L["flat_poses2d"].is_dir() and any(L["flat_poses2d"].rglob("*.json")):
        info("predict_keypoints_2d.py/split_keypoints_per_camera.py already complete on disk -- reusing flat keypoints.")
    else:
        run_script("predict_keypoints_2d.py", keypoint_args(args, L["flat_images"], L["flat_kp2d"], L["flat_fmasks"]),
                   conda_env=CONDA_ENV, label="predict_keypoints_2d.py (keypoints, flat production frame)")

        run_script("split_keypoints_per_camera.py", [
            "--kp2d_flat_dir", L["flat_kp2d"], "--out_dir", L["flat_poses2d"],
        ], label="split_keypoints_per_camera.py (split, flat production frame)")

    if L["flat_fmasks_clean"].is_dir() and any(L["flat_fmasks_clean"].glob("*.png")):
        info("clean_masks.py already complete on disk -- reusing cleaned masks.")
    else:
        run_script("clean_masks.py", [
            "--fmasks_dir", L["flat_fmasks"], "--kp2d_dir", L["flat_poses2d"],
            "--out_dir", L["flat_fmasks_clean"], "--images_dir", L["flat_images"], "--retry",
        ], conda_env=CONDA_ENV, label="clean_masks.py (mask cleanup, skeleton-guided, retry fallback)")

    run_script("triangulate_and_project_keypoints.py", [
        "--camera_path", L["flat_transforms"], "--kp2d_dir", L["flat_poses2d"],
        "--out_kp3d_dir", L["poses_3d_fullres"], "--out_pcd_dir", L["poses_pcd_fullres"],
    ], conda_env=CONDA_ENV, label="triangulate_and_project_keypoints.py (triangulate subject point cloud, real cameras)")


def stage_dataset4d(args, L, image_ext, sync_json: Path):
    """Build the 4D (D-NeRF-style) training dataset for the rotor trainer.

    Extracts a --train_window frame sequence, runs the standard per-instant
    chain (undistort, masks, keypoints, clean) on each frame, lays each
    instant out as a flipbook frame dir, and hands the whole sequence to
    build_flipbook_4dgs_dataset.py. Every step is CPU-capable, so CI can
    stop after this stage. Depends on the poses and masks stages: the
    refined transforms, the label map, and the flat transforms must exist.
    Frame extraction is PNG end-to-end -- the flipbook layout and the
    trainer's D-NeRF loader both expect .png."""
    banner("STAGE: DATASET4D (frame sequence -> 4D training dataset)")

    window = args.train_window
    extract_args = [str(args.video_dir), str(sync_json), str(L["seq_raw"]),
                    str(args.target_time_s), "--window", str(window),
                    "--output_ext", ".png"]
    if args.pp3_dir:
        extract_args += ["--pp3_dir", str(args.pp3_dir)]
    if L["seq_raw"].is_dir() and any(L["seq_raw"].rglob("*.png")):
        info("train sequence already extracted on disk -- reusing "
             f"(delete {L['seq_raw']} to force a redo)")
    else:
        run_script("extract_synced_frames.py", extract_args,
                   label=f"extract_synced_frames.py (train window of {window})")

    for k in range(window):
        f_raw, f_undist, f_pkl = _instant_dirs(
            (L["seq_raw"], L["seq_undist"], L["seq_pkls"]), window, k)
        frame_dir = L["flipbook_src"] / f"frame_{k:04d}"

        if (frame_dir / "fmasks_clean").is_dir() and any((frame_dir / "fmasks_clean").glob("*.png")):
            info(f"frame_{k:04d} already fully processed on disk -- reusing "
                 f"(delete {frame_dir} to force a redo)")
            continue

        run_script("undistort_frames.py",
                   undistort_args(args, f_raw, f_undist, f_pkl, ".png"),
                   conda_env=CONDA_ENV,
                   label=f"undistort_frames.py (train frame {k})")

        run_script("assemble_flipbook_frame.py", [
            "--undist_dir", f_undist, "--label_map", L["flat_label_map"],
            "--flat_transforms", L["flat_transforms"],
            "--out_frame_dir", frame_dir, "--image_ext", ".png",
        ], label=f"assemble_flipbook_frame.py (train frame {k})")

        frame_images = frame_dir / "images_flat"
        run_script("generate_masks.py", [
            "--images_dir", frame_images, "--out_fmasks_dir", frame_dir / "fmasks",
            "--image_ext", ".png",
        ], conda_env=CONDA_ENV, label=f"generate_masks.py (train frame {k})")

        run_script("predict_keypoints_2d.py",
                   keypoint_args(args, frame_images, frame_dir / "kp2d", frame_dir / "fmasks"),
                   conda_env=CONDA_ENV, label=f"predict_keypoints_2d.py (train frame {k})")

        run_script("split_keypoints_per_camera.py", [
            "--kp2d_flat_dir", frame_dir / "kp2d", "--out_dir", frame_dir / "poses_2d",
        ], label=f"split_keypoints_per_camera.py (train frame {k})")

        run_script("clean_masks.py", [
            "--fmasks_dir", frame_dir / "fmasks", "--kp2d_dir", frame_dir / "poses_2d",
            "--out_dir", frame_dir / "fmasks_clean", "--images_dir", frame_images, "--retry",
        ], conda_env=CONDA_ENV, label=f"clean_masks.py (train frame {k})")

    # The hull carve requires each candidate point inside the mask in at
    # least --hull_min_views views. The builder's default of 9 suits a
    # full rig; on a smaller rig (or the CI subset) it exceeds the camera
    # count and the hull is empty by construction. Clamp to the rig size.
    try:
        n_cams = len(json.loads(L["flat_transforms"].read_text()).get("frames", []))
    except (OSError, ValueError):
        n_cams = 0
    hull_min_views = min(args.hull_min_views, n_cams) if n_cams else args.hull_min_views
    build_args = [
        "--flipbook_root", L["flipbook_src"], "--out", L["dataset4d"],
        "--fps", str(args.train_fps), "--downscale", str(args.dataset_downscale),
        "--jobs", str(args.dataset_jobs),
        "--hull_min_views", str(hull_min_views),
    ]
    # The eval camera's stereo mates must leave training entirely: a held-out
    # camera whose near-duplicate mate keeps training measures leakage, not
    # novel-view quality (~5.7 dB of pure leak measured on an 11-camera rig).
    if args.eval_camera:
        build_args += ["--test_cameras", args.eval_camera]
        holdouts = [c for c in (args.holdout_cameras or []) if c != args.eval_camera]
        if holdouts:
            build_args += ["--holdout_cameras", *holdouts]
    run_script("build_flipbook_4dgs_dataset.py", build_args,
               conda_env=CONDA_ENV,
               label="build_flipbook_4dgs_dataset.py (4D dataset assembly)")


def stage_train4d(args, L):
    """Train the rotor 4DGS model, bake it to .sogst, and score it.

    GPU-gated: the trainer's rasterizer is CUDA-only, so this stage refuses
    to start without a visible GPU rather than fail deep inside training.
    The trainer is train_scratch.py in the vendored OMG4 fork (upstream
    4d-gaussian-splatting train.py plus this project's patches), run in the
    trainer conda env. Explicit --save_iterations makes the checkpoint name
    deterministic: <model_path>/chkpnt<iters>.pth."""
    banner("STAGE: TRAIN4D (rotor 4DGS training, bake, eval)")

    if not gpu_available():
        raise StageError(
            "stage 'train4d' requires a CUDA GPU (nvidia-smi found no device). "
            "Run through --stop_after_stage dataset4d on this machine and "
            "train on a GPU host.")

    trainer_repo = Path(args.trainer_repo).expanduser().resolve()
    train_script = trainer_repo / "train_scratch.py"
    if not train_script.is_file():
        raise StageError(
            f"trainer entry point not found at {train_script}. Initialise the "
            f"submodule (git submodule update --init deps/OMG4) or point "
            f"--trainer_repo at a patched OMG4 clone.")

    iters = args.total_train_iters
    duration_s = (args.train_window - 1) / args.train_fps
    if args.trainer_config:
        config_path = Path(args.trainer_config)
        info(f"Using caller-supplied trainer config {config_path} "
             "(template generation skipped)")
    else:
        template = (REPO_ROOT / "configs" / "gs4d_pretrain_template.yaml").read_text()
        config_path = L["train4d_config"]
        config_path.write_text(template.format_map({
            "time_max": f"{duration_s:.6f}",
            "num_pts": args.num_pts,
            "batch_size": args.batch_size4d,
            "source_path": str(L["dataset4d"].resolve()),
            "model_path": str(L["train4d_model"].resolve()),
            "iterations": iters,
            "densify_until_iter": args.densify_until_iter,
            "densify_until_num_points": args.densify_until_num_points,
        }))
        info(f"Wrote trainer config {config_path}")

    extra_env = {}
    if args.t_init_div:
        # Initial temporal sigma divisor: sqrt(duration / div). Unset (0)
        # preserves the trainer's upstream default of 5, which bakes several
        # frames of motion smear into the starting sigma.
        extra_env["GS4D_T_INIT_DIV"] = str(args.t_init_div)

    checkpoint = L["train4d_model"] / f"chkpnt{iters}.pth"
    if checkpoint.is_file():
        info(f"checkpoint {checkpoint} already on disk -- skipping training "
             "(delete it to force a retrain)")
    else:
        run_script(train_script, [
            "--config", config_path,
            "--test_iterations", str(iters),
            "--save_iterations", str(iters),
        ], conda_env=CONDA_ENV, cwd=trainer_repo, extra_env=extra_env,
            label="train_scratch.py (rotor 4DGS pretrain)")
    if not checkpoint.is_file():
        raise StageError(f"training finished but {checkpoint} does not exist")

    bake_args = [
        "--input", checkpoint, "--output", L["sogst_out"],
        "--time_min", "0", "--time_max", f"{duration_s:.6f}",
        "--fps", str(args.train_fps),
        # Lifetime mask-consistency filter: drops splats that project outside
        # the subject mask in most views across their active life. Removes
        # real silhouette-escaping junk; it is not a quality regulariser.
        "--mask_filter_root", L["dataset4d"],
    ]
    run_script("bake_sogst.py", bake_args, conda_env=CONDA_ENV,
               label="bake_sogst.py (bake checkpoint to .sogst)")

    if args.skip_eval or not args.eval_camera:
        info("eval skipped (no --eval_camera or --skip_eval given)")
        return
    run_script("eval_render.py", [
        "--model", L["sogst_out"],
        "--transforms", L["dataset4d"] / "transforms_test.json",
        "--gt-dir", L["dataset4d"] / "eval_gt_flat",
        "--every", str(args.eval_every),
        "--report_json", L["eval4d_report"],
    ], conda_env=CONDA_ENV, label="eval_render.py (held-out scoring)")


# ------------------------------------------------------------------------ CLI
CONFIGURABLE_DEFAULTS = {
    "sapiens_checkpoint_root",
    "multiframe_sfm_script", "hloc_feature_type", "hloc_resize_max", "hloc_max_keypoints",
    "trainer_repo",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON file of per-rig defaults (conda env names, --trainer_repo, "
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
    parser.add_argument("--sapiens_checkpoint_root", type=Path, default=None,
                        help="Overrides SAPIENS_CHECKPOINT_ROOT for predict_keypoints_2d.py. If omitted, "
                             "falls back to whatever SAPIENS_CHECKPOINT_ROOT is set to in this shell.")
    parser.add_argument("--sapiens_model_size", default="1b", choices=["0.4b", "0.8b", "1b", "5b"],
                        help="Sapiens2 pose checkpoint size (default 1b, production quality baseline). "
                             "Smaller sizes use dramatically less RAM at some accuracy cost -- see "
                             "predict_keypoints_2d.py's own --sapiens_model_size help.")
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
                             "during the HLOC stage; has headroom to coexist with a concurrent training "
                             "training job on a 24GB GPU, but not with two).")
    parser.add_argument("--hloc_max_keypoints", type=int, default=8192,
                        help="Max keypoints per image for HLOC feature extraction.")
    parser.add_argument("--total_train_iters", type=int, default=30000,
                        help="4D trainer iterations (also the checkpoint name: chkpnt<iters>.pth).")

    # ---- 4D training path (stage_dataset4d / stage_train4d) ----
    parser.add_argument("--train_window", type=int, default=48,
                        help="Frame count of the 4D training sequence, extracted at --target_time. "
                             "Clip duration is (window - 1) / --train_fps seconds.")
    parser.add_argument("--train_fps", type=float, default=30.0,
                        help="Frame rate the training sequence is extracted and trained at. "
                             "Must match the capture's real frame stepping.")
    parser.add_argument("--dataset_downscale", type=int, default=1,
                        help="build_flipbook_4dgs_dataset.py --downscale for the 4D dataset.")
    parser.add_argument("--dataset_jobs", type=int, default=8)
    parser.add_argument("--hull_min_views", type=int, default=9,
                        help="Minimum mask-consistent views for a visual-hull init point. "
                             "Clamped to the rig's camera count at run time.")
    parser.add_argument("--eval_camera", default=None,
                        help="2-digit camera label held out for eval_render.py scoring. "
                             "Omit to train on every camera and skip eval.")
    parser.add_argument("--holdout_cameras", nargs="*", default=[],
                        help="Additional 2-digit labels excluded from training WITHOUT being "
                             "scored -- the eval camera's stereo mates. A held-out camera whose "
                             "near-duplicate mate keeps training measures leakage, not quality.")
    parser.add_argument("--num_pts", type=int, default=100000,
                        help="Initial gaussian count for the 4D trainer (production default).")
    parser.add_argument("--batch_size4d", type=int, default=2)
    parser.add_argument("--densify_until_iter", type=int, default=25000)
    parser.add_argument("--densify_until_num_points", type=int, default=3000000)
    parser.add_argument("--t_init_div", type=int, default=100,
                        help="GS4D_T_INIT_DIV for the trainer: initial temporal sigma is "
                             "sqrt(duration / div). 0 leaves the env var unset (upstream div = 5, "
                             "which bakes several frames of motion smear into the initial sigma).")
    parser.add_argument("--trainer_config", type=Path, default=None,
                        help="Pre-written trainer yaml. Bypasses template generation entirely; "
                             "the template's source_path/model_path substitutions become the "
                             "caller's responsibility.")
    parser.add_argument("--trainer_repo", type=Path, default=REPO_ROOT / "deps" / "OMG4",
                        help="Patched OMG4 clone carrying train_scratch.py (default: the vendored "
                             "deps/OMG4 submodule).")
    parser.add_argument("--eval_every", type=int, default=10,
                        help="eval_render.py --every: score every Nth held-out frame.")
    parser.add_argument("--skip_eval", action="store_true")
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
    try:
        with open(pre_args.config) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        fail(f"--config {pre_args.config} is not valid JSON: {e}")
        sys.exit(1)

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
    if stop_idx is not None and stop_idx < start_idx:
        fail(f"--stop_after_stage {args.stop_after_stage!r} comes before "
             f"--start_from_stage {args.start_from_stage!r} in the pipeline order "
             f"{STAGE_KEYS} -- this would stop before any stage actually ran.")
        sys.exit(1)
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
        ], conda_env=CONDA_ENV, label=f"validate_stage_output.py ({key})")

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

        if should_run("dataset4d"):
            stage_dataset4d(args, L, image_ext, sync_json)
            validate_stage("dataset4d")
        else:
            info("Skipping stage 'dataset4d' (--start_from_stage)")
            check_expected(L["dataset4d"], "dataset4d")
        if hit_stop("dataset4d"):
            return

        if should_run("train4d"):
            stage_train4d(args, L)
            validate_stage("train4d")
        else:
            info("Skipping stage 'train4d' (--start_from_stage)")

    except StageError as e:
        fail(str(e))
        fail("Pipeline stopped. Re-run with --start_from_stage <stage> to resume "
             "(completed stages up to and including the one before the failure are reusable).")
        sys.exit(1)

    banner("PIPELINE COMPLETE")
    ok(f"Trained 4D splat baked to {L['sogst_out']}")


if __name__ == "__main__":
    main()
