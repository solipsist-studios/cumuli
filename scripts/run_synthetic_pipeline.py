#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
run_synthetic_pipeline.py

Blender scene plus camera rig spec, in; trained `.sogst` splat and a scored
experiment record, out. The synthetic counterpart to
run_unified_pipeline.py, which starts from real GoPro footage.

It serves two jobs that share almost all of their machinery:

  1. Producing splats of any character and animation. Author in Daz or
     Character Creator, normalise with prepare_blender_scene.py, then run
     this. Nothing here is specific to one subject.
  2. Comparing camera configurations cheaply. Change the rig spec, run
     again, and read the scores side by side. Because the subject, the
     lighting, and the animation are identical between runs, a difference
     in the numbers is a difference in the cameras.

--------------------------------------------------------------------------
STAGES
--------------------------------------------------------------------------
  render     blender_render_rig.py through render_blender_rig.py: rig and
             eval cameras over the frame window, subject mattes from the
             render's own alpha, optional backdrop plates and composites.
  localize   only for --poses hloc/refined. Runs the REAL pose chain on the
             composited frames (run_hloc.py, Sapiens keypoints,
             run_pose_refinement.py) and scores the result against the
             Blender ground truth with score_poses_vs_gt.py. The estimated
             poses then replace the ground-truth ones for training, with
             the same pixels, so any quality difference is attributable to
             pose error alone.
  dataset4d  build_flipbook_4dgs_dataset.py over the rendered flipbook,
             with the eval ring merged in through --eval_root.
  train4d    the shared stage from run_unified_pipeline.py, unmodified:
             rotor 4DGS training, bake to .sogst, then eval_render.py.

--------------------------------------------------------------------------
SCORING
--------------------------------------------------------------------------
Every configuration is scored on the SAME fixed ring of eval cameras,
which never train. Holding out a rig camera instead, as a real capture
must, moves the test viewpoint whenever the rig changes and makes two runs
incomparable.

LPIPS leads the reported metrics. On a masked subject most of the frame is
empty background that every model renders perfectly, so PSNR is dominated
by it and compresses the differences that matter. PSNR is still recorded,
because it is directly comparable with the trainer's own eval numbers.

Usage:
    python3 scripts/run_synthetic_pipeline.py \\
        --blend ~/Dev/datasets/ariana_src/ariana_packed.blend \\
        --rig_spec configs/rigs/ring16.json \\
        --out_dir ~/runs/ariana_ring16 \\
        --frame_start 100 --frame_count 48 --samples 128
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402

import camera_rig_spec as rig_spec  # noqa: E402
from run_unified_pipeline import (  # noqa: E402
    CONDA_ENV, StageError, add_hloc_args, add_keypoint_args, add_train4d_args,
    banner, check_expected, fail, info, keypoint_args, ok, run_hloc,
    run_script, stage_train4d, warn,
)

STAGE_KEYS = ["render", "localize", "dataset4d", "train4d"]


# --------------------------------------------------------------------- layout
def build_layout(out_dir: Path):
    """Paths this pipeline reads and writes.

    Names shared with run_unified_pipeline.py's layout keep their meaning,
    because stage_train4d is reused verbatim and looks them up by name."""
    return {
        "render": out_dir / "render",
        "flipbook_src": out_dir / "flipbook_src",
        "flipbook_est": out_dir / "flipbook_est",
        "eval_src": out_dir / "eval_src",
        "composite": out_dir / "composite",
        "calib": out_dir / "calib",
        "calib_undistorted": out_dir / "calib_undistorted",
        "rig_gt": out_dir / "rig_gt_transforms.json",

        "localize": out_dir / "localize",
        "production_undist": out_dir / "localize" / "production_undistorted",
        "production_pkls": out_dir / "localize" / "production_pkls",
        "hloc_final": out_dir / "localize" / "hloc_final",
        "transforms_refined": out_dir / "localize" / "transforms_refined.json",
        "pose_scores": out_dir / "pose_scores.json",
        "transforms_aligned": out_dir / "localize" / "transforms_aligned.json",

        "dataset4d": out_dir / "dataset_4dgs",
        "train4d_config": out_dir / "gs4d_config.yaml",
        "train4d_model": out_dir / "train4d_output",
        "sogst_out": out_dir / "splat_4d.sogst",
        "eval4d_report": out_dir / "eval_4d.json",
        "experiment": out_dir / "experiment.json",
    }


# ---------------------------------------------------------------- conversions
def blender_bbox_to_dataset(bbox):
    """Scene-manifest bounding box (Blender, Z up) to the dataset world.

    The dataset world is the OpenGL/nerfstudio convention every stage after
    the pose solve uses, so (x, y, z) becomes (x, z, -y) and the y and z
    extents swap ends."""
    lo = np.asarray(bbox["min"], dtype=np.float64)
    hi = np.asarray(bbox["max"], dtype=np.float64)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    converted = np.stack([corners[:, 0], corners[:, 2], -corners[:, 1]], axis=1)
    return converted.min(axis=0), converted.max(axis=0)


# -------------------------------------------------------------------- stages
def stage_render(args, L):
    banner("STAGE: RENDER (Blender rig over the frame window)")
    render_args = [
        "--blend", args.blend, "--rig_spec", args.rig_spec,
        "--out_dir", args.out_dir,
        "--frame_start", str(args.frame_start),
        "--frame_count", str(args.frame_count),
        "--frame_step", str(args.frame_step),
        "--samples", str(args.samples),
        "--engine", args.engine, "--device", args.device,
        "--jobs", str(args.dataset_jobs),
    ]
    if args.manifest:
        render_args += ["--manifest", str(args.manifest)]
    if args.action:
        render_args += ["--action", args.action]
    if args.no_denoise:
        render_args.append("--no_denoise")
    if args.view_transform:
        render_args += ["--view_transform", args.view_transform]
    if args.skip_existing:
        render_args.append("--skip_existing")
    if args.poses != "gt" or args.background_plates:
        # The localization path matches features on the backdrop; a subject
        # alone on transparency has nothing static to match against.
        render_args.append("--background_plates")
    run_script("render_blender_rig.py", render_args, conda_env=CONDA_ENV,
               label="render_blender_rig.py (rig + eval cameras)")


def spread_indices(count, n):
    """`n` frame indices spread across the clip, ends included.

    Pose refinement against human keypoints needs MANY instants: with one,
    the optimiser absorbs timing and pose error together instead of
    correcting the cameras."""
    n = max(1, min(int(n), int(count)))
    if n == 1:
        return [count // 2]
    step = (count - 1) / (n - 1)
    return sorted({int(round(i * step)) for i in range(n)})


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    return dst


def prepare_localize_frame(args, L, index, out_images, out_pkls, image_ext=".png"):
    """One instant of composited frames, ready for HLOC or keypoints.

    A fisheye rig renders distorted frames, so this runs the same
    single-warp undistortion a real capture gets, straight to the target
    pinhole geometry written by the render driver. A pinhole rig needs no
    warp and the composites are used as they are."""
    src = L["composite"] / f"frame_{index:04d}"
    if not src.is_dir():
        raise StageError(
            f"{src} does not exist. The localization path needs composited "
            "frames, which the render stage writes only when the scene has a "
            "background collection. Check the scene manifest's "
            "background_objects count.")
    if L["calib_undistorted"].is_dir():
        run_script("undistort_frames.py", [
            "--frames_dir", src, "--calib_dir", L["calib"],
            "--out_dir", out_images, "--out_pkl_dir", out_pkls,
            "--target_pkl_dir", L["calib_undistorted"],
            "--model", "OPENCV_FISHEYE", "--image_ext", image_ext,
        ], conda_env=CONDA_ENV,
            label=f"undistort_frames.py (localize frame {index})")
        return out_images

    out_images.mkdir(parents=True, exist_ok=True)
    out_pkls.mkdir(parents=True, exist_ok=True)
    for img in sorted(src.glob(f"*{image_ext}")):
        link_or_copy(img, out_images / img.name)
    for pkl in sorted(L["calib"].glob("*.pkl")):
        link_or_copy(pkl, out_pkls / pkl.name)
    return out_images


def stage_localize(args, L, n_frames):
    """Run the real pose chain on the rendered frames and score it.

    This is the measurement the ground-truth path cannot make: how well
    HLOC and keypoint refinement recover a rig whose true poses are known
    exactly."""
    banner("STAGE: LOCALIZE (real pose chain on rendered frames)")
    mid = n_frames // 2
    prepare_localize_frame(args, L, mid, L["production_undist"],
                           L["production_pkls"])

    transforms = run_hloc(args, L["production_undist"], L["production_pkls"],
                          L["hloc_final"], ".png",
                          "run_hloc.py (synthetic pose solve)")

    poses2d_dirs = []
    for k, index in enumerate(spread_indices(n_frames, args.refine_instants)):
        work = L["localize"] / f"instant_{k:02d}"
        images = prepare_localize_frame(args, L, index, work / "undistorted",
                                        work / "pkls")
        run_script("generate_masks.py", [
            "--images_dir", images, "--out_fmasks_dir", work / "fmasks",
            "--image_ext", ".png",
        ], conda_env=CONDA_ENV, label=f"generate_masks.py (instant {k})")
        run_script("predict_keypoints_2d.py",
                   keypoint_args(args, images, work / "kp2d", work / "fmasks"),
                   conda_env=CONDA_ENV,
                   label=f"predict_keypoints_2d.py (instant {k})")
        run_script("split_keypoints_per_camera.py", [
            "--kp2d_flat_dir", work / "kp2d", "--out_dir", work / "poses_2d",
        ], label=f"split_keypoints_per_camera.py (instant {k})")
        poses2d_dirs.append(work / "poses_2d")

    run_script("run_pose_refinement.py", [
        "--transforms", transforms,
        "--kp2d_dirs", ",".join(str(d) for d in poses2d_dirs),
        "--out_transforms", L["transforms_refined"],
    ], conda_env=CONDA_ENV, label="run_pose_refinement.py (keypoint refine)")

    estimated = transforms if args.poses == "hloc" else L["transforms_refined"]
    run_script("score_poses_vs_gt.py", [
        "--estimated", transforms,
        "--refined", L["transforms_refined"],
        "--ground_truth", L["rig_gt"],
        "--report_json", L["pose_scores"],
        "--align_source", estimated,
        "--write_aligned", L["transforms_aligned"],
    ], conda_env=CONDA_ENV, label="score_poses_vs_gt.py (pose error vs Blender)")

    build_estimated_flipbook(L, n_frames)
    return L["flipbook_est"]


def build_estimated_flipbook(L, n_frames):
    """A flipbook whose images are the ground-truth ones and whose cameras
    are the estimated ones.

    Reusing the same pixels is deliberate. Training on estimated poses with
    differently-generated images would confound pose error with mask and
    exposure differences, and the question here is what the pose error
    alone costs."""
    aligned = json.loads(L["transforms_aligned"].read_text())
    for index in range(n_frames):
        src = L["flipbook_src"] / f"frame_{index:04d}"
        dst = L["flipbook_est"] / f"frame_{index:04d}"
        for sub in ("images_flat", "fmasks_clean"):
            for img in sorted((src / sub).glob("*.png")):
                link_or_copy(img, dst / sub / img.name)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "transforms.json").write_text(json.dumps(aligned, indent=1))
    info(f"wrote {n_frames} estimated-pose frame(s) to {L['flipbook_est']}")


def stage_dataset4d(args, L, flipbook_root, manifest):
    banner("STAGE: DATASET4D (rendered flipbook -> 4D training dataset)")
    n_cams = len(json.loads(
        (flipbook_root / "frame_0000" / "transforms.json").read_text())["frames"])
    hull_min_views = min(args.hull_min_views, n_cams)

    build_args = [
        "--flipbook_root", flipbook_root, "--out", L["dataset4d"],
        "--eval_root", L["eval_src"],
        "--fps", str(args.train_fps),
        "--downscale", str(args.dataset_downscale),
        "--jobs", str(args.dataset_jobs),
        "--hull_min_views", str(hull_min_views),
    ]
    if manifest and manifest.get("subject_bbox"):
        lo, hi = blender_bbox_to_dataset(manifest["subject_bbox"])
        bbox = ",".join(f"{v:.6f}" for v in list(lo) + list(hi))
        # Joined with "=" rather than passed as a separate argument: a
        # subject whose bounds start left of the origin makes the value
        # begin with "-", and argparse then reads it as another flag.
        build_args.append(f"--init_bbox={bbox}")
        info("passing the subject bounding box from the scene manifest, so "
             "the hull carve does not have to find it by sampling")
    if args.eval_camera:
        build_args += ["--test_cameras", args.eval_camera]
    run_script("build_flipbook_4dgs_dataset.py", build_args, conda_env=CONDA_ENV,
               label="build_flipbook_4dgs_dataset.py (4D dataset assembly)")


# ---------------------------------------------------------------- experiment
def write_experiment(args, L, started, spec, manifest):
    """One record per run: what was rendered, and how well it scored.

    compare_experiments.py reads a directory of these. LPIPS first, for the
    reason in the module docstring."""
    record = {
        "run": Path(args.out_dir).name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": round(time.time() - started, 1),
        "rig_spec_path": str(args.rig_spec),
        "rig_spec": spec,
        "blend": str(args.blend),
        "subject_height_m": manifest.get("subject_height") if manifest else None,
        "action": args.action,
        "frames": {"start": args.frame_start, "count": args.frame_count,
                   "step": args.frame_step, "fps": args.train_fps},
        "render": {"samples": args.samples, "engine": args.engine,
                   "denoise": not args.no_denoise},
        "poses": args.poses,
        "masks": args.masks,
        "train": {"iterations": args.total_train_iters,
                  "num_pts": args.num_pts,
                  "downscale": args.dataset_downscale},
    }
    stats_path = L["render"] / "render_stats.json"
    if stats_path.is_file():
        record["render"].update(json.loads(stats_path.read_text()))
    if L["pose_scores"].is_file():
        record["pose_scores"] = json.loads(L["pose_scores"].read_text())
    if L["eval4d_report"].is_file():
        report = json.loads(L["eval4d_report"].read_text())
        mean = report.get("mean", {})
        record["eval"] = {
            "lpips": mean.get("lpips"),
            "psnr_db": mean.get("psnr_db"),
            "ssim": mean.get("ssim"),
            "views": len(report.get("views", [])),
        }
    L["experiment"].write_text(json.dumps(record, indent=2))
    return record


# ----------------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blend", required=True, type=Path,
                   help="Normalised .blend from prepare_blender_scene.py")
    p.add_argument("--rig_spec", required=True, type=Path,
                   help="Camera rig spec (see configs/rigs/)")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--manifest", type=Path, default=None,
                   help="scene_manifest.json (defaults to one beside --blend)")
    p.add_argument("--action", default=None,
                   help="Armature action to render. Defaults to the scene's.")
    p.add_argument("--frame_start", type=int, default=1)
    p.add_argument("--frame_count", type=int, default=48,
                   help="Frames to render and train on. Also the trainer's "
                        "window: clip length is (count - 1) / --train_fps.")
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--samples", type=int, default=128,
                   help="Cycles samples per pixel (default 128)")
    p.add_argument("--engine", default="CYCLES")
    p.add_argument("--device", default="OPTIX")
    p.add_argument("--no_denoise", action="store_true")
    p.add_argument("--view_transform", default=None)
    p.add_argument("--background_plates", action="store_true",
                   help="Render backdrop plates even on the ground-truth "
                        "path (they are always rendered for --poses hloc "
                        "or refined, which need them)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Keep rendered images already on disk")

    p.add_argument("--poses", choices=["gt", "hloc", "refined"], default="gt",
                   help="Cameras used for training. 'gt' takes Blender's own "
                        "poses and skips the localize stage. 'hloc' and "
                        "'refined' run the real pose chain on the rendered "
                        "frames and train on its estimate, which measures "
                        "what pose error costs.")
    p.add_argument("--masks", choices=["gt", "birefnet"], default="gt",
                   help="'gt' uses the render's own alpha, which is exact. "
                        "'birefnet' runs the production mask chain instead, "
                        "to measure what mask error costs.")
    p.add_argument("--refine_instants", type=int, default=10,
                   help="Instants sampled across the clip for keypoint pose "
                        "refinement (default 10). One instant lets the "
                        "optimiser absorb error rather than correct it.")

    p.add_argument("--start_from_stage", choices=STAGE_KEYS, default=STAGE_KEYS[0])
    p.add_argument("--stop_after_stage", choices=STAGE_KEYS, default=None)
    p.add_argument("--no_validate", action="store_true")

    add_keypoint_args(p)
    add_hloc_args(p)
    add_train4d_args(p)
    return p


def resolve_manifest(args):
    if args.manifest is None:
        guess = args.blend.expanduser().resolve().parent / "scene_manifest.json"
        if guess.is_file():
            args.manifest = guess
    if args.manifest and Path(args.manifest).is_file():
        return json.loads(Path(args.manifest).read_text())
    return None


def main():
    parser = build_parser()
    args = parser.parse_args()
    started = time.time()

    if not args.blend.is_file():
        fail(f"--blend {args.blend} not found")
        sys.exit(1)
    if not args.rig_spec.is_file():
        fail(f"--rig_spec {args.rig_spec} not found")
        sys.exit(1)

    manifest = resolve_manifest(args)
    try:
        # Loading validates the spec, so both it and the resolve below
        # report a bad spec as one clear line rather than a traceback.
        spec = rig_spec.load_spec(args.rig_spec)
        resolved = rig_spec.resolve_rig(
            spec, manifest,
            calib_loader=rig_spec.pickle_calib_loader(args.rig_spec))
        n_train, n_eval = len(resolved["train"]), len(resolved["eval"])
    except (rig_spec.RigSpecError, ValueError) as e:
        fail(f"rig spec {args.rig_spec}: {e}")
        sys.exit(1)

    # The trainer stage reads the window and rate off these names.
    args.train_window = args.frame_count
    if manifest and manifest.get("fps"):
        parser_default = parser.get_default("train_fps")
        if args.train_fps == parser_default:
            args.train_fps = float(manifest["fps"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    L = build_layout(args.out_dir)

    banner(f"SYNTHETIC PIPELINE -- {spec.get('name', args.rig_spec.stem)}: "
           f"{n_train} rig cameras, {n_eval} eval cameras, "
           f"{args.frame_count} frames @ {args.train_fps} fps")
    info(f"Working directory: {args.out_dir}")
    if args.masks == "birefnet":
        warn("--masks birefnet is not implemented yet; the render's own alpha "
             "is used. Mask ablations are tracked separately.")

    start_idx = STAGE_KEYS.index(args.start_from_stage)
    stop_idx = STAGE_KEYS.index(args.stop_after_stage) if args.stop_after_stage else None
    if stop_idx is not None and stop_idx < start_idx:
        fail(f"--stop_after_stage {args.stop_after_stage!r} comes before "
             f"--start_from_stage {args.start_from_stage!r}")
        sys.exit(1)

    def should_run(key):
        return STAGE_KEYS.index(key) >= start_idx

    def hit_stop(key):
        if stop_idx is not None and STAGE_KEYS.index(key) == stop_idx:
            banner(f"STOPPED after stage {key!r} (--stop_after_stage)")
            return True
        return False

    def validate(key):
        if args.no_validate or key not in ("dataset4d", "train4d"):
            return
        labels = ",".join(c.label for c in resolved["train"])
        run_script("validate_stage_output.py", [
            "--stage", key, "--out_dir", args.out_dir, "--real_cameras", labels,
        ], conda_env=CONDA_ENV, label=f"validate_stage_output.py ({key})")

    flipbook_root = L["flipbook_src"]
    try:
        if should_run("render"):
            stage_render(args, L)
        else:
            info("Skipping stage 'render' (--start_from_stage)")
            check_expected(L["flipbook_src"], "render")
        if hit_stop("render"):
            return

        if args.poses == "gt":
            info("--poses gt: skipping stage 'localize', training on "
                 "Blender's own camera poses")
        elif should_run("localize"):
            flipbook_root = stage_localize(args, L, args.frame_count)
        else:
            info("Skipping stage 'localize' (--start_from_stage)")
            check_expected(L["flipbook_est"], "localize")
            flipbook_root = L["flipbook_est"]
        if hit_stop("localize"):
            return

        if should_run("dataset4d"):
            stage_dataset4d(args, L, flipbook_root, manifest)
            validate("dataset4d")
        else:
            info("Skipping stage 'dataset4d' (--start_from_stage)")
            check_expected(L["dataset4d"], "dataset4d")
        if hit_stop("dataset4d"):
            write_experiment(args, L, started, spec, manifest)
            return

        if should_run("train4d"):
            stage_train4d(args, L)
            validate("train4d")
    except StageError as e:
        fail(str(e))
        fail("Pipeline stopped. Re-run with --start_from_stage <stage> to "
             "resume from the last completed stage.")
        write_experiment(args, L, started, spec, manifest)
        sys.exit(1)

    record = write_experiment(args, L, started, spec, manifest)
    banner("SYNTHETIC PIPELINE COMPLETE")
    ok(f"Trained 4D splat baked to {L['sogst_out']}")
    scores = record.get("eval")
    if scores and scores.get("lpips") is not None:
        ok(f"Eval over {scores['views']} novel views: "
           f"LPIPS {scores['lpips']:.4f}, PSNR {scores['psnr_db']:.2f} dB, "
           f"SSIM {scores['ssim']:.4f}")
    ok(f"Experiment record: {L['experiment']}")


if __name__ == "__main__":
    main()
