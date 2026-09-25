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
  render     render_rig_in_blender.py through render_blender_rig.py: rig and
             eval cameras over the frame window, subject mattes from the
             render's own alpha, optional backdrop plates and composites.
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
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402

import camera_rig_spec as rig_spec  # noqa: E402
from run_unified_pipeline import (  # noqa: E402
    CONDA_ENV, StageError, add_train4d_args, banner, check_expected, fail,
    info, ok, run_script, stage_train4d,
)

STAGE_KEYS = ["render", "dataset4d", "train4d"]


# --------------------------------------------------------------------- layout
def build_layout(out_dir: Path):
    """Paths this pipeline reads and writes.

    Names shared with run_unified_pipeline.py's layout keep their meaning,
    because stage_train4d is reused verbatim and looks them up by name."""
    return {
        "render": out_dir / "render",
        "flipbook_src": out_dir / "flipbook_src",
        "eval_src": out_dir / "eval_src",
        "composite": out_dir / "composite",
        "calib": out_dir / "calib",
        "calib_undistorted": out_dir / "calib_undistorted",
        "rig_gt": out_dir / "rig_gt_transforms.json",

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
    if args.background_plates:
        render_args.append("--background_plates")
    run_script("render_blender_rig.py", render_args, conda_env=CONDA_ENV,
               label="render_blender_rig.py (rig + eval cameras)")


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
        # Training always uses Blender's own poses and the render's own
        # alpha. Recorded so records stay comparable once estimated poses
        # or production masks become options.
        "poses": "gt",
        "masks": "gt",
        "train": {"iterations": args.total_train_iters,
                  "num_pts": args.num_pts,
                  "downscale": args.dataset_downscale},
    }
    stats_path = L["render"] / "render_stats.json"
    if stats_path.is_file():
        record["render"].update(json.loads(stats_path.read_text()))
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
                   help="Also render the backdrop alone, one plate per rig "
                        "camera, and composite the subject over it")
    p.add_argument("--skip_existing", action="store_true",
                   help="Keep rendered images already on disk")

    p.add_argument("--start_from_stage", choices=STAGE_KEYS, default=STAGE_KEYS[0])
    p.add_argument("--stop_after_stage", choices=STAGE_KEYS, default=None)
    p.add_argument("--no_validate", action="store_true")

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
