#!/usr/bin/env python3
"""
predict_keypoints_2d.py

Predicts 2D human keypoints per camera using the Sapiens2 Goliath model
(308 keypoints: body + hands + feet + face), via Diffuman4D's own
deps/Diffuman4D/scripts/preprocess/predict_keypoints.py (which drives
deps/Diffuman4D/scripts/preprocess/sapiens/2/demo/vis_pose.py).

Requires torchcodec in the active env (even though we never decode
video) -- vis_pose.py's sibling adhoc_video_dataset.py unconditionally
imports it at module load time.

conda env: sapiens2

Requires SAPIENS_CHECKPOINT_ROOT env var set to a directory containing
pose/sapiens2_1b_pose.safetensors (+ detector/detr-resnet-101-dc5, else
it's auto-downloaded). See --sapiens_checkpoint_root to override without
exporting the env var.

Usage:
    conda run -n sapiens2 python3 predict_keypoints_2d.py \\
        --images_dir /path/to/images_flat \\
        --out_kp2d_dir /path/to/poses_2d_flat \\
        --fmasks_dir /path/to/fmasks_flat
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIFFUMAN4D_ROOT = REPO_ROOT / "deps" / "Diffuman4D"

PREDICT_KEYPOINTS_SCRIPT = DIFFUMAN4D_ROOT / "scripts" / "preprocess" / "predict_keypoints.py"


def split_combined_predictions(predictions_json: Path, out_kp2d_dir: Path, images_dir_name: str):
    """vis_pose.py writes one combined
    <out_kp2d_dir>/<out_kp2d_dir.name>_predictions.json rather than a
    per-camera file per image. Split it into
    <out_kp2d_dir>/<images_dir_name>/<camera_label>.json to match the
    layout split_keypoints_per_camera.py already expects."""
    try:
        with open(predictions_json) as f:
            data = json.load(f)
        frames = data["frames"]
    except json.JSONDecodeError as e:
        print(f"Error: {predictions_json} is not valid JSON: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: {predictions_json} is missing expected key {e}")
        sys.exit(1)

    dest_dir = out_kp2d_dir / images_dir_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for frame in frames:
        try:
            label = Path(frame["image_name"]).stem
            instances = frame.get("instances") or []
            if not instances:
                print(f"  WARNING: no detected instance for {frame['image_name']}, skipping")
                continue
            inst = instances[0]
            out = {"instance_info": [{
                "keypoints": inst["keypoints"],
                "keypoint_scores": inst["keypoint_scores"],
            }]}
        except KeyError as e:
            print(f"  WARNING: malformed frame entry (missing key {e}), skipping: {frame}")
            continue
        with open(dest_dir / f"{label}.json", "w") as f:
            json.dump(out, f)
        n += 1
    print(f"  Split {n} camera(s) from {predictions_json} into {dest_dir}")


def run_goliath308(args, ckpt_root):
    if not PREDICT_KEYPOINTS_SCRIPT.is_file():
        print(f"Error: {PREDICT_KEYPOINTS_SCRIPT} not found -- does this Diffuman4D checkout have "
              "Sapiens2/Goliath support? (needs the solipsist-studios fork's hloc_validation branch "
              "or newer, not the upstream zju3dv/Diffuman4D repo)")
        sys.exit(1)

    args.out_kp2d_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if ckpt_root:
        env["SAPIENS_CHECKPOINT_ROOT"] = str(ckpt_root)

    cmd = [
        sys.executable, str(PREDICT_KEYPOINTS_SCRIPT),
        str(args.images_dir), str(args.out_kp2d_dir),
        "--fmasks_dir", str(args.fmasks_dir),
    ]
    if args.sapiens_model_size != "1b":
        # predict_keypoints.py's own defaults (both "1b") are hardcoded into
        # its function signature, but every parameter is exposed via
        # fire.Fire -- these two overrides just point at a same-named
        # smaller checkpoint + its matching config, both already vendored
        # in the sapiens2 package (facebook/sapiens2-pose-<size> is also
        # already a recognized auto-download repo id, see
        # _default_pose_repo_candidates in predict_keypoints.py).
        size = args.sapiens_model_size
        cmd += [
            "--sapiens_ckpt_path", f"{ckpt_root}/pose/sapiens2_{size}_pose.safetensors",
            "--config_path",
            f"configs/keypoints308/shutterstock_goliath_3po/sapiens2_{size}_keypoints308_shutterstock_goliath_3po-1024x768.py",
        ]
    print("Running:", " ".join(cmd), f"  [SAPIENS_CHECKPOINT_ROOT={ckpt_root}]" if ckpt_root else "")
    result = subprocess.run(cmd, cwd=str(DIFFUMAN4D_ROOT / "scripts" / "preprocess"), env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)

    predictions_json = args.out_kp2d_dir / f"{args.out_kp2d_dir.name}_predictions.json"
    if not predictions_json.is_file():
        print(f"Error: expected combined predictions file not found at {predictions_json}")
        sys.exit(1)
    split_combined_predictions(predictions_json, args.out_kp2d_dir, args.images_dir.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", required=True, type=Path)
    parser.add_argument("--out_kp2d_dir", required=True, type=Path)
    parser.add_argument("--fmasks_dir", required=True, type=Path)
    parser.add_argument("--sapiens_checkpoint_root", default=None,
                        help="Overrides SAPIENS_CHECKPOINT_ROOT for this call")
    parser.add_argument("--sapiens_model_size", default="1b",
                        choices=["0.4b", "0.8b", "1b", "5b"],
                        help="Sapiens2 pose checkpoint size (default 1b, matching "
                             "predict_keypoints.py's own default -- production quality "
                             "baseline). Smaller sizes use dramatically less RAM at some "
                             "accuracy cost; auto-downloads via the matching "
                             "facebook/sapiens2-pose-<size> repo if not already cached "
                             "under --sapiens_checkpoint_root.")
    args = parser.parse_args()

    ckpt_root = args.sapiens_checkpoint_root or os.environ.get("SAPIENS_CHECKPOINT_ROOT")
    run_goliath308(args, ckpt_root)


if __name__ == "__main__":
    main()
