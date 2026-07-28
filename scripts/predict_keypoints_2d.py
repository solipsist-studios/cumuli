#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
predict_keypoints_2d.py

Predicts 2D human keypoints per camera. Two models are supported:

  --model goliath308 (default): the Sapiens2 Goliath model (308 keypoints:
    body + hands + feet + face), via Diffuman4D's own
    deps/Diffuman4D/scripts/preprocess/predict_keypoints.py (which drives
    deps/Diffuman4D/scripts/preprocess/sapiens/2/demo/vis_pose.py). This is
    the higher-fidelity model and the one every downstream stage (pose
    refinement, triangulation, mask cleanup) is tuned against -- prefer
    it unless you have a specific reason not to.

  --model coco_wholebody133 (legacy fallback): the older Sapiens
    COCO-WholeBody model (133 keypoints), via
    deps/Diffuman4D/scripts/preprocess/sapiens/lite/demo/vis_pose.py
    directly. Requires mmdet (confirmed present in the sapiens_lite conda
    env; sapiens2 lacks it by default). Works around a python-fire
    --gpu_ids arg-quoting quirk in that older vis_pose.py internally --
    see the --gpu_ids build site below.

Both modes require torchcodec in the active env (even though we never
decode video) -- vis_pose.py's sibling adhoc_video_dataset.py
unconditionally imports it at module load time.

conda env: sapiens2 for --model goliath308 (default); sapiens_lite for
--model coco_wholebody133 (confirmed to have mmdet/json_tricks already).

Requires SAPIENS_CHECKPOINT_ROOT env var set to a directory containing
pose/sapiens2_1b_pose.safetensors (+ detector/detr-resnet-101-dc5, else
it's auto-downloaded) for goliath308, or the sapiens_lite torchscript
layout for coco_wholebody133. See --sapiens_checkpoint_root to override
without exporting the env var.

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

LITE_DEMO_DIR = DIFFUMAN4D_ROOT / "scripts" / "preprocess" / "sapiens" / "lite" / "demo"
LITE_VIS_POSE_SCRIPT = LITE_DEMO_DIR / "vis_pose.py"
LITE_DET_CONFIG = LITE_DEMO_DIR / "mmdetection_cfg" / "rtmdet_m_640-8xb32_coco-person_no_nms.py"


def split_combined_predictions(predictions_json: Path, out_kp2d_dir: Path, images_dir_name: str):
    """The Sapiens2 vis_pose.py writes one combined
    <out_kp2d_dir>/<out_kp2d_dir.name>_predictions.json rather than a
    per-camera file per image. Split it into
    <out_kp2d_dir>/<images_dir_name>/<camera_label>.json to match the
    layout split_keypoints_per_camera.py already expects (same layout
    the coco_wholebody133 lite path produces natively)."""
    with open(predictions_json) as f:
        data = json.load(f)

    dest_dir = out_kp2d_dir / images_dir_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for frame in data["frames"]:
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
    print("Running:", " ".join(cmd), f"  [SAPIENS_CHECKPOINT_ROOT={ckpt_root}]" if ckpt_root else "")
    result = subprocess.run(cmd, cwd=str(DIFFUMAN4D_ROOT / "scripts" / "preprocess"), env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)

    predictions_json = args.out_kp2d_dir / f"{args.out_kp2d_dir.name}_predictions.json"
    if not predictions_json.is_file():
        print(f"Error: expected combined predictions file not found at {predictions_json}")
        sys.exit(1)
    split_combined_predictions(predictions_json, args.out_kp2d_dir, args.images_dir.name)


def run_coco_wholebody133(args, ckpt_root):
    if not LITE_VIS_POSE_SCRIPT.is_file():
        print(f"Error: {LITE_VIS_POSE_SCRIPT} not found -- is the Diffuman4D submodule checked out?")
        sys.exit(1)
    if not ckpt_root:
        print("Error: SAPIENS_CHECKPOINT_ROOT env var is not set (e.g. export SAPIENS_CHECKPOINT_ROOT=~/sapiens), "
              "and --sapiens_checkpoint_root was not given")
        sys.exit(1)

    sapiens_ckpt_path = (
        f"{ckpt_root}/torchscript/pose/checkpoints/sapiens_2b/"
        "sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2"
    )
    detector_ckpt_path = (
        f"{ckpt_root}/detector/checkpoints/rtmpose/"
        "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    )

    args.out_kp2d_dir.mkdir(parents=True, exist_ok=True)

    # gpu_ids is quoted as a Python string literal (e.g. '"0"') so
    # python-fire keeps it as a str instead of literal-eval'ing an
    # unquoted "0" into int 0 or "0,1" into a tuple.
    cmd = [
        sys.executable, str(LITE_VIS_POSE_SCRIPT),
        sapiens_ckpt_path,
        "--det-checkpoint", detector_ckpt_path,
        "--det-config", str(LITE_DET_CONFIG),
        "--images_dir", str(args.images_dir),
        "--fmasks_dir", str(args.fmasks_dir),
        "--output_dir", str(args.out_kp2d_dir),
        "--skip_exists",
        f'--gpu_ids="{args.gpu_ids}"',
        "--num_workers", str(args.num_workers),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(LITE_DEMO_DIR))
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", required=True, type=Path)
    parser.add_argument("--out_kp2d_dir", required=True, type=Path)
    parser.add_argument("--fmasks_dir", required=True, type=Path)
    parser.add_argument("--model", choices=["goliath308", "coco_wholebody133"], default="goliath308",
                        help="goliath308 (default, 308kp, needs the Sapiens2 fork support) or "
                             "coco_wholebody133 (legacy, 133kp, needs mmdet)")
    parser.add_argument("--sapiens_checkpoint_root", default=None,
                        help="Overrides SAPIENS_CHECKPOINT_ROOT for this call")
    parser.add_argument("--gpu_ids", default="0", help="(coco_wholebody133 only) Comma-separated GPU ids")
    parser.add_argument("--num_workers", type=int, default=4, help="(coco_wholebody133 only)")
    args = parser.parse_args()

    ckpt_root = args.sapiens_checkpoint_root or os.environ.get("SAPIENS_CHECKPOINT_ROOT")

    if args.model == "goliath308":
        run_goliath308(args, ckpt_root)
    else:
        run_coco_wholebody133(args, ckpt_root)


if __name__ == "__main__":
    main()
