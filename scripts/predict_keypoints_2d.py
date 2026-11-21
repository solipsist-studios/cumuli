#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
predict_keypoints_2d.py

Predicts 2D human keypoints per camera using the Sapiens2 Goliath model
(308 keypoints: body + hands + feet + face). It resolves the checkpoint,
config and person detector, runs the vendored Sapiens2 demo
scripts/vendor/sapiens2/vis_pose.py in a subprocess, and splits that
script's combined predictions file into one JSON per camera.

conda env: cumuli

Requires a checkpoint root: SAPIENS_CHECKPOINT_ROOT, or
--sapiens_checkpoint_root. The pose checkpoint lives at
<root>/pose/sapiens2_<size>_pose.safetensors and the DETR person detector
at <root>/detector/detr-resnet-101-dc5. Either one is downloaded from
Hugging Face into that location when missing (facebook/sapiens2-pose-<size>,
or SAPIENS_HF_REPO; facebook/detr-resnet-101-dc5, or
SAPIENS_DETECTOR_HF_REPO). <size> is "1b" by default (production quality
baseline); see --sapiens_model_size for the smaller checkpoints. This
repo's own CPU integration test uses 0.4b, since the 1b checkpoint alone
needs more RAM than a GitHub Actions runner has (see
tests/integration/test_pipeline_end_to_end.py's CPU_SAPIENS_MODEL_SIZE for
the measured numbers).

Usage:
    conda run -n cumuli python3 predict_keypoints_2d.py \\
        --images_dir /path/to/images_flat \\
        --out_kp2d_dir /path/to/poses_2d_flat \\
        --fmasks_dir /path/to/fmasks_flat
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

VIS_POSE_SCRIPT = Path(__file__).resolve().parent / "vendor" / "sapiens2" / "vis_pose.py"

POSE_REPO_TEMPLATE = "facebook/sapiens2-pose-{size}"
DETECTOR_REPO = "facebook/detr-resnet-101-dc5"
CONFIG_TEMPLATE = ("keypoints308/shutterstock_goliath_3po/"
                   "sapiens2_{size}_keypoints308_shutterstock_goliath_3po-1024x768.py")

# vis_pose.py settings, as Diffuman4D's predict_keypoints.py passed them.
KPT_THR = 0.3             # keypoint score threshold (also the overlay threshold)
MAX_BONE_SIGMA = 0.0      # MAD bone-length filter: off (the 308-point graph mixes scales)
MAX_BILATERAL_RATIO = 3.5 # left/right bone-length ratio that zeroes the longer side


def ensure_pose_checkpoint(ckpt_path: Path, size: str) -> Path:
    """Download the Sapiens2 pose checkpoint into ckpt_path if it is missing."""
    if ckpt_path.is_file():
        return ckpt_path
    from huggingface_hub import hf_hub_download

    repo = os.environ.get("SAPIENS_HF_REPO") or POSE_REPO_TEMPLATE.format(size=size)
    print(f"{ckpt_path} not found; downloading {ckpt_path.name} from Hugging Face repo '{repo}'")
    downloaded = hf_hub_download(repo_id=repo, filename=ckpt_path.name)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, ckpt_path)
    return ckpt_path


def ensure_detector(detector_dir: Path) -> str:
    """Return a local DETR snapshot directory, downloading it when missing.
    If the download fails, return the repo id instead: vis_pose.py hands it
    straight to from_pretrained, which then uses the Hugging Face cache."""
    if detector_dir.is_dir() and any(detector_dir.iterdir()):
        return str(detector_dir)
    from huggingface_hub import snapshot_download

    repo = os.environ.get("SAPIENS_DETECTOR_HF_REPO", DETECTOR_REPO)
    print(f"Detector snapshot not found; downloading '{repo}' to {detector_dir}")
    try:
        snapshot_download(repo_id=repo, local_dir=str(detector_dir))
        return str(detector_dir)
    except Exception as e:  # noqa: BLE001 -- any failure falls back to the remote id
        print(f"WARNING: detector snapshot download failed ({e}); falling back to repo id '{repo}'")
        return repo


def sapiens_config_path(size: str) -> Path:
    """The pose config shipped inside the installed sapiens package."""
    import sapiens

    path = Path(sapiens.__file__).parent / "pose" / "configs" / CONFIG_TEMPLATE.format(size=size)
    if not path.is_file():
        raise FileNotFoundError(f"Sapiens2 config not found at {path}: is the sapiens2 package "
                                "installed at the commit environment.yml pins?")
    return path


def default_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_model_files(ckpt_root: Path, size: str):
    """(pose checkpoint, config, detector) for vis_pose.py, downloading what is missing."""
    pose_ckpt = ensure_pose_checkpoint(ckpt_root / "pose" / f"sapiens2_{size}_pose.safetensors", size)
    detector = ensure_detector(ckpt_root / "detector" / "detr-resnet-101-dc5")
    return pose_ckpt, sapiens_config_path(size), detector


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
    if not ckpt_root:
        print("Error: no Sapiens checkpoint root. Set SAPIENS_CHECKPOINT_ROOT or pass "
              "--sapiens_checkpoint_root (a directory holding pose/ and detector/; missing "
              "checkpoints are downloaded into it).")
        sys.exit(1)

    pose_ckpt, config, detector = resolve_model_files(Path(ckpt_root), args.sapiens_model_size)
    args.out_kp2d_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(VIS_POSE_SCRIPT),
        detector, str(config), str(pose_ckpt),
        "--input", str(args.images_dir),
        "--output", str(args.out_kp2d_dir),
        "--fmasks-dir", str(args.fmasks_dir),
        "--kpt-thr", str(KPT_THR),
        "--max-bone-sigma", str(MAX_BONE_SIGMA),
        "--max-bilateral-ratio", str(MAX_BILATERAL_RATIO),
        "--device", default_device(),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
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
                        help="Sapiens2 pose checkpoint size (default 1b, the production "
                             "quality baseline). Smaller sizes use dramatically less RAM at "
                             "some accuracy cost; auto-downloads via the matching "
                             "facebook/sapiens2-pose-<size> repo if not already cached "
                             "under the checkpoint root.")
    args = parser.parse_args()

    ckpt_root = args.sapiens_checkpoint_root or os.environ.get("SAPIENS_CHECKPOINT_ROOT")
    run_goliath308(args, ckpt_root)


if __name__ == "__main__":
    main()
