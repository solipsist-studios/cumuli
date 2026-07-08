#!/usr/bin/env python3
"""
09_split_keypoints_per_camera.py

Reorganizes the per-camera keypoint JSONs that
08_predict_keypoints_2d.py's split_combined_predictions() already wrote
to <out_kp2d_dir>/<images_dir_name>/<camera_label>.json (each in
{"instance_info": [{"keypoints": ..., "keypoint_scores": ...}]} format,
split there from vis_pose.py's single combined predictions file) into
the poses_2d/<camera_label>/<tem_label>.json layout
triangulate_skeleton.py expects.

Usage:
    python3 09_split_keypoints_per_camera.py \\
        --kp2d_flat_dir /path/to/poses_2d_flat \\
        --out_dir /path/to/poses_2d \\
        [--tem_label 000000]
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kp2d_flat_dir", required=True, type=Path,
                         help="Output dir passed to 08_predict_keypoints_2d.py as --out_kp2d_dir")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--tem_label", default="000000", help="6-digit frame label (default: 000000)")
    args = parser.parse_args()

    json_paths = sorted(args.kp2d_flat_dir.glob("*/*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No per-camera keypoint JSONs found under {args.kp2d_flat_dir}/*/*.json")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for src_path in json_paths:
        label = src_path.stem  # e.g. "00", "01", ...

        with open(src_path) as f:
            data = json.load(f)

        instance = data["instance_info"][0]
        if "keypoint_depths" not in instance:
            instance["keypoint_depths"] = [0.0] * len(instance["keypoints"])

        cam_dir = args.out_dir / label
        cam_dir.mkdir(parents=True, exist_ok=True)
        out_path = cam_dir / f"{args.tem_label}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        count += 1
        print(f"  {label} -> {out_path}")

    print(f"\nDone. Wrote structured 2D keypoints for {count} cameras to {args.out_dir}")


if __name__ == "__main__":
    main()
