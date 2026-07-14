#!/usr/bin/env python3
"""
run_hloc.py

Runs HLOC-based multi-camera pose estimation on a set of undistorted
single-frame images, producing camera poses in nerfstudio transforms.json
format. Wraps vendor/multiframe_sfm.py (override with
--multiframe_sfm_script if you're testing local changes to it).

This wraps two things that multiframe_sfm.py needs set up first:
  1. Restructuring flat undistorted images (<camera_id>.jpg) into the
     per-camera folder layout it expects (Camera_<camera_id>/0000.jpg).
     multiframe_sfm.py derives each camera_label directly from this
     folder name, and looks up calibration pkls via a glob on that same
     label (Camera_<camera_id>*.pkl) -- so no separate symlinking step is
     needed as long as the folder name and the calibration pkl name and
     the init_transforms.json camera_label all agree (all three use
     plain "Camera_<camera_id>" here).
  2. Building a bootstrap init_transforms.json from the undistorted
     (zero-distortion) calibration pkls -- identity poses, real
     intrinsics, since HLOC only needs a starting point for intrinsics
     and refines poses itself.

conda env: hloc (must contain hloc, pycolmap)

Usage:
    conda run -n hloc python3 run_hloc.py \\
        --undistorted_dir /path/to/undistorted \\
        --undistorted_pkl_dir /path/to/undistorted_pkls \\
        --outputs_dir /path/to/solipsist_out \\
        [--num_timestamps 1] [--feature_type superpoint]

Output:
    outputs_dir/transforms_multiframe.json -- real camera poses (nerfstudio
    format), consumed by every later stage.
"""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

from image_formats import SUPPORTED_IMAGE_EXTS


def restructure_flat_to_percam(undistorted_dir: Path, image_ext: str):
    """Move <camera_id>.jpg -> Camera_<camera_id>/0000.jpg in place.

    offline_undistort.py (undistort_frames.py) prefixes its output
    filenames with "undistorted_" (e.g. undistorted_0001.jpg) -- strip
    that back off so cam_id matches the calibration pkl naming
    (Camera_0001.pkl) rather than becoming "undistorted_0001".
    """
    camera_ids = []
    for f in sorted(undistorted_dir.glob(f"*{image_ext}")):
        cam_id = f.stem
        if cam_id.startswith("undistorted_"):
            cam_id = cam_id[len("undistorted_"):]
        cam_dir = undistorted_dir / f"Camera_{cam_id}"
        cam_dir.mkdir(exist_ok=True)
        target = cam_dir / f"0000{image_ext}"
        if not target.exists():
            f.rename(target)
        camera_ids.append(cam_id)
    return camera_ids


def build_init_transforms(undistorted_pkl_dir: Path, camera_ids, out_path: Path,
                           fallback_w=5568, fallback_h=4872,
                           fallback_fx=1786.75, fallback_fy=1786.84,
                           fallback_cx=2766.17, fallback_cy=2458.04):
    """Bootstrap transforms.json with identity poses and real intrinsics
    read from each camera's undistorted (zero-distortion) calibration pkl.
    HLOC refines the poses; it just needs correct intrinsics to start from.

    camera_label here MUST match the "Camera_<camera_id>" folder name
    that restructure_flat_to_percam() creates under frames_root, since
    multiframe_sfm.py matches init_transforms.json frames to cameras by
    that exact label (or a file_path/prefix match).
    """
    init_tf = {"camera_model": "PINHOLE", "frames": []}

    for cam_id in camera_ids:
        pkl_path = undistorted_pkl_dir / f"Camera_{cam_id}.pkl"
        label = f"Camera_{cam_id}"

        if pkl_path.is_file():
            with open(pkl_path, "rb") as f:
                calib = pickle.load(f)
            K = calib["camera_matrix"]
            img_size = calib["image_size"]  # (width, height)
            w, h = int(img_size[0]), int(img_size[1])
            fx, fy = float(K[0][0]), float(K[1][1])
            cx, cy = float(K[0][2]), float(K[1][2])
        else:
            print(f"  WARNING: no calibration pkl for {cam_id}, using fallback intrinsics")
            w, h = fallback_w, fallback_h
            fx, fy = fallback_fx, fallback_fy
            cx, cy = fallback_cx, fallback_cy

        init_tf["frames"].append({
            "camera_label": label,
            "file_path": f"images/{label}/0000.jpg",
            "w": w, "h": h,
            "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        })

    with open(out_path, "w") as f:
        json.dump(init_tf, f, indent=2)
    print(f"Wrote bootstrap {out_path} ({len(camera_ids)} cameras)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--undistorted_dir", required=True, type=Path)
    parser.add_argument("--undistorted_pkl_dir", required=True, type=Path)
    parser.add_argument("--outputs_dir", required=True, type=Path)
    parser.add_argument("--multiframe_sfm_script", type=Path,
                         default=Path(__file__).parent / "vendor" / "multiframe_sfm.py",
                         help="Path to multiframe_sfm.py (default: the vendored copy in scripts/vendor/)")
    parser.add_argument("--num_timestamps", type=int, default=1)
    parser.add_argument("--feature_type", default="superpoint", choices=["superpoint", "aliked"])
    parser.add_argument("--image_ext", default=".jpg", choices=SUPPORTED_IMAGE_EXTS)
    parser.add_argument("--skip_setup", action="store_true",
                         help="Skip restructure/init_transforms steps (if already done)")
    args = parser.parse_args()

    if not args.multiframe_sfm_script.is_file():
        print(f"Error: {args.multiframe_sfm_script} not found")
        sys.exit(1)

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    init_transforms_path = args.outputs_dir.parent / "init_transforms.json"

    if not args.skip_setup:
        print("Restructuring flat undistorted images into per-camera folders...")
        camera_ids = restructure_flat_to_percam(args.undistorted_dir, args.image_ext)
        print(f"  {len(camera_ids)} cameras: {camera_ids}")

        print("Building bootstrap init_transforms.json...")
        build_init_transforms(args.undistorted_pkl_dir, camera_ids, init_transforms_path)
    else:
        print("Skipping setup steps (--skip_setup); assuming already done.")

    cmd = [
        sys.executable, str(args.multiframe_sfm_script),
        "--frames_root", str(args.undistorted_dir),
        "--init_transforms", str(init_transforms_path),
        "--undistorted_calibration_dir", str(args.undistorted_pkl_dir),
        "--outputs_dir", str(args.outputs_dir),
        "--num_timestamps", str(args.num_timestamps),
        "--feature_type", args.feature_type,
    ]
    print("\nRunning multiframe_sfm.py:")
    print(" ", " ".join(cmd))
    env = {
        "__NV_PRIME_RENDER_OFFLOAD": "1",
        "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    import os
    full_env = {**os.environ, **env}
    result = subprocess.run(cmd, env=full_env)
    if result.returncode != 0:
        print("multiframe_sfm.py failed.")
        sys.exit(result.returncode)

    out_transforms = args.outputs_dir / "transforms_multiframe.json"
    print(f"\nDone. Real camera poses at: {out_transforms}")


if __name__ == "__main__":
    main()
