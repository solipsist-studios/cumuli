#!/usr/bin/env python3
"""
12_triangulate_and_project_keypoints.py

Wraps Diffuman4D's triangulate_skeleton.py to:
  1. Triangulate 3D keypoints from the real cameras' 2D detections
     (poses_2d, from 09_split_keypoints_per_camera.py / 10_resize_for_diffuman4d.py).
  2. Project those 3D keypoints into EVERY camera in the 48-cam ring
     (from 11_generate_camera_ring.py), producing the per-view skeleton
     keypoints Diffuman4D needs to condition its diffusion model on views
     that have no real image.

IMPORTANT: triangulate_skeleton.py's spa_labels_proj ALWAYS resolves to a
non-empty list (falling back to os.listdir(kp2d_dir)) even when neither
--spa_labels_proj nor --spa_labels_proj_range is given -- it then
unconditionally joins paths under out_kp2d_proj_dir for every one of
those labels, crashing with `TypeError: ... not NoneType` if
out_kp2d_proj_dir was never passed. So --out_kp2d_proj_dir must ALWAYS be
supplied; if the caller doesn't need the projected 2D keypoints (e.g.
triangulating for the real cameras only, no ring), this wrapper defaults
it to a throwaway directory alongside --out_kp3d_dir rather than leaving
it unset. To get skeleton maps for all 48 ring cameras, pass
--out_kp2d_proj_dir explicitly (this wrapper's --spa_labels_proj_range
covers 0..n_total by default in that case).

conda env: whichever has Diffuman4D's dependencies (easyvolcap, fire) --
confirmed usage ran this under the "queen" conda env.

Usage:
    python3 12_triangulate_and_project_keypoints.py \\
        --camera_path /path/to/transforms_48cam.json \\
        --kp2d_dir /path/to/poses_2d \\
        --out_kp3d_dir /path/to/poses_3d \\
        --out_pcd_dir /path/to/poses_pcd \\
        --out_kp2d_proj_dir /path/to/poses_2d_proj \\
        [--n_total 48]
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIFFUMAN4D_ROOT = REPO_ROOT / "deps" / "Diffuman4D"
SCRIPT = DIFFUMAN4D_ROOT / "scripts" / "preprocess" / "triangulate_skeleton.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera_path", required=True, type=Path)
    parser.add_argument("--kp2d_dir", required=True, type=Path)
    parser.add_argument("--out_kp3d_dir", required=True, type=Path)
    parser.add_argument("--out_pcd_dir", type=Path)
    parser.add_argument("--out_kp2d_proj_dir", type=Path)
    parser.add_argument("--n_total", type=int, default=48)
    args = parser.parse_args()

    if not SCRIPT.is_file():
        print(f"Error: {SCRIPT} not found -- is the Diffuman4D submodule checked out?")
        sys.exit(1)

    out_kp2d_proj_dir = args.out_kp2d_proj_dir
    if out_kp2d_proj_dir is None:
        out_kp2d_proj_dir = args.out_kp3d_dir.parent / f"{args.out_kp3d_dir.name}_kp2d_proj_unused"
        print(f"--out_kp2d_proj_dir not given; defaulting to {out_kp2d_proj_dir} "
              "(triangulate_skeleton.py requires a valid path here regardless -- see module docstring)")

    cmd = [
        sys.executable, str(SCRIPT),
        "--camera_path", str(args.camera_path),
        "--kp2d_dir", str(args.kp2d_dir),
        "--out_kp3d_dir", str(args.out_kp3d_dir),
        "--out_kp2d_proj_dir", str(out_kp2d_proj_dir),
    ]
    if args.out_pcd_dir:
        cmd += ["--out_pcd_dir", str(args.out_pcd_dir)]
    cmd += [f"--spa_labels_proj_range=[0,{args.n_total},1]"]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(DIFFUMAN4D_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
