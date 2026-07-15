#!/usr/bin/env python3
"""
triangulate_and_project_keypoints.py

Wraps Diffuman4D's triangulate_skeleton.py to triangulate 3D keypoints
from the real cameras' 2D detections (poses_2d, from
split_keypoints_per_camera.py) into --out_kp3d_dir/--out_pcd_dir.

IMPORTANT: triangulate_skeleton.py can't skip projection entirely --
`Ks_proj, Ts_proj = zip(*[...for spa_label in spa_labels_proj])` crashes
with ValueError on an empty spa_labels_proj, and spa_labels_proj ALWAYS
resolves to a non-empty list (falling back to os.listdir(kp2d_dir)) when
neither --spa_labels_proj nor --spa_labels_proj_range is given, then
crashes with `TypeError: ... not NoneType` joining paths under
out_kp2d_proj_dir if that was never passed. This projected-keypoints
output isn't used anywhere in this build (it would only feed a
ring-generation step that doesn't exist yet), so this wrapper does the
minimum unavoidable work to satisfy triangulate_skeleton.py: projects
into just 1 camera (--spa_labels_proj_range=[0,1,1]) and writes it to a
temp directory that's deleted once the subprocess exits.

conda env: whichever has Diffuman4D's dependencies (easyvolcap, fire) --
confirmed usage ran this under the "queen" conda env.

Usage:
    python3 triangulate_and_project_keypoints.py \\
        --camera_path /path/to/transforms.json \\
        --kp2d_dir /path/to/poses_2d \\
        --out_kp3d_dir /path/to/poses_3d \\
        --out_pcd_dir /path/to/poses_pcd
"""

import argparse
import subprocess
import sys
import tempfile
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
    args = parser.parse_args()

    if not SCRIPT.is_file():
        print(f"Error: {SCRIPT} not found -- is the Diffuman4D submodule checked out?")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, str(SCRIPT),
            "--camera_path", str(args.camera_path),
            "--kp2d_dir", str(args.kp2d_dir),
            "--out_kp3d_dir", str(args.out_kp3d_dir),
            "--out_kp2d_proj_dir", tmp,
            "--spa_labels_proj_range=[0,1,1]",
        ]
        if args.out_pcd_dir:
            cmd += ["--out_pcd_dir", str(args.out_pcd_dir)]

        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(DIFFUMAN4D_ROOT))

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
