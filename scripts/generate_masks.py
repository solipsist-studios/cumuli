#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
generate_masks.py

Thin wrapper around Diffuman4D's BiRefNet-based background removal
(deps/Diffuman4D/scripts/preprocess/remove_background.py), producing a
foreground mask per camera.

conda env: diffuman4d

Usage:
    conda run -n diffuman4d python3 generate_masks.py \\
        --images_dir /path/to/images_flat \\
        --out_fmasks_dir /path/to/fmasks_flat \\
        [--image_ext .png]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from image_formats import SUPPORTED_IMAGE_EXTS

REPO_ROOT = Path(__file__).resolve().parent.parent
DIFFUMAN4D_ROOT = REPO_ROOT / "deps" / "Diffuman4D"
SCRIPT = DIFFUMAN4D_ROOT / "scripts" / "preprocess" / "remove_background.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", required=True, type=Path)
    parser.add_argument("--out_fmasks_dir", required=True, type=Path)
    parser.add_argument("--image_ext", default=".png", choices=SUPPORTED_IMAGE_EXTS)
    args = parser.parse_args()

    if not SCRIPT.is_file():
        print(f"Error: {SCRIPT} not found -- is the Diffuman4D submodule checked out?")
        sys.exit(1)

    args.out_fmasks_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPT),
        str(args.images_dir),
        str(args.out_fmasks_dir),
        "--image_ext", args.image_ext,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(DIFFUMAN4D_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
