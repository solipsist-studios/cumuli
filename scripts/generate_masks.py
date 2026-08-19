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


def default_batch_size():
    """8 on GPU, 1 on CPU (see --batch_size help for the memory numbers).
    A missing torch also means CPU: this wrapper normally runs inside the
    diffuman4d env (which has torch), but if it doesn't, GPU inference
    isn't happening either way, and the conservative batch size is the
    only safe guess."""
    try:
        import torch
    except ImportError:
        return 1
    return 8 if torch.cuda.is_available() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", required=True, type=Path)
    parser.add_argument("--out_fmasks_dir", required=True, type=Path)
    parser.add_argument("--image_ext", default=".png", choices=SUPPORTED_IMAGE_EXTS)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Images per BiRefNet inference batch. Default: 8 on GPU "
                             "(remove_background.py's own default, tuned for ~16GB of "
                             "GPU memory in fp16), 1 on CPU -- CPU inference runs fp32, "
                             "and a batch of 8 peaks at ~21.6GB RSS (measured 2026-07-30, "
                             "11-camera fixture), which OOM-kills a GitHub runner VM (only "
                             "~7.75GB RAM -- confirmed directly via this session's own "
                             "[resmon] telemetry on a real failing run; an earlier '16GB' "
                             "assumption here was wrong, see integration-tests-cpu.yml's "
                             "resmon step).")
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = default_batch_size()

    if not SCRIPT.is_file():
        print(f"Error: {SCRIPT} not found -- is the Diffuman4D submodule checked out?")
        sys.exit(1)

    args.out_fmasks_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPT),
        str(args.images_dir),
        str(args.out_fmasks_dir),
        "--image_ext", args.image_ext,
        "--batch_size", str(args.batch_size),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(DIFFUMAN4D_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
