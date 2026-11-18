#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
generate_masks.py

Thin wrapper around birefnet_masks.py (BiRefNet background removal),
producing a foreground mask per camera. It picks a batch size that suits
the device and runs the model in a subprocess.

conda env: cumuli

Usage:
    conda run -n cumuli python3 generate_masks.py \\
        --images_dir /path/to/images_flat \\
        --out_fmasks_dir /path/to/fmasks_flat \\
        [--image_ext .png]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from image_formats import SUPPORTED_IMAGE_EXTS

SCRIPT = Path(__file__).resolve().parent / "birefnet_masks.py"


def default_batch_size():
    """8 on GPU, 1 on CPU (see --batch_size help for the memory numbers).
    A missing torch also means CPU: this wrapper normally runs inside the
    cumuli env (which has torch), but if it doesn't, GPU inference
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
                             "(birefnet_masks.py's own default, tuned for ~16GB of "
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

    args.out_fmasks_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPT),
        str(args.images_dir),
        str(args.out_fmasks_dir),
        "--image_ext", args.image_ext,
        "--batch_size", str(args.batch_size),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
