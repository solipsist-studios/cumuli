#!/usr/bin/env python3
"""
train_brush.py

Thin wrapper around the Brush gaussian-splat trainer (deps/brush),
defaulting to the flag values that produced the best confirmed results
historically (75k steps, lr-coeffs-sh-scale 80). Trains directly off a
transforms.json (no COLMAP sparse/0 required -- see prepare_brush_dataset.py),
or off a COLMAP sparse/0 dataset (see build_colmap_sparse.py).

Masked (subject-only) training: this brush_app build has NO --alpha-mode
flag (verified against `brush_app --help` -- there is no such option,
despite it being referenced in some older notes). Alpha transparency is
picked up automatically whenever the loaded images have an alpha
channel; --match_alpha_weight controls how strongly that alpha is
matched (Brush's own default: 0.1). So masked training just means:
bake masks into the image alpha channel (build_colmap_sparse.py
--masks_dir) and train normally -- no special flag needed. Brush has
been observed to silently ignore a separate masks/ folder placed
alongside same-named/same-extension images -- baked alpha is the
reliable route, not a separate masks dir.

Usage:
    python3 train_brush.py \\
        --data /path/to/brush_dataset \\
        --brush_app ~/brush-app-x86_64-unknown-linux-gnu/brush_app \\
        --export_path ~/brush_output \\
        --export_name heidi_75k_{iter}.ply \\
        [--total_steps 75000] [--with_viewer] [--match_alpha_weight 0.1]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="Dataset dir containing transforms.json")
    parser.add_argument("--brush_app", required=True, type=Path)
    parser.add_argument("--export_path", required=True, type=Path)
    parser.add_argument("--export_name", default="splat_{iter}.ply")
    parser.add_argument("--total_steps", type=int, default=75000)
    parser.add_argument("--growth_grad_threshold", type=float, default=0.000003)
    parser.add_argument("--growth_select_fraction", type=float, default=0.3)
    parser.add_argument("--refine_every", type=int, default=50)
    parser.add_argument("--growth_stop_iter", type=int, default=70000)
    parser.add_argument("--mean_noise_weight", type=float, default=20)
    parser.add_argument("--max_resolution", type=int, default=1024)
    parser.add_argument("--opac_loss_weight", type=float, default=0.0)
    parser.add_argument("--lr_coeffs_sh_scale", type=float, default=80)
    parser.add_argument("--export_every", type=int, default=10000)
    parser.add_argument("--with_viewer", action="store_true")
    parser.add_argument("--match_alpha_weight", type=float, default=None,
                         help="Weight of L1 loss on alpha for RGBA (masked) datasets. "
                              "Brush auto-detects the alpha channel; this just tunes how "
                              "strongly it's matched (Brush's own default: 0.1).")
    parser.add_argument("--display", default=":99")
    args = parser.parse_args()

    if not args.brush_app.is_file():
        print(f"Error: brush_app not found at {args.brush_app}")
        sys.exit(1)

    args.export_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.brush_app),
        str(args.data),
        "--total-steps", str(args.total_steps),
        "--growth-grad-threshold", str(args.growth_grad_threshold),
        "--growth-select-fraction", str(args.growth_select_fraction),
        "--refine-every", str(args.refine_every),
        "--growth-stop-iter", str(args.growth_stop_iter),
        "--mean-noise-weight", str(args.mean_noise_weight),
        "--max-resolution", str(args.max_resolution),
        "--opac-loss-weight", str(args.opac_loss_weight),
        "--lr-coeffs-sh-scale", str(args.lr_coeffs_sh_scale),
        "--export-every", str(args.export_every),
        "--export-path", str(args.export_path),
        "--export-name", args.export_name,
    ]
    if args.with_viewer:
        cmd.insert(2, "--with-viewer")
    if args.match_alpha_weight is not None:
        cmd += ["--match-alpha-weight", str(args.match_alpha_weight)]

    env = dict(os.environ)
    env["DISPLAY"] = args.display
    env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
