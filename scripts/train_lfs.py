#!/usr/bin/env python3
"""
train_lfs.py

Thin wrapper around the LichtFeld-Studio (LFS) gaussian-splat trainer
(deps/LichtFeld-Studio, compiled binary), defaulting to the flag values
that qualified against Brush on real captures. Consumes the same COLMAP
sparse/0 dataset build_colmap_sparse.py produces (RGBA-baked masks and
all -- LFS uses the image alpha channel as the training mask
automatically, `use_alpha_as_mask` is on by default in its parameter
defaults).

Why LFS: on a like-for-like single-frame A/B against Brush (same
dataset, 30k iterations each), LFS with the defaults below was sharper
on most novel-view renders, produced a 3-10x LOWER semi-transparent
halo/fringe ratio, kept ~60% more Gaussians through mask-consistency
filtering, and trained at native 4K in roughly half Brush's 2048-capped
wall-clock time.

Three settings are load-bearing; changing them un-qualifies the recipe:

  --mask-mode alpha_consistent   (exposed here as --mask_mode) Without it
        LFS's soft alpha handling floods the model with semi-transparent
        haze (halo ratios measured 20-70x worse -- the subject sits in
        smoke). alpha_consistent enforces the mask's exact alpha and was
        the difference between failing and decisively passing the A/B.
  --max-width 0                  (exposed as --max_width) LFS's default
        is 3840, which SILENTLY downscales wider media. 0 disables the
        cap and trains at native resolution.
  cold start (no --init)         Warm-starting each frame of a sequence
        from the previous frame's splat (--init) compounds: the
        inherited Gaussian count grows monotonically until it pins at
        the strategy's --max-cap (observed: every frame filling to the
        5M default cap, ~6x slower per frame, in an unvalidated
        regime). Seed per-frame geometry through the INIT POINT CLOUD
        instead: build_colmap_sparse.py --seed_splat_ply, which LFS's
        default COLMAP init consumes with no flag here at all.

The default strategy (mrnf) is deliberate: mcmc scored comparably on
sharpness in the A/B but left visible opaque floater specks that
mask-consistency filtering under-caught. --max_cap defaults to a
generous runaway safety net well above observed natural growth (~850K
cold start), not a working constraint.

Unlike brush_app, the LFS binary is genuinely headless (--headless, no X
display needed) and writes <output_name>.ply exactly once at completion
(plus checkpoints/checkpoint.resume) -- no {iter} templating.

conda env: none (compiled binary; see deps/LichtFeld-Studio for build
requirements).

Usage:
    python3 train_lfs.py \\
        --data /path/to/train_set \\
        --lfs_app deps/LichtFeld-Studio/build/LichtFeld-Studio \\
        --export_path ~/lfs_output \\
        --output_name heidi_30k \\
        [--iters 30000] [--strategy mrnf] [--mask_mode alpha_consistent]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path,
                         help="Dataset root containing sparse/0 (build_colmap_sparse.py output)")
    parser.add_argument("--lfs_app", required=True, type=Path,
                         help="Path to the compiled LichtFeld-Studio binary")
    parser.add_argument("--export_path", required=True, type=Path)
    parser.add_argument("--output_name", default="splat_lfs",
                         help="Output stem: <export_path>/<output_name>.ply (no {iter} templating; "
                              "LFS writes the ply once at completion)")
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument("--strategy", default="mrnf", choices=["mrnf", "mcmc", "igs+"],
                         help="Densification strategy (default mrnf -- see module docstring for "
                              "why mcmc is not the default despite comparable sharpness)")
    parser.add_argument("--mask_mode", default="alpha_consistent",
                         choices=["none", "segment", "ignore", "alpha_consistent"],
                         help="LFS --mask-mode (default alpha_consistent; load-bearing, see docstring)")
    parser.add_argument("--max_width", type=int, default=0,
                         help="LFS --max-width image cap; 0 = native resolution (LFS's own default "
                              "of 3840 silently downscales wider media)")
    parser.add_argument("--max_cap", type=int, default=2000000,
                         help="Max Gaussians (runaway safety net, not a working constraint)")
    parser.add_argument("--images_subdir", default="images_rgba",
                         help="Images folder under --data, passed to LFS as --images. Default "
                              "matches build_colmap_sparse.py's --rgba_subdir (LFS's own default "
                              "is 'images' and it errors if the folder is absent)")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[],
                         help="Anything after this flag is passed to the LFS binary verbatim")
    args = parser.parse_args()

    if not args.lfs_app.is_file():
        print(f"Error: LFS binary not found at {args.lfs_app} "
              f"(build deps/LichtFeld-Studio first)", file=sys.stderr)
        return 1
    if not (args.data / "sparse" / "0").is_dir():
        print(f"Error: {args.data} has no sparse/0 (run build_colmap_sparse.py first)", file=sys.stderr)
        return 1

    args.export_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.lfs_app), "--headless",
        "-d", str(args.data),
        "-o", str(args.export_path),
        "--output-name", args.output_name,
        "-i", str(args.iters),
        "--max-width", str(args.max_width),
        "--mask-mode", args.mask_mode,
        "--strategy", args.strategy,
        "--max-cap", str(args.max_cap),
        "--images", args.images_subdir,
    ] + args.extra_args

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"LFS exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    out_ply = args.export_path / f"{args.output_name}.ply"
    if not out_ply.is_file():
        print(f"Error: LFS exited 0 but {out_ply} was not written", file=sys.stderr)
        return 1
    print(f"Trained splat: {out_ply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
