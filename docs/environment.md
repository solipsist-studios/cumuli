# Environment setup

Precise setup instructions for the three conda environments and system
tools this pipeline needs, plus known-good versions this has actually
been tested against. `README.md` and `docs/pipeline.md` mention which
env each stage needs; this doc is the "how do I actually create them"
reference.

## Requirements

- Linux, NVIDIA GPU with recent drivers (tested on Pop!_OS 24.04 /
  Ubuntu Noble, driver 580.x). CPU-only or non-NVIDIA GPUs are not
  supported -- HLOC, Sapiens, and Diffuman4D all require CUDA.
  Comfortably tested on a single RTX 4090 (24GB VRAM); smaller cards
  may need reduced `--hloc_max_keypoints` or Diffuman4D batch settings.
- No system-wide CUDA toolkit install is required -- each conda env
  below pulls its own CUDA runtime via the `torch`/`torchvision` wheel
  (`+cuXXX` build tag). Don't rely on a system `nvcc`.
- [Miniforge](https://github.com/conda-forge/miniforge) (or any
  conda/mamba distribution).

## Conda environments

Three envs are wired into `scripts/21_run_unified_pipeline.py`, which
dispatches each stage into the right one automatically -- you don't
need to `conda activate` manually except when running a stage's script
directly.

### `hloc` -- pose estimation (script 05)

```bash
conda create -n hloc python=3.10 -y
conda activate hloc
pip install torch torchvision  # CUDA 12/13 wheel, matches your driver
pip install -r deps/camera-calibration/requirements.txt  # if not already covered
pip install pycolmap hloc  # or: pip install -e deps/Hierarchical-Localization if vendored
```

Known-good combination (verified 2026-07-13): Python 3.10.20, torch
2.12.0+cu130, torchvision 0.27.0+cu130, pycolmap 4.0.4, hloc 1.5, numpy
2.2.6, scipy 1.15.3, Pillow 12.2.0.

### `diffuman4d` -- masks, Diffuman4D inference, nerfstudio conversion (scripts 07, 14, 15, and 18's `--retry`)

```bash
conda create -n diffuman4d python=3.10 -y
conda activate diffuman4d
pip install -r deps/Diffuman4D/requirements.txt
pip install -r deps/BiRefNet/requirements.txt
```

Known-good combination: Python 3.10.20, torch 2.7.1+cu126, torchvision
0.22.1+cu126, pycolmap 4.0.4, diffusers 0.33.1, transformers 4.49.0,
numpy 2.2.6, scipy 1.15.3, Pillow 12.2.0. **This env intentionally pins
an older CUDA build than `hloc`/`sapiens2`** -- it follows whatever
Diffuman4D's own upstream `requirements.txt` specifies. Don't force it
to match the other envs' torch/CUDA versions without checking that
Diffuman4D still works against them.

### `sapiens2` -- 2D keypoint prediction (script 08)

```bash
conda create -n sapiens2 python=3.12 -y
conda activate sapiens2
pip install torch torchvision  # CUDA 12/13 wheel, matches your driver
pip install transformers
```

Known-good combination: Python 3.12.13, torch 2.12.0+cu130, torchvision
0.27.0+cu130, transformers 5.12.1, numpy 2.4.6, scipy 1.17.1, Pillow
12.2.0.

Requires the `SAPIENS_CHECKPOINT_ROOT` env var pointed at a directory
laid out like:

```
<root>/detector/detr-resnet-101-dc5/...
<root>/pose/sapiens2_1b_pose.safetensors
```

Download the detector and pose checkpoints from Sapiens' released
weights and place them in that layout; pass the root via
`--sapiens_checkpoint_root` to the orchestrator (or set
`SAPIENS_CHECKPOINT_ROOT` directly for standalone script 08 runs).

### Everything else

Scripts 01-04, 06, 09-13, 16, 17, 20 have no special conda env
requirement beyond numpy/scipy/Pillow/plyfile -- run them from any env
with those installed (`base`/`hloc`/etc. all work).

## System-level tools (no conda env)

- `ffmpeg` / `ffprobe` on PATH -- required by stages 01-02. Any recent
  build works (tested against 6.1.1).
- `rawtherapee-cli` on PATH, or a flatpak install of RawTherapee --
  optional, only needed for stage 02's `--pp3_dir` color correction.
- `brush_app` -- the compiled Brush gaussian-splat trainer, invoked as
  a binary (no conda env). Either build from the `deps/brush` submodule
  (needs a Rust toolchain -- see that repo's own build instructions) or
  download/build a release binary and point the orchestrator at it.

## Viewer display for `brush_app --with_viewer`

`--with_viewer` (the default; see `docs/pipeline.md` for why
`--no_viewer` isn't recommended on some setups) needs `brush_app` to
connect to a real, composited X or Xwayland display with something
actually driving it -- a headless/dummy display with no compositor or
client attached has been observed to hang the viewer thread rather than
fail cleanly. If you're running headless (e.g. over SSH with no
physical display), set one up via a streaming stack (Sunshine+gamescope,
VNC, etc.) or a virtual compositor, and pass its display number via
`--display` (e.g. `--display :1`).

## External script dependencies (not conda, not submodules)

Clone [solipsist-studios/4dgs-utils](https://github.com/solipsist-studios/4dgs-utils)
somewhere and point these flags at it:

- `--multiframe_sfm_script <path>/multiframe_sfm.py` -- required by
  stage 05
- `--refine_script <path>/refine_poses_with_keypoints.py` -- used by
  optional stage 20 (defaults to `~/4dgs-utils/refine_poses_with_keypoints.py`)

## Calibration data

Per-camera calibration PKLs (from `deps/camera-calibration`) live
outside the repo and are passed via `--calib_dir`. If you recalibrate a
subset of cameras later, double check they end up at the same native
resolution as the rest of the rig -- a mismatched-resolution camera in
an otherwise-consistent calib folder has caused measurably worse pose
estimation in testing here.
