# Environment setup

Precise setup instructions for the four conda environments and system
tools this pipeline needs, plus known-good versions this has actually
been tested against. `README.md` and `docs/pipeline.md` mention which
env each stage needs; this doc is the "how do I actually create them"
reference. `envs/*.yml` pin the exact known-good combination for each --
`conda env create -f envs/<name>.yml` is the fastest path; the manual
steps below explain what's actually in them.

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

Three envs are wired into `scripts/run_unified_pipeline.py`, which
dispatches each stage into the right one automatically -- you don't
need to `conda activate` manually except when running a stage's script
directly.

### `hloc` -- pose estimation (run_hloc.py)

```bash
conda env create -f envs/hloc.yml
```

That pins the exact known-good combination below. Equivalent by hand:

```bash
conda create -n hloc python=3.10 -y
conda activate hloc
pip install torch torchvision  # CUDA 12/13 wheel, matches your driver
pip install -r deps/camera-calibration/requirements.txt  # if not already covered
pip install pycolmap
pip install git+https://github.com/cvg/Hierarchical-Localization.git@c13273bd0ecc2917a35910fd843712a1c6243193  # hloc -- NOT on plain PyPI, see envs/hloc.yml
```

Known-good combination (verified 2026-07-13): Python 3.10.20, torch
2.12.0+cu130, torchvision 0.27.0+cu130, pycolmap 4.0.4, hloc 1.5 (commit
c13273b above -- this exact source was missing from both this doc and
envs/hloc.yml until 2026-07-29, when a fresh CI checkout failed trying to
`pip install hloc==1.5` from plain PyPI, which doesn't have it; recovered
from pip's own direct_url.json on the working dev env), numpy 2.2.6,
scipy 1.15.3, Pillow 12.2.0.

### `diffuman4d` -- masks, Diffuman4D inference, nerfstudio conversion (generate_masks.py, the not-yet-built Diffuman4D-branch scripts, and clean_masks.py's `--retry`)

```bash
conda env create -f envs/diffuman4d.yml
```

Equivalent by hand:

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

### `sapiens2` -- 2D keypoint prediction (predict_keypoints_2d.py)

```bash
conda env create -f envs/sapiens2.yml
```

Equivalent by hand:

```bash
conda create -n sapiens2 python=3.12 -y
conda activate sapiens2
pip install torch torchvision  # CUDA 12/13 wheel, matches your driver
pip install transformers
pip install git+https://github.com/facebookresearch/sapiens2.git@7e5bae88456ac418ff0e58e74106c9fe192055d4  # predict_keypoints_2d.py imports sapiens.pose
```

Known-good combination: Python 3.12.13, torch 2.12.0+cu130, torchvision
0.27.0+cu130, transformers 5.12.1, numpy 2.4.6, scipy 1.17.1, Pillow
12.2.0, sapiens 2.0.0 (commit 7e5bae8 above -- this package was missing
from both this doc and envs/sapiens2.yml until 2026-07-28; recovered
from pip's own direct_url.json on the working dev env, since neither
file had recorded it).

Requires the `SAPIENS_CHECKPOINT_ROOT` env var pointed at a directory
laid out like:

```
<root>/detector/detr-resnet-101-dc5/...
<root>/pose/sapiens2_1b_pose.safetensors
```

Download the detector and pose checkpoints from Sapiens' released
weights and place them in that layout; pass the root via
`--sapiens_checkpoint_root` to the orchestrator (or set
`SAPIENS_CHECKPOINT_ROOT` directly for standalone predict_keypoints_2d.py runs).

### `queen` -- generic scripts + triangulation (`--generic_env`/`--triangulate_env`)

```bash
conda env create -f envs/queen.yml
conda run -n queen pip install --no-deps git+https://github.com/zju3dv/EasyVolcap.git@4cb3c000a31b8764834c79792b355f110d947e75
```

This is `run_unified_pipeline.py`'s default target for `--generic_env`
(make_sync_grid.py, undistort_frames.py, build_flat_dataset.py,
build_colmap_sparse.py, run_pose_refinement.py, compute_sync_offsets.py)
and `--triangulate_env` (triangulate_and_project_keypoints.py, which
wraps Diffuman4D's triangulate_skeleton.py). The orchestrator never runs
these bare with its own launching Python -- always through this named
env -- so it has to actually exist, unlike a genuinely env-agnostic
script. Missing from every committed env spec entirely until 2026-07-29
(found when a fresh CI checkout failed with "Not a conda environment:
.../envs/queen").

Deliberately narrower than the real local `queen` env this was recovered
from (which is much larger -- also used for unrelated local
experimentation): this lists only what's actually imported by the real
code path, traced by hand, import by import, recursively, through to
easyvolcap's own internal utility modules. Notably, `open3d` isn't
imported anywhere in this path despite an earlier version of this repo's
own `--triangulate_env` help text claiming it's needed.

Known-good combination (verified 2026-07-29): Python 3.11.15, numpy
2.4.4, scipy 1.17.1, Pillow 12.2.0, plyfile 1.1.3, opencv-python
4.13.0.92, fire 0.7.1, torch 2.12.0 (CPU-safe -- the one real usage,
Diffuman4D's camera_parser.py, is plain tensor math, no `.cuda()`/device
placement), easyvolcap 0.0.0 (commit 4cb3c00 above, `--no-deps` --
skips its own heavy declared dependencies, none of which the real code
path here touches), pdbr 0.9.7, rich 15.0.0, ujson 5.13.0, ruamel.yaml
0.19.1, tqdm 4.67.3.

## System-level tools (no conda env)

- `ffmpeg` / `ffprobe` on PATH -- required by compute_sync_offsets.py
  and extract_synced_frames.py. Any recent build works (tested against
  6.1.1).
- `rawtherapee-cli` on PATH, or a flatpak install of RawTherapee --
  optional, only needed for extract_synced_frames.py's `--pp3_dir` color
  correction.
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

## Vendored script dependencies

`multiframe_sfm.py` (used by `run_hloc.py`) and
`refine_poses_with_keypoints.py` (used by the `run_pose_refinement.py`
wrapper) live directly in `scripts/` -- no external checkout needed.
Pass `--multiframe_sfm_script` / `--refine_script` only if you want to
point at a different copy (e.g. while testing local changes to them).

## Calibration data

Per-camera calibration PKLs (from `deps/camera-calibration`) live
outside the repo and are passed via `--calib_dir`. If you recalibrate a
subset of cameras later, double check they end up at the same native
resolution as the rest of the rig -- a mismatched-resolution camera in
an otherwise-consistent calib folder has caused measurably worse pose
estimation in testing here.
