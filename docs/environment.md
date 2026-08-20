<!--
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
-->

# Environment setup

Precise setup instructions for the single `cumuli` conda environment
and system tools this pipeline needs, plus the known-good versions this
has actually been tested against. Provision it with
`bash scripts/setup_cumuli_env.sh` -- NOT `conda env create` alone:
`envs/cumuli.yml` is the package manifest, and the script adds four
installs a yml cannot express (an editable, submodule-recursive hloc
checkout, easyvolcap `--no-deps`, the OMG4 trainer's CUDA extensions,
and the optional cupy/cuml SPM accelerators).

## Requirements

- Linux, NVIDIA GPU with recent drivers (tested on Pop!_OS 24.04 /
  Ubuntu Noble, driver 580.x). CPU-only or non-NVIDIA GPUs are not
  supported -- HLOC, Sapiens, and Diffuman4D all require CUDA.
  Comfortably tested on a single RTX 4090 (24GB VRAM). Smaller cards
  may need reduced `--hloc_max_keypoints` or Diffuman4D batch settings.
- No system-wide CUDA toolkit install is required -- the env pulls its own CUDA runtime via the `torch`/`torchvision` wheel
  (`+cuXXX` build tag). Don't rely on a system `nvcc`.
- [Miniforge](https://github.com/conda-forge/miniforge) (or any
  conda/mamba distribution).

## The `cumuli` environment

One env serves every stage. `scripts/run_unified_pipeline.py` dispatches
each stage into it automatically (you do not need to `conda activate`
except when running a stage's script directly), and each stage's env
remains overridable through the `--*_env` flags as an escape hatch for
machines that keep a split setup.

Known-good combination, verified 2026-08-20 by a full six-stage pipeline
run on the take01 fixture (drift vs the previous five-env setup:
reprojection 5.44 px vs 5.57 golden, eval PSNR within 0.02 dB, SSIM and
LPIPS identical to 4 decimals):

- Python 3.12, torch 2.13.0+cu130, torchvision (cu130 pair)
- transformers 5.12.1 (serves both BiRefNet's dynamic model code and
  Sapiens2's DETR classes -- BiRefNet was verified against it by a real
  mask-generation run, not only an import)
- numpy 2.4.6, scipy 1.17.1, Pillow 12.2.0, opencv-python 4.13.0.92
- pycolmap 4.0.4, hloc @c13273b (editable, --no-deps, submodules),
  lightglue @eb42fee, sapiens @7e5bae8, easyvolcap @4cb3c00 (--no-deps)
- gsplat, torchmetrics, lpips, dahuffman, open3d 0.19.0, and the
  trainer CUDA extensions built by the setup script

The anchor pin is torch 2.13.0+cu130: the trainer's CUDA extensions are
ABI-compiled against it, and cupy/cuml come from the CUDA-13 wheel
family. Changing torch means rebuilding the extensions (rerun the setup
script).

A CUDA GPU is required for the `train4d` stage only. Every stage before
it, including the full 4D dataset build, runs on CPU
(`--stop_after_stage dataset4d`); the setup script's `--cpu` mode
provisions a CPU-wheel variant without the extensions, which is what CI
uses.

Sapiens checkpoints are separate from the env: set
`SAPIENS_CHECKPOINT_ROOT` (or `--sapiens_checkpoint_root`) to a
directory holding `detector/` (facebook/detr-resnet-101-dc5 snapshot)
and `pose/sapiens2_<size>_pose.safetensors`.

## System-level tools (no conda env)

- `ffmpeg` / `ffprobe` on PATH -- required by compute_sync_offsets.py
  and extract_synced_frames.py. Any recent build works (tested against
  6.1.1).
- `rawtherapee-cli` on PATH, or a flatpak install of RawTherapee --
  optional, only needed for extract_synced_frames.py's `--pp3_dir` color
  correction.

## Vendored script dependencies

`multiframe_sfm.py` (used by `run_hloc.py`) and
`refine_poses_with_keypoints.py` (used by the `run_pose_refinement.py`
wrapper) live directly in `scripts/` -- no external checkout needed.
Pass `--multiframe_sfm_script` / `--refine_script` only if you want to
point at a different copy (for example while testing local changes to
them).

## Calibration data

Per-camera calibration PKLs (from `deps/camera-calibration`) live
outside the repo and are passed via `--calib_dir`. If you recalibrate a
subset of cameras later, double check they end up at the same native
resolution as the rest of the rig -- a mismatched-resolution camera in
an otherwise-consistent calib folder has caused measurably worse pose
estimation in testing here.
