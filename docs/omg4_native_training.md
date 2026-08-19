<!--
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
-->

# Native OMG4 training (4D Gaussian Splatting → .sogst)

End-to-end recipe for training a 4D gaussian splat natively from a
multi-camera capture and exporting it as a streamable `.sogst` asset for
the Solipsist Studios fork of the PlayCanvas SuperSplat Viewer.
Substitute your own dataset paths throughout.

An alternative quick-test path exists for assets you already have as a
per-frame SOG flipbook: `sog_flipbook_to_sogst.py` repacks such a
sequence directly. It drops splats to fit a budget, so use it only to
test existing assets, never as the production path.

"OMG4" in this document is the *trainer* (MinShirley/OMG4), never a
file format. The container it feeds is `.sogst`, specified in
[sogst-format.md](sogst-format.md).

Two conda environments are needed:

- a trainer environment: Python 3.7, torch 1.12, CUDA 11.6. It runs the
  4D Gaussian Splatting trainer.
- a bake environment: Python 3.11, torch 2.9. It runs the bake and pack
  steps (`bake_sogst.py`, `sogst_pack.py`).

Trainer repo: fudan-zvg/4d-gaussian-splatting ("rotor" 4DGS), **with
the patches listed under "Trainer patches" below. The patches are
required.**

## 1. Build the D-NeRF-style dataset

Input layout: per-camera `images/Camera_XXXX/NNNN.png` frames,
`fmasks/` foreground masks, and
`transforms_undistorted_intrinsics.json` (shared undistorted
intrinsics plus per-camera extrinsics from calibration).

```bash
# any env with numpy/PIL (the trainer env works)
python scripts/build_4dgs_dataset.py \
    --dataset_root <dataset_root> \
    --out <dataset_root>_4dgs_full \
    --start 100 --end 187 --fps 24 \
    --downscale 1 --jobs 12
```

What it does: composites RGBA frames from images and masks, writes
`transforms_train.json` / `transforms_test.json` with per-frame `time`
and the global scaled intrinsics, and initializes `points3d.ply` with
per-point `time` via an iterative bbox-refined visual hull carve.

Production choices that matter:

- `--downscale 1` (full resolution). A first run at reduced resolution
  was visibly softer.
- `--test_cameras` left empty means **all cameras train**.
  `transforms_test.json` then duplicates the first camera purely so
  the trainer's eval loop has something to chew on. Its PSNR is a
  *training-view* monitor, not a held-out score.

## 2. Train

Config lives in the trainer repo under `configs/`. Key baseline
settings: `gaussian_dim: 4`, `rot_4d: True`, `time_duration` set to
your clip length, `num_pts: 300_000`, `batch_size: 4`,
`resolution: 1`, `dataloader: True`, 30k iterations,
`densify_until_iter: 20_000`, `densify_until_num_points: 1_200_000`.

```bash
# note the stdin redirect so a remote ssh channel can close
ssh <host> 'cd <trainer-repo> && \
  GS4D_T_INIT_DIV=100 nohup \
  <trainer-env-python> train.py \
  --config configs/<scene>.yaml \
  > <scene>_train.log 2>&1 < /dev/null &'
```

Roughly 4 hours at full resolution on a 4090 (1.7–2.4 it/s, slower
with aggressive densification). Output:
`output/<model_path>/chkpnt30000.pth` (`chkpnt_best.pth` is the same
file for these runs).

Operational footguns, all hit in practice:

- `pkill -f train.py` over ssh **kills your own ssh shell**, because
  the pattern matches the remote command line. Use
  `pkill -f '[t]rain.py'`, and run the kill and the relaunch as
  *separate* ssh invocations.
- A launch with `nohup ... &` over ssh, without `< /dev/null`, leaves
  the ssh channel held open by the child's inherited stdin.
- The trainer is CPU-bound out of the box (DataLoader workers doing
  float64 composites). The `data_utils.py` patch below fixes it.

### Trainer patches

Apply these to a clean clone of fudan-zvg/4d-gaussian-splatting. All
five are required for this recipe.

1. **`gaussian_renderer/__init__.py` — FoV sentinel fix (critical).**
   The Blender-style loader sets `FovX = FovY = -1` when `fl_x/fl_y/
   cx/cy` are present, and the renderer then computes `tan(-0.5)` for
   the EWA Jacobian. The result is inflated splat footprints: a blur
   that `bake_sogst --aniso_boost/--scale_boost` used to paper over.
   Patch it to use `tanfov = image_size / (2 * fl)` whenever
   `fl_x > 0`.
2. **`gaussian_renderer/diff_gaussian_rasterization.py` — cached CUDA
   extension.** A system CUDA newer than 12 cannot JIT-build the old
   C++14 extension. The patch importlib-loads a previously built
   `diff_gaussian_rasterization.so` from the torch extensions cache
   (the spec name must be exactly `diff_gaussian_rasterization`). Do
   not clear that cache directory.
3. **`utils/data_utils.py` — fast composite path.** A uint8 torch
   fused `rgb*a + bg*(1-a)` replaces the float64 numpy chain. Roughly
   2× training throughput: the GPU idled at 0–15% before.
4. **`scene/dataset_readers.py` — `fetchPly`** copies the `time`
   field contiguously (`np.ascontiguousarray(...).astype(np.float32)`).
   plyfile returns a strided view that torch rejects.
5. **`scene/gaussian_model.py` — `GS4D_T_INIT_DIV` env var.** Initial
   temporal sigma is `sqrt(duration / div)`. Upstream hardcodes
   `div = 5`, which on a short clip bakes many frames of motion smear
   into the initial sigma. `GS4D_T_INIT_DIV=100` gives a much tighter
   start. When unset, upstream behavior is preserved.

## 2b. SPM compression (recommended before export)

The OMG4 paper's sampling → pruning → merging pipeline cuts the
Gaussian count several-fold with staged fine-tuning (~12k iterations
from the 30k-iteration pretrain) at minimal quality cost. It is where
the reference project gets most of its size win. As a reference point,
coffee_martini (from the open DyNeRF dataset) went through it and kept
146k splats. It runs entirely upstream of the container: fewer splats
in, same formats out, no viewer changes. Fewer splats also cut decode
time, mobile render load, and the bandwidth needed for gap-free
first-pass streaming (see `meta.streams` gating).

Needs: a local OMG4 clone, the bake environment, the training dataset
(for fine-tuning), and a scene yaml with the SPM `OptimizationParams`
block. Copy an existing custom config and adjust `source_path` /
`model_path` / `time_duration`. Key knobs: `tau_GS` (sampling
pressure: the paper's headline 0.2 keeps ~20%, and subject scenes have
shipped with the gentler 0.8) and `tau_GP` (pruning quantile). Then:

```bash
python scripts/spm_compress.py \
    --omg4-repo <path-to-OMG4> \
    --config configs/custom/<scene>_omg4.yaml \
    --checkpoint output/<scene>_pretrain/chkpnt30000.pth \
    --fps <fps> --output <scene>.sogst
```

The wrapper runs compute_gradient.py (SD score) → train.py
(S→P→M→SVQ → `comp.xz`) → `bake_sogst.py` (bakes the MLP appearance
to explicit SH, emitting the interchange PLY) → `sogst_pack.py`
(`--shn-count` auto-scales with the post-SPM splat count, because the
fixed-size SH centroid table dominates small scenes). Stages resume:
artifacts already present are skipped.

The bake stage passes `--no_filter_corrupted --sh_clamp 3.0`. The
bad_color/garbage corruption filters were calibrated for the legacy
SVQ pipeline's catastrophic MLP extrapolation, and they delete ~40% of
*healthy* splats on an SPM fine-tune (out-of-range bare f_dc is normal
SH representation there). Early subject-capture exports shipped with
the filters on and read as transparent and washed out. `sh_clamp 1.5`
compounds it: it zeroes the higher SH bands of any splat whose bare DC
exceeds 1.5, which are exactly the splats that need those bands to
land in range from real view directions.

Score the result against held-out views with `scripts/eval_render.py`
(gsplat + PSNR/SSIM/LPIPS, validated against the OMG4 trainer's own
coffee_martini numbers). Honest eval on custom rigs requires
`build_4dgs_dataset.py --test_cameras`, because the default duplicates
a training camera into the test split.

**Report PSNR on the subject, never full-frame.** A single subject can
occupy a few percent of the frame, and the remaining background is
trivially perfect. Full-frame PSNR can sit 10 dB or more above the
same render scored on foreground pixels alone.

## 2c. SPM-native compression (preferred for subject captures)

`spm_compress.py` above runs OMG4's *whole* pipeline, which after the
merge rounds distills appearance into 6-dim latents + MLPs + SVQ. The
exporter then bakes that back to explicit SH. On subject captures that
round-trip, not the splat-count reduction, is the dominant quality
loss: in one measured case, two exports at very different splat counts
scored within 0.3 dB of each other while both sat ~1.4 dB under the
bake-only pretrain, with visible face softening that survived a 16×
larger appearance codebook.

`spm_native.py` keeps the count reduction and drops the round-trip. It
drives the same sampling → gradient-pruning → merging schedule via
`train.py --spm_native_out` (a patched OMG4 clone, see the
availability note below). At the point the stock trainer would call
`construct_net()`, it instead fine-tunes the surviving **explicit-SH**
Gaussians for `--extra-iter` iterations and saves a rotor-style
checkpoint. The export then takes `bake_sogst.py`'s checkpoint path,
the same bake the full-quality pretrains use, with no MLPs to
evaluate.

```bash
python scripts/spm_native.py \
    --omg4-repo <path-to-OMG4> \
    --config configs/custom/<scene>_spm_native.yaml \
    --checkpoint output/<scene>_pretrain/chkpnt30000.pth \
    --fps <fps> --output <scene>_spm_native.sogst
```

Stages resume like `spm_compress.py`, and the SD-score gradients are
interchangeable between the two. Copy `{view,t}_grad.npy` into the
other model dir to A/B the two paths without recomputing them.

> **Availability note.** The `--spm_native_out` flag and the SPM-native
> schedule live on a patched OMG4 branch that has not been published
> yet. Until it is, `spm_native.py` requires that local patch set.
> `spm_compress.py` works against a stock OMG4 clone.

## 3. Export + pack

```bash
# bake environment, scripts from this repo's scripts/ dir
python scripts/bake_sogst.py \
    --input <trainer-repo>/output/<scene>/chkpnt30000.pth \
    --output <scene>.sogst \
    --time_min 0 --time_max <clip-seconds> --fps <fps>
```

The bake writes the container directly. The streamed per-segment
layout is the default whenever segmentation is on, so there is no
separate repack step. To hand the per-splat data to an external
encoder instead, add `--emit_ply <scene>.ply` (with or without
`--output`) and pack it with
`python scripts/sogst_pack.py --input ... --verify`.

No `--aniso_boost`/`--scale_boost`: those flags existed only to
compensate for the FoV-sentinel bug (patch 1) and must stay off now.

## 4. Deploy to the viewer

The target player is the Solipsist Studios fork of the PlayCanvas
SuperSplat Viewer. Serve the `.sogst` file and point the viewer at it.

Set the embed config's `sogstRotationDeg` to match your training
world's orientation. The viewer applies no rotation by default. Assets
trained from this pipeline's Y-up world use `[0, 180, 0]`. The
viewer's auto-framing fits horizontal FoV only and crops tall content,
so give portrait-framed subjects an explicit camera pose.

## Known limits

- **Motion smear** comes from temporal averaging inside each
  gaussian's lifetime: motion is only linear within a lifetime, so any
  non-linear motion inside that window (a face during head motion) is
  averaged away. Shorter temporal sigmas reduce it, and they also make
  temporal segment culling more effective.
- **The direct temporal levers saturate.** Cutting the initial sigma
  and making temporal densification more aggressive moved the final
  sigma distribution only slightly in measured runs. The optimizer
  chooses long lifetimes when the training views do not constrain it
  to do better. Past that point, more views are the lever, not more
  temporal capacity.
- **At export**, consider clamping `t_sigma` to the clip duration
  (rare outliers many times the clip length inflate the codebook
  range) and tuning `--segment_duration` per asset.
