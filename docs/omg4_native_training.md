# Native OMG4 training (4D Gaussian Splatting → .omg4)

End-to-end recipe for training a 4D gaussian splat natively from a
multi-camera capture and exporting it as a streamable OMG4 (v2) and
SOG-compressed OMG4 v3 asset for the GaussPlay viewer. This is the
pipeline that produced the `ariana_16_*` assets; substitute your own
dataset paths. It replaces the SOG-flipbook → `sog_flipbook_to_omg4.py`
repack path, which drops splats to fit a budget and was judged
unacceptable quality-wise.

All GPU stages run on `epoch` (RTX 4090 24GB, 32 cores). Two conda envs
there:

- `4d-gaussian-splatting` — py3.7 / torch 1.12 / cu116. Runs the trainer.
- `OMG4` — py3.11 / torch 2.9.1. Runs export/repack (`xz_to_omg4.py`,
  `omg4_repack.py`, `sog_pack.py`).

Trainer repo: `epoch:~/Dev/github/4d-gaussian-splatting`
(fudan-zvg/4d-gaussian-splatting, "rotor" 4DGS) **with local
uncommitted patches — see "Trainer patches" below; they are required.**

## 1. Build the D-NeRF-style dataset

Input layout (e.g. `epoch:/media/ai/datasets/ariana_16`): per-camera
`images/Camera_XXXX/NNNN.png` frames, `fmasks/` foreground masks, and
`transforms_undistorted_intrinsics.json` (shared undistorted intrinsics
+ per-camera extrinsics from calibration).

```bash
# epoch, any env with numpy/PIL (4d-gaussian-splatting env works)
cd ~/Dev/github/volumetric-capture-pipeline
python scripts/build_4dgs_dataset.py \
    --dataset_root /media/ai/datasets/ariana_16 \
    --out /media/ai/datasets/ariana_16_4dgs_full \
    --start 100 --end 187 --fps 24 \
    --downscale 1 --jobs 12
```

What it does: composites RGBA frames from images+masks, writes
`transforms_train.json`/`transforms_test.json` with per-frame `time`
and the global scaled intrinsics, and initializes `points3d.ply` with
per-point `time` via an iterative bbox-refined visual hull carve.

Production choices that matter:

- `--downscale 1` (full 3840×3360). The first run at reduced res was
  visibly softer.
- `--test_cameras` left empty → **all 16 cameras train**;
  `transforms_test.json` duplicates the first camera purely so the
  trainer's eval loop has something to chew on (its PSNR is a
  *training-view* monitor, not a held-out score).

## 2. Train

Config lives in the trainer repo: `configs/ariana_16_full.yaml`
(baseline production) and `configs/ariana_16_temporal.yaml`
(anti-smear; see run history). Key baseline settings: `gaussian_dim: 4`,
`rot_4d: True`, `time_duration: [0.0, 3.625]`, `num_pts: 300_000`,
`batch_size: 4`, `resolution: 1`, `dataloader: True`, 30k iterations,
`densify_until_iter: 20_000`, `densify_until_num_points: 1_200_000`.

```bash
# epoch — note stdin redirect so the ssh channel can close
ssh epoch 'cd ~/Dev/github/4d-gaussian-splatting && \
  GS4D_T_INIT_DIV=100 nohup \
  ~/miniconda3/envs/4d-gaussian-splatting/bin/python train.py \
  --config configs/ariana_16_temporal.yaml \
  > ariana_16_temporal_train.log 2>&1 < /dev/null &'
```

~4h at full res on the 4090 (1.7–2.4 it/s; slower with aggressive
densification). Output: `output/<model_path>/chkpnt30000.pth`
(`chkpnt_best.pth` is the same file for these runs).

Operational footguns (all hit in practice):

- `pkill -f train.py` over ssh **kills your own ssh shell** (the
  pattern matches the remote command line). Use `pkill -f '[t]rain.py'`
  and run kill and relaunch as *separate* ssh invocations.
- Launching with `nohup ... &` over ssh without `< /dev/null` leaves
  the ssh channel held open by the child's inherited stdin.
- The trainer is CPU-bound out of the box (DataLoader workers doing
  float64 composites) — the `data_utils.py` patch below fixes it.

### Trainer patches (uncommitted, in `epoch:~/Dev/github/4d-gaussian-splatting`)

1. **`gaussian_renderer/__init__.py` — FoV sentinel fix (critical).**
   The Blender-style loader sets `FovX = FovY = -1` when `fl_x/fl_y/
   cx/cy` are present, and the renderer then computed `tan(-0.5)` for
   the EWA Jacobian → inflated splat footprints (the blur that
   `xz_to_omg4 --aniso_boost/--scale_boost` used to paper over).
   Patched to use `tanfov = image_size / (2 * fl)` whenever `fl_x > 0`.
2. **`gaussian_renderer/diff_gaussian_rasterization.py` — cached CUDA
   extension.** System CUDA is 13.2, which can't JIT-build the old
   C++14 extension; the patch importlib-loads the cached
   `~/.cache/torch_extensions/py37_cu116/diff_gaussian_rasterization/
   diff_gaussian_rasterization.so` (spec name must be exactly
   `diff_gaussian_rasterization`). Don't clear that cache dir.
3. **`utils/data_utils.py` — fast composite path.** uint8 torch fused
   `rgb*a + bg*(1-a)` instead of the float64 numpy chain; ~2× training
   throughput (GPU was idling at 0–15% before).
4. **`scene/dataset_readers.py` — `fetchPly`** copies the `time` field
   contiguously (`np.ascontiguousarray(...).astype(np.float32)`);
   plyfile returns a strided view that torch rejects.
5. **`scene/gaussian_model.py` — `GS4D_T_INIT_DIV` env var.** Initial
   temporal sigma is `sqrt(duration / div)`; upstream hardcodes
   div = 5 (σ_t ≈ 0.85 s on a 3.6 s clip — ~20 frames of baked-in
   motion smear). `GS4D_T_INIT_DIV=100` → σ_t ≈ 0.19 s. Default (unset)
   preserves upstream behavior.

## 2b. SPM compression (recommended before export)

The OMG4 paper's sampling → pruning → merging pipeline cuts the Gaussian
count several-fold with staged fine-tuning (~12k iterations from the
30k-iteration pretrain) at minimal quality cost — it is where the
reference project gets most of its size win, and it was historically
skipped for the subject captures (`tatum_jump` shipped all 1.2M splats;
coffee_martini went through it and kept 146k). It runs entirely upstream
of the container: fewer splats in, same formats out, no viewer changes.
Fewer splats also cut decode time, mobile render load, and the bandwidth
needed for gap-free first-pass streaming (see `meta.streams` gating).

Needs: the local `~/Dev/github/OMG4` clone, the `omg4` conda env, the
training dataset (for fine-tuning), and a scene yaml with the SPM
`OptimizationParams` block — copy `configs/custom/perframe90_omg4.yaml`
and adjust `source_path` / `model_path` / `time_duration`. Key knobs:
`tau_GS` (sampling pressure; paper headline 0.2 = keep ~20%, subject
scenes have shipped with the gentler 0.8) and `tau_GP` (pruning
quantile). Then:

```bash
python scripts/spm_compress.py \
    --omg4-repo ~/Dev/github/OMG4 \
    --config configs/custom/<scene>_omg4.yaml \
    --checkpoint output/<scene>_pretrain/chkpnt30000.pth \
    --fps <fps> --output-v3 /tmp/<scene>_v3.omg4
```

The wrapper runs compute_gradient.py (SD score) → train.py (S→P→M→SVQ →
`comp.xz`) → `xz_to_omg4.py` (bakes the MLP appearance to explicit SH) →
`sog_pack.py` (v3; `--shn-count` auto-scales with the post-SPM splat
count — the fixed-size SH centroid table dominates small scenes).
Stages resume: artifacts already present are skipped.

Score the result against held-out views with `scripts/eval_render.py`
(gsplat + PSNR/SSIM/LPIPS; validated against the OMG4 trainer's own
coffee_martini numbers). Honest eval on custom rigs requires
`build_4dgs_dataset.py --test_cameras` — the default duplicates a
training camera into the test split.

## 3. Export + pack

```bash
# epoch, OMG4 env, scripts from this repo's scripts/ dir
python scripts/xz_to_omg4.py \
    --input ~/Dev/github/4d-gaussian-splatting/output/ariana_16_full/chkpnt30000.pth \
    --output /tmp/ariana_16_full.omg4 \
    --time_min 0 --time_max 3.625 --fps 24
# streamable v2 (tiled):
python scripts/omg4_repack.py --input /tmp/ariana_16_full.omg4 \
    --output /tmp/ariana_16_full_stream.omg4
# SOG-compressed v3:
python scripts/sog_pack.py --input /tmp/ariana_16_full.omg4 \
    --output /tmp/ariana_16_full_v3.omg4 --verify
```

No `--aniso_boost`/`--scale_boost` — those flags existed only to
compensate for the FoV-sentinel bug (patch 1) and must stay off now.

## 4. Deploy to GaussPlay

Copy the `_stream.omg4` / `_v3.omg4` into `gaussplay/public/splats/`
(gitignored — never commit them) and add a `viewer: "supersplat"`
gallery entry in `src/components/Gallery.tsx`. For assets trained from
this pipeline's Y-up world: `omg4Rotation: [0, 180, 0]` (the viewer's
default `[270,0,0]` is for COLMAP-world files), `position` to put feet
on the AR floor, and an explicit `cameraPose` (the viewer's auto-framing
fits horizontal FoV only and crops tall content).

## Run history (ariana_16, 88 frames @ 24 fps = 3.625 s)

| run | dataset / config | result |
| --- | --- | --- |
| `ariana_16` | downscaled, 2 held-out cams | Decent but ghosting from 3/4 low front-right (view extrapolation) + soft |
| `ariana_16_full` | full res, all 16 cams, `densify_grad_t_threshold 0.0002/80` | 4h10m, train PSNR 51.1, 525k splats (cap not hit), median σ_t 0.13 s. v2 stream 128 MB; v3 14 MB (9.6×, 86.7% persistent). Verdict: v3 much better, but off-angle motion smear remains (face during motion) — temporal under-capacity |
| `ariana_16_temporal` | as full + `densify_grad_t_threshold 0.0002/200` + `GS4D_T_INIT_DIV=100` | in progress 2026-07-30 |

## Known limits / next levers

- **View sparsity is the ghosting root cause** — 16 cams in two 8-cam
  rings (±45° azimuth spacing); intrinsics/extrinsics verified good
  (LOO visual-hull consistency 99.4% mean). Train PSNR ~51 vs ~36 on
  held-out views = overfitting to the rig. Best fix: render additional
  camera rings (the source is a synthetic Blender scene, so extra views
  are free).
- **Motion smear** = σ_t too long relative to motion (σ_t 0.13 s ≈ 3
  frames, linear motion within each gaussian's lifetime). Attacked by
  the `ariana_16_temporal` run; shorter sigmas also make v3's temporal
  segment culling effective (only ~12% culled at σ_t 0.13 s).
- **v3 export**: consider clamping σ_t to clip duration at export
  (rare outliers up to 4 s inflate the codebook range) and tuning
  `--segment_duration` per asset.
