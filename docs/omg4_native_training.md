# Native OMG4 training (4D Gaussian Splatting → .sogst)

End-to-end recipe for training a 4D gaussian splat natively from a
multi-camera capture and exporting it as a streamable `.sogst` asset for
the GaussPlay viewer. This is the pipeline that produced the
`ariana_16_*` assets; substitute your own dataset paths. It replaces the
SOG-flipbook → `sog_flipbook_to_sogst.py` repack path, which drops splats
to fit a budget and was judged unacceptable quality-wise.

"OMG4" here is the *trainer* (MinShirley/OMG4) and its conda env, not a
file format: the container it feeds is `.sogst`, specified in
[sogst-format.md](sogst-format.md).

All GPU stages run on `epoch` (RTX 4090 24GB, 32 cores). Two conda envs
there:

- `4d-gaussian-splatting` — py3.7 / torch 1.12 / cu116. Runs the trainer.
- `OMG4` — py3.11 / torch 2.9.1. Runs the bake and pack (`bake_sogst.py`,
  `sogst_pack.py`).

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
   `bake_sogst --aniso_boost/--scale_boost` used to paper over).
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
    --fps <fps> --output /tmp/<scene>.sogst
```

The wrapper runs compute_gradient.py (SD score) → train.py (S→P→M→SVQ →
`comp.xz`) → `bake_sogst.py` (bakes the MLP appearance to explicit SH,
emitting the interchange PLY) → `sogst_pack.py` (`--shn-count` auto-scales
with the post-SPM splat
count — the fixed-size SH centroid table dominates small scenes).
Stages resume: artifacts already present are skipped.

The bake stage passes `--no_filter_corrupted --sh_clamp 3.0`. The
bad_color/garbage corruption filters were calibrated for the legacy SVQ
pipeline's catastrophic MLP extrapolation and delete ~40% of *healthy*
splats on an SPM fine-tune (out-of-range bare f_dc is normal SH
representation there) — the first tatum SPM exports shipped with them on
and read as transparent/washed-out. sh_clamp 1.5 compounds it by zeroing
the higher SH bands of any splat whose bare DC exceeds 1.5 — exactly the
splats that need those bands to land in range from real view directions.

Score the result against held-out views with `scripts/eval_render.py`
(gsplat + PSNR/SSIM/LPIPS; validated against the OMG4 trainer's own
coffee_martini numbers). Honest eval on custom rigs requires
`build_4dgs_dataset.py --test_cameras` — the default duplicates a
training camera into the test split.

## 2c. SPM-native compression (preferred for subject captures)

`spm_compress.py` above runs OMG4's *whole* pipeline, which after the
merge rounds distills appearance into 6-dim latents + MLPs + SVQ; the
exporter then bakes that back to explicit SH. On subject captures that
round-trip — not the splat-count reduction — is the dominant quality
loss: on tatum, 27.6k and 97k splats scored within 0.3 dB of each other
while both sat ~1.4 dB under the bake-only pretrain, with visible face
softening that survived a 16× larger appearance codebook.

`spm_native.py` keeps the count reduction and drops the round-trip. It
drives the same sampling → gradient-pruning → merging schedule via
`train.py --spm_native_out` (OMG4 clone, `feature/ftgs-degree2` branch),
but at the point the stock trainer would call `construct_net()` it
instead fine-tunes the surviving **explicit-SH** Gaussians for
`--extra-iter` iterations and saves a rotor-style checkpoint. The export
then takes `bake_sogst.py`'s checkpoint path — the same bake the
full-quality pretrains use, with no MLPs to evaluate.

```bash
python scripts/spm_native.py \
    --omg4-repo ~/Dev/github/OMG4 \
    --config configs/custom/<scene>_spm_native.yaml \
    --checkpoint output/<scene>_pretrain/chkpnt30000.pth \
    --fps <fps> --output /tmp/<scene>_spm_native.sogst
```

Stages resume like `spm_compress.py`, and the SD-score gradients are
interchangeable between the two — copy `{view,t}_grad.npy` into the
other model dir to A/B the two paths without recomputing them.

## 3. Export + pack

```bash
# epoch, OMG4 env, scripts from this repo's scripts/ dir
python scripts/bake_sogst.py \
    --input ~/Dev/github/4d-gaussian-splatting/output/ariana_16_full/chkpnt30000.pth \
    --output /tmp/ariana_16_full.sogst \
    --time_min 0 --time_max 3.625 --fps 24
```

The bake writes the container directly; the streamed per-segment layout
is the default whenever segmentation is on, so there is no separate
repack step. To hand the per-splat data to an external encoder instead,
add `--emit_ply /tmp/ariana_16_full.ply` (with or without `--output`)
and pack it with `python scripts/sogst_pack.py --input ... --verify`.

No `--aniso_boost`/`--scale_boost` — those flags existed only to
compensate for the FoV-sentinel bug (patch 1) and must stay off now.

## 4. Deploy to GaussPlay

Copy the `.sogst` file into `gaussplay/public/splats/`
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
| `ariana_16_temporal` | as full + `densify_grad_t_threshold 0.0002/200` + `GS4D_T_INIT_DIV=100` | ~5h wall (paused mid-run), eval PSNR 50.3, 547k splats. σ_t p50 0.117 s / p95 0.387 s (vs 0.13 / 0.528); active splats per instant ~21% (vs ~18%). v2 stream 133.5 MB; v3 13.8 MB (9.7×, 89.7% drawn). Modest σ_t shrink — the loss still tolerates ~3-frame lifetimes; see levers below |

## Measured: where the quality gap actually comes from

Run 2026-07-30 on unit-02 (local 5090, `omg4` env, gsplat 1.5.3) with
`scripts/…/render_omg4.py` + `render_ply.py` (scratch copies; promote
them here if this gets repeated). Method: render **both** models at
`Camera_0015` — a view *both* were TRAINED on — and score each against
the identical ground-truth image. The two models live in different world
frames, so each is rendered with its own camera (4DGS
`transforms_train.json`; PostShot's COLMAP `sparse/0`), which is fine
because the reference image is shared. Intrinsics are identical in both
(PINHOLE 3840×3360, fl 2120.92).

Pipeline validated: rendering the OMG4 export at the trainer's own eval
camera over all 88 frames gives mean 49.2 dB vs the trainer's reported
50.3 dB — the ~1 dB is the export's SH bake/prune plus a different
rasterizer, so the harness is faithful.

**Report PSNR on the subject, never full-frame.** Ariana occupies only
**2.8–3.2%** of the frame; the remaining 97% is black background that is
trivially perfect. Full-frame PSNR ≈ 45–48 dB while the same render
scores ≈ 31–34 dB on foreground pixels. The headline training number
(50.3 dB) is ~15 dB of empty pixels.

| frame | GT motion | flipbook head | OMG4 4D head | gap |
| --- | --- | --- | --- | --- |
| 0185 | 0.09 (stillest) | 35.12 dB | 33.62 dB | **1.50 dB** |
| 0133 | 1.78 (fastest) | 34.97 dB | 31.65 dB | **3.32 dB** |
| 0143 | ~1.5 | 34.97 dB | 31.03 dB | 3.94 dB |

Across all 87 frame pairs at `Camera_0015` (19× motion range),
correlation between GT inter-frame motion and OMG4 head PSNR is
**−0.82**: 33.9 dB on the low-motion quartile vs 31.6 dB on the
high-motion quartile. The flipbook is essentially **motion-invariant**
(35.12 → 34.97, i.e. 0.15 dB from stillest to fastest frame).

So the ~3.3 dB deficit at speed decomposes roughly as:

- **~1.5 dB baseline** representation deficit, present even when the
  subject is nearly still — one 4D model at 547k gaussians simply
  carries less detail than a dedicated 141k-gaussian fit per frame.
- **~1.8 dB motion-dependent** loss on top — the temporal averaging
  inside each gaussian's ~3-frame lifetime.

Both arms saw the same 16 cameras, so **view count explains none of
this**. Fixing the smear means attacking the representation: more
capacity per instant, shorter lifetimes, or richer intra-lifetime motion
than a single linear velocity.

## Known limits / next levers

- **View sparsity is NOT sufficient to explain the gap** (corrected
  2026-07-30). The crisp SOG flipbook everyone compares against was
  trained by **PostShot v1.0.110, per-frame, on these exact same 16
  cameras at the same 3840×3360** — see "Flipbook provenance" below. 16
  views therefore demonstrably suffice for crisp novel-view rendering
  of this subject; what differs is the representation, not the capture.
  Adding rendered camera rings may still help (the rig is two 8-cam
  rings at ±45° azimuth, and train ~51 dB vs ~36 dB held-out in the
  first run does show rig overfitting) but it is no longer the obvious
  first lever.
- **Capacity per instant is comparable; capacity per *frame* is not.**
  The flipbook spends ~136–144k gaussians on each single frame,
  independently fitted — ~12.3M across the 88-frame clip. The 4D model
  has 547k total, ~21% active at any instant (~115k), so the per-instant
  budget is similar, but each of those gaussians must serve ~3 frames
  with only a linear velocity and a Gaussian temporal falloff. Any
  non-linear motion inside that window (face during head motion) is
  averaged away. That averaging — not splat count, not view count — is
  the mechanism behind the residual smear.
- **Motion smear** = σ_t too long relative to motion (σ_t ≈ 3 frames,
  and motion is only linear within each gaussian's lifetime). Shorter
  sigmas would also make v3's temporal segment culling effective (only
  ~10% culled at these lifetimes).
  **The direct temporal levers are mostly spent.** `ariana_16_temporal`
  cut the initial σ_t 4.5× (0.85 → 0.19 s) *and* made temporal
  densification 2.5× more aggressive, yet final σ_t p50 moved only
  0.130 → 0.117 s and the splat count only 525k → 547k (the 1.2M cap
  was never approached). Read that as: the optimizer is *choosing* long
  lifetimes because 16 views don't constrain it to do better — the
  training loss is already ~50 dB, so there is no error signal left to
  pay for finer temporal detail. Adding views, not temporal capacity,
  is the lever that should move next.
- **v3 export**: consider clamping σ_t to clip duration at export
  (rare outliers up to 4 s inflate the codebook range) and tuning
  `--segment_duration` per asset.

## Flipbook provenance (the crisp `public/splats/ariana/*.sog`)

Traced 2026-07-30, because the OMG4 results are always judged against
this asset and it matters what it actually is.

- **Trainer: PostShot v1.0.110** (`comment Postshot v1.0.110` in every
  PLY header), **not** Brush, and not any 4D method.
- **Per-frame, independently trained** — 88 separate static
  reconstructions, one per frame 0100–0187. Splat counts drift frame to
  frame (0100: 136,354 · 0120: 138,559 · 0150: 142,850 · 0187: 144,395),
  which is the signature of independent fits rather than one model
  sampled over time.
- **16 cameras, 3840×3360** — inputs at
  `epoch:/media/ai/output/results/ariana_16/data/Frame0100…0187/images/`
  are `Camera_0001…0016.png`, the same Blender renders (rendered
  2026-02-26) and same resolution later used for 4DGS training.
  **It was not trained on the 48-camera renders.** That 48-cam set is
  `/media/ai/datasets/ariana_4k/` (`undistorted_Camera_0001…0048`) and
  only ever covered frame 0100 in `images/` plus frames 0100–0118 in
  `images_frames/` — static single-frame experiments.
- **Chain**: PLYs written 2026-03-01 21:03 →
  `splats_compressed/*.sog` via splat-transform v0.16.1 on 2026-03-02
  18:44 → committed to gaussplay 21:30 the same day
  (`8a45028 Add Ariana synthetic splat`). The deployed
  `public/splats/ariana/0100.sog` is md5-identical to the epoch source.
- The working-tree "modification" of all 88 `.sog` files in gaussplay is
  only the git-lfs migration (`.gitattributes` now routes `*.sog`
  through the LFS clean filter); contents are byte-identical to the
  March commit.
