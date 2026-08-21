<!--
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
-->

# Volumetric capture pipeline

End-to-end flow from raw multi-camera GoPro footage to a trained 4D
gaussian splat, baked as a streamable `.sogst` asset. Calibration (sync,
poses, masks) happens at one target timestamp. Training then consumes a
frame window extracted around it. All example paths below use a
`take01_1500ms` example dataset (calibrated at 1.5s into the clip) --
substitute your own working directory.

Follow the sections below in order: that is the pipeline order. Each
script's own `--help` / docstring has the full flag reference. This doc
is the narrative walkthrough, and it names the conda envs each stage
needs.

Only the direct branch (training on the real cameras) is part of this
build. The Diffuman4D 48-camera dense-ring branch is a planned
addition (hallucinating extra ring views via Diffuman4D, then training
on real + synthetic views together) but it is not wired into any script
or this walkthrough yet. It will get its own doc sections once it has
been run and validated end-to-end.

## Prerequisites

- Per-camera calibration PKLs (from `deps/camera-calibration`), one per
  physical camera.
- Raw GoPro `.mp4` clips, one per camera, all recording the same take.
- The `cumuli` conda env (`bash scripts/setup_cumuli_env.sh`, see
  `docs/environment.md`). One env serves every stage. Keypoint
  prediction additionally needs `SAPIENS_CHECKPOINT_ROOT` set. Training
  needs a CUDA GPU. Every stage before it runs on CPU.

## Recommended: run everything with the unified orchestrator

`scripts/run_unified_pipeline.py` runs the full pipeline end-to-end in
one process: sync through mask cleanup and training, plus pose
refinement when a known-good sync is reused. It dispatches each stage
into the right conda env itself, so you do not need to `conda activate`
between steps the way the walkthrough below shows. Read the walkthrough
below first to understand what each stage actually does and why. Then
use the orchestrator to run them without the manual bookkeeping.

```bash
python3 scripts/run_unified_pipeline.py \
    --config configs/my_rig.json \
    --video_dir /path/to/movies \
    --calib_dir /path/to/calibration_pkls \
    --out_dir ~/pipeline_run \
    --target_time 2500ms \
    --start_from_stage production \
    --sapiens_checkpoint_root ~/sapiens/2
```

Key flags:

- **`--config`** -- JSON file of per-rig defaults (
  `--trainer_repo`, `SAPIENS_CHECKPOINT_ROOT`, HLOC settings) so
  you do not have to repeat them on every run. Explicit CLI flags always
  override the config. See `configs/README.md`.
- **`--start_from_stage` / `--stop_after_stage`** (`sync`, `production`,
  `poses`, `masks`, `dataset4d`, `train4d`): resume partway through, or
  stop early to inspect intermediate output before committing GPU time to
  training. On a machine with no CUDA GPU, run with
  `--stop_after_stage dataset4d` and train elsewhere: every stage through
  the dataset build is CPU-capable, and `train4d` refuses to start
  without a GPU rather than fail deep inside training.
  Resuming assumes the earlier stages' outputs already exist under
  `--out_dir`.
- **Sync: verify once, reuse, and do not re-trust automatically every run.**
  `compute_sync_offsets.py`'s sync method (envelope cross-correlation,
  always runs unless `--initial_sync_json` is given) had a sign-inversion
  bug and a confidence metric that could not tell a correct lock from a
  lucky one on periodic/rhythmic audio. Both are fixed and validated against
  a hand-verified sync file, and periodic-audio ambiguity is now flagged
  via `peak_ratio` rather than passing silently. Nothing in the pipeline
  auto-gates on that flag yet though (`validate_stage_output.py` only
  checks that every camera is present, not its confidence), so still
  treat the live result as a draft to visually confirm rather than
  something the pipeline itself will catch if wrong. The workflow that
  has worked every time in practice:
  1. Run `compute_sync_offsets.py` once, inspect `make_sync_grid.py`'s
     sync grid by eye, hand-tune any camera's `frame_offset` that
     is visibly off, and save the result (for example
     `sync_offsets_v5.json`).
  2. Pass that verified file back in via `--initial_sync_json` on the
     *next* run to seed it without recomputing.
  3. For every run after that on the same take (different timestamps,
     retries, etc.), skip `compute_sync_offsets.py` entirely: write the
     verified file's path into `<out_dir>/resolved_sync_json.txt`
     yourself and pass `--start_from_stage production`. This is the
     normal way to iterate without re-running (and re-trusting) sync
     search each time.
- **`--hloc_resize_max`** (default 4096) / **`--hloc_max_keypoints`**
  (default 8192): HLOC feature-extraction settings, passed through to
  `multiframe_sfm.py`. The 4096 default is deliberately higher than that
  script's own default of 2048: on an 11-camera rig with ~5312px native
  media, 2048 discarded enough detail to measurably hurt camera pose
  accuracy (one real run: post-refinement median reprojection error went
  from ~30px to ~5px after raising this, with the improvement concentrated
  in the previously-worst cameras).

## Sync and extract a frame

Every manual step below runs in the one `cumuli` env. Activate it once:

```bash
conda activate cumuli
```

```bash
python3 scripts/compute_sync_offsets.py \
    ~/captures/take01/movies \
    ~/take01_undist \
    0001.mp4
# -> ~/take01_undist/sync_offsets.json

# Hand-tune a few cameras' frame_offset by +/- a couple frames after
# visually checking make_sync_grid.py output at a few timestamps, and
# save as sync_offsets_v2.json, v3.json, etc. Point everything below at
# whichever version is your current best.

python3 scripts/extract_synced_frames.py \
    ~/captures/take01/movies \
    ~/take01_undist/sync_offsets_v5.json \
    ~/take01_1500ms/raw \
    1.5

python3 scripts/make_sync_grid.py ~/take01_1500ms/raw ~/take01_1500ms/sync_grid.jpg
# eyeball the grid -- every camera should show the same instant of action
```

**Color correction** (optional, needs a per-camera RawTherapee `.pp3`
profile directory, for example thumbnail sidecar files from an NLE): pass
`--pp3_dir` to `extract_synced_frames.py` and it color-corrects each
extracted frame before undistortion. Matters because uncorrected
per-GoPro exposure/saturation differences show up as color noise in the
trained splat.

## Undistort

```bash
python3 scripts/undistort_frames.py \
    --frames_dir ~/take01_1500ms/raw \
    --calib_dir /path/to/calibration_pkls \
    --out_dir ~/take01_1500ms/undistorted \
    --out_pkl_dir ~/take01_1500ms/undistorted_pkls \
    --model OPENCV_FISHEYE
```

**Calibration sanity check** happens automatically here. If a camera's
calibration `image_size` disagrees with the actual frame resolution by a
uniform scale, it is auto-corrected with a loud warning (this was a real
bug: a calibration made at the GoPro 5.3K full-sensor width, 5568px,
silently applied to 5312px-wide video warped every camera by ~113px of
focal). Read those warnings if they appear. They mean you are pointing
at the wrong calibration source. `--target_pkl_dir` also enables
single-warp mode: native fisheye straight to a known target pinhole
geometry in one resample, instead of downscale-then-undistort.

## HLOC pose estimation

```bash
python3 scripts/run_hloc.py \
    --undistorted_dir ~/take01_1500ms/undistorted \
    --undistorted_pkl_dir ~/take01_1500ms/undistorted_pkls \
    --outputs_dir ~/take01_1500ms/solipsist_out
# -> ~/take01_1500ms/solipsist_out/transforms_multiframe.json (real camera poses)
```

## Keypoint pose refinement

Background SfM features (HLOC, above) live on distant walls/windows, so
camera poses can fit the background well while still disagreeing by
dozens of pixels at the capture volume center, exactly where the
subject is. This step bundle-adjusts the camera poses against 2D human
keypoints instead ("human as calibration wand"), and has been reliable:
real runs have seen it take subject-space median reprojection error from
~30px to ~5px.

The optimization is meaningfully more constrained with keypoints from
several time instants of the same static rig instead of one (the subject
sweeping through the capture volume over a few seconds anchors the
cameras far more strongly than a single pose), so this is the validated
approach: extract a short candidate window per camera (separate from,
and in addition to, the single production frame extracted above),
predict keypoints on each instant, and refine against all of them at
once (10+ instants recommended, 5 shown here for brevity):

```bash
python3 scripts/extract_synced_frames.py \
    ~/captures/take01/movies \
    ~/take01_undist/sync_offsets_v5.json \
    ~/take01_1500ms/sync_candidates \
    1.5 \
    --window 5

# Run undistort_frames.py + generate_masks.py + predict_keypoints_2d.py +
# split_keypoints_per_camera.py on each instant subdir f0/..f4/
for k in 0 1 2 3 4; do
    python3 scripts/undistort_frames.py \
        --frames_dir ~/take01_1500ms/sync_candidates/f$k \
        --calib_dir /path/to/calibration_pkls \
        --out_dir ~/take01_1500ms/sync_candidates_undist/f$k \
        --out_pkl_dir ~/take01_1500ms/sync_candidates_pkls/f$k
    python3 scripts/generate_masks.py \
        --images_dir ~/take01_1500ms/sync_candidates_undist/f$k \
        --out_fmasks_dir ~/take01_1500ms/sync_candidates_fmasks/f$k
    python3 scripts/predict_keypoints_2d.py \
        --images_dir ~/take01_1500ms/sync_candidates_undist/f$k \
        --out_kp2d_dir ~/take01_1500ms/sync_candidates_kp2d/f$k \
        --fmasks_dir ~/take01_1500ms/sync_candidates_fmasks/f$k
    python3 scripts/split_keypoints_per_camera.py \
        --kp2d_flat_dir ~/take01_1500ms/sync_candidates_kp2d/f$k \
        --out_dir ~/take01_1500ms/sync_candidates_poses2d/f$k
done

python3 scripts/run_pose_refinement.py \
    --transforms ~/take01_1500ms/solipsist_out/transforms_multiframe.json \
    --kp2d_dirs ~/take01_1500ms/sync_candidates_poses2d/f0,~/take01_1500ms/sync_candidates_poses2d/f1,~/take01_1500ms/sync_candidates_poses2d/f2,~/take01_1500ms/sync_candidates_poses2d/f3,~/take01_1500ms/sync_candidates_poses2d/f4 \
    --out_transforms ~/take01_1500ms/transforms_refined.json \
    --report_only
# check the printed median px error, then re-run without --report_only
```

Use `transforms_refined.json` in place of `transforms_multiframe.json`
from here on. For a single instant instead (lighter-weight, less
constrained), skip the candidate-window loop above and call the
underlying script directly with this frame's own `poses_2d`. Note the
singular `--kp2d`, not `--kp2d_dirs`:

```bash
python3 scripts/refine_poses_with_keypoints.py \
    --transforms ~/take01_1500ms/solipsist_out/transforms_multiframe.json \
    --kp2d ~/take01_1500ms/poses_2d \
    --out_transforms ~/take01_1500ms/transforms_refined.json \
    --report_only
```

## Build the flat, 2-digit-labeled dataset

Downstream tooling looks up cameras by the literal `camera_label` string
in transforms.json, and assumes plain zero-padded 2-digit labels ("00",
"01", ...). This step is where HLOC's `Camera_undistorted_0001`-style
labels get converted to that convention. Every later stage inherits it.

```bash
python3 scripts/build_flat_dataset.py \
    --transforms ~/take01_1500ms/transforms_refined.json \
    --undistorted_dir ~/take01_1500ms/undistorted \
    --out_images_flat ~/take01_1500ms/images_flat \
    --out_transforms ~/take01_1500ms/transforms.json
```

## Masks and 2D keypoints for this frame

```bash
python3 scripts/generate_masks.py \
    --images_dir ~/take01_1500ms/images_flat \
    --out_fmasks_dir ~/take01_1500ms/fmasks_flat

python3 scripts/predict_keypoints_2d.py \
    --images_dir ~/take01_1500ms/images_flat \
    --out_kp2d_dir ~/take01_1500ms/poses_2d_flat \
    --fmasks_dir ~/take01_1500ms/fmasks_flat

python3 scripts/split_keypoints_per_camera.py \
    --kp2d_flat_dir ~/take01_1500ms/poses_2d_flat \
    --out_dir ~/take01_1500ms/poses_2d
```

**Mask quality**: clean masks before you trust them for anything. Raw
BiRefNet output can include bystanders/props and can also drop part of
the subject:

```bash
python3 scripts/clean_masks.py \
    --fmasks_dir ~/take01_1500ms/fmasks_flat \
    --kp2d_dir ~/take01_1500ms/poses_2d \
    --out_dir ~/take01_1500ms/fmasks_clean \
    --images_dir ~/take01_1500ms/images_flat \
    --retry
```

Score any evaluation against the CLEANED masks, not the raw ones. Raw
masks with bystanders make a good model's eval metrics look broken.
Masking/alignment quality on real captures is still an open problem
being actively worked (see project notes). Treat this as the current
best approach, not a solved one.

## Triangulate the subject point cloud

`transforms.json` here already carries the refined poses (pose
refinement ran before flattening, above), so nothing further needs to
reference `transforms_refined.json` from this point on.

```bash
python3 scripts/triangulate_and_project_keypoints.py \
    --camera_path ~/take01_1500ms/transforms.json \
    --kp2d_dir ~/take01_1500ms/poses_2d \
    --out_kp3d_dir ~/take01_1500ms/poses_3d \
    --out_pcd_dir ~/take01_1500ms/poses_pcd_fullres
```

## Build the 4D dataset and train (dataset4d / train4d)

The last two stages run through the orchestrator rather than one script
each: `dataset4d` loops the per-instant chain over the whole training
window, and `train4d` drives the vendored trainer. Resume from the
calibrated run directory:

```bash
python3 scripts/run_unified_pipeline.py \
    --video_dir <capture>/movies \
    --calib_dir /path/to/calibration_pkls \
    --out_dir ~/take01_1500ms \
    --target_time 1500ms \
    --start_from_stage dataset4d \
    --train_window 48 --train_fps 30 \
    --eval_camera 05 --holdout_cameras 06
```

What the flags mean:

- **`--train_window` / `--train_fps`** -- the training clip is
  `(window - 1) / fps` seconds of frames extracted at the verified sync.
  Each frame gets the full undistort/mask/keypoint/clean chain, so the
  dataset build is CPU-heavy but CPU-only.
- **`--eval_camera` / `--holdout_cameras`** -- the eval camera is held
  out and scored. Its stereo mates go in `--holdout_cameras`: they leave
  training WITHOUT being scored. Rigs built from near-duplicate camera
  pairs leak: holding out one camera whose mate keeps training measured
  ~5.7 dB of pure leakage on an 11-camera rig. That number is leak, not
  quality. Check your rig's pair structure before choosing the split.
- **`train4d` is CUDA-gated.** It generates the trainer config from
  `configs/gs4d_pretrain_template.yaml` (override with
  `--trainer_config`), trains `train_scratch.py` in `deps/OMG4` under
  the `omg4` env, bakes with `bake_sogst.py`, and scores the held-out
  camera with `eval_render.py`.

The masks are baked into the dataset's RGBA alpha during the dataset
build, so alpha supervision is part of training by construction. The
bake also applies a lifetime mask-consistency filter: splats that
project outside the subject mask in most views across their active life
are dropped as silhouette-escaping junk.

Outputs land in the run directory: `dataset_4dgs/` (the D-NeRF training
dataset), `splat_4d.sogst` (the baked asset), and `eval_4d.json` (the
held-out PSNR/SSIM/LPIPS report). Treat absolute PSNR on masked
captures as a relative gate only: most of the frame is background that
renders perfectly, so the number is dominated by it.

## Post-process: mask-consistency splat filtering

Masked training does not prevent floaters. Trained splats routinely
carry non-subject junk that alpha supervision never removed (measured
at 30-56% of Gaussians on warm-started per-frame sequences, and a few
percent even on clean single-frame runs). The junk occludes the
subject from novel views, and it corrupts anything computed *from* the
splat, for example a subject centroid used to aim novel-view render
cameras.

`filter_splat_by_masks.py` removes it with a direct geometric test: it
projects every Gaussian into every camera, and it drops the Gaussian
if it lands outside the subject mask in at least half the cameras
whose frustum resolves it. Score it against the cleaned masks (same
rule as eval: never the raw BiRefNet output):

```bash
python3 scripts/filter_splat_by_masks.py \
    --splat_ply <run>/splat_30000.ply \
    --transforms <run>/transforms.json \
    --masks_dir <run>/fmasks_clean \
    --out_ply <run>/splat_30000_maskfilt.ply
```

One class of junk survives the silhouette test. Gaussians that hide
*behind* the subject inside the silhouette frustum project inside the
mask from every camera, so this test cannot catch them (depth is
unobservable from silhouettes). If a downstream consumer computes
statistics from the splat, either bound them spatially or add the
optional `--subject_anchor_ply poses_pcd_fullres/<tem>.ply
--subject_radius 3.0` test, which also drops everything farther than
the radius from the triangulated subject's median.

Run `--report_only` first to see the keep/drop split before you write
anything.
