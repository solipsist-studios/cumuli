# Volumetric capture pipeline

End-to-end flow from raw multi-camera GoPro footage to a trained Brush
gaussian splat, for a single frame/timestamp. All example paths below use
a `heidi_1500ms` example dataset (frame extracted at 1.5s into the clip)
-- substitute your own working directory.

Follow the sections below in order -- that's the pipeline order. Each
script's own `--help` / docstring has the full flag reference; this doc
is the narrative walkthrough and the conda envs each stage needs.

Only the direct branch (4K masked training on the real cameras) is part
of this build. The Diffuman4D 48-camera dense-ring branch is a planned
addition (hallucinating extra ring views via Diffuman4D, then training
on real + synthetic views together) but isn't wired into any script or
this walkthrough yet -- it'll get its own doc sections once it's been
run and validated end-to-end.

## Prerequisites

- Per-camera calibration PKLs (from `deps/camera-calibration`), one per
  physical camera.
- Raw GoPro `.mp4` clips, one per camera, all recording the same take.
- Conda envs: `hloc` (HLOC + pycolmap), `diffuman4d` (BiRefNet background
  removal deps), `sapiens2` (Sapiens keypoint prediction, needs
  `SAPIENS_CHECKPOINT_ROOT` set). Brush training runs the compiled
  `brush_app` binary directly, no conda env needed for that step.

## Recommended: run everything with the unified orchestrator

`scripts/run_unified_pipeline.py` runs the full pipeline end-to-end in
one process -- sync through mask cleanup and training, plus pose
refinement when a known-good sync is reused -- dispatching each stage
into the right conda env itself, so you don't need to `conda activate`
between steps the way the walkthrough below shows. Read the walkthrough
below first to understand what each stage actually does and why; use the
orchestrator to run them without the manual bookkeeping.

```bash
python3 scripts/run_unified_pipeline.py \
    --video_dir /path/to/movies \
    --calib_dir /path/to/calibration_pkls \
    --out_dir ~/pipeline_run \
    --target_time 2500ms \
    --start_from_stage production \
    --sapiens_checkpoint_root ~/sapiens/2
```

Key flags:

- **`--start_from_stage` / `--stop_after_stage`** (`sync`, `production`,
  `poses`, `masks`, `branch`) -- resume partway through, or stop early to
  inspect intermediate output before committing GPU time to training.
  Resuming assumes the earlier stages' outputs already exist under
  `--out_dir`.
- **Sync: verify once, reuse, don't re-trust automatically every run.**
  `compute_sync_offsets.py`'s only sync method, raw audio
  cross-correlation (always runs unless `--initial_sync_json` is given),
  hasn't been reliable enough in real-world testing to trust unverified
  -- it can silently converge on a wrong offset for periodic/rhythmic
  audio; `compute_sync_offsets.py` just produces the QA grid below for
  manual verification. The workflow that's actually worked every time
  this session:
  1. Run `compute_sync_offsets.py` once, inspect `make_sync_grid.py`'s
     sync grid by eye, hand-tune any camera's `frame_offset` that's
     visibly off, save the result (e.g. `sync_offsets_v5.json`).
  2. Pass that verified file back in via `--initial_sync_json` on the
     *next* run to seed it without recomputing.
  3. For every run after that on the same take (different timestamps,
     retries, etc.), skip `compute_sync_offsets.py` entirely: write the
     verified file's path into `<out_dir>/resolved_sync_json.txt`
     yourself and pass `--start_from_stage production`. This is the
     normal way to iterate without re-running (and re-trusting) sync
     search each time.
- **`--with_viewer`** (default: on) / **`--no_viewer`** -- on this
  machine, `--no_viewer` has reliably hung (a busy-spin thread, not a
  blocked syscall) regardless of the display setup tried -- no display
  at all, a dummy Xorg instance, a real composited display with no
  client connected, and a real composited display with an active client
  all reproduced it identically. This may be specific to this machine's
  driver/library setup rather than a general limitation -- it hasn't
  been confirmed to fail (or work) elsewhere. `--wgpu_backend=vulkan`
  (forces wgpu's backend instead of auto-probing) is available as an
  experimental option but did not resolve the hang in testing here.
  Only pass `--no_viewer` if you've separately confirmed headless
  training completes in your own environment.
- **`--display`** (default `:2`) -- the X display `brush_app` connects
  to for `--with_viewer`. The default is specific to this machine's
  setup; override it for your own (needs a real, composited display --
  a detached/dummy Xorg instance is not sufficient).
- **`--hloc_resize_max`** (default 4096) / **`--hloc_max_keypoints`**
  (default 8192) -- HLOC feature-extraction settings, passed through to
  `multiframe_sfm.py`. The 4096 default is deliberately higher than that
  script's own default of 2048: on an 11-camera rig with ~5312px native
  media, 2048 discarded enough detail to measurably hurt camera pose
  accuracy (one real run: post-refinement median reprojection error went
  from ~30px to ~5px after raising this, with the improvement concentrated
  in the previously-worst cameras).

## Sync and extract a frame

```bash
python3 scripts/compute_sync_offsets.py \
    /media/ai/datasets/260521-105422/movies \
    ~/heidi_260521_undist \
    0001.mp4
# -> ~/heidi_260521_undist/sync_offsets.json

# Hand-tune a few cameras' frame_offset by +/- a couple frames after
# visually checking make_sync_grid.py output at a few timestamps, and
# save as sync_offsets_v2.json, v3.json, etc. Point everything below at
# whichever version is your current best.

python3 scripts/extract_synced_frames.py \
    /media/ai/datasets/260521-105422/movies \
    ~/heidi_260521_undist/sync_offsets_v5.json \
    ~/heidi_1500ms/raw \
    1.5

python3 scripts/make_sync_grid.py ~/heidi_1500ms/raw ~/heidi_1500ms/sync_grid.jpg
# eyeball the grid -- every camera should show the same instant of action
```

**Color correction** (optional, needs a per-camera RawTherapee `.pp3`
profile directory, e.g. thumbnail sidecar files from an NLE): pass
`--pp3_dir` to `extract_synced_frames.py` and it color-corrects each
extracted frame before undistortion. Matters because uncorrected
per-GoPro exposure/saturation differences show up as color noise in the
trained splat.

## Undistort

```bash
python3 scripts/undistort_frames.py \
    --frames_dir ~/heidi_1500ms/raw \
    --calib_dir /path/to/calibration_pkls \
    --out_dir ~/heidi_1500ms/undistorted \
    --out_pkl_dir ~/heidi_1500ms/undistorted_pkls \
    --model OPENCV_FISHEYE
```

**Calibration sanity check** happens automatically here -- if a camera's
calibration `image_size` disagrees with the actual frame resolution by a
uniform scale, it is auto-corrected with a loud warning (this was a real
bug: a calibration made at the GoPro 5.3K full-sensor width, 5568px,
silently applied to 5312px-wide video warped every camera by ~113px of
focal). Read those warnings if they appear -- they mean you're pointing
at the wrong calibration source. `--target_pkl_dir` also enables
single-warp mode: native fisheye straight to a known target pinhole
geometry in one resample, instead of downscale-then-undistort.

## HLOC pose estimation

```bash
conda activate hloc
python3 scripts/run_hloc.py \
    --undistorted_dir ~/heidi_1500ms/undistorted \
    --undistorted_pkl_dir ~/heidi_1500ms/undistorted_pkls \
    --outputs_dir ~/heidi_1500ms/solipsist_out
# -> ~/heidi_1500ms/solipsist_out/transforms_multiframe.json (real camera poses)
```

## Build the flat, 2-digit-labeled dataset

Downstream tooling looks up cameras by the literal `camera_label` string
in transforms.json, and assumes plain zero-padded 2-digit labels ("00",
"01", ...). This step is where HLOC's `Camera_undistorted_0001`-style
labels get converted to that convention -- every later stage inherits it.
This is purely a relabeling step, independent of pose quality, so it's
safe to do before pose refinement below (and lets that step reuse this
frame's masks/keypoints instead of predicting them twice).

```bash
python3 scripts/build_flat_dataset.py \
    --transforms ~/heidi_1500ms/solipsist_out/transforms_multiframe.json \
    --undistorted_dir ~/heidi_1500ms/undistorted \
    --out_images_flat ~/heidi_1500ms/images_flat \
    --out_transforms ~/heidi_1500ms/transforms.json
```

## Masks and 2D keypoints for this frame

```bash
conda activate diffuman4d
python3 scripts/generate_masks.py \
    --images_dir ~/heidi_1500ms/images_flat \
    --out_fmasks_dir ~/heidi_1500ms/fmasks_flat

conda activate sapiens2
python3 scripts/predict_keypoints_2d.py \
    --images_dir ~/heidi_1500ms/images_flat \
    --out_kp2d_dir ~/heidi_1500ms/poses_2d_flat \
    --fmasks_dir ~/heidi_1500ms/fmasks_flat

python3 scripts/split_keypoints_per_camera.py \
    --kp2d_flat_dir ~/heidi_1500ms/poses_2d_flat \
    --out_dir ~/heidi_1500ms/poses_2d
```

**Mask quality**: clean masks before trusting them for anything -- raw
BiRefNet output can include bystanders/props and can also drop part of
the subject:

```bash
python3 scripts/clean_masks.py \
    --fmasks_dir ~/heidi_1500ms/fmasks_flat \
    --kp2d_dir ~/heidi_1500ms/poses_2d \
    --out_dir ~/heidi_1500ms/fmasks_clean \
    --images_dir ~/heidi_1500ms/images_flat \
    --retry
```

Score any evaluation against the CLEANED masks, not the raw ones -- raw
masks with bystanders make a good model's eval metrics look broken.
Masking/alignment quality on real captures is still an open problem
being actively worked (see project notes) -- treat this as the current
best approach, not a solved one.

## Keypoint pose refinement

Background SfM features (HLOC, above) live on distant walls/windows, so
camera poses can fit the background well while still disagreeing by
dozens of pixels at the capture volume center -- exactly where the
subject is. This step bundle-adjusts the camera poses against this
frame's 2D human keypoints instead ("human as calibration wand"), and
has been reliable: real runs have seen it take subject-space median
reprojection error from ~30px to ~5px.

```bash
python3 scripts/refine_poses_with_keypoints.py \
    --transforms ~/heidi_1500ms/transforms.json \
    --kp2d ~/heidi_1500ms/poses_2d \
    --out_transforms ~/heidi_1500ms/transforms_refined.json \
    --report_only
# check the printed median px error, then re-run without --report_only
```

Use `transforms_refined.json` in place of `transforms.json` in the
triangulate/build/train step below.

This single-instant refinement is worthwhile on its own, but the
optimization gets meaningfully more constrained with keypoints from
several time instants of the same static rig instead of one (the subject
sweeping through the capture volume over a few seconds anchors the
cameras far more strongly than a single pose). `run_pose_refinement.py`
wraps the same underlying script for that case: extract a short
candidate window per camera, predict keypoints on each instant, and
refine against all of them at once (10+ instants recommended):

```bash
python3 scripts/extract_synced_frames.py \
    /media/ai/datasets/260521-105422/movies \
    ~/heidi_260521_undist/sync_offsets_v5.json \
    ~/heidi_1500ms/sync_candidates 1.5 --window 5

# Run undistort_frames.py + generate_masks.py + predict_keypoints_2d.py +
# split_keypoints_per_camera.py on each instant subdir f0/..f4/
for k in 0 1 2 3 4; do
    python3 scripts/undistort_frames.py --frames_dir ~/heidi_1500ms/sync_candidates/f$k \
        --calib_dir /path/to/calibration_pkls --out_dir ~/heidi_1500ms/sync_candidates_undist/f$k \
        --out_pkl_dir ~/heidi_1500ms/sync_candidates_pkls/f$k
    conda activate diffuman4d
    python3 scripts/generate_masks.py \
        --images_dir ~/heidi_1500ms/sync_candidates_undist/f$k \
        --out_fmasks_dir ~/heidi_1500ms/sync_candidates_fmasks/f$k
    conda activate sapiens2
    python3 scripts/predict_keypoints_2d.py \
        --images_dir ~/heidi_1500ms/sync_candidates_undist/f$k \
        --out_kp2d_dir ~/heidi_1500ms/sync_candidates_kp2d/f$k \
        --fmasks_dir ~/heidi_1500ms/sync_candidates_fmasks/f$k
    python3 scripts/split_keypoints_per_camera.py \
        --predictions_json ~/heidi_1500ms/sync_candidates_kp2d/f$k \
        --out_dir ~/heidi_1500ms/sync_candidates_poses2d/f$k
done

python3 scripts/run_pose_refinement.py \
    --transforms ~/heidi_1500ms/transforms.json \
    --kp2d_dirs ~/heidi_1500ms/sync_candidates_poses2d/f0,~/heidi_1500ms/sync_candidates_poses2d/f1,~/heidi_1500ms/sync_candidates_poses2d/f2,~/heidi_1500ms/sync_candidates_poses2d/f3,~/heidi_1500ms/sync_candidates_poses2d/f4 \
    --out_transforms ~/heidi_1500ms/transforms_refined.json --report_only
# check the printed median px error, then re-run without --report_only
```

## Triangulate, build the training set, and train Brush

```bash
python3 scripts/triangulate_and_project_keypoints.py \
    --camera_path ~/heidi_1500ms/transforms_refined.json \
    --kp2d_dir ~/heidi_1500ms/poses_2d \
    --out_kp3d_dir ~/heidi_1500ms/poses_3d \
    --out_pcd_dir ~/heidi_1500ms/poses_pcd_fullres
# real cameras only -- omit --out_kp2d_proj_dir/--n_total, those are for
# projecting into the (not-yet-built) 48-camera Diffuman4D ring
```

Then bake the cleaned masks into image alpha -- **do not** pass masks as
a separate folder next to same-named/same-extension images; Brush has
been observed to silently ignore that and train the full unmasked scene.
Training itself needs no special flag: `brush_app` has no `--alpha-mode`
option (verified against `brush_app --help`) -- it auto-detects the
alpha channel and applies its own `--match-alpha-weight` loss:

```bash
python3 scripts/build_colmap_sparse.py \
    --transforms ~/heidi_1500ms/transforms_refined.json \
    --points_ply ~/heidi_1500ms/poses_pcd_fullres/000000.ply \
    --out_dir ~/heidi_1500ms/train_set \
    --images_dir ~/heidi_1500ms/images_flat \
    --masks_dir ~/heidi_1500ms/fmasks_clean

python3 scripts/train_brush.py \
    --data ~/heidi_1500ms/train_set \
    --brush_app ~/brush-app-x86_64-unknown-linux-gnu/brush_app \
    --export_path ~/brush_output \
    --export_name heidi_1500ms_{iter}.ply
# opens Brush's live viewer by default (see "Recommended" section above for why);
# pass --no_viewer only if you've confirmed headless training works in your environment
```
