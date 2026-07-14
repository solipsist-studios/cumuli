# Volumetric capture pipeline

End-to-end flow from raw multi-camera GoPro footage to a trained Brush
gaussian splat, for a single frame/timestamp. All example paths below use
a `heidi_1500ms` example dataset (frame extracted at 1.5s into the clip)
-- substitute your own working directory.

Follow the sections below in order -- that's the pipeline order. Each
script's own `--help` / docstring has the full flag reference; this doc
is the narrative walkthrough and the conda envs each stage needs.

## Prerequisites

- Per-camera calibration PKLs (from `deps/camera-calibration`), one per
  physical camera.
- Raw GoPro `.mp4` clips, one per camera, all recording the same take.
- Conda envs: `hloc` (HLOC + pycolmap), `diffuman4d` (Diffuman4D
  inference deps, also used for BiRefNet background removal), `sapiens2`
  (Sapiens keypoint prediction, needs `SAPIENS_CHECKPOINT_ROOT` set).
  Brush training runs the compiled `brush_app` binary directly, no conda
  env needed for that step.

## Recommended: run everything with the unified orchestrator

`scripts/run_unified_pipeline.py` runs the full pipeline end-to-end in
one process -- sync through mask cleanup, plus pose refinement when a
known-good sync is reused -- dispatching each stage into the right conda
env itself, so you don't need to `conda activate` between steps the way
the walkthrough below shows. Read the walkthrough below first to
understand what each stage actually does and why; use the orchestrator
to run them without the manual bookkeeping.

```bash
python3 scripts/run_unified_pipeline.py \
    --video_dir /path/to/movies \
    --calib_dir /path/to/calibration_pkls \
    --out_dir ~/pipeline_run \
    --target_time 2500ms \
    --start_from_stage production \
    --no_diffuman \
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
  audio. (Automatic skeleton-based correction, `skeleton_sync_search.py`,
  isn't part of this build -- it has not produced good results in
  practice; `compute_sync_offsets.py` just produces the QA grid below for
  manual verification.) The workflow that's actually worked every time
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
- **`--no_diffuman` / `--use_diffuman`** (required, mutually exclusive) --
  branch A (direct 4K masked training on the real cameras, via
  `build_colmap_sparse.py` + `train_brush.py`, the only branch validated
  so far) vs. branch B (Diffuman4D's 48-camera dense ring). **Branch B
  isn't part of this build yet** -- selecting `--use_diffuman` fails
  immediately with a clear message; see the Diffuman4D-branch sections
  below.
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

## Undistort

```bash
python3 scripts/undistort_frames.py \
    --frames_dir ~/heidi_1500ms/raw \
    --calib_dir /path/to/calibration_pkls \
    --out_dir ~/heidi_1500ms/undistorted \
    --out_pkl_dir ~/heidi_1500ms/undistorted_pkls \
    --model OPENCV_FISHEYE
```

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

Diffuman4D's camera parser looks up cameras by the literal
`camera_label` string in transforms.json, and its own `--spa_labels` /
`--spa_label_range` args assume plain zero-padded 2-digit labels ("00",
"01", ...). This step is where HLOC's `Camera_undistorted_0001`-style
labels get converted to that convention -- every later stage inherits it.

```bash
python3 scripts/build_flat_dataset.py \
    --transforms ~/heidi_1500ms/solipsist_out/transforms_multiframe.json \
    --undistorted_dir ~/heidi_1500ms/undistorted \
    --out_images_flat ~/heidi_1500ms/images_flat \
    --out_transforms ~/heidi_1500ms/transforms.json
```

## Masks, 2D keypoints, split

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
    --predictions_json ~/heidi_1500ms/poses_2d_flat \
    --out_dir ~/heidi_1500ms/poses_2d
```

**The Diffuman4D dense-ring branch sections below are not yet included
in this build -- only the direct 4K masked branch (`--no_diffuman`, via
`build_colmap_sparse.py` + `train_brush.py`) has actually been run and
validated.** The scripts and orchestrator wiring for this branch will
land once it's been tested end-to-end. Kept here as a design reference
for that branch's intended shape.

## Resize for Diffuman4D

Center-crop + resize every real camera to 1024x1024 (Diffuman4D's own
training resolution), remapping intrinsics and keypoints to match.

```bash
python3 scripts/resize_for_diffuman4d.py \
    --images_dir ~/heidi_1500ms/images_flat \
    --fmasks_dir ~/heidi_1500ms/fmasks_flat \
    --transforms ~/heidi_1500ms/transforms.json \
    --poses_2d_dir ~/heidi_1500ms/poses_2d \
    --out_dir ~/heidi_1500ms_1024 \
    --size 1024
```

## Generate the 48-camera ring

First-draft geometry -- derives ring center/radius/height range/up-axis
from the real cameras. Inspect visually before trusting it; see the
script docstring for the tunable knobs.

```bash
python3 scripts/generate_camera_ring.py \
    --transforms ~/heidi_1500ms_1024/transforms.json \
    --out_transforms ~/heidi_1500ms_1024/transforms_48cam.json
```

## Triangulate, project, draw skeletons

```bash
python3 scripts/triangulate_and_project_keypoints.py \
    --camera_path ~/heidi_1500ms_1024/transforms_48cam.json \
    --kp2d_dir ~/heidi_1500ms_1024/poses_2d \
    --out_kp3d_dir ~/heidi_1500ms_1024/poses_3d \
    --out_pcd_dir ~/heidi_1500ms_1024/poses_pcd \
    --out_kp2d_proj_dir ~/heidi_1500ms_1024/poses_2d_proj

python3 scripts/draw_skeletons.py \
    --kp2d_dir ~/heidi_1500ms_1024/poses_2d_proj \
    --out_kpmap_dir ~/heidi_1500ms_1024/skeletons
```

## Diffuman4D inference

```bash
conda activate diffuman4d
python3 scripts/run_diffuman4d_inference.py \
    --images_dir ~/heidi_1500ms_1024/images_flat \
    --fmasks_dir ~/heidi_1500ms_1024/fmasks_flat \
    --skeletons_dir ~/heidi_1500ms_1024/skeletons \
    --transforms ~/heidi_1500ms_1024/transforms_48cam.json \
    --data_dir ~/diffuman4d_data \
    --scene_label 260521_1500ms_48cam \
    --n_real 11
# -> deps/Diffuman4D/output/results/demo_3d/260521_1500ms_48cam/
```

## Prepare and train Brush

`prepare_brush_dataset.py` (below) is diffuman-branch-only -- it wraps
Diffuman4D's own converter and requires the Diffuman4D inference step's
output, so it's not usable independent of that branch either; not part
of this build for the same reason as the rest of the Diffuman4D-branch
sections above. `train_brush.py` (also below) is shared and validated --
the direct branch uses it via `build_colmap_sparse.py`'s output instead
(see the "Quality improvements" section further down).

```bash
python3 scripts/prepare_brush_dataset.py \
    --diffuman4d_output_dir deps/Diffuman4D/output/results/demo_3d/260521_1500ms_48cam \
    --out_dir ~/brush_heidi_1500ms \
    --input_cameras 00,01,02,03,04,05,06,07,08,09,10

python3 scripts/train_brush.py \
    --data ~/brush_heidi_1500ms \
    --brush_app ~/brush-app-x86_64-unknown-linux-gnu/brush_app \
    --export_path ~/brush_output \
    --export_name heidi_1500ms_75k_{iter}.ply
# opens Brush's live viewer by default (see "Recommended" section above for why);
# pass --no_viewer only if you've confirmed headless training works in your environment
```

## Quality improvements (optional, recommended for production captures)

These came out of a deep-dive on a static-GoPro-rig capture that had
visible ghosting; each fixes a specific measured failure mode. They are
optional relative to the sections above (which are enough to get a first
splat), but for anything beyond a quick test, do them in this order:

**Calibration sanity check** happens automatically inside
`undistort_frames.py` now -- if a camera's calibration `image_size`
disagrees with the actual frame resolution by a uniform scale, it is
auto-corrected with a loud warning (this was a real bug: a calibration
made at the GoPro 5.3K full-sensor width, 5568px, silently applied to
5312px-wide video warped every camera by ~113px of focal). Read those
warnings if they appear -- they mean you're pointing at the wrong
calibration source. `--target_pkl_dir` on `undistort_frames.py` also
enables single-warp mode: native fisheye straight to a known target
pinhole geometry in one resample, instead of downscale-then-undistort.

**Color correction** (optional, needs a per-camera RawTherapee `.pp3`
profile directory, e.g. thumbnail sidecar files from an NLE): pass
`--pp3_dir` to `extract_synced_frames.py` and it color-corrects each
extracted frame before undistortion. Matters because uncorrected
per-GoPro exposure/saturation differences show up as color noise in the
trained splat.

**Multi-instant capture, sync search, and pose refinement** -- audio
sync (`compute_sync_offsets.py`) fails silently at live-music venues
(periodic beats create false correlation peaks; low confidence scores
are the tell). The subject's own skeleton is the intended fix for that
via `skeleton_sync_search.py` (steps below) -- **but that script isn't
part of this build yet**: it has not produced good results in practice
(see "Sync: verify once, reuse" above). Kept here as a design reference
for when it's validated and re-added. The **pose refinement** step
further below (`refine_poses_with_keypoints.py`) is a different thing --
camera *pose* accuracy, not *sync* timing -- and has been reliable: real
runs have seen it take subject-space median reprojection error from
~30px to ~5px.

```bash
# Extract N consecutive candidate frames per camera (also works for a
# single production instant if you pass --pp3_dir here for color correction)
python3 scripts/extract_synced_frames.py \
    /media/ai/datasets/CAPTURE/movies ~/capture/sync_offsets.json \
    ~/capture/sync_candidates 1.5 --window 5

# Run undistort_frames.py + generate_masks.py + predict_keypoints_2d.py on each instant subdir f0/..f4/
for k in 0 1 2 3 4; do
    python3 scripts/undistort_frames.py --frames_dir ~/capture/sync_candidates/f$k \
        --calib_dir /path/to/calibration_pkls --out_dir ~/capture/sync_candidates_undist/f$k \
        --out_pkl_dir ~/capture/sync_candidates_pkls/f$k
    conda activate diffuman4d
    python3 scripts/generate_masks.py \
        --images_dir ~/capture/sync_candidates_undist/f$k \
        --out_fmasks_dir ~/capture/sync_candidates_fmasks/f$k
    conda activate sapiens_lite  # or sapiens2, see predict_keypoints_2d.py's docstring
    python3 scripts/predict_keypoints_2d.py \
        --images_dir ~/capture/sync_candidates_undist/f$k \
        --out_kp2d_dir ~/capture/sync_candidates_kp2d/f$k \
        --fmasks_dir ~/capture/sync_candidates_fmasks/f$k
    python3 scripts/split_keypoints_per_camera.py \
        --kp2d_flat_dir ~/capture/sync_candidates_kp2d/f$k \
        --out_dir ~/capture/sync_candidates_poses2d/f$k
done

# Find the per-camera frame that minimizes skeleton reprojection error
python3 scripts/skeleton_sync_search.py \
    --transforms ~/capture/solipsist_out/transforms_multiframe.json \
    --kp2d_dirs ~/capture/sync_candidates_poses2d/f0,~/capture/sync_candidates_poses2d/f1,~/capture/sync_candidates_poses2d/f2,~/capture/sync_candidates_poses2d/f3,~/capture/sync_candidates_poses2d/f4 \
    --sync_json ~/capture/sync_offsets.json \
    --out_dir ~/capture/sync_search
# -> sync_search_report.json: a FLAT per-camera error curve means the
#    problem isn't temporal at all -- it's pose error, go straight to
#    refinement below. A V-shaped curve means re-extract with
#    sync_offsets_corrected.json.

# Bundle-adjust poses against the same multi-instant keypoints
# ("human as calibration wand" -- anchors accuracy at the subject,
# where background SfM features from run_hloc.py don't reach)
python3 scripts/refine_poses_with_keypoints.py \
    --transforms ~/capture/solipsist_out/transforms_multiframe.json \
    --kp2d_dirs ~/capture/sync_candidates_poses2d/f0,~/capture/sync_candidates_poses2d/f1,~/capture/sync_candidates_poses2d/f2,~/capture/sync_candidates_poses2d/f3,~/capture/sync_candidates_poses2d/f4 \
    --out_transforms ~/capture/transforms_refined.json --report_only
# check the printed median px error, then re-run without --report_only
```

Use `transforms_refined.json` in place of `transforms_multiframe.json`
in every stage from `build_flat_dataset.py` onward. On the capture this
was developed against, this took subject-space median reprojection error
from ~53px to ~5px and fixed between-camera midpoint renders that had
shown a dark ghost/fog around the subject.

**Masked (subject-only) training**: generate masks (`generate_masks.py`),
clean them before trusting them for anything -- raw BiRefNet output can
include bystanders/props and can also drop part of the subject:

```bash
python3 scripts/clean_masks.py \
    --fmasks_dir ~/capture/fmasks_flat --kp2d_dir ~/capture/poses_2d \
    --out_dir ~/capture/fmasks_clean --images_dir ~/capture/images_flat --retry
```

Then bake the cleaned masks into image alpha -- **do not** pass masks as
a separate folder next to same-named/same-extension images; Brush has
been observed to silently ignore that and train the full unmasked scene.
Training itself needs no special flag: `brush_app` has no `--alpha-mode`
option (verified against `brush_app --help`) -- it auto-detects the
alpha channel and applies its own `--match-alpha-weight` loss:

```bash
python3 scripts/build_colmap_sparse.py \
    --transforms ~/capture/transforms_refined.json \
    --points_ply ~/capture/poses_pcd_fullres/000000.ply \
    --out_dir ~/capture/train_set \
    --images_dir ~/capture/images_flat \
    --masks_dir ~/capture/fmasks_clean

python3 scripts/train_brush.py \
    --data ~/capture/train_set \
    --brush_app ~/brush-app-x86_64-unknown-linux-gnu/brush_app \
    --export_path ~/brush_output --export_name capture_masked_{iter}.ply
```

Score masked runs against the CLEANED masks, not the raw ones -- raw
masks with bystanders make a good model's eval metrics look broken.

## Known first-draft areas

These had no surviving ground-truth command from prior runs and are
first-draft designs -- validate them against a real run before trusting
the output:

- **generate_camera_ring.py**: ring geometry (center/up/radius/height
  tiers) is derived algorithmically from the real cameras, not copied
  from a known-good prior run. Check the printed center/up/radius values
  and look at the ring visually before running inference.
- **resize_for_diffuman4d.py**: assumes a center crop. If the subject
  isn't centered in the raw frame, this will crop them out.
