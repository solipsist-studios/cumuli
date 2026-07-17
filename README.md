# volumetric-capture-pipeline

Multi-camera capture -> gaussian splat pipeline: sync GoPro footage,
estimate camera poses (HLOC), predict human keypoints (Sapiens), and
train a splat (Brush) on the real cameras. A Diffuman4D dense-ring
branch (hallucinating a full 48-camera view ring) is planned but not
part of this build yet.

See [docs/pipeline.md](docs/pipeline.md) for the full walkthrough with
example commands, stage by stage, and the unified orchestrator that runs
all of it in one command (see Quick start below). See
[docs/environment.md](docs/environment.md) for exact conda env setup
commands and known-good package versions.

## Layout

```
scripts/   pipeline scripts, plus run_unified_pipeline.py (the
           recommended single-command entry point) and
           render_frame_sequence.py for multi-frame playback once you
           have posed cameras
deps/      git submodules for the external tools each stage wraps
configs/   per-rig JSON configs for run_unified_pipeline.py's --config flag
           (conda env names, --brush_app, --display, etc.) -- see
           configs/README.md
envs/      environment.yml per conda env (hloc, diffuman4d, sapiens2),
           pinning the known-good versions from docs/environment.md
docs/      pipeline.md walkthrough, environment.md conda setup
```

`clean_masks.py`, `build_colmap_sparse.py`, and
`refine_poses_with_keypoints.py` are part of the core walkthrough, not
optional extras -- calibration validation (built into
`undistort_frames.py`), mask cleanup (`clean_masks.py`), RGBA-baked
masked training (`clean_masks.py`, `build_colmap_sparse.py`
`--masks_dir`; Brush auto-detects the alpha channel, no training flag
needed), and keypoint pose refinement (`refine_poses_with_keypoints.py`,
reliable -- real runs have taken subject-space median reprojection error
from ~30px to ~5px) are all needed to get a good result, not just a
first splat. Color correction (`extract_synced_frames.py`'s
`--pp3_dir`) is the one genuinely optional extra, needing a per-camera
RawTherapee `.pp3` profile directory most captures won't have. See
`docs/pipeline.md` for the full walkthrough in order.

## Submodules

```bash
git submodule update --init --recursive
```

- `deps/camera-calibration` -- per-camera undistortion (offline_undistort.py)
- `deps/BiRefNet` -- background removal model (invoked via Diffuman4D's own wrapper script)
- `deps/sapiens` -- 2D keypoint prediction model (invoked via Diffuman4D's own wrapper script)
- `deps/Diffuman4D` -- preprocessing scripts (masks, keypoints, triangulation) used by the direct branch today; also has the diffusion inference model for the 48-camera ring, planned but not wired into this build yet
- `deps/brush` -- gaussian splat trainer

## Conda environments

- `hloc` -- HLOC + pycolmap, for pose estimation (`run_hloc.py`)
- `diffuman4d` -- Diffuman4D's own deps (BiRefNet masks, nerfstudio conversion) (`generate_masks.py`, `triangulate_and_project_keypoints.py`)
- `sapiens2` -- Sapiens keypoint prediction; requires `SAPIENS_CHECKPOINT_ROOT` env var set (`predict_keypoints_2d.py`)

Most scripts have no special conda env requirement beyond
numpy/scipy/Pillow (and ffmpeg/ffprobe on PATH for the sync/extraction
scripts; rawtherapee-cli on PATH, or flatpak with RawTherapee installed,
for `extract_synced_frames.py`'s optional `--pp3_dir` color correction).
`build_colmap_sparse.py` and `refine_poses_with_keypoints.py` need
numpy/scipy/plyfile (no special env); `clean_masks.py` needs scipy +
Pillow and, for its `--retry` flag, the `diffuman4d` env (it calls
remove_background.py).

## Quick start

Recommended: run everything with the unified orchestrator (see
`docs/pipeline.md`'s "Recommended: run everything with the unified
orchestrator" section for the full flag reference):

```bash
python3 scripts/run_unified_pipeline.py \
    --config configs/my_rig.json \
    --video_dir <movies_dir> \
    --calib_dir <calibration_pkls_dir> \
    --out_dir <out_dir> \
    --target_time <e.g. 2500ms>
```

`--config` loads per-rig defaults (conda env names, `--brush_app`,
`--display`, `SAPIENS_CHECKPOINT_ROOT`) from a JSON file so you don't have
to repeat them on every run -- copy [configs/example_rig.json](configs/example_rig.json)
and fill in your own paths. See [configs/README.md](configs/README.md).
Everything it sets can still be overridden on the command line.

Or run the individual stages by hand -- useful the first time, to
understand what each one does:

```bash
python3 scripts/compute_sync_offsets.py <movies_dir> <out_dir> <ref_video.mp4>
python3 scripts/extract_synced_frames.py <movies_dir> <sync_offsets.json> <out_dir> <seconds>
```

Continue through `docs/pipeline.md` for the rest of the pipeline.
