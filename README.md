# volumetric-capture-pipeline

Multi-camera capture -> gaussian splat pipeline: sync GoPro footage,
estimate camera poses (HLOC), predict human keypoints (Sapiens),
hallucinate a full 48-camera view ring (Diffuman4D), and train a splat
(Brush).

See [docs/pipeline.md](docs/pipeline.md) for the full walkthrough with
example commands, stage by stage, and the unified orchestrator that runs
all of it in one command (see Quick start below).

## Layout

```
scripts/   pipeline scripts, plus run_unified_pipeline.py (the
           recommended single-command entry point) and
           render_frame_sequence.py for multi-frame playback once you
           have posed cameras
deps/      git submodules for the external tools each stage wraps
configs/   placeholder for example configs / flag presets (empty for now)
docs/      pipeline.md walkthrough
```

`clean_masks.py`, `build_colmap_sparse.py`, and
`refine_poses_with_keypoints.py` are optional but recommended for
anything beyond a quick test -- calibration validation (built into
`undistort_frames.py`), color correction (`undistort_frames.py` /
`extract_synced_frames.py` flags), mask cleanup (`clean_masks.py`),
RGBA-baked masked training (`clean_masks.py`, `build_colmap_sparse.py`
`--masks_dir`; Brush auto-detects the alpha channel, no training flag
needed), and keypoint pose refinement (`refine_poses_with_keypoints.py`,
reliable -- real runs have taken subject-space median reprojection error
from ~30px to ~5px). Skeleton-based sync search (`skeleton_sync_search.py`,
intended for when audio sync is unreliable, e.g. live-music venues) isn't
part of this build yet -- it has not produced good results in practice;
see `docs/pipeline.md`'s "Sync: verify once, reuse" and "Quality
improvements" sections.

## Submodules

```bash
git submodule update --init --recursive
```

- `deps/camera-calibration` -- per-camera undistortion (offline_undistort.py)
- `deps/BiRefNet` -- background removal model (invoked via Diffuman4D's own wrapper script)
- `deps/sapiens` -- 2D keypoint prediction model (invoked via Diffuman4D's own wrapper script)
- `deps/Diffuman4D` -- preprocessing scripts (masks, keypoints, skeletons, triangulation) and the diffusion inference model that fills in the 48-camera ring
- `deps/brush` -- gaussian splat trainer

## Conda environments

- `hloc` -- HLOC + pycolmap, for pose estimation (`run_hloc.py`)
- `diffuman4d` -- Diffuman4D's own deps (BiRefNet masks, inference, nerfstudio conversion) (`generate_masks.py` and the not-yet-built Diffuman4D-branch scripts)
- `sapiens2` -- Sapiens keypoint prediction; requires `SAPIENS_CHECKPOINT_ROOT` env var set (`predict_keypoints_2d.py`)

Most scripts have no special conda env requirement beyond
numpy/scipy/Pillow (and ffmpeg/ffprobe on PATH for the sync/extraction
scripts; rawtherapee-cli on PATH, or flatpak with RawTherapee installed,
for `extract_synced_frames.py`'s optional `--pp3_dir` color correction).

Optional quality-improvement scripts: `build_colmap_sparse.py` and
`refine_poses_with_keypoints.py` need numpy/scipy/plyfile (no special
env); `clean_masks.py`/`skeleton_sync_search.py` need scipy + Pillow and,
for `clean_masks.py`'s `--retry`, the `diffuman4d` env (it calls
remove_background.py). `refine_poses_with_keypoints.py` shells out to
`scripts/vendor/refine_poses_with_keypoints.py` by default -- pass
`--refine_script` to point at a different copy instead.

## Quick start

Recommended: run everything with the unified orchestrator (see
`docs/pipeline.md`'s "Recommended: run everything with the unified
orchestrator" section for the full flag reference):

```bash
python3 scripts/run_unified_pipeline.py \
    --video_dir <movies_dir> --calib_dir <calibration_pkls_dir> \
    --out_dir <out_dir> --target_time <e.g. 2500ms> \
    --no_diffuman \
    --sapiens_checkpoint_root <path_to_sapiens_checkpoints>
```

Or run the individual stages by hand -- useful the first time, to
understand what each one does:

```bash
python3 scripts/compute_sync_offsets.py <movies_dir> <out_dir> <ref_video.mp4>
python3 scripts/extract_synced_frames.py <movies_dir> <sync_offsets.json> <out_dir> <seconds>
```

Continue through `docs/pipeline.md` for the rest of the pipeline.
