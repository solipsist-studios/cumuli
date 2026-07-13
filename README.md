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
scripts/   numbered pipeline scripts (01_... through 20_...), plus
           21_run_unified_pipeline.py, the recommended single-command
           entry point, and 22_render_frame_sequence.py for multi-frame
           playback once you have posed cameras
deps/      git submodules for the external tools each stage wraps
configs/   placeholder for example configs / flag presets (empty for now)
docs/      pipeline.md walkthrough
```

Stages 17, 18, and 20 are optional but recommended for anything beyond a
quick test -- calibration validation (built into 04), color correction
(04/02 flags), mask cleanup (18), RGBA-baked masked training (18, 17
`--masks_dir`; Brush auto-detects the alpha channel, no training flag
needed), and keypoint pose refinement (20, reliable -- real runs have
taken subject-space median reprojection error from ~30px to ~5px).
Skeleton-based sync search (stage 19, intended for when audio sync is
unreliable, e.g. live-music venues) isn't part of this build yet -- it
has not produced good results in practice; see `docs/pipeline.md`
"Sync: verify once, reuse". See `docs/pipeline.md`
§17-20.

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

- `hloc` -- HLOC + pycolmap, for pose estimation (script 05)
- `diffuman4d` -- Diffuman4D's own deps (BiRefNet masks, inference, nerfstudio conversion) (scripts 07, 14, 15)
- `sapiens2` -- Sapiens keypoint prediction; requires `SAPIENS_CHECKPOINT_ROOT` env var set (script 08)

Scripts 01-04, 06, 09-13, 16 have no special conda env requirement beyond
numpy/scipy/Pillow (and ffmpeg/ffprobe on PATH for 01-02; rawtherapee-cli
on PATH, or flatpak with RawTherapee installed, for 02's optional
`--pp3_dir` color correction).

Optional stages 17-20: 17/20 need numpy/scipy/plyfile (no special env);
18/19 need scipy + Pillow and, for 18's `--retry`, the `diffuman4d` env
(it calls remove_background.py). 20 shells out to
`~/4dgs-utils/refine_poses_with_keypoints.py` by default -- clone
[solipsist-studios/4dgs-utils](https://github.com/solipsist-studios/4dgs-utils)
there, or pass `--refine_script`.

## Quick start

Recommended: run everything with the unified orchestrator (see
`docs/pipeline.md`'s "Recommended: run everything with the unified
orchestrator" section for the full flag reference):

```bash
python3 scripts/21_run_unified_pipeline.py \
    --video_dir <movies_dir> --calib_dir <calibration_pkls_dir> \
    --out_dir <out_dir> --target_time <e.g. 2500ms> \
    --no_diffuman \
    --sapiens_checkpoint_root <path_to_sapiens_checkpoints> \
    --multiframe_sfm_script <path_to_4dgs-utils>/multiframe_sfm.py
```

Or run the individual numbered stages by hand -- useful the first time,
to understand what each one does:

```bash
python3 scripts/01_compute_sync_offsets.py <movies_dir> <out_dir> <ref_video.mp4>
python3 scripts/02_extract_synced_frames.py <movies_dir> <sync_offsets.json> <out_dir> <seconds>
```

Continue through `docs/pipeline.md` for the rest of the stages.
