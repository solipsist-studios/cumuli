# Changelog

## Unreleased

- Frame image formats. Frames can be written as lossless or lossy WebP, not
  only PNG. `.webp` with no quality is lossless; a quality (1-100) makes a lossy
  WebP whose alpha channel stays lossless. A quality with `.png` or `.jpg` is an
  error, and JPEG is not offered for training images (no alpha). Defaults are
  unchanged. New flags: `color_adjust.py --image_quality`,
  `build_flat_dataset.py --out_image_quality`,
  `undistort_frames.py --out_image_ext --out_image_quality`,
  `build_flipbook_4dgs_dataset.py --image_ext --out_image_ext --out_image_quality`,
  `build_4dgs_dataset.py --out_image_ext --out_image_quality`,
  `build_colmap_sparse.py --out_image_ext --out_image_quality`,
  `frame_motion.py`, `refine_sync_visual.py` and `associate_subjects.py`
  `--image_ext`. `extract_synced_frames.py --pp3_dir` accepts
  `--output_ext .webp`.
- Fixed: `color_adjust.py` and `build_flat_dataset.py` wrote `.webp` as lossy
  quality 80, and `undistort_frames.py` as OpenCV's lossy quality 95. All are
  lossless now.
- Fixed: `extract_synced_frames.py --window N` with `--output_ext .webp` wrote
  one animated file instead of N frames (ffmpeg picked the animated WebP
  encoder).
- A training frame that is not PNG carries its extension in its transforms
  `file_path`; PNG stays extensionless. Scored views (`--test_cameras`,
  `--eval_root`) and `eval_gt_flat/` stay lossless PNG.

## 0.2.0

Pins cumuli-core 0.2.0, cumuli-trainer 0.2.0 and camera-calibration 0.2.0.

- Degree-2 motion through the bake: SH-and-acceleration packing, windowed
  merge and seeding, and the mask-consistency filter along the full
  trajectory. Enable it in the trainer with `motion_degree: 2`.
- Sub-frame timing: `refine_sync_visual.py`, `frame_motion.py`,
  `build_flipbook_4dgs_dataset.py --timing_json --frame_weights`, and
  `eval_render.py --shutter`.
- Cumuli Studio support: `color_adjust.py`, `associate_subjects.py`,
  `stitch_window_plys.py`, `run_pose_refinement.py --report_json`, several
  moments per camera and `--refine_intrinsics` in `run_hloc.py`,
  `--aliked_model`, `--cross_camera_pairs_only`, parallel `undistort_frames.py`.
- Bake: optional dark-occluder filter, cropped-rig mask filter, smoothing
  default 0.3 px over 1 frame, low-bit drop pass-through, `fit_rate_target.py`.
- `.sogst` spec revisions 15 and 16: encoder precision, entry integrity,
  archive integrity and the TSOG section.
- Removed the Diffuman4D, BiRefNet and sapiens submodules. Masks come from
  `birefnet_masks.py`, keypoints from the vendored Sapiens2 demo in
  `scripts/vendor/sapiens2/`, and triangulation from
  `triangulate_and_project_keypoints.py`, which now rejects a camera file
  with nonzero distortion. See `scripts/vendor/diffuman4d/NOTICE.md`.
- Fixed six `undistort_frames.py` tests that still patched the removed
  subprocess call.

## 0.1.0

First release with the clean-room cumuli-trainer; OMG4 removed.
