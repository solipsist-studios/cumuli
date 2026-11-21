# Changelog

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
