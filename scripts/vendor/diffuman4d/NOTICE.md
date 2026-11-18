# Diffuman4D Attribution

Parts of two cumuli scripts derive from
[Diffuman4D](https://github.com/zju3dv/Diffuman4D) (Copyright 2025 ZJU3DV),
licensed under the Apache License 2.0. A copy
of that license is in [LICENSE](LICENSE).

Solipsist Studios modified the code in the `solipsist-studios/Diffuman4D`
fork and ported it into cumuli from that fork at `3656cd4` (tag `v0.0.1`),
when cumuli dropped the Diffuman4D submodule.

| Cumuli file | Derived from | Changes |
|---|---|---|
| `scripts/birefnet_masks.py` | `scripts/preprocess/remove_background.py` | argparse in place of `fire`; a thread pool in place of EasyVolcap's `parallel_execution`; path handling with `pathlib`; the rotation, RGBA-output and skip-existing options removed. |
| `scripts/triangulate_and_project_keypoints.py` | `scripts/preprocess/triangulate_skeleton.py`, `scripts/preprocess/utils/triang_utils.py`, `src/data/utils/camera_parser.py` | Only the per-camera-directory input path kept; the combined-JSON input, the 2D reprojection output, the CuPy path and the EasyVolcap camera format removed; distortion coefficients in the camera file now raise an error instead of being silently ignored. |

The modifications are Solipsist Studios' work and are licensed under this
repository's PolyForm Noncommercial license. The Apache-2.0 terms continue to
apply to the portions derived from Diffuman4D.
