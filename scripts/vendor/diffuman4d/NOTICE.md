# Diffuman4D Attribution and Statement of Changes

Two cumuli scripts are modified versions of code from
[Diffuman4D](https://github.com/zju3dv/Diffuman4D) (Copyright 2025 ZJU3DV),
which is licensed under the Apache License, Version 2.0. A copy of that
license is in [LICENSE](LICENSE). Diffuman4D ships no `NOTICE` file, and the
source files used here carry no copyright or attribution headers of their
own, so there are no further upstream notices to reproduce.

Solipsist Studios Inc. first modified the code in its
`solipsist-studios/Diffuman4D` fork, then ported it into cumuli from that
fork at `3656cd4` (tag `v0.0.1`) when cumuli dropped the Diffuman4D
submodule. The changes below cover both steps. Each modified file carries a
notice in its own header.

## `scripts/birefnet_masks.py`

Derived from `scripts/preprocess/remove_background.py`.

- Command line: `argparse` in place of `fire`.
- Parallelism: a standard-library thread pool in place of EasyVolcap's
  `parallel_execution`, and `tqdm` imported directly.
- Paths: `pathlib` in place of string replacement on the image directory.
- Images: each input file is decoded and closed, not left open.
- Imports: `torch`, `torchvision`, `tqdm` and `transformers` are imported
  where used, so the mask post-processing imports without them.
- Devices: one CPU model when no CUDA device is visible (the fork's change),
  and fp16 only on CUDA with an explicit float32 cast on CPU (the fork's
  change).
- Removed: the image rotation, RGBA-output, skip-existing and `gpu_ids`
  options.

## `scripts/triangulate_and_project_keypoints.py`

Derived from `scripts/preprocess/triangulate_skeleton.py`,
`scripts/preprocess/utils/triang_utils.py` and
`src/data/utils/camera_parser.py`, merged into one file.

- Input: only the per-camera-directory keypoint layout is kept. The combined
  predictions JSON input, intrinsics-scale auto-detection, keypoint padding,
  label-range options, skip-existing and parallel workers are removed.
- Output: the 2D re-projection output is removed (nothing consumed it),
  along with the dummy projection camera the old command needed to run.
- Cameras: read directly from the nerfstudio `transforms.json` in float64.
  The old parser built float32 tensors, so keypoints differ by under 6e-6
  rig units on well-conditioned points. The EasyVolcap camera format and
  scene normalisation are removed.
- Distortion: a camera with any nonzero `k1`..`k4`, `p1` or `p2` is now an
  error. The old parser read those fields and ignored them.
- Non-camera files in the keypoint directory are ignored.
- Removed: the CuPy GPU path.
- Kept: the `score_thr` option, now a command-line flag.

## Provenance of `triang_utils.py`

Checked 2026-11-20. GitHub's history page for the upstream file listed one
commit, the Diffuman4D initial release, though the page may not have shown
the full history. The file contains no comment naming another project as its source, and
the installed EasyVolcap contains no similar triangulation code. The
algorithm is a weighted linear (DLT) solve refined by a Huber-loss
reprojection fit, a standard method. This is an absence of evidence, not
proof that the code has no other origin.

## Licensing of the modifications

The modifications are Solipsist Studios' work and are licensed under this
repository's PolyForm Noncommercial license, as Section 4 of the Apache
License permits. The Apache License continues to govern the portions derived
from Diffuman4D.
