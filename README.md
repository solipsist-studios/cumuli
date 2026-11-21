<!--
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
-->

# volumetric-capture-pipeline

Multi-camera capture -> gaussian splat pipeline: sync GoPro footage,
estimate camera poses (HLOC), predict human keypoints (Sapiens), and
train a 4D gaussian splat on the real cameras, baked to a streamable
`.sogst` asset.

See [docs/pipeline.md](docs/pipeline.md) for the full walkthrough with
example commands, stage by stage, and the unified orchestrator that runs
all of it in one command (see Quick start below). See
[docs/environment.md](docs/environment.md) for exact conda env setup
commands and known-good package versions. See
[docs/integration-tests.md](docs/integration-tests.md) for how the real,
unmocked end-to-end pipeline test works, including the CPU-only variant
that runs automatically on every PR.

## Layout

```
scripts/   pipeline scripts, plus run_unified_pipeline.py (the
           recommended single-command entry point) and
           render_frame_sequence.py for multi-frame playback once you
           have posed cameras. scripts/vendor/ holds third-party code
           under its own license (see Third-party model licenses)
deps/      git submodules for the external tools each stage wraps
configs/   per-rig JSON configs for run_unified_pipeline.py's --config flag
           (SAPIENS_CHECKPOINT_ROOT, HLOC settings), see
           configs/README.md
environment.yml  the cumuli env manifest (see scripts/setup_cumuli_env.sh),
           pinning the known-good versions from docs/environment.md
tests/     unit/ (one file per script, mocked) and integration/ (real,
           unmocked end-to-end pipeline run against a committed fixture,
           see docs/integration-tests.md)
docs/      pipeline.md walkthrough, environment.md conda setup,
           integration-tests.md test suite
```

`clean_masks.py`, `build_colmap_sparse.py`, and
`refine_poses_with_keypoints.py` are part of the core walkthrough, not
optional extras. Calibration validation (built into
`undistort_frames.py`), mask cleanup (`clean_masks.py`), RGBA-baked
masked training (`clean_masks.py`, whose masks are baked into the
training dataset's RGBA alpha, so alpha supervision is part of training
by construction), and keypoint pose refinement (`refine_poses_with_keypoints.py`,
reliable: real runs have taken subject-space median reprojection error
from ~30px to ~5px) are all needed to get a good result, not just a
first splat. Color correction (`extract_synced_frames.py`'s
`--pp3_dir`) is the one genuinely optional extra, needing a per-camera
RawTherapee `.pp3` profile directory most captures will not have. See
`docs/pipeline.md` for the full walkthrough in order.

## Submodules

```bash
git submodule update --init --recursive
```

- `deps/camera-calibration` -- per-camera undistortion (offline_undistort.py)
- `deps/cumuli-trainer` -- the 4D gaussian splat trainer behind the `train4d` stage (Solipsist, private)
- `deps/cumuli-core` -- shared 4D library: interchange PLY, `.sogst` decoding, cameras, gsplat rendering, metrics (Solipsist, private)

## Conda environments

One env, `cumuli`, serves every stage. Provision it with
`bash scripts/setup_cumuli_env.sh` (see
[docs/environment.md](docs/environment.md), because a plain
`conda env create` misses the installs a yml cannot express). The
orchestrator dispatches every stage into it. Keypoint prediction additionally needs
`SAPIENS_CHECKPOINT_ROOT` set. ffmpeg/ffprobe must be on PATH for the
sync/extraction scripts, and rawtherapee-cli (or the RawTherapee
flatpak) only for `extract_synced_frames.py`'s optional `--pp3_dir`
color correction.

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

`--config` loads per-rig defaults (`SAPIENS_CHECKPOINT_ROOT`,
HLOC settings) from a JSON file so you do not have
to repeat them on every run. Copy [configs/example_rig.json](configs/example_rig.json)
and fill in your own paths. See [configs/README.md](configs/README.md).
Everything it sets can still be overridden on the command line.

Or run the individual stages by hand, which is useful the first time,
to understand what each one does:

```bash
python3 scripts/compute_sync_offsets.py <movies_dir> <out_dir> <ref_video.mp4>
python3 scripts/extract_synced_frames.py <movies_dir> <sync_offsets.json> <out_dir> <seconds>
```

Continue through `docs/pipeline.md` for the rest of the pipeline.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

```bash
pip install -r requirements-dev.txt
pytest tests/unit
```

The unit suite mocks out every wrapped tool, so it runs on any machine
(no GPU, no capture footage, no submodules), and it gates every pull
request. It cannot tell you whether a splat actually got better, though,
so changes to pose, mask, or training quality still need an end-to-end
before/after in the PR description.

External contributions need a signed Contributor License Agreement. We
use the Project Harmony agreements ([individual](cla/HA-CLA-I.md),
[entity](cla/HA-CLA-E.md)). You keep your copyright. The agreement lets us
license contributions commercially, while requiring that they stay
available under the project's public license too. Participation is
governed by the [Code of Conduct](CODE_OF_CONDUCT.md). Security issues go
to [SECURITY.md](SECURITY.md), not the public tracker.

## License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md).

This is a **source-available** license, not an OSI-approved open source
one. Any noncommercial purpose is permitted, and use by charitable
organizations, educational institutions, public research organizations,
and government institutions is permitted regardless of how that work is
funded, so academic and hobbyist use is unrestricted. Commercial use
requires a separate license: contact <jeff@solipsist.studio>.

A commercial license covers this repository's code only. It does not
grant rights to the third-party models the pipeline invokes. See
[Third-party model licenses](#third-party-model-licenses) below, and
note that the default `--feature_type superpoint` is itself restricted
to noncommercial research by Magic Leap.

### Third-party model licenses

Note that the external tools this pipeline wraps are **not**
covered by the PolyForm license: you obtain each of them directly from its
own project under its own terms. One of them restricts commercial use,
and that restriction is not lifted by buying a commercial license to
this pipeline:

- **SuperPoint** -- the default HLOC feature type
  (`--feature_type superpoint`) uses Magic Leap's SuperPoint detector
  and its `superpoint_v1.pth` weights, licensed "ACADEMIC OR NON-PROFIT
  ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY". Note that this holds
  even though HLOC and LightGlue are both Apache-2.0: the SuperPoint
  source file carries its own Magic Leap proprietary notice, and a
  repository-level Apache-2.0 label does not relicense it. Use
  `--feature_type aliked` (ALIKED, BSD-3-Clause, paired with LightGlue,
  Apache-2.0) for a fully permissive feature/matcher stack.

The keypoint stage uses **Sapiens2**, which is under Meta's own Sapiens2
License. Two Sapiens2 demo files, modified, ship
in this repository under `scripts/vendor/sapiens2/` with a copy of that
license; they are not covered by the PolyForm license. That
agreement *does* permit commercial use, but it carries an acceptable-use
list that prohibits, among other things, surveillance, **biometric
processing**, deepfakes and impersonation, and inferring sensitive
personal information without the required consents. It also grants Meta
audit rights and lets Meta amend the terms unilaterally. Because
"biometric processing" is not defined in the agreement and this is a
human capture pipeline, anyone deploying it commercially should read
the agreement and take their own legal advice rather than relying on
this summary.

The 4D training path no longer depends on OMG4's or Inria's
non-commercial code. The trainer, `deps/cumuli-trainer`, is Solipsist's
own code under the PolyForm Noncommercial license, the same terms as
this repository, and the shared library `deps/cumuli-core` is too. Both
render through gsplat, which is Apache-2.0.

Everything else wrapped by the pipeline (BiRefNet:
MIT code and weights, camera-calibration: MIT, HLOC: Apache-2.0,
LightGlue: Apache-2.0, ALIKED: BSD-3-Clause, COLMAP: BSD-3) is
permissively licensed. The mask and triangulation scripts contain code
ported from Diffuman4D (Apache-2.0); `scripts/vendor/diffuman4d/`
carries its license and the attribution notice.

Design references: `refine_sync_visual.py` is adapted from Open4DGS-Core
(MIT, https://open4dgs.asunemlabs.com), and its header carries that notice. The
sub-frame timing conventions and the blur-and-shift measure in `eval_render.py`
were informed by the documentation and a few source excerpts of Open4DGS-Core
and Open4DGS-Trigs (AGPL-3.0, https://github.com/ASU-NEM-Labs/open4dgs-trigs);
no Trigs code is included. `frame_motion.py` uses OpenCV's DIS optical flow
(Kroeger, Timofte, Dai and Van Gool, "Fast Optical Flow using Dense Inverse
Search", ECCV 2016).
