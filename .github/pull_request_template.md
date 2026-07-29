<!--
Please read CONTRIBUTING.md first. Note in particular that external
contributions cannot be merged until the CLA is finalized — open the PR
anyway if you want to discuss the approach, we just cannot merge yet.
-->

## What this changes

<!-- One or two sentences. Link the issue this came from, if there is one. -->

## Why

<!-- What was wrong or missing. -->

## How it was verified

<!--
The unit suite (`pytest tests/unit`) needs no GPU and no footage — please
run it. But it mocks out every wrapped tool, so it cannot tell you whether
a splat, mask, or pose actually got better. For changes to pose
estimation, masking, or training quality, add the end-to-end evidence
below. See CONTRIBUTING.md.
-->

**Unit tests:**

```
paste `pytest tests/unit` output here
```

**End-to-end check** (required for pose, mask, or training changes;
write "n/a" if this is a docs or plumbing change):

<!--
  1. The exact command you ran.
  2. The data: how many cameras, what resolution, single frame or sequence.
  3. Before/after on the same input.
  4. A number. `run_pose_refinement.py --report_only` prints subject-space
     median reprojection error without modifying anything.

If you could not run this (no GPU, no multi-camera footage), say so
plainly. That is a fine thing to admit and a bad thing to imply.
-->

```
paste commands and results here
```

## Checklist

- [ ] `pytest tests/unit` passes locally
- [ ] I added or updated unit tests for this change
- [ ] I read [CONTRIBUTING.md](../blob/main/CONTRIBUTING.md)
- [ ] New files carry the SPDX + Required Notice header
- [ ] Module docstrings updated if I added or changed a flag
- [ ] I checked whether this affects `render_frame_sequence.py` as well as
      `run_unified_pipeline.py` (they share stage functions)
- [ ] I checked whether this changes a stage's output layout, which is the
      next stage's input contract
- [ ] `docs/pipeline.md` updated if pipeline order or data conventions changed
