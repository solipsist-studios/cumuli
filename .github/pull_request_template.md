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
There is no test suite, so this section is the evidence. Please give:
  1. The exact command you ran.
  2. The data: how many cameras, what resolution, single frame or sequence.
  3. What you compared against — before/after on the same input.
  4. A number, for anything touching poses, masks, or training.
     `run_pose_refinement.py --report_only` prints subject-space median
     reprojection error without modifying anything.

If you could not test this (no GPU, no multi-camera footage), say so
plainly here. That is a fine thing to admit and a bad thing to imply.
-->

```
paste commands and results here
```

## Checklist

- [ ] I read [CONTRIBUTING.md](../blob/main/CONTRIBUTING.md)
- [ ] New files carry the SPDX + Required Notice header
- [ ] Module docstrings updated if I added or changed a flag
- [ ] I checked whether this affects `render_frame_sequence.py` as well as
      `run_unified_pipeline.py` (they share stage functions)
- [ ] I checked whether this changes a stage's output layout, which is the
      next stage's input contract
- [ ] `docs/pipeline.md` updated if pipeline order or data conventions changed
