# Integration tests

`tests/integration/` runs the real pipeline -- real ffmpeg, real HLOC,
real Sapiens, real BiRefNet, real Brush training, no mocking -- against a
small committed fixture capture, and checks that the output is actually
good, not just that nothing crashed. This is the opposite of
`tests/unit/` (646 tests, one per script, everything mocked): unit tests
verify each script's own logic in isolation; this verifies the whole
chain produces correct real output together.

## What it checks

Runs the full 5-stage pipeline (sync -> production -> poses -> masks ->
branch) via `run_unified_pipeline.py`, exactly as a real capture would be
processed, then:

- **Structural checks per stage** -- every camera present at every stage,
  valid transforms, non-empty masks/point clouds/exports. Largely a second
  look at what `validate_stage_output.py` already checks automatically
  (the orchestrator runs it after every stage; if that fails, the whole
  run fails before these tests even get their own chance to look).
- **Regression checks against a golden baseline** -- pose refinement's
  median reprojection error and per-camera mask coverage, compared with
  margin against `tests/integration/fixtures/heidi_11cam/golden_baseline.json`.
  That file records numbers from real runs of this fixture that a human
  reviewed and confirmed looked right. This is *not* a universal quality
  bar -- different rigs/subjects have no comparable absolute number -- it
  only answers "did this change make our one reference capture's output
  measurably worse than before."
- **PSNR on the final trained splat** -- Brush's own `--eval-split-every`
  holds out one camera during training; `--eval-save-to-disk` renders it.
  PSNR is computed independently (not parsed from Brush's own log),
  masked to the subject only (not the full frame -- background pixels are
  trivially easy to match and would otherwise dominate the score), between
  that render and the real held-out photo, checked against a floor derived
  from the golden runs. This is a **proxy signal, not a measure of real
  production quality** -- holding a camera out of training is something
  production never does (more cameras only ever improves a real splat) --
  see `test_pipeline_end_to_end.py`'s module docstring for the full
  reasoning, including why 11 cameras was chosen over a smaller subset:
  it makes this proxy both more representative of production (holding out
  1 of 11 is a much smaller departure than 1 of 5) and far more
  run-to-run reproducible (Brush training isn't bit-exact -- a 5-camera
  prototype showed a 4.6dB swing between identical runs; 11 cameras cut
  that to 0.2dB).

The whole pipeline runs from scratch every time -- no resuming from a
cached partial run -- so a regression in any stage (not just training)
gets caught.

## Running it

```bash
pytest tests/integration -q
```

Needs, on the machine running it:
- An NVIDIA GPU (`nvidia-smi`)
- The `hloc`, `diffuman4d`, and `sapiens2` conda envs (see `docs/environment.md`)
- `ffmpeg` on PATH
- A compiled `brush_app` binary (same default path `run_unified_pipeline.py`
  uses, or set `VCP_BRUSH_APP`)
- Sapiens checkpoints (`SAPIENS_CHECKPOINT_ROOT`, or `VCP_SAPIENS_CHECKPOINT_ROOT`
  to point at a different location than the orchestrator's own default)

Missing any of these -> the suite **skips with a specific reason**, it
doesn't fail. This is deliberate: the same test file works identically on
any machine, doing the real thing where the environment supports it and
skipping cleanly everywhere else, rather than needing CI-side conditionals
to know when it's safe to run.

## CI

`.github/workflows/integration-tests.yml` runs this on PRs that touch
something that could plausibly change real pipeline output (see the
workflow's `paths:` filter) -- a docs-only PR doesn't pay the ~20-minute
run cost or contend for the one GPU runner. It targets a
`self-hosted, gpu`-labeled runner, since none of the above exists on
GitHub's own hosted runners. The workflow sets
`VCP_REQUIRE_PIPELINE_PREREQS=1`, which flips the skip-cleanly behavior
above into a hard failure: on the CI runner specifically, a missing
prerequisite means the runner's setup broke, and skipping there would
show up as a green check with zero tests run.

Registering an actual runner with that label is a separate, deliberate
decision (not bundled into this workflow) -- once this repo is public, a
self-hosted runner executes whatever code a triggering PR checks out, so
it needs its own sign-off (and likely a required-reviewer gate for PRs
from forks). When that happens, also enable branch protection on `main`
requiring this workflow's check to pass: without it, a direct push to
`main` bypasses the PR gate entirely but still triggers the rolling
baseline update below, letting unreviewed output become the new baseline.

## Rolling baseline

`golden_baseline.json`'s numbers aren't static forever. After a PR merges
to `main` (one touching the same pipeline-relevant paths the PR gate
watches -- a docs-only merge can't change pipeline output, so it doesn't
trigger a re-measure), `.github/workflows/update-integration-baseline.yml`
re-runs `tests/integration/update_golden_baseline.py`, which:

1. Re-runs the real pipeline against the same fixture/settings.
2. Confirms the run itself succeeded structurally (reusing
   `test_pipeline_end_to_end.py`'s own per-stage checks, imported directly
   so the two can't drift apart) -- a crashed/broken run writes nothing.
3. Overwrites `golden_baseline.json`'s 3 metric fields with this run's
   numbers and commits the change back to `main`.
4. Appends one line to `golden_baseline.json`'s sibling
   `baseline_history.jsonl` -- an append-only record of every *merged*
   result over time (not failed/abandoned PRs), for trend visibility and
   as a reference point if a change needs reverting.

The baseline is always the single most recent merged result, deliberately
**not** an average across runs -- averaging would dilute a real,
intentional quality improvement the same way it would dilute noise. This
step does not compare against the old baseline or fail on drift; that
gating already happened on the PR's own `pull_request`-triggered
`integration-tests.yml` run before the code ever reached `main` -- its
only job is capturing fresh ground truth for the *next* PR to be checked
against.

Margins (`REPROJECTION_ERROR_MARGIN_PX`, `MASK_COVERAGE_MARGIN`,
`PSNR_MARGIN_DB` in `test_pipeline_end_to_end.py`) are derived from a real
7-run variance study, not guessed -- see the comments above those
constants and `golden_baseline.json`'s own `reproducibility_check` field.
Mask coverage is genuinely deterministic run to run (bit-identical across
every run so far). Reprojection error is *usually* similarly stable but
is not actually deterministic -- a 500-iter follow-up study found HLOC's
own initial pose estimate varies run to run (a real ~2.3px spread before
refinement even runs), which occasionally, not usually, carries through
into the final refined number too (one run out of 18+ landed 0.25px off
the rest, still comfortably inside the 1.0px margin). See the "IMPORTANT
caveat about run-to-run stability" paragraph in
`test_pipeline_end_to_end.py`'s module docstring for the full finding --
it's a separate source of variance from Brush's, living in HLOC/pycolmap's
own bundle adjustment instead. PSNR remains by far the noisiest metric and
the one actually capable of catching a real regression.

**Why not just pin a seed to kill the PSNR variance?** It's already
pinned -- `brush_app --seed` defaults to 42, and every run behind the
numbers above used that same default, unchanged. The variance exists
*despite* an identical seed every time, because it comes from GPU
floating-point reduction order (non-associative float addition across
massively parallel threads), not from the RNG -- pinning the seed harder
can't fix that. See the "Why seed-pinning doesn't fix this" paragraph in
`test_pipeline_end_to_end.py`'s module docstring for the full mechanism.

## The fixture

`tests/integration/fixtures/heidi_11cam/`: all 11 real cameras, full-length
clips (~3s), from the validated `heidi_260521` capture. An earlier 5-camera
subset was tried first (smaller/faster) but retired after real testing
showed it was measurably worse on two fronts: HLOC's final reconstruction
failed to register one camera on a first attempt (not enough background
feature overlap with so few cameras), and the held-out-camera PSNR check
swung 4.6dB between two otherwise-identical runs. All 11 cameras fixed
both -- registration succeeded cleanly across repeated runs, and the PSNR
swing dropped to 0.2dB (see `golden_baseline.json`'s
`reproducibility_check` field for the numbers).
