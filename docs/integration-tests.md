# Integration tests

`tests/integration/` runs the real pipeline (real ffmpeg, real HLOC,
real Sapiens, real BiRefNet, real 4D training, no mocking) against a
small committed fixture capture, and checks that the output is actually
good, not just that nothing crashed. This is the opposite of
`tests/unit/` (371 tests as of the 4D cutover, one file per script,
everything mocked): unit tests verify each script's own logic in
isolation, while this verifies the whole chain produces correct real
output together.

## What it checks

Runs the pipeline (sync -> production -> poses -> masks -> dataset4d,
plus train4d on a GPU machine) via `run_unified_pipeline.py`, exactly as
a real capture would be processed, then:

- **Structural checks per stage** -- every camera present at every stage,
  valid transforms, non-empty masks/point clouds/exports. Largely a second
  look at what `validate_stage_output.py` already checks automatically
  (the orchestrator runs it after every stage, and a failure there stops
  the run before these tests even get their own chance to look).
- **Regression checks against a golden baseline** -- pose refinement's
  median reprojection error and per-camera mask coverage, compared with
  margin against `tests/integration/fixtures/take01_11cam/golden_baseline.json`.
  That file records numbers from real runs of this fixture that a human
  reviewed and confirmed looked right. This is *not* a universal quality
  bar (different rigs/subjects have no comparable absolute number). It
  only answers "did this change make our one reference capture's output
  measurably worse than before."
- **Held-out metrics on the final 4D splat (GPU runs only)** -- the
  `train4d` stage scores the baked `.sogst` against a held-out camera
  with `eval_render.py` (PSNR/SSIM/LPIPS, written to `eval_4d.json`).
  The holdout is stereo-mate-aware: the eval camera's near-duplicate
  mates leave training WITHOUT being scored, because a held-out camera
  whose mate keeps training measures ~5.7 dB of pure leakage rather
  than quality. These are **proxy signals, not measures of real
  production quality**: production never holds a camera out (more
  cameras only ever improves a real splat), and absolute PSNR on a
  masked capture is dominated by background pixels, so the numbers are
  relative regression gates only. The `eval4d_*` fields in
  `golden_baseline.json` are informational until the first GPU-runner
  baseline lands.

The whole pipeline runs from scratch every time, with no resuming from
a cached partial run, so a regression in any stage (not just training)
gets caught.

## Running it

```bash
pytest tests/integration -q
```

Needs, on the machine running it:
- An NVIDIA GPU (`nvidia-smi`)
- The `cumuli` conda env, provisioned by `scripts/setup_cumuli_env.sh`
  (a plain `conda env create` lacks the trainer CUDA extensions the
  GPU-mode prereq check probes for, see `docs/environment.md`)
- The `deps/OMG4` submodule checked out (the vendored trainer)
- `ffmpeg` on PATH
- Sapiens checkpoints (`SAPIENS_CHECKPOINT_ROOT`, or `VCP_SAPIENS_CHECKPOINT_ROOT`
  to point at a different location than the orchestrator's own default)

Missing any of these -> the suite **skips with a specific reason**, it
does not fail. This is deliberate: the same test file works identically on
any machine, doing the real thing where the environment supports it and
skipping cleanly everywhere else, rather than needing CI-side conditionals
to know when it is safe to run.

### Running without a GPU

```bash
VCP_CPU_PIPELINE=1 pytest tests/integration -q
```

`VCP_CPU_PIPELINE=1` opts into running the pipeline without a GPU:
`_check_gpu()` stops requiring `nvidia-smi`, `pipeline_run` sets
`CUDA_VISIBLE_DEVICES=""` (hides any real GPU from PyTorch, so
masks/keypoints exercise the same CPU fallback a genuinely GPU-less
machine would use), and the run stops after the dataset build
(`--stop_after_stage dataset4d`). Training never runs in this mode:
the trainer's rasterizer is CUDA-only, and the `train4d` stage refuses
to start without a GPU by design.

**Hard-won findings that still apply** (the CPU-rendering findings for
the removed trainer are in git history):

1. (2026-07-30) **A real GitHub-hosted CI runner has only ~7.75GB RAM**,
   not the 16GB an earlier version of this doc assumed. This was
   confirmed directly via `[resmon]` telemetry added to
   `integration-tests-cpu.yml`.
   The 1b Sapiens checkpoint alone peaks at 6.5-7.6GB regardless of camera
   count (dominated by the model's fixed weight size, not image count),
   leaving so little headroom the runner swap-thrashed for ~40min then
   became unresponsive. CI-mode CPU testing therefore uses the smaller
   **0.4b checkpoint** (`CPU_SAPIENS_MODEL_SIZE`, measured 4.1GB peak,
   real margin) via `--sapiens_model_size`, wired through
   `predict_keypoints_2d.py`. Quality is unchecked in CPU mode anyway (see
   below), so the accuracy tradeoff costs nothing here. Separately, HLOC's
   SuperPoint feature extraction at the production `resize_max` of 4096
   peaks at 11.7GB RSS on CPU. It is also capped, to 1024
   (`CPU_HLOC_RESIZE_MAX`, `--hloc_resize_max`), the lowest setting that
   still reconstructs (512 fails outright).
2. (2026-07-30) **2 cameras is never enough for CI's reduced config**:
   every 2-camera pair tried made HLOC's COLMAP reconstruction fail
   outright ("Failed to create any sparse model"), confirmed via a fast
   standalone probe isolating just the reconstruction step across many
   camera-pair combinations, and every 3-camera trial succeeded. CI-mode
   CPU testing therefore uses a 3-camera subset (`CPU_REAL_CAMERAS`), not
   the full 11: a real functional minimum, not a scope choice. This also
   means CI's CPU path is structurally weaker than the full 11-camera
   fixture at registration robustness (see "The fixture" below on why 11
   was chosen for the GPU path). That is acceptable since CPU mode no
   longer checks reconstruction quality at all, only that the pipeline
   completes.
3. (2026-07-30) **`--sync_window 1` crashed outright**:
   `extract_synced_frames.py`'s `--window` has two different output
   layouts, not just "fewer frames": `window=1` writes flat files
   (`output_dir/0001.jpg`), any `window>1` writes per-instant subdirs
   (`output_dir/f0/`, `f1/`, ...). `run_unified_pipeline.py` assumed the
   subdir layout unconditionally. Fixed via a new `_instant_dirs()` helper
   that matches whichever layout is actually on disk. CI-mode CPU testing
   uses `CPU_SYNC_WINDOW = 1` (vs. the production default of 5) since
   Sapiens keypoint prediction, the dominant cost, runs once per
   candidate window frame. This cuts that cost 5x.

**What CPU mode validates:** structural completion through the dataset
build (`--stop_after_stage dataset4d`, the reduced config from the
findings above), plus the reprojection-error and mask-coverage golden
gates, which are pose/mask metrics and independent of training. The
held-out eval metrics require the GPU path. See
`test_pipeline_end_to_end.py`'s module docstring for the full
reasoning.

## CI

`.github/workflows/integration-tests-cpu.yml` runs the CPU path above on
**every** pull request, entirely on GitHub's own free hosted runners --
no self-hosted GPU runner required. It provisions the conda envs +
Sapiens checkpoints from scratch, each layer cached so only the first
run (or a cache-busting dependency change) pays the full cost. This is
the always-on safety net: it catches plumbing/logic regressions through
the dataset build on every PR. It produces no held-out eval numbers
(training never runs on CPU). See the CPU section above for the
reduced-config list and why.

`.github/workflows/integration-tests.yml` runs the full GPU-accurate path,
targeting a `self-hosted, gpu`-labeled runner since none of its
prerequisites exist on GitHub's own hosted runners. **Manual-dispatch
only** (`workflow_dispatch`, not `on: pull_request`): no runner with
that label has ever been registered, so an automatic trigger would just
queue forever on every PR with nothing to ever pick it up. Real GPU
validation happens locally today (`pytest tests/integration -q`, see
above) instead. The workflow sets `VCP_REQUIRE_PIPELINE_PREREQS=1`, which
flips the skip-cleanly behavior above into a hard failure: on a real CI
runner, a missing prerequisite means the runner's setup broke, and
skipping there would show up as a green check with zero tests run.

Registering an actual runner with that label is a separate, deliberate
decision (not bundled into this workflow). Once this repo is public, a
self-hosted runner executes whatever code a triggering PR checks out, so
it needs its own sign-off (and likely a required-reviewer gate for PRs
from forks). When that happens, switch the trigger back to `pull_request`
(with a `paths:` filter, so a docs-only PR does not pay the ~20-minute run
cost or contend for the one GPU runner) and enable branch protection on
`main` requiring this workflow's check to pass: without it, a direct push
to `main` bypasses the PR gate entirely but still triggers the rolling
baseline update below, letting unreviewed output become the new baseline.

## Rolling baseline

`golden_baseline.json`'s numbers are not static forever. After a PR
merges to `main` (one touching the same pipeline-relevant paths the PR
gate watches, because a docs-only merge cannot change pipeline output
and so does not trigger a re-measure),
`.github/workflows/update-integration-baseline.yml`
re-runs `tests/integration/update_golden_baseline.py`, which:

1. Re-runs the real pipeline against the same fixture/settings.
2. Confirms the run itself succeeded structurally (reusing
   `test_pipeline_end_to_end.py`'s own per-stage checks, imported directly
   so the two cannot drift apart). A crashed/broken run writes nothing.
3. Overwrites `golden_baseline.json`'s metric fields (reprojection
   error, mask coverage, and the `eval4d_*` metrics once the GPU runner
   populates them) with this run's numbers and commits the change back
   to `main`.
4. Appends one line to `golden_baseline.json`'s sibling
   `baseline_history.jsonl`, an append-only record of every *merged*
   result over time (not failed/abandoned PRs), for trend visibility and
   as a reference point if a change needs reverting.

The baseline is always the single most recent merged result, deliberately
**not** an average across runs. Averaging would dilute a real,
intentional quality improvement the same way it would dilute noise. This
step does not compare against the old baseline or fail on drift. That
gating already happened on the PR's own `pull_request`-triggered
`integration-tests.yml` run before the code ever reached `main`, so this
step's only job is capturing fresh ground truth for the *next* PR to be
checked against.

Margins (`REPROJECTION_ERROR_MARGIN_PX`, `MASK_COVERAGE_MARGIN` in
`test_pipeline_end_to_end.py`) are derived from a real 7-run variance
study, not guessed. See the comments above those constants and
`golden_baseline.json`'s own `reproducibility_check` field. The
`eval4d_*` margins start deliberately wide (PSNR -2 dB) until an
equivalent variance study exists for the 4D trainer.
Mask coverage is genuinely deterministic run to run (bit-identical across
every run so far). Reprojection error is *usually* similarly stable but
is not actually deterministic. A 500-iter follow-up study found HLOC's
own initial pose estimate varies run to run (a real ~2.3px spread before
refinement even runs), which occasionally, not usually, carries through
into the final refined number too (one run out of 18+ landed 0.25px off
the rest, still comfortably inside the 1.0px margin). See the "IMPORTANT
caveat about run-to-run stability" paragraph in
`test_pipeline_end_to_end.py`'s module docstring for the full finding --
it is a separate source of variance from training's, living in HLOC/pycolmap's
own bundle adjustment instead. Held-out PSNR remains by far the noisiest metric and the one actually
capable of catching a real regression.

**Why not just pin a seed to kill the PSNR variance?** Seed pinning
cannot fix it: the variance comes from GPU floating-point reduction
order (non-associative float addition across massively parallel
threads), not from the RNG. The removed trainer demonstrated this with
an identical seed on every run, and the mechanism applies equally to
the 4D trainer's CUDA rasterizer.

## The fixture

`tests/integration/fixtures/take01_11cam/`: all 11 real cameras, full-length
clips (~3s), from the validated `take01` capture (the same capture
`docs/pipeline.md` uses as its worked example). An earlier 5-camera
subset was tried first (smaller/faster) but retired after real testing
showed it was measurably worse on two fronts: HLOC's final reconstruction
failed to register one camera on a first attempt (not enough background
feature overlap with so few cameras), and the held-out-camera PSNR check
swung 4.6dB between two otherwise-identical training runs. All 11
cameras fixed both: registration succeeded cleanly across repeated
runs, and the PSNR swing dropped to 0.2dB (see `golden_baseline.json`'s
`reproducibility_check` field for the numbers, measured with the
previous trainer). The stability argument is about camera count, not
the trainer.
