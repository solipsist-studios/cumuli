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
  margin against `tests/integration/fixtures/take01_11cam/golden_baseline.json`.
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
- The `hloc`, `diffuman4d`, `sapiens2`, and `queen` conda envs (see `docs/environment.md`)
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

### Running without a GPU

```bash
VCP_ALLOW_CPU_RENDERING=1 VCP_BRUSH_APP=/path/to/source-built/brush-cli pytest tests/integration -q
```

(`VCP_BRUSH_APP` must point at a `brush-cli` built from source -- see
finding 2 below; the pinned commit `integration-tests-cpu.yml` builds is
a known-good one to match locally.)

`VCP_ALLOW_CPU_RENDERING=1` opts into running the whole pipeline without a
real GPU at all -- `_check_gpu()` stops requiring `nvidia-smi`, and
`pipeline_run` forces genuine software rendering plus `CUDA_VISIBLE_DEVICES=""`
(hides any real GPU from PyTorch too, so masks/keypoints exercise the same
CPU fallback a genuinely GPU-less machine would use, even when testing this
locally on a machine that does have a GPU).

**Hard-won findings**, roughly in the order they were hit:

1. (2026-07-28) `WGPU_BACKEND=gl` **alone does not avoid the GPU** on a
   machine with an NVIDIA driver installed -- it only picks the OpenGL
   API, not the implementation; glvnd still hands OpenGL to the NVIDIA
   stack. Forcing genuine software rendering also requires
   `LIBGL_ALWAYS_SOFTWARE=1` and pointing `__EGL_VENDOR_LIBRARY_FILENAMES`
   at Mesa's own EGL ICD (see `pipeline_run`'s env construction in
   `test_pipeline_end_to_end.py` for the exact recipe). An earlier claim
   in this doc that a fast, good-PSNR run had validated `WGPU_BACKEND=gl`
   alone was based on a run that, it turned out, had silently used the
   real NVIDIA GPU the whole time.
2. (2026-07-28) The **released `brush_app` v0.3.0 binary cannot run
   CPU-only at all**, independent of the above -- on genuine Mesa llvmpipe
   it hits (a) a panic launching GPU subgroup/"plane" kernels llvmpipe
   doesn't support, then (b) a `BufferTooBig` panic at the production 4096
   resolution, and even past both of those, (c) a real, non-deterministic
   deadlock: one thread spins forever on a `spin::Mutex` inside
   burn-fusion (`MutexFusionClient::register_tensor`) that another, parked
   thread never releases -- confirmed via instruction-pointer sampling,
   not inferred. That code path has since been removed from burn's `main`
   branch (the `client/mutex.rs` module doesn't exist there anymore).
   **CPU-rendering mode therefore requires `brush-cli` built from source**
   at a commit newer than that fix (`VCP_BRUSH_APP` pointed at the build)
   -- the release binary will not work here. `--max-resolution` is also
   capped lower for CPU mode (`--brush_max_resolution`, wired through
   `run_unified_pipeline.py`) since llvmpipe rejects the buffer sizes 4096
   needs.
3. (2026-07-30) **A real GitHub-hosted CI runner has only ~7.75GB RAM**,
   not the 16GB an earlier version of this doc assumed -- confirmed
   directly via `[resmon]` telemetry added to `integration-tests-cpu.yml`.
   The 1b Sapiens checkpoint alone peaks at 6.5-7.6GB regardless of camera
   count (dominated by the model's fixed weight size, not image count),
   leaving so little headroom the runner swap-thrashed for ~40min then
   became unresponsive. CI-mode CPU testing therefore uses the smaller
   **0.4b checkpoint** (`CPU_SAPIENS_MODEL_SIZE`, measured 4.1GB peak --
   real margin) via `--sapiens_model_size`, wired through
   `predict_keypoints_2d.py`. Quality is unchecked in CPU mode anyway (see
   below), so the accuracy tradeoff costs nothing here. Separately, HLOC's
   SuperPoint feature extraction at the production `resize_max` of 4096
   peaks at 11.7GB RSS on CPU -- also capped, to 1024 (`CPU_HLOC_RESIZE_MAX`,
   `--hloc_resize_max`), the lowest setting that still reconstructs (512
   fails outright).
4. (2026-07-30) **2 cameras is never enough for CI's reduced config** --
   every 2-camera pair tried made HLOC's COLMAP reconstruction fail
   outright ("Failed to create any sparse model"), confirmed via a fast
   standalone probe isolating just the reconstruction step across many
   camera-pair combinations; every 3-camera trial succeeded. CI-mode CPU
   testing therefore uses a 3-camera subset (`CPU_REAL_CAMERAS`), not the
   full 11 -- a real functional minimum, not a scope choice. This also
   means CI's CPU path is structurally weaker than the full 11-camera
   fixture at registration robustness (see "The fixture" below on why 11
   was chosen for the GPU path) -- acceptable since CPU mode no longer
   checks reconstruction quality at all, only that the pipeline completes.
5. (2026-07-30) **`--sync_window 1` crashed outright** --
   `extract_synced_frames.py`'s `--window` has two different output
   layouts, not just "fewer frames": `window=1` writes flat files
   (`output_dir/0001.jpg`), any `window>1` writes per-instant subdirs
   (`output_dir/f0/`, `f1/`, ...). `run_unified_pipeline.py` assumed the
   subdir layout unconditionally. Fixed via a new `_instant_dirs()` helper
   that matches whichever layout is actually on disk. CI-mode CPU testing
   uses `CPU_SYNC_WINDOW = 1` (vs. the production default of 5) since
   Sapiens keypoint prediction -- the dominant cost -- runs once per
   candidate window frame; this cuts that cost 5x.
6. (2026-07-31) **A bare GitHub-hosted runner has zero graphics packages
   installed at all** -- not even a software rasterizer. Getting findings
   1-2 right (the env vars, a source-built binary) still panicked with "No
   possible adapter available for backend" the first time this actually
   ran on real CI, because there was no llvmpipe (OpenGL) or lavapipe
   (Vulkan) implementation present for those env vars to select in the
   first place -- this only surfaced on CI since every earlier validation
   ran on a dev machine that already had *some* GL/Vulkan stack installed.
   Fixed by installing `libgl1-mesa-dri` (llvmpipe) and
   `mesa-vulkan-drivers` (lavapipe) in `integration-tests-cpu.yml` --
   installing both rather than resolving exactly which one `cubecl-wgpu`
   wants first.

**What actually gets validated in CPU mode differs by context.** A full
local run (2026-07-28/29, all 11 cameras, the production sync window and
checkpoint size, ~1h20-25min) validated the software-rendering technique
itself end-to-end -- reprojection error and mask coverage matched the GPU
golden baseline exactly, masked PSNR landed within 0.5dB. But
`integration-tests-cpu.yml`'s actual CI config is the much smaller one
from findings 3-5 above (3 cameras, sync window 1, 0.4b checkpoint,
`resize_max` 1024, `CPU_TOTAL_TRAIN_ITERS` instead of a full training run)
-- and **every golden-baseline quality check (reprojection error, mask
coverage, PSNR) is skipped entirely in that mode**, not just
"not comparable" -- CI-mode CPU testing validates structural completion
only ("did every stage produce valid, non-degenerate output"), never
reconstruction quality. See `test_pipeline_end_to_end.py`'s module
docstring for the full reasoning.

## CI

`.github/workflows/integration-tests-cpu.yml` runs the CPU-rendering path
above on **every** pull request, entirely on GitHub's own free hosted
runners -- no self-hosted GPU runner required. It builds `brush-cli` from
source (pinned to a specific validated commit, not a moving `main`
target -- see the workflow's own comments for why) and provisions the
conda envs + Sapiens checkpoints from scratch, each layer cached so only
the first run (or a cache-busting dependency change) pays the full cost.
This is the always-on safety net: it catches plumbing/logic regressions
across all 5 stages on every PR. It does not produce a PSNR number at all
(no camera is held out of the reduced 3-camera training set) or run any
other golden-baseline comparison -- see the CPU-rendering section above
for the full reduced-config list and why.

`.github/workflows/integration-tests.yml` runs the full GPU-accurate path,
targeting a `self-hosted, gpu`-labeled runner since none of its
prerequisites exist on GitHub's own hosted runners. **Manual-dispatch
only** (`workflow_dispatch`, not `on: pull_request`) -- no runner with
that label has ever been registered, so an automatic trigger would just
queue forever on every PR with nothing to ever pick it up. Real GPU
validation happens locally today (`pytest tests/integration -q`, see
above) instead. The workflow sets `VCP_REQUIRE_PIPELINE_PREREQS=1`, which
flips the skip-cleanly behavior above into a hard failure: on a real CI
runner, a missing prerequisite means the runner's setup broke, and
skipping there would show up as a green check with zero tests run.

Registering an actual runner with that label is a separate, deliberate
decision (not bundled into this workflow) -- once this repo is public, a
self-hosted runner executes whatever code a triggering PR checks out, so
it needs its own sign-off (and likely a required-reviewer gate for PRs
from forks). When that happens, switch the trigger back to `pull_request`
(with a `paths:` filter, so a docs-only PR doesn't pay the ~20-minute run
cost or contend for the one GPU runner) and enable branch protection on
`main` requiring this workflow's check to pass: without it, a direct push
to `main` bypasses the PR gate entirely but still triggers the rolling
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

`tests/integration/fixtures/take01_11cam/`: all 11 real cameras, full-length
clips (~3s), from the validated `take01` capture (the same capture `docs/pipeline.md` uses as its worked example). An earlier 5-camera
subset was tried first (smaller/faster) but retired after real testing
showed it was measurably worse on two fronts: HLOC's final reconstruction
failed to register one camera on a first attempt (not enough background
feature overlap with so few cameras), and the held-out-camera PSNR check
swung 4.6dB between two otherwise-identical runs. All 11 cameras fixed
both -- registration succeeded cleanly across repeated runs, and the PSNR
swing dropped to 0.2dB (see `golden_baseline.json`'s
`reproducibility_check` field for the numbers).
