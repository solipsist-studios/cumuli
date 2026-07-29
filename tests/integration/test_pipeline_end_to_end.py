"""
tests/integration/test_pipeline_end_to_end.py

Real, unmocked pipeline test: runs the full 5-stage volumetric capture
pipeline (sync -> production -> poses -> masks -> branch) against the
committed heidi_11cam fixture (tests/integration/fixtures/heidi_11cam/,
all 11 real cameras -- an earlier 5-camera prototype was retired after
comparison testing showed 11 cameras is both more robust (clean HLOC
registration every time; a 5-camera subset failed to register one camera
on a first attempt) and gives a far more reproducible PSNR number, see
below), then checks quality, not just "didn't crash":

  - validate_stage_output.py's own structural sanity checks (already run
    by run_unified_pipeline.py itself -- no --no_validate here).
  - median pose-refinement reprojection error and per-camera mask coverage
    compared against a golden baseline recorded from one human-reviewed
    run of this same fixture (golden_baseline.json) -- a regression guard
    against THIS fixture's own history, not a universal quality bar (see
    that file's own comment for why).
  - the final trained splat's PSNR, computed independently from Brush's
    own held-out eval render (--eval-split-every/--eval-save-to-disk)
    against the real held-out photo, MASKED to the subject only (not the
    full frame -- see PSNR_MARGIN_DB below for why).

IMPORTANT caveat about the PSNR check, confirmed with user: training with
one camera held out (required to get any held-out render at all) is NOT
how this pipeline is ever actually used in production -- production always
trains on every available camera, since splatting quality improves with
more views, never fewer. This PSNR number is a PROXY, not a measure of
real production splat quality: its job is narrower -- "can the model
reconstruct a view it never saw", which is sensitive to the same upstream
breakage (bad sync, bad masks, bad poses) a real bug would cause, even
though the training configuration itself doesn't match production. With
11 cameras, holding one out is a much smaller departure from production
(~9% of views) than the original 5-camera prototype (20% of views) --
part of why 11 was chosen over 5, alongside the reproducibility finding
below.

Also masked to the subject only: an early pass computed PSNR over the
full frame and got a deceptively good-looking score -- ~98% of every
pixel in this fixture is plain black background, trivially easy to
match, so an unmasked score is dominated by background agreement, not
subject fidelity.

IMPORTANT caveat about run-to-run stability: mask generation is genuinely
deterministic (confirmed empirically -- bit-identical to 4 decimal places
across 18+ repeat runs of the same fixture/settings, no exceptions).
Reprojection error is NOT actually deterministic, despite earlier language
here claiming otherwise -- a 500-iter, 5-run study found HLOC's own
initial pose estimate (multiframe_sfm.py, before the keypoint-refinement
step even runs) varies by a real ~2.3px across identical runs (25.71px to
28.00px), almost certainly RANSAC's internal randomness and/or GPU float
non-determinism inside pycolmap's own bundle adjustment -- the same
general category of issue as Brush's non-determinism below, just in a
different part of the pipeline. The keypoint-guided refinement step
usually (not always) absorbs this variation and lands at nearly the same
final number regardless of starting point -- across 18+ runs at 1000/250
iters the final value stayed within a 0.06-0.09px range -- but that same
500-iter study saw one run refine to 5.35px against every other run's
5.5-5.6px cluster, a real ~0.25px deviation traced directly to that run
having the worst (28.00px) initial HLOC estimate of the batch. Still
comfortably inside REPROJECTION_ERROR_MARGIN_PX's current 1.0px margin,
so nothing here changes what's shipped -- but "deterministic" was the
wrong word for this metric; "usually very stable, occasionally not" is
the accurate one. Brush training is NOT bit-reproducible either,
even with a fixed seed -- almost certainly non-bit-exact GPU float
reductions in its differentiable rasterizer, compounding over training
steps. A 5-camera prototype fixture showed this clearly: identical
settings, two runs, masked PSNR of 15.1dB and 10.5dB -- a 4.6dB swing
from nothing but re-running. Switching to all 11 cameras (see
golden_baseline.json's own "reproducibility_check" field) cut that swing
to 0.2dB, which is why PSNR_MARGIN_DB below can reasonably be a tight-ish
hard gate rather than purely informational.

Why seed-pinning doesn't fix this: `brush_app --seed` defaults to 42
(confirmed against `brush_app --help`'s own `[default: 42]`), and every
run referenced above -- the 5-camera prototype, the 2-run golden capture,
and the full 7-run variance study -- used that same default value,
unchanged, every time. So the seed was ALREADY effectively pinned across
every single one of those runs, and PSNR still varied. The seed only
fixes which pseudo-random numbers get drawn; it has no control over the
order in which the GPU actually performs floating-point arithmetic.
Brush's rasterizer does massively parallel reduction operations (summing
gradients across thousands of GPU threads), and floating-point addition
isn't associative -- `(a+b)+c` can differ by a tiny rounding amount from
`a+(b+c)` -- so a different thread-scheduling order on the same hardware
with the same code can produce a slightly different sum. Individually
negligible, but compounding over hundreds/thousands of training steps,
these differences add up to the measurable PSNR swings seen above. Seed-
pinning harder cannot fix non-determinism that comes from hardware
execution order rather than the RNG.

See conftest.py for how this suite skips cleanly (not fails) on a machine
that doesn't have the GPU/conda-envs/brush_app/checkpoints this needs. The
whole pipeline runs fresh from raw video on every test session -- nothing
is cached/resumed across runs, so a regression anywhere in the chain gets
caught (see docs/integration-tests.md).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import run_unified_pipeline as unified
from baseline_metrics import held_out_psnr, mask_coverage, parse_reprojection_error
from validate_stage_output import ply_vertex_count

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

TARGET_TIME = "1500ms"
TOTAL_TRAIN_ITERS = 1000  # small on purpose -- see docs/integration-tests.md
EVAL_SPLIT_EVERY = 11  # holds out exactly 1 of the fixture's 11 cameras
CPU_MAX_RESOLUTION = 1024  # CPU-rendering mode only -- llvmpipe rejects the buffer
                            # sizes the production 4096 needs (BufferTooBig), and 16x
                            # the pixels under software rendering would be impractically
                            # slow. GPU runs keep the orchestrator's 4096 default.
PIPELINE_TIMEOUT_S_GPU = 3000  # 50min -- comfortably above the ~15-20min real GPU runs take,
                                 # below integration-tests.yml's 60-min job timeout so this cleaner
                                 # failure fires first. brush_app has genuinely hung on this
                                 # project's own dev machine before; without this, a hang stalls
                                 # pytest forever on a local run.
PIPELINE_TIMEOUT_S_CPU = 7200  # 2h -- CPU rendering is dramatically slower than GPU, not just for
                                 # Brush training itself: Sapiens pose estimation (predict_keypoints_2d.py)
                                 # is CPU-bound regardless of Brush and runs once per sync-correction
                                 # candidate frame batch (several) plus once for the real production
                                 # frames, so most of the added wall time here is Sapiens, not Brush.
                                 # Observed empirically (2026-07-28): sync-correction alone (candidate
                                 # scoring, before the main pipeline even starts) took >45min on a
                                 # 32-core dev machine and hadn't finished -- comfortably below
                                 # integration-tests-cpu.yml's job timeout so this cleaner failure
                                 # fires first there too.

REAL_CAMERAS = ["0001", "0002", "0003", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012"]

# brush_app's --eval-split-every 11 on this 11-image dataset holds out
# whichever image build_colmap_sparse.py wrote first as flat label "00" --
# for this fixture that's real camera 0001 (see baseline_metrics.flat_label_for(),
# which reads the pipeline's own camera_label_map.json rather than
# assuming a fixed sort order).
HELD_OUT_REAL_CAMERA = "0001"

with open(Path(__file__).resolve().parent / "fixtures" / "heidi_11cam" / "golden_baseline.json") as f:
    GOLDEN = json.load(f)

# Margins below are derived from a real 7-run variance study (identical
# fixture/settings, --total_train_iters 1000, re-run 7 times back to back)
# -- see ~/vcp_variance_study_report.md and
# INTEGRATION_TESTS_COMMIT_PLAN.md's rolling-baseline entry. Mask coverage
# is genuinely deterministic (bit-identical across every run so far), so
# its margin is tight -- not a strong regression signal on its own given
# it never varies naturally, but a wide margin would make it check almost
# nothing at all. Reprojection error is USUALLY similarly tight (0.06px
# range across the main 7-run study) but is not actually deterministic --
# a separate 500-iter study found HLOC's own initial pose estimate varies
# run to run, which occasionally (not usually) propagates into the final
# refined number too (one run out of 18+ landed 0.25px off the rest) --
# see the module docstring above for the full finding. Its 1.0px margin
# comfortably covers that observed case, so it's kept the same. PSNR is
# still the noisiest one by far (Brush's
# differentiable rasterizer isn't bit-reproducible even at a fixed seed --
# confirmed empirically, not fixable by seed-pinning) and is the primary
# metric this suite relies on to catch a real regression; its margin is
# picked with real headroom (~9x the observed 0.49dB range) so it doesn't
# flake on natural noise while still tripping on a multi-dB real break
# (the retired 5-camera prototype's known-bad case swung 4.56dB).
REPROJECTION_ERROR_MARGIN_PX = 1.0  # observed range across 7 runs: 0.06px (baseline 5.57px)
MASK_COVERAGE_MARGIN = 0.005  # +/- absolute fraction slack per camera; observed: bit-identical
                               # across 7 runs/11 cameras, and true values are only ~0.01-0.023,
                               # so anything close to the old 0.05 margin checked almost nothing
PSNR_MARGIN_DB = 1.5  # subtracted from the golden masked PSNR for the floor; observed range across
                       # 7 runs: 0.49dB (stdev 0.17dB, mean 22.17dB) -- note cutting
                       # --total_train_iters for a faster test roughly doubles this variance
                       # (0.91dB range on a 4-run 250-iter sample), so don't reduce iteration
                       # count without re-widening this margin


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory, sapiens_checkpoint_root, brush_app, fixture_dir, allow_cpu_rendering):
    """Runs the real orchestrator once, for the whole test session, as an
    actual subprocess (matching how anyone would really invoke it -- the
    orchestrator itself dispatches each stage into its own conda env via
    subprocess regardless of how it's launched). Every test function below
    reads this one run's output; nothing here re-runs the pipeline."""
    out_dir = tmp_path_factory.mktemp("vcp_integration_run")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "run_unified_pipeline.py"),
        "--video_dir", str(fixture_dir / "movies"),
        "--calib_dir", str(fixture_dir / "calibration_pkls"),
        "--out_dir", str(out_dir),
        "--target_time", TARGET_TIME,
        "--sapiens_checkpoint_root", sapiens_checkpoint_root,
        "--brush_app", brush_app,
        "--total_train_iters", str(TOTAL_TRAIN_ITERS),
        "--export_every", str(TOTAL_TRAIN_ITERS),
        "--eval_split_every", str(EVAL_SPLIT_EVERY),
        "--eval_save_to_disk",
    ]

    env = dict(os.environ)
    if allow_cpu_rendering:
        # Force brush onto Mesa's llvmpipe software rasterizer -- ALL
        # THREE of these matter, discovered the hard way (2026-07-28):
        #
        # WGPU_BACKEND=gl alone only picks the OpenGL API, NOT the
        # implementation -- on a machine with an NVIDIA driver installed,
        # glvnd hands OpenGL to the NVIDIA stack and training silently
        # runs on the real GPU (verified: the process held /dev/nvidia*
        # fds and 281MiB of GPU memory while nvidia-smi read 0%
        # utilization -- an earlier claim here that WGPU_BACKEND=gl had
        # been "empirically verified" to avoid the GPU was based on
        # exactly that misreading). LIBGL_ALWAYS_SOFTWARE requests
        # software rendering from Mesa, and pointing
        # __EGL_VENDOR_LIBRARY_FILENAMES at Mesa's own ICD stops glvnd
        # from ever offering the NVIDIA implementation -- set only when
        # that file exists (dual-driver dev machines); a real GPU-less CI
        # runner has no NVIDIA ICD to exclude.
        #
        # CUDA_VISIBLE_DEVICES="" additionally hides any real GPU from
        # PyTorch (masks/keypoints stages) -- without it, a machine that
        # DOES have a GPU (unlike the CI runner this is actually for)
        # would silently keep using it for those stages, making a local
        # test here unrepresentative of what a real GPU-less runner sees.
        env["WGPU_BACKEND"] = "gl"
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"
        mesa_icd = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
        if mesa_icd.is_file():
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(mesa_icd)
        env["CUDA_VISIBLE_DEVICES"] = ""
        # Headless (no viewer window -- there's no display on a CI
        # runner) and 1024 res instead of the production 4096: llvmpipe
        # rejects the GL buffer sizes 4096 needs (BufferTooBig), and
        # software-rendering 16x the pixels would be impractically slow.
        # This makes the CPU profile's PSNR NOT directly comparable to
        # the golden baseline recorded at 4096 -- see test_psnr_meets_baseline.
        cmd += ["--no_viewer", "--brush_max_resolution", str(CPU_MAX_RESOLUTION)]

    timeout_s = PIPELINE_TIMEOUT_S_CPU if allow_cpu_rendering else PIPELINE_TIMEOUT_S_GPU
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                                timeout=timeout_s)
        returncode, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        # e.stdout/e.stderr can be raw bytes here even though text=True was
        # passed to subprocess.run above -- on POSIX, a timeout interrupts
        # Popen.communicate() before its normal text-decoding step runs, so
        # whatever was buffered so far is packaged into the exception as-is.
        # Discovered the hard way: a run that genuinely timed out (the exact
        # case this handler exists for) crashed here instead with an
        # unrelated "can't concat str to bytes" TypeError, masking the real
        # timeout as a confusing pytest ERROR.
        returncode = -1
        stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        stderr += f"\npipeline killed after exceeding its timeout ({timeout_s}s)"

    log_path = ARTIFACTS_DIR / "pipeline_run.log"
    log_path.write_text(stdout + "\n" + stderr)

    L = unified.build_layout(out_dir)
    return {"out_dir": out_dir, "L": L, "returncode": returncode, "log": stdout}


def test_pipeline_completes_successfully(pipeline_run):
    assert pipeline_run["returncode"] == 0, (
        f"run_unified_pipeline.py exited {pipeline_run['returncode']} -- "
        f"see {ARTIFACTS_DIR / 'pipeline_run.log'} for the full log"
    )


# --------------------------------------------------------------------------
# Per-stage structural checks. run_unified_pipeline.py already runs
# validate_stage_output.py's own sanity checks after every stage (no
# --no_validate above), so test_pipeline_completes_successfully already
# proves those passed. These are a second, independent look at the
# specific files each stage promises, from the test's own perspective
# rather than trusting the pipeline's self-report.
# --------------------------------------------------------------------------

def test_sync_stage_produced_offsets_for_every_camera(pipeline_run):
    L = pipeline_run["L"]
    data = json.loads(L["sync_offsets"].read_text())
    offset_cams = sorted(Path(name).stem for name in data["offsets"])
    assert offset_cams == sorted(REAL_CAMERAS)
    assert L["sync_grid"].is_file()


def test_production_stage_undistorted_every_camera(pipeline_run):
    """By the time the full run finishes, run_hloc.py's
    restructure_flat_to_percam() side effect (stage 'poses') has already
    turned production_undist/ from flat <camera>.jpg files into
    Camera_<id>/0000.jpg subdirs -- checking for the pre-restructure flat
    layout here would silently check the wrong thing once every stage has
    run (caught by running this test against real output before trusting
    it -- see INTEGRATION_TESTS_COMMIT_PLAN.md)."""
    L = pipeline_run["L"]
    found = sorted(p.name for p in L["production_undist"].glob("Camera_*") if p.is_dir())
    expected = sorted(f"Camera_{c}" for c in REAL_CAMERAS)
    assert found == expected
    for cam_dir in expected:
        assert any((L["production_undist"] / cam_dir).glob("*.jpg")), f"{cam_dir} has no frame"


def test_poses_stage_transforms_cover_every_camera(pipeline_run):
    L = pipeline_run["L"]
    data = json.loads(L["transforms_refined"].read_text())
    labels = sorted(fr["camera_label"] for fr in data["frames"])
    assert labels == sorted(f"Camera_{c}" for c in REAL_CAMERAS)


def test_poses_stage_reprojection_error_within_baseline(pipeline_run):
    """Parses run_pose_refinement.py's own printed 'Median residual: X ->
    Y' line (there's no machine-readable report file for this) and checks
    it against golden_baseline.json's recorded value, with slack -- a
    regression guard for THIS fixture, not an absolute accuracy bar (see
    golden_baseline.json)."""
    final_median_px = parse_reprojection_error(pipeline_run["log"])
    assert final_median_px is not None, "could not find pose-refinement's 'Median residual: ... -> ...px' line in the run log"
    baseline = GOLDEN["median_reprojection_error_px"]
    assert final_median_px <= baseline + REPROJECTION_ERROR_MARGIN_PX, (
        f"pose refinement's median reprojection error ({final_median_px:.2f}px) drifted more than "
        f"{REPROJECTION_ERROR_MARGIN_PX}px past the golden baseline ({baseline:.2f}px)"
    )


def test_masks_stage_coverage_within_baseline(pipeline_run):
    """Cleaned-mask coverage fraction per camera vs. golden_baseline.json,
    with slack -- catches a mask silhouette shrinking/ballooning even
    though it'd still pass validate_stage_output.py's much looser
    0.5%-90% sanity range."""
    L = pipeline_run["L"]
    for cam in REAL_CAMERAS:
        frac = mask_coverage(L, cam)
        baseline = GOLDEN["mask_coverage"][cam]
        assert abs(frac - baseline) <= MASK_COVERAGE_MARGIN, (
            f"{cam}: cleaned mask coverage {frac:.3f} drifted more than {MASK_COVERAGE_MARGIN} "
            f"from the golden baseline ({baseline:.3f})"
        )


def test_branch_stage_exported_a_nonempty_splat(pipeline_run):
    L = pipeline_run["L"]
    plys = sorted(L["brush_output"].glob("*.ply"))
    assert plys, f"no .ply exported under {L['brush_output']}"
    newest = max(plys, key=lambda p: p.stat().st_mtime)
    assert ply_vertex_count(newest) >= 1

    # Copy the trained splat out to the artifacts dir for CI upload. The
    # mkdir matters: this function is also called by
    # update_golden_baseline.py's structural gate, where the pipeline_run
    # fixture (which normally creates ARTIFACTS_DIR) never runs -- and the
    # dir is gitignored, so it doesn't exist on a fresh checkout.
    import shutil
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(newest, ARTIFACTS_DIR / newest.name)


def test_psnr_meets_baseline(pipeline_run):
    """Masked PSNR between Brush's held-out eval render (camera
    HELD_OUT_REAL_CAMERA, excluded from training via --eval_split_every)
    and the real photo from that same camera -- see the module docstring
    for why this is a proxy signal, not a production-quality measurement,
    and why it's masked to the subject rather than the full frame."""
    L = pipeline_run["L"]
    score, render_path = held_out_psnr(L, TOTAL_TRAIN_ITERS, HELD_OUT_REAL_CAMERA)
    baseline = GOLDEN["psnr_db"]

    # Report + render are written BEFORE the assertion below: the one run
    # where they're actually needed for diagnosis (a real PSNR regression)
    # is exactly the run where an assert-first ordering would skip them.
    report = {"held_out_camera": HELD_OUT_REAL_CAMERA, "masked_psnr_db": round(score, 2),
               "golden_baseline_db": baseline}
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "psnr_report.json").write_text(json.dumps(report, indent=2))
    import shutil
    eval_renders_dir = ARTIFACTS_DIR / "eval_renders"
    eval_renders_dir.mkdir(exist_ok=True)
    shutil.copy(render_path, eval_renders_dir / render_path.name)

    assert score >= baseline - PSNR_MARGIN_DB, (
        f"masked PSNR on held-out camera {HELD_OUT_REAL_CAMERA} ({score:.2f}dB) dropped more than "
        f"{PSNR_MARGIN_DB}dB below the golden baseline ({baseline:.2f}dB)"
    )
