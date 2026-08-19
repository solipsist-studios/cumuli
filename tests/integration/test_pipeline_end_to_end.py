"""
tests/integration/test_pipeline_end_to_end.py

Real, unmocked pipeline test: runs the full 5-stage volumetric capture
pipeline (sync -> production -> poses -> masks -> branch) against the
committed take01_11cam fixture (tests/integration/fixtures/take01_11cam/,
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

CPU mode (VCP_ALLOW_CPU_RENDERING=1) runs a DELIBERATELY SMALLER config
than the paragraphs above describe, and skips the three golden-baseline
quality checks entirely (see each test's own `if allow_cpu_rendering:
pytest.skip(...)`) -- structural completion only ("did every stage
produce valid, non-degenerate output", not "does it match a quality
baseline"). Real CI finding (2026-07-30): the full config's dominant cost
is Sapiens keypoint prediction (predict_keypoints_2d.py), which runs once
per --sync_window candidate frame (5 by default) PLUS once more for the
production frame -- 6 total passes, ~39min each measured locally on an
11-camera candidate set, i.e. hours of CPU wall time before Brush training
even starts. A first real CI run of the full config ran past 2 hours
without finishing. CPU mode cuts this via CPU_REAL_CAMERAS (3 cameras
instead of 11 -- see CPU_REAL_CAMERAS's own comment: every 2-camera
pair tried made HLOC's COLMAP reconstruction fail outright with
insufficient baseline/parallax, confirmed via a fast standalone probe
that isolates just the reconstruction step -- 3 is the real functional
minimum, not a scope choice) and CPU_SYNC_WINDOW (1 instead of 5, so
keypoints/masks run 2x total instead of 6x -- see CPU_SYNC_WINDOW's own
comment: this required a real fix to run_unified_pipeline.py, not just
a config value, since --sync_window 1 crashed outright before that fix),
plus CPU_TOTAL_TRAIN_ITERS (a small
Brush training count -- quality is unchecked in this mode, so there's no
reason to train to convergence). The golden-baseline quality checks (and
the full 11-camera/window-5 config) still run as-is in GPU mode --
unchanged by any of this. Known tradeoff, not a hidden one: an earlier
5-camera prototype (see above) failed to register a camera on its first
HLOC attempt -- fewer cameras measurably makes multi-view registration
less robust, not just faster, and 2 cameras genuinely isn't enough at
all (see above, confirmed twice). 3 cameras is a real, honest test of
"does the pipeline run end-to-end", just a structurally weaker one than
11 -- acceptable here because CPU mode no longer claims to test
reconstruction quality at all.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import run_unified_pipeline as unified
from baseline_metrics import clean_artifact_name, held_out_psnr, mask_coverage, parse_reprojection_error
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
PIPELINE_TIMEOUT_S_CPU = 7200  # 2h -- CPU mode now runs CPU_REAL_CAMERAS (3, not 11) and
                                 # CPU_SYNC_WINDOW (1, not 5), so Sapiens keypoint prediction (the
                                 # dominant cost -- see module docstring) runs on far less data than
                                 # the old 4h estimate assumed (that number was calibrated against
                                 # 11 cameras x up to 6 keypoint passes). Still real headroom above
                                 # the reduced config's expected runtime for CI-runner variance and
                                 # shared-machine contention, without masking a genuine hang for as
                                 # long as the old 4h did -- comfortably below
                                 # integration-tests-cpu.yml's job timeout so this cleaner failure
                                 # fires first.

REAL_CAMERAS = ["0001", "0002", "0003", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012"]

# CPU mode only -- see module docstring for why (Sapiens keypoint prediction
# cost scales with camera count x sync window; 3 cameras x 2 window instead
# of 11 x 5 is what actually makes CI wall time tractable -- every 2-camera
# pair tried made HLOC's COLMAP reconstruction fail outright, see this
# constant's own comment below). Real movie files for these three IDs exist
# in the committed fixture; a CPU_REAL_CAMERAS-only subset dir is built by
# the pipeline_run fixture below rather than adding a second, separate
# fixture.
CPU_REAL_CAMERAS = ["0006", "0007", "0008"]  # NOT 2 cameras -- real finding (2026-07-30):
                     # HLOC's COLMAP reconstruction failed outright with EVERY 2-camera pair
                     # tried (0001+0002, then 0006+0007), both "Failed to create any sparse
                     # model" / "Could not reconstruct any model!" -- confirmed via a fast
                     # standalone probe (run_hloc.py directly against the real 11-camera
                     # production frame, bypassing masks/keypoints/Sapiens entirely -- seconds
                     # per trial instead of 20+min) that this is a genuine 2-view-vs-3-view
                     # SfM limitation, not specific to which 2 cameras: every 2-camera trial
                     # failed, every 3-camera trial succeeded (0006+0007+0008 -> 333 3D points,
                     # 0001+0002+0003 -> 506 points). 3 cameras is the real minimum.
CPU_SYNC_WINDOW = 1  # Real bug found running this locally (2026-07-30):
                      # extract_synced_frames.py's --window has two different output
                      # layouts, not just "fewer frames": window=1 writes flat files
                      # (output_dir/0001.jpg), any window>1 writes per-instant subdirs
                      # (output_dir/f0/, f1/, ...). run_unified_pipeline.py's
                      # prepare_candidate_window()/stage_poses() always assumed the
                      # subdir layout, so --sync_window 1 crashed immediately (affected
                      # manual --sync_window 1 use too, not just this test). FIXED
                      # (2026-07-30) in run_unified_pipeline.py's new _instant_dirs()
                      # helper, which matches whichever layout extract_synced_frames.py
                      # actually produces. 1 cuts the dominant Sapiens-keypoint cost
                      # from 5x to 1x candidate pass -- measured locally: 22.7min total
                      # vs 27.8min at window=2, real end-to-end pipeline success
                      # (including the triangulate_and_project_keypoints.py step).
CPU_TOTAL_TRAIN_ITERS = 60  # clears one multiple of refine_every's default (50) with margin,
                             # so at least one real growth/refine step executes inside brush_app,
                             # not just export of the raw COLMAP-seeded gaussians -- quality is
                             # unchecked in CPU mode (see module docstring), so there's no reason
                             # to train further than that.
CPU_HLOC_RESIZE_MAX = 1024  # NOT the 4096 production default -- real CI failure
                             # (2026-07-31, CI run 30636025137): run_hloc.py's SuperPoint
                             # feature extraction at resize_max 4096 peaks at 11.7GB RSS
                             # on CPU (measured locally via /usr/bin/time -v on the same
                             # 3-camera images, OMP_NUM_THREADS=4) -- can never fit the
                             # runner's ~7.75GB. [resmon] telemetry showed the run die 21s
                             # after the candidates Sapiens pass finished, a single python3
                             # at 6.9GB: exactly run_hloc.py's slot in the stage order.
                             # Measured floor (same 3 cameras, CPU): 4096 = 11.7GB/468
                             # 3D points; 2048 = 3.4GB/350 pts; 1024 = 1.4GB/133 pts;
                             # 512 = reconstruction fails outright. 1024 is the lowest
                             # working setting. If 3-camera reconstruction ever turns
                             # flaky at 1024 (133 pts is a thin margin), bump to 2048
                             # first -- still fits the runner easily.
CPU_SAPIENS_MODEL_SIZE = "0.4b"  # NOT the 1b production default -- real CI failure
                                   # (2026-07-30, CI run 30581141971): the actual GitHub
                                   # runner has only ~7.75GB RAM (confirmed directly via
                                   # this session's own [resmon] telemetry -- earlier
                                   # "16GB" assumptions elsewhere in this file predate
                                   # this finding and were wrong), and the 1b Sapiens
                                   # checkpoint alone peaks at 6.5-7.6GB regardless of
                                   # camera count (the memory cost is dominated by the
                                   # model's own fixed weight size, not image count) --
                                   # left so little headroom that the run swap-thrashed
                                   # for ~40min then the whole VM became unresponsive and
                                   # was killed externally. 0.4b measured at 4.1GB peak
                                   # on the real 3-camera fixture -- real margin, not just
                                   # barely under. Quality is unchecked in CPU mode anyway
                                   # (see module docstring), so the accuracy tradeoff of a
                                   # smaller checkpoint doesn't cost anything here.

# brush_app's --eval-split-every 11 on this 11-image dataset holds out
# whichever image build_colmap_sparse.py wrote first as flat label "00" --
# for this fixture that's real camera 0001 (see baseline_metrics.flat_label_for(),
# which reads the pipeline's own camera_label_map.json rather than
# assuming a fixed sort order).
HELD_OUT_REAL_CAMERA = "0001"

with open(Path(__file__).resolve().parent / "fixtures" / "take01_11cam" / "golden_baseline.json") as f:
    GOLDEN = json.load(f)

# Margins below are derived from a real 7-run variance study (identical
# fixture/settings, --total_train_iters 1000, re-run 7 times back to back).
# Mask coverage
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

    if allow_cpu_rendering:
        # CPU_REAL_CAMERAS-only subset: symlink just those movie files
        # into a fresh dir rather than adding a second fixture -- see
        # module docstring for why CPU mode uses fewer cameras.
        # --calib_dir stays pointed at the full fixture: unused per-camera
        # calibration files for cameras not in this subset are just
        # ignored by whatever looks calibration up per-camera-label.
        video_dir = tmp_path_factory.mktemp("vcp_cpu_video_subset")
        for cam in CPU_REAL_CAMERAS:
            src = fixture_dir / "movies" / f"{cam}.mp4"
            (video_dir / src.name).symlink_to(src)
    else:
        video_dir = fixture_dir / "movies"

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "run_unified_pipeline.py"),
        "--video_dir", str(video_dir),
        "--calib_dir", str(fixture_dir / "calibration_pkls"),
        "--out_dir", str(out_dir),
        "--target_time", TARGET_TIME,
        "--sapiens_checkpoint_root", sapiens_checkpoint_root,
        "--brush_app", brush_app,
    ]
    if allow_cpu_rendering:
        cmd += [
            "--sync_window", str(CPU_SYNC_WINDOW),
            "--sapiens_model_size", CPU_SAPIENS_MODEL_SIZE,
            "--hloc_resize_max", str(CPU_HLOC_RESIZE_MAX),
            "--total_train_iters", str(CPU_TOTAL_TRAIN_ITERS),
            "--export_every", str(CPU_TOTAL_TRAIN_ITERS),
            # No --eval_split_every/--eval_save_to_disk: the PSNR check is
            # skipped entirely in CPU mode (test_psnr_meets_baseline), so
            # there's no reason to hold a camera out of training or render
            # an eval frame -- trains on all of CPU_REAL_CAMERAS instead.
        ]
    else:
        cmd += [
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

def test_sync_stage_produced_offsets_for_every_camera(pipeline_run, allow_cpu_rendering):
    cameras = CPU_REAL_CAMERAS if allow_cpu_rendering else REAL_CAMERAS
    L = pipeline_run["L"]
    data = json.loads(L["sync_offsets"].read_text())
    offset_cams = sorted(Path(name).stem for name in data["offsets"])
    assert offset_cams == sorted(cameras)
    assert L["sync_grid"].is_file()


def test_production_stage_undistorted_every_camera(pipeline_run, allow_cpu_rendering):
    """By the time the full run finishes, run_hloc.py's
    restructure_flat_to_percam() side effect (stage 'poses') has already
    turned production_undist/ from flat <camera>.jpg files into
    Camera_<id>/0000.jpg subdirs -- checking for the pre-restructure flat
    layout here would silently check the wrong thing once every stage has
    run (caught by running this test against real output before trusting
    it)."""
    cameras = CPU_REAL_CAMERAS if allow_cpu_rendering else REAL_CAMERAS
    L = pipeline_run["L"]
    found = sorted(p.name for p in L["production_undist"].glob("Camera_*") if p.is_dir())
    expected = sorted(f"Camera_{c}" for c in cameras)
    assert found == expected
    for cam_dir in expected:
        assert any((L["production_undist"] / cam_dir).glob("*.jpg")), f"{cam_dir} has no frame"


def test_poses_stage_transforms_cover_every_camera(pipeline_run, allow_cpu_rendering):
    cameras = CPU_REAL_CAMERAS if allow_cpu_rendering else REAL_CAMERAS
    L = pipeline_run["L"]
    data = json.loads(L["transforms_refined"].read_text())
    labels = sorted(fr["camera_label"] for fr in data["frames"])
    assert labels == sorted(f"Camera_{c}" for c in cameras)


def test_poses_stage_reprojection_error_within_baseline(pipeline_run, allow_cpu_rendering):
    """Parses run_pose_refinement.py's own printed 'Median residual: X ->
    Y' line (there's no machine-readable report file for this) and checks
    it against golden_baseline.json's recorded value, with slack -- a
    regression guard for THIS fixture, not an absolute accuracy bar (see
    golden_baseline.json). GPU-only -- see module docstring for why CPU
    mode skips every golden-baseline quality check."""
    if allow_cpu_rendering:
        pytest.skip("golden-baseline quality checks are GPU-only in CPU mode -- see module docstring")
    final_median_px = parse_reprojection_error(pipeline_run["log"])
    assert final_median_px is not None, "could not find pose-refinement's 'Median residual: ... -> ...px' line in the run log"
    baseline = GOLDEN["median_reprojection_error_px"]
    assert final_median_px <= baseline + REPROJECTION_ERROR_MARGIN_PX, (
        f"pose refinement's median reprojection error ({final_median_px:.2f}px) drifted more than "
        f"{REPROJECTION_ERROR_MARGIN_PX}px past the golden baseline ({baseline:.2f}px)"
    )


def test_masks_stage_coverage_within_baseline(pipeline_run, allow_cpu_rendering):
    """Cleaned-mask coverage fraction per camera vs. golden_baseline.json,
    with slack -- catches a mask silhouette shrinking/ballooning even
    though it'd still pass validate_stage_output.py's much looser
    0.5%-90% sanity range. GPU-only -- see module docstring for why CPU
    mode skips every golden-baseline quality check."""
    if allow_cpu_rendering:
        pytest.skip("golden-baseline quality checks are GPU-only in CPU mode -- see module docstring")
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


def test_psnr_meets_baseline(pipeline_run, allow_cpu_rendering):
    """Masked PSNR between Brush's held-out eval render (camera
    HELD_OUT_REAL_CAMERA, excluded from training via --eval_split_every)
    and the real photo from that same camera -- see the module docstring
    for why this is a proxy signal, not a production-quality measurement,
    and why it's masked to the subject rather than the full frame.
    GPU-only -- CPU mode doesn't hold a camera out at all (see the
    pipeline_run fixture), so there's no eval render to score, and every
    golden-baseline quality check is skipped in that mode regardless (see
    module docstring)."""
    if allow_cpu_rendering:
        pytest.skip("golden-baseline quality checks are GPU-only in CPU mode -- see module docstring")
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
    shutil.copy(render_path, eval_renders_dir / clean_artifact_name(render_path.name))

    assert score >= baseline - PSNR_MARGIN_DB, (
        f"masked PSNR on held-out camera {HELD_OUT_REAL_CAMERA} ({score:.2f}dB) dropped more than "
        f"{PSNR_MARGIN_DB}dB below the golden baseline ({baseline:.2f}dB)"
    )
