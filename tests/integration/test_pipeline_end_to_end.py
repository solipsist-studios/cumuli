"""
tests/integration/test_pipeline_end_to_end.py

Real, unmocked pipeline test: runs the 6-stage volumetric capture
pipeline (sync -> production -> poses -> masks -> dataset4d -> train4d)
against the committed take01_11cam fixture
(tests/integration/fixtures/take01_11cam/, all 11 real cameras -- an
earlier 5-camera prototype was retired after comparison testing showed 11
cameras is more robust: clean HLOC registration every time, where a
5-camera subset failed to register one camera on a first attempt).

Two modes:

  - GPU mode (default): the full pipeline, ending in rotor 4DGS training
    (train_scratch.py in deps/OMG4), a baked .sogst, and eval_render.py
    scoring of a held-out camera (eval_4d.json). Checks structure AND
    quality: reprojection error and mask coverage against the golden
    baseline, plus the eval4d metrics once a baseline for them exists
    (see test_eval4d_metrics_meet_baseline -- informational until the
    first GPU-runner baseline run records eval4d_* fields).
  - CPU mode (VCP_CPU_PIPELINE=1): a DELIBERATELY SMALLER config that
    stops after the dataset4d stage (--stop_after_stage dataset4d) --
    structural completion only ("did every stage produce valid,
    non-degenerate output"), no training, no GPU, no quality claims.

IMPORTANT caveat about the eval camera: training with one camera held out
(required to get any held-out score at all) is NOT how this pipeline is
used in production -- production trains on every available camera. The
eval score is a PROXY: "can the model reconstruct a view it never saw",
which is sensitive to the same upstream breakage (bad sync, bad masks,
bad poses) a real bug would cause. Two further honesty rules, measured
the hard way on earlier captures: the held-out camera's stereo MATES must
also leave training or the score measures leakage (~5.7 dB of pure leak
observed on an 11-camera paired rig), and masked/subject-only comparison
is mandatory because ~98% of every frame here is black background that
scores trivially well. eval_render.py + build_flipbook_4dgs_dataset.py's
same-pass GT emission handle the mis-framed-GT trap; the golden margins
stay WIDE until a real variance study of the 4D metrics exists (the old
Brush-era 7-run study does not transfer to a different trainer).

Run-to-run stability notes that survive from the Brush era because they
are about upstream stages, not the trainer: mask generation is
deterministic (bit-identical across 18+ runs); reprojection error is
USUALLY stable (0.06px range) but not deterministic -- HLOC's own initial
estimate varies ~2.3px run to run and one run in 18+ landed 0.25px off
the refined cluster. REPROJECTION_ERROR_MARGIN_PX covers that observed
case.

See conftest.py for how this suite skips cleanly (not fails) on a machine
without the envs/checkpoints/GPU it needs. The whole pipeline runs fresh
from raw video on every test session -- nothing is cached/resumed across
runs, so a regression anywhere in the chain gets caught.

CPU-mode sizing (all real findings, see each constant): CPU_REAL_CAMERAS
(3 cameras -- every 2-camera pair made HLOC's reconstruction fail
outright; 3 is the functional minimum), CPU_SYNC_WINDOW (1 -- cuts the
dominant Sapiens keypoint cost), CPU_TRAIN_WINDOW (2 -- the smallest
sequence that still exercises the per-frame dataset4d chain twice),
CPU_HLOC_RESIZE_MAX / CPU_SAPIENS_MODEL_SIZE (runner memory ceilings,
measured).
"""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import run_unified_pipeline as unified
from baseline_metrics import eval4d_metrics, mask_coverage, parse_reprojection_error

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

TARGET_TIME = "1500ms"

# GPU-mode 4D training config: small on purpose. 2000 iterations and a
# 4x-downscaled 8-frame window keep the run inside the 60-minute GPU job
# (the production config is 30000 iterations at full resolution over a
# much longer window). Calibrate on the runner before tightening any
# margin that depends on these numbers.
TOTAL_TRAIN_ITERS = 2000
TRAIN_WINDOW = 8
DATASET_DOWNSCALE = 4
# Flat 2-digit label of the held-out eval camera. For this fixture flat
# "00" is real camera 0001 (see baseline_metrics.flat_label_for, which
# reads the pipeline's own camera_label_map.json). No stereo-mate
# holdouts are configured yet: this fixture's pair map has not been
# measured. Record it in golden_baseline.json (eval4d_holdout) before
# trusting absolute eval numbers -- see the module docstring.
EVAL_CAMERA = "00"

PIPELINE_TIMEOUT_S_GPU = 3000  # 50min -- headroom over the reduced 4D config
                                 # (2000 iters, downscale 4, window 8) while
                                 # staying under integration-tests.yml's 60-min
                                 # job timeout so this cleaner failure fires
                                 # first. Calibrate after the first real runs.
PIPELINE_TIMEOUT_S_CPU = 7200  # 2h -- CPU mode runs CPU_REAL_CAMERAS (3, not 11),
                                 # CPU_SYNC_WINDOW (1, not 5), and a train window
                                 # of 2, then stops after dataset4d. Real headroom
                                 # for CI-runner variance without masking a genuine
                                 # hang for long -- comfortably below
                                 # integration-tests-cpu.yml's job timeout.

REAL_CAMERAS = ["0001", "0002", "0003", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012"]

# CPU mode only -- Sapiens keypoint cost scales with camera count x frame
# count; 3 cameras is what makes CI wall time tractable. NOT 2 cameras --
# real finding (2026-07-30): HLOC's COLMAP reconstruction failed outright
# with EVERY 2-camera pair tried, confirmed via a fast standalone probe
# (run_hloc.py directly against the real 11-camera production frame):
# every 2-camera trial failed, every 3-camera trial succeeded. 3 is the
# real minimum.
CPU_REAL_CAMERAS = ["0006", "0007", "0008"]
CPU_SYNC_WINDOW = 1  # window=1 is a genuinely different extract_synced_frames.py
                      # output layout (flat files, no f0/ subdir) -- handled by
                      # run_unified_pipeline.py's _instant_dirs(); cuts the
                      # dominant Sapiens cost from 5x to 1x candidate pass.
CPU_TRAIN_WINDOW = 2  # the smallest window that runs the per-frame dataset4d
                       # chain more than once, so the frame loop (and its
                       # reuse-on-disk skip) is genuinely exercised.
CPU_EVAL_CAMERA = "02"  # holds one of the three CPU cameras out of the train
                         # split so the dataset build's holdout plumbing runs.
CPU_HLOC_RESIZE_MAX = 1024  # NOT the 4096 production default -- SuperPoint at
                             # 4096 peaks at 11.7GB RSS on CPU; the runner has
                             # ~7.75GB. 1024 is the lowest working setting
                             # (512 fails reconstruction outright).
CPU_SAPIENS_MODEL_SIZE = "0.4b"  # the 1b checkpoint alone peaks at 6.5-7.6GB
                                   # and swap-thrashed the ~7.75GB runner to
                                   # death; 0.4b measured 4.1GB peak.

with open(Path(__file__).resolve().parent / "fixtures" / "take01_11cam" / "golden_baseline.json") as f:
    GOLDEN = json.load(f)

# Reprojection/coverage margins carry over from the Brush era unchanged:
# they gate stages upstream of the trainer, and their variance studies
# (7 runs) remain valid. The 4D eval margin is WIDE on purpose -- no
# variance study of the 4D metrics exists yet; tighten it only from
# measured run-to-run data, the way the old PSNR margin was set.
REPROJECTION_ERROR_MARGIN_PX = 1.0  # observed range across 7 runs: 0.06px (baseline 5.57px)
MASK_COVERAGE_MARGIN = 0.005  # observed: bit-identical across 7 runs/11 cameras
EVAL4D_PSNR_MARGIN_DB = 2.0  # provisional -- see comment above


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory, sapiens_checkpoint_root, fixture_dir, cpu_pipeline):
    """Runs the real orchestrator once, for the whole test session, as an
    actual subprocess (matching how anyone would really invoke it -- the
    orchestrator itself dispatches each stage into its own conda env via
    subprocess regardless of how it's launched). Every test function below
    reads this one run's output; nothing here re-runs the pipeline."""
    out_dir = tmp_path_factory.mktemp("vcp_integration_run")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if cpu_pipeline:
        # CPU_REAL_CAMERAS-only subset: symlink just those movie files
        # into a fresh dir rather than adding a second fixture. --calib_dir
        # stays pointed at the full fixture: unused per-camera calibration
        # files are ignored by whatever looks calibration up per camera.
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
    ]
    if cpu_pipeline:
        cmd += [
            "--stop_after_stage", "dataset4d",
            "--train_window", str(CPU_TRAIN_WINDOW),
            "--eval_camera", CPU_EVAL_CAMERA,
            "--sync_window", str(CPU_SYNC_WINDOW),
            "--sapiens_model_size", CPU_SAPIENS_MODEL_SIZE,
            "--hloc_resize_max", str(CPU_HLOC_RESIZE_MAX),
        ]
    else:
        cmd += [
            "--train_window", str(TRAIN_WINDOW),
            "--dataset_downscale", str(DATASET_DOWNSCALE),
            "--total_train_iters", str(TOTAL_TRAIN_ITERS),
            "--eval_camera", EVAL_CAMERA,
        ]

    env = dict(os.environ)
    timeout_s = PIPELINE_TIMEOUT_S_CPU if cpu_pipeline else PIPELINE_TIMEOUT_S_GPU
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                                timeout=timeout_s)
        returncode, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        # e.stdout/e.stderr can be raw bytes here even though text=True was
        # passed to subprocess.run above -- on POSIX, a timeout interrupts
        # Popen.communicate() before its normal text-decoding step runs.
        # Discovered the hard way: a genuine timeout crashed this handler
        # with "can't concat str to bytes" instead of reporting itself.
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

def test_sync_stage_produced_offsets_for_every_camera(pipeline_run, cpu_pipeline):
    cameras = CPU_REAL_CAMERAS if cpu_pipeline else REAL_CAMERAS
    L = pipeline_run["L"]
    data = json.loads(L["sync_offsets"].read_text())
    offset_cams = sorted(Path(name).stem for name in data["offsets"])
    assert offset_cams == sorted(cameras)
    assert L["sync_grid"].is_file()


def test_production_stage_undistorted_every_camera(pipeline_run, cpu_pipeline):
    """By the time the full run finishes, run_hloc.py's
    restructure_flat_to_percam() side effect (stage 'poses') has already
    turned production_undist/ from flat <camera>.jpg files into
    Camera_<id>/0000.jpg subdirs -- checking for the pre-restructure flat
    layout here would silently check the wrong thing once every stage has
    run (caught by running this test against real output before trusting
    it)."""
    cameras = CPU_REAL_CAMERAS if cpu_pipeline else REAL_CAMERAS
    L = pipeline_run["L"]
    found = sorted(p.name for p in L["production_undist"].glob("Camera_*") if p.is_dir())
    expected = sorted(f"Camera_{c}" for c in cameras)
    assert found == expected
    for cam_dir in expected:
        assert any((L["production_undist"] / cam_dir).glob("*.jpg")), f"{cam_dir} has no frame"


def test_poses_stage_transforms_cover_every_camera(pipeline_run, cpu_pipeline):
    cameras = CPU_REAL_CAMERAS if cpu_pipeline else REAL_CAMERAS
    L = pipeline_run["L"]
    data = json.loads(L["transforms_refined"].read_text())
    labels = sorted(fr["camera_label"] for fr in data["frames"])
    assert labels == sorted(f"Camera_{c}" for c in cameras)


def test_poses_stage_reprojection_error_within_baseline(pipeline_run, cpu_pipeline):
    """Parses run_pose_refinement.py's own printed 'Median residual: X ->
    Y' line (there's no machine-readable report file for this) and checks
    it against golden_baseline.json's recorded value, with slack -- a
    regression guard for THIS fixture, not an absolute accuracy bar (see
    golden_baseline.json). GPU-only: the golden number was measured on the
    full 11-camera config, which CPU mode does not run."""
    if cpu_pipeline:
        pytest.skip("golden-baseline quality checks are GPU-only -- CPU mode runs a reduced config")
    final_median_px = parse_reprojection_error(pipeline_run["log"])
    assert final_median_px is not None, "could not find pose-refinement's 'Median residual: ... -> ...px' line in the run log"
    baseline = GOLDEN["median_reprojection_error_px"]
    assert final_median_px <= baseline + REPROJECTION_ERROR_MARGIN_PX, (
        f"pose refinement's median reprojection error ({final_median_px:.2f}px) drifted more than "
        f"{REPROJECTION_ERROR_MARGIN_PX}px past the golden baseline ({baseline:.2f}px)"
    )


def test_masks_stage_coverage_within_baseline(pipeline_run, cpu_pipeline):
    """Cleaned-mask coverage fraction per camera vs. golden_baseline.json,
    with slack -- catches a mask silhouette shrinking/ballooning even
    though it'd still pass validate_stage_output.py's much looser
    0.5%-90% sanity range. GPU-only: the golden numbers cover all 11
    cameras, which CPU mode does not run."""
    if cpu_pipeline:
        pytest.skip("golden-baseline quality checks are GPU-only -- CPU mode runs a reduced config")
    L = pipeline_run["L"]
    for cam in REAL_CAMERAS:
        frac = mask_coverage(L, cam)
        baseline = GOLDEN["mask_coverage"][cam]
        assert abs(frac - baseline) <= MASK_COVERAGE_MARGIN, (
            f"{cam}: cleaned mask coverage {frac:.3f} drifted more than {MASK_COVERAGE_MARGIN} "
            f"from the golden baseline ({baseline:.3f})"
        )


def test_dataset4d_layout_is_valid(pipeline_run, cpu_pipeline):
    """Structural check on the 4D training dataset, in BOTH modes -- this
    is the CPU run's terminal stage and the GPU run's training input. The
    checks mirror validate_dataset4d's core promises from the test's own
    perspective: per-view intrinsics on every frame entry (the flipbook
    layout deliberately has no global intrinsics block) and a time-aware
    point-cloud init."""
    L = pipeline_run["L"]
    train = json.loads((L["dataset4d"] / "transforms_train.json").read_text())
    test = json.loads((L["dataset4d"] / "transforms_test.json").read_text())
    assert train["frames"], "transforms_train.json carries no frames"
    for frame in train["frames"] + test["frames"]:
        for key in ("fl_x", "fl_y", "cx", "cy"):
            assert key in frame, f"frame entry missing per-view intrinsic {key!r}"
        assert "time" in frame
    header = (L["dataset4d"] / "points3d.ply").read_bytes()[:2048].decode("latin-1")
    assert "element vertex" in header
    assert "property float time" in header, "points3d.ply lacks the per-point time property"


def test_train4d_produced_sogst(pipeline_run, cpu_pipeline):
    """The baked .sogst is the pipeline's actual product. GPU-only: CPU
    mode stops after dataset4d and never trains."""
    if cpu_pipeline:
        pytest.skip("train4d is GPU-only -- CPU mode stops after dataset4d")
    L = pipeline_run["L"]
    assert zipfile.is_zipfile(L["sogst_out"]), f"{L['sogst_out']} is not a ZIP archive"
    with zipfile.ZipFile(L["sogst_out"]) as zf:
        meta = json.loads(zf.read("meta.json"))
    assert meta["format"] == "sogst"
    assert meta["count"] >= 1

    # Copy the baked asset out for CI upload. The mkdir matters: this
    # function is also called by update_golden_baseline.py's structural
    # gate, where the pipeline_run fixture (which normally creates
    # ARTIFACTS_DIR) never runs.
    import shutil
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(L["sogst_out"], ARTIFACTS_DIR / L["sogst_out"].name)


def test_eval4d_metrics_meet_baseline(pipeline_run, cpu_pipeline):
    """eval_render.py's held-out metrics vs the golden baseline. GPU-only.
    Informational until the first GPU-runner baseline run records the
    eval4d_* fields: comparing against a number nobody has measured would
    either always fail or gate on an invented value."""
    if cpu_pipeline:
        pytest.skip("train4d/eval are GPU-only -- CPU mode stops after dataset4d")
    L = pipeline_run["L"]
    report = eval4d_metrics(L)

    # Written BEFORE the assertion: the run where the report matters for
    # diagnosis (a real regression) is exactly the run where an
    # assert-first ordering would skip it.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "eval_4d.json").write_text(json.dumps(report, indent=2))

    if "eval4d_psnr_db" not in GOLDEN:
        pytest.skip("golden_baseline.json has no eval4d_* fields yet -- run "
                    "tests/integration/update_golden_baseline.py on the GPU runner to record them")
    score = report["mean"]["psnr_db"]
    baseline = GOLDEN["eval4d_psnr_db"]
    assert score >= baseline - EVAL4D_PSNR_MARGIN_DB, (
        f"held-out eval PSNR ({score:.2f}dB) dropped more than "
        f"{EVAL4D_PSNR_MARGIN_DB}dB below the golden baseline ({baseline:.2f}dB)"
    )
