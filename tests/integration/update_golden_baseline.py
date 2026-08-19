#!/usr/bin/env python3
"""
tests/integration/update_golden_baseline.py

Post-merge companion to test_pipeline_end_to_end.py: re-runs the real
pipeline against the same take01_11cam fixture and settings, then
overwrites golden_baseline.json with this run's fresh numbers and appends
one line to baseline_history.jsonl -- the append-only record of every
merged result over time. Meant to run from
.github/workflows/update-integration-baseline.yml on every push to main
(i.e. after a PR merges), NOT on every PR attempt -- see that workflow's
own comments for why. GPU-runner only: the pipeline's train4d stage is
CUDA-gated, and the eval4d_* baseline fields come from its eval report.

Deliberately does NOT compare this run's numbers against the OLD
golden_baseline.json or fail on drift -- that gating already happened on
the PR's own integration run before this code ever reached main. This
script's only job is capturing fresh ground truth. It DOES still confirm
the run itself succeeded structurally (pipeline exit code, every stage's
expected output present -- the same checks test_pipeline_end_to_end.py
uses, imported directly rather than re-implemented so the two can't
drift apart) before writing anything: a crashed or infra-broken run must
never poison the baseline history.

The baseline is always the single most recent result, never an average
across runs -- averaging would dilute a real, intentional quality
improvement the same way it would dilute noise, and this project already
decided against that.

Usage (as CI runs it):
    python3 tests/integration/update_golden_baseline.py --commit <sha>

Exits non-zero (via the underlying AssertionError) without writing
anything if the run didn't structurally succeed.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_unified_pipeline as unified  # noqa: E402
import test_pipeline_end_to_end as t  # noqa: E402
from baseline_metrics import eval4d_metrics, mask_coverage, parse_reprojection_error  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "take01_11cam"
GOLDEN_PATH = FIXTURE_DIR / "golden_baseline.json"
HISTORY_PATH = FIXTURE_DIR / "baseline_history.jsonl"


def run_pipeline(out_dir: Path, sapiens_checkpoint_root: str) -> dict:
    """Same command test_pipeline_end_to_end.py's pipeline_run fixture
    uses in GPU mode -- constants (TARGET_TIME, TOTAL_TRAIN_ITERS,
    TRAIN_WINDOW, DATASET_DOWNSCALE, EVAL_CAMERA) are imported from that
    module rather than re-typed here, so the baseline is always measured
    at exactly the config the gate compares against."""
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_unified_pipeline.py"),
        "--video_dir", str(FIXTURE_DIR / "movies"),
        "--calib_dir", str(FIXTURE_DIR / "calibration_pkls"),
        "--out_dir", str(out_dir),
        "--target_time", t.TARGET_TIME,
        "--sapiens_checkpoint_root", sapiens_checkpoint_root,
        "--train_window", str(t.TRAIN_WINDOW),
        "--dataset_downscale", str(t.DATASET_DOWNSCALE),
        "--total_train_iters", str(t.TOTAL_TRAIN_ITERS),
        "--eval_camera", t.EVAL_CAMERA,
    ]

    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=dict(os.environ),
                                capture_output=True, text=True,
                                timeout=t.PIPELINE_TIMEOUT_S_GPU)
        returncode, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        # e.stdout/e.stderr can be raw bytes here even with text=True above
        # -- see the matching comment in test_pipeline_end_to_end.py's
        # pipeline_run fixture for why.
        returncode = -1
        stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        stderr += f"\npipeline killed after exceeding its timeout ({t.PIPELINE_TIMEOUT_S_GPU}s)"

    # Persist the full log to the same artifacts dir the test fixture uses
    # -- update-integration-baseline.yml uploads that dir when this script
    # fails, and without this write a failed run leaves no diagnostics.
    t.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (t.ARTIFACTS_DIR / "pipeline_run.log").write_text(stdout + "\n" + stderr)

    L = unified.build_layout(out_dir)
    return {"out_dir": out_dir, "L": L, "returncode": returncode, "log": stdout}


def check_structurally_sound(run: dict) -> None:
    """Reuses test_pipeline_end_to_end.py's own structural per-stage
    checks (import, not a copy). Raises AssertionError if the run didn't
    actually succeed. Deliberately does NOT call the golden-baseline
    comparison tests -- those compare against the baseline this script is
    about to overwrite, which isn't "did the run itself work"."""
    t.test_pipeline_completes_successfully(run)
    t.test_sync_stage_produced_offsets_for_every_camera(run, cpu_pipeline=False)
    t.test_production_stage_undistorted_every_camera(run, cpu_pipeline=False)
    t.test_poses_stage_transforms_cover_every_camera(run, cpu_pipeline=False)
    t.test_dataset4d_layout_is_valid(run, cpu_pipeline=False)
    t.test_train4d_produced_sogst(run, cpu_pipeline=False)


def measure(run: dict) -> dict:
    L = run["L"]
    reprojection_error = parse_reprojection_error(run["log"])
    if reprojection_error is None:
        raise AssertionError("could not find pose-refinement's 'Median residual: ... -> ...px' line in the run log")
    coverage = {cam: round(mask_coverage(L, cam), 4) for cam in t.REAL_CAMERAS}
    report = eval4d_metrics(L)
    mean = report["mean"]
    return {
        "median_reprojection_error_px": round(reprojection_error, 2),
        "mask_coverage": coverage,
        "eval4d_psnr_db": round(mean["psnr_db"], 2),
        "eval4d_ssim": round(mean["ssim"], 4),
        "eval4d_lpips": round(mean["lpips"], 4),
    }


def write_golden_baseline(metrics: dict, commit: str) -> None:
    """Overwrites the metric fields plus provenance (run_date,
    source_commit) in place -- keeps the file's own _comment and notes
    fields as historical documentation rather than regenerating them from
    a single run every time. eval4d_holdout/eval4d_config record the
    exact eval configuration the numbers were measured under, without
    which a later comparison is meaningless."""
    golden = json.loads(GOLDEN_PATH.read_text())
    golden["run_date"] = datetime.date.today().isoformat()
    golden["source_commit"] = commit
    golden["median_reprojection_error_px"] = metrics["median_reprojection_error_px"]
    golden["mask_coverage"] = metrics["mask_coverage"]
    golden["eval4d_psnr_db"] = metrics["eval4d_psnr_db"]
    golden["eval4d_ssim"] = metrics["eval4d_ssim"]
    golden["eval4d_lpips"] = metrics["eval4d_lpips"]
    golden["eval4d_holdout"] = {"eval_camera": t.EVAL_CAMERA, "excluded_mates": []}
    golden["eval4d_config"] = {
        "train_window": t.TRAIN_WINDOW,
        "dataset_downscale": t.DATASET_DOWNSCALE,
        "total_train_iters": t.TOTAL_TRAIN_ITERS,
        "target_time": t.TARGET_TIME,
    }
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2) + "\n")


def append_history(metrics: dict, commit: str) -> None:
    """Append-only: one JSON object per line, one line per merged result
    (never rewritten/edited)."""
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "median_reprojection_error_px": metrics["median_reprojection_error_px"],
        "mask_coverage_avg": round(sum(metrics["mask_coverage"].values()) / len(metrics["mask_coverage"]), 4),
        "eval4d_psnr_db": metrics["eval4d_psnr_db"],
        "eval4d_ssim": metrics["eval4d_ssim"],
        "eval4d_lpips": metrics["eval4d_lpips"],
    }
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=None, help="Defaults to a fresh temp dir.")
    p.add_argument("--sapiens_checkpoint_root", default=None,
                   help="Defaults to $VCP_SAPIENS_CHECKPOINT_ROOT / $SAPIENS_CHECKPOINT_ROOT, "
                        "else ~/sapiens/2 -- same resolution order tests/integration/conftest.py uses.")
    p.add_argument("--commit", default=None, help="Commit SHA to record; defaults to `git rev-parse HEAD`.")
    return p


def main():
    args = build_parser().parse_args()
    commit = args.commit
    if commit is None:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
        ).stdout.strip()
    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="vcp_baseline_update_"))
    sapiens_checkpoint_root = (args.sapiens_checkpoint_root
                               or os.environ.get("VCP_SAPIENS_CHECKPOINT_ROOT")
                               or os.environ.get("SAPIENS_CHECKPOINT_ROOT")
                               or str(Path.home() / "sapiens" / "2"))

    print(f"Running pipeline against {FIXTURE_DIR} -> {out_dir} (commit {commit[:12]})")
    run = run_pipeline(out_dir, sapiens_checkpoint_root)

    check_structurally_sound(run)  # raises AssertionError -> nonzero exit, nothing written, on a bad run
    metrics = measure(run)
    print(f"Measured: {json.dumps(metrics, indent=2)}")

    write_golden_baseline(metrics, commit)
    append_history(metrics, commit)
    print(f"Updated {GOLDEN_PATH} and appended to {HISTORY_PATH}")

    # Only after a fully successful update, and only if this script created
    # the temp dir itself: each run's output is large, and on a runner that
    # executes this on every merge, never cleaning up would eventually fill
    # the disk. A failed run keeps its dir for debugging (the exception
    # above exits before reaching this).
    if args.out_dir is None:
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"Cleaned up {out_dir}")


if __name__ == "__main__":
    main()
