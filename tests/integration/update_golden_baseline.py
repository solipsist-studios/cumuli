#!/usr/bin/env python3
"""
tests/integration/update_golden_baseline.py

Post-merge companion to test_pipeline_end_to_end.py: re-runs the real
pipeline against the same heidi_11cam fixture and settings, then
overwrites golden_baseline.json with this run's fresh numbers and appends
one line to baseline_history.jsonl -- the append-only record of every
merged result over time. Meant to run from
.github/workflows/update-integration-baseline.yml on every push to main
(i.e. after a PR merges), NOT on every PR attempt -- see that workflow's
own comments for why.

Deliberately does NOT compare this run's numbers against the OLD
golden_baseline.json or fail on drift -- that gating already happened on
the PR's own pull_request-triggered integration-tests.yml run before this
code ever reached main. This script's only job is capturing fresh ground
truth. It DOES still confirm the run itself succeeded structurally
(pipeline exit code, every stage's expected output present -- the same
checks test_pipeline_end_to_end.py uses, imported directly rather than
re-implemented so the two can't drift apart) before writing anything: a
crashed or infra-broken run must never poison the baseline history.

The baseline is always the single most recent result, never an average
across runs -- averaging would dilute a real, intentional quality
improvement the same way it would dilute noise, and this project already
decided against that (see INTEGRATION_TESTS_COMMIT_PLAN.md).

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
from baseline_metrics import held_out_psnr, mask_coverage, parse_reprojection_error  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "heidi_11cam"
GOLDEN_PATH = FIXTURE_DIR / "golden_baseline.json"
HISTORY_PATH = FIXTURE_DIR / "baseline_history.jsonl"


def run_pipeline(out_dir: Path, sapiens_checkpoint_root: str, brush_app: str) -> dict:
    """Same command test_pipeline_end_to_end.py's pipeline_run fixture
    uses -- constants (TARGET_TIME, TOTAL_TRAIN_ITERS, EVAL_SPLIT_EVERY)
    are imported from that module rather than re-typed here."""
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_unified_pipeline.py"),
        "--video_dir", str(FIXTURE_DIR / "movies"),
        "--calib_dir", str(FIXTURE_DIR / "calibration_pkls"),
        "--out_dir", str(out_dir),
        "--target_time", t.TARGET_TIME,
        "--sapiens_checkpoint_root", sapiens_checkpoint_root,
        "--brush_app", brush_app,
        "--total_train_iters", str(t.TOTAL_TRAIN_ITERS),
        "--export_every", str(t.TOTAL_TRAIN_ITERS),
        "--eval_split_every", str(t.EVAL_SPLIT_EVERY),
        "--eval_save_to_disk",
    ]

    env = dict(os.environ)
    if os.environ.get("VCP_ALLOW_CPU_RENDERING") == "1":
        # Not exercised by CI today -- update-integration-baseline.yml
        # always targets the self-hosted GPU runner and never sets this,
        # so golden_baseline.json stays GPU-only (its numbers aren't
        # comparable to a CPU/llvmpipe run's -- see
        # tests/integration/conftest.py's module docstring). Kept correct
        # for local/manual use, matching test_pipeline_end_to_end.py's
        # pipeline_run fixture exactly (see its comments for why each of
        # these matters -- WGPU_BACKEND=gl alone does NOT avoid a real
        # GPU on a machine with an NVIDIA driver installed).
        env["WGPU_BACKEND"] = "gl"
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"
        mesa_icd = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
        if mesa_icd.is_file():
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(mesa_icd)
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd += ["--no_viewer", "--brush_max_resolution", str(t.CPU_MAX_RESOLUTION)]

    timeout_s = t.PIPELINE_TIMEOUT_S_CPU if os.environ.get("VCP_ALLOW_CPU_RENDERING") == "1" else t.PIPELINE_TIMEOUT_S_GPU
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                                timeout=timeout_s)
        returncode, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        # e.stdout/e.stderr can be raw bytes here even with text=True above
        # -- see the matching comment in test_pipeline_end_to_end.py's
        # pipeline_run fixture for why.
        returncode = -1
        stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        stderr += f"\npipeline killed after exceeding its timeout ({timeout_s}s)"

    # Persist the full log to the same artifacts dir the test fixture uses
    # -- update-integration-baseline.yml uploads that dir when this script
    # fails, and without this write a failed run leaves no diagnostics at
    # all (stdout/stderr would otherwise die with the process).
    t.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (t.ARTIFACTS_DIR / "pipeline_run.log").write_text(stdout + "\n" + stderr)

    L = unified.build_layout(out_dir)
    return {"out_dir": out_dir, "L": L, "returncode": returncode, "log": stdout}


def check_structurally_sound(run: dict) -> None:
    """Reuses test_pipeline_end_to_end.py's own structural per-stage
    checks (import, not a copy). Raises AssertionError if the run didn't
    actually succeed. Deliberately does NOT call the 3 golden-baseline
    comparison tests (reprojection/mask/PSNR) -- those compare against the
    baseline this script is about to overwrite, which isn't "did the run
    itself work"."""
    t.test_pipeline_completes_successfully(run)
    t.test_sync_stage_produced_offsets_for_every_camera(run, allow_cpu_rendering=False)
    t.test_production_stage_undistorted_every_camera(run, allow_cpu_rendering=False)
    t.test_poses_stage_transforms_cover_every_camera(run, allow_cpu_rendering=False)
    t.test_branch_stage_exported_a_nonempty_splat(run)


def measure(run: dict) -> dict:
    L = run["L"]
    reprojection_error = parse_reprojection_error(run["log"])
    if reprojection_error is None:
        raise AssertionError("could not find pose-refinement's 'Median residual: ... -> ...px' line in the run log")
    coverage = {cam: round(mask_coverage(L, cam), 4) for cam in t.REAL_CAMERAS}
    psnr, _render_path = held_out_psnr(L, t.TOTAL_TRAIN_ITERS, t.HELD_OUT_REAL_CAMERA)
    return {
        "median_reprojection_error_px": round(reprojection_error, 2),
        "mask_coverage": coverage,
        "psnr_db": round(psnr, 2),
    }


def write_golden_baseline(metrics: dict, commit: str) -> None:
    """Overwrites the 3 metric fields plus provenance (run_date,
    source_commit) in place -- keeps the file's own _comment/
    reproducibility_check/psnr_notes fields as historical documentation
    rather than regenerating them from a single run every time."""
    golden = json.loads(GOLDEN_PATH.read_text())
    golden["run_date"] = datetime.date.today().isoformat()
    golden["source_commit"] = commit
    golden["median_reprojection_error_px"] = metrics["median_reprojection_error_px"]
    golden["mask_coverage"] = metrics["mask_coverage"]
    golden["psnr_db"] = metrics["psnr_db"]
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2) + "\n")


def append_history(metrics: dict, commit: str) -> None:
    """Append-only: one JSON object per line, one line per merged result
    (never rewritten/edited -- see baseline_history.jsonl's own header
    entry for the format)."""
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "median_reprojection_error_px": metrics["median_reprojection_error_px"],
        "mask_coverage_avg": round(sum(metrics["mask_coverage"].values()) / len(metrics["mask_coverage"]), 4),
        "psnr_db": metrics["psnr_db"],
    }
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=None, help="Defaults to a fresh temp dir.")
    p.add_argument("--sapiens_checkpoint_root", default=None,
                   help="Defaults to $VCP_SAPIENS_CHECKPOINT_ROOT / $SAPIENS_CHECKPOINT_ROOT, "
                        "else ~/sapiens/2 -- same resolution order tests/integration/conftest.py uses.")
    p.add_argument("--brush_app", default=None,
                   help="Defaults to $VCP_BRUSH_APP, else run_unified_pipeline.py's own default -- "
                        "same resolution tests/integration/conftest.py uses.")
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
    brush_app = (args.brush_app
                 or os.environ.get("VCP_BRUSH_APP")
                 or str(unified.build_parser().get_default("brush_app")))

    print(f"Running pipeline against {FIXTURE_DIR} -> {out_dir} (commit {commit[:12]})")
    run = run_pipeline(out_dir, sapiens_checkpoint_root, brush_app)

    check_structurally_sound(run)  # raises AssertionError -> nonzero exit, nothing written, on a bad run
    metrics = measure(run)
    print(f"Measured: {json.dumps(metrics, indent=2)}")

    write_golden_baseline(metrics, commit)
    append_history(metrics, commit)
    print(f"Updated {GOLDEN_PATH} and appended to {HISTORY_PATH}")

    # Only after a fully successful update, and only if this script created
    # the temp dir itself: each run's output is ~720MB, and on a runner
    # that executes this on every merge, never cleaning up would
    # eventually fill the disk. A failed run keeps its dir for debugging
    # (the exception above exits before reaching this).
    if args.out_dir is None:
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"Cleaned up {out_dir}")


if __name__ == "__main__":
    main()
