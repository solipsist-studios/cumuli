"""
tests/unit/test_integration_baseline_checks.py

Proves tests/integration/test_pipeline_end_to_end.py's golden-baseline
regression checks (reprojection error, mask coverage, the eval4d PSNR
gate) actually fail on bad input, not just pass on good input -- real
runs produce good results, so without this file none of those checks
would ever have been observed to catch anything. Same "prove the tests
aren't padding" practice already applied throughout tests/unit/:
temporarily confirm the check fails against deliberately bad data, then
confirm it passes again against good data.

Deliberately lives in tests/unit/, not tests/integration/: these checks
are pure comparison logic (arithmetic on synthetic data), need no real
GPU/conda-envs/pipeline run, and tests/integration/conftest.py's
autouse pipeline_prereqs fixture would otherwise skip this file on any
machine without that real setup -- defeating the point of a fast, always
-runnable proof that the comparison logic itself is sound. Imports
directly from the integration test module (not a copy) so this can never
drift out of sync with the real checks.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "integration"))
import run_unified_pipeline as unified  # noqa: E402
import test_pipeline_end_to_end as t  # noqa: E402


def test_reprojection_error_check_catches_regression(monkeypatch):
    monkeypatch.setattr(t, "GOLDEN", {**t.GOLDEN, "median_reprojection_error_px": 5.0})
    monkeypatch.setattr(t, "REPROJECTION_ERROR_MARGIN_PX", 10.0)

    good_run = {"log": "some preamble\nMedian residual: 100.00px -> 8.00px\nmore log"}
    t.test_poses_stage_reprojection_error_within_baseline(good_run, cpu_pipeline=False)  # must not raise

    bad_run = {"log": "some preamble\nMedian residual: 100.00px -> 30.00px\nmore log"}
    with pytest.raises(AssertionError, match="reprojection error"):
        t.test_poses_stage_reprojection_error_within_baseline(bad_run, cpu_pipeline=False)


def test_mask_coverage_check_catches_regression(monkeypatch, tmp_path):
    L = unified.build_layout(tmp_path)
    L["flat_fmasks_clean"].mkdir(parents=True)
    L["flat_label_map"].write_text(json.dumps({"00": "Camera_0001"}))

    monkeypatch.setattr(t, "REAL_CAMERAS", ["0001"])
    monkeypatch.setattr(t, "GOLDEN", {**t.GOLDEN, "mask_coverage": {"0001": 0.05}})
    monkeypatch.setattr(t, "MASK_COVERAGE_MARGIN", 0.02)
    pipeline_run = {"L": L}

    # Good: ~5% coverage, within the 0.05 +/- 0.02 window.
    good_mask = np.zeros((20, 20), dtype=np.uint8)
    good_mask[:5, :4] = 255  # 20/400 = 5%
    Image.fromarray(good_mask).save(L["flat_fmasks_clean"] / "00.png")
    t.test_masks_stage_coverage_within_baseline(pipeline_run, cpu_pipeline=False)  # must not raise

    # Bad: subject dropped out of the mask entirely -- 0% coverage.
    bad_mask = np.zeros((20, 20), dtype=np.uint8)
    Image.fromarray(bad_mask).save(L["flat_fmasks_clean"] / "00.png")
    with pytest.raises(AssertionError, match="mask coverage"):
        t.test_masks_stage_coverage_within_baseline(pipeline_run, cpu_pipeline=False)


def test_eval4d_check_catches_regression(monkeypatch, tmp_path):
    L = unified.build_layout(tmp_path)

    monkeypatch.setattr(t, "GOLDEN", {**t.GOLDEN, "eval4d_psnr_db": 30.0})
    monkeypatch.setattr(t, "EVAL4D_PSNR_MARGIN_DB", 2.0)
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(t, "ARTIFACTS_DIR", artifacts_dir)
    pipeline_run = {"L": L}

    def write_report(psnr):
        L["eval4d_report"].parent.mkdir(parents=True, exist_ok=True)
        L["eval4d_report"].write_text(json.dumps(
            {"mean": {"psnr_db": psnr, "ssim": 0.9, "lpips": 0.1}, "views": []}))

    # Good: within margin of the golden value.
    write_report(29.0)
    t.test_eval4d_metrics_meet_baseline(pipeline_run, cpu_pipeline=False)  # must not raise

    # Bad: a multi-dB regression.
    write_report(20.0)
    with pytest.raises(AssertionError, match="eval PSNR"):
        t.test_eval4d_metrics_meet_baseline(pipeline_run, cpu_pipeline=False)


def test_eval4d_check_skips_until_a_baseline_exists(monkeypatch, tmp_path):
    """The eval4d gate must be informational before the first GPU-runner
    baseline run records eval4d_* fields -- gating on an invented number
    would fail every run or check nothing."""
    L = unified.build_layout(tmp_path)
    golden = {k: v for k, v in t.GOLDEN.items() if not k.startswith("eval4d_")}
    monkeypatch.setattr(t, "GOLDEN", golden)
    monkeypatch.setattr(t, "ARTIFACTS_DIR", tmp_path / "artifacts")
    L["eval4d_report"].parent.mkdir(parents=True, exist_ok=True)
    L["eval4d_report"].write_text(json.dumps(
        {"mean": {"psnr_db": 25.0, "ssim": 0.9, "lpips": 0.1}, "views": []}))
    with pytest.raises(pytest.skip.Exception, match="no eval4d"):
        t.test_eval4d_metrics_meet_baseline({"L": L}, cpu_pipeline=False)
