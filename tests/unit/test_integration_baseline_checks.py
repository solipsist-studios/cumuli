"""
tests/unit/test_integration_baseline_checks.py

Proves tests/integration/test_pipeline_end_to_end.py's three golden-
baseline regression checks (reprojection error, mask coverage, PSNR)
actually fail on bad input, not just pass on good input -- both real
runs so far produced good results, so until this file, none of those
checks had ever been observed to catch anything. Same "prove the tests
aren't padding" practice already applied throughout tests/unit/ (see
INTEGRATION_TESTS_COMMIT_PLAN.md / COMMIT_PLAN.md): temporarily confirm
the check fails against deliberately bad data, then confirm it passes
again against good data.

Deliberately lives in tests/unit/, not tests/integration/: these checks
are pure comparison logic (arithmetic on synthetic data), need no real
GPU/conda-envs/brush_app/pipeline run, and tests/integration/conftest.py's
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
    t.test_poses_stage_reprojection_error_within_baseline(good_run, allow_cpu_rendering=False)  # must not raise

    bad_run = {"log": "some preamble\nMedian residual: 100.00px -> 30.00px\nmore log"}
    with pytest.raises(AssertionError, match="reprojection error"):
        t.test_poses_stage_reprojection_error_within_baseline(bad_run, allow_cpu_rendering=False)


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
    t.test_masks_stage_coverage_within_baseline(pipeline_run, allow_cpu_rendering=False)  # must not raise

    # Bad: subject dropped out of the mask entirely -- 0% coverage.
    bad_mask = np.zeros((20, 20), dtype=np.uint8)
    Image.fromarray(bad_mask).save(L["flat_fmasks_clean"] / "00.png")
    with pytest.raises(AssertionError, match="mask coverage"):
        t.test_masks_stage_coverage_within_baseline(pipeline_run, allow_cpu_rendering=False)


def test_psnr_check_catches_regression(monkeypatch, tmp_path):
    L = unified.build_layout(tmp_path)
    L["flat_label_map"].write_text(json.dumps({"00": "Camera_0001"}))
    eval_dir = L["brush_output"] / f"eval_{t.TOTAL_TRAIN_ITERS}"
    eval_dir.mkdir(parents=True)
    rgba_dir = L["train_set"] / "images_rgba"
    rgba_dir.mkdir(parents=True)

    monkeypatch.setattr(t, "HELD_OUT_REAL_CAMERA", "0001")
    monkeypatch.setattr(t, "GOLDEN", {**t.GOLDEN, "psnr_db": 20.0})
    monkeypatch.setattr(t, "PSNR_MARGIN_DB", 3.0)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    monkeypatch.setattr(t, "ARTIFACTS_DIR", artifacts_dir)
    pipeline_run = {"L": L}

    size = (16, 16)
    gt = np.zeros((*size[::-1], 4), dtype=np.uint8)
    gt[4:12, 4:12] = [128, 128, 128, 255]  # opaque subject square, transparent elsewhere
    Image.fromarray(gt, mode="RGBA").save(rgba_dir / "00.png")

    # Good: render closely matches the ground-truth subject region -> high PSNR.
    good_render = np.zeros((*size[::-1], 3), dtype=np.uint8)
    good_render[4:12, 4:12] = [130, 128, 126]
    Image.fromarray(good_render, mode="RGB").save(eval_dir / "00.png")
    t.test_psnr_meets_baseline(pipeline_run, allow_cpu_rendering=False)  # must not raise

    # Bad: render wildly disagrees with ground truth in the subject region -> low PSNR.
    bad_render = np.zeros((*size[::-1], 3), dtype=np.uint8)
    bad_render[4:12, 4:12] = [0, 255, 0]
    Image.fromarray(bad_render, mode="RGB").save(eval_dir / "00.png")
    with pytest.raises(AssertionError, match="masked PSNR"):
        t.test_psnr_meets_baseline(pipeline_run, allow_cpu_rendering=False)
