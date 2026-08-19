"""
tests/unit/test_update_golden_baseline.py

Proves tests/integration/update_golden_baseline.py's structural gate and
write-back logic actually work, not just "looks right on read" -- same
"prove the tests aren't padding" practice used throughout this project
(see tests/unit/test_integration_baseline_checks.py): confirm the
structural gate correctly raises on bad synthetic data AND passes on
good data, and confirm
measure()/write_golden_baseline()/append_history() round-trip real values
correctly.

Deliberately in tests/unit/, not tests/integration/: pure logic against
synthetic tmp_path data, no GPU/conda-envs/trainer needed -- same
reasoning as test_integration_baseline_checks.py. Imports directly from
tests/integration/update_golden_baseline.py so this can never drift out
of sync with the real code.
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
import update_golden_baseline as ugb  # noqa: E402


@pytest.fixture(autouse=True)
def _artifacts_dir_in_tmp(monkeypatch, tmp_path):
    """check_structurally_sound() reaches test_train4d_produced_sogst(),
    which copies the .sogst into the integration module's ARTIFACTS_DIR
    as a side effect -- point that at tmp so these tests never write into
    the real (gitignored) tests/integration/artifacts/ nor depend on it
    existing (it doesn't, on a fresh checkout)."""
    monkeypatch.setattr(t, "ARTIFACTS_DIR", tmp_path / "artifacts")


REAL_CAMERAS = ["0001", "0002", "0003", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012"]


def _write_minimal_ply(path, n_vertices=1, with_time=True):
    time_prop = "property float time\n" if with_time else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = "0 0 0 0\n" if with_time else "0 0 0\n"
    path.write_text(
        "ply\nformat ascii 1.0\n"
        f"element vertex {n_vertices}\n"
        "property float x\nproperty float y\nproperty float z\n"
        + time_prop + "end_header\n" + row * n_vertices
    )


def _write_minimal_sogst(path):
    import zipfile
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("meta.json", json.dumps({"format": "sogst", "version": 1, "count": 3}))


def _build_good_run(out_dir, with_metrics_data=False):
    """A minimal but complete synthetic run satisfying every structural
    check in check_structurally_sound(). Pass with_metrics_data=True to
    also populate what measure() needs (masks, camera_label_map, the
    eval_4d.json report) -- kept optional since the structural-gate tests
    don't need it."""
    L = unified.build_layout(out_dir)

    offsets = {f"{cam}.mp4": {"frame_offset": 0} for cam in REAL_CAMERAS}
    L["sync_offsets"].parent.mkdir(parents=True, exist_ok=True)
    L["sync_offsets"].write_text(json.dumps({"offsets": offsets}))
    L["sync_grid"].write_bytes(b"fake jpg")

    for cam in REAL_CAMERAS:
        cam_dir = L["production_undist"] / f"Camera_{cam}"
        cam_dir.mkdir(parents=True)
        (cam_dir / "0000.jpg").write_bytes(b"fake jpg")

    L["transforms_refined"].write_text(json.dumps(
        {"frames": [{"camera_label": f"Camera_{cam}"} for cam in REAL_CAMERAS]}
    ))

    # dataset4d: transforms pair with per-view intrinsics + a referenced
    # RGBA image + a time-aware points3d.ply.
    L["dataset4d"].mkdir(parents=True)
    intr = {"fl_x": 100.0, "fl_y": 100.0, "cx": 8.0, "cy": 8.0, "time": 0.0}
    frame = {"file_path": "realcams/cam00/frame_00000.png",
             "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], **intr}
    (L["dataset4d"] / "transforms_train.json").write_text(json.dumps({"frames": [frame]}))
    (L["dataset4d"] / "transforms_test.json").write_text(json.dumps({"frames": [frame]}))
    img_path = L["dataset4d"] / "realcams" / "cam00" / "frame_00000.png"
    img_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((16, 16, 4), dtype=np.uint8), mode="RGBA").save(img_path)
    _write_minimal_ply(L["dataset4d"] / "points3d.ply", n_vertices=1, with_time=True)

    # train4d: a checkpoint and a valid .sogst archive.
    L["train4d_model"].mkdir(parents=True)
    (L["train4d_model"] / "chkpnt2000.pth").write_bytes(b"x")
    _write_minimal_sogst(L["sogst_out"])

    log = "Median residual: 30.00px -> 5.00px\n"
    run = {"out_dir": out_dir, "L": L, "returncode": 0, "log": log}

    if with_metrics_data:
        label_map = {f"{i:02d}": f"Camera_{cam}" for i, cam in enumerate(REAL_CAMERAS)}
        L["flat_label_map"].write_text(json.dumps(label_map))

        L["flat_fmasks_clean"].mkdir(parents=True)
        for flat in label_map:
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[:2, :5] = 255  # 10/100 = 0.10 coverage, same for every camera
            Image.fromarray(mask).save(L["flat_fmasks_clean"] / f"{flat}.png")

        L["eval4d_report"].write_text(json.dumps(
            {"mean": {"psnr_db": 27.5, "ssim": 0.91, "lpips": 0.08}, "views": []}))

    return run


def test_check_structurally_sound_passes_on_good_run(tmp_path):
    run = _build_good_run(tmp_path)
    ugb.check_structurally_sound(run)  # must not raise


def test_check_structurally_sound_catches_nonzero_returncode(tmp_path):
    run = _build_good_run(tmp_path)
    run["returncode"] = 1
    with pytest.raises(AssertionError, match="exited"):
        ugb.check_structurally_sound(run)


def test_check_structurally_sound_catches_missing_camera_in_poses(tmp_path):
    run = _build_good_run(tmp_path)
    data = json.loads(run["L"]["transforms_refined"].read_text())
    data["frames"] = [f for f in data["frames"] if f["camera_label"] != "Camera_0001"]
    run["L"]["transforms_refined"].write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        ugb.check_structurally_sound(run)


def test_check_structurally_sound_catches_missing_sogst(tmp_path):
    run = _build_good_run(tmp_path)
    run["L"]["sogst_out"].unlink()
    with pytest.raises(AssertionError, match="ZIP"):
        ugb.check_structurally_sound(run)


def test_measure_extracts_all_metrics_from_a_good_run(tmp_path):
    run = _build_good_run(tmp_path, with_metrics_data=True)
    metrics = ugb.measure(run)

    assert metrics["median_reprojection_error_px"] == 5.0
    assert metrics["mask_coverage"].keys() == set(REAL_CAMERAS)
    assert all(frac == pytest.approx(0.10) for frac in metrics["mask_coverage"].values())
    assert metrics["eval4d_psnr_db"] == 27.5
    assert metrics["eval4d_ssim"] == 0.91
    assert metrics["eval4d_lpips"] == 0.08


def test_measure_raises_when_reprojection_line_missing(tmp_path):
    run = _build_good_run(tmp_path, with_metrics_data=True)
    run["log"] = "no residual line in this log at all"
    with pytest.raises(AssertionError, match="Median residual"):
        ugb.measure(run)


def test_write_golden_baseline_preserves_docs_and_updates_only_metrics(tmp_path, monkeypatch):
    golden_path = tmp_path / "golden_baseline.json"
    original = {
        "_comment": "hand-written documentation, must survive",
        "run_date": "2020-01-01",
        "reproducibility_check": "also hand-written, must survive",
        "median_reprojection_error_px": 1.0,
        "mask_coverage": {"old": 0.5},
        "eval4d_notes": "must survive too",
    }
    golden_path.write_text(json.dumps(original))
    monkeypatch.setattr(ugb, "GOLDEN_PATH", golden_path)

    metrics = {"median_reprojection_error_px": 5.57, "mask_coverage": {"0001": 0.0157},
               "eval4d_psnr_db": 27.5, "eval4d_ssim": 0.91, "eval4d_lpips": 0.08}
    ugb.write_golden_baseline(metrics, "abc123")

    written = json.loads(golden_path.read_text())
    assert written["_comment"] == original["_comment"]
    assert written["reproducibility_check"] == original["reproducibility_check"]
    assert written["eval4d_notes"] == original["eval4d_notes"]
    assert written["median_reprojection_error_px"] == 5.57
    assert written["mask_coverage"] == {"0001": 0.0157}
    assert written["eval4d_psnr_db"] == 27.5
    assert written["eval4d_holdout"]["eval_camera"] == t.EVAL_CAMERA
    assert written["eval4d_config"]["total_train_iters"] == t.TOTAL_TRAIN_ITERS
    assert written["source_commit"] == "abc123"
    assert written["run_date"] != "2020-01-01"


def test_append_history_creates_file_then_appends_without_touching_prior_lines(tmp_path, monkeypatch):
    history_path = tmp_path / "baseline_history.jsonl"
    monkeypatch.setattr(ugb, "HISTORY_PATH", history_path)

    metrics_1 = {"median_reprojection_error_px": 5.57, "mask_coverage": {"0001": 0.01, "0002": 0.03},
                 "eval4d_psnr_db": 27.5, "eval4d_ssim": 0.91, "eval4d_lpips": 0.08}
    ugb.append_history(metrics_1, "commit1")
    assert history_path.is_file()
    lines = history_path.read_text().splitlines()
    assert len(lines) == 1
    entry_1 = json.loads(lines[0])
    assert entry_1["commit"] == "commit1"
    assert entry_1["mask_coverage_avg"] == pytest.approx(0.02)

    metrics_2 = {"median_reprojection_error_px": 5.60, "mask_coverage": {"0001": 0.02},
                 "eval4d_psnr_db": 27.9, "eval4d_ssim": 0.92, "eval4d_lpips": 0.07}
    ugb.append_history(metrics_2, "commit2")
    lines = history_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == entry_1  # first line untouched, not rewritten
    assert json.loads(lines[1])["commit"] == "commit2"
