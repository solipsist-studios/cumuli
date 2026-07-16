import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL")

import numpy as np
from PIL import Image

import run_unified_pipeline as unified
import validate_stage_output as vso

REAL_CAMERAS = ["0001", "0002", "0003"]
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def make_transform(label, tx=0.0, ty=0.0, tz=0.0):
    m = [row[:] for row in IDENTITY]
    m[0][3], m[1][3], m[2][3] = tx, ty, tz
    return {"camera_label": label, "transform_matrix": m}


# --------------------------------------------------------------------------
# validate_sync
# --------------------------------------------------------------------------

def test_validate_sync_passes_when_complete(tmp_path):
    L = unified.build_layout(tmp_path)
    L["sync_grid"].parent.mkdir(parents=True, exist_ok=True)
    L["sync_grid"].write_bytes(b"jpg")
    L["sync_offsets"].write_text(json.dumps({"offsets": {f"{c}.mp4": {} for c in REAL_CAMERAS}}))

    vso.validate_sync(L, REAL_CAMERAS)  # should not raise


def test_validate_sync_fails_when_grid_missing(tmp_path):
    L = unified.build_layout(tmp_path)
    L["sync_offsets"].parent.mkdir(parents=True, exist_ok=True)
    L["sync_offsets"].write_text(json.dumps({"offsets": {}}))

    with pytest.raises(vso.ValidationError, match="sync_grid"):
        vso.validate_sync(L, REAL_CAMERAS)


def test_validate_sync_fails_when_camera_missing_from_offsets(tmp_path):
    L = unified.build_layout(tmp_path)
    L["sync_grid"].parent.mkdir(parents=True, exist_ok=True)
    L["sync_grid"].write_bytes(b"jpg")
    # only 2 of 3 real cameras present -- e.g. 0003 dropped mid-sync
    L["sync_offsets"].write_text(json.dumps({"offsets": {"0001.mp4": {}, "0002.mp4": {}}}))

    with pytest.raises(vso.ValidationError, match="0003"):
        vso.validate_sync(L, REAL_CAMERAS)


# --------------------------------------------------------------------------
# validate_production
# --------------------------------------------------------------------------

def write_image(path, w=1920, h=1080):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h)).save(path)


def test_validate_production_passes_when_complete(tmp_path):
    L = unified.build_layout(tmp_path)
    for cam in REAL_CAMERAS:
        write_image(L["production_undist"] / f"{cam}.jpg")

    vso.validate_production(L, REAL_CAMERAS)  # should not raise


def test_validate_production_fails_when_camera_missing(tmp_path):
    L = unified.build_layout(tmp_path)
    for cam in REAL_CAMERAS[:-1]:
        write_image(L["production_undist"] / f"{cam}.jpg")

    with pytest.raises(vso.ValidationError, match="expected one undistorted frame"):
        vso.validate_production(L, REAL_CAMERAS)


def test_validate_production_fails_when_frame_suspiciously_small(tmp_path):
    L = unified.build_layout(tmp_path)
    for cam in REAL_CAMERAS:
        write_image(L["production_undist"] / f"{cam}.jpg", w=100, h=100)

    with pytest.raises(vso.ValidationError, match="suspiciously small"):
        vso.validate_production(L, REAL_CAMERAS)


# --------------------------------------------------------------------------
# validate_poses
# --------------------------------------------------------------------------

def test_validate_poses_passes_with_distinct_camera_positions(tmp_path):
    L = unified.build_layout(tmp_path)
    frames = [make_transform(f"Camera_{c}", tx=i * 1.5) for i, c in enumerate(REAL_CAMERAS)]
    L["transforms_refined"].parent.mkdir(parents=True, exist_ok=True)
    L["transforms_refined"].write_text(json.dumps({"frames": frames}))

    vso.validate_poses(L, REAL_CAMERAS)  # should not raise


def test_validate_poses_fails_on_wrong_frame_count(tmp_path):
    L = unified.build_layout(tmp_path)
    frames = [make_transform("Camera_0001", tx=1.0)]
    L["transforms_refined"].parent.mkdir(parents=True, exist_ok=True)
    L["transforms_refined"].write_text(json.dumps({"frames": frames}))

    with pytest.raises(vso.ValidationError, match="expected 3"):
        vso.validate_poses(L, REAL_CAMERAS)


def test_validate_poses_fails_on_duplicate_camera_label(tmp_path):
    L = unified.build_layout(tmp_path)
    frames = [make_transform("Camera_0001", tx=i) for i in range(3)]  # same label 3x
    L["transforms_refined"].parent.mkdir(parents=True, exist_ok=True)
    L["transforms_refined"].write_text(json.dumps({"frames": frames}))

    with pytest.raises(vso.ValidationError, match="duplicate camera_label"):
        vso.validate_poses(L, REAL_CAMERAS)


def test_validate_poses_fails_on_non_finite_transform(tmp_path):
    L = unified.build_layout(tmp_path)
    frames = [make_transform(f"Camera_{c}", tx=i) for i, c in enumerate(REAL_CAMERAS)]
    frames[0]["transform_matrix"][0][3] = float("nan")
    L["transforms_refined"].parent.mkdir(parents=True, exist_ok=True)
    L["transforms_refined"].write_text(json.dumps({"frames": frames}))

    with pytest.raises(vso.ValidationError, match="invalid transform_matrix"):
        vso.validate_poses(L, REAL_CAMERAS)


def test_validate_poses_fails_when_all_cameras_collapse_to_same_position(tmp_path):
    L = unified.build_layout(tmp_path)
    # bug scenario: pose solve/refinement silently fell back to the identity
    # bootstrap pose for every camera instead of actually solving anything
    frames = [make_transform(f"Camera_{c}") for c in REAL_CAMERAS]
    L["transforms_refined"].parent.mkdir(parents=True, exist_ok=True)
    L["transforms_refined"].write_text(json.dumps({"frames": frames}))

    with pytest.raises(vso.ValidationError, match="suspiciously identical"):
        vso.validate_poses(L, REAL_CAMERAS)


# --------------------------------------------------------------------------
# validate_masks
# --------------------------------------------------------------------------

def write_mask(path, foreground_frac):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 100, 100
    arr = np.zeros((h, w), dtype=np.uint8)
    n_fg = int(foreground_frac * h * w)
    arr.flat[:n_fg] = 255
    Image.fromarray(arr, mode="L").save(path)


def test_validate_masks_passes_with_plausible_coverage(tmp_path):
    L = unified.build_layout(tmp_path)
    labels = ["00", "01", "02"]
    L["flat_transforms"].parent.mkdir(parents=True, exist_ok=True)
    L["flat_transforms"].write_text(json.dumps({"frames": [{"camera_label": l} for l in labels]}))
    for label in labels:
        write_mask(L["flat_fmasks_clean"] / f"{label}.png", foreground_frac=0.15)

    vso.validate_masks(L, REAL_CAMERAS)  # should not raise


def test_validate_masks_fails_when_mask_missing(tmp_path):
    L = unified.build_layout(tmp_path)
    labels = ["00", "01", "02"]
    L["flat_transforms"].parent.mkdir(parents=True, exist_ok=True)
    L["flat_transforms"].write_text(json.dumps({"frames": [{"camera_label": l} for l in labels]}))
    for label in labels[:-1]:
        write_mask(L["flat_fmasks_clean"] / f"{label}.png", foreground_frac=0.15)

    with pytest.raises(vso.ValidationError, match="missing cleaned mask"):
        vso.validate_masks(L, REAL_CAMERAS)


def test_validate_masks_fails_when_subject_dropped_entirely(tmp_path):
    L = unified.build_layout(tmp_path)
    labels = ["00", "01", "02"]
    L["flat_transforms"].parent.mkdir(parents=True, exist_ok=True)
    L["flat_transforms"].write_text(json.dumps({"frames": [{"camera_label": l} for l in labels]}))
    for label in labels:
        write_mask(L["flat_fmasks_clean"] / f"{label}.png", foreground_frac=0.0)

    with pytest.raises(vso.ValidationError, match="implausible mask coverage"):
        vso.validate_masks(L, REAL_CAMERAS)


def test_validate_masks_fails_when_mask_keeps_background_instead(tmp_path):
    L = unified.build_layout(tmp_path)
    labels = ["00"]
    L["flat_transforms"].parent.mkdir(parents=True, exist_ok=True)
    L["flat_transforms"].write_text(json.dumps({"frames": [{"camera_label": l} for l in labels]}))
    write_mask(L["flat_fmasks_clean"] / "00.png", foreground_frac=0.99)

    with pytest.raises(vso.ValidationError, match="implausible mask coverage"):
        vso.validate_masks(L, ["0001"])


# --------------------------------------------------------------------------
# validate_branch
# --------------------------------------------------------------------------

def write_sparse(sparse_dir, n_cams, n_points):
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (sparse_dir / "cameras.txt").write_text("# header\n" + "\n".join(f"{i} PINHOLE" for i in range(n_cams)))
    img_lines = "\n".join(f"{i} IMAGE_LINE\n" for i in range(n_cams))  # + blank points2D line each
    (sparse_dir / "images.txt").write_text("# header\n" + img_lines)
    (sparse_dir / "points3D.txt").write_text(
        "# header\n" + "\n".join(f"{i} 0 0 0 128 128 128 1.0" for i in range(n_points)))


def test_validate_branch_passes_when_complete(tmp_path):
    L = unified.build_layout(tmp_path)
    write_sparse(L["train_set"] / "sparse" / "0", n_cams=3, n_points=100)
    L["brush_output"].mkdir(parents=True, exist_ok=True)
    (L["brush_output"] / "run_30000.ply").write_bytes(b"ply")

    vso.validate_branch(L, REAL_CAMERAS)  # should not raise


def test_validate_branch_fails_on_missing_sparse_file(tmp_path):
    L = unified.build_layout(tmp_path)
    sparse_dir = L["train_set"] / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (sparse_dir / "cameras.txt").write_text("# header\n")
    # images.txt / points3D.txt missing entirely

    with pytest.raises(vso.ValidationError, match="was not produced"):
        vso.validate_branch(L, REAL_CAMERAS)


def test_validate_branch_fails_on_zero_triangulated_points(tmp_path):
    L = unified.build_layout(tmp_path)
    write_sparse(L["train_set"] / "sparse" / "0", n_cams=3, n_points=0)
    L["brush_output"].mkdir(parents=True, exist_ok=True)
    (L["brush_output"] / "run_30000.ply").write_bytes(b"ply")

    with pytest.raises(vso.ValidationError, match="zero triangulated points"):
        vso.validate_branch(L, REAL_CAMERAS)


def test_validate_branch_fails_when_no_ply_exported(tmp_path):
    L = unified.build_layout(tmp_path)
    write_sparse(L["train_set"] / "sparse" / "0", n_cams=3, n_points=100)
    L["brush_output"].mkdir(parents=True, exist_ok=True)
    # no .ply written

    with pytest.raises(vso.ValidationError, match="no .ply exported"):
        vso.validate_branch(L, REAL_CAMERAS)


# --------------------------------------------------------------------------
# main()'s CLI wiring
# --------------------------------------------------------------------------

def test_main_exits_1_and_prints_error_on_validation_failure(tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "argv", [
        "prog", "--stage", "sync", "--out_dir", str(tmp_path), "--real_cameras", "0001,0002",
    ])
    with pytest.raises(SystemExit) as exc_info:
        vso.main()
    assert exc_info.value.code == 1
    assert "failed validation" in capsys.readouterr().out


def test_main_exits_0_and_prints_ok_on_success(tmp_path, monkeypatch, capsys):
    import sys
    L = unified.build_layout(tmp_path)
    L["sync_grid"].parent.mkdir(parents=True, exist_ok=True)
    L["sync_grid"].write_bytes(b"jpg")
    L["sync_offsets"].write_text(json.dumps({"offsets": {"0001.mp4": {}, "0002.mp4": {}}}))

    monkeypatch.setattr(sys, "argv", [
        "prog", "--stage", "sync", "--out_dir", str(tmp_path), "--real_cameras", "0001,0002",
    ])
    vso.main()  # should not raise / not exit

    assert "OK" in capsys.readouterr().out
