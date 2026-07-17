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
    L["sync_offsets"].write_text(json.dumps(
        {"offsets": {f"{c}.mp4": {"fps": 30.0, "frame_offset": 2} for c in REAL_CAMERAS}}))

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

def write_ply(path, n_vertices):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n_vertices}\nproperty float x\nend_header\n")
    path.write_bytes(header.encode("ascii") + b"\x00" * (4 * n_vertices))


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
    write_ply(L["brush_output"] / "run_30000.ply", n_vertices=5000)

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
    write_ply(L["brush_output"] / "run_30000.ply", n_vertices=5000)

    with pytest.raises(vso.ValidationError, match="zero triangulated points"):
        vso.validate_branch(L, REAL_CAMERAS)


def test_validate_branch_fails_when_no_ply_exported(tmp_path):
    L = unified.build_layout(tmp_path)
    write_sparse(L["train_set"] / "sparse" / "0", n_cams=3, n_points=100)
    L["brush_output"].mkdir(parents=True, exist_ok=True)
    # no .ply written

    with pytest.raises(vso.ValidationError, match="no .ply exported"):
        vso.validate_branch(L, REAL_CAMERAS)


def test_validate_branch_fails_on_truncated_ply(tmp_path):
    # Brush crashing mid-write still leaves a .ply on disk -- an existence
    # check alone would pass a 0-byte/garbage export.
    L = unified.build_layout(tmp_path)
    write_sparse(L["train_set"] / "sparse" / "0", n_cams=3, n_points=100)
    L["brush_output"].mkdir(parents=True, exist_ok=True)
    (L["brush_output"] / "run_30000.ply").write_bytes(b"")

    with pytest.raises(vso.ValidationError, match="zero splats"):
        vso.validate_branch(L, REAL_CAMERAS)


def test_validate_branch_fails_on_zero_vertex_ply(tmp_path):
    L = unified.build_layout(tmp_path)
    write_sparse(L["train_set"] / "sparse" / "0", n_cams=3, n_points=100)
    write_ply(L["brush_output"] / "run_30000.ply", n_vertices=0)

    with pytest.raises(vso.ValidationError, match="zero splats"):
        vso.validate_branch(L, REAL_CAMERAS)


def test_validate_branch_checks_newest_ply_not_lexically_last(tmp_path):
    # run_5000.ply sorts lexically AFTER run_30000.ply -- the check must go
    # by mtime, so a good final export isn't masked (or falsely failed) by
    # checkpoint naming.
    import os
    L = unified.build_layout(tmp_path)
    write_sparse(L["train_set"] / "sparse" / "0", n_cams=3, n_points=100)
    write_ply(L["brush_output"] / "run_5000.ply", n_vertices=0)  # early, truncated
    write_ply(L["brush_output"] / "run_30000.ply", n_vertices=5000)  # final, good
    os.utime(L["brush_output"] / "run_5000.ply", (1_000_000, 1_000_000))
    os.utime(L["brush_output"] / "run_30000.ply", (2_000_000, 2_000_000))

    vso.validate_branch(L, REAL_CAMERAS)  # should not raise


# --------------------------------------------------------------------------
# validate_sync -- entry content, not just name coverage
# --------------------------------------------------------------------------

def write_sync(L, entries):
    L["sync_grid"].parent.mkdir(parents=True, exist_ok=True)
    L["sync_grid"].write_bytes(b"jpg")
    L["sync_offsets"].write_text(json.dumps({"offsets": entries}))


GOOD_ENTRY = {"fps": 30.0, "frame_offset": 4, "offset_seconds": 0.1333}


def test_validate_sync_fails_on_error_stub_entry(tmp_path):
    # compute_sync_offsets.py keeps a camera whose audio extraction failed in
    # the json as {"error": ...} -- name coverage alone would pass it.
    L = unified.build_layout(tmp_path)
    write_sync(L, {"0001.mp4": dict(GOOD_ENTRY), "0002.mp4": dict(GOOD_ENTRY),
                   "0003.mp4": {"error": "ffmpeg exited 1"}})

    with pytest.raises(vso.ValidationError, match="no usable sync data"):
        vso.validate_sync(L, REAL_CAMERAS)


def test_validate_sync_fails_on_entry_missing_frame_offset(tmp_path):
    L = unified.build_layout(tmp_path)
    write_sync(L, {"0001.mp4": dict(GOOD_ENTRY), "0002.mp4": dict(GOOD_ENTRY),
                   "0003.mp4": {"fps": 30.0}})

    with pytest.raises(vso.ValidationError, match="missing frame_offset"):
        vso.validate_sync(L, REAL_CAMERAS)


def test_validate_sync_fails_on_degenerate_fps(tmp_path):
    L = unified.build_layout(tmp_path)
    write_sync(L, {"0001.mp4": dict(GOOD_ENTRY), "0002.mp4": dict(GOOD_ENTRY),
                   "0003.mp4": {"fps": 0.0, "frame_offset": 4}})

    with pytest.raises(vso.ValidationError, match="degenerate"):
        vso.validate_sync(L, REAL_CAMERAS)


def test_validate_sync_passes_with_full_entries(tmp_path):
    L = unified.build_layout(tmp_path)
    write_sync(L, {f"{c}.mp4": dict(GOOD_ENTRY) for c in REAL_CAMERAS})

    vso.validate_sync(L, REAL_CAMERAS)  # should not raise


# --------------------------------------------------------------------------
# validate_production -- .png frames from a --pp3_dir color-corrected run
# --------------------------------------------------------------------------

def test_validate_production_accepts_png_frames_from_pp3_run(tmp_path):
    # Regression test: with --pp3_dir, run_unified_pipeline.py's image_ext is
    # .png, so undistort_frames.py writes <cam>.png -- a hard-coded *.jpg
    # glob falsely failed (and aborted) every color-corrected run.
    L = unified.build_layout(tmp_path)
    for cam in REAL_CAMERAS:
        write_image(L["production_undist"] / f"{cam}.png")

    vso.validate_production(L, REAL_CAMERAS)  # should not raise


# --------------------------------------------------------------------------
# validate_poses -- rotation validity, not just position spread
# --------------------------------------------------------------------------

def write_poses(L, frames):
    L["transforms_refined"].parent.mkdir(parents=True, exist_ok=True)
    L["transforms_refined"].write_text(json.dumps({"frames": frames}))


def test_validate_poses_fails_on_bad_bottom_row(tmp_path):
    L = unified.build_layout(tmp_path)
    frames = [make_transform(f"Camera_{c}", tx=i * 1.5) for i, c in enumerate(REAL_CAMERAS)]
    frames[1]["transform_matrix"][3] = [0.1, 0.0, 0.0, 1.0]
    write_poses(L, frames)

    with pytest.raises(vso.ValidationError, match="bottom row"):
        vso.validate_poses(L, REAL_CAMERAS)


def test_validate_poses_fails_on_non_rotation_block(tmp_path):
    # Distinct positions but a zeroed rotation block: the old spread-only
    # check passed this outright.
    L = unified.build_layout(tmp_path)
    frames = [make_transform(f"Camera_{c}", tx=i * 1.5) for i, c in enumerate(REAL_CAMERAS)]
    for row in range(3):
        for col in range(3):
            frames[0]["transform_matrix"][row][col] = 0.0
    write_poses(L, frames)

    with pytest.raises(vso.ValidationError, match="not a proper rotation"):
        vso.validate_poses(L, REAL_CAMERAS)


def test_validate_poses_fails_on_reflected_rotation(tmp_path):
    # det = -1: a mirrored "camera" cannot come out of a sane pose solve.
    L = unified.build_layout(tmp_path)
    frames = [make_transform(f"Camera_{c}", tx=i * 1.5) for i, c in enumerate(REAL_CAMERAS)]
    frames[2]["transform_matrix"][0][0] = -1.0
    write_poses(L, frames)

    with pytest.raises(vso.ValidationError, match="not a proper rotation"):
        vso.validate_poses(L, REAL_CAMERAS)


def test_validate_poses_passes_with_real_rotations(tmp_path):
    # Cameras on a ring looking inward: genuine non-identity rotations must
    # not trip the orthonormality check.
    L = unified.build_layout(tmp_path)
    frames = []
    for i, c in enumerate(REAL_CAMERAS):
        theta = 2 * np.pi * i / len(REAL_CAMERAS)
        c_, s_ = np.cos(theta), np.sin(theta)
        m = [[c_, 0.0, s_, 3.0 * s_],
             [0.0, 1.0, 0.0, 1.2],
             [-s_, 0.0, c_, 3.0 * c_],
             [0.0, 0.0, 0.0, 1.0]]
        frames.append({"camera_label": f"Camera_{c}", "transform_matrix": m})
    write_poses(L, frames)

    vso.validate_poses(L, REAL_CAMERAS)  # should not raise


# --------------------------------------------------------------------------
# validate_masks -- boundary coverage and duplicate labels
# --------------------------------------------------------------------------

def write_flat(L, labels):
    L["flat_transforms"].parent.mkdir(parents=True, exist_ok=True)
    L["flat_transforms"].write_text(json.dumps({"frames": [{"camera_label": l} for l in labels]}))


def test_validate_masks_fails_on_duplicate_labels(tmp_path):
    L = unified.build_layout(tmp_path)
    write_flat(L, ["00", "00", "01"])  # 3 frames, but a camera flattened twice
    for label in ("00", "01"):
        write_mask(L["flat_fmasks_clean"] / f"{label}.png", foreground_frac=0.15)

    with pytest.raises(vso.ValidationError, match="duplicate camera_label"):
        vso.validate_masks(L, REAL_CAMERAS)


def test_validate_masks_boundary_coverage(tmp_path):
    # Just inside the plausible band passes; just outside fails, both ends.
    L = unified.build_layout(tmp_path)
    write_flat(L, ["00"])

    for frac, ok in ((0.01, True), (0.004, False), (0.85, True), (0.95, False)):
        write_mask(L["flat_fmasks_clean"] / "00.png", foreground_frac=frac)
        if ok:
            vso.validate_masks(L, ["0001"])
        else:
            with pytest.raises(vso.ValidationError, match="implausible mask coverage"):
                vso.validate_masks(L, ["0001"])


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
    L["sync_offsets"].write_text(json.dumps(
        {"offsets": {"0001.mp4": {"fps": 30.0, "frame_offset": 0},
                     "0002.mp4": {"fps": 30.0, "frame_offset": 3}}}))

    monkeypatch.setattr(sys, "argv", [
        "prog", "--stage", "sync", "--out_dir", str(tmp_path), "--real_cameras", "0001,0002",
    ])
    vso.main()  # should not raise / not exit

    assert "OK" in capsys.readouterr().out
