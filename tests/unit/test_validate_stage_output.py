# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

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
    L["flat_transforms"].write_text(json.dumps({"frames": [{"camera_label": label} for label in labels]}))
    for label in labels:
        write_mask(L["flat_fmasks_clean"] / f"{label}.png", foreground_frac=0.15)

    vso.validate_masks(L, REAL_CAMERAS)  # should not raise


def test_validate_masks_fails_when_mask_missing(tmp_path):
    L = unified.build_layout(tmp_path)
    labels = ["00", "01", "02"]
    L["flat_transforms"].parent.mkdir(parents=True, exist_ok=True)
    L["flat_transforms"].write_text(json.dumps({"frames": [{"camera_label": label} for label in labels]}))
    for label in labels[:-1]:
        write_mask(L["flat_fmasks_clean"] / f"{label}.png", foreground_frac=0.15)

    with pytest.raises(vso.ValidationError, match="missing cleaned mask"):
        vso.validate_masks(L, REAL_CAMERAS)


def test_validate_masks_fails_when_subject_dropped_entirely(tmp_path):
    L = unified.build_layout(tmp_path)
    labels = ["00", "01", "02"]
    L["flat_transforms"].parent.mkdir(parents=True, exist_ok=True)
    L["flat_transforms"].write_text(json.dumps({"frames": [{"camera_label": label} for label in labels]}))
    for label in labels:
        write_mask(L["flat_fmasks_clean"] / f"{label}.png", foreground_frac=0.0)

    with pytest.raises(vso.ValidationError, match="implausible mask coverage"):
        vso.validate_masks(L, REAL_CAMERAS)


def test_validate_masks_fails_when_mask_keeps_background_instead(tmp_path):
    L = unified.build_layout(tmp_path)
    labels = ["00"]
    L["flat_transforms"].parent.mkdir(parents=True, exist_ok=True)
    L["flat_transforms"].write_text(json.dumps({"frames": [{"camera_label": label} for label in labels]}))
    write_mask(L["flat_fmasks_clean"] / "00.png", foreground_frac=0.99)

    with pytest.raises(vso.ValidationError, match="implausible mask coverage"):
        vso.validate_masks(L, ["0001"])


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
    L["flat_transforms"].write_text(json.dumps({"frames": [{"camera_label": label} for label in labels]}))


def test_validate_masks_fails_on_duplicate_labels(tmp_path):
    L = unified.build_layout(tmp_path)
    write_flat(L, ["00", "00", "01"])  # 3 frames, but a camera flattened twice
    for label in ("00", "01"):
        write_mask(L["flat_fmasks_clean"] / f"{label}.png", foreground_frac=0.15)

    with pytest.raises(vso.ValidationError, match="duplicate camera_label"):
        vso.validate_masks(L, REAL_CAMERAS)


@pytest.mark.parametrize("frac,ok", [
    (0.01, True),   # just inside the plausible band, low side
    (0.004, False), # just outside, low side
    (0.85, True),   # just inside, high side
    (0.95, False),  # just outside, high side
])
def test_validate_masks_boundary_coverage(tmp_path, frac, ok):
    L = unified.build_layout(tmp_path)
    write_flat(L, ["00"])
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


# --------------------------------------------------------------------------
# validate_dataset4d / validate_train4d
# --------------------------------------------------------------------------

def layout4d(tmp_path):
    """Layout dict for the 4D validators. setdefault keeps this correct both
    before and after build_layout() itself grows these keys."""
    L = dict(unified.build_layout(tmp_path))
    L.setdefault("dataset4d", tmp_path / "dataset_4dgs")
    L.setdefault("train4d_model", tmp_path / "train4d_output")
    L.setdefault("sogst_out", tmp_path / "splat_4d.sogst")
    return L


def frame_entry(label, i, with_intrinsics=True):
    entry = {
        "file_path": f"realcams/cam{label}/frame_{i + 1:05d}",
        "camera_label": label,
        "time": i / 30.0,
        "w": 8, "h": 8,
        "transform_matrix": IDENTITY,
    }
    if with_intrinsics:
        entry.update({"fl_x": 4.0, "fl_y": 4.0, "cx": 4.0, "cy": 4.0})
    return entry


def write_time_ply(path, n_vertices, with_time=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    props = "property float x\nproperty float y\nproperty float z\n"
    if with_time:
        props += "property float time\n"
    header = (f"ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n_vertices}\n{props}end_header\n")
    width = 4 if with_time else 3
    path.write_bytes(header.encode("ascii") + b"\x00" * (4 * width * n_vertices))


def write_dataset4d(root, image_mode="RGBA", with_intrinsics=True,
                    with_time=True, n_frames=2):
    frames = [frame_entry("00", i, with_intrinsics) for i in range(n_frames)]
    root.mkdir(parents=True, exist_ok=True)
    (root / "transforms_train.json").write_text(
        json.dumps({"camera_model": "OPENCV", "frames": frames}))
    (root / "transforms_test.json").write_text(
        json.dumps({"camera_model": "OPENCV", "frames": frames[:1]}))
    img = root / (frames[0]["file_path"] + ".png")
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.new(image_mode, (8, 8)).save(img)
    write_time_ply(root / "points3d.ply", n_vertices=10, with_time=with_time)
    return frames


def test_validate_dataset4d_passes_when_complete(tmp_path):
    L = layout4d(tmp_path)
    write_dataset4d(L["dataset4d"])

    vso.validate_dataset4d(L, REAL_CAMERAS)  # should not raise


def test_validate_dataset4d_fails_on_missing_transforms(tmp_path):
    L = layout4d(tmp_path)
    L["dataset4d"].mkdir(parents=True, exist_ok=True)

    with pytest.raises(vso.ValidationError, match="was not produced"):
        vso.validate_dataset4d(L, REAL_CAMERAS)


def test_validate_dataset4d_fails_on_invalid_json(tmp_path):
    L = layout4d(tmp_path)
    write_dataset4d(L["dataset4d"])
    (L["dataset4d"] / "transforms_train.json").write_text("{not json")

    with pytest.raises(vso.ValidationError, match="not valid JSON"):
        vso.validate_dataset4d(L, REAL_CAMERAS)


def test_validate_dataset4d_fails_without_per_view_intrinsics(tmp_path):
    L = layout4d(tmp_path)
    write_dataset4d(L["dataset4d"], with_intrinsics=False)

    with pytest.raises(vso.ValidationError, match="lack 'fl_x'"):
        vso.validate_dataset4d(L, REAL_CAMERAS)


def test_validate_dataset4d_fails_on_non_rgba_image(tmp_path):
    L = layout4d(tmp_path)
    write_dataset4d(L["dataset4d"], image_mode="RGB")

    with pytest.raises(vso.ValidationError, match="expected RGBA"):
        vso.validate_dataset4d(L, REAL_CAMERAS)


def test_validate_dataset4d_fails_without_time_property(tmp_path):
    L = layout4d(tmp_path)
    write_dataset4d(L["dataset4d"], with_time=False)

    with pytest.raises(vso.ValidationError, match="'time' property"):
        vso.validate_dataset4d(L, REAL_CAMERAS)


def _write_sogst_zip(path, count=5, fmt="sogst"):
    import zipfile
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("meta.json", json.dumps({"version": 1, "format": fmt, "count": count}))


def test_validate_train4d_passes_when_complete(tmp_path):
    L = layout4d(tmp_path)
    L["train4d_model"].mkdir(parents=True, exist_ok=True)
    (L["train4d_model"] / "chkpnt2000.pth").write_bytes(b"\x00" * 2_000_000)
    _write_sogst_zip(L["sogst_out"])

    vso.validate_train4d(L, REAL_CAMERAS)  # should not raise


def test_validate_train4d_fails_without_checkpoint(tmp_path):
    L = layout4d(tmp_path)
    L["train4d_model"].mkdir(parents=True, exist_ok=True)
    _write_sogst_zip(L["sogst_out"])

    with pytest.raises(vso.ValidationError, match="no chkpnt"):
        vso.validate_train4d(L, REAL_CAMERAS)


def test_validate_train4d_fails_on_undersized_checkpoint(tmp_path):
    L = layout4d(tmp_path)
    L["train4d_model"].mkdir(parents=True, exist_ok=True)
    (L["train4d_model"] / "chkpnt2000.pth").write_bytes(b"\x00" * 1024)
    _write_sogst_zip(L["sogst_out"])

    with pytest.raises(vso.ValidationError, match="truncated checkpoint"):
        vso.validate_train4d(L, REAL_CAMERAS)


def test_validate_train4d_fails_on_invalid_zip(tmp_path):
    L = layout4d(tmp_path)
    L["train4d_model"].mkdir(parents=True, exist_ok=True)
    (L["train4d_model"] / "chkpnt2000.pth").write_bytes(b"\x00" * 2_000_000)
    L["sogst_out"].write_bytes(b"not a zip")

    with pytest.raises(vso.ValidationError, match="not a readable .sogst"):
        vso.validate_train4d(L, REAL_CAMERAS)


def test_validate_train4d_fails_on_zero_splats(tmp_path):
    L = layout4d(tmp_path)
    L["train4d_model"].mkdir(parents=True, exist_ok=True)
    (L["train4d_model"] / "chkpnt2000.pth").write_bytes(b"\x00" * 2_000_000)
    _write_sogst_zip(L["sogst_out"], count=0)

    with pytest.raises(vso.ValidationError, match="zero splats"):
        vso.validate_train4d(L, REAL_CAMERAS)


def test_validate_train4d_fails_on_wrong_format(tmp_path):
    L = layout4d(tmp_path)
    L["train4d_model"].mkdir(parents=True, exist_ok=True)
    (L["train4d_model"] / "chkpnt2000.pth").write_bytes(b"\x00" * 2_000_000)
    _write_sogst_zip(L["sogst_out"], fmt="sog")

    with pytest.raises(vso.ValidationError, match="expected 'sogst'"):
        vso.validate_train4d(L, REAL_CAMERAS)
