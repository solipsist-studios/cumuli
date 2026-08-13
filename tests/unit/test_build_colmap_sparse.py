# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

import json
import sys

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

import build_colmap_sparse as bcs


# --------------------------------------------------------------------------
# opengl_c2w_to_colmap_w2c -- real coordinate-conversion math
# --------------------------------------------------------------------------

def test_opengl_c2w_to_colmap_w2c_identity():
    # Negating Y/Z columns of an identity c2w gives diag(1,-1,-1,1), which
    # is its own inverse -- an exact, hand-derivable expected value.
    c2w = np.eye(4)
    R, t = bcs.opengl_c2w_to_colmap_w2c(c2w)
    assert np.allclose(R, np.diag([1.0, -1.0, -1.0]))
    assert np.allclose(t, [0.0, 0.0, 0.0])


def test_opengl_c2w_to_colmap_w2c_does_not_mutate_input():
    c2w = np.eye(4)
    original = c2w.copy()
    bcs.opengl_c2w_to_colmap_w2c(c2w)
    assert np.array_equal(c2w, original)


@given(
    quat=st.tuples(*[st.floats(-1, 1, allow_nan=False)] * 4).filter(lambda q: np.linalg.norm(q) > 1e-6),
    translation=st.tuples(*[st.floats(-100, 100, allow_nan=False)] * 3),
)
@settings(max_examples=40, deadline=None)
def test_opengl_c2w_to_colmap_w2c_round_trips_for_any_pose(quat, translation):
    # Property test: negate-then-invert must be its own algebraic inverse
    # when undone in reverse order (invert-then-negate), for any valid
    # rotation/translation -- not just the one hand-picked identity case.
    R_c2w = Rotation.from_quat(np.array(quat) / np.linalg.norm(quat)).as_matrix()
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = translation

    R_w2c, t_w2c = bcs.opengl_c2w_to_colmap_w2c(c2w)
    w2c = np.eye(4)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c

    recovered_c2w_cv = np.linalg.inv(w2c)
    recovered_c2w = recovered_c2w_cv.copy()
    recovered_c2w[:3, 1] *= -1
    recovered_c2w[:3, 2] *= -1

    assert np.allclose(recovered_c2w, c2w, atol=1e-6)


# --------------------------------------------------------------------------
# validate_frame -- pure logic, no mocking
# --------------------------------------------------------------------------

def make_valid_frame(label="00"):
    return {
        "camera_label": label,
        "file_path": f"images_flat/{label}.png",
        "transform_matrix": np.eye(4).tolist(),
        "w": 100, "h": 100, "fl_x": 50, "fl_y": 50, "cx": 50, "cy": 50,
    }


def test_validate_frame_accepts_well_formed_frame():
    assert bcs.validate_frame(make_valid_frame()) is None


@pytest.mark.parametrize("missing_key", list(bcs.REQUIRED_CAMERA_KEYS) + ["transform_matrix", "file_path"])
def test_validate_frame_rejects_each_missing_key(missing_key):
    fr = make_valid_frame()
    del fr[missing_key]
    reason = bcs.validate_frame(fr)
    assert reason is not None
    assert missing_key in reason


def test_validate_frame_rejects_singular_transform_matrix():
    fr = make_valid_frame()
    fr["transform_matrix"] = np.zeros((4, 4)).tolist()
    reason = bcs.validate_frame(fr)
    assert reason is not None
    assert "not invertible" in reason


def test_validate_frame_reports_all_missing_keys_at_once():
    fr = {"camera_label": "00"}
    reason = bcs.validate_frame(fr)
    for key in bcs.REQUIRED_CAMERA_KEYS:
        assert key in reason
    assert "transform_matrix" in reason
    assert "file_path" in reason


# --------------------------------------------------------------------------
# bake_rgba -- real PIL/numpy math, no mocking
# --------------------------------------------------------------------------

def test_bake_rgba_zeroes_rgb_where_mask_is_zero(tmp_path):
    img = Image.new("RGB", (4, 4), color=(200, 150, 100))
    mask_arr = np.zeros((4, 4), dtype=np.uint8)
    mask_arr[1:3, 1:3] = 255  # foreground only in the center 2x2
    mask = Image.fromarray(mask_arr, mode="L")
    image_path, mask_path, out_path = tmp_path / "img.png", tmp_path / "mask.png", tmp_path / "out.png"
    img.save(image_path)
    mask.save(mask_path)

    bcs.bake_rgba(image_path, mask_path, out_path)

    result = np.array(Image.open(out_path))
    assert result.shape == (4, 4, 4)
    assert tuple(result[0, 0]) == (0, 0, 0, 0)          # background: RGB zeroed, alpha 0
    assert tuple(result[1, 1]) == (200, 150, 100, 255)  # foreground: RGB preserved, alpha 255


def test_bake_rgba_resizes_mismatched_mask_size(tmp_path):
    img = Image.new("RGB", (10, 10), color=(1, 2, 3))
    mask = Image.new("L", (5, 5), color=255)  # different size than the image
    image_path, mask_path, out_path = tmp_path / "img.png", tmp_path / "mask.png", tmp_path / "out.png"
    img.save(image_path)
    mask.save(mask_path)

    bcs.bake_rgba(image_path, mask_path, out_path)

    with Image.open(out_path) as result:
        assert result.size == (10, 10)  # resized to match the image, not left at mask's native size


def test_bake_rgba_creates_parent_dir(tmp_path):
    img = Image.new("RGB", (2, 2))
    mask = Image.new("L", (2, 2), color=255)
    image_path, mask_path = tmp_path / "img.png", tmp_path / "mask.png"
    img.save(image_path)
    mask.save(mask_path)
    out_path = tmp_path / "nested" / "dir" / "out.png"

    bcs.bake_rgba(image_path, mask_path, out_path)
    assert out_path.is_file()


# --------------------------------------------------------------------------
# bake_masks -- per-camera fault isolation
# --------------------------------------------------------------------------

def write_image_and_mask(images_dir, masks_dir, label, ext=".png"):
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), color=(9, 9, 9)).save(images_dir / f"{label}{ext}")
    Image.new("L", (10, 10), color=255).save(masks_dir / f"{label}.png")


def test_bake_masks_bakes_all_present_cameras(tmp_path, capsys):
    images_dir, masks_dir, out_dir = tmp_path / "images", tmp_path / "masks", tmp_path / "out"
    write_image_and_mask(images_dir, masks_dir, "00")
    write_image_and_mask(images_dir, masks_dir, "01")
    frames = [make_valid_frame("00"), make_valid_frame("01")]

    bcs.bake_masks(frames, masks_dir, images_dir, out_dir, "images_rgba")

    assert (out_dir / "images_rgba" / "00.png").is_file()
    assert (out_dir / "images_rgba" / "01.png").is_file()
    assert "Baked 2/2 cameras" in capsys.readouterr().out


def test_bake_masks_missing_image_skips_and_warns(tmp_path, capsys):
    images_dir, masks_dir, out_dir = tmp_path / "images", tmp_path / "masks", tmp_path / "out"
    masks_dir.mkdir(parents=True)
    Image.new("L", (10, 10), color=255).save(masks_dir / "00.png")
    images_dir.mkdir(parents=True)  # no 00.png in images_dir
    frames = [make_valid_frame("00")]

    bcs.bake_masks(frames, masks_dir, images_dir, out_dir, "images_rgba")

    assert not (out_dir / "images_rgba" / "00.png").exists()
    out = capsys.readouterr().out
    assert "Baked 0/1 cameras" in out
    assert "missing image or mask for cameras: ['00']" in out


def test_bake_masks_missing_mask_skips_and_warns(tmp_path, capsys):
    images_dir, masks_dir, out_dir = tmp_path / "images", tmp_path / "masks", tmp_path / "out"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(images_dir / "00.png")
    masks_dir.mkdir(parents=True)  # no 00.png in masks_dir
    frames = [make_valid_frame("00")]

    bcs.bake_masks(frames, masks_dir, images_dir, out_dir, "images_rgba")
    assert "missing image or mask for cameras: ['00']" in capsys.readouterr().out


def test_bake_masks_corrupted_image_treated_as_missing_good_camera_survives(tmp_path, capsys):
    # Regression test: a corrupted source image used to crash the entire
    # batch with a raw UnidentifiedImageError -- confirmed a good second
    # camera never got processed either. Now treated the same as "missing."
    images_dir, masks_dir, out_dir = tmp_path / "images", tmp_path / "masks", tmp_path / "out"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    (images_dir / "00.png").write_bytes(b"not a real png")
    Image.new("L", (10, 10), color=255).save(masks_dir / "00.png")
    write_image_and_mask(images_dir, masks_dir, "01")
    frames = [make_valid_frame("00"), make_valid_frame("01")]

    bcs.bake_masks(frames, masks_dir, images_dir, out_dir, "images_rgba")  # must not raise

    assert not (out_dir / "images_rgba" / "00.png").exists()
    assert (out_dir / "images_rgba" / "01.png").is_file()
    out = capsys.readouterr().out
    assert "could not bake 00" in out
    assert "Baked 1/2 cameras" in out


# --------------------------------------------------------------------------
# write_cameras_txt / write_images_txt / write_points3d_txt
# --------------------------------------------------------------------------

def test_write_cameras_txt_format_and_sequential_ids(tmp_path):
    frames = [make_valid_frame("00"), make_valid_frame("01")]
    frames[1].update(w=200, h=150, fl_x=99, fl_y=98, cx=97, cy=96)
    bcs.write_cameras_txt(tmp_path, frames)
    lines = [ln for ln in (tmp_path / "cameras.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    assert lines[0] == "1 PINHOLE 100 100 50 50 50 50"
    assert lines[1] == "2 PINHOLE 200 150 99 98 97 96"


def test_write_images_txt_identity_pose_and_empty_points2d_line(tmp_path):
    frames = [make_valid_frame("00")]
    bcs.write_images_txt(tmp_path, frames, "images_flat", None, "images_rgba")
    lines = (tmp_path / "images.txt").read_text().splitlines()
    data_lines = [ln for ln in lines if ln and not ln.startswith("#")]
    assert len(data_lines) == 1  # the points2D line is empty (blank), not a "#"-comment
    parts = data_lines[0].split()
    image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts
    assert image_id == camera_id == "1"
    # identity c2w -> w2c = diag(1,-1,-1,1) -> 180 deg rotation about X
    assert pytest.approx(float(qw), abs=1e-6) == 0.0
    assert pytest.approx(abs(float(qx)), abs=1e-6) == 1.0
    assert pytest.approx(float(ty), abs=1e-6) == 0.0
    assert name == "images_flat/00.png"


def test_write_images_txt_uses_rgba_name_when_masks_dir_given(tmp_path):
    frames = [make_valid_frame("00")]
    bcs.write_images_txt(tmp_path, frames, "images_flat", tmp_path, "images_rgba")  # masks_dir just needs to be non-None
    text = (tmp_path / "images.txt").read_text()
    assert "images_rgba/00.png" in text
    assert "images_flat/00.png" not in text.replace("images_rgba/00.png", "")


def test_write_images_txt_sequential_camera_and_image_ids(tmp_path):
    frames = [make_valid_frame("00"), make_valid_frame("01"), make_valid_frame("02")]
    bcs.write_images_txt(tmp_path, frames, "images_flat", None, "images_rgba")
    data_lines = [ln for ln in (tmp_path / "images.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    ids = [(ln.split()[0], ln.split()[8]) for ln in data_lines]
    assert ids == [("1", "1"), ("2", "2"), ("3", "3")]


def write_ply(path, n=3, with_color=False):
    if with_color:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    else:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    verts = np.zeros(n, dtype=dtype)
    verts["x"] = np.arange(n)
    verts["y"] = np.arange(n) * 2
    verts["z"] = np.arange(n) * 3
    if with_color:
        verts["red"] = 10
        verts["green"] = 20
        verts["blue"] = 30
    el = PlyElement.describe(verts, "vertex")
    PlyData([el]).write(str(path))


def test_write_points3d_txt_without_color_uses_gray_placeholder(tmp_path):
    ply_path = tmp_path / "p.ply"
    write_ply(ply_path, n=2, with_color=False)
    verts = bcs.write_points3d_txt(tmp_path, ply_path)
    assert len(verts) == 2
    lines = [ln for ln in (tmp_path / "points3D.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    assert lines[0] == "1 0.0 0.0 0.0 128 128 128 1.0"
    assert lines[1] == "2 1.0 2.0 3.0 128 128 128 1.0"


def test_write_points3d_txt_with_color_uses_real_values(tmp_path):
    ply_path = tmp_path / "p.ply"
    write_ply(ply_path, n=1, with_color=True)
    bcs.write_points3d_txt(tmp_path, ply_path)
    line = [ln for ln in (tmp_path / "points3D.txt").read_text().splitlines() if ln and not ln.startswith("#")][0]
    assert line == "1 0.0 0.0 0.0 10 20 30 1.0"


# --------------------------------------------------------------------------
# main() -- full CLI wiring + regression tests for all bugs found via real
# execution against broken inputs.
# --------------------------------------------------------------------------

def base_argv(transforms, points_ply, out_dir, extra=None):
    argv = ["prog", "--transforms", str(transforms), "--points_ply", str(points_ply), "--out_dir", str(out_dir)]
    return argv + (extra or [])


def write_transforms(path, frames, extra_top_level=None):
    data = {"frames": frames, **(extra_top_level or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_main_errors_cleanly_on_invalid_json(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "bad.json"
    transforms.write_text("not valid json {{{")
    monkeypatch.setattr(sys, "argv", base_argv(transforms, tmp_path / "p.ply", tmp_path / "out"))
    with pytest.raises(SystemExit) as exc_info:
        bcs.main()
    assert exc_info.value.code == 1
    assert "is not valid JSON" in capsys.readouterr().out


def test_main_errors_cleanly_on_missing_frames_key(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [])
    transforms.write_text(json.dumps({"not_frames": []}))
    monkeypatch.setattr(sys, "argv", base_argv(transforms, tmp_path / "p.ply", tmp_path / "out"))
    with pytest.raises(SystemExit) as exc_info:
        bcs.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "is missing expected key" in out
    assert "'frames'" in out


def test_main_errors_cleanly_on_frame_missing_camera_label(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [{"not_camera_label": "x"}])
    monkeypatch.setattr(sys, "argv", base_argv(transforms, tmp_path / "p.ply", tmp_path / "out"))
    with pytest.raises(SystemExit) as exc_info:
        bcs.main()
    assert exc_info.value.code == 1
    assert "'camera_label'" in capsys.readouterr().out


def test_main_errors_cleanly_when_points_ply_missing(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [])
    monkeypatch.setattr(sys, "argv", base_argv(transforms, tmp_path / "nope.ply", tmp_path / "out"))
    with pytest.raises(SystemExit) as exc_info:
        bcs.main()
    assert exc_info.value.code == 1
    assert "not found" in capsys.readouterr().out


def test_main_errors_cleanly_on_corrupted_ply(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [])
    ply_path = tmp_path / "bad.ply"
    ply_path.write_bytes(b"not a real ply file")
    monkeypatch.setattr(sys, "argv", base_argv(transforms, ply_path, tmp_path / "out"))
    with pytest.raises(SystemExit) as exc_info:
        bcs.main()
    assert exc_info.value.code == 1
    assert "could not be read as a point cloud" in capsys.readouterr().out


@pytest.mark.parametrize("bad_frame_fn", [
    lambda fr: fr.pop("transform_matrix"),
    lambda fr: [fr.pop(k) for k in bcs.REQUIRED_CAMERA_KEYS],
    lambda fr: fr.__setitem__("transform_matrix", np.zeros((4, 4)).tolist()),
])
def test_main_skips_malformed_frame_good_camera_survives(tmp_path, monkeypatch, bad_frame_fn):
    transforms = tmp_path / "t.json"
    bad = make_valid_frame("00")
    bad_frame_fn(bad)
    good = make_valid_frame("01")
    write_transforms(transforms, [bad, good])
    ply_path = tmp_path / "p.ply"
    write_ply(ply_path, n=1)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(transforms, ply_path, out_dir))
    bcs.main()  # must not raise

    cameras_txt = (out_dir / "sparse" / "0" / "cameras.txt").read_text()
    data_lines = [ln for ln in cameras_txt.splitlines() if ln and not ln.startswith("#")]
    assert len(data_lines) == 1  # only the good camera made it through


def test_main_all_frames_malformed_still_writes_empty_valid_output(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    bad = {"camera_label": "00"}
    write_transforms(transforms, [bad])
    ply_path = tmp_path / "p.ply"
    write_ply(ply_path, n=1)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(transforms, ply_path, out_dir))
    bcs.main()  # must not raise
    data_lines = [ln for ln in (out_dir / "sparse" / "0" / "cameras.txt").read_text().splitlines()
                  if ln and not ln.startswith("#")]
    assert data_lines == []


def test_main_masks_dir_triggers_baking_and_bad_bake_is_skipped(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [make_valid_frame("00")])
    ply_path = tmp_path / "p.ply"
    write_ply(ply_path, n=1)
    images_dir = tmp_path / "images_flat"
    masks_dir = tmp_path / "masks"
    (images_dir).mkdir()
    (images_dir / "00.png").write_bytes(b"not a real png")  # corrupted
    masks_dir.mkdir()
    Image.new("L", (10, 10), color=255).save(masks_dir / "00.png")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(
        transforms, ply_path, out_dir, ["--masks_dir", str(masks_dir), "--images_dir", str(images_dir)],
    ))
    bcs.main()  # must not raise despite the corrupted bake
    assert "could not bake 00" in capsys.readouterr().out


def test_main_images_dir_defaults_to_out_dir_image_subdir(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [make_valid_frame("00")])
    ply_path = tmp_path / "p.ply"
    write_ply(ply_path, n=1)
    out_dir = tmp_path / "out"
    # images live under out_dir/images_flat (the default --image_subdir),
    # NOT passed via --images_dir -- proves the default reaches bake_masks
    default_images_dir = out_dir / "images_flat"
    masks_dir = tmp_path / "masks"
    write_image_and_mask(default_images_dir, masks_dir, "00")
    monkeypatch.setattr(sys, "argv", base_argv(transforms, ply_path, out_dir, ["--masks_dir", str(masks_dir)]))
    bcs.main()
    assert (out_dir / "images_rgba" / "00.png").is_file()


def test_main_no_masks_dir_skips_baking_entirely(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [make_valid_frame("00")])
    ply_path = tmp_path / "p.ply"
    write_ply(ply_path, n=1)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(transforms, ply_path, out_dir))
    bcs.main()
    assert not (out_dir / "images_rgba").exists()
    assert "images_flat/00.png" in (out_dir / "sparse" / "0" / "images.txt").read_text()


@pytest.mark.parametrize("missing_flag", ["--transforms", "--points_ply", "--out_dir"])
def test_main_errors_when_required_arg_missing(tmp_path, monkeypatch, missing_flag):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [])
    argv = base_argv(transforms, tmp_path / "p.ply", tmp_path / "out")
    idx = argv.index(missing_flag)
    argv = argv[:idx] + argv[idx + 2:]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        bcs.main()
    assert exc_info.value.code == 2
