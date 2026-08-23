"""Unit tests for build_frame_dataset.py.

The point of this script is to make a bakeoff probe honest, so the tests that
matter are the ones that catch a holdout that is not actually held out: an
excluded camera slipping back into the training set, or a typo in --exclude
passing silently and training on everything."""

import json

import numpy as np
import pytest
from PIL import Image

import build_frame_dataset as bfd


def make_transforms(tmp_path, cameras=("cam01", "cam02", "cam03", "cam04"), frames=(1, 68),
                    sizes=None, write_images=True):
    """A 4D transforms.json plus per-camera image directories, mirroring the
    full4d layout: one entry per camera AND frame, extensionless file_path,
    per-frame principal point, per-camera image size."""
    sizes = sizes or {c: 64 + 16 * i for i, c in enumerate(cameras)}
    entries = []
    for camera in cameras:
        for frame in frames:
            entries.append({
                "file_path": f"{camera}/frame_{frame:05d}",
                "transform_matrix": np.eye(4).tolist(),
                "fl_x": 1300.0, "fl_y": 1300.0,
                "cx": 500.0 + frame, "cy": 400.0 + frame,   # moves per frame, as the crops do
                "time": frame / 30.0,
            })
            if write_images:
                directory = tmp_path / camera
                directory.mkdir(parents=True, exist_ok=True)
                size = sizes[camera]
                Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(
                    directory / f"frame_{frame:05d}.jpg")
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps({"w": 1024, "h": 1024, "frames": entries}))
    return path


# --------------------------------------------------------------------------
# Frame selection
# --------------------------------------------------------------------------

def test_frame_entries_picks_one_timestamp(tmp_path):
    transforms = make_transforms(tmp_path)
    entries = bfd.frame_entries(transforms, 68)
    assert sorted(entries) == ["cam01", "cam02", "cam03", "cam04"]
    assert all(e["file_path"].endswith("00068") for e in entries.values())


def test_frame_entries_is_empty_for_an_absent_frame(tmp_path):
    assert bfd.frame_entries(make_transforms(tmp_path), 999) == {}


def test_build_rejects_an_absent_frame(tmp_path):
    transforms = make_transforms(tmp_path)
    with pytest.raises(ValueError, match="no entry"):
        bfd.build(transforms, 999, tmp_path / "out", set())


# --------------------------------------------------------------------------
# The holdout
# --------------------------------------------------------------------------

def test_excluded_cameras_are_absent_from_the_dataset(tmp_path):
    transforms = make_transforms(tmp_path)
    summary = bfd.build(transforms, 68, tmp_path / "out", {"cam03", "cam04"})

    assert summary["cameras"] == ["cam01", "cam02"]
    written = json.loads((tmp_path / "out" / "transforms.json").read_text())
    labels = {f["camera_label"] for f in written["frames"]}
    assert labels == {"cam01", "cam02"}
    assert not (tmp_path / "out" / "images" / "cam04.jpg").exists()


def test_a_misspelled_exclusion_is_an_error(tmp_path):
    # Silently training on everything because --exclude had a typo would produce
    # a probe that looks held out and is not, which is the one failure this
    # script exists to prevent.
    transforms = make_transforms(tmp_path)
    with pytest.raises(ValueError, match="absent at frame 68"):
        bfd.build(transforms, 68, tmp_path / "out", {"cam99"})


def test_no_exclusion_keeps_every_camera(tmp_path):
    summary = bfd.build(make_transforms(tmp_path), 68, tmp_path / "out", set())
    assert summary["cameras"] == ["cam01", "cam02", "cam03", "cam04"]


# --------------------------------------------------------------------------
# Intrinsics and sizes
# --------------------------------------------------------------------------

def test_intrinsics_come_from_the_requested_frame(tmp_path):
    # The principal point moves frame to frame; taking frame 1's would misalign
    # every camera by that drift.
    transforms = make_transforms(tmp_path)
    bfd.build(transforms, 68, tmp_path / "out", set())
    written = json.loads((tmp_path / "out" / "transforms.json").read_text())
    assert {f["cx"] for f in written["frames"]} == {568.0}
    assert {f["cy"] for f in written["frames"]} == {468.0}


def test_size_is_read_from_each_image_not_the_transforms_header(tmp_path):
    # The top-level w/h says 1024 and disagrees with every camera on this capture.
    sizes = {"cam01": 64, "cam02": 96, "cam03": 80, "cam04": 112}
    transforms = make_transforms(tmp_path, sizes=sizes)
    bfd.build(transforms, 68, tmp_path / "out", set())
    written = json.loads((tmp_path / "out" / "transforms.json").read_text())
    by_label = {f["camera_label"]: (f["w"], f["h"]) for f in written["frames"]}
    assert by_label == {c: (s, s) for c, s in sizes.items()}


def test_written_paths_point_at_the_copied_images(tmp_path):
    bfd.build(make_transforms(tmp_path), 68, tmp_path / "out", set())
    written = json.loads((tmp_path / "out" / "transforms.json").read_text())
    for frame in written["frames"]:
        assert (tmp_path / "out" / frame["file_path"]).exists()


# --------------------------------------------------------------------------
# Image resolution
# --------------------------------------------------------------------------

def test_resolve_image_sniffs_the_extension(tmp_path):
    # file_path entries in these transforms carry no extension at all.
    (tmp_path / "cam01").mkdir()
    (tmp_path / "cam01" / "frame_00068.jpg").write_bytes(b"")
    transforms = tmp_path / "transforms.json"
    transforms.write_text("{}")
    assert bfd.resolve_image(transforms, "cam01/frame_00068") == tmp_path / "cam01" / "frame_00068.jpg"


def test_resolve_image_returns_none_when_absent(tmp_path):
    transforms = tmp_path / "transforms.json"
    transforms.write_text("{}")
    assert bfd.resolve_image(transforms, "cam01/frame_00068") is None


def test_missing_images_are_reported_not_silently_dropped(tmp_path):
    transforms = make_transforms(tmp_path)
    (tmp_path / "cam02" / "frame_00068.jpg").unlink()
    summary = bfd.build(transforms, 68, tmp_path / "out", set())
    assert summary["missing"] == ["cam02"]
    assert "cam02" not in summary["cameras"]


def test_build_fails_when_no_image_resolves(tmp_path):
    transforms = make_transforms(tmp_path, write_images=False)
    with pytest.raises(ValueError, match="no images resolved"):
        bfd.build(transforms, 68, tmp_path / "out", set())


def test_link_mode_symlinks_rather_than_copying(tmp_path):
    bfd.build(make_transforms(tmp_path), 68, tmp_path / "out", set(), link=True)
    assert (tmp_path / "out" / "images" / "cam01.jpg").is_symlink()


def test_rebuilding_over_an_existing_dataset_succeeds(tmp_path):
    transforms = make_transforms(tmp_path)
    bfd.build(transforms, 68, tmp_path / "out", set(), link=True)
    bfd.build(transforms, 68, tmp_path / "out", {"cam04"}, link=True)
    written = json.loads((tmp_path / "out" / "transforms.json").read_text())
    assert {f["camera_label"] for f in written["frames"]} == {"cam01", "cam02", "cam03"}
