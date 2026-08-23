"""Unit tests for window_keypoints_3d.py (frame selection and mask derivation;
the Sapiens and triangulation stages are subprocesses).

The mask tests matter because the keypoint stage requires masks and will happily
run on wrong ones: an all-white mask produces keypoints detected against the
background, which triangulate into 3D points that look plausible and are wrong."""

import numpy as np
import pytest
from PIL import Image

import window_keypoints_3d as wk


def write_transforms(tmp_path, cameras=("cam01", "cam02"), frames=(1, 58, 74)):
    import json
    entries = [{"file_path": f"{c}/frame_{n:05d}", "transform_matrix": np.eye(4).tolist(),
                "fl_x": 1300.0, "fl_y": 1300.0, "cx": 500.0, "cy": 400.0, "time": n / 30.0}
               for c in cameras for n in frames]
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps({"w": 1024, "h": 1024, "frames": entries}))
    return path


# --------------------------------------------------------------------------
# Frame selection
# --------------------------------------------------------------------------

def test_frame_numbers_defaults_to_every_frame(tmp_path):
    assert wk.frame_numbers(write_transforms(tmp_path), None) == [1, 58, 74]


def test_frame_numbers_honours_a_range(tmp_path):
    assert wk.frame_numbers(write_transforms(tmp_path), "58-74") == [58, 74]


def test_frame_numbers_rejects_a_malformed_range(tmp_path):
    with pytest.raises(ValueError, match="LO-HI"):
        wk.frame_numbers(write_transforms(tmp_path), "58..74")


def test_frame_numbers_reports_what_the_rig_actually_has(tmp_path):
    # An empty selection from a typo would otherwise look like "nothing to do".
    with pytest.raises(ValueError, match="the rig has 1-74"):
        wk.frame_numbers(write_transforms(tmp_path), "200-300")


# --------------------------------------------------------------------------
# Mask derivation
# --------------------------------------------------------------------------

def test_masks_come_from_alpha_when_present(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    array = np.zeros((16, 16, 4), dtype=np.uint8)
    array[..., :3] = 200          # opaque-looking colour everywhere
    array[4:12, 4:12, 3] = 255    # but only the middle is actually covered
    Image.fromarray(array, mode="RGBA").save(images / "00.png")

    assert wk.write_masks(images, tmp_path / "masks") == 1
    mask = np.asarray(Image.open(tmp_path / "masks" / "00.png"))
    assert mask[8, 8] == 255
    assert mask[0, 0] == 0


def test_masks_come_from_non_black_when_there_is_no_alpha(tmp_path):
    # Background-removed captures arrive as RGB against black.
    images = tmp_path / "images"
    images.mkdir()
    array = np.zeros((16, 16, 3), dtype=np.uint8)
    array[4:12, 4:12] = 180
    Image.fromarray(array).save(images / "00.jpg")

    wk.write_masks(images, tmp_path / "masks")
    mask = np.asarray(Image.open(tmp_path / "masks" / "00.png"))
    assert mask[8, 8] == 255
    assert mask[0, 0] == 0


def test_near_black_pixels_count_as_background(tmp_path):
    # JPEG ringing leaves the background at small non-zero values; a plain
    # "> 0" test would mask the whole frame as subject.
    images = tmp_path / "images"
    images.mkdir()
    array = np.full((16, 16, 3), 2, dtype=np.uint8)
    array[4:12, 4:12] = 180
    Image.fromarray(array).save(images / "00.png")

    wk.write_masks(images, tmp_path / "masks")
    mask = np.asarray(Image.open(tmp_path / "masks" / "00.png"))
    assert mask[0, 0] == 0
    assert mask[8, 8] == 255


def test_masks_are_written_for_every_image(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("00.png", "01.png", "02.png"):
        Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(images / name)
    (images / "notes.txt").write_text("ignored")

    assert wk.write_masks(images, tmp_path / "masks") == 3
    assert sorted(p.name for p in (tmp_path / "masks").iterdir()) == ["00.png", "01.png", "02.png"]


def test_mask_names_match_the_image_stems(tmp_path):
    # The keypoint stage pairs an image with its mask by stem, so a .jpg image
    # must yield a .png mask of the same stem.
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(images / "07.jpg")
    wk.write_masks(images, tmp_path / "masks")
    assert (tmp_path / "masks" / "07.png").exists()
