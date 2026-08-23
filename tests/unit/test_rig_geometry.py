"""Unit tests for rig_geometry.py.

Everything the novel-view stages do rests on these conversions, and a drift here
is silent: synthetic views land in the wrong place and the only symptom is a
worse score. So the camera convention is pinned against build_colmap_sparse's
own implementation, and load_rig is pinned against both transforms layouts this
project has produced."""

import json

import numpy as np
import pytest

import build_colmap_sparse as bcs
import rig_geometry as rg


UP = np.array([0.0, 0.0, 1.0])
TARGET = np.array([0.3, -0.2, 1.1])


# --------------------------------------------------------------------------
# Camera convention
# --------------------------------------------------------------------------

def test_opengl_c2w_to_w2c_matches_build_colmap_sparse():
    # The whole pipeline hinges on one c2w/w2c convention; this must not drift
    # from the reference implementation the COLMAP writer uses.
    rng = np.random.default_rng(0)
    c2w = np.eye(4)
    c2w[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    c2w[:3, 3] = rng.normal(size=3)

    w2c = rg.opengl_c2w_to_w2c(c2w)
    R, t = bcs.opengl_c2w_to_colmap_w2c(c2w)
    assert np.allclose(w2c[:3, :3], R)
    assert np.allclose(w2c[:3, 3], t)


def test_opengl_c2w_to_w2c_does_not_mutate_its_input():
    c2w = np.eye(4)
    before = c2w.copy()
    rg.opengl_c2w_to_w2c(c2w)
    assert np.array_equal(c2w, before)


def test_lookat_places_the_camera_and_aims_it():
    eye = np.array([4.0, 0.0, 0.0])
    w2c = rg.lookat_w2c(eye, np.zeros(3), -UP)
    c2w = np.linalg.inv(w2c)
    assert c2w[:3, 3] == pytest.approx(eye)
    forward = c2w[:3, 2]
    assert forward == pytest.approx(-eye / np.linalg.norm(eye), abs=1e-9)


# --------------------------------------------------------------------------
# Spherical parameterisation
# --------------------------------------------------------------------------

def test_spherical_and_position_at_round_trip_exactly():
    # The endpoint homography is exact only if a real camera's centre survives
    # the trip through (azimuth, elevation, radius) unchanged.
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(UP, e1)
    rng = np.random.default_rng(7)
    for _ in range(20):
        center = TARGET + rng.normal(size=3) * 3.0
        azimuth, elevation, radius = rg.spherical(center, TARGET, UP, e1, e2)
        assert np.allclose(rg.position_at(TARGET, UP, e1, e2, azimuth, elevation, radius), center)


@pytest.mark.parametrize("start, end, expected", [
    (10.0, 40.0, 30.0), (40.0, 10.0, -30.0), (170.0, -170.0, 20.0), (-170.0, 170.0, -20.0),
])
def test_shortest_arc(start, end, expected):
    assert rg.shortest_arc(start, end) == pytest.approx(expected)


def test_orbit_basis_points_at_the_first_camera():
    cameras = [{"center": TARGET + np.array([5.0, 0.0, 2.0])}]
    e1, e2 = rg.orbit_basis(cameras, TARGET, UP)
    assert np.dot(e1, UP) == pytest.approx(0.0, abs=1e-12)   # horizontal
    assert np.linalg.norm(e1) == pytest.approx(1.0)
    assert np.dot(e1, np.array([1.0, 0.0, 0.0])) > 0.99
    assert np.dot(np.cross(e1, e2), UP) > 0                  # right-handed about up


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path, expected", [
    ("cam01/frame_00042", "cam01"),
    ("cam01/sub/frame_00042.jpg", "cam01"),
    ("00.png", ""),          # a flat layout implies no camera directory
    ("", ""),
])
def test_camera_label_from_path(path, expected):
    assert rg.camera_label_from_path(path) == expected


@pytest.mark.parametrize("path, expected", [
    ("cam01/frame_00042.jpg", 42),
    ("frame_00000", 0),
    ("00.png", 0),
    ("no_digits.png", None),
])
def test_trailing_number(path, expected):
    assert rg.trailing_number(path) == expected


# --------------------------------------------------------------------------
# load_rig, both layouts
# --------------------------------------------------------------------------

def write_transforms(tmp_path, frames, **top):
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps({"frames": frames, **top}))
    return path


def flat_frame(label, w=1024):
    return {"file_path": f"images/{label}.png", "camera_label": label,
            "transform_matrix": np.eye(4).tolist(),
            "fl_x": 1300.0, "fl_y": 1300.0, "cx": 512.0, "cy": 512.0, "w": w, "h": w}


def test_load_rig_reads_the_documented_flat_layout(tmp_path):
    path = write_transforms(tmp_path, [flat_frame("00"), flat_frame("01")])
    cameras, up, focal, width = rg.load_rig(path)
    assert [c["label"] for c in cameras] == ["00", "01"]
    assert focal == pytest.approx(1300.0)
    assert width == pytest.approx(1024.0)
    assert np.linalg.norm(up) == pytest.approx(1.0)


def test_load_rig_collapses_the_4d_layout_to_one_entry_per_camera(tmp_path):
    frames = []
    for camera in ("cam01", "cam02"):
        for number in (1, 2, 3):
            frames.append({"file_path": f"{camera}/frame_{number:05d}",
                           "transform_matrix": np.eye(4).tolist(),
                           "fl_x": 1300.0, "fl_y": 1300.0,
                           "cx": 500.0 + number, "cy": 400.0, "time": number / 30.0})
    path = write_transforms(tmp_path, frames, w=1024, h=1024)

    cameras, _, focal, width = rg.load_rig(path)
    assert [c["label"] for c in cameras] == ["cam01", "cam02"]
    assert focal == pytest.approx(1300.0)
    assert width == pytest.approx(1024.0)     # taken from the top level
    assert sorted(cameras[0]["per_frame"]) == [1, 2, 3]


def test_load_rig_keeps_per_frame_intrinsics(tmp_path):
    # Subject-tracking crops move the principal point every frame while the
    # focal and pose hold still. Collapsing those would misalign every warp.
    frames = [{"file_path": f"cam01/frame_{n:05d}", "transform_matrix": np.eye(4).tolist(),
               "fl_x": 1300.0, "fl_y": 1300.0, "cx": 500.0 + n, "cy": 400.0 - n, "time": n / 30.0}
              for n in (1, 58)]
    path = write_transforms(tmp_path, frames, w=1024, h=1024)

    camera = rg.load_rig(path)[0][0]
    assert camera["per_frame"][1]["intrinsics"][0, 2] == pytest.approx(501.0)
    assert camera["per_frame"][58]["intrinsics"][0, 2] == pytest.approx(558.0)
    assert camera["per_frame"][58]["intrinsics"][1, 2] == pytest.approx(342.0)


def test_load_rig_records_each_frames_time(tmp_path):
    frames = [{"file_path": f"cam01/frame_{n:05d}", "transform_matrix": np.eye(4).tolist(),
               "fl_x": 1300.0, "fl_y": 1300.0, "cx": 500.0, "cy": 400.0, "time": n / 30.0}
              for n in (1, 58)]
    path = write_transforms(tmp_path, frames, w=1024, h=1024)
    camera = rg.load_rig(path)[0][0]
    assert camera["per_frame"][58]["time"] == pytest.approx(58 / 30.0)


def test_load_rig_time_is_none_when_the_transforms_has_none(tmp_path):
    path = write_transforms(tmp_path, [flat_frame("00")])
    assert rg.load_rig(path)[0][0]["per_frame"][0]["time"] is None


def test_load_rig_rejects_an_empty_transforms(tmp_path):
    path = write_transforms(tmp_path, [])
    with pytest.raises(ValueError, match="no frames"):
        rg.load_rig(path)


def test_load_rig_rejects_a_transforms_with_no_width(tmp_path):
    frame = flat_frame("00")
    del frame["w"]
    path = write_transforms(tmp_path, [frame])
    with pytest.raises(ValueError, match="no image width"):
        rg.load_rig(path)


def test_load_rig_projection_matches_its_parts(tmp_path):
    path = write_transforms(tmp_path, [flat_frame("00")])
    camera = rg.load_rig(path)[0][0]
    assert np.allclose(camera["projection"], camera["intrinsics"] @ camera["w2c"][:3, :])


# --------------------------------------------------------------------------
# Subject anchors
# --------------------------------------------------------------------------

def test_subject_anchors_finds_the_centroid_and_a_head_above_it():
    rng = np.random.default_rng(3)
    body = rng.normal(scale=0.2, size=(500, 3))
    body[:, 2] += 1.0
    opacity = np.full(len(body), 0.9)
    centroid, head = rg.subject_anchors(body, opacity, UP)
    assert centroid[2] == pytest.approx(1.0, abs=0.1)
    assert head[2] > centroid[2]


def test_subject_anchors_ignores_transparent_splats():
    solid = np.tile(np.array([0.0, 0.0, 1.0]), (300, 1))
    ghost = np.tile(np.array([50.0, 50.0, 50.0]), (300, 1))
    means = np.vstack([solid, ghost])
    opacity = np.concatenate([np.full(300, 0.9), np.full(300, 0.01)])
    centroid, _ = rg.subject_anchors(means, opacity, UP)
    assert centroid == pytest.approx([0.0, 0.0, 1.0])


def test_subject_anchors_falls_back_when_nothing_is_opaque():
    means = np.tile(np.array([0.0, 0.0, 2.0]), (50, 1))
    centroid, head = rg.subject_anchors(means, np.full(50, 0.01), UP)
    assert centroid == pytest.approx([0.0, 0.0, 2.0])
    assert head[2] == pytest.approx(2.0 + rg.DEFAULT_HEAD_OFFSET)
