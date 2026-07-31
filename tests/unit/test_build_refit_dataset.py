"""Unit tests for build_refit_dataset.py."""

import json

import numpy as np
import pytest

import build_colmap_sparse as bcs
import build_refit_dataset as brd


def random_c2w(seed):
    rng = np.random.default_rng(seed)
    c2w = np.eye(4)
    c2w[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    c2w[:3, 3] = rng.normal(size=3)
    return c2w


# --------------------------------------------------------------------------
# Pose conversions
# --------------------------------------------------------------------------

def test_colmap_quaternion_to_c2w_inverts_the_colmap_writer():
    # build_colmap_sparse writes c2w -> (R, t); this must read it straight back,
    # or every head-crop view lands at the wrong pose.
    from scipy.spatial.transform import Rotation

    c2w = random_c2w(3)
    R, t = bcs.opengl_c2w_to_colmap_w2c(c2w)
    qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()

    assert np.allclose(brd.colmap_quaternion_to_c2w(qw, qx, qy, qz, t), c2w)


def test_w2c_to_c2w_round_trips_the_render_pose_convention():
    # render_orbit_views stores w2c; the trainer wants c2w.
    import render_orbit_views as rov

    c2w = random_c2w(7)
    assert np.allclose(brd.w2c_to_c2w(rov.opengl_c2w_to_w2c(c2w)), c2w)


# --------------------------------------------------------------------------
# Frame numbering
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("frame_0007", 7), ("frame07", 7), ("frame_0000", 0), ("f123", 123), ("frame_0010", 10),
])
def test_frame_index_reads_the_trailing_number(tmp_path, name, expected):
    assert brd.frame_index(tmp_path / name) == expected


def test_frame_index_rejects_a_name_without_a_number(tmp_path):
    with pytest.raises(ValueError, match="frame number"):
        brd.frame_index(tmp_path / "orbit")


# --------------------------------------------------------------------------
# COLMAP reading
# --------------------------------------------------------------------------

def test_read_colmap_sparse_round_trips_what_build_colmap_sparse_writes(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    frames = [{"camera_label": "head_0001", "file_path": "images/head_0001.png",
               "transform_matrix": random_c2w(11).tolist(),
               "w": 400, "h": 400, "fl_x": 1700.0, "fl_y": 1700.0, "cx": 180.0, "cy": 210.0}]
    bcs.write_cameras_txt(sparse, frames)
    bcs.write_images_txt(sparse, frames, "images", None, "images_rgba")

    cameras, images = brd.read_colmap_sparse(sparse)
    assert len(images) == 1
    name, (camera_id, c2w) = next(iter(images.items()))
    assert name.endswith("head_0001.png")
    assert cameras[camera_id] == (400, 400, 1700.0, 1700.0, 180.0, 210.0)
    assert np.allclose(c2w, np.array(frames[0]["transform_matrix"]))


def test_read_colmap_sparse_skips_the_points2d_continuation_lines(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("# comment\n1 PINHOLE 100 100 50 50 50 50\n")
    (sparse / "images.txt").write_text(
        "# comment\n1 1 0 0 0 0 0 0 1 images/head_0001.png\n\n"
        "2 1 0 0 0 0 0 0 1 images/head_0002.png\n1.0 2.0 3\n")

    _, images = brd.read_colmap_sparse(sparse)
    assert set(images) == {"images/head_0001.png", "images/head_0002.png"}


# --------------------------------------------------------------------------
# Real-view linking
# --------------------------------------------------------------------------

def make_real_dataset(tmp_path, times):
    dataset = tmp_path / "real"
    frames = []
    for i, t in enumerate(times):
        camera = f"cam{i:02d}"
        (dataset / camera).mkdir(parents=True, exist_ok=True)
        (dataset / camera / "0000.jpg").write_bytes(b"jpeg")
        frames.append({"file_path": f"{camera}/0000", "time": t, "transform_matrix": np.eye(4).tolist(),
                       "fl_x": 1.0, "fl_y": 1.0, "cx": 0.5, "cy": 0.5})
    return dataset, frames


def test_link_real_views_keeps_only_the_window(tmp_path):
    dataset, frames = make_real_dataset(tmp_path, [0.0, 0.5, 1.0])
    out = tmp_path / "out"
    out.mkdir()

    linked = brd.link_real_views(frames, dataset, out, 0.4, 1.1)
    assert [f["time"] for f in linked] == [0.5, 1.0]


def test_link_real_views_renames_jpg_to_png(tmp_path):
    # The trainer's loader applies ONE extension to every frame and PIL sniffs
    # content, so reals must be reachable under a .png name.
    dataset, frames = make_real_dataset(tmp_path, [0.0])
    out = tmp_path / "out"
    out.mkdir()

    linked = brd.link_real_views(frames, dataset, out, 0.0, 1.0)
    assert linked[0]["file_path"] == "realcams/cam00/0000"
    linked_path = out / "realcams" / "cam00" / "0000.png"
    assert linked_path.is_symlink()
    assert linked_path.read_bytes() == b"jpeg"


def test_link_real_views_skips_frames_whose_image_is_missing(tmp_path):
    dataset, frames = make_real_dataset(tmp_path, [0.0, 0.5])
    (dataset / "cam01" / "0000.jpg").unlink()
    out = tmp_path / "out"
    out.mkdir()

    assert len(brd.link_real_views(frames, dataset, out, 0.0, 1.0)) == 1


def test_link_real_views_is_idempotent(tmp_path):
    dataset, frames = make_real_dataset(tmp_path, [0.0])
    out = tmp_path / "out"
    out.mkdir()
    first = brd.link_real_views(frames, dataset, out, 0.0, 1.0)
    second = brd.link_real_views(frames, dataset, out, 0.0, 1.0)
    assert first == second


# --------------------------------------------------------------------------
# Repaired-view collection
# --------------------------------------------------------------------------

def make_repaired_frame(root, name, n_body=2, n_head=2):
    frame_dir = root / name
    frame_dir.mkdir(parents=True)
    meta = {"fl_x": 900.0, "fl_y": 900.0, "cx": 768.0, "cy": 768.0,
            "frames": [{"idx": i, "w2c": np.eye(4).tolist()} for i in range(n_body)]}
    if n_head:
        meta["head"] = {"fl_x": 6000.0, "fl_y": 6000.0, "cx": 768.0, "cy": 768.0,
                        "frames": [{"idx": i, "w2c": np.eye(4).tolist()} for i in range(n_head)]}
    (frame_dir / "cameras.json").write_text(json.dumps(meta))
    for i in range(n_body):
        (frame_dir / f"body_{i:03d}.png").write_bytes(b"png")
    for i in range(n_head):
        (frame_dir / f"head_{i:03d}.png").write_bytes(b"png")
    return frame_dir


def test_head_views_are_excluded_by_default(tmp_path):
    # The default matters: head views were measured to poison shared-window refits.
    repaired = tmp_path / "repaired"
    make_repaired_frame(repaired, "frame_0000")
    out = tmp_path / "out"
    out.mkdir()

    body_only = brd.collect_repaired_views(repaired, out, [0.0], 0.0, 1.0, include_head=False)
    with_head = brd.collect_repaired_views(repaired, out, [0.0], 0.0, 1.0, include_head=True)
    assert len(body_only) == 2
    assert len(with_head) == 4
    assert all("body_" in view["file_path"] for view in body_only)


def test_repaired_views_get_their_frames_timestamp(tmp_path):
    repaired = tmp_path / "repaired"
    make_repaired_frame(repaired, "frame_0000", n_head=0)
    make_repaired_frame(repaired, "frame_0002", n_head=0)
    out = tmp_path / "out"
    out.mkdir()

    views = brd.collect_repaired_views(repaired, out, [0.0, 0.1, 0.2], 0.0, 1.0, include_head=False)
    assert sorted({view["time"] for view in views}) == [0.0, 0.2]


def test_repaired_view_paths_are_relative_to_the_output_dir(tmp_path):
    # A hardcoded relative path once pointed training at a stale directory while
    # the existence check looked at the right one.
    repaired = tmp_path / "elsewhere" / "repaired"
    make_repaired_frame(repaired, "frame_0000", n_head=0)
    out = tmp_path / "nested" / "out"
    out.mkdir(parents=True)

    views = brd.collect_repaired_views(repaired, out, [0.0], 0.0, 1.0, include_head=False)
    resolved = (out / views[0]["file_path"]).parent.resolve()
    assert resolved == (repaired / "frame_0000").resolve()


def test_repaired_views_outside_the_window_are_dropped(tmp_path):
    repaired = tmp_path / "repaired"
    make_repaired_frame(repaired, "frame_0000", n_head=0)
    make_repaired_frame(repaired, "frame_0001", n_head=0)
    out = tmp_path / "out"
    out.mkdir()

    views = brd.collect_repaired_views(repaired, out, [0.0, 0.5], 0.4, 1.0, include_head=False)
    assert {view["time"] for view in views} == {0.5}


def test_missing_image_files_are_skipped_not_listed(tmp_path):
    repaired = tmp_path / "repaired"
    frame_dir = make_repaired_frame(repaired, "frame_0000", n_body=3, n_head=0)
    (frame_dir / "body_001.png").unlink()
    out = tmp_path / "out"
    out.mkdir()

    views = brd.collect_repaired_views(repaired, out, [0.0], 0.0, 1.0, include_head=False)
    assert len(views) == 2
