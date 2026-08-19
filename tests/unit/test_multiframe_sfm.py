# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
Tests real math/data-structure logic against a REAL pycolmap (installed as a
dev dependency, see requirements-dev.txt) rather than mocking it away --
mocking pycolmap would make tests like test_build_database_* circular (they'd
just check "did the mock receive the arguments I told it to expect").

hloc's own deep-learning feature-extraction/matching (extract_features,
match_features) is NOT installed here (heavy, GPU-oriented) -- a lightweight
fake `hloc` package is injected into sys.modules below purely so
`import multiframe_sfm` succeeds. `hloc.reconstruction`'s two thin COLMAP-
wrapper functions this module actually calls before any deep-learning step
(create_empty_db, get_image_ids) are given real pycolmap-backed
implementations, so build_database() is still tested against real behavior.
Everything downstream of the deep-learning boundary (average_camera_poses,
extract_refined_intrinsics, export_points_ply, write_transforms) is exercised
with duck-typed stand-ins for pycolmap.Reconstruction objects, matching the
FakeProcess technique already used in test_run_unified_pipeline.py for a similarly
expensive-to-construct real object.
"""
import json
import pickle
import sys
import types

import numpy as np
import pytest

pytest.importorskip("pycolmap")
import pycolmap


def _install_fake_hloc():
    if getattr(sys.modules.get("hloc"), "_is_fake", False):
        return

    hloc = types.ModuleType("hloc")
    extract_features = types.ModuleType("hloc.extract_features")
    match_features = types.ModuleType("hloc.match_features")
    pairs_from_exhaustive = types.ModuleType("hloc.pairs_from_exhaustive")
    reconstruction = types.ModuleType("hloc.reconstruction")

    extract_features.main = lambda *a, **kw: None
    match_features.main = lambda *a, **kw: None
    match_features.confs = {"superpoint+lightglue": {}, "aliked+lightglue": {}}
    pairs_from_exhaustive.main = lambda *a, **kw: None

    def create_empty_db(path):
        # Matches real hloc's own create_empty_db: delete any existing file,
        # then open (which creates the COLMAP sqlite schema).
        from pathlib import Path
        p = Path(path)
        if p.exists():
            p.unlink()
        pycolmap.Database.open(str(path)).close()

    def get_image_ids(db_path):
        db = pycolmap.Database.open(str(db_path))
        try:
            return {im.name: im.image_id for im in db.read_all_images()}
        finally:
            db.close()

    reconstruction.create_empty_db = create_empty_db
    reconstruction.get_image_ids = get_image_ids
    reconstruction.import_features = lambda *a, **kw: None
    reconstruction.import_matches = lambda *a, **kw: None
    reconstruction.estimation_and_geometric_verification = lambda *a, **kw: None
    reconstruction.run_reconstruction = lambda *a, **kw: None

    hloc.extract_features = extract_features
    hloc.match_features = match_features
    hloc.pairs_from_exhaustive = pairs_from_exhaustive
    hloc.reconstruction = reconstruction
    hloc._is_fake = True

    sys.modules["hloc"] = hloc
    sys.modules["hloc.extract_features"] = extract_features
    sys.modules["hloc.match_features"] = match_features
    sys.modules["hloc.pairs_from_exhaustive"] = pairs_from_exhaustive
    sys.modules["hloc.reconstruction"] = reconstruction


_install_fake_hloc()
import multiframe_sfm as mfs


# --------------------------------------------------------------------------
# probe_video / extract_frame -- subprocess.run is monkeypatched (real
# ffprobe/ffmpeg, not runnable here)
# --------------------------------------------------------------------------

def test_probe_video_parses_ffprobe_json(tmp_path, monkeypatch):
    ffprobe_json = json.dumps({"streams": [{
        "nb_read_frames": "100", "r_frame_rate": "30/1", "width": "1920", "height": "1080",
    }]})

    def fake_run(cmd, capture_output=True, text=True, check=True):
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, stdout=ffprobe_json, stderr="")
    monkeypatch.setattr(mfs.subprocess, "run", fake_run)

    info = mfs.probe_video(tmp_path / "0001.mp4")
    assert info == {"nb_frames": 100, "fps": 30.0, "width": 1920, "height": 1080}


def test_probe_video_handles_fractional_frame_rate(tmp_path, monkeypatch):
    ffprobe_json = json.dumps({"streams": [{
        "nb_read_frames": "50", "r_frame_rate": "30000/1001", "width": "100", "height": "100",
    }]})

    def fake_run(cmd, capture_output=True, text=True, check=True):
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, stdout=ffprobe_json, stderr="")
    monkeypatch.setattr(mfs.subprocess, "run", fake_run)

    info = mfs.probe_video(tmp_path / "0001.mp4")
    assert info["fps"] == pytest.approx(29.97, abs=0.01)


def test_extract_frame_raises_when_ffmpeg_writes_nothing(tmp_path, monkeypatch):
    def fake_run(cmd, check=True, capture_output=True):
        import subprocess
        return subprocess.CompletedProcess(cmd, 0)  # succeeds but writes no file
    monkeypatch.setattr(mfs.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Failed to extract frame"):
        mfs.extract_frame(tmp_path / "0001.mp4", 5, tmp_path / "out" / "0001_05.jpg")


def test_extract_frame_creates_parent_dir(tmp_path, monkeypatch):
    out_path = tmp_path / "nested" / "dir" / "frame.jpg"

    def fake_run(cmd, check=True, capture_output=True):
        import subprocess
        out_path.write_bytes(b"fake jpg")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(mfs.subprocess, "run", fake_run)

    mfs.extract_frame(tmp_path / "0001.mp4", 5, out_path)
    assert out_path.is_file()


# --------------------------------------------------------------------------
# load_init_intrinsics -- pure logic, real json/dict handling
# --------------------------------------------------------------------------

def write_init_transforms(path, frames, camera_model="PINHOLE", w=None, h=None):
    data = {"camera_model": camera_model, "frames": frames}
    if w is not None:
        data["w"] = w
    if h is not None:
        data["h"] = h
    path.write_text(json.dumps(data))


def test_load_init_intrinsics_pinhole(tmp_path):
    p = tmp_path / "init.json"
    write_init_transforms(p, [{"camera_label": "Camera_0001", "fl_x": 10.0, "fl_y": 11.0, "cx": 50.0, "cy": 60.0, "w": 100, "h": 200}])
    result = mfs.load_init_intrinsics(p, ["Camera_0001"])
    assert result == {"Camera_0001": ("PINHOLE", 100, 200, [10.0, 11.0, 50.0, 60.0])}


def test_load_init_intrinsics_opencv_includes_distortion(tmp_path):
    p = tmp_path / "init.json"
    write_init_transforms(p, [{
        "camera_label": "Camera_0001", "camera_model": "OPENCV",
        "fl_x": 10.0, "fl_y": 11.0, "cx": 50.0, "cy": 60.0, "w": 100, "h": 200,
        "k1": 0.1, "k2": 0.2, "p1": 0.01, "p2": 0.02,
    }], camera_model="OPENCV")
    model, w, h, params = mfs.load_init_intrinsics(p, ["Camera_0001"])["Camera_0001"]
    assert model == "OPENCV"
    assert params == [10.0, 11.0, 50.0, 60.0, 0.1, 0.2, 0.01, 0.02]


def test_load_init_intrinsics_opencv_fisheye_includes_k_params(tmp_path):
    p = tmp_path / "init.json"
    write_init_transforms(p, [{
        "camera_label": "Camera_0001", "camera_model": "OPENCV_FISHEYE",
        "fl_x": 10.0, "fl_y": 11.0, "cx": 50.0, "cy": 60.0, "w": 100, "h": 200,
        "k1": 0.1, "k2": 0.2, "k3": 0.3, "k4": 0.4,
    }], camera_model="OPENCV_FISHEYE")
    model, _, _, params = mfs.load_init_intrinsics(p, ["Camera_0001"])["Camera_0001"]
    assert model == "OPENCV_FISHEYE"
    assert params == [10.0, 11.0, 50.0, 60.0, 0.1, 0.2, 0.3, 0.4]


def test_load_init_intrinsics_missing_distortion_defaults_to_zero(tmp_path):
    p = tmp_path / "init.json"
    write_init_transforms(p, [{
        "camera_label": "Camera_0001", "camera_model": "OPENCV",
        "fl_x": 10.0, "fl_y": 11.0, "cx": 50.0, "cy": 60.0, "w": 100, "h": 200,
    }], camera_model="OPENCV")
    _, _, _, params = mfs.load_init_intrinsics(p, ["Camera_0001"])["Camera_0001"]
    assert params == [10.0, 11.0, 50.0, 60.0, 0.0, 0.0, 0.0, 0.0]


def test_load_init_intrinsics_matches_by_file_path_stem(tmp_path):
    # A frame with no camera_label but a file_path whose FILENAME (not a
    # parent directory -- Path.stem only strips the extension off the final
    # path component) contains the label.
    p = tmp_path / "init.json"
    write_init_transforms(p, [{"file_path": "images/Camera_0001.jpg", "fl_x": 1.0, "fl_y": 1.0, "cx": 1.0, "cy": 1.0, "w": 10, "h": 10}])
    result = mfs.load_init_intrinsics(p, ["Camera_0001"])
    assert "Camera_0001" in result


def test_load_init_intrinsics_no_match_raises_keyerror(tmp_path):
    p = tmp_path / "init.json"
    write_init_transforms(p, [{"camera_label": "Camera_0002", "fl_x": 1.0, "fl_y": 1.0, "cx": 1.0, "cy": 1.0, "w": 10, "h": 10}])
    with pytest.raises(KeyError, match="Camera_0001"):
        mfs.load_init_intrinsics(p, ["Camera_0001"])


def test_load_init_intrinsics_unsupported_model_raises(tmp_path):
    p = tmp_path / "init.json"
    write_init_transforms(p, [{"camera_label": "Camera_0001", "camera_model": "BOGUS_MODEL",
                               "fl_x": 1.0, "fl_y": 1.0, "cx": 1.0, "cy": 1.0, "w": 10, "h": 10}])
    with pytest.raises(ValueError, match="Unsupported camera model"):
        mfs.load_init_intrinsics(p, ["Camera_0001"])


def test_load_init_intrinsics_falls_back_to_top_level_w_h(tmp_path):
    p = tmp_path / "init.json"
    write_init_transforms(p, [{"camera_label": "Camera_0001", "fl_x": 1.0, "fl_y": 1.0, "cx": 1.0, "cy": 1.0}],
                          w=640, h=480)
    _, w, h, _ = mfs.load_init_intrinsics(p, ["Camera_0001"])["Camera_0001"]
    assert (w, h) == (640, 480)


# --------------------------------------------------------------------------
# robust_pose_average -- real numpy math, no mocking
# --------------------------------------------------------------------------

def make_pose(tx=0.0, ty=0.0, tz=0.0):
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T


def test_robust_pose_average_identical_poses_returns_same_pose():
    T = make_pose(1.0, 2.0, 3.0)
    T_avg, stats = mfs.robust_pose_average([T.copy() for _ in range(3)])
    assert np.allclose(T_avg, T)
    assert stats == {
        "num_views": 3, "num_inliers": 3,
        "center_spread": 0.0, "center_spread_max": 0.0,
        "rot_spread_deg": 0.0, "rot_spread_deg_max": 0.0,
    }


def test_robust_pose_average_rejects_wild_position_outlier():
    good = [make_pose(1.0, 2.0, 3.0) for _ in range(4)]
    outlier = make_pose(1000.0, 1000.0, 1000.0)
    T_avg, stats = mfs.robust_pose_average(good + [outlier])
    assert np.allclose(T_avg[:3, 3], [1.0, 2.0, 3.0], atol=1e-6)
    assert stats["num_inliers"] == 4
    assert stats["num_views"] == 5


def test_robust_pose_average_rejects_moderate_rotational_outlier():
    good = [np.eye(4) for _ in range(4)]
    theta = np.radians(45)
    outlier = np.eye(4)
    outlier[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    T_avg, stats = mfs.robust_pose_average(good + [outlier], max_rot_spread_deg=10.0)
    assert stats["num_inliers"] == 4
    assert np.allclose(T_avg[:3, :3], np.eye(3), atol=1e-6)


@pytest.mark.parametrize("outlier_deg", [15, 30, 45, 60, 90, 120, 150, 179])
def test_robust_pose_average_rejects_rotational_outlier_at_any_magnitude(outlier_deg):
    # Regression test: a single-shot "average everyone, then see who's close
    # to the average" approach used to let an EXTREME outlier (60+ degrees)
    # drag the average far enough that even the good poses failed the
    # deviation check, triggering an empty-selection fallback that reverted
    # to keeping everyone -- counterintuitively, a MORE extreme outlier was
    # MORE likely to survive unfiltered than a moderate one. Fixed with a
    # vote-based approach (whichever pose has the most other poses agreeing
    # with it defines the inlier set), which never computes a
    # outlier-contaminated average in the first place. This sweeps the full
    # plausible outlier range to prove the fix holds everywhere, not just at
    # the couple of points that happened to be checked by hand.
    good = [np.eye(4) for _ in range(4)]
    theta = np.radians(outlier_deg)
    outlier = np.eye(4)
    outlier[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    T_avg, stats = mfs.robust_pose_average(good + [outlier], max_rot_spread_deg=10.0)
    assert stats["num_inliers"] == 4
    assert np.allclose(T_avg[:3, :3], np.eye(3), atol=1e-6)


def test_robust_pose_average_no_false_rejection_on_realistic_small_noise():
    # The vote-based fix must not become trigger-happy on ordinary
    # estimation noise when there's no real outlier at all.
    rng = np.random.default_rng(0)
    poses = []
    for _ in range(10):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.radians(rng.uniform(-1, 1))  # ~1 degree of noise
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        T = np.eye(4)
        T[:3, :3] = R
        poses.append(T)
    _, stats = mfs.robust_pose_average(poses, max_rot_spread_deg=10.0)
    assert stats["num_inliers"] == 10


def test_robust_pose_average_single_pose_does_not_crash():
    T = make_pose(1.0, 2.0, 3.0)
    T_avg, stats = mfs.robust_pose_average([T])
    assert np.allclose(T_avg, T)
    assert stats["num_views"] == 1 and stats["num_inliers"] == 1


@pytest.mark.parametrize("seed", range(10))
def test_robust_pose_average_all_outliers_edge_case_does_not_crash(seed):
    # keep.sum() == 0 fallback path: every pose is simultaneously a position
    # AND rotation outlier relative to the others -- must fall back to using
    # all poses rather than crash on an empty selection.
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(3):
        T = np.eye(4)
        T[:3, 3] = rng.uniform(-1000, 1000, 3)
        poses.append(T)
    T_avg, stats = mfs.robust_pose_average(poses)
    assert stats["num_inliers"] >= 1


# --------------------------------------------------------------------------
# write_transforms -- real numpy coordinate conversion + json writing
# --------------------------------------------------------------------------

def test_write_transforms_identity_pose_opencv_to_opengl(tmp_path):
    out_path = tmp_path / "transforms.json"
    mfs.write_transforms(out_path, {"Camera_0001": np.eye(4)},
                         {"Camera_0001": ("PINHOLE", 100, 200, [10.0, 11.0, 50.0, 60.0])})
    data = json.loads(out_path.read_text())
    fr = data["frames"][0]
    assert fr["camera_label"] == "Camera_0001"
    assert fr["file_path"] == "images/Camera_0001.png"
    # cam_from_world=I -> c2w=I -> negate Y/Z columns -> diag(1,-1,-1)
    m = np.array(fr["transform_matrix"])
    assert np.allclose(m, np.diag([1.0, -1.0, -1.0, 1.0]))


def test_write_transforms_opencv_distortion_keys(tmp_path):
    out_path = tmp_path / "transforms.json"
    mfs.write_transforms(out_path, {"Camera_0001": np.eye(4)},
                         {"Camera_0001": ("OPENCV", 100, 200, [10.0, 11.0, 50.0, 60.0, 0.1, 0.2, 0.01, 0.02])})
    fr = json.loads(out_path.read_text())["frames"][0]
    assert (fr["k1"], fr["k2"], fr["p1"], fr["p2"]) == (0.1, 0.2, 0.01, 0.02)


def test_write_transforms_fisheye_distortion_keys(tmp_path):
    out_path = tmp_path / "transforms.json"
    mfs.write_transforms(out_path, {"Camera_0001": np.eye(4)},
                         {"Camera_0001": ("OPENCV_FISHEYE", 100, 200, [10.0, 11.0, 50.0, 60.0, 0.1, 0.2, 0.3, 0.4])})
    fr = json.loads(out_path.read_text())["frames"][0]
    assert (fr["k1"], fr["k2"], fr["k3"], fr["k4"]) == (0.1, 0.2, 0.3, 0.4)


def test_write_transforms_prefers_undistorted_intrinsics_when_given(tmp_path):
    out_path = tmp_path / "transforms.json"
    mfs.write_transforms(
        out_path, {"Camera_0001": np.eye(4)},
        intrinsics_by_camera={"Camera_0001": ("OPENCV", 100, 200, [10.0, 11.0, 50.0, 60.0, 0, 0, 0, 0])},
        undistorted_intrinsics={"Camera_0001": ("PINHOLE", 999, 888, [1.0, 2.0, 3.0, 4.0])},
    )
    fr = json.loads(out_path.read_text())["frames"][0]
    assert fr["camera_model"] == "PINHOLE"
    assert (fr["w"], fr["h"]) == (999, 888)
    assert (fr["fl_x"], fr["fl_y"]) == (1.0, 2.0)


def test_write_transforms_includes_ply_file_path_when_given(tmp_path):
    out_path = tmp_path / "transforms.json"
    mfs.write_transforms(out_path, {"Camera_0001": np.eye(4)},
                         {"Camera_0001": ("PINHOLE", 100, 200, [1.0, 1.0, 1.0, 1.0])},
                         ply_file_path="background_points.ply")
    data = json.loads(out_path.read_text())
    assert data["ply_file_path"] == "background_points.ply"


def test_write_transforms_omits_ply_file_path_when_not_given(tmp_path):
    out_path = tmp_path / "transforms.json"
    mfs.write_transforms(out_path, {"Camera_0001": np.eye(4)},
                         {"Camera_0001": ("PINHOLE", 100, 200, [1.0, 1.0, 1.0, 1.0])})
    assert "ply_file_path" not in json.loads(out_path.read_text())


def test_write_transforms_sorts_frames_by_label(tmp_path):
    out_path = tmp_path / "transforms.json"
    intrinsics = {
        "Camera_0002": ("PINHOLE", 1, 1, [1.0, 1.0, 1.0, 1.0]),
        "Camera_0001": ("PINHOLE", 1, 1, [1.0, 1.0, 1.0, 1.0]),
    }
    mfs.write_transforms(out_path, {"Camera_0002": np.eye(4), "Camera_0001": np.eye(4)}, intrinsics)
    labels = [fr["camera_label"] for fr in json.loads(out_path.read_text())["frames"]]
    assert labels == ["Camera_0001", "Camera_0002"]


# --------------------------------------------------------------------------
# load_undistorted_intrinsics -- real pickle reads, no mocking
# --------------------------------------------------------------------------

def write_calib_pkl(path, w=100, h=200, fx=10.0, fy=11.0, cx=50.0, cy=60.0):
    with open(path, "wb") as f:
        pickle.dump({"camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "image_size": (w, h)}, f)


def test_load_undistorted_intrinsics_matches_undistorted_prefix(tmp_path):
    write_calib_pkl(tmp_path / "undistorted_Camera_0001.pkl")
    result = mfs.load_undistorted_intrinsics(tmp_path, ["Camera_0001"])
    assert result["Camera_0001"][0] == "OPENCV"
    assert result["Camera_0001"][1:3] == (100, 200)


def test_load_undistorted_intrinsics_matches_bare_label(tmp_path):
    write_calib_pkl(tmp_path / "Camera_0001_calib.pkl")
    result = mfs.load_undistorted_intrinsics(tmp_path, ["Camera_0001"])
    assert "Camera_0001" in result


def test_load_undistorted_intrinsics_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Camera_0001"):
        mfs.load_undistorted_intrinsics(tmp_path, ["Camera_0001"])


def test_load_undistorted_intrinsics_zero_distortion_params(tmp_path):
    write_calib_pkl(tmp_path / "Camera_0001.pkl")
    _, _, _, params = mfs.load_undistorted_intrinsics(tmp_path, ["Camera_0001"])["Camera_0001"]
    assert params[4:] == [0.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------------
# load_sync_shifts -- KNOWN LIMITATION, not fixed (see conversation record):
# this reads a JSON shape ("cameras"/"frame_shift") that no longer matches
# compute_sync_offsets.py's real output ("offsets"/"frame_offset"). The
# --videos_dir/--sync_json code path is not exercised by the real pipeline
# (run_hloc.py only ever calls this script with --frames_root), so this is
# left unfixed by explicit decision. These tests document CURRENT behavior
# only -- they are not an assertion that this is correct or usable as-is.
# --------------------------------------------------------------------------

def test_load_sync_shifts_none_returns_empty_dict():
    assert mfs.load_sync_shifts(None) == {}


def test_load_sync_shifts_current_expected_shape(tmp_path):
    # Documents current behavior against the shape this function actually
    # reads today -- NOT the shape compute_sync_offsets.py actually produces.
    p = tmp_path / "sync.json"
    p.write_text(json.dumps({"cameras": {"0001": {"frame_shift": 5}, "0002": {"frame_shift": -3}}}))
    assert mfs.load_sync_shifts(p) == {"0001": 5, "0002": -3}


def test_load_sync_shifts_crashes_on_real_compute_sync_offsets_format(tmp_path):
    # KNOWN BUG, not fixed by decision: compute_sync_offsets.py (the only
    # sync-offsets producer anywhere in this pipeline) writes {"offsets":
    # {...: {"frame_offset": ...}}}, not {"cameras": {...: {"frame_shift":
    # ...}}}. This test exists to make the gap explicit and visible, not to
    # assert it's acceptable -- if this ever starts passing because someone
    # "fixes" the KeyError without addressing the underlying format
    # mismatch, that's worth a second look.
    p = tmp_path / "sync_offsets.json"
    p.write_text(json.dumps({
        "reference_camera": "0001.mp4",
        "offsets": {"0001.mp4": {"frame_offset": 0, "fps": 30.0},
                    "0002.mp4": {"frame_offset": 5, "fps": 30.0}},
    }))
    with pytest.raises(KeyError, match="cameras"):
        mfs.load_sync_shifts(p)


# --------------------------------------------------------------------------
# collect_frames_from_root -- real filesystem logic, no mocking
# --------------------------------------------------------------------------

def test_collect_frames_from_root_groups_by_camera_dir(tmp_path):
    (tmp_path / "Camera_0001").mkdir()
    (tmp_path / "Camera_0001" / "000010.jpg").write_text("x")
    (tmp_path / "Camera_0001" / "000020.jpg").write_text("x")
    (tmp_path / "Camera_0002").mkdir()
    (tmp_path / "Camera_0002" / "000010.jpg").write_text("x")

    result = mfs.collect_frames_from_root(tmp_path)
    assert result["Camera_0001"] == ["Camera_0001/000010.jpg", "Camera_0001/000020.jpg"]
    assert result["Camera_0002"] == ["Camera_0002/000010.jpg"]


def test_collect_frames_from_root_ignores_non_image_files(tmp_path):
    (tmp_path / "Camera_0001").mkdir()
    (tmp_path / "Camera_0001" / "000010.jpg").write_text("x")
    (tmp_path / "Camera_0001" / "notes.txt").write_text("stray")
    result = mfs.collect_frames_from_root(tmp_path)
    assert result["Camera_0001"] == ["Camera_0001/000010.jpg"]


def test_collect_frames_from_root_ignores_non_directory_entries(tmp_path):
    (tmp_path / "Camera_0001").mkdir()
    (tmp_path / "Camera_0001" / "000010.jpg").write_text("x")
    (tmp_path / "stray_file.jpg").write_text("not inside a camera dir")
    result = mfs.collect_frames_from_root(tmp_path)
    assert set(result.keys()) == {"Camera_0001"}


def test_collect_frames_from_root_errors_when_empty(tmp_path):
    with pytest.raises(SystemExit, match="No frames found"):
        mfs.collect_frames_from_root(tmp_path)


# --------------------------------------------------------------------------
# gather_frames -- dispatch logic
# --------------------------------------------------------------------------

class Args:
    def __init__(self, videos_dir=None, frames_root=None, sync_json=None, num_timestamps=1, timestamps=None):
        self.videos_dir = videos_dir
        self.frames_root = frames_root
        self.sync_json = sync_json
        self.num_timestamps = num_timestamps
        self.timestamps = timestamps


def test_gather_frames_uses_frames_root_when_given(tmp_path):
    (tmp_path / "Camera_0001").mkdir()
    (tmp_path / "Camera_0001" / "000010.jpg").write_text("x")
    args = Args(frames_root=str(tmp_path))
    result, frames_dir = mfs.gather_frames(args, tmp_path / "unused_frames_dir")
    assert "Camera_0001" in result
    assert frames_dir == tmp_path


def test_gather_frames_errors_when_neither_given():
    args = Args()
    with pytest.raises(SystemExit, match="Provide --videos_dir or --frames_root"):
        mfs.gather_frames(args, None)


# --------------------------------------------------------------------------
# build_database -- REAL pycolmap, no mocking
# --------------------------------------------------------------------------

def test_build_database_creates_one_shared_camera_per_physical_camera(tmp_path):
    db_path = tmp_path / "database.db"
    image_names_by_camera = {
        "Camera_0002": ["Camera_0002/000000.jpg", "Camera_0002/000010.jpg"],
        "Camera_0001": ["Camera_0001/000000.jpg"],
    }
    intrinsics_by_camera = {
        "Camera_0001": ("PINHOLE", 100, 200, [10.0, 11.0, 50.0, 60.0]),
        "Camera_0002": ("PINHOLE", 100, 200, [10.0, 11.0, 50.0, 60.0]),
    }
    camera_ids = mfs.build_database(db_path, image_names_by_camera, intrinsics_by_camera)

    # sorted() assignment -- Camera_0001 gets id 1, Camera_0002 gets id 2
    assert camera_ids == {"Camera_0001": 1, "Camera_0002": 2}

    db = pycolmap.Database.open(str(db_path))
    try:
        assert db.num_cameras() == 2
        assert db.num_images() == 3
        images_by_name = {im.name: im.camera_id for im in db.read_all_images()}
        assert images_by_name["Camera_0001/000000.jpg"] == 1
        assert images_by_name["Camera_0002/000000.jpg"] == 2
        assert images_by_name["Camera_0002/000010.jpg"] == 2
    finally:
        db.close()


def test_build_database_recreates_existing_db_file(tmp_path):
    db_path = tmp_path / "database.db"
    intrinsics = {"Camera_0001": ("PINHOLE", 100, 200, [10.0, 11.0, 50.0, 60.0])}
    mfs.build_database(db_path, {"Camera_0001": ["Camera_0001/a.jpg"]}, intrinsics)
    mfs.build_database(db_path, {"Camera_0001": ["Camera_0001/a.jpg"]}, intrinsics)

    db = pycolmap.Database.open(str(db_path))
    try:
        assert db.num_images() == 1  # not doubled by the second run
    finally:
        db.close()


# --------------------------------------------------------------------------
# export_points_ply -- duck-typed fake Reconstruction (real pycolmap
# Reconstruction objects with 3D points are expensive to construct; the
# function's own filtering/PLY-writing logic is what's under test, matching
# the FakeProcess technique already used in test_run_unified_pipeline.py)
# --------------------------------------------------------------------------

class FakeTrack:
    def __init__(self, length):
        self._length = length
    def length(self):
        return self._length


class FakePoint3D:
    def __init__(self, xyz, color, error, track_length):
        self.xyz = np.array(xyz)
        self.color = color
        self.error = error
        self.track = FakeTrack(track_length)


class FakeRec:
    def __init__(self, points=None, images=None, cameras=None):
        self.points3D = {i: p for i, p in enumerate(points or [])}
        self.images = {i: im for i, im in enumerate(images or [])}
        self.cameras = cameras or {}


def test_export_points_ply_filters_by_error_and_track_length(tmp_path):
    rec = FakeRec([
        FakePoint3D([1.0, 2.0, 3.0], [255, 0, 0], error=1.0, track_length=5),  # keep
        FakePoint3D([4.0, 5.0, 6.0], [0, 255, 0], error=5.0, track_length=5),  # error too high
        FakePoint3D([7.0, 8.0, 9.0], [0, 0, 255], error=1.0, track_length=1),  # track too short
    ])
    out_path = tmp_path / "points.ply"
    n = mfs.export_points_ply(rec, out_path, max_error=2.0, min_track_length=3)
    assert n == 1
    assert out_path.is_file()


def test_export_points_ply_zero_points_remaining_writes_valid_empty_ply(tmp_path):
    rec = FakeRec([FakePoint3D([1, 2, 3], [1, 2, 3], error=99.0, track_length=1)])
    out_path = tmp_path / "points.ply"
    n = mfs.export_points_ply(rec, out_path, max_error=2.0, min_track_length=3)
    assert n == 0
    header = out_path.read_bytes()
    assert b"element vertex 0" in header


def test_export_points_ply_writes_readable_binary_ply(tmp_path):
    rec = FakeRec([FakePoint3D([1.5, 2.5, 3.5], [10, 20, 30], error=0.5, track_length=10)])
    out_path = tmp_path / "points.ply"
    mfs.export_points_ply(rec, out_path, max_error=2.0, min_track_length=3)

    raw = out_path.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    body = raw[header_end:]
    xyz = np.frombuffer(body[:12], dtype="<f4")
    rgb = np.frombuffer(body[12:15], dtype=np.uint8)
    assert np.allclose(xyz, [1.5, 2.5, 3.5])
    assert list(rgb) == [10, 20, 30]


# --------------------------------------------------------------------------
# average_camera_poses / extract_refined_intrinsics -- duck-typed fake
# Reconstruction images/cameras, real robust_pose_average math underneath
# --------------------------------------------------------------------------

class FakePose:
    def __init__(self, T):
        self._T = T
    def matrix(self):
        return self._T[:3, :]


class FakeImage:
    def __init__(self, name, T, camera_id=None):
        self.name = name
        self.camera_id = camera_id if camera_id is not None else int(name.split("/")[0].split("_")[1])
        self._T = T
    def cam_from_world(self):
        return FakePose(self._T)


def test_average_camera_poses_averages_across_timestamps():
    T1 = np.eye(4)
    T2 = make_pose(5.0, 0.0, 0.0)
    rec = FakeRec(images=[
        FakeImage("Camera_0001/000000.jpg", T1),
        FakeImage("Camera_0001/000010.jpg", T1),
        FakeImage("Camera_0002/000000.jpg", T2),
    ])
    avg_poses, stats = mfs.average_camera_poses(rec, ["Camera_0001", "Camera_0002"], max_rot_spread_deg=10.0)
    assert set(avg_poses.keys()) == {"Camera_0001", "Camera_0002"}
    assert stats["Camera_0001"]["num_views"] == 2
    assert stats["Camera_0002"]["num_views"] == 1


def test_average_camera_poses_warns_on_camera_with_no_registered_images(capsys):
    rec = FakeRec(images=[FakeImage("Camera_0001/000000.jpg", np.eye(4))])
    avg_poses, _ = mfs.average_camera_poses(rec, ["Camera_0001", "Camera_0099"], max_rot_spread_deg=10.0)
    assert "Camera_0099" not in avg_poses
    assert "Camera_0099" in capsys.readouterr().out


class FakeModel:
    def __init__(self, name):
        self.name = name


class FakeCamera:
    def __init__(self, model_name, width, height, params):
        self.model = FakeModel(model_name)
        self.width = width
        self.height = height
        self.params = params


def test_extract_refined_intrinsics_maps_camera_to_label():
    rec = FakeRec(
        images=[FakeImage("Camera_0001/000000.jpg", np.eye(4), camera_id=1)],
        cameras={1: FakeCamera("PINHOLE", 100, 200, [10.0, 11.0, 50.0, 60.0])},
    )
    refined = mfs.extract_refined_intrinsics(rec)
    assert refined == {"Camera_0001": ("PINHOLE", 100, 200, [10.0, 11.0, 50.0, 60.0])}


def test_extract_refined_intrinsics_skips_camera_with_no_images():
    rec = FakeRec(images=[], cameras={1: FakeCamera("PINHOLE", 100, 200, [1.0, 1.0, 1.0, 1.0])})
    assert mfs.extract_refined_intrinsics(rec) == {}


# --------------------------------------------------------------------------
# parse_args -- CLI wiring
# --------------------------------------------------------------------------

def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "prog", "--init_transforms", "t.json", "--outputs_dir", "out",
    ])
    args = mfs.parse_args()
    assert args.num_timestamps == 12
    assert args.feature_type == "superpoint"
    assert args.resize_max == 2048
    assert args.max_keypoints == 8192
    assert args.refine_intrinsics is False


def test_parse_args_rejects_invalid_feature_type(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "prog", "--init_transforms", "t.json", "--outputs_dir", "out", "--feature_type", "bogus",
    ])
    with pytest.raises(SystemExit) as exc_info:
        mfs.parse_args()
    assert exc_info.value.code == 2


@pytest.mark.parametrize("missing_flag", ["--init_transforms", "--outputs_dir"])
def test_parse_args_errors_when_required_arg_missing(monkeypatch, missing_flag):
    argv = ["prog", "--init_transforms", "t.json", "--outputs_dir", "out"]
    idx = argv.index(missing_flag)
    argv = argv[:idx] + argv[idx + 2:]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        mfs.parse_args()
    assert exc_info.value.code == 2


# --------------------------------------------------------------------------
# main() -- full orchestration wiring. extract_and_match_features and
# run_sfm_reconstruction are monkeypatched (the true hloc/deep-learning/
# COLMAP-mapper boundary, equivalent to mocking ffmpeg elsewhere); everything
# downstream (average_camera_poses, extract_refined_intrinsics,
# export_results) runs for real against a duck-typed fake Reconstruction.
# --------------------------------------------------------------------------

def make_frames_root_rig(tmp_path, cams=("Camera_0001", "Camera_0002")):
    frames_root = tmp_path / "frames_root"
    for cam in cams:
        (frames_root / cam).mkdir(parents=True)
        (frames_root / cam / "000000.jpg").write_text("x")
    init_transforms = tmp_path / "init.json"
    write_init_transforms(init_transforms, [
        {"camera_label": cam, "fl_x": 10.0, "fl_y": 11.0, "cx": 50.0, "cy": 60.0, "w": 100, "h": 200}
        for cam in cams
    ])
    return frames_root, init_transforms


def test_main_runs_full_pipeline_and_writes_outputs(tmp_path, monkeypatch, capsys):
    frames_root, init_transforms = make_frames_root_rig(tmp_path)
    outputs_dir = tmp_path / "outputs"

    fake_rec = FakeRec(
        images=[
            FakeImage("Camera_0001/000000.jpg", np.eye(4), camera_id=1),
            FakeImage("Camera_0002/000000.jpg", make_pose(5.0, 0.0, 0.0), camera_id=2),
        ],
        cameras={
            1: FakeCamera("PINHOLE", 100, 200, [10.0, 11.0, 50.0, 60.0]),
            2: FakeCamera("PINHOLE", 100, 200, [10.0, 11.0, 50.0, 60.0]),
        },
    )
    fake_rec.num_points3D = lambda: 0
    fake_rec.compute_mean_track_length = lambda: 0.0
    fake_rec.compute_mean_reprojection_error = lambda: 0.0
    fake_rec.num_reg_images = lambda: 2

    monkeypatch.setattr(mfs, "extract_and_match_features", lambda *a, **kw: ("feat", "pairs", "match"))
    monkeypatch.setattr(mfs, "run_sfm_reconstruction", lambda *a, **kw: (fake_rec, outputs_dir / "sfm"))

    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_root", str(frames_root), "--init_transforms", str(init_transforms),
        "--outputs_dir", str(outputs_dir),
    ])
    mfs.main()

    assert (outputs_dir / "transforms_multiframe.json").is_file()
    assert (outputs_dir / "report.json").is_file()
    assert (outputs_dir / "background_points.ply").is_file()
    report = json.loads((outputs_dir / "report.json").read_text())
    assert report["num_cameras"] == 2
    assert "Camera_0001" in report["per_camera_pose_stats"]


def test_main_errors_when_neither_videos_dir_nor_frames_root_given(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "prog", "--init_transforms", str(tmp_path / "t.json"), "--outputs_dir", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit, match="Provide --videos_dir or --frames_root"):
        mfs.main()


def test_main_aliked_feature_type_uses_aliked_config(tmp_path, monkeypatch):
    frames_root, init_transforms = make_frames_root_rig(tmp_path, cams=("Camera_0001",))
    outputs_dir = tmp_path / "outputs"
    captured = {}

    def fake_extract_and_match(args, frames_dir, out_dir, all_images):
        captured["feature_type"] = args.feature_type
        return "feat", "pairs", "match"

    fake_rec = FakeRec(images=[FakeImage("Camera_0001/000000.jpg", np.eye(4), camera_id=1)],
                       cameras={1: FakeCamera("PINHOLE", 100, 200, [10.0, 11.0, 50.0, 60.0])})
    fake_rec.num_points3D = lambda: 0
    fake_rec.compute_mean_track_length = lambda: 0.0
    fake_rec.compute_mean_reprojection_error = lambda: 0.0
    fake_rec.num_reg_images = lambda: 1

    monkeypatch.setattr(mfs, "extract_and_match_features", fake_extract_and_match)
    monkeypatch.setattr(mfs, "run_sfm_reconstruction", lambda *a, **kw: (fake_rec, outputs_dir / "sfm"))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_root", str(frames_root), "--init_transforms", str(init_transforms),
        "--outputs_dir", str(outputs_dir), "--feature_type", "aliked",
    ])
    mfs.main()
    assert captured["feature_type"] == "aliked"


def test_extract_and_match_features_builds_superpoint_config(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(mfs.extract_features, "main", lambda conf, *a, **kw: calls.setdefault("feat_conf", conf) or "feature_path")
    monkeypatch.setattr(mfs.pairs_from_exhaustive, "main", lambda sfm_pairs, image_list: sfm_pairs.write_text(""))
    monkeypatch.setattr(mfs.match_features, "main", lambda conf, *a, **kw: calls.setdefault("match_conf", conf) or "match_path")

    args = Args()
    args.feature_type = "superpoint"
    args.max_keypoints = 4096
    args.resize_max = 1024

    mfs.extract_and_match_features(args, tmp_path, tmp_path, ["Camera_0001/000000.jpg"])

    assert calls["feat_conf"]["model"]["name"] == "superpoint"
    assert calls["feat_conf"]["model"]["max_keypoints"] == 4096
    assert calls["feat_conf"]["preprocessing"]["resize_max"] == 1024


def test_extract_and_match_features_builds_aliked_config(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(mfs.extract_features, "main", lambda conf, *a, **kw: calls.setdefault("feat_conf", conf) or "feature_path")
    monkeypatch.setattr(mfs.pairs_from_exhaustive, "main", lambda sfm_pairs, image_list: sfm_pairs.write_text(""))
    monkeypatch.setattr(mfs.match_features, "main", lambda conf, *a, **kw: "match_path")

    args = Args()
    args.feature_type = "aliked"
    args.max_keypoints = 2048
    args.resize_max = 512

    mfs.extract_and_match_features(args, tmp_path, tmp_path, ["Camera_0001/000000.jpg"])

    assert calls["feat_conf"]["model"]["name"] == "aliked"
    assert calls["feat_conf"]["model"]["max_num_keypoints"] == 2048
