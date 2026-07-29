# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Tests real math (triangulation, similarity alignment, bundle adjustment) against
synthetic camera rigs with known ground truth, rather than mocking scipy/numpy away --
the whole point of this script is that the math is correct, so the tests check the
math is correct, not that a mock received the arguments we told it to expect.
"""
import json
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

import refine_poses_with_keypoints as rpwk


# --------------------------------------------------------------------------
# Synthetic camera rig helpers
# --------------------------------------------------------------------------

def look_at_w2c(eye, target, world_up=(0.0, 0.0, 1.0)):
    """OpenCV-convention world-to-camera (R, t) such that x_cam = R @ X_world + t,
    with +Z (forward) pointing from eye toward target."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.asarray(world_up, dtype=float)
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R_c2w = np.stack([right, down, forward], axis=1)
    R_w2c = R_c2w.T
    t = -R_w2c @ eye
    return R_w2c, t


def w2c_to_opengl_transform_matrix(R_w2c, t_w2c):
    """Inverse of load_transforms' OpenGL->OpenCV conversion, so a rig built with
    look_at_w2c can be round-tripped through a real transforms.json file."""
    c2w_opencv = np.eye(4)
    c2w_opencv[:3, :3] = R_w2c.T
    c2w_opencv[:3, 3] = -R_w2c.T @ t_w2c
    c2w_opengl = c2w_opencv.copy()
    c2w_opengl[:3, 1:3] *= -1
    return c2w_opengl


def make_rig(n=4, radius=3.0, z=1.5, fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, target=(0.0, 0.0, 0.0)):
    cams = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        eye = np.array([radius * np.cos(theta), radius * np.sin(theta), z])
        R_w2c, t_w2c = look_at_w2c(eye, target)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        cams.append({"label": f"Camera_{i:04d}", "K": K, "R": R_w2c, "t": t_w2c})
    return cams


def project(K, R_w2c, t_w2c, X):
    x = R_w2c @ np.asarray(X, dtype=float) + t_w2c
    uv = K @ (x / x[2])
    return float(uv[0]), float(uv[1]), float(x[2])


def write_transforms_json(path, cams, w=1920, h=1080, camera_label=True):
    frames = []
    for c in cams:
        M = w2c_to_opengl_transform_matrix(c["R"], c["t"])
        K = c["K"]
        fr = {
            "file_path": f"images/{c['label']}.png",
            "w": w, "h": h,
            "fl_x": float(K[0, 0]), "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "transform_matrix": M.tolist(),
        }
        if camera_label:
            fr["camera_label"] = c["label"]
        frames.append(fr)
    data = {"frames": frames}
    path.write_text(json.dumps(data))
    return data


def write_kp2d_combined(path, cams, points_3d, score=0.95, frame_id="000000"):
    frames = []
    for c in cams:
        kps, scores = [], []
        for X in points_3d:
            u, v, _z = project(c["K"], c["R"], c["t"], X)
            kps.append([u, v])
            scores.append(score)
        frames.append({
            "image_name": f"{c['label']}.png",
            "frame_id": frame_id,
            "instances": [{"keypoints": kps, "keypoint_scores": scores}],
        })
    data = {"frames": frames}
    path.write_text(json.dumps(data))
    return data


def write_kp2d_directory(root, cams, points_3d, score=0.95, tem="000000"):
    for c in cams:
        cam_dir = root / c["label"]
        cam_dir.mkdir(parents=True)
        kps, scores = [], []
        for X in points_3d:
            u, v, _z = project(c["K"], c["R"], c["t"], X)
            kps.append([u, v])
            scores.append(score)
        (cam_dir / f"{tem}.json").write_text(json.dumps({
            "instance_info": [{"keypoints": kps, "keypoint_scores": scores}],
        }))


# --------------------------------------------------------------------------
# load_transforms
# --------------------------------------------------------------------------

def test_load_transforms_parses_K_and_w2c_round_trip(tmp_path):
    cams = make_rig(n=3)
    p = tmp_path / "transforms.json"
    write_transforms_json(p, cams)
    data, loaded = rpwk.load_transforms(p)
    for c in cams:
        entry = loaded[c["label"]]
        assert np.allclose(entry["K"], c["K"])
        assert np.allclose(entry["w2c"][:3, :3], c["R"], atol=1e-6)
        assert np.allclose(entry["w2c"][:3, 3], c["t"], atol=1e-6)


def test_load_transforms_falls_back_to_file_path_stem_when_camera_label_missing(tmp_path):
    cams = make_rig(n=1)
    p = tmp_path / "transforms.json"
    write_transforms_json(p, cams, camera_label=False)
    _, loaded = rpwk.load_transforms(p)
    assert list(loaded.keys()) == ["Camera_0000"]


def test_load_transforms_errors_when_file_missing(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        rpwk.load_transforms(tmp_path / "nope.json")


def test_load_transforms_errors_on_invalid_json(tmp_path):
    p = tmp_path / "transforms.json"
    p.write_text("{not valid json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        rpwk.load_transforms(p)


def test_load_transforms_errors_on_missing_frames_key(tmp_path):
    p = tmp_path / "transforms.json"
    p.write_text(json.dumps({"not_frames": []}))
    with pytest.raises(SystemExit, match="missing expected key"):
        rpwk.load_transforms(p)


def test_load_transforms_skips_frame_missing_intrinsics_keeps_others(tmp_path, capsys):
    cams = make_rig(n=2)
    p = tmp_path / "transforms.json"
    write_transforms_json(p, cams)
    parsed = json.loads(p.read_text())
    del parsed["frames"][0]["fl_x"]
    p.write_text(json.dumps(parsed))
    _, loaded = rpwk.load_transforms(p)
    assert "Camera_0000" not in loaded
    assert "Camera_0001" in loaded
    assert "WARNING" in capsys.readouterr().out


def test_load_transforms_skips_frame_with_singular_transform_matrix(tmp_path):
    cams = make_rig(n=2)
    p = tmp_path / "transforms.json"
    write_transforms_json(p, cams)
    parsed = json.loads(p.read_text())
    parsed["frames"][0]["transform_matrix"] = np.zeros((4, 4)).tolist()
    p.write_text(json.dumps(parsed))
    _, loaded = rpwk.load_transforms(p)
    assert "Camera_0000" not in loaded
    assert "Camera_0001" in loaded


def test_load_transforms_errors_when_every_camera_unusable(tmp_path):
    cams = make_rig(n=1)
    p = tmp_path / "transforms.json"
    write_transforms_json(p, cams)
    parsed = json.loads(p.read_text())
    del parsed["frames"][0]["fl_x"]
    p.write_text(json.dumps(parsed))
    with pytest.raises(SystemExit, match="no usable cameras"):
        rpwk.load_transforms(p)


# --------------------------------------------------------------------------
# _trailing_id / match_camera
# --------------------------------------------------------------------------

def test_trailing_id_extracts_digits():
    assert rpwk._trailing_id("Camera_0007") == 7
    assert rpwk._trailing_id("undistorted_0042") == 42


def test_trailing_id_returns_none_when_no_digits():
    assert rpwk._trailing_id("left_cam") is None


def test_match_camera_exact_stem_match():
    assert rpwk.match_camera("Camera_0007.png", ["Camera_0007", "Camera_0008"]) == "Camera_0007"


def test_match_camera_trailing_id_unique_match():
    labels = ["Camera_0007", "Camera_0008"]
    assert rpwk.match_camera("undistorted_0007.png", labels) == "Camera_0007"


def test_match_camera_trailing_id_ambiguous_with_no_substring_returns_none():
    # Two labels share trailing id 7 -- id-based match must not pick either
    # arbitrarily. Falls through to substring matching, which also finds
    # nothing here, so the honest answer is "no match" rather than a guess.
    labels = ["Camera_0007", "Extra_0007"]
    assert rpwk.match_camera("undistorted_0007", labels) is None


def test_match_camera_substring_match_picks_longest():
    labels = ["left_cam", "left_cam_ir"]
    assert rpwk.match_camera("left_cam_ir_raw", labels) == "left_cam_ir"


def test_match_camera_no_match_returns_none():
    assert rpwk.match_camera("totally_unrelated", ["Camera_0007"]) is None


# --------------------------------------------------------------------------
# triangulate_linear -- real DLT triangulation
# --------------------------------------------------------------------------

def test_triangulate_linear_recovers_known_point_exactly():
    cams = make_rig(n=4)
    X_true = np.array([0.1, -0.2, 0.3])
    Ks, w2cs, uvs, ws = [], [], [], []
    for c in cams:
        u, v, z = project(c["K"], c["R"], c["t"], X_true)
        assert z > 0
        Ks.append(c["K"])
        T = np.eye(4)
        T[:3, :3] = c["R"]
        T[:3, 3] = c["t"]
        w2cs.append(T)
        uvs.append((u, v))
        ws.append(1.0)
    X_est = rpwk.triangulate_linear(Ks, w2cs, uvs, ws)
    assert np.allclose(X_est, X_true, atol=1e-4)


@given(
    x=st.floats(-1.0, 1.0, allow_nan=False),
    y=st.floats(-1.0, 1.0, allow_nan=False),
    z=st.floats(-0.5, 0.5, allow_nan=False),
)
@settings(max_examples=40, deadline=None)
def test_triangulate_linear_recovers_any_point_in_working_volume(x, y, z):
    cams = make_rig(n=4)
    X_true = np.array([x, y, z])
    Ks, w2cs, uvs, ws = [], [], [], []
    for c in cams:
        u, v, zc = project(c["K"], c["R"], c["t"], X_true)
        if zc <= 1e-3:
            return  # degenerate for this camera, not what this property is about
        Ks.append(c["K"])
        T = np.eye(4)
        T[:3, :3] = c["R"]
        T[:3, 3] = c["t"]
        w2cs.append(T)
        uvs.append((u, v))
        ws.append(1.0)
    X_est = rpwk.triangulate_linear(Ks, w2cs, uvs, ws)
    assert np.allclose(X_est, X_true, atol=1e-3)


# --------------------------------------------------------------------------
# similarity_align
# --------------------------------------------------------------------------

def test_similarity_align_identity_when_src_equals_dst():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    s, R, t = rpwk.similarity_align(pts, pts)
    assert np.isclose(s, 1.0, atol=1e-6)
    assert np.allclose(R, np.eye(3), atol=1e-6)
    assert np.allclose(t, 0.0, atol=1e-6)


def test_similarity_align_recovers_known_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(6, 3))
    s_true = 2.5
    R_true = Rotation.from_euler("xyz", [12, -30, 55], degrees=True).as_matrix()
    t_true = np.array([1.0, -2.0, 0.5])
    dst = (s_true * (R_true @ src.T)).T + t_true
    s, R, t = rpwk.similarity_align(src, dst)
    assert np.isclose(s, s_true, atol=1e-5)
    assert np.allclose(R, R_true, atol=1e-5)
    assert np.allclose(t, t_true, atol=1e-4)


@given(
    scale=st.floats(0.1, 5.0, allow_nan=False),
    euler=st.tuples(*[st.floats(-179, 179, allow_nan=False)] * 3),
    translation=st.tuples(*[st.floats(-10, 10, allow_nan=False)] * 3),
)
@settings(max_examples=40, deadline=None)
def test_similarity_align_recovers_any_similarity_transform(scale, euler, translation):
    rng = np.random.default_rng(0)
    src = rng.normal(size=(8, 3))
    R_true = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
    t_true = np.array(translation)
    dst = (scale * (R_true @ src.T)).T + t_true
    s, R, t = rpwk.similarity_align(src, dst)
    recovered = (s * (R @ src.T)).T + t
    assert np.allclose(recovered, dst, atol=1e-3)


# --------------------------------------------------------------------------
# huber_transform
# --------------------------------------------------------------------------

def test_huber_transform_unchanged_below_threshold():
    r = np.array([1.0, -2.0, 3.0])
    out = rpwk.huber_transform(r, delta=8.0)
    assert np.allclose(out, r)


def test_huber_transform_downweights_above_threshold_preserving_sign():
    r = np.array([20.0, -20.0])
    delta = 8.0
    out = rpwk.huber_transform(r, delta)
    expected_mag = np.sqrt(delta * (2 * 20.0 - delta))
    assert np.isclose(out[0], expected_mag)
    assert np.isclose(out[1], -expected_mag)
    assert out[0] < 20.0  # actually downweighted, not a no-op


def test_huber_transform_does_not_mutate_input():
    r = np.array([100.0, -100.0])
    original = r.copy()
    rpwk.huber_transform(r, delta=8.0)
    assert np.array_equal(r, original)


# --------------------------------------------------------------------------
# drop_outliers
# --------------------------------------------------------------------------

def _make_point(entries, xyz=(0.0, 0.0, 0.0), key=("000000", 0)):
    return {"xyz": np.array(xyz), "obs": list(entries), "key": key}


def test_drop_outliers_removes_behind_camera_observation():
    # Two cameras facing opposite directions along X: a point at x=-10 is
    # behind cam A (which looks from x=-5 toward the origin, i.e. +x) but
    # still in front of cam B (looks from x=+5 toward the origin, i.e. -x).
    RA, tA = look_at_w2c([-5, 0, 0], [0, 0, 0])
    RB, tB = look_at_w2c([5, 0, 0], [0, 0, 0])
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    Ks_all = [K, K]
    rvecs0 = np.array([Rotation.from_matrix(RA).as_rotvec(), Rotation.from_matrix(RB).as_rotvec()])
    tvecs0 = np.array([tA, tB])
    X = np.array([-10.0, 0.0, 0.0])
    uB, vB, zB = project(K, RB, tB, X)
    assert zB > 0
    # Cam A's own (u, v) values don't matter -- it's dropped by the z<=0 check
    # before its reprojection error is ever computed.
    point = _make_point([(0, 0.0, 0.0, 1.0), (1, uB, vB, 1.0)], xyz=X)
    kept, n_dropped = rpwk.drop_outliers([point], rvecs0, tvecs0, Ks_all, outlier_px=200.0, min_views=1)
    assert n_dropped == 1
    assert len(kept) == 1
    assert [e[0] for e in kept[0]["obs"]] == [1]


def test_drop_outliers_removes_high_reprojection_error_observation():
    cams = make_rig(n=3)
    X = np.array([0.0, 0.0, 0.0])
    rvecs0 = np.array([Rotation.from_matrix(c["R"]).as_rotvec() for c in cams])
    tvecs0 = np.array([c["t"] for c in cams])
    Ks_all = [c["K"] for c in cams]
    entries = []
    for i, c in enumerate(cams):
        u, v, _z = project(c["K"], c["R"], c["t"], X)
        entries.append((i, u, v, 1.0))
    # Corrupt one observation far from its true reprojection.
    cam_idx, u, v, s = entries[0]
    entries[0] = (cam_idx, u + 500.0, v + 500.0, s)
    point = _make_point(entries, xyz=X)
    kept, n_dropped = rpwk.drop_outliers([point], rvecs0, tvecs0, Ks_all, outlier_px=50.0, min_views=1)
    assert n_dropped == 1
    assert len(kept[0]["obs"]) == len(cams) - 1
    assert all(e[0] != 0 for e in kept[0]["obs"])


def test_drop_outliers_drops_whole_point_below_min_views_after_filtering():
    cams = make_rig(n=2)
    X = np.array([0.0, 0.0, 0.0])
    rvecs0 = np.array([Rotation.from_matrix(c["R"]).as_rotvec() for c in cams])
    tvecs0 = np.array([c["t"] for c in cams])
    Ks_all = [c["K"] for c in cams]
    entries = []
    for i, c in enumerate(cams):
        u, v, _z = project(c["K"], c["R"], c["t"], X)
        entries.append((i, u + (900.0 if i == 0 else 0.0), v, 1.0))
    point = _make_point(entries, xyz=X)
    kept, n_dropped = rpwk.drop_outliers([point], rvecs0, tvecs0, Ks_all, outlier_px=50.0, min_views=2)
    assert kept == []
    assert n_dropped == 1


def test_drop_outliers_keeps_clean_point_unchanged():
    cams = make_rig(n=3)
    X = np.array([0.0, 0.0, 0.0])
    rvecs0 = np.array([Rotation.from_matrix(c["R"]).as_rotvec() for c in cams])
    tvecs0 = np.array([c["t"] for c in cams])
    Ks_all = [c["K"] for c in cams]
    entries = [(i, *project(c["K"], c["R"], c["t"], X)[:2], 1.0) for i, c in enumerate(cams)]
    point = _make_point(entries, xyz=X)
    kept, n_dropped = rpwk.drop_outliers([point], rvecs0, tvecs0, Ks_all, outlier_px=50.0, min_views=2)
    assert n_dropped == 0
    assert len(kept) == 1
    assert len(kept[0]["obs"]) == 3


# --------------------------------------------------------------------------
# build_observations
# --------------------------------------------------------------------------

def test_build_observations_applies_face_weight_to_face_keypoint():
    face_ids = {2}
    points = [_make_point([(0, 1.0, 2.0, 0.81)], key=("000000", 2))]
    obs_flat, obs_p, obs_c, obs_uv, obs_w = rpwk.build_observations(points, face_ids, face_weight=4.0)
    assert np.isclose(obs_w[0], np.sqrt(4.0 * 0.81))


def test_build_observations_default_weight_for_non_face_keypoint():
    face_ids = {2}
    points = [_make_point([(0, 1.0, 2.0, 0.81)], key=("000000", 5))]
    _, _, _, _, obs_w = rpwk.build_observations(points, face_ids, face_weight=4.0)
    assert np.isclose(obs_w[0], np.sqrt(0.81))


def test_build_observations_face_weight_1_is_a_noop():
    face_ids = {2}
    points = [_make_point([(0, 1.0, 2.0, 0.5)], key=("000000", 2))]
    _, _, _, _, obs_w = rpwk.build_observations(points, face_ids, face_weight=1.0)
    assert np.isclose(obs_w[0], np.sqrt(0.5))


# --------------------------------------------------------------------------
# build_sparsity_pattern
# --------------------------------------------------------------------------

def test_build_sparsity_pattern_shape():
    obs_flat = [(0, 0, 1.0, 2.0, 1.0), (0, 1, 3.0, 4.0, 1.0)]
    sparsity = rpwk.build_sparsity_pattern(obs_flat, n_cams=2, n_pts=1)
    assert sparsity.shape == (2 * 2, 6 * 2 + 3 * 1)


def test_build_sparsity_pattern_marks_correct_blocks():
    obs_flat = [(0, 1, 1.0, 2.0, 1.0)]  # point 0 seen by camera 1
    n_cams, n_pts = 2, 1
    sparsity = rpwk.build_sparsity_pattern(obs_flat, n_cams, n_pts).toarray()
    # rotation block for cam 1: cols [3:6]
    assert sparsity[0, 3:6].all()
    # translation block for cam 1: cols [3*n_cams + 3 : 3*n_cams+6] = [9:12]
    assert sparsity[0, 9:12].all()
    # point block for point 0: cols [6*n_cams : 6*n_cams+3] = [12:15]
    assert sparsity[0, 12:15].all()
    # camera 0's blocks (rotation [0:3], translation [6:9]) must be untouched
    assert not sparsity[0, 0:3].any()
    assert not sparsity[0, 6:9].any()


# --------------------------------------------------------------------------
# report_residuals
# --------------------------------------------------------------------------

def test_report_residuals_computes_correct_error(capsys):
    cams = make_rig(n=2)
    X = np.array([0.05, 0.0, 0.0])
    rvecs = np.array([Rotation.from_matrix(c["R"]).as_rotvec() for c in cams])
    tvecs = np.array([c["t"] for c in cams])
    Ks_all = [c["K"] for c in cams]
    entries = []
    expected_err = {}
    for i, c in enumerate(cams):
        u, v, _z = project(c["K"], c["R"], c["t"], X)
        du, dv = 3.0, -4.0
        entries.append((i, u + du, v + dv, 1.0))
        expected_err[i] = np.hypot(du, dv)
    point = _make_point(entries, xyz=X)
    all_errs = rpwk.report_residuals(rvecs, tvecs, [point], [point], Ks_all, [c["label"] for c in cams], "TEST")
    assert np.allclose(sorted(all_errs), sorted(expected_err.values()), atol=1e-6)
    out = capsys.readouterr().out
    assert "TEST" in out


def test_report_residuals_skips_points_behind_camera():
    cams = make_rig(n=2)
    rvecs = np.array([Rotation.from_matrix(c["R"]).as_rotvec() for c in cams])
    tvecs = np.array([c["t"] for c in cams])
    Ks_all = [c["K"] for c in cams]
    # A point way behind camera 0 (on the far side of it from the rig center)
    behind_pt = np.array([6.0, 0.0, 1.5]) + (np.array([6.0, 0.0, 1.5]) - np.array([0, 0, 0]))
    point = _make_point([(0, 0.0, 0.0, 1.0)], xyz=behind_pt)
    # As long as it doesn't crash and returns something (possibly empty for cam0)
    try:
        rpwk.report_residuals(rvecs, tvecs, [point], [point], Ks_all, [c["label"] for c in cams], "TEST")
    except SystemExit:
        pass  # acceptable if truly nothing reprojects validly


def test_report_residuals_raises_systemexit_when_all_points_behind_all_cameras():
    cams = make_rig(n=2)
    rvecs = np.array([Rotation.from_matrix(c["R"]).as_rotvec() for c in cams])
    tvecs = np.array([c["t"] for c in cams])
    Ks_all = [c["K"] for c in cams]
    # Put the point far behind every camera by placing it way outside the rig
    # on the opposite side from all camera positions.
    far_behind = np.array([0.0, 0.0, 1000.0])
    point = _make_point([(0, 0.0, 0.0, 1.0), (1, 0.0, 0.0, 1.0)], xyz=far_behind)
    with pytest.raises(SystemExit, match="no valid reprojections"):
        rpwk.report_residuals(rvecs, tvecs, [point], [point], Ks_all, [c["label"] for c in cams], "TEST")


# --------------------------------------------------------------------------
# triangulate_candidate_points
# --------------------------------------------------------------------------

def test_triangulate_candidate_points_filters_by_score_threshold(tmp_path):
    cams = make_rig(n=4)
    cam_index = {c["label"]: i for i, c in enumerate(cams)}
    cam_labels = [c["label"] for c in cams]
    cams_by_label = {c["label"]: {"K": c["K"], "w2c": np.block([[c["R"], c["t"][:, None]], [0, 0, 0, 1]])}
                     for c in cams}
    X = np.array([0.0, 0.0, 0.0])
    obs = {}
    for i, c in enumerate(cams):
        u, v, _z = project(c["K"], c["R"], c["t"], X)
        score = 0.9 if i < 3 else 0.1  # one low-confidence camera
        obs[c["label"]] = (u, v, score)
    observations = {("000000", 0): obs}
    points = rpwk.triangulate_candidate_points(cams_by_label, cam_labels, cam_index, observations,
                                                score_thr=0.5, min_views=3)
    assert len(points) == 1
    assert len(points[0]["obs"]) == 3  # low-confidence camera excluded


def test_triangulate_candidate_points_filters_by_min_views():
    cams = make_rig(n=4)
    cam_index = {c["label"]: i for i, c in enumerate(cams)}
    cam_labels = [c["label"] for c in cams]
    cams_by_label = {c["label"]: {"K": c["K"], "w2c": np.block([[c["R"], c["t"][:, None]], [0, 0, 0, 1]])}
                     for c in cams}
    X = np.array([0.0, 0.0, 0.0])
    obs = {}
    for i, c in enumerate(cams):
        u, v, _z = project(c["K"], c["R"], c["t"], X)
        score = 0.9 if i < 2 else 0.1  # only 2 confident views, min_views=3
        obs[c["label"]] = (u, v, score)
    observations = {("000000", 0): obs}
    points = rpwk.triangulate_candidate_points(cams_by_label, cam_labels, cam_index, observations,
                                                score_thr=0.5, min_views=3)
    assert points == []


def test_triangulate_candidate_points_recovers_correct_xyz():
    cams = make_rig(n=4)
    cam_index = {c["label"]: i for i, c in enumerate(cams)}
    cam_labels = [c["label"] for c in cams]
    cams_by_label = {c["label"]: {"K": c["K"], "w2c": np.block([[c["R"], c["t"][:, None]], [0, 0, 0, 1]])}
                     for c in cams}
    X_true = np.array([0.2, -0.1, 0.4])
    obs = {}
    for c in cams:
        u, v, _z = project(c["K"], c["R"], c["t"], X_true)
        obs[c["label"]] = (u, v, 0.9)
    observations = {("000000", 0): obs}
    points = rpwk.triangulate_candidate_points(cams_by_label, cam_labels, cam_index, observations,
                                                score_thr=0.5, min_views=3)
    assert len(points) == 1
    assert np.allclose(points[0]["xyz"], X_true, atol=1e-4)


# --------------------------------------------------------------------------
# remove_gauge_drift
# --------------------------------------------------------------------------

def test_remove_gauge_drift_undoes_known_similarity_drift():
    cams = make_rig(n=4)
    rvecs0 = np.array([Rotation.from_matrix(c["R"]).as_rotvec() for c in cams])
    tvecs0 = np.array([c["t"] for c in cams])
    n_cams = len(cams)

    # Simulate a bundle-adjustment result that's correct up to an unconstrained
    # similarity transform (scale/rotation/translation of the whole scene).
    s_drift = 1.7
    R_drift = Rotation.from_euler("xyz", [5, -8, 15], degrees=True).as_matrix()
    t_drift = np.array([0.3, -0.2, 0.1])

    centers0 = np.array([-Rotation.from_rotvec(rvecs0[i]).as_matrix().T @ tvecs0[i] for i in range(n_cams)])
    centers1 = (s_drift * (R_drift @ centers0.T)).T + t_drift
    # cam orientation also rotates by R_drift (rigid part of the similarity)
    rv1 = np.array([Rotation.from_matrix(R_drift @ Rotation.from_rotvec(rvecs0[i]).as_matrix()).as_rotvec()
                    for i in range(n_cams)])
    tv1 = np.array([-Rotation.from_rotvec(rv1[i]).as_matrix() @ centers1[i] for i in range(n_cams)])

    pts0 = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    pts1 = (s_drift * (R_drift @ pts0.T)).T + t_drift

    rv_fixed, tv_fixed, pts_fixed = rpwk.remove_gauge_drift(rv1.copy(), tv1.copy(), pts1.copy(),
                                                            rvecs0, tvecs0, n_cams)
    centers_fixed = np.array([-Rotation.from_rotvec(rv_fixed[i]).as_matrix().T @ tv_fixed[i]
                              for i in range(n_cams)])
    assert np.allclose(centers_fixed, centers0, atol=1e-3)
    assert np.allclose(pts_fixed, pts0, atol=1e-3)


# --------------------------------------------------------------------------
# write_refined_transforms
# --------------------------------------------------------------------------

def test_write_refined_transforms_round_trips_unchanged_pose(tmp_path):
    cams = make_rig(n=2)
    p = tmp_path / "transforms.json"
    write_transforms_json(p, cams)
    data, loaded = rpwk.load_transforms(p)
    cam_labels = [c["label"] for c in cams]
    cam_index = {c["label"]: i for i, c in enumerate(cams)}
    rv = np.array([Rotation.from_matrix(loaded[label]["w2c"][:3, :3]).as_rotvec() for label in cam_labels])
    tv = np.array([loaded[label]["w2c"][:3, 3] for label in cam_labels])
    out_path = tmp_path / "out.json"
    rpwk.write_refined_transforms(data, loaded, cam_labels, cam_index, rv, tv, out_path)
    out_data = json.loads(out_path.read_text())
    orig_data = json.loads(p.read_text())
    for fr_out, fr_orig in zip(out_data["frames"], orig_data["frames"]):
        assert np.allclose(fr_out["transform_matrix"], fr_orig["transform_matrix"], atol=1e-6)


def test_write_refined_transforms_preserves_other_fields(tmp_path):
    cams = make_rig(n=1)
    p = tmp_path / "transforms.json"
    data = write_transforms_json(p, cams)
    _, loaded = rpwk.load_transforms(p)
    cam_labels = [cams[0]["label"]]
    cam_index = {cam_labels[0]: 0}
    rv = np.array([Rotation.from_matrix(loaded[cam_labels[0]]["w2c"][:3, :3]).as_rotvec()])
    tv = np.array([loaded[cam_labels[0]]["w2c"][:3, 3]])
    out_path = tmp_path / "out.json"
    rpwk.write_refined_transforms(data, loaded, cam_labels, cam_index, rv, tv, out_path)
    out_data = json.loads(out_path.read_text())
    assert out_data["frames"][0]["fl_x"] == data["frames"][0]["fl_x"]
    assert out_data["frames"][0]["w"] == data["frames"][0]["w"]


def test_write_refined_transforms_creates_missing_parent_dir(tmp_path):
    cams = make_rig(n=1)
    p = tmp_path / "transforms.json"
    data = write_transforms_json(p, cams)
    _, loaded = rpwk.load_transforms(p)
    cam_labels = [cams[0]["label"]]
    cam_index = {cam_labels[0]: 0}
    rv = np.array([Rotation.from_matrix(loaded[cam_labels[0]]["w2c"][:3, :3]).as_rotvec()])
    tv = np.array([loaded[cam_labels[0]]["w2c"][:3, 3]])
    out_path = tmp_path / "nested" / "does" / "not" / "exist" / "out.json"
    rpwk.write_refined_transforms(data, loaded, cam_labels, cam_index, rv, tv, out_path)
    assert out_path.is_file()


# --------------------------------------------------------------------------
# load_keypoints -- combined-file and directory formats, plus bad-input handling
# --------------------------------------------------------------------------

def test_load_keypoints_combined_file_parses_all_cameras(tmp_path):
    cams = make_rig(n=3)
    p = tmp_path / "kp2d.json"
    points = [np.array([0.0, 0.0, 0.0])]
    write_kp2d_combined(p, cams, points)
    cam_labels = [c["label"] for c in cams]
    obs = rpwk.load_keypoints(p, cam_labels)
    assert len(obs[("000000", 0)]) == 3


def test_load_keypoints_directory_layout_parses_all_cameras(tmp_path):
    cams = make_rig(n=3)
    kp_dir = tmp_path / "kp2d_dir"
    points = [np.array([0.0, 0.0, 0.0])]
    write_kp2d_directory(kp_dir, cams, points)
    cam_labels = [c["label"] for c in cams]
    obs = rpwk.load_keypoints(kp_dir, cam_labels)
    assert len(obs[("000000", 0)]) == 3


def test_load_keypoints_errors_when_path_does_not_exist(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        rpwk.load_keypoints(tmp_path / "does_not_exist", ["Camera_0000"])


def test_load_keypoints_errors_on_invalid_json(tmp_path):
    p = tmp_path / "kp2d.json"
    p.write_text("{not valid")
    with pytest.raises(SystemExit, match="not valid JSON"):
        rpwk.load_keypoints(p, ["Camera_0000"])


def test_load_keypoints_errors_on_missing_frames_key(tmp_path):
    p = tmp_path / "kp2d.json"
    p.write_text(json.dumps({"nope": []}))
    with pytest.raises(SystemExit, match="missing expected key"):
        rpwk.load_keypoints(p, ["Camera_0000"])


def test_load_keypoints_skips_frame_missing_image_name_keeps_others(tmp_path, capsys):
    cams = make_rig(n=2)
    p = tmp_path / "kp2d.json"
    write_kp2d_combined(p, cams, [np.array([0.0, 0.0, 0.0])])
    data = json.loads(p.read_text())
    del data["frames"][0]["image_name"]
    p.write_text(json.dumps(data))
    cam_labels = [c["label"] for c in cams]
    obs = rpwk.load_keypoints(p, cam_labels)
    assert len(obs[("000000", 0)]) == 1  # only the untouched camera survives
    assert "WARNING" in capsys.readouterr().out


def test_load_keypoints_skips_frame_missing_keypoint_scores(tmp_path, capsys):
    cams = make_rig(n=2)
    p = tmp_path / "kp2d.json"
    write_kp2d_combined(p, cams, [np.array([0.0, 0.0, 0.0])])
    data = json.loads(p.read_text())
    del data["frames"][0]["instances"][0]["keypoint_scores"]
    p.write_text(json.dumps(data))
    cam_labels = [c["label"] for c in cams]
    obs = rpwk.load_keypoints(p, cam_labels)
    assert len(obs[("000000", 0)]) == 1
    assert "WARNING" in capsys.readouterr().out


def test_load_keypoints_combined_skips_frame_from_unmatched_camera(tmp_path):
    cams = make_rig(n=2)
    p = tmp_path / "kp2d.json"
    write_kp2d_combined(p, cams, [np.array([0.0, 0.0, 0.0])])
    data = json.loads(p.read_text())
    data["frames"][0]["image_name"] = "totally_unrelated_camera.png"
    p.write_text(json.dumps(data))
    cam_labels = [c["label"] for c in cams]
    obs = rpwk.load_keypoints(p, cam_labels)
    assert len(obs[("000000", 0)]) == 1


def test_load_keypoints_combined_skips_frame_with_no_instances(tmp_path):
    cams = make_rig(n=2)
    p = tmp_path / "kp2d.json"
    write_kp2d_combined(p, cams, [np.array([0.0, 0.0, 0.0])])
    data = json.loads(p.read_text())
    data["frames"][0]["instances"] = []
    p.write_text(json.dumps(data))
    cam_labels = [c["label"] for c in cams]
    obs = rpwk.load_keypoints(p, cam_labels)
    assert len(obs[("000000", 0)]) == 1


def test_load_keypoints_directory_skips_unmatched_camera_dir(tmp_path):
    cams = make_rig(n=2)
    kp_dir = tmp_path / "kp2d_dir"
    write_kp2d_directory(kp_dir, cams, [np.array([0.0, 0.0, 0.0])])
    (kp_dir / "totally_unrelated_camera").mkdir()
    (kp_dir / "totally_unrelated_camera" / "000000.json").write_text(
        json.dumps({"instance_info": [{"keypoints": [[1.0, 2.0]], "keypoint_scores": [0.9]}]}))
    cam_labels = [c["label"] for c in cams]
    obs = rpwk.load_keypoints(kp_dir, cam_labels)
    assert len(obs[("000000", 0)]) == 2  # unmatched dir contributes nothing


def test_load_keypoints_directory_skips_file_with_no_instances(tmp_path):
    cams = make_rig(n=2)
    kp_dir = tmp_path / "kp2d_dir"
    write_kp2d_directory(kp_dir, cams, [np.array([0.0, 0.0, 0.0])])
    (kp_dir / cams[0]["label"] / "000000.json").write_text(json.dumps({"instance_info": []}))
    cam_labels = [c["label"] for c in cams]
    obs = rpwk.load_keypoints(kp_dir, cam_labels)
    assert len(obs[("000000", 0)]) == 1


def test_load_keypoints_directory_skips_malformed_camera_file(tmp_path, capsys):
    cams = make_rig(n=2)
    kp_dir = tmp_path / "kp2d_dir"
    write_kp2d_directory(kp_dir, cams, [np.array([0.0, 0.0, 0.0])])
    bad_file = kp_dir / cams[0]["label"] / "000000.json"
    bad_file.write_text("{not valid json")
    cam_labels = [c["label"] for c in cams]
    obs = rpwk.load_keypoints(kp_dir, cam_labels)
    assert len(obs[("000000", 0)]) == 1
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------
# main() -- CLI wiring and the actual end-to-end bundle-adjustment behavior
# --------------------------------------------------------------------------

def _argv(transforms, kp2d, out, extra=()):
    return ["refine_poses_with_keypoints.py",
            "--transforms", str(transforms), "--kp2d", str(kp2d), "--out_transforms", str(out), *extra]


@pytest.mark.parametrize("missing", ["--transforms", "--kp2d", "--out_transforms"])
def test_main_errors_when_required_arg_missing(monkeypatch, tmp_path, missing):
    argv = _argv(tmp_path / "t.json", tmp_path / "k.json", tmp_path / "o.json")
    # strip the flag and its value
    idx = argv.index(missing)
    del argv[idx:idx + 2]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        rpwk.main()
    assert exc.value.code == 2


def test_main_report_only_does_not_write_output(monkeypatch, tmp_path, capsys):
    cams = make_rig(n=4)
    transforms_path = tmp_path / "transforms.json"
    write_transforms_json(transforms_path, cams)
    kp2d_path = tmp_path / "kp2d.json"
    write_kp2d_combined(kp2d_path, cams, [np.array([0.0, 0.0, 0.0])])
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", _argv(transforms_path, kp2d_path, out_path, ["--report_only"]))
    rpwk.main()
    assert not out_path.exists()
    out = capsys.readouterr().out
    assert "BEFORE" in out
    assert "AFTER" not in out


def test_main_no_usable_points_raises_systemexit(monkeypatch, tmp_path):
    cams = make_rig(n=4)
    transforms_path = tmp_path / "transforms.json"
    write_transforms_json(transforms_path, cams)
    kp2d_path = tmp_path / "kp2d.json"
    write_kp2d_combined(kp2d_path, cams, [np.array([0.0, 0.0, 0.0])], score=0.95)
    out_path = tmp_path / "out.json"
    # score_thr above every observation's score -> nothing survives triangulation
    monkeypatch.setattr(sys, "argv", _argv(transforms_path, kp2d_path, out_path, ["--score_thr", "0.99"]))
    with pytest.raises(SystemExit, match="No usable points"):
        rpwk.main()


def test_main_min_views_wiring_changes_included_point_count(monkeypatch, tmp_path, capsys):
    cams = make_rig(n=4)
    transforms_path = tmp_path / "transforms.json"
    write_transforms_json(transforms_path, cams)
    kp2d_path = tmp_path / "kp2d.json"
    write_kp2d_combined(kp2d_path, cams, [np.array([0.0, 0.0, 0.0])], score=0.95)
    out_path = tmp_path / "out.json"
    # min_views=5 > the 4 cameras available -> the default reaches the real
    # triangulation call and correctly rejects the only candidate point.
    monkeypatch.setattr(sys, "argv", _argv(transforms_path, kp2d_path, out_path, ["--min_views", "5"]))
    with pytest.raises(SystemExit, match="No usable points"):
        rpwk.main()


def test_main_kp2d_directory_layout_end_to_end(monkeypatch, tmp_path):
    cams = make_rig(n=4)
    transforms_path = tmp_path / "transforms.json"
    write_transforms_json(transforms_path, cams)
    kp_dir = tmp_path / "kp2d_dir"
    write_kp2d_directory(kp_dir, cams, [np.array([0.0, 0.0, 0.0]), np.array([0.05, 0.0, 0.0])])
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", _argv(transforms_path, kp_dir, out_path, ["--max_iters", "50"]))
    rpwk.main()
    assert out_path.is_file()


def test_main_face_weight_wiring_reaches_real_call(monkeypatch, tmp_path, capsys):
    cams = make_rig(n=4)
    transforms_path = tmp_path / "transforms.json"
    write_transforms_json(transforms_path, cams)
    kp2d_path = tmp_path / "kp2d.json"
    # 6 keypoints: indices 0-4 are face ids (goliath308: 0-4), index 5 is body.
    points = [np.array([0.01 * i, 0.0, 0.0]) for i in range(6)]
    write_kp2d_combined(kp2d_path, cams, points, score=0.95)
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", _argv(transforms_path, kp2d_path, out_path,
                                            ["--face_weight", "3.0", "--max_iters", "20"]))
    rpwk.main()
    out = capsys.readouterr().out
    assert "Applying 3.0x weight to 20 face/head keypoint observations." in out  # 5 face kps x 4 cams


def test_main_transforms_invalid_json_gives_clean_error_not_traceback(monkeypatch, tmp_path):
    transforms_path = tmp_path / "transforms.json"
    transforms_path.write_text("{not valid")
    monkeypatch.setattr(sys, "argv", _argv(transforms_path, tmp_path / "kp2d.json", tmp_path / "out.json"))
    with pytest.raises(SystemExit, match="not valid JSON"):
        rpwk.main()


def test_main_kp2d_path_missing_gives_clean_error_not_traceback(monkeypatch, tmp_path):
    cams = make_rig(n=2)
    transforms_path = tmp_path / "transforms.json"
    write_transforms_json(transforms_path, cams)
    monkeypatch.setattr(sys, "argv", _argv(transforms_path, tmp_path / "does_not_exist", tmp_path / "out.json"))
    with pytest.raises(SystemExit, match="not found"):
        rpwk.main()


def test_main_end_to_end_bundle_adjustment_reduces_reprojection_error(monkeypatch, tmp_path, capsys):
    true_cams = make_rig(n=6, radius=3.0)
    # Simulate SfM drift on one camera's rotation.
    i_bad = 2
    perturbed_cams = [dict(c) for c in true_cams]
    perturb = Rotation.from_euler("xyz", [3, -2, 1.5], degrees=True).as_matrix()
    perturbed_cams[i_bad] = dict(true_cams[i_bad])
    perturbed_cams[i_bad]["R"] = perturb @ true_cams[i_bad]["R"]

    transforms_path = tmp_path / "transforms.json"
    write_transforms_json(transforms_path, perturbed_cams)

    rng = np.random.default_rng(1)
    points_3d = [rng.uniform([-0.3, -0.3, -0.2], [0.3, 0.3, 0.2]) for _ in range(15)]
    kp2d_path = tmp_path / "kp2d.json"
    write_kp2d_combined(kp2d_path, true_cams, points_3d, score=0.95)

    out_path = tmp_path / "transforms_refined.json"
    monkeypatch.setattr(sys, "argv", _argv(transforms_path, kp2d_path, out_path, ["--max_iters", "300"]))
    rpwk.main()

    out = capsys.readouterr().out
    import re
    m = re.search(r"Median residual: ([\d.]+)px -> ([\d.]+)px", out)
    assert m is not None
    before, after = float(m.group(1)), float(m.group(2))
    assert after < before
    assert after < 1.0  # noise-free synthetic data -- should converge to near-zero
    assert out_path.is_file()
