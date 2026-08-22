"""Unit tests for project_skeleton_conditioning.py.

The conditioning map tells the video model where the body is in views no camera
saw, so two things have to hold or the control signal actively misleads it:
confidence must fall as reprojection error RISES (the field it comes from is an
error, not a score), and face keypoints must fade as the sweep passes behind the
head instead of being drawn through the skull."""

import json

import numpy as np
import pytest

import project_skeleton_conditioning as psc


def identity_camera(focal=800.0, principal=512.0):
    """A camera at the origin looking down +Z, so a point at (0, 0, d) lands
    dead centre at depth d."""
    intrinsics = np.array([[focal, 0.0, principal], [0.0, focal, principal], [0.0, 0.0, 1.0]])
    return intrinsics, np.eye(4)


def write_kp3d_file(path, keypoints, reproj=None):
    instance = {"keypoints": np.asarray(keypoints, dtype=float).tolist()}
    if reproj is not None:
        instance["keypoint_reproj"] = np.asarray(reproj, dtype=float).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"instance_info": [instance]}))


# --------------------------------------------------------------------------
# Loading and validity
# --------------------------------------------------------------------------

def test_load_kp3d_reads_keypoints_and_error(tmp_path):
    path = tmp_path / "000000.json"
    write_kp3d_file(path, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [0.5, 9.0])

    keypoints, reproj = psc.load_kp3d(path)
    assert keypoints.shape == (2, 3)
    assert reproj.tolist() == [0.5, 9.0]


def test_load_kp3d_defaults_the_error_when_absent(tmp_path):
    path = tmp_path / "000000.json"
    write_kp3d_file(path, [[0.0, 1.0, 2.0]])
    _, reproj = psc.load_kp3d(path)
    assert reproj.tolist() == [0.0]


def test_load_kp3d_rejects_two_dimensional_keypoints(tmp_path):
    # A poses_2d file handed in by mistake would silently project nonsense.
    path = tmp_path / "000000.json"
    write_kp3d_file(path, [[10.0, 20.0], [30.0, 40.0]])
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        psc.load_kp3d(path)


def test_valid_mask_rejects_the_invalid_sentinel():
    keypoints = np.array([[0.0, 0.0, 1.0], [psc.INVALID, psc.INVALID, psc.INVALID], [1.0, 1.0, 1.0]])
    assert psc.valid_mask(keypoints).tolist() == [True, False, True]


def test_valid_mask_rejects_non_finite_keypoints():
    keypoints = np.array([[0.0, 0.0, 1.0], [np.nan, 0.0, 1.0], [np.inf, 0.0, 1.0]])
    assert psc.valid_mask(keypoints).tolist() == [True, False, False]


# --------------------------------------------------------------------------
# Confidence from reprojection error
# --------------------------------------------------------------------------

def test_score_falls_as_reprojection_error_rises():
    # The field is an ERROR; reading it as a score would invert every confidence.
    valid = np.ones(4, dtype=bool)
    scores = psc.reproj_to_score(np.array([0.0, 2.5, 5.0, 10.0]), valid, tau=10.0)
    assert scores.tolist() == pytest.approx([1.0, 0.75, 0.5, 0.0])


def test_score_clips_at_zero_beyond_tau():
    valid = np.ones(2, dtype=bool)
    assert psc.reproj_to_score(np.array([25.0, 1e5]), valid, tau=10.0).tolist() == [0.0, 0.0]


def test_score_is_zero_for_invalid_keypoints():
    valid = np.array([True, False])
    scores = psc.reproj_to_score(np.array([0.0, 0.0]), valid, tau=10.0)
    assert scores.tolist() == [1.0, 0.0]


def test_score_is_zero_when_the_error_is_the_invalid_sentinel():
    # Untriangulated keypoints carry INVALID in the error field too; its absolute
    # value is enormous, so a naive mapping would clip to 0 by luck. Be explicit.
    valid = np.ones(1, dtype=bool)
    assert psc.reproj_to_score(np.array([psc.INVALID]), valid, tau=10.0).tolist() == [0.0]


# --------------------------------------------------------------------------
# Face orientation
# --------------------------------------------------------------------------

UP = np.array([0.0, 1.0, 0.0])


def face_keypoints(n=100, facing=(0.0, 0.0, -1.0)):
    """A head whose face points along `facing`, padded out to a full layout.

    Anatomically placed: the eyes straddle the head's side axis and the nose tip
    sits forward of the eye midpoint AND below it. The vertical drop is what
    makes this a real test rather than a degenerate one."""
    keypoints = np.zeros((n, 3))
    forward = np.asarray(facing, dtype=float)
    forward = forward / np.linalg.norm(forward)
    side = np.cross(UP, forward)
    keypoints[psc.LEFT_EYE_KP] = -side * 0.03
    keypoints[psc.RIGHT_EYE_KP] = side * 0.03
    keypoints[psc.NOSE_KP] = forward * 0.05 - UP * 0.02
    return keypoints


def test_face_normal_points_the_way_the_face_points():
    keypoints = face_keypoints(facing=(0.0, 0.0, -1.0))
    normal = psc.face_normal(keypoints, np.ones(len(keypoints), dtype=bool), UP)
    assert normal is not None
    assert np.dot(normal, np.array([0.0, 0.0, -1.0])) == pytest.approx(1.0, abs=1e-6)


def test_face_normal_ignores_the_nose_drop():
    # The nose sits below the eyes; the facing direction must be horizontal
    # regardless, or the fade would track head height instead of head yaw.
    normal = psc.face_normal(face_keypoints(facing=(1.0, 0.0, 0.0)),
                             np.ones(100, dtype=bool), UP)
    assert np.dot(normal, UP) == pytest.approx(0.0, abs=1e-9)
    assert np.dot(normal, np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-6)


def test_face_normal_is_none_without_all_three_face_keypoints():
    keypoints = face_keypoints()
    valid = np.ones(len(keypoints), dtype=bool)
    valid[psc.NOSE_KP] = False
    assert psc.face_normal(keypoints, valid, UP) is None


def test_face_scores_survive_a_head_on_camera():
    # Camera looks along +Z; the face points back along -Z, straight at it.
    keypoints = face_keypoints(facing=(0.0, 0.0, -1.0))
    normal = psc.face_normal(keypoints, np.ones(len(keypoints), dtype=bool), UP)
    faded = psc.fade_face_scores(np.ones(100), normal, np.eye(4))
    assert faded[psc.NOSE_KP] == pytest.approx(1.0)
    assert faded[50] == pytest.approx(1.0)


def test_face_scores_vanish_from_directly_behind():
    # The face points the same way the camera looks, so the camera is behind the
    # head. Drawing eyes and nose here paints them through the skull.
    keypoints = face_keypoints(facing=(0.0, 0.0, 1.0))
    normal = psc.face_normal(keypoints, np.ones(len(keypoints), dtype=bool), UP)
    faded = psc.fade_face_scores(np.ones(100), normal, np.eye(4))
    assert faded[psc.NOSE_KP] == pytest.approx(0.0, abs=1e-6)
    assert faded[50] == pytest.approx(0.0, abs=1e-6)


def test_face_scores_are_half_way_side_on():
    keypoints = face_keypoints(facing=(1.0, 0.0, 0.0))
    normal = psc.face_normal(keypoints, np.ones(len(keypoints), dtype=bool), UP)
    faded = psc.fade_face_scores(np.ones(100), normal, np.eye(4))
    assert faded[psc.NOSE_KP] == pytest.approx(0.5, abs=1e-6)


def test_face_fade_leaves_body_keypoints_alone():
    keypoints = face_keypoints(facing=(0.0, 0.0, 1.0))
    normal = psc.face_normal(keypoints, np.ones(len(keypoints), dtype=bool), UP)
    faded = psc.fade_face_scores(np.ones(100), normal, np.eye(4))
    assert faded[psc.FACE_KP_END:].tolist() == pytest.approx([1.0] * (100 - psc.FACE_KP_END))
    assert faded[5] == pytest.approx(1.0)   # shoulders and below are untouched


def test_world_up_prefers_the_recorded_axis():
    up = np.array([0.0, 0.0, 1.0])
    assert psc.world_up({"up": up.tolist()}, np.eye(4)) == pytest.approx(up, abs=1e-12)


def test_world_up_normalises_the_recorded_axis():
    assert psc.world_up({"up": [0.0, 0.0, 4.0]}, np.eye(4)) == pytest.approx([0.0, 0.0, 1.0])


def test_world_up_falls_back_to_the_camera_only_approximately():
    # lookat_w2c orthogonalizes down against forward, so an ELEVATED camera's
    # image-down axis is tilted off world up. The fallback is close for a level
    # camera and drifts with elevation, which is why the sweep records the axis.
    import render_orbit_views as rov

    up = np.array([0.0, 0.0, 1.0])
    level = rov.lookat_w2c(np.array([4.0, 0.0, 0.0]), np.zeros(3), -up)
    assert psc.world_up({}, level) == pytest.approx(up, abs=1e-9)

    elevated = rov.lookat_w2c(np.array([4.0, 0.0, 0.5]), np.zeros(3), -up)
    assert psc.world_up({}, elevated) != pytest.approx(up, abs=1e-3)


def test_face_fade_is_a_no_op_without_a_normal():
    scores = np.linspace(0.1, 1.0, 100)
    assert psc.fade_face_scores(scores, None, np.eye(4)).tolist() == pytest.approx(scores.tolist())


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------

def test_projection_puts_an_on_axis_point_at_the_principal_point():
    intrinsics, w2c = identity_camera(focal=800.0, principal=512.0)
    uv, depth = psc.project(np.array([[0.0, 0.0, 4.0]]), intrinsics, w2c)
    assert uv[0].tolist() == pytest.approx([512.0, 512.0])
    assert depth[0] == pytest.approx(4.0)


def test_projection_scales_with_focal_length_over_depth():
    intrinsics, w2c = identity_camera(focal=800.0, principal=512.0)
    uv, _ = psc.project(np.array([[1.0, 0.0, 4.0]]), intrinsics, w2c)
    assert uv[0][0] == pytest.approx(512.0 + 800.0 / 4.0)


def test_projection_marks_points_behind_the_camera_invalid():
    intrinsics, w2c = identity_camera()
    uv, depth = psc.project(np.array([[0.0, 0.0, -3.0], [0.0, 0.0, 0.0]]), intrinsics, w2c)
    assert (uv == psc.INVALID).all()
    assert (depth == psc.INVALID).all()


def test_projection_reports_true_camera_space_depth():
    # Depth is what lets the drawer sort limbs back-to-front in a novel view.
    intrinsics, w2c = identity_camera()
    _, depth = psc.project(np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 7.0]]), intrinsics, w2c)
    assert depth.tolist() == pytest.approx([2.0, 7.0])


# --------------------------------------------------------------------------
# Output format
# --------------------------------------------------------------------------

def test_write_kp2d_matches_diffuman4d_format(tmp_path):
    path = tmp_path / "nested" / "0000.json"
    psc.write_kp2d(path, np.array([[1.0, 2.0]]), np.array([3.0]), np.array([0.75]))

    instance = json.loads(path.read_text())["instance_info"][0]
    assert instance["keypoints"] == [[1.0, 2.0]]
    assert instance["keypoint_depths"] == [3.0]
    assert instance["keypoint_scores"] == [0.75]


# --------------------------------------------------------------------------
# Locating the 3D keypoints
# --------------------------------------------------------------------------

def test_resolve_kp3d_finds_the_per_frame_layout(tmp_path):
    frame = tmp_path / "frame_0000"
    (frame / "poses_3d").mkdir(parents=True)
    (frame / "poses_3d" / "000000.json").write_text("{}")
    assert psc.resolve_kp3d(frame, "poses_3d", "000000", None, 0) == frame / "poses_3d" / "000000.json"


def test_resolve_kp3d_falls_back_to_the_frame_index(tmp_path):
    frame = tmp_path / "frame_0007"
    (frame / "poses_3d").mkdir(parents=True)
    (frame / "poses_3d" / "000007.json").write_text("{}")
    assert psc.resolve_kp3d(frame, "poses_3d", "000000", None, 7) == frame / "poses_3d" / "000007.json"


def test_resolve_kp3d_honours_a_flat_directory(tmp_path):
    flat = tmp_path / "poses_3d"
    flat.mkdir()
    (flat / "000003.json").write_text("{}")
    assert psc.resolve_kp3d(tmp_path / "unused", "poses_3d", "000000", flat, 3) == flat / "000003.json"


def test_resolve_kp3d_returns_none_when_absent(tmp_path):
    assert psc.resolve_kp3d(tmp_path, "poses_3d", "000000", None, 0) is None


# --------------------------------------------------------------------------
# End to end over a sweep
# --------------------------------------------------------------------------

def make_sweep(tmp_path, n_frames=3, reproj=0.5, with_keypoints=True):
    """A minimal render_pair_sweep.py output plus matching per-frame keypoints."""
    sweep = tmp_path / "03_to_07"
    sweep.mkdir(exist_ok=True)
    frames = []
    for index in range(n_frames):
        frame_dir = tmp_path / f"frame_{index:04d}"
        frame_dir.mkdir(exist_ok=True)
        if with_keypoints:
            keypoints = face_keypoints(n=100, facing=(0.0, 0.0, -1.0))
            keypoints[:, 2] += 4.0  # push the whole body in front of the camera
            write_kp3d_file(frame_dir / "poses_3d" / "000000.json", keypoints, [reproj] * 100)
        frames.append({"idx": index, "frame_dir": str(frame_dir), "time": index / 30.0,
                       "w2c": np.eye(4).tolist()})
    (sweep / "cameras.json").write_text(json.dumps(
        {"pair": ["03", "07"], "res": 1024, "fl_x": 800.0, "fl_y": 800.0,
         "cx": 512.0, "cy": 512.0, "up": UP.tolist(), "frames": frames}))
    return sweep


def test_project_sweep_writes_one_file_per_frame(tmp_path):
    sweep = make_sweep(tmp_path, n_frames=4)
    out = tmp_path / "skeletons"

    summary = psc.project_sweep(sweep, out)
    assert summary["frames"] == 4
    assert sorted(p.name for p in (out / "kp2d" / "03_to_07").glob("*.json")) == \
        ["0000.json", "0001.json", "0002.json", "0003.json"]


def test_project_sweep_reports_the_error_distribution(tmp_path):
    # The distribution is how you calibrate --reproj_tau; a default guess at the
    # wrong capture resolution silently empties every map.
    sweep = make_sweep(tmp_path, reproj=3.0)
    summary = psc.project_sweep(sweep, tmp_path / "skeletons")
    assert summary["reproj_px"]["median"] == pytest.approx(3.0)


def test_project_sweep_scores_reflect_the_error(tmp_path):
    sweep = make_sweep(tmp_path, n_frames=1, reproj=5.0)
    out = tmp_path / "skeletons"
    psc.project_sweep(sweep, out, reproj_tau=10.0)

    instance = json.loads((out / "kp2d" / "03_to_07" / "0000.json").read_text())["instance_info"][0]
    assert instance["keypoint_scores"][50] == pytest.approx(0.5)


def test_project_sweep_records_missing_frames_without_failing(tmp_path):
    sweep = make_sweep(tmp_path, n_frames=3)
    (tmp_path / "frame_0001" / "poses_3d" / "000000.json").unlink()

    summary = psc.project_sweep(sweep, tmp_path / "skeletons")
    assert summary["frames"] == 2
    assert summary["missing_frames"] == [1]


def test_project_sweep_fails_when_no_frame_has_keypoints(tmp_path):
    sweep = make_sweep(tmp_path, with_keypoints=False)
    with pytest.raises(FileNotFoundError, match="triangulate_and_project_keypoints.py"):
        psc.project_sweep(sweep, tmp_path / "skeletons")


def test_project_sweep_needs_a_rendered_sweep(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="render_pair_sweep.py first"):
        psc.project_sweep(tmp_path / "empty", tmp_path / "skeletons")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_projects_and_reports(tmp_path, monkeypatch, capsys):
    sweep = make_sweep(tmp_path)
    monkeypatch.setattr("sys.argv", ["project_skeleton_conditioning.py",
                                     "--sweep_dir", str(sweep),
                                     "--out_dir", str(tmp_path / "skeletons")])
    assert psc.main() == 0
    assert "projected 3 frames" in capsys.readouterr().out


def test_cli_warns_when_tau_would_empty_the_maps(tmp_path, monkeypatch, capsys):
    sweep = make_sweep(tmp_path, reproj=20.0)
    monkeypatch.setattr("sys.argv", ["project_skeleton_conditioning.py",
                                     "--sweep_dir", str(sweep),
                                     "--out_dir", str(tmp_path / "skeletons"),
                                     "--reproj_tau", "5.0"])
    assert psc.main() == 0
    assert "MEDIAN error is at or past" in capsys.readouterr().err


def test_cli_reports_a_missing_sweep(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["project_skeleton_conditioning.py",
                                     "--sweep_dir", str(tmp_path / "absent"),
                                     "--out_dir", str(tmp_path / "skeletons")])
    assert psc.main() == 1
    assert "render_pair_sweep.py first" in capsys.readouterr().err
