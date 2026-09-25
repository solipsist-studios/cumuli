# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Scoring an estimated rig against the rig that actually rendered.

The load-bearing property is that a reconstruction differing from truth
only by a similarity transform scores as perfect. Structure from motion
cannot recover global rotation, translation, or scale, so a scorer that
counted those as error would report a large number for a flawless solve
and hide the errors that matter.
"""

import json
import math

import numpy as np
import pytest

import score_poses_vs_gt as scorer


def look_at(position, target=(0.0, 0.0, 0.0)):
    """An OpenGL camera-to-world looking from `position` at `target`."""
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    forward = forward / np.linalg.norm(forward)
    z = -forward
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(up, z)) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, position
    return c2w


def ring_poses(n=6, radius=3.0, height=1.0):
    poses = {}
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        poses[f"{i:02d}"] = look_at(
            [radius * math.cos(angle), height, radius * math.sin(angle)],
            [0.0, height, 0.0])
    return poses


def rotation_about_y(degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def apply_similarity(poses, scale, R, t):
    out = {}
    for label, c2w in poses.items():
        moved = np.eye(4)
        moved[:3, :3] = R @ c2w[:3, :3]
        moved[:3, 3] = scale * R @ c2w[:3, 3] + t
        out[label] = moved
    return out


def gt_payload(poses, w=640, h=480, fl=500.0):
    return {
        "cameras": [
            {"label": label, "name": f"Camera_{label}", "role": "train",
             "transform_matrix": c2w.tolist(),
             "fl_x": fl, "fl_y": fl, "cx": w / 2, "cy": h / 2, "w": w, "h": h}
            for label, c2w in poses.items()
        ]
    }


# ------------------------------------------------------------- label match
def test_hloc_style_labels_match_rendered_labels():
    """HLOC names a camera by its folder, the renderer by its index."""
    pairs = scorer.match_labels(["Camera_00", "Camera_01"], ["00", "01"])
    assert pairs == {"Camera_00": "00", "Camera_01": "01"}


def test_identical_labels_match_directly():
    assert scorer.match_labels(["00"], ["00"]) == {"00": "00"}


def test_an_unmatchable_label_is_left_out_rather_than_guessed():
    assert scorer.match_labels(["Camera_99"], ["00", "01"]) == {}


# ------------------------------------------------------------- alignment
def test_a_perfect_solve_up_to_a_similarity_scores_as_perfect():
    """Rotate, translate, and scale the whole reconstruction. None of that
    is visible in any image, so none of it is error."""
    gt = ring_poses()
    est = apply_similarity(gt, 0.4, rotation_about_y(37.0),
                           np.array([5.0, -2.0, 1.5]))
    pairs = scorer.match_labels(list(est), list(gt))
    scored = scorer.align_and_score(est, gt, pairs)

    assert scored["scale"] == pytest.approx(1.0 / 0.4, rel=1e-6)
    for camera in scored["cameras"]:
        assert camera["position_error_m"] < 1e-9
        assert camera["rotation_error_deg"] < 1e-6


def test_the_recovered_scale_converts_units_to_metres():
    """On a real capture this number has to be guessed; here it is read."""
    gt = ring_poses()
    est = apply_similarity(gt, 0.25, np.eye(3), np.zeros(3))
    pairs = scorer.match_labels(list(est), list(gt))
    assert scorer.align_and_score(est, gt, pairs)["scale"] == pytest.approx(4.0)


def test_a_displaced_camera_is_reported_in_metres():
    gt = ring_poses()
    est = {k: v.copy() for k, v in gt.items()}
    est["02"][:3, 3] += np.array([0.05, 0.0, 0.0])
    pairs = scorer.match_labels(list(est), list(gt))
    scored = scorer.align_and_score(est, gt, pairs)

    worst = max(scored["cameras"], key=lambda c: c["position_error_m"])
    assert worst["camera_label"] == "02"
    assert worst["position_error_m"] > 0.03


def test_a_rotated_camera_is_reported_in_degrees():
    gt = ring_poses()
    est = {k: v.copy() for k, v in gt.items()}
    est["03"][:3, :3] = rotation_about_y(2.0) @ est["03"][:3, :3]
    pairs = scorer.match_labels(list(est), list(gt))
    scored = scorer.align_and_score(est, gt, pairs)

    worst = max(scored["cameras"], key=lambda c: c["rotation_error_deg"])
    assert worst["camera_label"] == "03"
    assert worst["rotation_error_deg"] == pytest.approx(2.0, abs=0.3)


def test_too_few_matched_cameras_is_refused():
    gt = ring_poses(n=6)
    est = {"00": gt["00"], "01": gt["01"]}
    pairs = scorer.match_labels(list(est), list(gt))
    with pytest.raises(SystemExit, match="similarity fit needs 3"):
        scorer.align_and_score(est, gt, pairs)


# ------------------------------------------------------------ projection
def test_rotation_angle_is_zero_for_identical_rotations():
    R = rotation_about_y(20.0)
    assert scorer.rotation_angle_deg(R, R) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("degrees", [0.5, 5.0, 45.0, 179.0])
def test_rotation_angle_recovers_a_known_rotation(degrees):
    R = rotation_about_y(0.0)
    assert scorer.rotation_angle_deg(R, rotation_about_y(degrees)) == \
        pytest.approx(degrees, abs=1e-6)


def test_joint_points_convert_from_blender_to_the_dataset_world():
    got = scorer.blender_to_opengl_points([[1.0, 2.0, 3.0]])
    assert np.allclose(got, [[1.0, 3.0, -2.0]])


def test_projection_puts_a_point_on_the_optical_axis_at_the_centre():
    c2w = look_at([0.0, 0.0, 3.0], [0.0, 0.0, 0.0])
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    u, v, front = scorer.project_opengl(c2w, K, np.zeros((1, 3)))
    assert front[0]
    assert u[0] == pytest.approx(320.0, abs=1e-6)
    assert v[0] == pytest.approx(240.0, abs=1e-6)


def test_points_behind_the_camera_are_flagged():
    c2w = look_at([0.0, 0.0, 3.0], [0.0, 0.0, 0.0])
    K = np.eye(3)
    _, _, front = scorer.project_opengl(c2w, K, np.array([[0.0, 0.0, 10.0]]))
    assert not front[0]


# ---------------------------------------------------------- write aligned
def test_aligned_output_lands_in_the_ground_truth_frame(tmp_path):
    """A splat trained on the aligned rig is metric and can be scored
    against the same eval cameras as the ground-truth run."""
    gt = ring_poses()
    est = apply_similarity(gt, 0.3, rotation_about_y(-20.0),
                           np.array([1.0, 2.0, 3.0]))
    est_path = tmp_path / "est.json"
    est_path.write_text(json.dumps({"frames": [
        {"camera_label": f"Camera_{label}", "transform_matrix": c2w.tolist()}
        for label, c2w in est.items()]}))

    pairs = scorer.match_labels([f"Camera_{k}" for k in est], list(gt))
    est_by_hloc_label = {f"Camera_{k}": v for k, v in est.items()}
    scored = scorer.align_and_score(est_by_hloc_label, gt, pairs)
    gt_cams = {c["label"]: c for c in gt_payload(gt)["cameras"]}

    out = tmp_path / "aligned.json"
    n = scorer.write_aligned(est_path, scored, pairs, gt_cams, out)
    assert n == len(gt)

    written = json.loads(out.read_text())
    for entry in written["frames"]:
        expected = gt[entry["camera_label"]]
        got = np.asarray(entry["transform_matrix"])
        assert np.allclose(got, expected, atol=1e-9)
        # Labels are rewritten to the ground-truth ones, because every later
        # stage looks a camera up by that literal string.
        assert entry["camera_label"] in gt
        assert entry["k1"] == 0.0


def test_ground_truth_without_training_cameras_is_refused(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"cameras": [
        {"label": "e00", "role": "eval", "transform_matrix": np.eye(4).tolist()}]}))
    with pytest.raises(SystemExit, match="no training cameras"):
        scorer.load_ground_truth(path)


def test_transforms_with_repeated_cameras_keep_the_first(tmp_path):
    """A multi-timestamp solve lists each camera once per instant."""
    path = tmp_path / "t.json"
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 9.0
    path.write_text(json.dumps({"frames": [
        {"camera_label": "00", "transform_matrix": first.tolist()},
        {"camera_label": "00", "transform_matrix": second.tolist()}]}))
    _, poses = scorer.load_transform_map(path)
    assert len(poses) == 1
    assert poses["00"][0, 3] == 0.0


def test_align_source_must_be_one_of_the_scored_solves(tmp_path, monkeypatch):
    """The similarity used to move a rig has to be the one fitted to that
    rig. Combining one solve's poses with another's alignment would look
    plausible and be wrong."""
    import sys

    gt = ring_poses()
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt_payload(gt)))
    est_path = tmp_path / "est.json"
    est_path.write_text(json.dumps({"frames": [
        {"camera_label": label, "transform_matrix": c2w.tolist()}
        for label, c2w in gt.items()]}))
    other = tmp_path / "other.json"
    other.write_text(est_path.read_text())

    monkeypatch.setattr(sys, "argv", [
        "score_poses_vs_gt.py", "--estimated", str(est_path),
        "--ground_truth", str(gt_path), "--align_source", str(other),
        "--write_aligned", str(tmp_path / "aligned.json")])
    with pytest.raises(SystemExit, match="neither --estimated nor --refined"):
        scorer.main()


def test_refined_is_preferred_when_no_source_is_named(tmp_path, monkeypatch, capsys):
    import sys

    gt = ring_poses()
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt_payload(gt)))
    frames = [{"camera_label": label, "transform_matrix": c2w.tolist()}
              for label, c2w in gt.items()]
    est_path = tmp_path / "est.json"
    est_path.write_text(json.dumps({"frames": frames}))
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps({"frames": frames}))

    monkeypatch.setattr(sys, "argv", [
        "score_poses_vs_gt.py", "--estimated", str(est_path),
        "--refined", str(ref_path),
        "--ground_truth", str(gt_path),
        "--write_aligned", str(tmp_path / "aligned.json")])
    scorer.main()
    assert "from the refined solve" in capsys.readouterr().out
