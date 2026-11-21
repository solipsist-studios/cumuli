# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

import json
import sys

import numpy as np
import pytest

import triangulate_and_project_keypoints as tpk

W, H, F = 640, 480, 500.0


def ring_c2w(n, radius=4.0):
    """n OpenGL camera-to-world matrices on a ring around the origin, each
    looking at the origin (OpenGL: camera looks down -z, y up)."""
    poses = []
    for i in range(n):
        a = 2 * np.pi * i / n
        eye = np.array([radius * np.cos(a), 0.3, radius * np.sin(a)])
        back = eye / np.linalg.norm(eye)          # camera +z points away from the target
        right = np.cross([0.0, 1.0, 0.0], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, back, eye
        poses.append(c2w)
    return poses


def write_transforms(path, labels, extra_frame_fields=None):
    frames = []
    for label, c2w in zip(labels, ring_c2w(len(labels))):
        frame = {"camera_label": label, "file_path": f"images/{label}.png",
                 "transform_matrix": c2w.tolist(), "fl_x": F, "fl_y": F, "cx": W / 2, "cy": H / 2}
        frame.update(extra_frame_fields or {})
        frames.append(frame)
    path.write_text(json.dumps({"w": W, "h": H, "frames": frames}))
    return path


def project(points, camera_path):
    """Pixel projections of world points in every camera of camera_path."""
    _, by_cam = tpk.load_cameras(camera_path)
    out = {}
    for label, (K, T) in by_cam.items():
        cam = (T[:3, :3] @ points.T + T[:3, 3:4]).T
        out[label] = (K @ cam.T).T[:, :2] / cam[:, 2:3]
    return out


POINTS = np.array([[0.0, 0.0, 0.0], [0.2, 0.5, -0.1], [-0.3, 1.0, 0.2]])


def write_kp2d_dir(kp2d_dir, pixels, scores=None, tem="000000"):
    for label, px in pixels.items():
        (kp2d_dir / label).mkdir(parents=True, exist_ok=True)
        s = scores[label] if scores else [0.9] * len(px)
        (kp2d_dir / label / f"{tem}.json").write_text(json.dumps(
            {"instance_info": [{"keypoints": px.tolist(), "keypoint_scores": list(s)}]}))


# --------------------------------------------------------------------------
# Triangulation math
# --------------------------------------------------------------------------

def test_triangulate_recovers_known_points(tmp_path):
    cams = write_transforms(tmp_path / "t.json", [f"{i:02d}" for i in range(6)])
    by_cam_tem, by_cam = tpk.load_cameras(cams)
    labels = sorted(by_cam)
    Ks = np.stack([by_cam[c][0] for c in labels])
    Ts = np.stack([by_cam[c][1] for c in labels])
    px = project(POINTS, cams)
    kp2d = np.stack([px[c] for c in labels])
    kp3d, reproj, n_views = tpk.triangulate_points(Ks, Ts, kp2d, np.full(kp2d.shape[:2], 0.9))
    np.testing.assert_allclose(kp3d, POINTS, atol=1e-6)
    assert (reproj < 1e-3).all()
    assert (n_views == 6).all()


def test_keypoint_seen_by_fewer_than_three_confident_views_is_invalid(tmp_path):
    cams = write_transforms(tmp_path / "t.json", [f"{i:02d}" for i in range(6)])
    _, by_cam = tpk.load_cameras(cams)
    labels = sorted(by_cam)
    Ks = np.stack([by_cam[c][0] for c in labels])
    Ts = np.stack([by_cam[c][1] for c in labels])
    px = project(POINTS, cams)
    kp2d = np.stack([px[c] for c in labels])
    scores = np.full(kp2d.shape[:2], 0.9)
    scores[2:, 1] = 0.1  # keypoint 1: only two views pass score_thr=0.6
    kp3d, reproj, n_views = tpk.triangulate_points(Ks, Ts, kp2d, scores)
    assert (kp3d[1] == tpk.INVALID).all()
    assert reproj[1] == tpk.INVALID
    assert n_views[1] == 2
    np.testing.assert_allclose(kp3d[[0, 2]], POINTS[[0, 2]], atol=1e-6)


def test_triangulate_points_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="Ks must have shape"):
        tpk.triangulate_points(np.zeros((2, 3, 3)), np.zeros((3, 4, 4)), np.zeros((3, 5, 2)), np.zeros((3, 5)))


# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", tpk.DISTORTION_KEYS)
def test_nonzero_distortion_is_rejected(tmp_path, key):
    cams = write_transforms(tmp_path / "t.json", ["00", "01", "02"], {key: 0.08})
    with pytest.raises(ValueError, match="nonzero distortion"):
        tpk.load_cameras(cams)


def test_global_distortion_is_rejected(tmp_path):
    cams = write_transforms(tmp_path / "t.json", ["00", "01", "02"])
    tfs = json.loads(cams.read_text())
    tfs["k1"] = -0.02
    cams.write_text(json.dumps(tfs))
    with pytest.raises(ValueError, match="nonzero distortion"):
        tpk.load_cameras(cams)


def test_zero_distortion_fields_are_accepted(tmp_path):
    cams = write_transforms(tmp_path / "t.json", ["00", "01", "02"],
                            {k: 0.0 for k in tpk.DISTORTION_KEYS})
    _, by_cam = tpk.load_cameras(cams)
    assert sorted(by_cam) == ["00", "01", "02"]


def test_missing_intrinsics_raise(tmp_path):
    cams = write_transforms(tmp_path / "t.json", ["00"])
    tfs = json.loads(cams.read_text())
    del tfs["frames"][0]["fl_x"]
    cams.write_text(json.dumps(tfs))
    with pytest.raises(ValueError, match="missing fl_x"):
        tpk.load_cameras(cams)


def test_camera_lookup_tolerates_zero_padding(tmp_path):
    cams = write_transforms(tmp_path / "t.json", ["00", "01"])
    by_cam_tem, by_cam = tpk.load_cameras(cams)
    K, T = tpk.camera_matrices(by_cam_tem, by_cam, "1", "000000")
    np.testing.assert_array_equal(T, by_cam["01"][1])
    with pytest.raises(KeyError):
        tpk.camera_matrices(by_cam_tem, by_cam, "07", "000000")


def test_tem_label_inference():
    assert tpk._infer_tem_label({"file_path": "images/00.png"}, "00") == "0001"
    assert tpk._infer_tem_label({"file_path": "images/cam_0003.png"}, "00") == "0003"
    assert tpk._infer_tem_label({"file_path": "x.png", "frame_id": 7}, "00") == "7"
    assert tpk._infer_tem_label({}, "00") is None


# --------------------------------------------------------------------------
# File round trip and CLI
# --------------------------------------------------------------------------

def test_read_kp2d_rescales_legacy_133_finger_scores(tmp_path):
    scores = np.full(133, 0.8)
    scores[91] = 0.5
    path = tmp_path / "kp.json"
    path.write_text(json.dumps({"instance_info": [{"keypoints": np.zeros((133, 2)).tolist(),
                                                   "keypoint_scores": scores.tolist()}]}))
    _, s = tpk.read_kp2d(path)
    assert s[95] == pytest.approx(0.8 * 0.25)
    assert s[50] == pytest.approx(0.8)


def test_triangulate_skeleton_writes_json(tmp_path):
    labels = [f"{i:02d}" for i in range(5)]
    cams = write_transforms(tmp_path / "t.json", labels)
    kp2d_dir = tmp_path / "poses_2d"
    write_kp2d_dir(kp2d_dir, project(POINTS, cams))
    (kp2d_dir / "association_report.json").write_text("{}")  # stray files are not cameras

    tpk.triangulate_skeleton(cams, kp2d_dir, tmp_path / "poses_3d")

    out = json.loads((tmp_path / "poses_3d" / "000000.json").read_text())["instance_info"][0]
    np.testing.assert_allclose(out["keypoints"], POINTS, atol=1e-6)
    assert len(out["keypoint_reproj"]) == len(POINTS)


def test_triangulate_skeleton_writes_point_cloud(tmp_path):
    pytest.importorskip("open3d")
    labels = [f"{i:02d}" for i in range(5)]
    cams = write_transforms(tmp_path / "t.json", labels)
    kp2d_dir = tmp_path / "poses_2d"
    write_kp2d_dir(kp2d_dir, project(POINTS, cams))
    tpk.triangulate_skeleton(cams, kp2d_dir, tmp_path / "poses_3d", tmp_path / "pcd")
    assert (tmp_path / "pcd" / "000000.ply").is_file()


def base_argv(camera_path, kp2d_dir, out_kp3d_dir, extra=None):
    return ["prog", "--camera_path", str(camera_path), "--kp2d_dir", str(kp2d_dir),
            "--out_kp3d_dir", str(out_kp3d_dir)] + (extra or [])


def test_main_exits_1_on_distorted_cameras(tmp_path, monkeypatch, capsys):
    cams = write_transforms(tmp_path / "t.json", ["00", "01", "02"], {"k1": 0.05})
    kp2d_dir = tmp_path / "poses_2d"
    write_kp2d_dir(kp2d_dir, {c: np.zeros((3, 2)) for c in ["00", "01", "02"]})
    monkeypatch.setattr(sys, "argv", base_argv(cams, kp2d_dir, tmp_path / "out"))
    with pytest.raises(SystemExit) as exc_info:
        tpk.main()
    assert exc_info.value.code == 1
    assert "transforms_hloc.json" in capsys.readouterr().out


def test_main_exits_1_on_missing_kp2d_dir(tmp_path, monkeypatch, capsys):
    cams = write_transforms(tmp_path / "t.json", ["00"])
    monkeypatch.setattr(sys, "argv", base_argv(cams, tmp_path / "nope", tmp_path / "out"))
    with pytest.raises(SystemExit) as exc_info:
        tpk.main()
    assert exc_info.value.code == 1
    assert "kp2d_dir not found" in capsys.readouterr().out


@pytest.mark.parametrize("missing_flag", ["--camera_path", "--kp2d_dir", "--out_kp3d_dir"])
def test_main_errors_when_required_arg_missing(tmp_path, monkeypatch, missing_flag):
    argv = base_argv(tmp_path / "t.json", tmp_path / "kp2d", tmp_path / "kp3d")
    idx = argv.index(missing_flag)
    argv = argv[:idx] + argv[idx + 2:]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        tpk.main()
    assert exc_info.value.code == 2
