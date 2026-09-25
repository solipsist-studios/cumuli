# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""build_hybrid_dataset.py's --gen_run fan-out: several 4DAnyone runs must
merge into one flat pool of generated views, namespaced so camera-id
collisions between independently-numbered runs cannot alias, with culling
and the init cloud both still behaving correctly across the whole pool.
"""

import json
import math
import sys

import numpy as np
from PIL import Image

import build_hybrid_dataset as bhd

SIDE = 8  # square real-camera crop, chosen so und/sfm grid widths can match 1:1


def _real_c2w(azimuth_deg, radius=3.0, height=1.5):
    """OpenGL c2w looking at the origin from `azimuth_deg` about +Y."""
    az = math.radians(azimuth_deg)
    eye = np.array([radius * math.cos(az), height, radius * math.sin(az)])
    forward = -eye / np.linalg.norm(eye)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, true_up, -forward, eye
    return c2w


def write_real_fixtures(root, azimuths_deg):
    """A minimal real capture: one frame per camera, square crops so the
    crop-origin recovery is trivial (und/sfm grids equal, full cx==frame cx)."""
    real_dir = root / "real"
    root_dir = root / "real_root"
    masks_dir = root / "masks"
    real_dir.mkdir()
    root_dir.mkdir()
    masks_dir.mkdir()

    frames = []
    und_frames = []
    for i, az in enumerate(azimuths_deg):
        cam = f"cam{i:02d}"
        camera_id = f"{i:04d}"
        c2w = _real_c2w(az)
        frames.append({
            "file_path": f"{cam}/frame_00001",
            "camera_label": camera_id,
            "transform_matrix": c2w.tolist(),
            "fl_x": 500.0, "fl_y": 500.0, "cx": SIDE / 2, "cy": SIDE / 2,
            "w": SIDE, "h": SIDE, "time": 0.0,
        })
        und_frames.append({"camera_label": camera_id, "cx": SIDE / 2, "cy": SIDE / 2})

        (root_dir / cam).mkdir(exist_ok=True)
        Image.fromarray(np.full((SIDE, SIDE, 3), 120, dtype=np.uint8)).save(
            root_dir / f"{cam}/frame_00001.jpg")
        (masks_dir / camera_id).mkdir(exist_ok=True)
        Image.fromarray(np.full((SIDE, SIDE), 255, dtype=np.uint8)).save(
            masks_dir / camera_id / "f0001.png")

    real_transforms = root / "real_transforms.json"
    real_transforms.write_text(json.dumps({"w": SIDE, "h": SIDE, "frames": frames}))
    undistorted_transforms = root / "undistorted_transforms.json"
    undistorted_transforms.write_text(json.dumps({"frames": und_frames}))
    return real_transforms, root_dir, masks_dir, undistorted_transforms


def write_gen_run(root, name, cam_labels, n_frames=2, with_points=False):
    """A minimal converted 4DAnyone dataset: RGBA frames plus optionally a
    tiny points3d.ply, matching fdanyone_to_omg4.py's output shape."""
    gen_dir = root / f"gen_{name}"
    frames = []
    for cam in cam_labels:
        (gen_dir / cam).mkdir(parents=True, exist_ok=True)
        c2w = np.eye(4)
        c2w[:3, 3] = [1.0, 1.5, 1.0]  # arbitrary, distinct from any real camera
        for f in range(1, n_frames + 1):
            frames.append({
                "file_path": f"{cam}/frame_{f:05d}",
                "transform_matrix": c2w.tolist(),
                "fl_x": 500.0, "fl_y": 500.0, "cx": SIDE / 2, "cy": SIDE / 2,
                "w": SIDE, "h": SIDE,
            })
            rgba = np.zeros((SIDE, SIDE, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            Image.fromarray(rgba, mode="RGBA").save(gen_dir / cam / f"frame_{f:05d}.png")
    (gen_dir / "transforms_train.json").write_text(
        json.dumps({"w": SIDE, "h": SIDE, "frames": frames}))
    if with_points:
        from plyfile import PlyData, PlyElement
        n = 5
        data = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                  ("time", "f4")])
        data["x"] = np.arange(n, dtype=np.float32)
        PlyData([PlyElement.describe(data, "vertex")]).write(str(gen_dir / "points3d.ply"))
    return gen_dir


def write_motion_dir(root, n_frames=2):
    motion_dir = root / "motion"
    motion_dir.mkdir()
    (motion_dir / "motion.json").write_text(
        json.dumps({"source_frame_indices": list(range(n_frames))}))
    return motion_dir


def write_identity_transform(root):
    path = root / "transform.json"
    path.write_text(json.dumps({"T_rig_from_canonical": np.eye(4).tolist()}))
    return path


def run_bhd(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["build_hybrid_dataset.py"] + argv)
    bhd.main()


def base_argv(tmp_path, real_transforms, real_root, masks_root, undistorted_transforms,
              motion_dir, transform, out_dir, gen_runs, **extra):
    argv = [
        "--real_transforms", str(real_transforms),
        "--real_root", str(real_root),
        "--masks_root", str(masks_root),
        "--undistorted_transforms", str(undistorted_transforms),
        "--motion_dir", str(motion_dir),
        "--transform", str(transform),
        "--out_dir", str(out_dir),
        "--und_grid_width", str(SIDE), "--sfm_grid_width", str(SIDE),
        "--holdout", "cam99",  # no real camera has this label: nothing held out
    ]
    for name, gen_dir in gen_runs:
        argv += ["--gen_run", name, str(gen_dir)]
    for k, v in extra.items():
        argv += [f"--{k}", str(v)]
    return argv


# --------------------------------------------------------------- fan-out
def test_gen_run_namespaces_cameras_by_run_name(tmp_path, monkeypatch):
    real_transforms, real_root, masks_root, und = write_real_fixtures(tmp_path, [0.0])
    motion_dir = write_motion_dir(tmp_path)
    transform = write_identity_transform(tmp_path)
    gen_a = write_gen_run(tmp_path, "a", ["gen00"])
    gen_b = write_gen_run(tmp_path, "b", ["gen00"])  # same camera-id, different run
    out_dir = tmp_path / "out"

    run_bhd(monkeypatch, base_argv(
        tmp_path, real_transforms, real_root, masks_root, und, motion_dir, transform,
        out_dir, gen_runs=[("a", gen_a), ("b", gen_b)]))

    train = json.loads((out_dir / "transforms_train.json").read_text())
    gen_paths = sorted(f["file_path"] for f in train["frames"] if f["file_path"].startswith("gen/"))
    assert gen_paths == [
        "gen/a_gen00/frame_00001", "gen/a_gen00/frame_00002",
        "gen/b_gen00/frame_00001", "gen/b_gen00/frame_00002",
    ]
    assert (out_dir / "gen" / "a_gen00" / "frame_00001.png").is_file()
    assert (out_dir / "gen" / "b_gen00" / "frame_00001.png").is_file()


def test_motion_dir_and_transform_are_shared_across_runs(tmp_path, monkeypatch):
    """Every --gen_run entry is retimed and reposed by the SAME single
    --motion_dir/--transform, matching fit_rig_motion.py running once per
    capture (verified separately) rather than once per generation run."""
    real_transforms, real_root, masks_root, und = write_real_fixtures(tmp_path, [0.0])
    motion_dir = write_motion_dir(tmp_path)
    transform = write_identity_transform(tmp_path)
    gen_a = write_gen_run(tmp_path, "a", ["gen00"])
    gen_b = write_gen_run(tmp_path, "b", ["gen01"])
    out_dir = tmp_path / "out"

    run_bhd(monkeypatch, base_argv(
        tmp_path, real_transforms, real_root, masks_root, und, motion_dir, transform,
        out_dir, gen_runs=[("a", gen_a), ("b", gen_b)]))

    train = json.loads((out_dir / "transforms_train.json").read_text())
    gen_entries = [f for f in train["frames"] if f["file_path"].startswith("gen/")]
    assert len(gen_entries) == 4
    # Identity transform: generated poses pass through unchanged.
    for e in gen_entries:
        assert np.allclose(np.asarray(e["transform_matrix"])[:3, 3], [1.0, 1.5, 1.0])


# ------------------------------------------------------------------ culling
def test_min_separation_deg_culls_across_the_whole_pool(tmp_path, monkeypatch):
    """A generated view close to a real camera's azimuth is culled
    regardless of which --gen_run produced it -- every run is one flat pool
    for this purpose."""
    real_transforms, real_root, masks_root, und = write_real_fixtures(tmp_path, [0.0, 90.0])
    motion_dir = write_motion_dir(tmp_path, n_frames=1)
    transform = write_identity_transform(tmp_path)

    # gen00 in run "near" sits AT the same position as real cam00 (azimuth 0,
    # radius 3, height 1.5) -- a duplicate. gen00 in run "far" sits opposite
    # the rig entirely.
    near_dir = tmp_path / "gen_near"
    (near_dir / "gen00").mkdir(parents=True)
    near_c2w = _real_c2w(0.0)
    (near_dir / "transforms_train.json").write_text(json.dumps({
        "frames": [{"file_path": "gen00/frame_00001",
                    "transform_matrix": near_c2w.tolist(),
                    "fl_x": 500.0, "fl_y": 500.0, "cx": SIDE / 2, "cy": SIDE / 2,
                    "w": SIDE, "h": SIDE}]}))
    rgba = np.zeros((SIDE, SIDE, 4), dtype=np.uint8); rgba[..., 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(near_dir / "gen00" / "frame_00001.png")

    far_dir = write_gen_run(tmp_path, "far", ["gen00"], n_frames=1)
    far_json = json.loads((far_dir / "transforms_train.json").read_text())
    far_c2w = _real_c2w(200.0)  # far from both real cameras (0, 90)
    far_json["frames"][0]["transform_matrix"] = far_c2w.tolist()
    (far_dir / "transforms_train.json").write_text(json.dumps(far_json))

    out_dir = tmp_path / "out"
    run_bhd(monkeypatch, base_argv(
        tmp_path, real_transforms, real_root, masks_root, und, motion_dir, transform,
        out_dir, gen_runs=[("near", near_dir), ("far", far_dir)],
        min_separation_deg="15"))

    train = json.loads((out_dir / "transforms_train.json").read_text())
    gen_labels = {f["file_path"].split("/")[1] for f in train["frames"]
                 if f["file_path"].startswith("gen/")}
    assert gen_labels == {"far_gen00"}   # near_gen00 culled as a real-camera duplicate


# --------------------------------------------------------------- init cloud
def test_only_the_first_runs_init_cloud_is_used(tmp_path, monkeypatch):
    real_transforms, real_root, masks_root, und = write_real_fixtures(tmp_path, [0.0])
    motion_dir = write_motion_dir(tmp_path)
    transform = write_identity_transform(tmp_path)
    gen_a = write_gen_run(tmp_path, "a", ["gen00"], with_points=True)
    gen_b = write_gen_run(tmp_path, "b", ["gen01"], with_points=True)
    out_dir = tmp_path / "out"

    from plyfile import PlyData
    run_bhd(monkeypatch, base_argv(
        tmp_path, real_transforms, real_root, masks_root, und, motion_dir, transform,
        out_dir, gen_runs=[("a", gen_a), ("b", gen_b)]))

    out_ply = PlyData.read(str(out_dir / "points3d.ply"))
    a_ply = PlyData.read(str(gen_a / "points3d.ply"))
    assert len(out_ply["vertex"]) == len(a_ply["vertex"])
    assert np.allclose(np.asarray(out_ply["vertex"]["x"]), np.asarray(a_ply["vertex"]["x"]))
