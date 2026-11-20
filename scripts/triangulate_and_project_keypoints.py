#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 AND Apache-2.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
# MODIFIED FROM Diffuman4D (Copyright 2025 ZJU3DV, Apache License 2.0),
# scripts/preprocess/triangulate_skeleton.py, utils/triang_utils.py and
# src/data/utils/camera_parser.py, by Solipsist Studios Inc.: this file
# merges and changes that code (one input path kept, camera handling
# rewritten, distortion now rejected). The license is
# scripts/vendor/diffuman4d/LICENSE and the full list of changes is in
# scripts/vendor/diffuman4d/NOTICE.md.

"""
triangulate_and_project_keypoints.py

Triangulates 3D keypoints from the real cameras' 2D detections (poses_2d,
from split_keypoints_per_camera.py) and writes them to --out_kp3d_dir, plus
a point cloud per frame to --out_pcd_dir.

Each keypoint is solved independently. Views scoring below --score_thr (or
below the score that keeps at most 24 views) are dropped, a keypoint seen by
fewer than 3 views is marked invalid, and the rest get a score-weighted
linear (DLT) estimate refined by a Huber-loss reprojection fit. Invalid
keypoints are written as -1e6 in every coordinate.

The camera file must describe undistorted pinhole cameras: the solve uses
only fl_x/fl_y/cx/cy. A camera with a nonzero k1..k4/p1/p2 is rejected,
because triangulating distorted images against a pinhole model gives wrong
3D keypoints that look plausible. Pass the post-undistort transforms
(transforms_refined.json), not transforms_hloc.json.

Ported from Diffuman4D's triangulate_skeleton.py, keeping only the path
cumuli runs; see scripts/vendor/diffuman4d/NOTICE.md. The name is
historical: the 2D re-projection output is gone, since nothing consumed it.

conda env: cumuli

Usage:
    python3 triangulate_and_project_keypoints.py \\
        --camera_path /path/to/transforms.json \\
        --kp2d_dir /path/to/poses_2d \\
        --out_kp3d_dir /path/to/poses_3d \\
        --out_pcd_dir /path/to/poses_pcd

Input layout: <kp2d_dir>/<camera_label>/<tem_label>.json, each
{"instance_info": [{"keypoints": [[x, y], ...], "keypoint_scores": [...]}]}.
Output: <out_kp3d_dir>/<tem_label>.json ({"instance_info": [{"keypoints",
"keypoint_reproj"}]}) and <out_pcd_dir>/<tem_label>.ply.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

INVALID = -1e6
MIN_VIEWS = 3
MAX_VIEWS = 24
DISTORTION_KEYS = ("k1", "k2", "k3", "k4", "p1", "p2")


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

def _infer_tem_label(frame: dict, camera_label: str) -> str | None:
    """Frame label for a transforms entry: an explicit field, else "0001"
    for a single-instant capture whose file stem is the camera label, else
    the last number in the file stem."""
    for key in ("tem_label", "temporal_label", "frame_label", "time_label", "frame_id"):
        if frame.get(key) is not None:
            return str(frame[key])
    file_path = frame.get("file_path")
    if not file_path:
        return None
    basename = str(file_path).replace("\\", "/").split("/")[-1]
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename
    if stem == camera_label:
        return "0001"
    digits = "".join(c if c.isdigit() else " " for c in stem.split(".")[0]).split()
    return digits[-1] if digits else None


def _label_candidates(label: str, width: int) -> list[str]:
    """label, then its unpadded and width-padded forms when it is numeric."""
    out = [label]
    if label.isdigit():
        out += [str(int(label)), f"{int(label):0{width}d}"]
    return list(dict.fromkeys(out))


def check_undistorted(tfs: dict, camera_path) -> None:
    """Refuse a camera file with any nonzero distortion coefficient."""
    bad = []
    for frame in tfs.get("frames", []):
        for key in DISTORTION_KEYS:
            value = frame.get(key, tfs.get(key, 0.0)) or 0.0
            if abs(float(value)) > 0.0:
                bad.append(f"{frame.get('camera_label', '?')}:{key}={value}")
    if bad:
        raise ValueError(
            f"{camera_path} has nonzero distortion coefficients ({', '.join(bad[:6])}"
            f"{', ...' if len(bad) > 6 else ''}). Triangulation uses a pinhole model and "
            "would silently produce wrong 3D keypoints. Pass the post-undistort transforms "
            "(transforms_refined.json or transforms.json), not transforms_hloc.json.")


def load_cameras(camera_path: Path, dtype=np.float64):
    """Read a nerfstudio transforms.json into K (3x3) and world-to-camera T
    (4x4, OpenCV axes) per camera. Returns ({(camera, tem): (K, T)},
    {camera: (K, T)}); the second maps each label to its last entry."""
    with open(camera_path) as f:
        tfs = json.load(f)
    check_undistorted(tfs, camera_path)

    by_cam_tem, by_cam = {}, {}
    for frame in tfs.get("frames", []):
        cam = frame.get("camera_label")
        if cam is None:
            continue
        cam = str(cam)
        fx = frame.get("fl_x", tfs.get("fl_x"))
        fy = frame.get("fl_y", tfs.get("fl_y"))
        cx = frame.get("cx", tfs.get("cx"))
        cy = frame.get("cy", tfs.get("cy"))
        if any(v is None for v in (fx, fy, cx, cy)):
            raise ValueError(f"{camera_path}: camera {cam} is missing fl_x/fl_y/cx/cy "
                             "(per frame or global)")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype)
        c2w = np.array(frame["transform_matrix"], dtype=dtype)
        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV camera axes
        T = np.linalg.inv(c2w)
        by_cam[cam] = (K, T)
        tem = _infer_tem_label(frame, cam)
        if tem is not None:
            by_cam_tem[(cam, tem)] = (K, T)
    return by_cam_tem, by_cam


def camera_matrices(by_cam_tem, by_cam, camera_label: str, tem_label: str):
    """(K, T) for a camera at a frame: the per-frame entry if one matches,
    else the camera's static entry. Labels may differ in zero padding."""
    for cam in _label_candidates(camera_label, 2):
        for tem in _label_candidates(tem_label, 6):
            if (cam, tem) in by_cam_tem:
                return by_cam_tem[(cam, tem)]
    for cam in _label_candidates(camera_label, 2):
        if cam in by_cam:
            return by_cam[cam]
    raise KeyError(f"Camera label not found in the camera file: {camera_label}")


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def project_one_point(kp3d, Ks, Ts):
    """kp3d (3,) world -> pixel (m, 2) and depth (m,) in each of m cameras."""
    if (np.asarray(kp3d) == INVALID).any():
        return np.full((Ks.shape[0], 2), INVALID), np.full((Ks.shape[0],), INVALID)
    kp3d_h = np.concatenate([np.asarray(kp3d).ravel(), [1.0]])
    kp2d_h = (Ks @ Ts[:, :3]) @ kp3d_h
    depth = kp2d_h[:, 2]
    return kp2d_h[:, :2] / (depth[:, None] + 1e-9), depth


def triangulate_one_point(Ks, Ts, kp2d, kp2d_score, score_thr=0.6, min_views=MIN_VIEWS,
                          max_views=MAX_VIEWS):
    """One keypoint seen in m cameras: kp2d (m, 2), kp2d_score (m,).
    Returns (kp3d (3,), mean reprojection error in px, views used), or
    (None, None, views) when too few views pass the score gate."""
    if score_thr is not None:
        if max_views is not None:
            max_views = min(max_views, kp2d.shape[0])
            # also gate on the score that keeps only the best max_views views
            score_thr = max(score_thr, np.percentile(kp2d_score, 100 * (1 - max_views / kp2d.shape[0])))
        mask = kp2d_score >= score_thr
        n_views = mask.sum()
        if n_views < min_views:
            return None, None, n_views
        Ks, Ts, kp2d, kp2d_score = Ks[mask], Ts[mask], kp2d[mask], kp2d_score[mask]
    else:
        n_views = kp2d.shape[0]

    # 1. score-weighted linear (DLT) solution
    rows, weights = [], []
    for (u, v), K, T, s in zip(kp2d, Ks, Ts, kp2d_score):
        if s <= 0 or u < 0 or v < 0:
            continue
        P = K @ T[:3]
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
        weights.extend([s, s])
    if not rows:
        return None, None, 0
    A = np.diag(np.sqrt(weights)) @ np.stack(rows)
    _, _, Vt = np.linalg.svd(A)
    kp3d_lin = Vt[-1][:3] / (Vt[-1][3] + 1e-9)

    # 2. robust non-linear refinement of the reprojection error
    coord_w = np.repeat(np.sqrt(kp2d_score), 2)

    def residual(kp3d):
        pred, _ = project_one_point(kp3d, Ks, Ts)
        return (pred.reshape(-1) - kp2d.reshape(-1)) * coord_w

    kp3d = least_squares(residual, kp3d_lin, method="trf", loss="huber", f_scale=1.0, max_nfev=50).x

    # 3. score-weighted mean reprojection error
    kp2d_hat, _ = project_one_point(kp3d, Ks, Ts)
    err_px = np.linalg.norm(kp2d_hat - kp2d, axis=1)
    reproj = (err_px * kp2d_score).sum() / (kp2d_score.sum() + 1e-9)
    return kp3d, reproj, n_views


def triangulate_points(Ks, Ts, kp2d, kp2d_score, score_thr=0.6):
    """n cameras, k keypoints: Ks (n,3,3), Ts (n,4,4), kp2d (n,k,2),
    kp2d_score (n,k). Returns kp3d (k,3), reproj (k,), n_views (k,), with
    INVALID where a keypoint could not be solved."""
    n, k, _ = kp2d.shape
    for name, arr, shape in (("kp2d_score", kp2d_score, (n, k)), ("Ks", Ks, (n, 3, 3)),
                             ("Ts", Ts, (n, 4, 4))):
        if arr.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")

    kp3d = np.full((k, 3), INVALID)
    reproj = np.full((k,), INVALID)
    n_views = np.full((k,), INVALID)
    for i in range(k):
        p, r, v = triangulate_one_point(Ks, Ts, kp2d[:, i], kp2d_score[:, i], score_thr=score_thr)
        if p is not None:
            kp3d[i] = p
        if r is not None:
            reproj[i] = r
        n_views[i] = v
    return kp3d, reproj, n_views


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_kp2d(path: Path, dtype=np.float64):
    """Keypoints (k, 2) and scores (k,) of the first instance (scores are 1
    when the file has none)."""
    with open(path) as f:
        instance = json.load(f)["instance_info"][0]
    kp = np.array(instance["keypoints"], dtype=dtype)
    if "keypoint_scores" in instance:
        score = np.array(instance["keypoint_scores"], dtype=dtype)
    else:
        score = np.ones(kp.shape[0], dtype=dtype)
    # Legacy COCO-WholeBody 133 layout: weight finger scores by their hand
    # root's confidence. The 308-point goliath layout is left alone.
    if score.shape[0] == 133:
        score[92:112] *= score[91] ** 2
        score[113:133] *= score[112] ** 2
    return kp, score


def write_kp3d(path: Path, kp3d, kp3d_reproj):
    path.parent.mkdir(parents=True, exist_ok=True)
    pose = {"instance_info": [{"keypoints": kp3d.tolist(), "keypoint_reproj": kp3d_reproj.tolist()}]}
    with open(path, "w") as f:
        json.dump(pose, f, indent=4)


def write_kp3d_pcd(path: Path, kp3d):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(kp3d)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def triangulate_skeleton(camera_path: Path, kp2d_dir: Path, out_kp3d_dir: Path,
                         out_pcd_dir: Path | None = None, score_thr: float = 0.6):
    if not kp2d_dir.is_dir():
        raise FileNotFoundError(f"kp2d_dir not found: {kp2d_dir}")
    camera_labels = sorted(p.name for p in kp2d_dir.iterdir() if p.is_dir())
    if not camera_labels:
        raise FileNotFoundError(f"No per-camera keypoint directories under {kp2d_dir}")
    tem_labels = sorted(p.name.split(".")[0] for p in (kp2d_dir / camera_labels[0]).iterdir())
    print(f"Using cameras: {camera_labels}")
    print(f"Triangulating {len(tem_labels)} frame(s)")

    by_cam_tem, by_cam = load_cameras(camera_path)
    for tem in tem_labels:
        Ks, Ts = zip(*[camera_matrices(by_cam_tem, by_cam, cam, tem) for cam in camera_labels])
        kp2d, score = zip(*[read_kp2d(kp2d_dir / cam / f"{tem}.json") for cam in camera_labels])
        kp3d, reproj, _ = triangulate_points(np.stack(Ks), np.stack(Ts), np.stack(kp2d), np.stack(score),
                                             score_thr=score_thr)
        write_kp3d(out_kp3d_dir / f"{tem}.json", kp3d, reproj)
        if out_pcd_dir is not None:
            write_kp3d_pcd(out_pcd_dir / f"{tem}.ply", kp3d)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera_path", required=True, type=Path,
                        help="Undistorted nerfstudio transforms.json with camera_label per frame")
    parser.add_argument("--kp2d_dir", required=True, type=Path)
    parser.add_argument("--out_kp3d_dir", required=True, type=Path)
    parser.add_argument("--out_pcd_dir", type=Path)
    parser.add_argument("--score_thr", type=float, default=0.6,
                        help="Minimum 2D keypoint score for a view to count (default 0.6)")
    args = parser.parse_args()

    try:
        triangulate_skeleton(args.camera_path, args.kp2d_dir, args.out_kp3d_dir,
                             args.out_pcd_dir, score_thr=args.score_thr)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
