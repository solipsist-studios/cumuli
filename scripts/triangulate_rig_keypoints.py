# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Triangulate the rig's goliath308 keypoints into rig world, every frame.

WHY
---
Aligning 4DAnyone's generated ring to the rig with ONE global similarity leaves
the two sources disagreeing about the subject's position by 10.9 cm on average
and up to 37.9 cm, which costs the hybrid 6.2 dB against a real-only control.
Fixing that needs a per-frame transform, and a per-frame similarity needs at
least three correspondences per frame -- more than the single mask centroid the
diagnostic used.

The rig already has what is required: Sapiens goliath308 predictions for all 12
cameras across all 146 frames, at the 4096x3584 undistorted resolution. This
turns them into 3D.

GEOMETRY
--------
The 2D predictions live on the full undistorted frame while the dataset's
intrinsics are crop-local, so the full-frame principal point is recovered from
the SfM intrinsics scaled 5312 -> 4096. Poses come from the dataset unchanged
(cropping moves the principal point, never the camera).

Each joint is triangulated by the standard linear (DLT) method over every
camera that scores it above --min_score, weighted by that score, then refined
by rejecting cameras whose reprojection error exceeds --max_reproj_px and
re-solving. Joints seen by fewer than --min_views cameras are written as the
INVALID sentinel that Diffuman4D's tooling already uses, so downstream code
that filters on it keeps working.

See also triangulate_and_project_keypoints.py, an older wrapper around
Diffuman4D's own triangulate_skeleton.py (conda env "queen"). Kept side by
side deliberately, not merged: that one's ring-generation-facing output
was unused until now, and pulling in its heavier Diffuman4D/easyvolcap
dependency here for what fit_rig_motion.py needs would be a worse trade
than this self-contained DLT implementation.

conda env: cumuli.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INVALID = -1e6
UND_W = 4096
SFM_W = 5312


def camera_matrices(dataset_frames, undistorted, cameras):
    """Projection matrix per camera, on the full undistorted grid."""
    scale = UND_W / SFM_W
    out = {}
    for cam in cameras:
        entry = next(f for f in dataset_frames if f["file_path"].startswith(f"{cam}/"))
        label = f"{int(cam[3:]):04d}"
        full = undistorted[label]
        K = np.array([[entry["fl_x"], 0, full["cx"] * scale],
                      [0, entry["fl_y"], full["cy"] * scale],
                      [0, 0, 1.0]])
        c2w = np.asarray(entry["transform_matrix"], dtype=np.float64)
        # nerfstudio/OpenGL c2w -> OpenCV world-to-camera
        c2w = c2w.copy()
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        out[cam] = (K @ w2c[:3, :], label)
    return out


def triangulate(points, projections, weights):
    """Weighted DLT: each view contributes two rows of the null-space system."""
    rows = []
    for (u, v), P, w in zip(points, projections, weights):
        rows.append(w * (u * P[2] - P[0]))
        rows.append(w * (v * P[2] - P[1]))
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def reproject(X, P):
    x = P @ np.append(X, 1.0)
    return x[:2] / x[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transforms", required=True, type=Path,
                    help="omg4_full4d/transforms_train.json (poses + per-camera focal)")
    ap.add_argument("--undistorted_transforms", required=True, type=Path)
    ap.add_argument("--kp2d_root", required=True, type=Path,
                    help="full4d_kp2d/<camera_id>/<camera_id>_predictions.json")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--min_score", type=float, default=0.3)
    ap.add_argument("--min_views", type=int, default=3)
    ap.add_argument("--max_reproj_px", type=float, default=25.0)
    args = ap.parse_args()

    dataset = json.loads(args.transforms.read_text())["frames"]
    undistorted = {f["camera_label"]: f
                   for f in json.loads(args.undistorted_transforms.read_text())["frames"]}
    cameras = sorted({f["file_path"].split("/")[0] for f in dataset})
    mats = camera_matrices(dataset, undistorted, cameras)

    preds = {}
    for cam in cameras:
        _, label = mats[cam]
        path = args.kp2d_root / label / f"{label}_predictions.json"
        preds[cam] = {f["image_name"]: f for f in json.loads(path.read_text())["frames"]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_names = sorted({n for cam in cameras for n in preds[cam]})
    n_joints = 308
    stats = []

    for name in frame_names:
        capture_frame = int(name.lstrip("f").split(".")[0])
        obs = []
        for cam in cameras:
            frame = preds[cam].get(name)
            if not frame or not frame["instances"]:
                continue
            inst = frame["instances"][0]
            obs.append((np.asarray(inst["keypoints"], dtype=np.float64),
                        np.asarray(inst["keypoint_scores"], dtype=np.float64),
                        mats[cam][0]))
        if len(obs) < args.min_views:
            continue

        points = np.full((n_joints, 3), INVALID)
        errors = []
        for j in range(n_joints):
            uv, w, P = [], [], []
            for kp, sc, proj in obs:
                if sc[j] >= args.min_score:
                    uv.append(kp[j])
                    w.append(sc[j])
                    P.append(proj)
            if len(uv) < args.min_views:
                continue
            X = triangulate(uv, P, w)
            # Drop the views this solution does not explain, then re-solve.
            keep = [i for i in range(len(uv))
                    if np.linalg.norm(reproject(X, P[i]) - uv[i]) <= args.max_reproj_px]
            if len(keep) >= args.min_views and len(keep) < len(uv):
                X = triangulate([uv[i] for i in keep], [P[i] for i in keep],
                                [w[i] for i in keep])
            elif len(keep) < args.min_views:
                continue
            points[j] = X
            errors.append(np.mean([np.linalg.norm(reproject(X, P[i]) - uv[i]) for i in keep]))

        valid = int((points[:, 0] != INVALID).sum())
        stats.append((valid, np.mean(errors) if errors else np.nan))
        (args.out_dir / f"{capture_frame:06d}.json").write_text(json.dumps(
            {"instance_info": [{"keypoints": points.tolist()}]}))

    valid = np.asarray([s[0] for s in stats])
    err = np.asarray([s[1] for s in stats])
    print(f"wrote {len(stats)} frames to {args.out_dir}")
    print(f"valid joints per frame: mean {valid.mean():.0f} of {n_joints} "
          f"(min {valid.min()}, max {valid.max()})")
    print(f"reprojection error: mean {np.nanmean(err):.2f} px, max {np.nanmax(err):.2f} px "
          f"on a {UND_W}px frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
