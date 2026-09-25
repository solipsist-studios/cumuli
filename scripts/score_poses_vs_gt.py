#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
score_poses_vs_gt.py - measure estimated camera poses against known truth.

A rendered rig is the only case where the true camera poses are known
exactly, which makes it the only place the pose chain can be scored rather
than merely inspected. This compares what run_hloc.py and
run_pose_refinement.py recovered against what Blender actually rendered
from, and reports the error in metres and degrees.

Structure-from-motion recovers geometry up to a similarity: the whole
reconstruction may be rotated, translated, and uniformly scaled without
changing a single image. So the comparison first fits the best similarity
transform between the estimated camera centres and the true ones, then
reports what is left. The fitted scale is itself worth reading: it converts
the reconstruction's arbitrary units into metres, and on a real capture
that number has to be guessed.

Reported per camera and in summary:

  position_error_m     distance between true and aligned estimated centre
  rotation_error_deg   angle of the residual rotation
  scale                one number for the run: metres per reconstruction unit
  joint_reprojection   ground-truth armature joints projected through the
                       estimated cameras, in pixels. This is the error that
                       actually matters to a splat, because it is measured
                       where the subject is rather than out on the walls
                       where background features happen to lie.

With --write_aligned the estimated rig is written back in the ground-truth
world frame, so a splat trained on it is metric and can be scored against
the same eval cameras as the ground-truth run.

Usage:
    python3 scripts/score_poses_vs_gt.py \\
        --estimated <run>/localize/hloc_final/transforms_multiframe.json \\
        --refined <run>/localize/transforms_refined.json \\
        --ground_truth <run>/rig_gt_transforms.json \\
        --report_json <run>/pose_scores.json \\
        --align_source <run>/localize/transforms_refined.json \\
        --write_aligned <run>/localize/transforms_aligned.json
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from refine_poses_with_keypoints import similarity_align  # noqa: E402


def trailing_id(label):
    m = re.search(r"(\d+)$", str(label))
    return int(m.group(1)) if m else None


def match_labels(estimated_labels, gt_labels):
    """Pair estimated camera labels with ground-truth ones.

    HLOC labels a camera by its folder ("Camera_00") while the rendered rig
    labels it "00", so an exact match is the exception. Matching on the
    trailing number is what the rest of the pipeline does; this refuses an
    ambiguous match rather than guessing."""
    pairs = {}
    gt_by_id = {}
    for label in gt_labels:
        gt_by_id.setdefault(trailing_id(label), []).append(label)
    for est in estimated_labels:
        if est in gt_labels:
            pairs[est] = est
            continue
        candidates = gt_by_id.get(trailing_id(est), [])
        if len(candidates) == 1:
            pairs[est] = candidates[0]
    return pairs


def load_transform_map(path):
    """{camera_label: 4x4 OpenGL camera-to-world} from a transforms file."""
    data = json.loads(Path(path).read_text())
    frames = data.get("frames") or data.get("cameras") or []
    out = {}
    for fr in frames:
        label = str(fr.get("camera_label") or fr.get("label"))
        if label in out:
            continue                      # multi-timestamp files repeat cameras
        out[label] = np.asarray(fr["transform_matrix"], dtype=np.float64)
    if not out:
        sys.exit(f"ERROR: {path} contains no camera entries")
    return data, out


def load_ground_truth(path):
    data = json.loads(Path(path).read_text())
    cams = {c["label"]: c for c in data["cameras"] if c.get("role") == "train"}
    if not cams:
        sys.exit(f"ERROR: {path} lists no training cameras")
    poses = {k: np.asarray(v["transform_matrix"], dtype=np.float64)
             for k, v in cams.items()}
    return data, cams, poses


def rotation_angle_deg(R_a, R_b):
    """Angle of the rotation taking R_a to R_b.

    Uses atan2 of the antisymmetric part against the trace rather than
    acos of the trace alone. Near zero, acos is ill-conditioned: an error
    of eps in its argument becomes sqrt(2*eps) in the angle, so a perfect
    match reports a few microdegrees of noise. Sub-degree pose error is
    precisely the regime this scorer reports on."""
    R = np.asarray(R_a).T @ np.asarray(R_b)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / 2.0
    sin_theta = float(np.linalg.norm(axis))
    cos_theta = (float(np.trace(R)) - 1.0) / 2.0
    return math.degrees(math.atan2(sin_theta, cos_theta))


def align_and_score(est_poses, gt_poses, pairs):
    """Fit the similarity from estimated to true, then score the residual."""
    est_labels = sorted(pairs)
    if len(est_labels) < 3:
        sys.exit(f"ERROR: only {len(est_labels)} camera(s) matched between the "
                 "estimate and the ground truth; a similarity fit needs 3")
    src = np.array([est_poses[e][:3, 3] for e in est_labels])
    dst = np.array([gt_poses[pairs[e]][:3, 3] for e in est_labels])
    scale, R, t = similarity_align(src, dst)

    rows = []
    for est_label in est_labels:
        gt_label = pairs[est_label]
        est = est_poses[est_label]
        centre = scale * R @ est[:3, 3] + t
        # A similarity's rotation applies to the camera basis unchanged: the
        # uniform scale does not tilt any axis.
        basis = R @ est[:3, :3]
        rows.append({
            "estimated_label": est_label,
            "camera_label": gt_label,
            "position_error_m": float(np.linalg.norm(
                centre - gt_poses[gt_label][:3, 3])),
            "rotation_error_deg": float(rotation_angle_deg(
                basis, gt_poses[gt_label][:3, :3])),
        })
    return {"scale": float(scale), "rotation": R, "translation": t,
            "cameras": rows}


def project_opengl(c2w, K, points):
    """World points through an OpenGL camera-to-world into pixels."""
    colmap = np.asarray(c2w, dtype=np.float64).copy()
    colmap[:3, 1:3] *= -1.0                  # OpenGL -> OpenCV camera axes
    w2c = np.linalg.inv(colmap)
    cam = (w2c[:3, :3] @ points.T + w2c[:3, 3:4]).T
    z = cam[:, 2]
    front = z > 1e-6
    zs = np.where(front, z, 1.0)
    u = K[0, 0] * cam[:, 0] / zs + K[0, 2]
    v = K[1, 1] * cam[:, 1] / zs + K[1, 2]
    return u, v, front


def blender_to_opengl_points(points):
    p = np.asarray(points, dtype=np.float64)
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1)


def joint_reprojection(alignment, est_poses, gt_cams, pairs, joints_path):
    """Reprojection error of known 3D joints, in pixels.

    Camera-centre error alone understates what a splat suffers: a small
    rotation about a distant camera moves the subject a long way in the
    image. Projecting points that sit ON the subject measures the error
    where it costs something."""
    if not joints_path or not Path(joints_path).is_file():
        return None
    payload = json.loads(Path(joints_path).read_text())
    heads = [j["head"] for j in payload["joints"].values()]
    if not heads:
        return None
    points = blender_to_opengl_points(heads)

    scale, R, t = alignment["scale"], alignment["rotation"], alignment["translation"]
    rows = []
    for est_label, gt_label in sorted(pairs.items()):
        gt = gt_cams[gt_label]
        K = np.array([[gt["fl_x"], 0.0, gt["cx"]],
                      [0.0, gt["fl_y"], gt["cy"]],
                      [0.0, 0.0, 1.0]])
        aligned = np.eye(4)
        aligned[:3, :3] = R @ est_poses[est_label][:3, :3]
        aligned[:3, 3] = scale * R @ est_poses[est_label][:3, 3] + t

        u_gt, v_gt, ok_gt = project_opengl(gt["transform_matrix"], K, points)
        u_es, v_es, ok_es = project_opengl(aligned, K, points)
        ok = ok_gt & ok_es & (u_gt >= 0) & (u_gt < gt["w"]) & \
            (v_gt >= 0) & (v_gt < gt["h"])
        if not ok.any():
            continue
        err = np.hypot(u_es[ok] - u_gt[ok], v_es[ok] - v_gt[ok])
        rows.append({"camera_label": gt_label,
                     "points": int(ok.sum()),
                     "median_px": float(np.median(err)),
                     "max_px": float(err.max())})
    if not rows:
        return None
    return {"per_camera": rows,
            "median_px": float(np.median([r["median_px"] for r in rows])),
            "max_px": float(max(r["max_px"] for r in rows))}


def summarise(name, scored, joints):
    pos = [c["position_error_m"] for c in scored["cameras"]]
    rot = [c["rotation_error_deg"] for c in scored["cameras"]]
    out = {
        "source": name,
        "cameras_matched": len(scored["cameras"]),
        "scale_metres_per_unit": scored["scale"],
        "position_error_m": {"median": float(np.median(pos)),
                             "max": float(np.max(pos))},
        "rotation_error_deg": {"median": float(np.median(rot)),
                               "max": float(np.max(rot))},
        "per_camera": scored["cameras"],
    }
    if joints:
        out["joint_reprojection_px"] = {"median": joints["median_px"],
                                        "max": joints["max_px"],
                                        "per_camera": joints["per_camera"]}
    return out


def write_aligned(align_path, alignment, pairs, gt_cams, out_path):
    """The estimated rig expressed in the ground-truth world frame.

    Camera labels are rewritten to the ground-truth ones, because every
    stage downstream looks a camera up by that literal string."""
    _data, poses = load_transform_map(align_path)
    scale, R, t = alignment["scale"], alignment["rotation"], alignment["translation"]
    frames = []
    for est_label, gt_label in sorted(pairs.items(), key=lambda kv: kv[1]):
        gt = gt_cams[gt_label]
        aligned = np.eye(4)
        aligned[:3, :3] = R @ poses[est_label][:3, :3]
        aligned[:3, 3] = scale * R @ poses[est_label][:3, 3] + t
        frames.append({
            "camera_label": gt_label,
            "file_path": f"images_flat/{gt_label}.png",
            "transform_matrix": aligned.tolist(),
            "fl_x": gt["fl_x"], "fl_y": gt["fl_y"],
            "cx": gt["cx"], "cy": gt["cy"],
            "w": gt["w"], "h": gt["h"],
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {"camera_model": "OPENCV", "frames": frames}, indent=1))
    return len(frames)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimated", required=True,
                    help="transforms from the pose solve (run_hloc.py)")
    ap.add_argument("--refined", default=None,
                    help="transforms after keypoint refinement, scored too")
    ap.add_argument("--ground_truth", required=True,
                    help="rig_gt_transforms.json from the render step")
    ap.add_argument("--joints", default=None,
                    help="joints/frame_NNNN.json for the reprojection check "
                         "(defaults to the render's middle frame)")
    ap.add_argument("--report_json", default=None)
    ap.add_argument("--align_source", default=None,
                    help="which transforms --write_aligned transforms "
                         "(default: --refined when given, else --estimated)")
    ap.add_argument("--write_aligned", default=None)
    args = ap.parse_args()

    _gt_data, gt_cams, gt_poses = load_ground_truth(args.ground_truth)

    joints_path = args.joints
    if joints_path is None:
        render_dir = Path(args.ground_truth).parent / "render" / "joints"
        if render_dir.is_dir():
            files = sorted(render_dir.glob("frame_*.json"))
            if files:
                joints_path = files[len(files) // 2]

    report = {"ground_truth": str(args.ground_truth), "sources": []}
    alignments = {}
    for name, path in (("hloc", args.estimated), ("refined", args.refined)):
        if not path or not Path(path).is_file():
            continue
        _, est_poses = load_transform_map(path)
        pairs = match_labels(list(est_poses), list(gt_poses))
        unmatched = sorted(set(est_poses) - set(pairs))
        if unmatched:
            print(f"  WARNING: {len(unmatched)} estimated camera(s) did not "
                  f"match a ground-truth label: {unmatched[:5]}")
        scored = align_and_score(est_poses, gt_poses, pairs)
        joints = joint_reprojection(scored, est_poses, gt_cams, pairs, joints_path)
        report["sources"].append(summarise(name, scored, joints))
        alignments[name] = (scored, pairs, path)

        summary = report["sources"][-1]
        print(f"{name}: {summary['cameras_matched']} cameras | "
              f"position median {summary['position_error_m']['median'] * 1000:.1f} mm "
              f"max {summary['position_error_m']['max'] * 1000:.1f} mm | "
              f"rotation median {summary['rotation_error_deg']['median']:.3f} deg "
              f"max {summary['rotation_error_deg']['max']:.3f} deg | "
              f"scale {summary['scale_metres_per_unit']:.4f} m/unit")
        if joints:
            print(f"        subject joints reproject at "
                  f"{joints['median_px']:.2f} px median, "
                  f"{joints['max_px']:.2f} px max")

    if not report["sources"]:
        sys.exit("ERROR: neither --estimated nor --refined could be read")

    if args.write_aligned:
        # The similarity used to move the rig must be the one fitted to the
        # rig being moved. Pick by matching the requested source against the
        # files already scored, rather than combining one solve's transform
        # with another's alignment.
        pick = None
        if args.align_source:
            wanted = Path(args.align_source).resolve()
            for name, (_scored, _pairs, path) in alignments.items():
                if Path(path).resolve() == wanted:
                    pick = name
                    break
            if pick is None:
                sys.exit(
                    f"ERROR: --align_source {args.align_source} is neither "
                    "--estimated nor --refined, so no similarity was fitted "
                    "for it. Pass one of those two.")
        else:
            pick = "refined" if "refined" in alignments else "hloc"

        scored, pairs, source_path = alignments[pick]
        n = write_aligned(source_path, scored, pairs, gt_cams, args.write_aligned)
        print(f"  wrote {n} aligned camera(s) to {args.write_aligned} "
              f"(from the {pick} solve, in the ground-truth world frame)")
        report["aligned_from"] = pick

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"  wrote {args.report_json}")


if __name__ == "__main__":
    main()
