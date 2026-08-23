#!/usr/bin/env python3
"""
project_skeleton_conditioning.py

Project already-triangulated 3D keypoints into a pair sweep's novel camera poses
and write the per-frame 2D keypoint files a skeleton conditioning map is drawn
from. Runs after render_pair_sweep.py; its output feeds the video-to-video
repair pass as a structural control channel (see docs/render_and_repair.md).

Why this exists: a splat render is right about appearance and wrong about
geometry exactly where the rig had no coverage -- holes, floaters, smear. A
skeleton is the opposite: it says nothing about appearance and everything about
where the body is, from any direction, because 3D keypoints are view-independent.
Feed the render as the video model's base image and the skeleton as its control
signal, and each supplies what the other lacks. Handing the model a synthetic
body render INSTEAD of the splat render is the trap: body geometry carries no
clothing, hair or identity, so a denoise strong enough to dress it is a denoise
strong enough to invent the subject, which is the measured 18.0 -> 13.3 dB
failure.

Nothing here re-solves any geometry. triangulate_and_project_keypoints.py has
already produced poses_3d/<tem_label>.json for every frame; this projects those
same points through the sweep's cameras.

TWO THINGS THAT MAKE A SKELETON USABLE AT NOVEL VIEWS

  * Depth. Every keypoint gets its true camera-space depth in the swept view, so
    Diffuman4D's drawer sorts limbs back-to-front correctly instead of painting
    an arm that is behind the torso on top of it. Real-view pipelines often have
    no depth to give it; projecting from 3D, we always do.

  * Face fade. As the sweep swings behind the subject, face keypoints must fade
    out rather than be drawn straight through the back of the skull. Scores for
    the face keypoints are scaled by the angle between the face normal
    (triangulated from nose and eyes) and the camera's view direction, mirroring
    the same demotion in Diffuman4D's project_points.

CONFIDENCE

Scores come from each keypoint's triangulation reprojection error, which is what
poses_3d actually records (`keypoint_reproj`, a score-weighted pixel error --
LOW is good). The mapping is score = clip(1 - err / --reproj_tau, 0, 1), and
Diffuman4D's drawer then drops any link below 0.5 and fades colour up to 0.9.

--reproj_tau is in SOURCE image pixels, so its right value depends on your
capture resolution; the script prints the actual error distribution every run so
you can set it from data rather than from this default. Keypoints that failed to
triangulate are marked with Diffuman4D's INVALID sentinel and scored 0.

conda env: none for the projection (numpy only). --draw shells out to
Diffuman4D's draw_skeleton.py, which needs that submodule's own dependencies
(cv2, fire, easyvolcap) -- run this script under an env that has them, as
triangulate_and_project_keypoints.py does.

Usage:
    python3 project_skeleton_conditioning.py \\
        --sweep_dir /path/to/sweeps/03_to_07 \\
        --out_dir /path/to/skeletons/03_to_07 \\
        --kp3d_dir /path/to/window_kp/poses_3d \\
        [--reproj_tau 10.0] [--draw]

Output:
    out_dir/kp2d/<sweep>/NNNN.json   Diffuman4D-format 2D keypoints, with
                                     keypoint_depths and keypoint_scores
    out_dir/kpmap/<sweep>/NNNN.png   (with --draw) the skeleton conditioning map
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DIFFUMAN4D_ROOT = REPO_ROOT / "deps" / "Diffuman4D"
DRAW_SCRIPT = DIFFUMAN4D_ROOT / "scripts" / "preprocess" / "draw_skeleton.py"

INVALID = -1e6         # Diffuman4D's sentinel for a keypoint that failed to triangulate
INVALID_ATOL = 1.0     # anything within this of the sentinel counts as it
NOSE_KP, LEFT_EYE_KP, RIGHT_EYE_KP = 0, 1, 2  # goliath308 face keypoint indices
FACE_KP_END = 91       # keypoints [23, 91) are the face group in the 308/133 layouts
DEFAULT_REPROJ_TAU = 10.0


def load_kp3d(path: Path) -> tuple:
    """(keypoints (N,3), reprojection error (N,)) from a poses_3d JSON.

    `keypoint_reproj` is an ERROR in pixels, not a confidence -- low is good.
    triangulate_one_point's docstring calls its second return `kp3d_score`, but
    the code returns the reprojection error there, so reading it as a score
    would invert every confidence in the sweep."""
    instance = json.loads(path.read_text())["instance_info"][0]
    keypoints = np.asarray(instance["keypoints"], dtype=np.float64)
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        raise ValueError(f"{path}: expected (N, 3) 3D keypoints, got {keypoints.shape}")
    reproj = np.asarray(instance.get("keypoint_reproj", np.zeros(len(keypoints))), dtype=np.float64)
    return keypoints, reproj


def valid_mask(keypoints: np.ndarray) -> np.ndarray:
    """False wherever triangulation failed and left the INVALID sentinel."""
    return np.isfinite(keypoints).all(axis=1) & (np.abs(keypoints - INVALID) > INVALID_ATOL).all(axis=1)


def reproj_to_score(reproj: np.ndarray, valid: np.ndarray, tau: float) -> np.ndarray:
    """Per-keypoint confidence in [0, 1] from reprojection error, zero where the
    keypoint is invalid or its error was never recorded."""
    with np.errstate(invalid="ignore"):
        score = 1.0 - np.abs(reproj) / tau
    score = np.clip(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    recorded = np.isfinite(reproj) & (np.abs(reproj - INVALID) > INVALID_ATOL)
    return np.where(valid & recorded, score, 0.0)


def face_normal(keypoints: np.ndarray, valid: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    """Horizontal unit vector out of the face, or None when the nose and both
    eyes did not all triangulate.

    Construction: the nose tip sits forward of the eye midpoint, so
    `nose - eye_mid` with the vertical component projected out is the facing
    direction, no handedness to get wrong. This is exactly what
    this project has always used for a facing direction, and it is deliberately NOT
    Diffuman4D's `get_face_normal`, which takes cross(eye line, nose offset).
    That cross product is perpendicular to both inputs, so it is dominated by
    the head's up axis and only acquires a forward component from how far the
    nose sits BELOW the eyes; it also flips sign with the left/right eye
    convention. Nothing is lost by departing from it: their own preprocessing
    calls project_points with kp3d_score=None, which skips the face demotion
    altogether, so there is no conditioning behaviour here to stay consistent
    with."""
    needed = [NOSE_KP, LEFT_EYE_KP, RIGHT_EYE_KP]
    if len(keypoints) <= RIGHT_EYE_KP or not valid[needed].all():
        return None
    eye_mid = (keypoints[LEFT_EYE_KP] + keypoints[RIGHT_EYE_KP]) / 2.0
    forward = keypoints[NOSE_KP] - eye_mid
    forward = forward - (forward @ up) * up
    norm = np.linalg.norm(forward)
    return forward / norm if norm > 1e-9 else None


def world_up(meta: dict, w2c: np.ndarray) -> np.ndarray:
    """The sweep's world up axis: the exact one render_pair_sweep recorded, or an
    approximation from the camera for sweeps written before it was.

    The fallback is only an approximation, and knowing why matters. lookat_w2c
    orthogonalizes its down vector against forward, so an elevated camera's
    image-down axis is tilted off world up by the elevation angle -- around 7
    degrees of error at a typical rig elevation. That tilt leaks head height into
    a measurement meant to capture head yaw, so prefer the recorded value."""
    recorded = meta.get("up")
    up = np.asarray(recorded, dtype=np.float64) if recorded else -w2c[1, :3]
    norm = np.linalg.norm(up)
    return up / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])


def fade_face_scores(scores: np.ndarray, normal: np.ndarray | None, w2c: np.ndarray) -> np.ndarray:
    """Scale face keypoint scores by how squarely the camera faces the face.

    A skeleton projected into a novel view has no idea the head is opaque, so
    without this the sweep draws eyes and nose through the back of the skull the
    moment it passes behind the subject. The factor is (1 + cos)/2 against the
    camera's view direction: 1 head-on, 0 from directly behind. The shape is
    Diffuman4D's, from the demotion in its project_points; the facing vector it
    is applied to is this repo's (see face_normal)."""
    if normal is None or len(scores) <= NOSE_KP:
        return scores
    view_direction = w2c[2, :3]  # camera +Z in world coordinates, i.e. where it looks
    facing = float(-np.dot(view_direction, normal)) * 0.5 + 0.5

    faded = scores.copy()
    faded[:min(3, len(scores))] *= facing
    if len(scores) > 23:
        faded[23:min(FACE_KP_END, len(scores))] *= facing
    return faded


def project(keypoints: np.ndarray, intrinsics: np.ndarray, w2c: np.ndarray) -> tuple:
    """(uv (N,2), depth (N,)) of world keypoints in one camera. Points at or
    behind the image plane come back at the INVALID sentinel, which is what the
    drawer treats as unusable."""
    homogeneous = np.concatenate([keypoints, np.ones((len(keypoints), 1))], axis=1)
    uvw = (intrinsics @ w2c[:3, :] @ homogeneous.T).T
    depth = uvw[:, 2]
    in_front = depth > 1e-6
    uv = np.full((len(keypoints), 2), INVALID)
    uv[in_front] = uvw[in_front, :2] / depth[in_front, None]
    return uv, np.where(in_front, depth, INVALID)


def write_kp2d(path: Path, uv: np.ndarray, depth: np.ndarray, score: np.ndarray) -> None:
    """Diffuman4D's exact kp2d format (see its triangulate_skeleton.write_kp2d),
    so draw_skeleton.py consumes these unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"instance_info": [{
        "keypoints": uv.tolist(),
        "keypoint_depths": depth.tolist(),
        "keypoint_scores": score.tolist(),
    }]}, indent=1))


def resolve_kp3d(kp3d_dir: Path, frame: int | None, index: int) -> Path | None:
    """That swept frame's 3D keypoints from a flat directory.

    Keyed by CAPTURE frame number rather than by position in the clip, because a
    sweep can start anywhere in a take: frame 58 is `000058.json` whether it is
    the clip's first frame or its tenth. Falls back to the sweep index only when
    the sweep records no capture numbering (a single-instant rig)."""
    keys = [frame, index] if frame is not None else [index]
    for key in keys:
        candidate = kp3d_dir / f"{key:06d}.json"
        if candidate.exists():
            return candidate
    return None


def project_sweep(sweep_dir: Path, out_dir: Path, *, kp3d_dir: Path,
                  reproj_tau: float = DEFAULT_REPROJ_TAU) -> dict:
    """Project every swept frame's keypoints. Returns a summary of what it wrote."""
    cameras_path = sweep_dir / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"{cameras_path} not found -- run render_pair_sweep.py first")
    meta = json.loads(cameras_path.read_text())

    intrinsics = np.array([[meta["fl_x"], 0.0, meta["cx"]],
                           [0.0, meta["fl_y"], meta["cy"]],
                           [0.0, 0.0, 1.0]])
    kp2d_root = out_dir / "kp2d" / sweep_dir.name

    written, missing, errors, faded_frames = 0, [], [], 0
    for record in meta["frames"]:
        index = record["idx"]
        kp3d_path = resolve_kp3d(kp3d_dir, record.get("frame"), index)
        if kp3d_path is None:
            missing.append(index)
            continue

        keypoints, reproj = load_kp3d(kp3d_path)
        valid = valid_mask(keypoints)
        w2c = np.array(record["w2c"])

        uv, depth = project(keypoints, intrinsics, w2c)
        score = reproj_to_score(reproj, valid, reproj_tau)
        normal = face_normal(keypoints, valid, world_up(meta, w2c))
        score = fade_face_scores(score, normal, w2c)
        score[~valid] = 0.0
        score[depth <= INVALID + INVALID_ATOL] = 0.0
        faded_frames += normal is not None

        write_kp2d(kp2d_root / f"{index:04d}.json", uv, depth, score)
        errors.extend(reproj[valid & np.isfinite(reproj)].tolist())
        written += 1

    if written == 0:
        raise FileNotFoundError(
            f"no 3D keypoints found for any frame of {sweep_dir.name}. Looked in {kp3d_dir} for "
            "<capture frame>.json -- run window_keypoints_3d.py over this frame range first")

    summary = {"sweep": sweep_dir.name, "frames": written, "kp2d_dir": str(kp2d_root),
               "res": meta["res"], "missing_frames": missing,
               "faces_oriented": faded_frames, "reproj_tau": reproj_tau}
    if errors:
        percentiles = np.percentile(errors, [50, 90, 99])
        summary["reproj_px"] = {"median": float(percentiles[0]), "p90": float(percentiles[1]),
                                "p99": float(percentiles[2])}
    return summary


def draw_maps(out_dir: Path, sweep_name: str, res: int, image_ext: str = ".png") -> int:
    """Render the conditioning maps with Diffuman4D's own drawer.

    Wrapped rather than reimplemented so the goliath308 link topology, per-joint
    palette and depth sorting match the maps that model was conditioned on --
    a repainted skeleton in different colours is a different control signal."""
    if not DRAW_SCRIPT.is_file():
        print(f"ERROR: {DRAW_SCRIPT} not found -- is the Diffuman4D submodule checked out?",
              file=sys.stderr)
        return 1
    # --spa_labels is deliberately NOT passed: draw_skeleton formats it as
    # f"{label:02d}", which assumes numeric camera labels and dies with "Unknown
    # format code 'd' for object of type 'str'" on a sweep directory name.
    # Omitting it makes the drawer list the directory instead, which is what we
    # want anyway since kp2d holds exactly this one sweep.
    command = [
        sys.executable, str(DRAW_SCRIPT),
        "--kp2d_dir", str(out_dir / "kp2d"),
        "--out_kpmap_dir", str(out_dir / "kpmap"),
        f"--kp2d_canvas_shape=({res},{res})",
        f"--out_kpmap_shape=({res},{res})",
        "--image_ext", image_ext,
    ]
    print("Running:", " ".join(command))
    return subprocess.run(command, cwd=str(DIFFUMAN4D_ROOT)).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True, type=Path,
                    help="render_pair_sweep.py output directory, holding cameras.json")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="destination for kp2d/ and, with --draw, kpmap/")
    ap.add_argument("--kp3d_dir", required=True, type=Path,
                    help="directory of <capture frame>.json triangulated keypoints, as "
                         "window_keypoints_3d.py writes")
    ap.add_argument("--reproj_tau", type=float, default=DEFAULT_REPROJ_TAU,
                    help="reprojection error, in SOURCE image pixels, at which a keypoint's "
                         "confidence reaches zero. The printed error distribution is the right "
                         "basis for this; the drawer drops links scoring below 0.5")
    ap.add_argument("--draw", action="store_true",
                    help="also render the conditioning maps via Diffuman4D's draw_skeleton.py "
                         "(needs that submodule's deps: cv2, fire, easyvolcap)")
    ap.add_argument("--image_ext", default=".png", help="conditioning map extension for --draw")
    args = ap.parse_args()

    try:
        summary = project_sweep(args.sweep_dir, args.out_dir, kp3d_dir=args.kp3d_dir,
                                reproj_tau=args.reproj_tau)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"{summary['sweep']}: projected {summary['frames']} frames -> {summary['kp2d_dir']}")
    if summary.get("reproj_px"):
        stats = summary["reproj_px"]
        print(f"  reprojection error px: median {stats['median']:.2f}, p90 {stats['p90']:.2f}, "
              f"p99 {stats['p99']:.2f} (--reproj_tau {summary['reproj_tau']:.1f})")
        if stats["median"] >= summary["reproj_tau"]:
            print("  WARNING: the MEDIAN error is at or past --reproj_tau, so most keypoints score "
                  "0 and the drawer will emit near-empty maps. Raise --reproj_tau", file=sys.stderr)
    if summary["faces_oriented"] < summary["frames"]:
        print(f"  note: face orientation resolved on {summary['faces_oriented']}/{summary['frames']} "
              "frames; the rest keep un-faded face scores and may draw the face from behind")
    if summary["missing_frames"]:
        shown = summary["missing_frames"][:8]
        print(f"  WARNING: no keypoints for {len(summary['missing_frames'])} frame(s): {shown}"
              f"{'...' if len(summary['missing_frames']) > 8 else ''}", file=sys.stderr)

    if args.draw:
        return draw_maps(args.out_dir, args.sweep_dir.name, summary["res"], args.image_ext)
    print(f"  to draw the maps: rerun with --draw (or call {DRAW_SCRIPT.name} on kp2d/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
