#!/usr/bin/env python3
"""
render_orbit_views.py

Render a ring of synthetic novel views around the subject from one frame's
trained splat -- the "render" half of render-and-repair (see
docs/render_and_repair.md). Runs after filter_splat_by_masks.py, whose
cleaned .ply it consumes; its output feeds klein_repair_views.py.

Why this exists: a 12-camera rig covers barely half the azimuth circle, so
a 4D model trained on real views alone has no supervision behind or beside
the subject and fills those directions with floaters and smear. Rendering
the (already good) per-frame splat from novel directions, repairing each
render with a generative pass, and training on real + repaired views gives
the 4D fit supervision everywhere. This script produces the un-repaired
renders.

Two passes per frame:

  * BODY (--elev_rows x --row_azimuths, default 3x12 = 36): a full 360
    degree azimuth sweep per elevation row, DENSIFIED RELATIVE TO THE REAL
    RIG. Real camera azimuths are clustered (this rig is 6 stereo pairs,
    each pair 0.3-0.8 degrees apart), so a uniform sweep wastes samples on
    near-duplicates and a sweep spanning only the rig's arc leaves most
    poses on the subject's side/back. Instead: merge real azimuths within
    --merge_deg into effective anchors, then subdivide each gap between
    anchors. This biases samples toward wherever the rig is dense (in
    practice, wherever the subject faces) while still covering the circle.
    Inserted samples may not walk more than --max_reach_deg from their
    anchor: a sample at the exact middle of a wide gap sits at the point
    farthest from any real camera in either direction -- the worst-supported
    position possible -- and renders visibly incoherent. Wide gaps get less
    coverage rather than one maximally-bad sample.

  * HEAD (--head_views, default 8): physically zoomed cameras (--head_zoom,
    default 11.0 vs the body's 1.6) aimed at the head, rendered NATIVE at
    the same --res, NOT cropped from a body render -- cropping an already
    1536px image adds no information. This mirrors into the synthetic domain
    the real head-crop views that build_densification_crops.py adds, so the
    repair pass has genuine spatial detail on the face to work with. The arc
    is centered on the subject's ACTUAL facing direction for this frame,
    triangulated in 3D from nose/eye keypoints across every camera with a
    confident detection (--kp2d_dir); the rig's clusters often contain no
    genuinely head-on camera, so picking the "most frontal" single real
    camera still yields a 3/4 view. Falls back to the rig arc's midpoint
    when fewer than two cameras resolve the face.

NOTE: head views are useful for per-frame supervision but have been measured
to POISON shared-window 4D refits on backlit captures -- see
docs/render_and_repair.md. build_refit_dataset.py --no_head_views excludes
them at dataset-build time; render them here regardless, the cost is small.

conda env: diffuman4d (gsplat + torch + numpy + PIL + scipy + plyfile).
Any env with gsplat works; cv2 is deliberately not used.

Usage:
    python3 render_orbit_views.py \\
        --splat_ply /path/to/frame_0000_30000_maskfilt.ply \\
        --transforms /path/to/transforms.json \\
        --out_dir /path/to/orbit/frame_0000 \\
        [--kp2d_dir /path/to/poses_2d --tem_label 000000] \\
        [--subject_anchor_ply /path/to/poses_pcd_fullres/000000.ply \\
         --subject_radius 3.0] \\
        [--res 1536] [--zoom 1.6] [--elev_rows 3] [--row_azimuths 12] \\
        [--head_views 8] [--head_zoom 11.0] [--no_head_views]

Output:
    out_dir/body_NNN.png + head_NNN.png (RGBA, --res square, alpha from the
    rasterized coverage) and cameras.json recording every pose's w2c matrix
    and the shared intrinsics for each pass -- build_refit_dataset.py reads
    that file to place the repaired views in the refit dataset.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

from clean_masks import find_keypoints_json

NOSE_KP, LEFT_EYE_KP, RIGHT_EYE_KP = 0, 1, 2  # goliath-308 face keypoint indices
FACE_SCORE_THRESHOLD = 0.3   # per-keypoint Sapiens confidence needed to use a camera for triangulation
HEAD_ANCHOR_PERCENTILE = 85  # head = splats in the top 15% of subject extent along the up axis
HIGH_OPACITY = 0.5           # opacity above which a splat counts toward the subject centroid
MIN_CENTROID_SPLATS = 100    # below this many high-opacity splats, fall back to the full cloud
MIN_HEAD_SPLATS = 10         # below this many head splats, fall back to a fixed offset up the axis
DEFAULT_HEAD_OFFSET = 0.6    # world units up from the centroid when the head cannot be located


def opengl_c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """nerfstudio/OpenGL c2w -> COLMAP-convention w2c, matching the conversion
    documented in build_colmap_sparse.py (negate Y/Z axis columns, invert)."""
    m = c2w.copy()
    m[:3, 1:3] *= -1
    return np.linalg.inv(m)


def lookat_w2c(eye: np.ndarray, target: np.ndarray, down: np.ndarray) -> np.ndarray:
    """World-to-camera matrix for a camera at `eye` looking at `target`, with
    `down` giving the +Y (image-down) direction to orthogonalize against."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    dn = down - (down @ forward) * forward
    dn = dn / np.linalg.norm(dn)
    right = np.cross(dn, forward)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, dn, forward, eye
    return np.linalg.inv(c2w)


def triangulate_dlt(projections: list, uvs: list) -> np.ndarray:
    """Direct linear transform triangulation of one 3D point from its 2D
    observation in 2+ cameras with known 3x4 projection matrices."""
    rows = []
    for proj, (u, v) in zip(projections, uvs):
        rows.append(u * proj[2] - proj[0])
        rows.append(v * proj[2] - proj[1])
    _, _, vt = np.linalg.svd(np.stack(rows, axis=0))
    homogeneous = vt[-1]
    return homogeneous[:3] / homogeneous[3]


def densified_azimuths(real_azimuths: np.ndarray, count: int,
                       merge_deg: float, max_reach_deg: float) -> list:
    """Azimuth samples covering the full circle, densified toward the real
    cameras. See the module docstring for why this beats a uniform sweep.

    Real azimuths within `merge_deg` of each other collapse to one averaged
    anchor (a stereo pair's near-zero internal gap would otherwise subdivide
    into visual duplicates); each gap between anchors is then subdivided into
    `count / n_anchors` steps, and no inserted sample walks more than
    `max_reach_deg` past its anchor."""
    ordered = np.sort(np.asarray(real_azimuths, dtype=np.float64))
    clusters = [[ordered[0]]]
    for az in ordered[1:]:
        if az - clusters[-1][-1] < merge_deg:
            clusters[-1].append(az)
        else:
            clusters.append([az])
    # wraparound: the last cluster may be within merge_deg of the first across the 360 seam
    if len(clusters) > 1 and (clusters[0][0] + 360.0 - clusters[-1][-1]) < merge_deg:
        clusters[0] = clusters.pop() + clusters[0]

    anchors = np.sort([float(np.mean(c)) for c in clusters])
    per_anchor = max(1, round(count / len(anchors)))
    gaps = np.diff(np.concatenate([anchors, [anchors[0] + 360.0]]))
    return [float(anchor + min(gap * k / per_anchor, max_reach_deg))
            for anchor, gap in zip(anchors, gaps)
            for k in range(per_anchor)]


def load_rig(transforms_path: Path):
    """Real camera geometry from a nerfstudio transforms.json: per-camera label,
    center, 3x4 projection matrix, plus the rig-average up axis and reference
    intrinsics (first frame's focal and width) used to scale the render frustum."""
    data = json.loads(transforms_path.read_text())
    frames = data["frames"]
    if not frames:
        raise ValueError(f"{transforms_path} has no frames")

    c2ws = [np.array(fr["transform_matrix"], dtype=np.float64) for fr in frames]
    up = np.mean([m[:3, 1] for m in c2ws], axis=0)
    up = up / np.linalg.norm(up)

    cameras = []
    for fr, c2w in zip(frames, c2ws):
        w2c = opengl_c2w_to_w2c(c2w)
        intrinsics = np.array([[fr["fl_x"], 0.0, fr["cx"]],
                               [0.0, fr["fl_y"], fr["cy"]],
                               [0.0, 0.0, 1.0]])
        cameras.append({
            "label": str(fr.get("camera_label", "")),
            "center": c2w[:3, 3],
            "projection": intrinsics @ w2c[:3, :],
        })
    reference = frames[0]
    return cameras, up, float(reference["fl_x"]), float(reference["w"])


def load_face_keypoints(kp2d_dir: Path, tem_label: str, cameras: list) -> dict:
    """{camera_label: (nose_uv, left_eye_uv, right_eye_uv)} for every camera whose
    Sapiens prediction resolves all three face keypoints confidently."""
    found = {}
    for cam in cameras:
        json_path = find_keypoints_json(kp2d_dir, cam["label"])
        if json_path is None:
            continue
        if json_path.stem != tem_label and (kp2d_dir / cam["label"]).is_dir():
            candidate = kp2d_dir / cam["label"] / f"{tem_label}.json"
            if candidate.exists():
                json_path = candidate
        data = json.loads(json_path.read_text())
        instances = data.get("instance_info") or data.get("instances")
        if not instances:
            continue
        inst = instances[0]
        points = np.asarray(inst["keypoints"], dtype=np.float64)
        scores = np.asarray(inst["keypoint_scores"], dtype=np.float64)
        if len(points) <= RIGHT_EYE_KP:
            continue
        if min(scores[NOSE_KP], scores[LEFT_EYE_KP], scores[RIGHT_EYE_KP]) < FACE_SCORE_THRESHOLD:
            continue
        found[cam["label"]] = (points[NOSE_KP, :2], points[LEFT_EYE_KP, :2], points[RIGHT_EYE_KP, :2])
    return found


def facing_azimuth(face_kps: dict, cameras: list, up: np.ndarray,
                   e1: np.ndarray, e2: np.ndarray, fallback: float) -> float:
    """Azimuth (in the e1/e2 basis) a camera must sit at to see the face head-on.

    Forward is (nose - eye midpoint) with the up component projected out:
    anatomically the nose tip sits forward of the eye line, which gives a
    horizontal facing direction without having to disambiguate the sign of an
    eye-line cross product. Needs two cameras to triangulate; returns
    `fallback` otherwise."""
    by_label = {cam["label"]: cam for cam in cameras}
    labels = [lab for lab in face_kps if lab in by_label]
    if len(labels) < 2:
        return fallback
    projections = [by_label[lab]["projection"] for lab in labels]
    nose = triangulate_dlt(projections, [face_kps[lab][0] for lab in labels])
    left_eye = triangulate_dlt(projections, [face_kps[lab][1] for lab in labels])
    right_eye = triangulate_dlt(projections, [face_kps[lab][2] for lab in labels])

    forward = nose - (left_eye + right_eye) / 2.0
    forward = forward - (forward @ up) * up
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return fallback
    forward = forward / norm
    return float(np.degrees(np.arctan2(forward @ e2, forward @ e1)))


def load_splat(splat_ply: Path, anchor: np.ndarray | None, radius: float | None):
    """Gaussian parameters as numpy arrays, optionally bounded to `radius`
    around `anchor`.

    The spatial bound matters: mask-consistency filtering cannot remove junk
    hiding BEHIND the subject inside the silhouette frustum (see
    filter_splat_by_masks.py's known limitation). When that junk dominates,
    the high-opacity median flips to the junk cluster, orbit cameras aim at
    background, and spherical harmonics evaluated from out-of-distribution
    directions render solid black."""
    vertex = PlyData.read(str(splat_ply))["vertex"]
    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    n_rest = len([p.name for p in vertex.properties if p.name.startswith("f_rest_")])
    dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=1)[:, None, :]
    rest = np.stack([vertex[f"f_rest_{i}"] for i in range(n_rest)], axis=1)
    rest = rest.reshape(len(means), 3, n_rest // 3).transpose(0, 2, 1)
    sh = np.concatenate([dc, rest], axis=1)
    quats = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1)
    scales = np.exp(np.stack([vertex[f"scale_{i}"] for i in range(3)], axis=1))
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float64)))

    keep = np.isfinite(means).all(axis=1)
    if anchor is not None:
        keep &= np.linalg.norm(means - anchor[None, :], axis=1) < radius

    sh_degree = int(round(np.sqrt(sh.shape[1]))) - 1
    if (sh_degree + 1) ** 2 != sh.shape[1]:
        raise ValueError(f"{splat_ply}: {sh.shape[1]} SH coefficients is not a whole degree")
    return means[keep], quats[keep], scales[keep], opacity[keep], sh[keep], sh_degree


def subject_anchors(means: np.ndarray, opacity: np.ndarray, up: np.ndarray):
    """(subject centroid, head anchor) from the cleaned splat. The centroid is
    the median of high-opacity splats; the head is the median of those in the
    top 15% of the subject's extent along the up axis."""
    high = opacity > HIGH_OPACITY
    centroid = np.median(means[high], axis=0) if high.sum() > MIN_CENTROID_SPLATS else np.median(means, axis=0)

    high_means = means[high]
    if len(high_means) > 20:
        along_up = (high_means - centroid[None, :]) @ up
        head_sel = along_up >= np.percentile(along_up, HEAD_ANCHOR_PERCENTILE)
        if head_sel.sum() > MIN_HEAD_SPLATS:
            return centroid, np.median(high_means[head_sel], axis=0)
    return centroid, centroid + up * DEFAULT_HEAD_OFFSET


def render_orbit(splat_ply: Path, transforms: Path, out_dir: Path, *,
                 kp2d_dir: Path | None = None, tem_label: str = "000000",
                 subject_anchor_ply: Path | None = None, subject_radius: float | None = None,
                 res: int = 1536, zoom: float = 1.6, elev_rows: int = 3, row_azimuths: int = 12,
                 merge_deg: float = 3.0, max_reach_deg: float = 20.0,
                 head_views: int = 8, head_zoom: float = 11.0) -> dict:
    """Render one frame's orbit. Returns the cameras.json metadata it wrote."""
    import torch
    from gsplat import rasterization

    cameras, up, reference_focal, reference_width = load_rig(transforms)

    anchor = None
    if subject_anchor_ply is not None:
        anchor_vertex = PlyData.read(str(subject_anchor_ply))["vertex"]
        anchor = np.median(np.stack([anchor_vertex["x"], anchor_vertex["y"], anchor_vertex["z"]], axis=1), axis=0)

    means, quats, scales, opacity, sh, sh_degree = load_splat(splat_ply, anchor, subject_radius)
    if len(means) == 0:
        raise ValueError(f"{splat_ply}: no splats survived the spatial bound -- check --subject_anchor_ply/--subject_radius")
    centroid, head_anchor = subject_anchors(means, opacity, up)

    def to_cuda(a):
        return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device="cuda")

    gs = [to_cuda(means), to_cuda(quats), to_cuda(scales), to_cuda(opacity), to_cuda(sh)]

    # orbit basis: e1 points from the subject toward the first real camera, e2 completes
    # the right-handed horizontal frame, so azimuths are directly comparable to the rig's
    first = cameras[0]["center"] - centroid
    e1 = first - (first @ up) * up
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    offsets = np.stack([cam["center"] for cam in cameras]) - centroid[None, :]
    real_azimuths = np.degrees(np.arctan2(offsets @ e2, offsets @ e1))
    real_elevations = np.degrees(np.arcsin((offsets @ up) / np.linalg.norm(offsets, axis=1)))
    radius = float(np.mean(np.linalg.norm(offsets, axis=1)))
    elev_lo, elev_hi = float(real_elevations.min()), float(real_elevations.max())

    out_dir.mkdir(parents=True, exist_ok=True)

    def render_pass(pass_zoom, poses, target, name_fmt):
        focal = float(reference_focal * (res / reference_width) * pass_zoom)
        principal = res / 2.0
        intrinsics = torch.tensor([[focal, 0.0, principal], [0.0, focal, principal], [0.0, 0.0, 1.0]],
                                  dtype=torch.float32, device="cuda")[None]
        records = []
        for idx, (elev, azim) in enumerate(poses):
            ring = np.cos(np.radians(azim)) * e1 + np.sin(np.radians(azim)) * e2
            position = target + radius * (np.cos(np.radians(elev)) * ring + np.sin(np.radians(elev)) * up)
            w2c = lookat_w2c(position, target, -up)
            viewmat = torch.tensor(w2c, dtype=torch.float32, device="cuda")[None]
            with torch.no_grad():
                colors, alphas, _ = rasterization(*gs, viewmat, intrinsics, res, res,
                                                  sh_degree=sh_degree, render_mode="RGB")
            rgb = (colors[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            alpha = (alphas[0, ..., 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").save(out_dir / name_fmt.format(idx))
            records.append({"idx": idx, "elev": float(elev), "azim": float(azim),
                            "w2c": w2c.tolist(), "position": position.tolist()})
        return focal, principal, records

    body_azimuths = densified_azimuths(real_azimuths, row_azimuths, merge_deg, max_reach_deg)
    body_poses = [(elev, azim)
                  for elev in np.linspace(elev_lo, elev_hi, elev_rows)
                  for azim in body_azimuths]
    focal, principal, body_records = render_pass(zoom, body_poses, centroid, "body_{:03d}.png")
    meta = {"res": res, "zoom": zoom, "fl_x": focal, "fl_y": focal, "cx": principal, "cy": principal,
            "target": centroid.tolist(), "frames": body_records}

    if head_views > 0:
        face_kps = load_face_keypoints(kp2d_dir, tem_label, cameras) if kp2d_dir else {}
        fallback = float((real_azimuths.min() + real_azimuths.max()) / 2.0)
        center = facing_azimuth(face_kps, cameras, up, e1, e2, fallback)
        elev_mid = (elev_lo + elev_hi) / 2.0
        head_poses = [(elev_mid, center - 90.0 + 180.0 * j / max(head_views - 1, 1))
                      for j in range(head_views)]
        head_focal, head_principal, head_records = render_pass(head_zoom, head_poses, head_anchor, "head_{:03d}.png")
        meta["head"] = {"zoom": head_zoom, "fl_x": head_focal, "fl_y": head_focal,
                        "cx": head_principal, "cy": head_principal,
                        "target": head_anchor.tolist(), "azim_center": center,
                        "faces_triangulated": len(face_kps), "frames": head_records}

    (out_dir / "cameras.json").write_text(json.dumps(meta, indent=1))
    n_head = len(meta.get("head", {}).get("frames", []))
    print(f"{out_dir.name}: {len(body_records)} body + {n_head} head views at {res}px, "
          f"centroid {np.round(centroid, 3).tolist()}, "
          f"azimuth anchors {len(body_azimuths)}/row, elevation [{elev_lo:.1f}, {elev_hi:.1f}]"
          + (f", face-front {meta['head']['azim_center']:.1f} deg "
             f"({meta['head']['faces_triangulated']} cameras)" if n_head else ""))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splat_ply", required=True, type=Path,
                    help="one frame's trained splat, mask-filtered (filter_splat_by_masks.py output)")
    ap.add_argument("--transforms", required=True, type=Path,
                    help="nerfstudio transforms.json for the real rig -- supplies the orbit "
                         "radius, elevation range, azimuth anchors and reference intrinsics")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--kp2d_dir", type=Path, default=None,
                    help="poses_2d/<camera_label>/<tem_label>.json keypoints; used only to aim "
                         "the head pass at the true facing direction (falls back to the rig midpoint)")
    ap.add_argument("--tem_label", default="000000", help="6-digit frame label inside --kp2d_dir (default: 000000)")
    ap.add_argument("--subject_anchor_ply", type=Path, default=None,
                    help="subject point cloud (e.g. poses_pcd_fullres/<tem>.ply); with "
                         "--subject_radius, bounds the splat before rendering -- strongly "
                         "recommended, background junk otherwise drags the look-at off-subject")
    ap.add_argument("--subject_radius", type=float, default=None,
                    help="radius (world units) around the subject anchor; requires --subject_anchor_ply")
    ap.add_argument("--res", type=int, default=1536,
                    help="square render resolution (default 1536, the repair model's native size)")
    ap.add_argument("--zoom", type=float, default=1.6, help="body-pass focal multiplier (default 1.6)")
    ap.add_argument("--elev_rows", type=int, default=3,
                    help="elevation rows spanning the real rig's elevation range (default 3)")
    ap.add_argument("--row_azimuths", type=int, default=12,
                    help="azimuth samples per elevation row (default 12, so 36 body views total)")
    ap.add_argument("--merge_deg", type=float, default=3.0,
                    help="real azimuths closer than this merge into one anchor (default 3.0)")
    ap.add_argument("--max_reach_deg", type=float, default=20.0,
                    help="how far an inserted sample may walk from its anchor (default 20.0)")
    ap.add_argument("--head_views", type=int, default=8, help="zoomed head poses (default 8)")
    ap.add_argument("--head_zoom", type=float, default=11.0, help="head-pass focal multiplier (default 11.0)")
    ap.add_argument("--no_head_views", action="store_true", help="skip the head pass entirely")
    args = ap.parse_args()

    if (args.subject_radius is None) != (args.subject_anchor_ply is None):
        ap.error("--subject_anchor_ply and --subject_radius must be used together")

    try:
        render_orbit(args.splat_ply, args.transforms, args.out_dir,
                     kp2d_dir=args.kp2d_dir, tem_label=args.tem_label,
                     subject_anchor_ply=args.subject_anchor_ply, subject_radius=args.subject_radius,
                     res=args.res, zoom=args.zoom, elev_rows=args.elev_rows,
                     row_azimuths=args.row_azimuths, merge_deg=args.merge_deg,
                     max_reach_deg=args.max_reach_deg,
                     head_views=0 if args.no_head_views else args.head_views,
                     head_zoom=args.head_zoom)
    except (OSError, ValueError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
