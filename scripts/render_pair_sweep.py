#!/usr/bin/env python3
"""
render_pair_sweep.py

Render a camera-pair "tween" sweep: a short, temporally advancing arc from one
real camera to the next, at poses the renderer chooses rather than poses a
generative model invents. This is the render half of video render-and-repair --
the video-model successor to the per-frame image repair in
docs/render_and_repair.md. Its output feeds a video-to-video repair pass and,
after that, build_refit_dataset.py.

Why a sweep instead of an orbit (render_orbit_views.py): an image model repairs
each view alone, so neighbouring views only agree by luck -- klein_repair_views.py
holds a single seed across the orbit precisely to force that agreement. A video
model produces a coherent sequence natively, but it wants a *sequence*: frames
that move, in order. This script produces one.

Why the arc advances in time as it turns: a sweep that freezes time asks the
video model to do the one thing video models are worst at, hold a subject
perfectly still while the camera flies. Advancing one capture frame per sweep
frame means the motion in the clip is the subject's real motion, and it gives
every generated frame a distinct (pose, time) rather than many views of one
instant. Run a pair in both directions to get two synthetic poses per instant.

THE ENDPOINT PROPERTY, which is the point of this script:

    Every swept frame looks at the subject centroid through one shared square
    frustum, but its CENTRE is interpolated between the two real camera centres
    and, at the endpoints, equals a real centre exactly. Two cameras sharing a
    centre and differing only in aim and intrinsics are related by an exact
    homography (K_new R_new R_real^-1 K_real^-1) -- no depth, no
    approximation. So at each endpoint this script can warp the REAL photo into
    the sweep frustum losslessly (up to resampling) and write it alongside the
    render.

    That gives the repair pass real pixels to pin its first and last frame to,
    which is what keeps generated colour, exposure and micro-detail on the real
    cameras' manifold. Feeding a model's own idea of the subject into training
    was measured to make held-out novel-view reconstruction WORSE
    (18.0 dB -> 13.3 dB); see docs/render_and_repair.md.

    The same property makes the bakeoff possible. --holdout_label names a real
    camera that sits between the pair and is excluded from the sweep's
    endpoints; the frame nearest it is snapped to its exact centre, so that
    camera's real photo warps into the probe frustum and becomes ground truth
    for whatever the video model generated there. score_novel_views.py consumes
    the `probe` block this writes.

conda env: diffuman4d (gsplat + torch + numpy + PIL + scipy + plyfile), same as
render_orbit_views.py, whose rig loading, splat loading and look-at conventions
this reuses.

Usage:
    python3 render_pair_sweep.py \\
        --sequence_root /path/to/run \\
        --transforms /path/to/transforms.json \\
        --out_dir /path/to/sweeps \\
        --pair 03 04 \\
        [--holdout_label 04] \\
        [--frames 0-80] [--fps 29.97] [--res 1024] [--zoom 1.6] \\
        [--splat_pattern 'exports/*_maskfilt.ply'] \\
        [--subject_anchor_pattern 'poses_pcd_fullres/*.ply'] [--subject_radius 3.0]

    --pair may be given more than once; --adjacent_pairs sweeps every
    neighbouring pair in azimuth instead.

PER-FRAME LOSS WEIGHTS

    cameras.json carries a `loss_weight` per frame, peaking at the real-pinned
    endpoints and falling to 1.0 at the arc's midpoint (see
    angular_loss_weights). Hwang et al., "4D Human-Scene Reconstruction from
    Low-Overlap Captures" (SIGGRAPH 2026), measured this on the same problem and
    reported it as Equation S2. Weighting every generated frame equally spends as
    much of the fit's attention on the least supported view as on the best.

    That paper is worth reading before trusting this whole approach: it
    reconstructs BACKGROUNDS from generated views and masks the moving human out
    of that supervision entirely, fitting the person from real views only. Its
    caution and this repo's own 18.0 -> 13.3 dB result point the same way, so
    treat generated supervision of the subject as the thing the bakeoff has to
    prove rather than assume.

Output, per pair, in out_dir/<A>_to_<B>/:
    sweep_NNNN.png    RGBA renders, one per capture frame, in clip order
    real_first.png    the real photo at camera A, warped into the sweep frustum
    real_last.png     the real photo at camera B, likewise
    probe_real.png    (with --holdout_label) the held-out camera's real photo
    cameras.json      per-frame w2c, time, frame dir and loss_weight, plus the
                      shared intrinsics and a `probe` block naming the probe index
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy import ndimage

from image_formats import SUPPORTED_IMAGE_EXTS
from render_and_repair_sequence import find_splat, resolve_frames
from render_orbit_views import lookat_w2c, load_rig, load_splat, subject_anchors

WARP_ORDER = 1  # bilinear; the homography is exact, so resampling is the only error
LAMBDA_GT = 2.0  # angular loss-weight strength; Hwang et al. (SIGGRAPH 2026) Eq. S2


def spherical(center: np.ndarray, target: np.ndarray, up: np.ndarray,
              e1: np.ndarray, e2: np.ndarray) -> tuple:
    """(azimuth deg, elevation deg, radius) of `center` about `target`, in the
    same e1/e2/up basis render_orbit_views.py uses, so the two scripts' poses
    are directly comparable."""
    offset = center - target
    radius = float(np.linalg.norm(offset))
    elevation = float(np.degrees(np.arcsin((offset @ up) / radius)))
    azimuth = float(np.degrees(np.arctan2(offset @ e2, offset @ e1)))
    return azimuth, elevation, radius


def position_at(target: np.ndarray, up: np.ndarray, e1: np.ndarray, e2: np.ndarray,
                azimuth: float, elevation: float, radius: float) -> np.ndarray:
    """Inverse of `spherical`. Round-trips a real camera centre exactly, which is
    what makes the endpoint homography exact rather than approximate."""
    ring = np.cos(np.radians(azimuth)) * e1 + np.sin(np.radians(azimuth)) * e2
    return target + radius * (np.cos(np.radians(elevation)) * ring
                              + np.sin(np.radians(elevation)) * up)


def shortest_arc(start_deg: float, end_deg: float) -> float:
    """Signed azimuth delta taking the short way round, so a pair straddling the
    +/-180 wrap sweeps across the gap between them rather than the long way
    through the whole rig."""
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def sweep_parameters(cam_a: dict, cam_b: dict, target: np.ndarray, up: np.ndarray,
                     e1: np.ndarray, e2: np.ndarray, count: int,
                     holdout: dict | None = None) -> tuple:
    """Per-frame (azimuth, elevation, radius) from camera A to camera B, plus the
    index snapped to the holdout camera (or None).

    Endpoints are the real cameras' own spherical coordinates, so frame 0 and
    frame count-1 sit exactly at their centres. The holdout frame is snapped the
    same way: its interpolated parameters are replaced wholesale by the real
    camera's, so the probe render shares that camera's centre exactly and its
    photo warps in without approximation. Snapping perturbs the arc's spacing
    slightly, and that is the right trade -- an inexact probe would not be
    ground truth at all."""
    if count < 2:
        raise ValueError(f"a sweep needs at least 2 frames, got {count}")

    azimuth_a, elevation_a, radius_a = spherical(cam_a["center"], target, up, e1, e2)
    azimuth_b, elevation_b, radius_b = spherical(cam_b["center"], target, up, e1, e2)
    span = shortest_arc(azimuth_a, azimuth_b)

    fractions = np.linspace(0.0, 1.0, count)
    parameters = [(azimuth_a + span * f,
                   elevation_a + (elevation_b - elevation_a) * f,
                   radius_a + (radius_b - radius_a) * f) for f in fractions]

    probe_index = None
    if holdout is not None:
        azimuth_h, elevation_h, radius_h = spherical(holdout["center"], target, up, e1, e2)
        offset = shortest_arc(azimuth_a, azimuth_h)
        if span == 0.0 or not 0.0 <= offset / span <= 1.0:
            raise ValueError(
                f"holdout camera {holdout['label']!r} at azimuth {azimuth_h:.1f} deg does not lie "
                f"between {cam_a['label']!r} ({azimuth_a:.1f}) and {cam_b['label']!r} ({azimuth_b:.1f}) "
                "-- a probe outside the swept arc is never rendered")
        # nearest interior frame; the endpoints stay real-pinned and are never the probe
        interior = range(1, count - 1)
        probe_index = min(interior, key=lambda i: abs(shortest_arc(parameters[i][0], azimuth_h)))
        parameters[probe_index] = (azimuth_h, elevation_h, radius_h)

    return parameters, probe_index


def angular_loss_weights(parameters: list, lambda_gt: float = LAMBDA_GT) -> list:
    """Per-frame training weight, highest at the two real cameras and falling to
    1.0 at the middle of the arc:

        w = 1 + lambda_gt * (1 + cos(pi * d / d_max)) / 2

    with d the angular distance to the nearer endpoint. This is Equation S2 of
    Hwang et al., "4D Human-Scene Reconstruction from Low-Overlap Captures"
    (SIGGRAPH 2026), which used lambda_gt = 2 -- so a frame sitting on a real
    camera is worth 3x one at the arc's midpoint.

    The reasoning transfers directly to a sweep: both endpoints are pinned to
    real pixels, so a generated frame's trustworthiness degrades smoothly with
    how far it has travelled from them. A flat weight would spend as much of the
    fit's attention on the least supported frame as on the best.

    Only the ENDPOINTS anchor the weighting. The bakeoff holdout is an interior
    frame, so it never attracts endpoint weight -- which is what keeps the
    held-out camera out of the training signal that its own score judges."""
    if len(parameters) < 2:
        return [1.0] * len(parameters)

    span = abs(shortest_arc(parameters[0][0], parameters[-1][0]))
    reach = span / 2.0  # the midpoint is the farthest any frame gets from both ends
    if reach <= 0:
        return [1.0 + lambda_gt] * len(parameters)

    weights = []
    for azimuth, _, _ in parameters:
        travelled = abs(shortest_arc(parameters[0][0], azimuth))
        distance = min(travelled, abs(span - travelled))
        falloff = (1.0 + np.cos(np.pi * min(distance / reach, 1.0))) / 2.0
        weights.append(float(1.0 + lambda_gt * falloff))
    return weights


def homography_to_frustum(camera: dict, w2c_new: np.ndarray, focal: float, principal: float) -> np.ndarray:
    """3x3 mapping NEW image pixels back to `camera`'s real image pixels.

    Valid only because the two cameras share a centre: with translation gone, a
    world point's ray is the same for both and only rotation and intrinsics
    differ. Returned inverted (new -> real) because resampling pulls."""
    intrinsics_new = np.array([[focal, 0.0, principal], [0.0, focal, principal], [0.0, 0.0, 1.0]])
    rotation = camera["w2c"][:3, :3] @ w2c_new[:3, :3].T
    return camera["intrinsics"] @ rotation @ np.linalg.inv(intrinsics_new)


def warp_real_image(image_path: Path, homography: np.ndarray, res: int) -> Image.Image:
    """Resample a real photo into the square sweep frustum through `homography`.

    Pixels whose ray leaves the real image (or falls behind the camera) come back
    transparent, so the repair pass and the scorer can both tell "no real
    coverage here" from "real black here"."""
    source = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    height, width = source.shape[:2]

    ys, xs = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    rays = homography @ np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    depth = rays[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = rays[0] / depth, rays[1] / depth

    inside = (depth > 0) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    coordinates = np.stack([np.where(inside, v, 0.0), np.where(inside, u, 0.0)])

    channels = [ndimage.map_coordinates(source[..., c], coordinates, order=WARP_ORDER, mode="nearest")
                for c in range(3)]
    rgb = np.stack(channels, axis=-1).reshape(res, res, 3).clip(0, 255).astype(np.uint8)
    alpha = (inside.reshape(res, res) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")


def resolve_real_image(camera: dict, transforms: Path, images_dir: Path | None) -> Path | None:
    """The real photo behind a transforms.json frame. `file_path` is normally
    relative to the transforms file; --images_dir overrides that for datasets
    moved after they were written, and the extension is re-sniffed because
    build_refit_dataset.py's uniform-.png naming means the recorded name may not
    be the one on disk."""
    recorded = camera.get("file_path") or ""
    if not recorded:
        return None
    candidates = [transforms.parent / recorded]
    if images_dir is not None:
        candidates.append(images_dir / Path(recorded).name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
        for extension in SUPPORTED_IMAGE_EXTS:
            alternative = candidate.with_suffix(extension)
            if alternative.exists():
                return alternative
    return None


def render_pair_sweep(sequence_root: Path, transforms: Path, out_dir: Path, *,
                      pair: tuple, frame_range: str | None = None, fps: float = 29.97,
                      res: int = 1024, zoom: float = 1.6,
                      splat_pattern: str = "exports/*_maskfilt.ply",
                      subject_anchor_pattern: str = "poses_pcd_fullres/*.ply",
                      subject_radius: float | None = None,
                      holdout_label: str | None = None,
                      images_dir: Path | None = None,
                      lambda_gt: float = LAMBDA_GT) -> dict:
    """Render one pair's sweep. Returns the cameras.json metadata it wrote."""
    import torch
    from gsplat import rasterization

    cameras, up, reference_focal, reference_width = load_rig(transforms)
    by_label = {camera["label"]: camera for camera in cameras}
    for label in (*pair, *( [holdout_label] if holdout_label else [] )):
        if label not in by_label:
            raise KeyError(f"camera {label!r} is not in {transforms} "
                           f"(have {sorted(by_label)[:8]}{'...' if len(by_label) > 8 else ''})")
    if holdout_label in pair:
        raise ValueError(f"--holdout_label {holdout_label!r} is one of the swept endpoints; "
                         "a camera pinned as real ground truth cannot also be held out")

    frame_dirs = resolve_frames(sequence_root, frame_range)
    if len(frame_dirs) < 2:
        raise ValueError(f"{sequence_root} yielded {len(frame_dirs)} frame(s) for range {frame_range!r}; "
                         "a sweep needs at least 2")

    # the arc is fixed by the FIRST frame's subject: re-deriving the centroid per
    # frame would make the camera path chase the subject's own motion, which reads
    # as a wobble in the clip and, worse, means no two frames share a pose basis
    first_splat = find_splat(frame_dirs[0], splat_pattern)
    if first_splat is None:
        raise FileNotFoundError(f"no splat matching {splat_pattern!r} in {frame_dirs[0]}")
    anchor = None
    anchors = sorted(frame_dirs[0].glob(subject_anchor_pattern))
    if anchors:
        vertex = PlyData.read(str(anchors[0]))["vertex"]
        anchor = np.median(np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1), axis=0)
    means, _, _, opacity, _, _ = load_splat(first_splat, anchor, subject_radius)
    if len(means) == 0:
        raise ValueError(f"{first_splat}: no splats survived the spatial bound -- check --subject_radius")
    centroid, _ = subject_anchors(means, opacity, up)

    first_offset = cameras[0]["center"] - centroid
    e1 = first_offset - (first_offset @ up) * up
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    parameters, probe_index = sweep_parameters(
        by_label[pair[0]], by_label[pair[1]], centroid, up, e1, e2,
        len(frame_dirs), by_label.get(holdout_label) if holdout_label else None)

    focal = float(reference_focal * (res / reference_width) * zoom)
    principal = res / 2.0
    intrinsics = torch.tensor([[focal, 0.0, principal], [0.0, focal, principal], [0.0, 0.0, 1.0]],
                              dtype=torch.float32, device="cuda")[None]
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_cuda(array):
        return torch.tensor(np.ascontiguousarray(array), dtype=torch.float32, device="cuda")

    weights = angular_loss_weights(parameters, lambda_gt)
    records = []
    for index, (frame_dir, (azimuth, elevation, radius)) in enumerate(zip(frame_dirs, parameters)):
        splat_ply = find_splat(frame_dir, splat_pattern)
        if splat_ply is None:
            raise FileNotFoundError(f"no splat matching {splat_pattern!r} in {frame_dir}")
        frame_means, quats, scales, frame_opacity, sh, sh_degree = load_splat(splat_ply, anchor, subject_radius)

        position = position_at(centroid, up, e1, e2, azimuth, elevation, radius)
        w2c = lookat_w2c(position, centroid, -up)
        viewmat = torch.tensor(w2c, dtype=torch.float32, device="cuda")[None]

        with torch.no_grad():
            colors, alphas, _ = rasterization(
                to_cuda(frame_means), to_cuda(quats), to_cuda(scales), to_cuda(frame_opacity), to_cuda(sh),
                viewmat, intrinsics, res, res, sh_degree=sh_degree, render_mode="RGB")
        rgb = (colors[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        alpha = (alphas[0, ..., 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").save(out_dir / f"sweep_{index:04d}.png")

        records.append({"idx": index, "frame_dir": str(frame_dir), "time": index / fps,
                        "azim": azimuth, "elev": elevation, "radius": radius,
                        "loss_weight": weights[index],
                        "w2c": w2c.tolist(), "position": position.tolist()})

    meta = {"pair": list(pair), "res": res, "zoom": zoom, "fps": fps, "lambda_gt": lambda_gt,
            "fl_x": focal, "fl_y": focal, "cx": principal, "cy": principal,
            "target": centroid.tolist(), "frames": records}

    # real-pixel endpoints and, when asked, the held-out probe
    pinned = {"real_first": (by_label[pair[0]], 0),
              "real_last": (by_label[pair[1]], len(records) - 1)}
    if probe_index is not None:
        pinned["probe_real"] = (by_label[holdout_label], probe_index)

    warped = {}
    for name, (camera, index) in pinned.items():
        image_path = resolve_real_image(camera, transforms, images_dir)
        if image_path is None:
            print(f"  WARNING: no real image on disk for camera {camera['label']!r} "
                  f"({camera.get('file_path')!r}) -- {name}.png not written; "
                  "the repair pass will have nothing to pin this end to", file=sys.stderr)
            continue
        homography = homography_to_frustum(camera, np.array(records[index]["w2c"]), focal, principal)
        warp_real_image(image_path, homography, res).save(out_dir / f"{name}.png")
        warped[name] = {"camera": camera["label"], "idx": index, "source": str(image_path)}

    meta["real_endpoints"] = warped
    if probe_index is not None:
        meta["probe"] = {"camera": holdout_label, "idx": probe_index,
                         "image": "probe_real.png" if "probe_real" in warped else None,
                         "render": f"sweep_{probe_index:04d}.png"}

    (out_dir / "cameras.json").write_text(json.dumps(meta, indent=1))
    print(f"{out_dir.name}: {len(records)} frames, cameras {pair[0]}->{pair[1]}, "
          f"azimuth {parameters[0][0]:.1f} -> {parameters[-1][0]:.1f} deg at {res}px"
          + (f", probe {holdout_label!r} at index {probe_index}" if probe_index is not None else "")
          + (f", pinned {sorted(warped)}" if warped else ", NO real endpoints"))
    return meta


def adjacent_pairs(cameras: list, centroid: np.ndarray, up: np.ndarray,
                   e1: np.ndarray, e2: np.ndarray, merge_deg: float) -> list:
    """Neighbouring camera pairs in azimuth order, skipping pairs closer together
    than `merge_deg`. The rig is stereo pairs 0.3-0.8 degrees apart
    (render_orbit_views.py), and sweeping between the two halves of one stereo
    pair generates a clip of near-identical frames."""
    ordered = sorted(cameras, key=lambda c: spherical(c["center"], centroid, up, e1, e2)[0])
    pairs = []
    for left, right in zip(ordered, ordered[1:]):
        gap = abs(shortest_arc(spherical(left["center"], centroid, up, e1, e2)[0],
                               spherical(right["center"], centroid, up, e1, e2)[0]))
        if gap >= merge_deg:
            pairs.append((left["label"], right["label"]))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence_root", required=True, type=Path,
                    help="per-frame run root (render_frame_sequence.py output), one dir per frame")
    ap.add_argument("--transforms", required=True, type=Path,
                    help="nerfstudio transforms.json for the real rig")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="parent directory; each pair gets an <A>_to_<B> subdirectory")
    ap.add_argument("--pair", action="append", nargs=2, metavar=("A", "B"), default=None,
                    help="camera_label pair to sweep between; repeatable")
    ap.add_argument("--adjacent_pairs", action="store_true",
                    help="sweep every neighbouring pair in azimuth instead of named pairs")
    ap.add_argument("--holdout_label", default=None,
                    help="real camera between the pair to reserve as bakeoff ground truth: the "
                         "nearest swept frame is snapped to its exact centre and its photo warped "
                         "in as probe_real.png. Exclude this camera from training too")
    ap.add_argument("--frames", default=None, metavar="LO-HI",
                    help="limit to a frame-number range; one sweep frame per capture frame, so "
                         "this also sets the clip length (match your video model's native length)")
    ap.add_argument("--fps", type=float, default=29.97, help="capture fps, for per-frame timestamps")
    ap.add_argument("--res", type=int, default=1024,
                    help="square render size; match the video model's native resolution")
    ap.add_argument("--zoom", type=float, default=1.6, help="frustum tightening vs the real cameras")
    ap.add_argument("--splat_pattern", default="exports/*_maskfilt.ply",
                    help="glob inside each frame directory for that frame's mask-filtered splat")
    ap.add_argument("--subject_anchor_pattern", default="poses_pcd_fullres/*.ply",
                    help="glob inside the first frame dir for the triangulated subject cloud")
    ap.add_argument("--subject_radius", type=float, default=None,
                    help="bound splats to this radius around the anchor (see render_orbit_views.py)")
    ap.add_argument("--merge_deg", type=float, default=3.0,
                    help="with --adjacent_pairs, skip pairs closer than this in azimuth")
    ap.add_argument("--lambda_gt", type=float, default=LAMBDA_GT,
                    help="angular loss-weight strength written into cameras.json: a frame on a real "
                         "camera is worth (1 + lambda_gt) times one at the arc's midpoint. 0 gives "
                         "every frame equal weight (Hwang et al. SIGGRAPH 2026 measured 2.0)")
    ap.add_argument("--images_dir", type=Path, default=None,
                    help="override directory holding the real photos, when transforms.json's "
                         "file_path entries no longer resolve")
    args = ap.parse_args()

    if bool(args.pair) == bool(args.adjacent_pairs):
        ap.error("give either --pair (one or more) or --adjacent_pairs, not both or neither")
    if args.holdout_label and (args.adjacent_pairs or len(args.pair) > 1):
        ap.error("--holdout_label applies to a single --pair: the probe camera has to lie between "
                 "that pair's endpoints, which cannot hold for every pair at once. Run the bakeoff "
                 "one pair at a time")

    pairs = [tuple(p) for p in args.pair] if args.pair else None
    if pairs is None:
        cameras, up, _, _ = load_rig(args.transforms)
        frame_dirs = resolve_frames(args.sequence_root, args.frames)
        if not frame_dirs:
            print(f"ERROR: no frame directories in {args.sequence_root} for range {args.frames!r}",
                  file=sys.stderr)
            return 1
        splat_ply = find_splat(frame_dirs[0], args.splat_pattern)
        if splat_ply is None:
            print(f"ERROR: no splat matching {args.splat_pattern!r} in {frame_dirs[0]}", file=sys.stderr)
            return 1
        means, _, _, opacity, _, _ = load_splat(splat_ply, None, None)
        centroid, _ = subject_anchors(means, opacity, up)
        first_offset = cameras[0]["center"] - centroid
        e1 = first_offset - (first_offset @ up) * up
        e1 = e1 / np.linalg.norm(e1)
        pairs = adjacent_pairs(cameras, centroid, up, e1, np.cross(up, e1), args.merge_deg)
        print(f"{len(pairs)} adjacent pairs at >= {args.merge_deg} deg separation")

    for pair in pairs:
        render_pair_sweep(
            args.sequence_root, args.transforms, args.out_dir / f"{pair[0]}_to_{pair[1]}",
            pair=pair, frame_range=args.frames, fps=args.fps, res=args.res, zoom=args.zoom,
            splat_pattern=args.splat_pattern, subject_anchor_pattern=args.subject_anchor_pattern,
            subject_radius=args.subject_radius, holdout_label=args.holdout_label,
            images_dir=args.images_dir, lambda_gt=args.lambda_gt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
