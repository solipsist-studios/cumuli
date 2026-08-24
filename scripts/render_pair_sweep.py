#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""render_pair_sweep.py - camera-pair "tween" sweeps from a trained 4D model.

Renders a short arc from one real camera to the next, advancing in time as it
turns, at poses this script chooses rather than poses a generative model
invents. It is the render half of video render-and-repair: its output is what a
video model repairs, and after that a refit dataset is built from real plus
repaired views.

Why render before generating: a generative model handed two real photos and
asked to invent the views between them produces frames with no camera pose
attached, and a 4D fit needs an exact pose per image. Measured on this rig, that
approach scored 8.5 dB BELOW simply rendering the model (21.23 vs 29.78 against
a held-out camera), because the model interpolates along a path it imagines
rather than the true camera arc. Rendering first fixes the geometry and leaves
the model only the job it is good at.

Why time advances as the camera turns: a sweep that freezes time asks a video
model to hold a subject perfectly still while the camera flies, which is the one
thing video models are worst at. Advancing one capture frame per sweep frame
means the motion in the clip is the subject's real motion, and every generated
frame gets a distinct (pose, time). Run a pair in both directions for two
synthetic poses per instant.

THE ENDPOINT PROPERTY, which is the point of this script:

    Every swept frame looks at the subject through one shared square frustum,
    but its CENTRE is interpolated between the two real camera centres and, at
    the endpoints, equals a real centre exactly. Two cameras sharing a centre
    and differing only in aim and intrinsics are related by an exact homography
    (K_new R_new R_real^-1 K_real^-1), with no depth term and no approximation.
    So at each endpoint the REAL photo warps into the sweep frustum losslessly
    (up to resampling) and is written alongside the render.

    That gives the repair pass real pixels to pin its first and last frame to,
    which keeps generated colour, exposure and micro-detail on the real cameras'
    manifold.

    The same property makes a bakeoff possible. --holdout_label names a real
    camera between the pair; the swept frame nearest it is snapped to its exact
    centre, so that camera's photo warps into the probe frustum and becomes
    ground truth for whatever a model generated there. score_novel_views.py
    consumes the `probe` block this writes. Hold that camera out of training too,
    or the probe is not a probe.

Temporal evaluation matches eval_render.py exactly -- mean(t) = xyz + v*dt
(+ a*dt^2), alpha(t) = sigmoid(opacity) * exp(-0.5*(dt/t_sigma)^2) -- so a sweep
frame and an eval render of the same model at the same instant agree.

conda env: cumuli (gsplat + torch + CUDA for the render; numpy/PIL/scipy for the
rest).

Usage:
    python scripts/render_pair_sweep.py \\
        --model /path/to/splat_4d.sogst \\
        --transforms /path/to/transforms.json \\
        --out_dir /path/to/sweeps \\
        --pair cam02 cam06 \\
        [--holdout_label cam04] [--frames 58-74] \\
        [--res 1024] [--zoom 1.6] [--lambda_gt 2.0]

    --pair may be repeated; --adjacent_pairs sweeps every neighbouring pair.
    Clip length is the frame count: match your video model's native length
    (Wan needs 4k+1, so 17, 21, 33...).

Output, per pair, in out_dir/<A>_to_<B>/:
    sweep_NNNN.png    RGBA renders, one per capture frame, in clip order
    real_first.png    the real photo at camera A, warped into the sweep frustum
    real_last.png     the real photo at camera B, likewise
    probe_real.png    (with --holdout_label) the held-out camera's real photo
    cameras.json      per-frame w2c, time, loss_weight; shared intrinsics; the
                      rig `up` axis; and a `probe` block naming the probe index
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from image_formats import SUPPORTED_IMAGE_EXTS
from rig_geometry import (
    load_rig, lookat_w2c, orbit_basis, position_at, shortest_arc, spherical, subject_anchors,
)

WARP_ORDER = 1   # bilinear; the homography is exact, so resampling is the only error
LAMBDA_GT = 2.0  # angular loss-weight strength; Hwang et al. (SIGGRAPH 2026) Eq. S2


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
    slightly, and that is the right trade -- an inexact probe would not be ground
    truth at all."""
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
                f"between {cam_a['label']!r} ({azimuth_a:.1f}) and {cam_b['label']!r} "
                f"({azimuth_b:.1f}) -- a probe outside the swept arc is never rendered")
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
    (SIGGRAPH 2026), which used lambda_gt = 2, so a frame sitting on a real
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


def homography_to_frustum(camera: dict, w2c_new: np.ndarray, focal: float, principal: float,
                          intrinsics_real: np.ndarray | None = None) -> np.ndarray:
    """3x3 mapping NEW image pixels back to `camera`'s real image pixels.

    Valid only because the two cameras share a centre: with translation gone, a
    world point's ray is the same for both and only rotation and intrinsics
    differ. Returned inverted (new -> real) because resampling pulls.

    Pass `intrinsics_real` for the specific frame being warped. The camera-level
    intrinsics are the first frame's, and on a subject-cropped dataset the
    principal point moves every frame."""
    intrinsics_new = np.array([[focal, 0.0, principal], [0.0, focal, principal], [0.0, 0.0, 1.0]])
    real = camera["intrinsics"] if intrinsics_real is None else intrinsics_real
    rotation = camera["w2c"][:3, :3] @ w2c_new[:3, :3].T
    return real @ rotation @ np.linalg.inv(intrinsics_new)


def warp_real_image(image_path: Path, homography: np.ndarray, res: int) -> Image.Image:
    """Resample a real photo into the square sweep frustum through `homography`.

    Pixels whose ray leaves the real image (or falls behind the camera) come back
    transparent, so the repair pass and the scorer can both tell "no real
    coverage here" from "real black here"."""
    with Image.open(image_path) as opened:
        source = np.asarray(opened.convert("RGB"), dtype=np.float32)
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


def resolve_real_view(camera: dict, transforms: Path, images_dir: Path | None,
                      frame_number: int | None = None) -> tuple:
    """(path to the real photo, that photo's intrinsics) for one camera at one
    instant, or (None, None) when the file is missing.

    Both halves are per-instant, and the intrinsics half is the one that bites.
    Datasets that crop every frame around the moving subject shift the principal
    point frame to frame while the focal and pose hold still; warping frame 58's
    photo through frame 1's principal point misaligned it by 249 px vertically on
    one measured camera, enough to make a "real pixel" pin anchor the clip to the
    wrong place."""
    per_frame = camera.get("per_frame") or {}
    entry = per_frame.get(frame_number) if frame_number is not None else None
    if entry is None and len(per_frame) == 1:
        entry = next(iter(per_frame.values()))  # a static rig has one image per camera
    recorded = (entry or {}).get("file_path") or camera.get("file_path") or ""
    intrinsics = (entry or {}).get("intrinsics")
    if intrinsics is None:
        intrinsics = camera["intrinsics"]
    if not recorded:
        return None, None

    candidates = [Path(transforms).parent / recorded]
    if images_dir is not None:
        candidates.append(images_dir / Path(recorded).name)
    for candidate in candidates:
        if candidate.exists():
            return candidate, intrinsics
        for extension in SUPPORTED_IMAGE_EXTS:
            alternative = candidate.with_suffix(extension)
            if alternative.exists():
                return alternative, intrinsics
    return None, None


def resolve_frame_range(cameras: list, frame_range: str | None) -> list:
    """Capture frame numbers to sweep, in order, from what the rig actually has.

    Frames come from the transforms rather than from a directory listing, so the
    sweep can only ask for instants the dataset really contains."""
    available = sorted(n for camera in cameras for n in camera["per_frame"] if n is not None)
    unique = sorted(set(available))
    if not unique:
        return [None]  # a single-instant rig: one frame, no numbering
    if frame_range is None:
        return unique
    try:
        lo_text, hi_text = frame_range.split("-")
        lo, hi = int(lo_text), int(hi_text)
    except ValueError:
        raise ValueError(f"--frames must look like LO-HI, got {frame_range!r}") from None
    selected = [n for n in unique if lo <= n <= hi]
    if not selected:
        raise ValueError(f"no frames in {frame_range!r}; the rig has {unique[0]}-{unique[-1]}")
    return selected


def frame_time(cameras: list, frame_number: int | None, fps: float) -> float:
    """The dataset's own timestamp for a capture frame, falling back to the frame
    index over fps when the transforms records no `time`."""
    for camera in cameras:
        entry = camera["per_frame"].get(frame_number)
        if entry and entry.get("time") is not None:
            return float(entry["time"])
    return 0.0 if frame_number is None else float(frame_number) / fps


def adjacent_pairs(cameras: list, centroid: np.ndarray, up: np.ndarray,
                   e1: np.ndarray, e2: np.ndarray, merge_deg: float) -> list:
    """Neighbouring camera pairs in azimuth order, skipping pairs closer together
    than `merge_deg`. Rigs built from stereo pairs put their two halves under a
    degree apart, and sweeping between them generates a clip of near-identical
    frames."""
    ordered = sorted(cameras, key=lambda c: spherical(c["center"], centroid, up, e1, e2)[0])
    pairs = []
    for left, right in zip(ordered, ordered[1:]):
        gap = abs(shortest_arc(spherical(left["center"], centroid, up, e1, e2)[0],
                               spherical(right["center"], centroid, up, e1, e2)[0]))
        if gap >= merge_deg:
            pairs.append((left["label"], right["label"]))
    return pairs


def load_4d_model(model_path: Path):
    """The trained 4D model as numpy arrays, decoded the way the shipping viewer
    would. Delegates to eval_render.load_model so a sweep and an eval of the same
    asset can never disagree about what the model contains."""
    from eval_render import load_model
    return load_model(str(model_path))


def render_pair_sweep(model: Path, transforms: Path, out_dir: Path, *,
                      pair: tuple, frame_range: str | None = None, fps: float = 29.97,
                      res: int = 1024, zoom: float = 1.6,
                      holdout_label: str | None = None, images_dir: Path | None = None,
                      lambda_gt: float = LAMBDA_GT, time_scale: float | None = None,
                      point_render: float = 0.0) -> dict:
    """Render one pair's sweep. Returns the cameras.json metadata it wrote."""
    import torch
    from gsplat import rasterization

    cameras, up, reference_focal, reference_width = load_rig(transforms)
    by_label = {camera["label"]: camera for camera in cameras}
    for label in (*pair, *([holdout_label] if holdout_label else [])):
        if label not in by_label:
            raise KeyError(f"camera {label!r} is not in {transforms} "
                           f"(have {sorted(by_label)[:8]}{'...' if len(by_label) > 8 else ''})")
    if holdout_label in pair:
        raise ValueError(f"--holdout_label {holdout_label!r} is one of the swept endpoints; "
                         "a camera pinned as real ground truth cannot also be held out")

    frames = resolve_frame_range(cameras, frame_range)
    if len(frames) < 2:
        raise ValueError(f"{transforms} yielded {len(frames)} frame(s) for range {frame_range!r}; "
                         "a sweep needs at least 2")

    header, fields = load_4d_model(model)
    device = torch.device("cuda")

    def to_gpu(array):
        return torch.tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)

    xyz = to_gpu(np.stack([fields["x"], fields["y"], fields["z"]], axis=1))
    velocity = to_gpu(np.stack([fields["vx"], fields["vy"], fields["vz"]], axis=1))
    accel = to_gpu(np.stack([fields["ax"], fields["ay"], fields["az"]], axis=1)) \
        if "ax" in fields else None
    quats = to_gpu(np.stack([fields[f"rot_{i}"] for i in range(4)], axis=1))
    scales = torch.exp(to_gpu(np.stack([fields[f"scale_{i}"] for i in range(3)], axis=1)))
    if point_render > 0:
        # Collapse every Gaussian to a near-isotropic dot: the same geometry
        # rendered as a POINT CLOUD rather than a continuous surface, which is
        # the style Uni3C was trained to consume. A world-size around
        # radius/focal covers about one pixel at our frustum.
        scales = torch.full_like(scales, float(point_render))
    op_logit = to_gpu(fields["opacity"])
    t_center = to_gpu(fields["t_center"])
    t_sigma = to_gpu(np.maximum(np.abs(fields["t_sigma"]), 1e-6))

    count = header["count"]
    sh_degree = 3 if "f_rest" in fields else 0
    shs = torch.zeros((count, (sh_degree + 1) ** 2, 3), dtype=torch.float32, device=device)
    for channel in range(3):
        shs[:, 0, channel] = to_gpu(fields[f"f_dc_{channel}"])
    if sh_degree:
        shs[:, 1:, :] = to_gpu(fields["f_rest"]).reshape(count, 3, 15).permute(0, 2, 1)

    scale = 1.0 if time_scale is None else time_scale

    # Aim the arc at where the subject actually is in the MIDDLE of the window,
    # evaluated the same way every frame is. Using the static xyz instead would
    # aim at each Gaussian's position at its own t_center, which for a moving
    # subject is a blur of the whole trajectory rather than a place it ever is.
    mid_dt = (header["time_min"] + frame_time(cameras, frames[len(frames) // 2], fps) * scale) \
        - t_center
    mid_means = xyz + velocity * mid_dt[:, None]
    if accel is not None:
        mid_means = mid_means + accel * (mid_dt * mid_dt)[:, None]
    mid_alpha = torch.sigmoid(op_logit) * torch.exp(-0.5 * (mid_dt / t_sigma) ** 2)
    centroid, _ = subject_anchors(mid_means.cpu().numpy(), mid_alpha.cpu().numpy(), up)
    e1, e2 = orbit_basis(cameras, centroid, up)

    parameters, probe_index = sweep_parameters(
        by_label[pair[0]], by_label[pair[1]], centroid, up, e1, e2,
        len(frames), by_label.get(holdout_label) if holdout_label else None)
    weights = angular_loss_weights(parameters, lambda_gt)

    focal = float(reference_focal * (res / reference_width) * zoom)
    principal = res / 2.0
    intrinsics = torch.tensor([[focal, 0.0, principal], [0.0, focal, principal], [0.0, 0.0, 1.0]],
                              dtype=torch.float32, device=device)[None]
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, (frame_number, (azimuth, elevation, radius)) in enumerate(zip(frames, parameters)):
        camera_time = frame_time(cameras, frame_number, fps)
        model_time = header["time_min"] + camera_time * scale
        dt = model_time - t_center
        means = xyz + velocity * dt[:, None]
        if accel is not None:
            means = means + accel * (dt * dt)[:, None]
        alpha = torch.sigmoid(op_logit) * torch.exp(-0.5 * (dt / t_sigma) ** 2)

        position = position_at(centroid, up, e1, e2, azimuth, elevation, radius)
        w2c = lookat_w2c(position, centroid, -up)
        viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)[None]
        with torch.no_grad():
            colors, alphas, _ = rasterization(means, quats, scales, alpha, shs, viewmat,
                                              intrinsics, res, res, sh_degree=sh_degree,
                                              render_mode="RGB")
        rgb = (colors[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        coverage = (alphas[0, ..., 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(np.dstack([rgb, coverage]), mode="RGBA").save(
            out_dir / f"sweep_{index:04d}.png")

        records.append({"idx": index, "frame": frame_number, "time": camera_time,
                        "model_time": float(model_time), "azim": azimuth, "elev": elevation,
                        "radius": radius, "loss_weight": weights[index],
                        "w2c": w2c.tolist(), "position": position.tolist()})

    # `up` is recorded because it cannot be recovered exactly from the poses:
    # lookat_w2c orthogonalizes the down vector against forward, so an elevated
    # camera's image-down axis is tilted off world up by its elevation angle.
    meta = {"pair": list(pair), "res": res, "zoom": zoom, "fps": fps, "lambda_gt": lambda_gt,
            "model": str(model), "fl_x": focal, "fl_y": focal, "cx": principal, "cy": principal,
            "target": centroid.tolist(), "up": up.tolist(), "frames": records}

    pinned = {"real_first": (by_label[pair[0]], 0),
              "real_last": (by_label[pair[1]], len(records) - 1)}
    if probe_index is not None:
        pinned["probe_real"] = (by_label[holdout_label], probe_index)

    warped = {}
    for name, (camera, index) in pinned.items():
        frame_number = records[index]["frame"]
        image_path, intrinsics_real = resolve_real_view(camera, transforms, images_dir, frame_number)
        if image_path is None:
            print(f"  WARNING: no real image on disk for camera {camera['label']!r} at frame "
                  f"{frame_number} -- {name}.png not written; the repair pass will have nothing "
                  "to pin this end to", file=sys.stderr)
            continue
        homography = homography_to_frustum(camera, np.array(records[index]["w2c"]), focal,
                                           principal, intrinsics_real)
        warp_real_image(image_path, homography, res).save(out_dir / f"{name}.png")
        warped[name] = {"camera": camera["label"], "idx": index, "frame": frame_number,
                        "source": str(image_path)}

    meta["real_endpoints"] = warped
    if probe_index is not None:
        meta["probe"] = {"camera": holdout_label, "idx": probe_index,
                         "image": "probe_real.png" if "probe_real" in warped else None,
                         "render": f"sweep_{probe_index:04d}.png"}

    (out_dir / "cameras.json").write_text(json.dumps(meta, indent=1))
    print(f"{out_dir.name}: {len(records)} frames"
          f"{' POINT-RENDER' if point_render > 0 else ''}, cameras {pair[0]}->{pair[1]}, "
          f"azimuth {parameters[0][0]:.1f} -> {parameters[-1][0]:.1f} deg at {res}px, "
          f"t {records[0]['time']:.3f}-{records[-1]['time']:.3f}s"
          + (f", probe {holdout_label!r} at index {probe_index}" if probe_index is not None else "")
          + (f", pinned {sorted(warped)}" if warped else ", NO real endpoints"))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path,
                    help="trained 4D model: a .sogst archive or a 4D interchange PLY")
    ap.add_argument("--transforms", required=True, type=Path,
                    help="transforms.json for the real rig (poses, intrinsics, per-frame times)")
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
                    help="limit to a capture-frame range; one sweep frame per capture frame, so "
                         "this sets the clip length (Wan needs 4k+1: 17, 21, 33...)")
    ap.add_argument("--fps", type=float, default=29.97,
                    help="fallback frame rate, used only when the transforms records no per-frame time")
    ap.add_argument("--time_scale", type=float, default=None,
                    help="rescale camera time into the model's range; see eval_render.py")
    ap.add_argument("--res", type=int, default=1024,
                    help="square render size; match the video model's native resolution")
    ap.add_argument("--zoom", type=float, default=1.6, help="frustum tightening vs the real cameras")
    ap.add_argument("--merge_deg", type=float, default=3.0,
                    help="with --adjacent_pairs, skip pairs closer than this in azimuth")
    ap.add_argument("--lambda_gt", type=float, default=LAMBDA_GT,
                    help="angular loss-weight strength written into cameras.json: a frame on a real "
                         "camera is worth (1 + lambda_gt) times one at the arc's midpoint. 0 gives "
                         "every frame equal weight")
    ap.add_argument("--point_render", type=float, default=0.0,
                    help="render as a POINT CLOUD instead of splats, collapsing every Gaussian "
                         "to a dot of this world size (try 0.002 at our scale). This is the "
                         "style Uni3C expects for its guidance video")
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
        header, fields = load_4d_model(args.model)
        means = np.stack([fields["x"], fields["y"], fields["z"]], axis=1)
        opacity = 1.0 / (1.0 + np.exp(-np.asarray(fields["opacity"], dtype=np.float64)))
        centroid, _ = subject_anchors(means, opacity, up)
        e1, e2 = orbit_basis(cameras, centroid, up)
        pairs = adjacent_pairs(cameras, centroid, up, e1, e2, args.merge_deg)
        print(f"{len(pairs)} adjacent pairs at >= {args.merge_deg} deg separation")

    for pair in pairs:
        render_pair_sweep(
            args.model, args.transforms, args.out_dir / f"{pair[0]}_to_{pair[1]}",
            pair=pair, frame_range=args.frames, fps=args.fps, res=args.res, zoom=args.zoom,
            holdout_label=args.holdout_label, images_dir=args.images_dir,
            lambda_gt=args.lambda_gt, time_scale=args.time_scale,
            point_render=args.point_render)
    return 0


if __name__ == "__main__":
    sys.exit(main())
