#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""rig_geometry.py - camera-rig geometry shared by the novel-view stages.

render_pair_sweep.py, project_skeleton_conditioning.py and
build_frame_dataset.py all need the same things: read a rig out of a
transforms.json, convert between this project's camera conventions, and aim a
synthetic camera at a subject. This module is where those live so the three do
not each carry a copy.

Camera conventions follow build_colmap_sparse.py and eval_render.py exactly:
transforms.json holds nerfstudio/OpenGL c2w, and everything downstream works in
COLMAP-convention w2c, reached by negating the Y and Z axis columns and
inverting. Any drift here silently mis-places every synthetic view, so the
conversion is tested against build_colmap_sparse's own implementation.

conda env: cumuli (numpy only).
"""

import json
from pathlib import Path

import numpy as np

HIGH_OPACITY = 0.5           # opacity above which a splat counts toward the subject centroid
MIN_CENTROID_SPLATS = 100    # below this many high-opacity splats, fall back to the full cloud
HEAD_ANCHOR_PERCENTILE = 85  # head = splats in the top 15% of subject extent along the up axis
MIN_HEAD_SPLATS = 10         # below this many head splats, fall back to a fixed offset up the axis
DEFAULT_HEAD_OFFSET = 0.6    # world units up from the centroid when the head cannot be located


def opengl_c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """nerfstudio/OpenGL c2w -> COLMAP-convention w2c (negate Y/Z axis columns,
    invert). Matches build_colmap_sparse.opengl_c2w_to_colmap_w2c and the inline
    conversion in eval_render.load_cameras."""
    m = c2w.copy()
    m[:3, 1:3] *= -1
    return np.linalg.inv(m)


def lookat_w2c(eye: np.ndarray, target: np.ndarray, down: np.ndarray) -> np.ndarray:
    """World-to-camera matrix for a camera at `eye` looking at `target`, with
    `down` giving the +Y (image-down) direction to orthogonalize against.

    Note that `down` is orthogonalized against forward, so the resulting image-
    down axis is NOT `down` itself for an elevated camera; it is tilted by the
    elevation angle. Callers needing the true world up must carry it separately
    rather than recovering it from the returned matrix."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    dn = down - (down @ forward) * forward
    dn = dn / np.linalg.norm(dn)
    right = np.cross(dn, forward)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, dn, forward, eye
    return np.linalg.inv(c2w)


def camera_label_from_path(file_path: str) -> str:
    """Camera label implied by a file_path's first path component, for transforms
    that carry no explicit camera_label (`cam01/frame_00042` -> `cam01`)."""
    parts = [p for p in file_path.replace("\\", "/").split("/") if p]
    return parts[0] if len(parts) > 1 else ""


def trailing_number(file_path: str) -> int | None:
    """The trailing integer of a file_path's stem (`frame_00042.jpg` -> 42), or
    None when it ends in no digits. This keys each camera's per-instant views."""
    stem = file_path.replace("\\", "/").split("/")[-1].split(".")[0]
    digits = ""
    for char in reversed(stem):
        if not char.isdigit():
            break
        digits = char + digits
    return int(digits) if digits else None


def load_rig(transforms_path: Path):
    """Real camera geometry from a transforms.json: per-camera label, centre, 3x4
    projection, w2c, 3x3 intrinsics and source file_path, plus the rig-average up
    axis and the reference focal and width used to scale a synthetic frustum.

    Accepts both layouts this project has produced. The documented one is
    build_flat_dataset.py's: one entry per camera, `camera_label` set, `w` on the
    frame. The 4D layout (an entry per camera AND timestamp, `w`/`h` at the top
    level, the camera implied by the file_path's first path component) collapses
    to one entry per camera, since a static rig's pose does not depend on time.

    What DOES depend on time is the principal point. Datasets that crop each
    frame around the moving subject shift `cx`/`cy` frame to frame while the
    focal and the pose stay fixed (measured on 260529-171110: one focal, 84
    distinct `cx` values over 146 frames, and `cy` moving 249 px between frame 1
    and frame 58). So each camera keeps `per_frame`, mapping frame number to that
    frame's file_path and intrinsics; a caller warping a real photo must use the
    matching frame's intrinsics or it misaligns by that shift. The camera-level
    `intrinsics` is the first frame's, for callers needing only a rough frustum.
    """
    data = json.loads(Path(transforms_path).read_text())
    frames = data["frames"]
    if not frames:
        raise ValueError(f"{transforms_path} has no frames")

    c2ws = [np.array(fr["transform_matrix"], dtype=np.float64) for fr in frames]
    up = np.mean([m[:3, 1] for m in c2ws], axis=0)
    up = up / np.linalg.norm(up)

    cameras: dict = {}
    for index, (fr, c2w) in enumerate(zip(frames, c2ws)):
        file_path = str(fr.get("file_path", ""))
        label = str(fr.get("camera_label", "")) or camera_label_from_path(file_path) or str(index)
        intrinsics = np.array([[fr.get("fl_x", data.get("fl_x")), 0.0, fr.get("cx", data.get("cx"))],
                               [0.0, fr.get("fl_y", data.get("fl_y")), fr.get("cy", data.get("cy"))],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
        if label not in cameras:
            w2c = opengl_c2w_to_w2c(c2w)
            cameras[label] = {
                "label": label,
                "center": c2w[:3, 3],
                "projection": intrinsics @ w2c[:3, :],
                "w2c": w2c,
                "intrinsics": intrinsics,
                "file_path": file_path,
                "per_frame": {},
            }
        cameras[label]["per_frame"][trailing_number(file_path)] = {
            "file_path": file_path, "intrinsics": intrinsics,
            # the dataset's own timestamp for this capture frame. A sweep
            # evaluates the 4D model at this time rather than deriving one from
            # an assumed fps, so the two never drift apart.
            "time": float(fr["time"]) if "time" in fr else None,
        }

    reference = frames[0]
    focal = reference.get("fl_x", data.get("fl_x"))
    width = reference.get("w", data.get("w"))
    if focal is None:
        raise ValueError(f"{transforms_path}: no fl_x on the first frame or at the top level")
    if width is None:
        raise ValueError(f"{transforms_path}: no image width on the first frame or at the top level")
    return list(cameras.values()), up, float(focal), float(width)


def subject_anchors(means: np.ndarray, opacity: np.ndarray, up: np.ndarray):
    """(subject centroid, head anchor) from a splat cloud. The centroid is the
    median of high-opacity splats; the head is the median of those in the top 15%
    of the subject's extent along the up axis."""
    high = opacity > HIGH_OPACITY
    centroid = np.median(means[high], axis=0) if high.sum() > MIN_CENTROID_SPLATS \
        else np.median(means, axis=0)

    high_means = means[high]
    if len(high_means) > 20:
        along_up = (high_means - centroid[None, :]) @ up
        head_sel = along_up >= np.percentile(along_up, HEAD_ANCHOR_PERCENTILE)
        if head_sel.sum() > MIN_HEAD_SPLATS:
            return centroid, np.median(high_means[head_sel], axis=0)
    return centroid, centroid + up * DEFAULT_HEAD_OFFSET


def orbit_basis(cameras: list, centroid: np.ndarray, up: np.ndarray):
    """(e1, e2): a horizontal right-handed frame with e1 pointing from the subject
    toward the first real camera, so azimuths are directly comparable to the
    rig's."""
    first = cameras[0]["center"] - centroid
    e1 = first - (first @ up) * up
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def spherical(center: np.ndarray, target: np.ndarray, up: np.ndarray,
              e1: np.ndarray, e2: np.ndarray) -> tuple:
    """(azimuth deg, elevation deg, radius) of `center` about `target`."""
    offset = center - target
    radius = float(np.linalg.norm(offset))
    elevation = float(np.degrees(np.arcsin((offset @ up) / radius)))
    azimuth = float(np.degrees(np.arctan2(offset @ e2, offset @ e1)))
    return azimuth, elevation, radius


def position_at(target: np.ndarray, up: np.ndarray, e1: np.ndarray, e2: np.ndarray,
                azimuth: float, elevation: float, radius: float) -> np.ndarray:
    """Inverse of `spherical`. Round-trips a real camera centre exactly, which is
    what lets a swept frame sit at a real camera and warp its photo in through an
    exact homography."""
    ring = np.cos(np.radians(azimuth)) * e1 + np.sin(np.radians(azimuth)) * e2
    return target + radius * (np.cos(np.radians(elevation)) * ring
                              + np.sin(np.radians(elevation)) * up)


def shortest_arc(start_deg: float, end_deg: float) -> float:
    """Signed azimuth delta taking the short way round, so a pair straddling the
    +/-180 wrap sweeps the gap between them rather than the long way through the
    whole rig. At exactly half a turn either sign is correct."""
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0
