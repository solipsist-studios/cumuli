#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
camera_rig_spec.py - resolve a JSON camera-rig spec into concrete cameras.

Pure Python plus numpy, no bpy, so the same geometry runs inside Blender's
bundled interpreter and in the `cumuli` env for tests and the render driver.

A rig spec describes where cameras go and what they see. Specs are small
enough to read, version, and diff, which is the point: comparing camera
configurations means comparing two files, not two hand-built scenes.

    {
      "name": "ring16",
      "layout": "rings",
      "target": "subject_center",
      "rings": [
        {"count": 8, "radius": {"subject_heights": 1.8}, "height": {"subject_fraction": 0.55}},
        {"count": 8, "radius": {"subject_heights": 1.8}, "height": {"subject_fraction": 1.1},
         "azimuth_offset_deg": 22.5}
      ],
      "resolution": [1920, 1680],
      "camera_model": "PINHOLE",
      "intrinsics": {"lens_mm": 32, "sensor_width_mm": 36},
      "eval": {"count": 4},
      "lights": {"count": 12, "power_w": 100}
    }

--------------------------------------------------------------------------
RING OPTIONS
--------------------------------------------------------------------------
Each entry of `rings` takes `count` and `radius`, plus:

    height                 one altitude for the whole ring
    heights                a CYCLE of altitudes, camera i taking
                           heights[i % len(heights)]. Staggers elevation
                           while leaving the azimuths evenly spaced.
    azimuth_offset_deg     azimuth of the first camera (default 0)
    azimuth_span_deg       makes the ring an ARC of this angular width,
                           with cameras spread inclusive of both ends,
                           rather than a closed circle
    azimuth_centre_deg     centres an arc on this azimuth, instead of
                           starting it at azimuth_offset_deg. Only valid
                           together with azimuth_span_deg.

A ring of `count: 1` is a single camera at its offset, which is how a lone
rear camera is written.

--------------------------------------------------------------------------
SUBJECT-RELATIVE VALUES
--------------------------------------------------------------------------
Any length or height may be a literal number or a small object resolved
against the scene manifest that prepare_blender_scene.py writes:

    {"subject_heights": k}   k times the subject's height (a LENGTH, for
                             radius and similar distances)
    {"subject_fraction": k}  the world Z at k of the way up the subject
                             from its floor (a HEIGHT, for camera altitude;
                             k may exceed 1 to sit above the subject)

Vectors accept the strings "subject_center" (the bounding box centre, at
mid height) and "subject_floor_center" (the same point at floor level), or
a literal [x, y, z].

This is what lets one rig spec serve subjects of different sizes, which is
the difference between a camera testbed and a tool that produces splats of
whatever character is loaded.

--------------------------------------------------------------------------
CONVENTIONS
--------------------------------------------------------------------------
Positions and matrices are in Blender world space, Z up. Cameras look down
their local -Z with +Y up, which is Blender's camera convention and also
OpenGL's, so converting a camera pose to the nerfstudio/OpenGL world
convention every downstream stage expects is a change of WORLD axes only:

    (x, y, z)_blender -> (x, z, -y)_opengl

`blender_to_opengl_world` applies it. Verified against the March Ariana
export: its Camera_0002 sits at Blender (2, 0, 2) and appears in
transforms_gt.json at (2, 2, 0).

Camera orientation reproduces Blender's `direction.to_track_quat('-Z','Y')`
exactly, which is what camera_cage.py used, so a ported cage spec places
cameras where the original script placed them (agreement measured at 7e-7
over random directions, including the degenerate straight-up and
straight-down cases).
"""

import json
import math
from pathlib import Path

import numpy as np

BLENDER_TO_OPENGL = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

WORLD_UP = np.array([0.0, 0.0, 1.0])

LAYOUTS = ("cage", "rings", "explicit")

SPEC_KEYS = {
    "name", "layout", "target", "center", "cage", "rings", "cameras",
    "resolution", "camera_model", "intrinsics", "calibration_pkl", "eval",
    "lights", "description",
}

DEFAULT_EVAL_COUNT = 4


class RigSpecError(ValueError):
    """Raised for a spec that cannot be resolved into cameras."""


class RigCamera:
    """One resolved camera. Plain attributes so Blender's interpreter and
    the test env agree without importing anything exotic."""

    __slots__ = ("name", "label", "role", "position", "look_at", "c2w_blender",
                 "resolution", "calib", "ring")

    def __init__(self, name, label, role, position, look_at, c2w_blender,
                 resolution, calib, ring=None):
        self.name = name
        self.label = label
        self.role = role
        self.position = np.asarray(position, dtype=np.float64)
        self.look_at = np.asarray(look_at, dtype=np.float64)
        self.c2w_blender = np.asarray(c2w_blender, dtype=np.float64)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.calib = calib
        self.ring = ring

    @property
    def c2w_opengl(self):
        return BLENDER_TO_OPENGL @ self.c2w_blender

    def as_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "role": self.role,
            "position": self.position.tolist(),
            "look_at": self.look_at.tolist(),
            "c2w_blender": self.c2w_blender.tolist(),
            "c2w_opengl": self.c2w_opengl.tolist(),
            "resolution": list(self.resolution),
            "ring": self.ring,
        }


# ----------------------------------------------------------------- geometry
def look_at_matrix(position, target, up=WORLD_UP):
    """Blender camera-to-world matrix looking from `position` at `target`.

    Reproduces `(target - position).to_track_quat('-Z', 'Y')`: the camera's
    -Z axis points at the target and its +Y axis is as close to world up as
    the view direction allows. Falls back to a different up vector when the
    view is within a whisker of vertical, which is exactly the case a top
    camera in a cage rig hits."""
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - position
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise RigSpecError(f"camera at {position.tolist()} looks at its own position")
    forward = forward / norm

    z_axis = -forward
    up_vec = np.asarray(up, dtype=np.float64)
    x_axis = np.cross(up_vec, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        up_vec = np.array([0.0, 1.0, 0.0]) if abs(z_axis[2]) > 0.9 else WORLD_UP
        x_axis = np.cross(up_vec, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    c2w = np.eye(4)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = position
    return c2w


def blender_to_opengl_world(matrix):
    """Blender world (Z up) to nerfstudio/OpenGL world (Y up)."""
    return BLENDER_TO_OPENGL @ np.asarray(matrix, dtype=np.float64)


def label_width(count):
    """Fixed label width so labels sort in camera order.

    build_flat_dataset.py formats labels as `f"{i:02d}"`, which stops being
    fixed width past 99 cameras and then sorts wrongly. This widens instead.
    Rigs above 100 cameras therefore label consistently here but must not be
    round-tripped through build_flat_dataset.py without the same fix."""
    return max(2, len(str(max(int(count) - 1, 0))))


def make_labels(count, prefix=""):
    width = label_width(count)
    return [f"{prefix}{i:0{width}d}" for i in range(int(count))]


# ------------------------------------------------- subject-relative resolve
def subject_metrics(manifest):
    """Centre, floor, and height of the subject from a scene manifest."""
    if not manifest:
        return None
    bbox = manifest.get("subject_bbox")
    if not bbox:
        return None
    lo = np.asarray(bbox["min"], dtype=np.float64)
    hi = np.asarray(bbox["max"], dtype=np.float64)
    height = float(hi[2] - lo[2])
    centre = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0,
                       lo[2] + height / 2.0])
    return {"min": lo, "max": hi, "height": height, "centre": centre,
            "floor": float(lo[2])}


def resolve_scalar(value, manifest, what="value"):
    """A literal number, or a subject-relative length or height."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, dict):
        raise RigSpecError(f"{what}: expected a number or a subject-relative "
                           f"object, got {value!r}")
    metrics = subject_metrics(manifest)
    if metrics is None:
        raise RigSpecError(
            f"{what} is subject-relative ({value!r}) but no scene manifest "
            "with a subject_bbox was supplied. Run prepare_blender_scene.py "
            "first, or use literal numbers in the spec.")
    if "subject_heights" in value:
        return float(value["subject_heights"]) * metrics["height"]
    if "subject_fraction" in value:
        return metrics["floor"] + float(value["subject_fraction"]) * metrics["height"]
    raise RigSpecError(f"{what}: unknown subject-relative form {sorted(value)}")


def resolve_vector(value, manifest, what="point"):
    if isinstance(value, str):
        metrics = subject_metrics(manifest)
        if metrics is None:
            raise RigSpecError(
                f"{what} is {value!r} but no scene manifest with a "
                "subject_bbox was supplied")
        if value == "subject_center":
            return metrics["centre"].copy()
        if value == "subject_floor_center":
            return np.array([metrics["centre"][0], metrics["centre"][1],
                             metrics["floor"]])
        raise RigSpecError(f"{what}: unknown named point {value!r}")
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise RigSpecError(f"{what}: expected 3 numbers, got {value!r}")
    return arr


# --------------------------------------------------------------- intrinsics
def pickle_calib_loader(spec_path):
    """Loader for a spec's `calibration_pkl`, resolving relative paths.

    A relative path is tried against the spec's own directory first, then
    the repository root, so a spec can name `configs/rigs/gopro.pkl` from
    anywhere. Kept here rather than duplicated in each caller."""
    import pickle

    def load(path):
        candidates = [Path(path).expanduser()]
        if not candidates[0].is_absolute():
            spec_dir = Path(spec_path).expanduser().resolve().parent
            candidates = [spec_dir / path,
                          Path(__file__).resolve().parent.parent / path]
        for cand in candidates:
            if cand.is_file():
                with open(cand, "rb") as f:
                    return pickle.load(f)
        raise RigSpecError(
            f"calibration_pkl {path!r} not found. Tried: "
            f"{[str(c) for c in candidates]}")

    return load


def resolve_calibration(spec, resolution, calib_loader=None):
    """The calibration every camera in the rig shares.

    Either `calibration_pkl` (a real lens, loaded through `calib_loader`) or
    `intrinsics` with a focal length in millimetres. Returns a calibration
    dict in the pkl schema so both paths look the same downstream."""
    w, h = int(resolution[0]), int(resolution[1])
    model = spec.get("camera_model", "PINHOLE")

    pkl = spec.get("calibration_pkl")
    if pkl:
        if calib_loader is None:
            raise RigSpecError(
                "spec sets calibration_pkl but no loader was provided")
        calib = dict(calib_loader(pkl))
        calib.setdefault("model", model)
        return calib

    intr = spec.get("intrinsics") or {}
    sensor_width = float(intr.get("sensor_width_mm", 36.0))
    if "lens_mm" in intr:
        focal_px = float(intr["lens_mm"]) / sensor_width * w
    elif "focal_px" in intr:
        focal_px = float(intr["focal_px"])
    elif "fov_deg" in intr:
        focal_px = (w / 2.0) / math.tan(math.radians(float(intr["fov_deg"])) / 2.0)
    else:
        raise RigSpecError(
            "spec needs `calibration_pkl`, or `intrinsics` with one of "
            "lens_mm, focal_px, or fov_deg")
    if model != "PINHOLE":
        raise RigSpecError(
            f"camera_model {model!r} needs a calibration_pkl: distortion "
            "coefficients cannot be invented from a focal length")
    cx = float(intr.get("cx", w / 2.0))
    cy = float(intr.get("cy", h / 2.0))
    return {
        "camera_matrix": np.array([[focal_px, 0.0, cx],
                                   [0.0, focal_px, cy],
                                   [0.0, 0.0, 1.0]]),
        "distortion_coefficients": np.zeros(5),
        "image_size": (w, h),
        "model": "PINHOLE",
        "sensor_width_mm": sensor_width,
    }


# ------------------------------------------------------------ rig generators
def _cage_positions(cfg, centre, manifest):
    """Port of camera_cage.py: azimuth columns by elevation rows on a
    sphere, truncated to a height band, with the per-row azimuth stagger.

    min_height and max_height are world Z values, as in the original script,
    and are converted here to offsets from the sphere centre."""
    n_theta = int(cfg.get("theta", cfg.get("num_theta", 12)))
    n_phi = int(cfg.get("phi", cfg.get("num_phi", 4)))
    if n_theta < 1 or n_phi < 1:
        raise RigSpecError("cage needs at least one theta column and one phi row")
    radius = resolve_scalar(cfg.get("radius", 3.0), manifest, "cage.radius")
    if radius <= 0.0:
        raise RigSpecError("cage.radius must be positive")

    z_max = resolve_scalar(cfg["max_height"], manifest, "cage.max_height") - centre[2] \
        if "max_height" in cfg else radius
    z_min = resolve_scalar(cfg["min_height"], manifest, "cage.min_height") - centre[2] \
        if "min_height" in cfg else 0.0

    safe_max = min(z_max, radius)
    safe_min = max(z_min, -radius)
    if safe_min >= safe_max:
        raise RigSpecError(
            f"cage height band is empty: min {z_min:.3f} max {z_max:.3f} "
            f"relative to centre z {centre[2]:.3f} with radius {radius:.3f}")

    phi_start = math.acos(safe_max / radius)
    phi_end = math.acos(safe_min / radius)
    phi_step = (phi_end - phi_start) / n_phi
    stagger = bool(cfg.get("stagger", True))

    out = []
    for j in range(n_phi):
        phi = phi_start + (j + 1) * phi_step
        for i in range(n_theta):
            theta = i * (2.0 * math.pi / n_theta)
            if stagger:
                theta += j * math.pi / (n_theta * 2.0)
            out.append((centre + np.array([
                radius * math.sin(phi) * math.cos(theta),
                radius * math.sin(phi) * math.sin(theta),
                radius * math.cos(phi)]), j))
    return out


def _ring_heights(ring, centre, manifest, label):
    """The height, or cycle of heights, cameras in this ring sit at.

    `heights` staggers elevation WITHOUT disturbing azimuth: camera i takes
    heights[i % len(heights)], so twelve cameras at an even thirty degrees
    can alternate between two or three altitudes. Expressing the same shape
    as several rings would work, but it spaces each ring's cameras
    independently, and the point of a stagger is that the azimuths stay
    even."""
    if "heights" in ring:
        values = ring["heights"]
        if not values:
            raise RigSpecError(f"{label}.heights is empty")
        return [resolve_scalar(v, manifest, f"{label}.heights[{n}]")
                for n, v in enumerate(values)]
    return [resolve_scalar(ring.get("height", centre[2]), manifest,
                           f"{label}.height")]


def _ring_angles(ring, count, label):
    """Camera azimuths in radians.

    A closed ring divides the full circle, so the last camera does not
    land on the first. An arc (`azimuth_span_deg`) spreads its cameras
    INCLUSIVE of both ends instead, which is what makes an eleven-camera
    front arc reach both extremes of the span rather than stopping one
    step short."""
    has_centre = "azimuth_centre_deg" in ring
    offset = math.radians(float(ring.get("azimuth_offset_deg", 0.0)))

    if "azimuth_span_deg" not in ring:
        if has_centre:
            raise RigSpecError(
                f"{label}.azimuth_centre_deg only means something with "
                "azimuth_span_deg: a closed ring has no centre. Use "
                "azimuth_offset_deg to rotate it.")
        step = 2.0 * math.pi / count
        return [offset + i * step for i in range(count)]

    span = math.radians(float(ring["azimuth_span_deg"]))
    step = span / (count - 1) if count > 1 else 0.0
    if has_centre:
        start = math.radians(float(ring["azimuth_centre_deg"])) - span / 2.0
    else:
        start = offset
    return [start + i * step for i in range(count)]


def _ring_positions(rings, centre, manifest, key="rings"):
    out = []
    for idx, ring in enumerate(rings):
        label = f"{key}[{idx}]"
        count = int(ring.get("count", 0))
        if count < 1:
            raise RigSpecError(f"{label}.count must be at least 1")
        radius = resolve_scalar(ring.get("radius", 3.0), manifest,
                                f"{label}.radius")
        heights = _ring_heights(ring, centre, manifest, label)
        for i, angle in enumerate(_ring_angles(ring, count, label)):
            out.append((np.array([centre[0] + radius * math.cos(angle),
                                  centre[1] + radius * math.sin(angle),
                                  heights[i % len(heights)]]), idx))
    return out


def _explicit_positions(cameras, manifest):
    out = []
    for idx, entry in enumerate(cameras):
        pos = resolve_vector(entry["position"], manifest, f"cameras[{idx}].position")
        out.append((pos, entry.get("look_at")))
    return out


# --------------------------------------------------------------- public API
def load_spec(path):
    spec = json.loads(Path(path).read_text())
    validate_spec(spec)
    return spec


def validate_spec(spec):
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise RigSpecError(
            f"unknown rig spec key(s): {sorted(unknown)} -- valid keys are "
            f"{sorted(SPEC_KEYS)}")
    layout = spec.get("layout")
    if layout not in LAYOUTS:
        raise RigSpecError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if layout == "cage" and "cage" not in spec:
        raise RigSpecError("layout 'cage' needs a `cage` block")
    if layout == "rings" and not spec.get("rings"):
        raise RigSpecError("layout 'rings' needs a non-empty `rings` list")
    if layout == "explicit" and not spec.get("cameras"):
        raise RigSpecError("layout 'explicit' needs a non-empty `cameras` list")
    res = spec.get("resolution")
    if not res or len(res) != 2 or int(res[0]) < 1 or int(res[1]) < 1:
        raise RigSpecError("spec needs a `resolution` of [width, height]")
    return spec


def resolve_rig(spec, manifest=None, calib_loader=None):
    """Resolve a spec into training cameras, eval cameras, and lights.

    Returns {"train": [RigCamera], "eval": [RigCamera], "lights": [dict],
             "target": [x,y,z], "centre": [x,y,z], "calibration": dict}."""
    validate_spec(spec)
    resolution = (int(spec["resolution"][0]), int(spec["resolution"][1]))
    calib = resolve_calibration(spec, resolution, calib_loader)

    target = resolve_vector(spec.get("target", [0.0, 0.0, 0.0]), manifest, "target")
    centre = resolve_vector(spec["center"], manifest, "center") \
        if "center" in spec else target.copy()

    layout = spec["layout"]
    if layout == "cage":
        placed = _cage_positions(spec["cage"], centre, manifest)
        entries = [(pos, target, ring) for pos, ring in placed]
    elif layout == "rings":
        placed = _ring_positions(spec["rings"], centre, manifest)
        entries = [(pos, target, ring) for pos, ring in placed]
    else:
        entries = []
        for pos, look in _explicit_positions(spec["cameras"], manifest):
            aim = resolve_vector(look, manifest, "cameras[].look_at") \
                if look is not None else target
            entries.append((pos, aim, None))

    labels = make_labels(len(entries))
    train = [
        RigCamera(name=f"Camera_{label}", label=label, role="train",
                  position=pos, look_at=aim,
                  c2w_blender=look_at_matrix(pos, aim),
                  resolution=resolution, calib=calib, ring=ring)
        for label, (pos, aim, ring) in zip(labels, entries)
    ]
    if not train:
        raise RigSpecError("spec resolved to zero cameras")

    evals = _resolve_eval(spec, manifest, centre, target, resolution, calib, train)
    lights = _resolve_lights(spec.get("lights"), manifest, centre, target)
    return {"train": train, "eval": evals, "lights": lights,
            "target": target.tolist(), "centre": centre.tolist(),
            "calibration": calib}


def _resolve_eval(spec, manifest, centre, target, resolution, calib, train):
    """The scored novel-view cameras.

    These are deliberately NOT rig cameras. Holding out a rig camera scores
    a different view for every rig, so the numbers cannot be compared across
    configurations; a fixed eval ring can. The default azimuth offset puts
    eval views half a step off the first training ring, which is the hardest
    place for the model rather than the easiest."""
    cfg = spec.get("eval")
    if cfg is None:
        cfg = {}
    if cfg.get("count", DEFAULT_EVAL_COUNT) == 0:
        return []

    count = int(cfg.get("count", DEFAULT_EVAL_COUNT))
    first_ring = spec["rings"][0] if spec.get("layout") == "rings" and spec.get("rings") else {}
    radius = resolve_scalar(
        cfg.get("radius", first_ring.get("radius", 3.0)), manifest, "eval.radius")
    # A staggered ring has `heights` rather than `height`; fall back to its
    # first altitude. Comparable runs should set eval explicitly anyway, so
    # that every rig is scored from exactly the same viewpoints.
    ring_height = first_ring.get("height")
    if ring_height is None and first_ring.get("heights"):
        ring_height = first_ring["heights"][0]
    if ring_height is None:
        ring_height = centre[2]
    height = resolve_scalar(cfg.get("height", ring_height), manifest, "eval.height")
    if "azimuth_offset_deg" in cfg:
        offset_deg = float(cfg["azimuth_offset_deg"])
    else:
        ring_count = int(first_ring.get("count", count)) or count
        offset_deg = 180.0 / ring_count
    eval_res = tuple(int(v) for v in cfg["resolution"]) if "resolution" in cfg else resolution

    placed = _ring_positions(
        [{"count": count, "radius": radius, "height": height,
          "azimuth_offset_deg": offset_deg}], centre, manifest, key="eval")
    labels = make_labels(count, prefix="e")
    return [
        RigCamera(name=f"Eval_{label}", label=label, role="eval", position=pos,
                  look_at=target, c2w_blender=look_at_matrix(pos, target),
                  resolution=eval_res, calib=calib, ring=None)
        for label, (pos, _) in zip(labels, placed)
    ]


def _resolve_lights(cfg, manifest, centre, target):
    """Bar-light ring, ported from camera_cage.py.

    Lights sit between the camera columns rather than behind them, which is
    the half-step offset the original script applied."""
    if not cfg:
        return []
    count = int(cfg.get("count", 12))
    if count < 1:
        return []
    radius = resolve_scalar(cfg.get("radius", 4.0), manifest, "lights.radius")
    radius += float(cfg.get("radius_offset", 0.0))
    height = resolve_scalar(cfg.get("height", target[2]), manifest, "lights.height")
    power = float(cfg.get("power_w", 100.0))
    bar_h = float(cfg.get("bar_height", 2.0))
    bar_w = float(cfg.get("bar_width", 0.2))

    lights = []
    for i in range(count):
        angle = i * (2.0 * math.pi / count) + math.pi / count
        pos = np.array([centre[0] + radius * math.cos(angle),
                        centre[1] + radius * math.sin(angle),
                        height])
        aim = np.array([target[0], target[1], height])
        lights.append({
            "name": f"Bar_Light_{i + 1}",
            "position": pos.tolist(),
            "c2w_blender": look_at_matrix(pos, aim if np.linalg.norm(aim - pos) > 1e-6
                                          else target).tolist(),
            "energy": power,
            "size": bar_w,
            "size_y": bar_h,
            "shape": "RECTANGLE",
        })
    return lights
