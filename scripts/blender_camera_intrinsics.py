#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
blender_camera_intrinsics.py - convert OpenCV camera calibrations to Blender
camera settings and back.

This module is pure Python plus numpy. It imports no bpy, so it runs both
inside Blender's bundled interpreter and in the `cumuli` env (tests,
render_blender_rig.py, verify_blender_intrinsics.py).

Three camera models are supported, named exactly as the calibration pkls
name them (see deps/camera-calibration/offline_undistort.py):

  PINHOLE          zero distortion. Blender PERSP camera, exact.
  OPENCV           radial/tangential distortion. Blender renders a PERSP
                   camera, which applies NO distortion, so the rendered
                   frames are geometrically undistorted by construction.
                   The coefficients are carried for provenance only.
  OPENCV_FISHEYE   equidistant fisheye with 4 coefficients. Blender PANO
                   camera with panorama_type FISHEYE_LENS_POLYNOMIAL.

--------------------------------------------------------------------------
THE FISHEYE POLYNOMIAL CONVENTION
--------------------------------------------------------------------------
Blender's documentation describes the fisheye polynomial loosely enough
that getting it right by reading is unreliable, and getting it wrong warps
every rendered frame silently. The convention below was measured directly
against Blender 5.1.2 (scripts/verify_blender_intrinsics.py reproduces the
measurement, and tests/unit/test_blender_camera_intrinsics.py pins it):

    theta = | k0 + k1*r + k2*r^2 + k3*r^3 + k4*r^4 |

where `theta` is the ray angle from the optical axis in radians and `r` is
the distance from the image centre measured ON THE SENSOR IN MILLIMETRES:

    r_mm = r_px * sensor_fit_mm / resolution_along_fit_axis

The polynomial maps RADIUS TO ANGLE, not angle to radius. Three ablations
established this, each rendering emissive markers at known angles:

  - Doubling `sensor_width` halved every marker's pixel radius, so `r`
    carries physical sensor units rather than a normalised fraction.
  - Doubling the render resolution doubled every marker's pixel radius,
    which is what the millimetre-to-pixel conversion above predicts.
  - `fisheye_lens` and `fisheye_fov` changed no marker position at all.
    Those settings belong to the equisolid and equidistant panorama types;
    `fisheye_fov` only limits the field.

Evaluating the polynomial at the measured radii reproduced the marker
angles to within 0.15%. The reference camera for the measurement was the
March Ariana rig (16 GoPro-derived cameras): k1 = -0.3055 with a 6.74 mm
sensor at 3840 px implies 1865 px/rad, against the GoPro fisheye
calibration's fx of 1845, a 1% residual from whoever fitted it.

Blender's own sign convention has k1 negative for an ordinary fisheye, so
`fit_fisheye_polynomial` negates the fitted curve. A wrong sign rotates the
rendered image by 180 degrees, which the verification script detects.

--------------------------------------------------------------------------
THE PRINCIPAL POINT
--------------------------------------------------------------------------
Blender expresses the principal point as `shift_x`/`shift_y`, with shift_y
signed opposite to the image v axis because Blender's sensor y axis points
up. Two details were measured rather than assumed, because both are silent
when wrong:

  1. The NORMALISER differs by camera type. Rendering one marker at shift 0
     and again at shift 0.1, at 1920x1680, moved it by:

         PERSP   shift_x  1920 px      shift_y  1920 px
         PANO    shift_x  1920 px      shift_y  1680 px

     A perspective camera scales both shifts by the fitted sensor axis; a
     panoramic camera scales each shift by its own image dimension.
  2. Blender's raster sits HALF A PIXEL from OpenCV's. With zero shift the
     optical axis lands at array index (w/2 - 0.5, h/2 - 0.5), while
     OpenCV's cx = w/2 means index w/2. Markers placed by unprojecting
     through K landed a constant (-0.50, -0.50) from target, with 0.08 px
     of scatter behind that offset.

Both corrections are applied in `principal_point_to_shift`. With them, a
rendered marker lands within 0.15 px of where the dataset's own intrinsics
say it should, on a pinhole rig, a deliberately off-centre pinhole rig, and
the GoPro fisheye rig. Without the second, every rendered frame is offset
by half a pixel; without the first, a fisheye rig's vertical principal
point is out by about a pixel, which is roughly the error the March Ariana
rig carried.

--------------------------------------------------------------------------
A TRAP THIS MODULE EXISTS TO PREVENT
--------------------------------------------------------------------------
The March export wrote `fl_x = fl_y = lens / sensor_width * w` into
transforms_gt.json for cameras that were PANO fisheye. A PANO camera
ignores `lens` entirely, so those intrinsics never described the rendered
images and only the poses in that file were usable.
`blender_settings_to_calib` raises rather than return pinhole intrinsics
for a panoramic camera.
"""

import math

import numpy as np

# Calibration pkl model strings, matching offline_undistort.py.
MODEL_PINHOLE = "PINHOLE"
MODEL_OPENCV = "OPENCV"
MODEL_FISHEYE = "OPENCV_FISHEYE"
SUPPORTED_MODELS = (MODEL_PINHOLE, MODEL_OPENCV, MODEL_FISHEYE)

# Blender's default sensor width. The fisheye path treats sensor size as a
# free scale on r, so any positive value works as long as the polynomial is
# fitted against the same one. The PERSP path needs it to derive `lens`.
DEFAULT_SENSOR_WIDTH_MM = 36.0

# Anisotropic pixels cannot be expressed by one Blender camera: PERSP has a
# single `lens`, and the fisheye polynomial is radially symmetric. Real
# calibrations of square-pixel sensors land far inside this tolerance (the
# GoPro reference differs by 3e-5).
FOCAL_ASPECT_TOL = 0.01


class IntrinsicsError(ValueError):
    """Raised when a calibration cannot be represented by a Blender camera."""


# --------------------------------------------------------------- sensor fit
def resolve_sensor_fit(sensor_fit, width, height):
    """Resolve sensor_fit AUTO to the axis Blender will actually fit."""
    if sensor_fit not in ("AUTO", "HORIZONTAL", "VERTICAL"):
        raise IntrinsicsError(f"unknown sensor_fit {sensor_fit!r}")
    if sensor_fit != "AUTO":
        return sensor_fit
    return "HORIZONTAL" if width >= height else "VERTICAL"


def fit_extent(width, height, sensor_width_mm, sensor_height_mm, sensor_fit):
    """Pixels and millimetres along the fitted sensor axis.

    Returns (fit_px, fit_mm). Every normalisation in this module divides by
    one of these, so the AUTO/HORIZONTAL/VERTICAL question is answered once
    here rather than at each call site."""
    axis = resolve_sensor_fit(sensor_fit, width, height)
    if axis == "HORIZONTAL":
        return float(width), float(sensor_width_mm)
    return float(height), float(sensor_height_mm)


def mm_per_pixel(width, height, sensor_width_mm, sensor_height_mm, sensor_fit):
    fit_px, fit_mm = fit_extent(width, height, sensor_width_mm,
                                sensor_height_mm, sensor_fit)
    return fit_mm / fit_px


# ------------------------------------------------------- OpenCV fisheye math
def fisheye_theta_d(theta, dist):
    """OpenCV fisheye forward distortion: theta -> theta_d.

    theta_d = theta * (1 + d1*theta^2 + d2*theta^4 + d3*theta^6 + d4*theta^8)
    """
    d1, d2, d3, d4 = (float(x) for x in np.asarray(dist, dtype=np.float64).reshape(-1)[:4])
    t = np.asarray(theta, dtype=np.float64)
    t2 = t * t
    return t * (1.0 + t2 * (d1 + t2 * (d2 + t2 * (d3 + t2 * d4))))


def fisheye_dtheta_d(theta, dist):
    """d(theta_d)/d(theta), used to find where the model stops being
    monotonic. Past that angle the OpenCV polynomial folds back on itself
    and no longer describes a lens, so fits must stop there."""
    d1, d2, d3, d4 = (float(x) for x in np.asarray(dist, dtype=np.float64).reshape(-1)[:4])
    t = np.asarray(theta, dtype=np.float64)
    t2 = t * t
    return 1.0 + t2 * (3.0 * d1 + t2 * (5.0 * d2 + t2 * (7.0 * d3 + t2 * 9.0 * d4)))


def fisheye_monotonic_limit(dist, hard_limit=math.pi):
    """Largest theta for which the OpenCV fisheye model still increases."""
    thetas = np.linspace(1e-6, hard_limit, 4000)
    slope = fisheye_dtheta_d(thetas, dist)
    bad = np.nonzero(slope <= 0.0)[0]
    if bad.size == 0:
        return hard_limit
    return float(thetas[max(bad[0] - 1, 0)])


def fisheye_theta_from_theta_d(theta_d, dist, max_theta=None):
    """Invert theta_d(theta) by Newton iteration with a bisection guard."""
    limit = fisheye_monotonic_limit(dist) if max_theta is None else float(max_theta)
    targets = np.atleast_1d(np.asarray(theta_d, dtype=np.float64))
    out = np.zeros_like(targets)
    for i, target in enumerate(targets):
        lo, hi = 0.0, limit
        t = min(max(float(target), 0.0), limit)
        for _ in range(100):
            f = float(fisheye_theta_d(t, dist)) - target
            if abs(f) < 1e-12:
                break
            if f > 0.0:
                hi = t
            else:
                lo = t
            slope = float(fisheye_dtheta_d(t, dist))
            step = t - f / slope if slope > 1e-9 else 0.5 * (lo + hi)
            t = step if lo < step < hi else 0.5 * (lo + hi)
        out[i] = t
    return out if np.ndim(theta_d) else float(out[0])


# ------------------------------------------------ Blender fisheye polynomial
def blender_fisheye_theta(r_mm, coeffs):
    """Blender's mapping, radius in sensor millimetres -> ray angle.

    This is the measured convention documented in the module docstring, and
    it is the function the render path is verified against."""
    k0, k1, k2, k3, k4 = (float(c) for c in coeffs)
    r = np.asarray(r_mm, dtype=np.float64)
    return np.abs(k0 + r * (k1 + r * (k2 + r * (k3 + r * k4))))


def fit_fisheye_polynomial(camera_matrix, dist, image_size, sensor_width_mm,
                           sensor_height_mm=None, sensor_fit="AUTO",
                           samples=512):
    """Fit Blender's degree-4 radius-to-angle polynomial to an OpenCV
    fisheye calibration.

    The fit domain is the sensor radius actually reachable in the rendered
    frame, that is the corner of the image measured from the principal
    point, capped where the OpenCV model stops being monotonic. Fitting
    beyond either bound is what makes a fisheye render subtly wrong at the
    edges while looking fine in the middle.

    Returns (coeffs, report) where coeffs is [k0..k4] with Blender's sign
    convention and report carries the fit residual in radians and in pixels
    plus the fitted domain."""
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    w, h = (int(v) for v in image_size)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if sensor_height_mm is None:
        sensor_height_mm = sensor_width_mm * h / w
    check_isotropic_focal(fx, fy)

    pitch = mm_per_pixel(w, h, sensor_width_mm, sensor_height_mm, sensor_fit)

    # Farthest image point from the principal point: an image corner.
    r_corner_px = math.hypot(max(cx, w - cx), max(cy, h - cy))
    theta_limit = fisheye_monotonic_limit(dist)
    r_limit_px = float(fisheye_theta_d(theta_limit, dist)) * fx
    r_max_px = min(r_corner_px, r_limit_px)

    r_px = np.linspace(0.0, r_max_px, int(samples))
    theta = fisheye_theta_from_theta_d(r_px / fx, dist, max_theta=theta_limit)
    r_mm = r_px * pitch

    # numpy fits highest power first; Blender stores lowest first. The
    # negation puts k1 below zero, which is Blender's convention for an
    # ordinary fisheye (a positive fit renders the image rotated by 180).
    fitted = np.polyfit(r_mm, theta, 4)
    coeffs = [float(-c) for c in fitted[::-1]]

    predicted = blender_fisheye_theta(r_mm, coeffs)
    resid_rad = float(np.max(np.abs(predicted - theta)))
    # Convert the angular residual to pixels at the local magnification so
    # the number can be compared against a pixel tolerance.
    resid_px = resid_rad * fx
    report = {
        "max_residual_rad": resid_rad,
        "max_residual_px": resid_px,
        "fit_r_max_mm": float(r_mm[-1]),
        "fit_r_max_px": float(r_max_px),
        "fit_theta_max_rad": float(theta[-1]),
        "clipped_by_monotonic_limit": bool(r_limit_px < r_corner_px),
        "mm_per_pixel": pitch,
    }
    return coeffs, report


# ------------------------------------------------------------- shift helpers
# Blender's raster and OpenCV's pixel coordinates differ by half a pixel.
# With zero shift Blender puts the optical axis at array index
# (w/2 - 0.5, h/2 - 0.5), while OpenCV's cx = w/2 means array index w/2,
# because index i denotes the centre of pixel i. Measured directly: markers
# placed by unprojecting through K landed a constant (-0.50, -0.50) from
# their target, on both a centred and a deliberately off-centre
# calibration, with only 0.08 px of scatter behind that offset. Adding half
# a pixel to the principal point before converting removes it, so a point
# the dataset's intrinsics place at index (u, v) is rendered there.
RASTER_HALF_PIXEL = 0.5


def shift_normalisers(width, height, sensor_width_mm, sensor_height_mm,
                      sensor_fit, camera_type):
    """Pixels of image movement per unit of shift_x and shift_y.

    Blender is not consistent between camera types here, and the difference
    is invisible until a reconstruction comes out subtly wrong. Measured by
    rendering one marker at shift 0 and again at shift 0.1, at 1920x1680:

        PERSP   shift_x -> 1920 px,  shift_y -> 1920 px
        PANO    shift_x -> 1920 px,  shift_y -> 1680 px

    So a perspective camera scales BOTH shifts by the fitted sensor axis,
    while a panoramic camera scales each shift by its own image dimension.
    Using the perspective rule for a fisheye put its vertical principal
    point about a pixel out on the reference GoPro rig."""
    fit_px, _ = fit_extent(width, height, sensor_width_mm, sensor_height_mm,
                           sensor_fit)
    if camera_type == "PANO":
        return float(width), float(height)
    return fit_px, fit_px


def principal_point_to_shift(cx, cy, width, height, sensor_width_mm,
                             sensor_height_mm, sensor_fit, camera_type="PERSP"):
    norm_x, norm_y = shift_normalisers(width, height, sensor_width_mm,
                                       sensor_height_mm, sensor_fit, camera_type)
    cx = cx + RASTER_HALF_PIXEL
    cy = cy + RASTER_HALF_PIXEL
    return ((width / 2.0 - cx) / norm_x, (cy - height / 2.0) / norm_y)


def shift_to_principal_point(shift_x, shift_y, width, height, sensor_width_mm,
                             sensor_height_mm, sensor_fit, camera_type="PERSP"):
    norm_x, norm_y = shift_normalisers(width, height, sensor_width_mm,
                                       sensor_height_mm, sensor_fit, camera_type)
    cx = width / 2.0 - shift_x * norm_x - RASTER_HALF_PIXEL
    cy = height / 2.0 + shift_y * norm_y - RASTER_HALF_PIXEL
    return (cx, cy)


def check_isotropic_focal(fx, fy):
    if fx <= 0.0 or fy <= 0.0:
        raise IntrinsicsError(f"non-positive focal length (fx={fx}, fy={fy})")
    if abs(fy / fx - 1.0) > FOCAL_ASPECT_TOL:
        raise IntrinsicsError(
            f"anisotropic pixels (fx={fx:.3f}, fy={fy:.3f}, ratio "
            f"{fy / fx:.4f}) cannot be represented by a single Blender "
            "camera. Set sensor_fit explicitly and adjust sensor_height, or "
            "rescale the calibration to square pixels first.")


# ------------------------------------------------------------ main direction
def calib_to_blender(calib, render_size=None, sensor_width_mm=DEFAULT_SENSOR_WIDTH_MM,
                     sensor_fit="AUTO"):
    """Blender camera settings for an OpenCV calibration.

    `calib` is a calibration dict in the pkl schema: camera_matrix,
    distortion_coefficients, image_size, model. `render_size` defaults to
    the calibration's own image_size; pass it to render at a different
    resolution (the intrinsics are rescaled the same way
    deps/camera-calibration/convert_intrinsics.py rescales them, which is a
    pure scale plus a principal-point shift, with distortion unchanged).

    Returns a dict of bpy camera-data attributes plus a `_meta` block. The
    caller assigns each non-underscore key straight onto a bpy Camera."""
    model = calib.get("model") or infer_model(calib.get("distortion_coefficients"))
    if model not in SUPPORTED_MODELS:
        raise IntrinsicsError(f"unsupported camera model {model!r}")

    K, dist, (w, h) = rescale_calibration(calib, render_size)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    check_isotropic_focal(fx, fy)

    sensor_height_mm = sensor_width_mm * h / w
    camera_type = "PANO" if model == MODEL_FISHEYE else "PERSP"
    shift_x, shift_y = principal_point_to_shift(
        cx, cy, w, h, sensor_width_mm, sensor_height_mm, sensor_fit,
        camera_type=camera_type)

    settings = {
        "sensor_width": float(sensor_width_mm),
        "sensor_height": float(sensor_height_mm),
        "sensor_fit": sensor_fit,
        "shift_x": float(shift_x),
        "shift_y": float(shift_y),
    }
    meta = {
        "model": model,
        "render_size": [int(w), int(h)],
        "camera_matrix": K.tolist(),
        "distortion_coefficients": np.asarray(dist, dtype=np.float64).reshape(-1).tolist(),
    }

    if model == MODEL_FISHEYE:
        coeffs, report = fit_fisheye_polynomial(
            K, dist, (w, h), sensor_width_mm, sensor_height_mm, sensor_fit)
        settings["type"] = "PANO"
        settings["panorama_type"] = "FISHEYE_LENS_POLYNOMIAL"
        for i, c in enumerate(coeffs):
            settings[f"fisheye_polynomial_k{i}"] = c
        # Field limit, with a margin so the fitted domain is never clipped
        # by the field itself. Cycles uses this only to bound the field.
        settings["fisheye_fov"] = float(min(2.0 * report["fit_theta_max_rad"] * 1.05,
                                            2.0 * math.pi))
        meta["fisheye_fit"] = report
    else:
        # A PERSP camera renders no distortion. For MODEL_OPENCV that makes
        # the render an undistorted image of the same pinhole geometry, so
        # the pipeline must not undistort it a second time.
        fit_px, fit_mm = fit_extent(w, h, sensor_width_mm, sensor_height_mm,
                                    sensor_fit)
        settings["type"] = "PERSP"
        settings["lens"] = float(fx * fit_mm / fit_px)
        meta["renders_distortion"] = False
        if model == MODEL_OPENCV and np.any(np.abs(np.asarray(dist, dtype=np.float64)) > 1e-9):
            meta["distortion_not_rendered"] = True
            # The meta block describes the RENDERED image, which downstream
            # writes out as each camera's calibration pkl. Keeping the
            # source distortion there would tell the pose solve to treat an
            # undistorted render as distorted. The source stays on record.
            meta["source_model"] = model
            meta["source_distortion_coefficients"] = meta["distortion_coefficients"]
            meta["model"] = MODEL_PINHOLE
            meta["distortion_coefficients"] = [0.0] * len(meta["distortion_coefficients"])

    settings["_meta"] = meta
    return settings


def blender_settings_to_calib(settings, render_size, samples=512):
    """The inverse: a calibration dict for a camera authored in Blender.

    Fisheye cameras are fitted back to a 4-coefficient OpenCV fisheye model
    so a rendered rig can be handed to undistort_frames.py unchanged.

    Raises for a panoramic camera whose settings claim a PERSP reading.
    That guard is the transforms_gt.json bug: `lens` is meaningless on a
    PANO camera, and reporting it as a focal length silently produced
    intrinsics that never matched the rendered pixels."""
    w, h = (int(v) for v in render_size)
    cam_type = settings.get("type", "PERSP")
    sensor_width_mm = float(settings.get("sensor_width", DEFAULT_SENSOR_WIDTH_MM))
    sensor_height_mm = float(settings.get("sensor_height",
                                          sensor_width_mm * h / w))
    sensor_fit = settings.get("sensor_fit", "AUTO")
    fit_px, fit_mm = fit_extent(w, h, sensor_width_mm, sensor_height_mm, sensor_fit)
    cx, cy = shift_to_principal_point(
        float(settings.get("shift_x", 0.0)), float(settings.get("shift_y", 0.0)),
        w, h, sensor_width_mm, sensor_height_mm, sensor_fit,
        camera_type=cam_type)

    if cam_type == "PERSP":
        f = float(settings["lens"]) * fit_px / fit_mm
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        return {"camera_matrix": K,
                "distortion_coefficients": np.zeros(5),
                "image_size": (w, h),
                "model": MODEL_PINHOLE}

    if cam_type != "PANO":
        raise IntrinsicsError(f"camera type {cam_type!r} has no calibration form")
    pano = settings.get("panorama_type")
    if pano != "FISHEYE_LENS_POLYNOMIAL":
        raise IntrinsicsError(
            f"panorama_type {pano!r} cannot be expressed as an OpenCV "
            "calibration. Only FISHEYE_LENS_POLYNOMIAL is supported; a "
            "PERSP reading of a panoramic camera (lens / sensor_width * w) "
            "is meaningless and produced wrong ground-truth intrinsics once "
            "already.")

    coeffs = [float(settings[f"fisheye_polynomial_k{i}"]) for i in range(5)]
    pitch = fit_mm / fit_px
    r_corner_px = math.hypot(max(cx, w - cx), max(cy, h - cy))
    r_px = np.linspace(1e-6, r_corner_px, int(samples))
    theta = blender_fisheye_theta(r_px * pitch, coeffs)

    # Least squares on r_px = fx * theta * (1 + d1 t^2 + d2 t^4 + d3 t^6 + d4 t^8).
    # Linear in (fx, fx*d1, ..., fx*d4), so one solve recovers everything.
    basis = np.stack([theta, theta ** 3, theta ** 5, theta ** 7, theta ** 9], axis=1)
    sol, *_ = np.linalg.lstsq(basis, r_px, rcond=None)
    fx = float(sol[0])
    dist = (sol[1:] / fx).astype(np.float64)
    resid_px = float(np.max(np.abs(basis @ sol - r_px)))

    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
    return {"camera_matrix": K,
            "distortion_coefficients": dist,
            "image_size": (w, h),
            "model": MODEL_FISHEYE,
            "fit_max_residual_px": resid_px}


# ------------------------------------------------------------------ helpers
def infer_model(dist):
    """Same rule offline_undistort.py uses for calibrations with no model
    field: four coefficients means fisheye, anything else means OPENCV."""
    if dist is None:
        return MODEL_PINHOLE
    flat = np.asarray(dist, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.any(np.abs(flat) > 1e-12):
        return MODEL_PINHOLE
    return MODEL_FISHEYE if flat.size == 4 else MODEL_OPENCV


def rescale_calibration(calib, render_size=None):
    """Calibration matrix, distortion, and size at the render resolution.

    Scaling matches convert_intrinsics.py in deps/camera-calibration: take
    the smaller axis scale, then shift the principal point by half the crop
    (or half the letterbox band, when the target is wider). Distortion
    coefficients are unitless in this model and do not change."""
    K = np.asarray(calib["camera_matrix"], dtype=np.float64).reshape(3, 3).copy()
    dist = np.asarray(calib.get("distortion_coefficients", []),
                      dtype=np.float64).reshape(-1)
    src = calib.get("image_size")
    if src is None:
        if render_size is None:
            raise IntrinsicsError(
                "calibration has no image_size and no render_size was given")
        return K, dist, (int(render_size[0]), int(render_size[1]))
    w_in, h_in = (int(v) for v in src)
    if render_size is None:
        return K, dist, (w_in, h_in)

    w_out, h_out = (int(v) for v in render_size)
    if (w_out, h_out) == (w_in, h_in):
        return K, dist, (w_out, h_out)

    scale = min(w_out / w_in, h_out / h_in)
    K[:2, :] *= scale
    K[0, 2] -= (w_in * scale - w_out) / 2.0
    K[1, 2] -= (h_in * scale - h_out) / 2.0
    return K, dist, (w_out, h_out)
