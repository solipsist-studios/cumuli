# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Calibration to Blender camera settings, and back.

Every number here was established by measurement against Blender 5.1.2
(scripts/verify_blender_intrinsics.py reproduces the measurement end to
end). These tests pin those conventions so a Blender upgrade that changes
one fails loudly, rather than shifting every rendered frame by a fraction
of a pixel that nothing else would notice.
"""

import math

import numpy as np
import pytest

import blender_camera_intrinsics as bci


# The real GoPro calibration the March Ariana rig was built from.
GOPRO_K = np.array([[1844.97025145, 0.0, 1905.77203213],
                    [0.0, 1845.03139881, 1667.95822578],
                    [0.0, 0.0, 1.0]])
GOPRO_D = np.array([0.03426853, 0.05935758, -0.04352849, 0.00874438])
GOPRO_SIZE = (3840, 3360)

# The coefficients stored in that rig's .blend, at sensor width 6.74 mm.
ARIANA_K = [-3.9394761552102864e-4, -0.30550360679626465, -6.013988051563501e-3,
            4.655892960727215e-3, -5.011643515899777e-4]
ARIANA_SENSOR_MM = 6.74


def gopro_calib():
    return {"camera_matrix": GOPRO_K.copy(),
            "distortion_coefficients": GOPRO_D.copy(),
            "image_size": GOPRO_SIZE,
            "model": bci.MODEL_FISHEYE}


def pinhole_calib(w=1920, h=1680, fx=1706.667, cx=None, cy=None):
    cx = w / 2.0 if cx is None else cx
    cy = h / 2.0 if cy is None else cy
    return {"camera_matrix": np.array([[fx, 0.0, cx], [0.0, fx, cy],
                                       [0.0, 0.0, 1.0]]),
            "distortion_coefficients": np.zeros(5),
            "image_size": (w, h),
            "model": bci.MODEL_PINHOLE}


# ------------------------------------------------------------- sensor fit
@pytest.mark.parametrize("w,h,expected", [
    (1920, 1080, "HORIZONTAL"),
    (1080, 1920, "VERTICAL"),
    (1000, 1000, "HORIZONTAL"),
])
def test_auto_sensor_fit_follows_the_longer_axis(w, h, expected):
    assert bci.resolve_sensor_fit("AUTO", w, h) == expected


def test_unknown_sensor_fit_is_rejected():
    with pytest.raises(bci.IntrinsicsError):
        bci.resolve_sensor_fit("DIAGONAL", 100, 100)


# -------------------------------------------------------- principal point
def test_shift_normalisers_differ_between_camera_types():
    """Measured: a PANO camera scales shift_y by image height while a
    PERSP camera scales it by the fitted sensor axis. Using the perspective
    rule for a fisheye put the vertical principal point about a pixel out."""
    persp = bci.shift_normalisers(1920, 1680, 36.0, 31.5, "AUTO", "PERSP")
    pano = bci.shift_normalisers(1920, 1680, 36.0, 31.5, "AUTO", "PANO")
    assert persp == (1920.0, 1920.0)
    assert pano == (1920.0, 1680.0)


def test_centred_principal_point_still_carries_the_half_pixel():
    """Blender's zero-shift axis lands at index (w/2 - 0.5, h/2 - 0.5), so a
    centred OpenCV principal point needs a non-zero shift."""
    shift_x, shift_y = bci.principal_point_to_shift(
        960.0, 840.0, 1920, 1680, 36.0, 31.5, "AUTO", "PERSP")
    assert shift_x == pytest.approx(-0.5 / 1920.0)
    assert shift_y == pytest.approx(0.5 / 1920.0)


@pytest.mark.parametrize("camera_type", ["PERSP", "PANO"])
@pytest.mark.parametrize("cx,cy", [(960.0, 840.0), (860.0, 900.0), (1010.5, 799.25)])
def test_principal_point_round_trips(camera_type, cx, cy):
    args = (1920, 1680, 36.0, 31.5, "AUTO", camera_type)
    shift_x, shift_y = bci.principal_point_to_shift(cx, cy, *args)
    back = bci.shift_to_principal_point(shift_x, shift_y, *args)
    assert back[0] == pytest.approx(cx, abs=1e-9)
    assert back[1] == pytest.approx(cy, abs=1e-9)


def test_principal_point_left_of_centre_gives_a_positive_x_shift():
    """Sign checked against a render with cx deliberately 100 px left of
    centre; the markers stayed aligned."""
    shift_x, _ = bci.principal_point_to_shift(
        860.0, 840.0, 1920, 1680, 36.0, 31.5, "AUTO", "PERSP")
    assert shift_x > 0


# ------------------------------------------------------- pinhole cameras
def test_pinhole_lens_follows_from_the_focal_length():
    settings = bci.calib_to_blender(pinhole_calib(fx=1706.6667),
                                    sensor_width_mm=36.0)
    assert settings["type"] == "PERSP"
    assert settings["lens"] == pytest.approx(1706.6667 * 36.0 / 1920)


def test_pinhole_round_trips_through_blender_settings():
    calib = pinhole_calib(fx=1500.0, cx=930.0, cy=870.0)
    settings = bci.calib_to_blender(calib, sensor_width_mm=36.0)
    back = bci.blender_settings_to_calib(settings, (1920, 1680))
    assert back["camera_matrix"][0, 0] == pytest.approx(1500.0, rel=1e-9)
    assert back["camera_matrix"][0, 2] == pytest.approx(930.0, abs=1e-9)
    assert back["camera_matrix"][1, 2] == pytest.approx(870.0, abs=1e-9)


def test_anisotropic_pixels_are_rejected_with_a_usable_message():
    calib = pinhole_calib()
    calib["camera_matrix"][1, 1] *= 1.2
    with pytest.raises(bci.IntrinsicsError, match="anisotropic"):
        bci.calib_to_blender(calib)


def test_opencv_distortion_is_carried_but_flagged_as_not_rendered():
    calib = pinhole_calib()
    calib["model"] = bci.MODEL_OPENCV
    calib["distortion_coefficients"] = np.array([0.1, -0.05, 0.001, 0.002, 0.0])
    settings = bci.calib_to_blender(calib)
    assert settings["type"] == "PERSP"
    assert settings["_meta"]["distortion_not_rendered"] is True
    # What is written as the camera's calibration must match the pixels,
    # which carry no distortion; the source calibration stays on record.
    assert settings["_meta"]["model"] == bci.MODEL_PINHOLE
    assert not any(settings["_meta"]["distortion_coefficients"])
    assert settings["_meta"]["source_model"] == bci.MODEL_OPENCV
    assert settings["_meta"]["source_distortion_coefficients"][0] == 0.1


# ------------------------------------------------------- fisheye cameras
def test_fisheye_becomes_a_polynomial_panorama_camera():
    settings = bci.calib_to_blender(gopro_calib(), sensor_width_mm=ARIANA_SENSOR_MM)
    assert settings["type"] == "PANO"
    assert settings["panorama_type"] == "FISHEYE_LENS_POLYNOMIAL"
    assert settings["fisheye_polynomial_k1"] < 0    # Blender's own sign


def test_fitted_fisheye_matches_the_reference_rig_coefficients():
    """The fit reproduces the coefficients the March rig actually used, to
    a small fraction of a degree across the whole image. That is the
    evidence that this module reconstructs the historical procedure rather
    than merely producing something self-consistent."""
    settings = bci.calib_to_blender(gopro_calib(), sensor_width_mm=ARIANA_SENSOR_MM)
    fitted = [settings[f"fisheye_polynomial_k{i}"] for i in range(5)]
    report = settings["_meta"]["fisheye_fit"]

    r_mm = np.linspace(0.0, report["fit_r_max_mm"], 200)
    ours = bci.blender_fisheye_theta(r_mm, fitted)
    theirs = bci.blender_fisheye_theta(r_mm, ARIANA_K)
    assert math.degrees(np.max(np.abs(ours - theirs))) < 0.1


def test_fisheye_fit_residual_is_sub_pixel():
    settings = bci.calib_to_blender(gopro_calib(), sensor_width_mm=ARIANA_SENSOR_MM)
    assert settings["_meta"]["fisheye_fit"]["max_residual_px"] < 1.0


def test_fisheye_polynomial_reproduces_the_opencv_curve():
    """The fit is checked where it matters: the angle Blender will assign a
    given image radius, against the angle OpenCV assigns it."""
    settings = bci.calib_to_blender(gopro_calib(), sensor_width_mm=ARIANA_SENSOR_MM)
    coeffs = [settings[f"fisheye_polynomial_k{i}"] for i in range(5)]
    pitch = settings["_meta"]["fisheye_fit"]["mm_per_pixel"]

    r_px = np.linspace(0.0, settings["_meta"]["fisheye_fit"]["fit_r_max_px"], 100)
    expected = bci.fisheye_theta_from_theta_d(r_px / GOPRO_K[0, 0], GOPRO_D)
    got = bci.blender_fisheye_theta(r_px * pitch, coeffs)
    assert np.max(np.abs(got - expected)) * GOPRO_K[0, 0] < 1.0


def test_fisheye_round_trips_back_to_an_opencv_calibration():
    settings = bci.calib_to_blender(gopro_calib(), sensor_width_mm=ARIANA_SENSOR_MM)
    back = bci.blender_settings_to_calib(settings, GOPRO_SIZE)
    assert back["model"] == bci.MODEL_FISHEYE
    assert back["camera_matrix"][0, 0] == pytest.approx(GOPRO_K[0, 0], rel=1e-3)
    assert back["camera_matrix"][0, 2] == pytest.approx(GOPRO_K[0, 2], abs=1e-6)
    assert back["camera_matrix"][1, 2] == pytest.approx(GOPRO_K[1, 2], abs=1e-6)


def test_reading_a_panoramic_camera_as_a_pinhole_is_refused():
    """The March export wrote lens/sensor*w as a focal length for cameras
    that ignore `lens` entirely, and only the poses in that file survived.
    A render places points about 1800 px from where those intrinsics say."""
    settings = bci.calib_to_blender(gopro_calib(), sensor_width_mm=ARIANA_SENSOR_MM)
    settings["panorama_type"] = "EQUIRECTANGULAR"
    with pytest.raises(bci.IntrinsicsError, match="meaningless"):
        bci.blender_settings_to_calib(settings, GOPRO_SIZE)


def test_fisheye_theta_d_inverts_itself():
    theta = np.linspace(0.0, 1.2, 50)
    theta_d = bci.fisheye_theta_d(theta, GOPRO_D)
    back = bci.fisheye_theta_from_theta_d(theta_d, GOPRO_D)
    assert np.allclose(back, theta, atol=1e-9)


def test_monotonic_limit_stops_where_the_model_folds_back():
    """Past this angle the OpenCV polynomial decreases, so it no longer
    describes a lens and a fit through it would be nonsense."""
    # d1 = -0.5 makes the derivative 1 - 1.5*theta^2, which reaches zero at
    # theta = sqrt(2/3) ~ 0.8165 rad.
    folding = np.array([-0.5, 0.0, 0.0, 0.0])
    limit = bci.fisheye_monotonic_limit(folding)
    assert limit == pytest.approx(math.sqrt(2.0 / 3.0), abs=2e-3)
    assert bci.fisheye_dtheta_d(limit, folding) > 0
    assert bci.fisheye_dtheta_d(limit + 0.05, folding) < 0


def test_monotonic_limit_returns_the_whole_range_for_a_well_behaved_lens():
    assert bci.fisheye_monotonic_limit(GOPRO_D) == pytest.approx(math.pi)


# ----------------------------------------------------------- model choice
@pytest.mark.parametrize("dist,expected", [
    (None, bci.MODEL_PINHOLE),
    (np.zeros(5), bci.MODEL_PINHOLE),
    (np.array([0.1, 0.2, 0.3, 0.4]), bci.MODEL_FISHEYE),
    (np.array([0.1, 0.2, 0.3, 0.4, 0.5]), bci.MODEL_OPENCV),
])
def test_model_inference_matches_offline_undistort(dist, expected):
    assert bci.infer_model(dist) == expected


def test_unsupported_model_is_rejected():
    calib = pinhole_calib()
    calib["model"] = "EQUIRECT"
    with pytest.raises(bci.IntrinsicsError, match="unsupported camera model"):
        bci.calib_to_blender(calib)


# ------------------------------------------------------------- rescaling
def test_rescaling_to_half_resolution_halves_the_intrinsics():
    K, dist, size = bci.rescale_calibration(gopro_calib(), (1920, 1680))
    assert size == (1920, 1680)
    assert K[0, 0] == pytest.approx(GOPRO_K[0, 0] / 2)
    assert K[0, 2] == pytest.approx(GOPRO_K[0, 2] / 2)
    assert np.allclose(dist, GOPRO_D)      # unitless, unchanged by rescaling


def test_rescaling_to_a_new_aspect_shifts_the_principal_point():
    """A crop moves the principal point by half the removed width, which is
    what convert_intrinsics.py in deps/camera-calibration does."""
    K, _, size = bci.rescale_calibration(pinhole_calib(1920, 1080, cx=960, cy=540),
                                         (1080, 1080))
    assert size == (1080, 1080)
    assert K[0, 2] == pytest.approx(540.0)


def test_rescaling_is_a_no_op_at_the_native_size():
    K, _, size = bci.rescale_calibration(gopro_calib(), GOPRO_SIZE)
    assert size == GOPRO_SIZE
    assert np.allclose(K, GOPRO_K)


def test_missing_image_size_without_a_render_size_is_an_error():
    calib = gopro_calib()
    del calib["image_size"]
    with pytest.raises(bci.IntrinsicsError, match="image_size"):
        bci.rescale_calibration(calib, None)
