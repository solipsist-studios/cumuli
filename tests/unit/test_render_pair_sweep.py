"""Unit tests for render_pair_sweep.py (pure geometry and image warping -- gsplat
and torch are imported lazily inside render_pair_sweep, so these run without a GPU).

The properties under test are the ones the whole approach rests on: a swept frame
that sits at a real camera's centre must reproduce that centre exactly, and the
homography from that frame back to the real photo must be exact rather than
approximate. If either drifts, the "real pixel" endpoints stop being real pixels
and the bakeoff's ground truth stops being ground truth."""

import numpy as np
import pytest
from PIL import Image

import rig_geometry as rg
import render_pair_sweep as rps


UP = np.array([0.0, 0.0, 1.0])
E1 = np.array([1.0, 0.0, 0.0])
E2 = np.cross(UP, E1)
TARGET = np.array([0.3, -0.2, 1.1])


def make_camera(label, azimuth_deg, elevation_deg=5.0, radius=4.0, focal=1200.0, size=1600):
    """A real camera at a known spherical position, aimed at TARGET, with the
    full w2c/intrinsics that the homography needs."""
    center = rg.position_at(TARGET, UP, E1, E2, azimuth_deg, elevation_deg, radius)
    w2c = rg.lookat_w2c(center, TARGET, -UP)
    intrinsics = np.array([[focal, 0.0, size / 2.0], [0.0, focal, size / 2.0], [0.0, 0.0, 1.0]])
    return {"label": label, "center": center, "w2c": w2c, "intrinsics": intrinsics,
            "projection": intrinsics @ w2c[:3, :], "file_path": f"images/{label}.png"}


# --------------------------------------------------------------------------
# Spherical parameterisation
# --------------------------------------------------------------------------

def test_spherical_and_position_at_round_trip_exactly():
    # The endpoint homography is exact only if a real camera's centre survives
    # the trip through (azimuth, elevation, radius) unchanged.
    rng = np.random.default_rng(7)
    for _ in range(20):
        center = TARGET + rng.normal(size=3) * 3.0
        azimuth, elevation, radius = rg.spherical(center, TARGET, UP, E1, E2)
        assert np.allclose(rg.position_at(TARGET, UP, E1, E2, azimuth, elevation, radius), center)


def test_spherical_uses_the_orbit_basis():
    center = TARGET + 5.0 * E1
    azimuth, elevation, radius = rg.spherical(center, TARGET, UP, E1, E2)
    assert azimuth == pytest.approx(0.0)
    assert elevation == pytest.approx(0.0)
    assert radius == pytest.approx(5.0)


@pytest.mark.parametrize("start, end, expected", [
    (10.0, 40.0, 30.0),
    (40.0, 10.0, -30.0),
    (170.0, -170.0, 20.0),    # across the wrap, the short way
    (-170.0, 170.0, -20.0),
])
def test_shortest_arc(start, end, expected):
    assert rg.shortest_arc(start, end) == pytest.approx(expected)


def test_shortest_arc_at_exactly_half_a_turn_picks_one_direction():
    # Antipodal cameras have no shorter way round; either sign is correct, and
    # the only thing worth pinning is that the magnitude is a half turn.
    assert abs(rg.shortest_arc(0.0, 180.0)) == pytest.approx(180.0)


def test_shortest_arc_never_takes_the_long_way():
    # A pair straddling +/-180 must sweep the small gap between the cameras, not
    # the long way round through every other camera in the rig.
    rng = np.random.default_rng(3)
    for _ in range(50):
        start, end = rng.uniform(-180, 180, size=2)
        assert abs(rg.shortest_arc(start, end)) <= 180.0 + 1e-9


# --------------------------------------------------------------------------
# Sweep parameters
# --------------------------------------------------------------------------

def test_sweep_endpoints_land_on_the_real_camera_centres():
    a, b = make_camera("03", 10.0, 4.0, 4.0), make_camera("07", 50.0, 9.0, 4.5)
    parameters, probe = rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=24)

    assert probe is None
    assert np.allclose(rg.position_at(TARGET, UP, E1, E2, *parameters[0]), a["center"])
    assert np.allclose(rg.position_at(TARGET, UP, E1, E2, *parameters[-1]), b["center"])


def test_sweep_advances_monotonically_between_the_endpoints():
    a, b = make_camera("03", 10.0), make_camera("07", 50.0)
    parameters, _ = rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=17)
    azimuths = [p[0] for p in parameters]
    assert len(parameters) == 17
    assert all(later > earlier for earlier, later in zip(azimuths, azimuths[1:]))


def test_sweep_interpolates_elevation_and_radius_too():
    # Real rigs are not on a perfect circle; a sweep that held radius fixed would
    # drift off the shell the cameras actually occupy.
    a, b = make_camera("03", 0.0, elevation_deg=0.0, radius=4.0), make_camera("07", 40.0, 10.0, 6.0)
    parameters, _ = rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=3)
    assert parameters[1][1] == pytest.approx(5.0)
    assert parameters[1][2] == pytest.approx(5.0)


def test_probe_frame_is_snapped_to_the_holdout_camera_exactly():
    a, b = make_camera("03", 0.0, 4.0, 4.0), make_camera("07", 60.0, 8.0, 5.0)
    holdout = make_camera("05", 31.0, elevation_deg=2.0, radius=4.7)
    parameters, probe = rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=25, holdout=holdout)

    assert probe is not None
    # snapped wholesale -- not merely nearby, or the warped photo is not ground truth
    assert np.allclose(rg.position_at(TARGET, UP, E1, E2, *parameters[probe]), holdout["center"])


def test_probe_is_never_an_endpoint():
    # The endpoints carry the real-pixel pins; making one of them the probe would
    # hand the model the answer it is being scored against.
    a, b = make_camera("03", 0.0), make_camera("07", 60.0)
    for azimuth in (0.05, 59.95):
        _, probe = rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=12,
                                        holdout=make_camera("05", azimuth))
        assert probe not in (0, 11)


def test_probe_outside_the_arc_is_an_error():
    a, b = make_camera("03", 0.0), make_camera("07", 40.0)
    with pytest.raises(ValueError, match="does not lie between"):
        rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=12, holdout=make_camera("09", 75.0))


def test_a_sweep_needs_at_least_two_frames():
    a, b = make_camera("03", 0.0), make_camera("07", 40.0)
    with pytest.raises(ValueError, match="at least 2 frames"):
        rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=1)


# --------------------------------------------------------------------------
# Angular loss weighting (Hwang et al. SIGGRAPH 2026, Eq. S2)
# --------------------------------------------------------------------------

def test_loss_weight_peaks_at_the_real_endpoints():
    a, b = make_camera("03", 0.0), make_camera("07", 40.0)
    parameters, _ = rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=21)
    weights = rps.angular_loss_weights(parameters)

    assert weights[0] == pytest.approx(3.0)    # 1 + lambda_gt, lambda_gt = 2
    assert weights[-1] == pytest.approx(3.0)
    assert weights[10] == pytest.approx(1.0)   # the midpoint, farthest from both


def test_loss_weight_falls_monotonically_from_each_end():
    a, b = make_camera("03", 0.0), make_camera("07", 40.0)
    parameters, _ = rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=21)
    weights = rps.angular_loss_weights(parameters)

    first_half, second_half = weights[:11], weights[10:]
    assert all(later <= earlier for earlier, later in zip(first_half, first_half[1:]))
    assert all(later >= earlier for earlier, later in zip(second_half, second_half[1:]))


def test_loss_weight_is_symmetric_about_the_midpoint():
    a, b = make_camera("03", 0.0), make_camera("07", 40.0)
    parameters, _ = rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=21)
    weights = rps.angular_loss_weights(parameters)
    assert weights == pytest.approx(weights[::-1])


def test_loss_weight_never_leaves_the_documented_range():
    a, b = make_camera("03", 10.0, 3.0, 4.0), make_camera("07", 95.0, 9.0, 5.0)
    parameters, _ = rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=33)
    weights = rps.angular_loss_weights(parameters)
    assert all(1.0 - 1e-9 <= w <= 3.0 + 1e-9 for w in weights)


def test_lambda_gt_zero_weights_every_frame_equally():
    # The escape hatch for testing whether the weighting helps at all.
    a, b = make_camera("03", 0.0), make_camera("07", 40.0)
    parameters, _ = rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=9)
    assert rps.angular_loss_weights(parameters, lambda_gt=0.0) == pytest.approx([1.0] * 9)


def test_holdout_probe_does_not_receive_endpoint_weight():
    # The probe sits at a REAL camera's centre, so a nearest-real-camera rule
    # would hand it weight 3 and train on the view being held out to score the
    # result. Only endpoints may anchor the weighting.
    a, b = make_camera("03", 0.0), make_camera("07", 60.0)
    holdout = make_camera("05", 30.0)
    parameters, probe = rps.sweep_parameters(a, b, TARGET, UP, E1, E2, count=21, holdout=holdout)
    weights = rps.angular_loss_weights(parameters)

    assert weights[probe] < 1.5
    assert weights[probe] < weights[0]


# --------------------------------------------------------------------------
# The endpoint homography
# --------------------------------------------------------------------------

def project(camera, points):
    """Pixel coordinates of world points in a real camera."""
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    uvw = (camera["projection"] @ homogeneous.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def test_homography_maps_the_frustum_back_to_the_real_image_exactly():
    # The claim the whole design rests on: two cameras sharing a centre are
    # related by a homography with NO depth term. Verified by projecting world
    # points at wildly different depths and checking they land in the same place
    # whether projected directly or carried through the homography.
    camera = make_camera("05", 20.0, elevation_deg=6.0, radius=4.0)
    res, focal = 1024, 900.0
    w2c_new = rg.lookat_w2c(camera["center"], TARGET, -UP)
    homography = rps.homography_to_frustum(camera, w2c_new, focal, res / 2.0)

    rng = np.random.default_rng(11)
    directions = rng.normal(size=(40, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # same directions from the shared centre, at depths spanning two orders of magnitude
    depths = rng.uniform(0.5, 50.0, size=(40, 1))
    points = camera["center"] + directions * depths

    intrinsics_new = np.array([[focal, 0.0, res / 2.0], [0.0, focal, res / 2.0], [0.0, 0.0, 1.0]])
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    uvw_new = (intrinsics_new @ w2c_new[:3, :] @ homogeneous.T).T
    in_front = uvw_new[:, 2] > 1e-6
    pixels_new = uvw_new[in_front, :2] / uvw_new[in_front, 2:3]

    carried = (homography @ np.concatenate([pixels_new, np.ones((len(pixels_new), 1))], axis=1).T).T
    carried = carried[:, :2] / carried[:, 2:3]

    assert in_front.sum() > 5
    assert np.allclose(carried, project(camera, points[in_front]), atol=1e-6)


def test_homography_is_identity_when_nothing_changes():
    camera = make_camera("05", 20.0, radius=4.0, focal=900.0, size=1024)
    homography = rps.homography_to_frustum(camera, camera["w2c"], 900.0, 512.0)
    assert np.allclose(homography / homography[2, 2], np.eye(3), atol=1e-9)


# --------------------------------------------------------------------------
# Warping a real photo into the frustum
# --------------------------------------------------------------------------

def test_warp_under_the_identity_homography_reproduces_the_image(tmp_path):
    rng = np.random.default_rng(5)
    source = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    path = tmp_path / "real.png"
    Image.fromarray(source).save(path)

    warped = rps.warp_real_image(path, np.eye(3), 64)
    array = np.asarray(warped)
    assert warped.mode == "RGBA"
    assert np.array_equal(array[..., :3], source)
    assert (array[..., 3] == 255).all()


def test_warp_marks_out_of_frame_pixels_transparent(tmp_path):
    # Half the destination frustum looks where the real photo has no pixels; those
    # must come back transparent so the scorer never compares against padding.
    source = np.full((40, 40, 3), 200, dtype=np.uint8)
    path = tmp_path / "real.png"
    Image.fromarray(source).save(path)

    shifted = np.eye(3)
    shifted[0, 2] = -30.0  # pull destination pixels off the left edge of the source
    array = np.asarray(rps.warp_real_image(path, shifted, 40))
    assert (array[..., 3] == 0).any()
    assert (array[..., 3] == 255).any()
    assert (array[:, :29, 3] == 0).all()


def test_warp_treats_points_behind_the_camera_as_uncovered(tmp_path):
    source = np.full((32, 32, 3), 128, dtype=np.uint8)
    path = tmp_path / "real.png"
    Image.fromarray(source).save(path)

    behind = np.eye(3)
    behind[2, 2] = -1.0  # every ray's w goes negative
    assert (np.asarray(rps.warp_real_image(path, behind, 32))[..., 3] == 0).all()


# --------------------------------------------------------------------------
# Real-image resolution
# --------------------------------------------------------------------------

STATIC_K = np.array([[900.0, 0.0, 512.0], [0.0, 900.0, 512.0], [0.0, 0.0, 1.0]])


def test_resolve_real_view_finds_the_recorded_path(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "05.png").write_bytes(b"")
    transforms = tmp_path / "transforms.json"
    transforms.write_text("{}")

    found, _ = rps.resolve_real_view({"file_path": "images/05.png", "intrinsics": STATIC_K},
                                     transforms, None)
    assert found == tmp_path / "images" / "05.png"


def test_resolve_real_view_re_sniffs_the_extension(tmp_path):
    # build_refit_dataset.py writes uniform .png names over real .jpg frames, so
    # the recorded extension routinely disagrees with what is on disk.
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "05.jpg").write_bytes(b"")
    transforms = tmp_path / "transforms.json"
    transforms.write_text("{}")

    found, _ = rps.resolve_real_view({"file_path": "images/05.png", "intrinsics": STATIC_K},
                                     transforms, None)
    assert found == tmp_path / "images" / "05.jpg"


def test_resolve_real_view_honours_the_override_directory(tmp_path):
    elsewhere = tmp_path / "moved"
    elsewhere.mkdir()
    (elsewhere / "05.png").write_bytes(b"")
    transforms = tmp_path / "transforms.json"
    transforms.write_text("{}")

    found, _ = rps.resolve_real_view({"file_path": "gone/05.png", "intrinsics": STATIC_K},
                                     transforms, elsewhere)
    assert found == elsewhere / "05.png"


def test_resolve_real_view_returns_none_when_absent(tmp_path):
    transforms = tmp_path / "transforms.json"
    transforms.write_text("{}")
    camera = {"file_path": "images/05.png", "intrinsics": STATIC_K}
    assert rps.resolve_real_view(camera, transforms, None)[0] is None
    assert rps.resolve_real_view({"file_path": "", "intrinsics": STATIC_K}, transforms, None)[0] is None


def test_resolve_real_view_returns_the_requested_frames_intrinsics(tmp_path):
    # The full4d crops move the principal point every frame. Warping frame 58's
    # photo through frame 1's principal point misaligned it by 249 px.
    (tmp_path / "cam04").mkdir()
    (tmp_path / "cam04" / "frame_00058.jpg").write_bytes(b"")
    transforms = tmp_path / "transforms.json"
    transforms.write_text("{}")

    early = np.array([[1309.5, 0.0, 1231.6], [0.0, 1309.5, 753.7], [0.0, 0.0, 1.0]])
    late = np.array([[1309.5, 0.0, 1262.6], [0.0, 1309.5, 504.7], [0.0, 0.0, 1.0]])
    camera = {"file_path": "cam04/frame_00001.jpg", "intrinsics": early, "per_frame": {
        1: {"file_path": "cam04/frame_00001.jpg", "intrinsics": early},
        58: {"file_path": "cam04/frame_00058.jpg", "intrinsics": late},
    }}

    found, intrinsics = rps.resolve_real_view(camera, transforms, None, frame_number=58)
    assert found == tmp_path / "cam04" / "frame_00058.jpg"
    assert intrinsics[1, 2] == pytest.approx(504.7)


def test_resolve_real_view_uses_the_sole_entry_for_a_static_rig(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "05.png").write_bytes(b"")
    transforms = tmp_path / "transforms.json"
    transforms.write_text("{}")

    camera = {"file_path": "images/05.png", "intrinsics": STATIC_K,
              "per_frame": {None: {"file_path": "images/05.png", "intrinsics": STATIC_K}}}
    found, intrinsics = rps.resolve_real_view(camera, transforms, None, frame_number=58)
    assert found == tmp_path / "images" / "05.png"
    assert intrinsics[0, 0] == pytest.approx(900.0)


def test_homography_uses_the_frames_own_intrinsics():
    camera = make_camera("04", 20.0, radius=4.0, focal=1300.0, size=1024)
    shifted = camera["intrinsics"].copy()
    shifted[1, 2] -= 249.0  # the measured per-frame principal-point drift

    default = rps.homography_to_frustum(camera, camera["w2c"], 900.0, 512.0)
    per_frame = rps.homography_to_frustum(camera, camera["w2c"], 900.0, 512.0, shifted)
    assert not np.allclose(default, per_frame)
    # the shift lands where a vertical principal-point move should, and only there
    assert per_frame[1, 2] == pytest.approx(default[1, 2] - 249.0)
    assert per_frame[0, 2] == pytest.approx(default[0, 2])


# --------------------------------------------------------------------------
# Adjacent pairing
# --------------------------------------------------------------------------

def test_adjacent_pairs_skips_stereo_twins():
    # This rig is stereo pairs 0.3-0.8 degrees apart; sweeping between the halves
    # of one pair would generate a clip of near-identical frames.
    cameras = [make_camera("00", 0.0), make_camera("01", 0.5),
               make_camera("02", 30.0), make_camera("03", 30.4)]
    pairs = rps.adjacent_pairs(cameras, TARGET, UP, E1, E2, merge_deg=3.0)
    assert pairs == [("01", "02")]


def test_adjacent_pairs_orders_by_azimuth_not_by_label():
    cameras = [make_camera("09", 40.0), make_camera("02", 0.0), make_camera("05", 20.0)]
    assert rps.adjacent_pairs(cameras, TARGET, UP, E1, E2, merge_deg=3.0) == [("02", "05"), ("05", "09")]
