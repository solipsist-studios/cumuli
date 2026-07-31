"""Unit tests for render_orbit_views.py (pure geometry only -- gsplat and torch
are imported lazily inside render_orbit, so these run without a GPU)."""

import numpy as np
import pytest

import build_colmap_sparse as bcs
import render_orbit_views as rov


# --------------------------------------------------------------------------
# Camera convention
# --------------------------------------------------------------------------

def test_opengl_c2w_to_w2c_matches_build_colmap_sparse():
    # The whole pipeline hinges on one c2w/w2c convention; this must not drift
    # from the reference implementation the COLMAP writer uses.
    rng = np.random.default_rng(0)
    c2w = np.eye(4)
    c2w[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    c2w[:3, 3] = rng.normal(size=3)

    w2c = rov.opengl_c2w_to_w2c(c2w)
    R, t = bcs.opengl_c2w_to_colmap_w2c(c2w)
    assert np.allclose(w2c[:3, :3], R)
    assert np.allclose(w2c[:3, 3], t)


def test_opengl_c2w_to_w2c_does_not_mutate_its_input():
    c2w = np.eye(4)
    original = c2w.copy()
    rov.opengl_c2w_to_w2c(c2w)
    assert np.array_equal(c2w, original)


# --------------------------------------------------------------------------
# Look-at
# --------------------------------------------------------------------------

def test_lookat_puts_the_target_on_the_optical_axis():
    eye = np.array([0.0, 0.0, -5.0])
    target = np.array([1.0, 2.0, 3.0])
    w2c = rov.lookat_w2c(eye, target, np.array([0.0, -1.0, 0.0]))

    in_camera = w2c[:3, :3] @ target + w2c[:3, 3]
    assert in_camera[0] == pytest.approx(0.0, abs=1e-9)
    assert in_camera[1] == pytest.approx(0.0, abs=1e-9)
    assert in_camera[2] == pytest.approx(np.linalg.norm(target - eye))


def test_lookat_rotation_is_orthonormal():
    w2c = rov.lookat_w2c(np.array([3.0, 1.0, 0.0]), np.zeros(3), np.array([0.0, -1.0, 0.0]))
    R = w2c[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_lookat_places_the_camera_at_the_requested_eye():
    eye = np.array([2.0, -1.0, 4.0])
    w2c = rov.lookat_w2c(eye, np.zeros(3), np.array([0.0, -1.0, 0.0]))
    recovered = -w2c[:3, :3].T @ w2c[:3, 3]
    assert np.allclose(recovered, eye)


# --------------------------------------------------------------------------
# Triangulation
# --------------------------------------------------------------------------

def make_projection(eye, target):
    K = np.array([[800.0, 0.0, 512.0], [0.0, 800.0, 512.0], [0.0, 0.0, 1.0]])
    return K @ rov.lookat_w2c(eye, target, np.array([0.0, -1.0, 0.0]))[:3, :]


def test_triangulate_recovers_a_known_point():
    point = np.array([0.1, 0.2, 0.3])
    projections = [make_projection(np.array([3.0, 0.0, 0.0]), np.zeros(3)),
                   make_projection(np.array([0.0, 0.0, 3.0]), np.zeros(3))]
    uvs = []
    for P in projections:
        homogeneous = P @ np.append(point, 1.0)
        uvs.append(homogeneous[:2] / homogeneous[2])

    assert np.allclose(rov.triangulate_dlt(projections, uvs), point, atol=1e-6)


def test_triangulate_uses_every_camera_given():
    point = np.array([-0.4, 0.1, 0.25])
    eyes = [np.array([3.0, 0.0, 0.0]), np.array([0.0, 0.0, 3.0]), np.array([0.0, 3.0, 0.1])]
    projections = [make_projection(eye, np.zeros(3)) for eye in eyes]
    uvs = []
    for P in projections:
        homogeneous = P @ np.append(point, 1.0)
        uvs.append(homogeneous[:2] / homogeneous[2])

    assert np.allclose(rov.triangulate_dlt(projections, uvs), point, atol=1e-6)


# --------------------------------------------------------------------------
# Azimuth densification -- the load-bearing sampling logic
# --------------------------------------------------------------------------

def test_densified_azimuths_returns_the_requested_count_for_even_anchors():
    azimuths = rov.densified_azimuths(np.array([0.0, 90.0, 180.0, 270.0]), 12, 3.0, 180.0)
    assert len(azimuths) == 12


def test_stereo_pairs_collapse_to_one_anchor_each():
    # Six pairs 0.5 degrees apart: without merging, each pair's near-zero gap
    # would subdivide into visually identical duplicate views.
    pairs = np.array([a + d for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0) for d in (0.0, 0.5)])
    merged = rov.densified_azimuths(pairs, 12, 3.0, 180.0)
    unmerged = rov.densified_azimuths(pairs, 12, 0.1, 180.0)

    assert len(np.unique(np.round(merged, 3))) == len(merged)
    # with a 0.1 degree merge threshold the 12 anchors each get 1 sample -- the
    # pair members land 0.5 degrees apart, i.e. effectively duplicate viewpoints
    close_pairs = np.sum(np.diff(np.sort(unmerged)) < 1.0)
    assert close_pairs > 0
    assert np.all(np.diff(np.sort(merged)) >= 1.0)


def test_samples_never_walk_further_than_max_reach_from_an_anchor():
    # A 180 degree blind spot must not receive a sample at its midpoint: that is
    # the position farthest from any real camera and renders incoherently.
    anchors = np.array([0.0, 90.0, 180.0])
    samples = rov.densified_azimuths(anchors, 9, 3.0, 20.0)
    for sample in samples:
        distance = min(abs(sample - a) for a in list(anchors) + [360.0])
        assert distance <= 20.0 + 1e-9


def test_narrow_gaps_are_unaffected_by_the_reach_cap():
    # Gaps whose natural subdivision is already inside the cap must subdivide normally.
    anchors = np.array([0.0, 10.0, 20.0, 30.0])
    capped = rov.densified_azimuths(anchors, 8, 3.0, 20.0)
    uncapped = rov.densified_azimuths(anchors, 8, 3.0, 1e6)
    assert capped[:len(anchors)] == pytest.approx(uncapped[:len(anchors)])


def test_wraparound_cluster_merges_across_the_360_seam():
    # 359.5 and 0.2 are 0.7 degrees apart the short way round, not 359.3.
    merged = rov.densified_azimuths(np.array([0.2, 120.0, 240.0, 359.5]), 3, 3.0, 180.0)
    assert len(merged) == 3


def test_every_sample_starts_at_or_after_its_anchor():
    anchors = np.array([0.0, 100.0, 200.0])
    samples = rov.densified_azimuths(anchors, 6, 3.0, 30.0)
    assert min(samples) >= anchors.min() - 1e-9


# --------------------------------------------------------------------------
# Subject / head anchoring
# --------------------------------------------------------------------------

def test_head_anchor_sits_above_the_centroid_along_the_up_axis():
    rng = np.random.default_rng(1)
    up = np.array([0.0, 1.0, 0.0])
    means = rng.uniform(-1, 1, size=(500, 3))
    opacity = np.full(500, 0.9)

    centroid, head = rov.subject_anchors(means, opacity, up)
    assert (head - centroid) @ up > 0


def test_centroid_ignores_low_opacity_outliers():
    up = np.array([0.0, 1.0, 0.0])
    subject = np.zeros((300, 3))
    junk = np.full((300, 3), 50.0)
    means = np.vstack([subject, junk])
    opacity = np.concatenate([np.full(300, 0.9), np.full(300, 0.01)])

    centroid, _ = rov.subject_anchors(means, opacity, up)
    assert np.allclose(centroid, 0.0)


def test_head_anchor_falls_back_when_too_few_splats_resolve():
    up = np.array([0.0, 1.0, 0.0])
    means = np.zeros((5, 3))
    opacity = np.full(5, 0.9)
    centroid, head = rov.subject_anchors(means, opacity, up)
    assert np.allclose(head, centroid + up * rov.DEFAULT_HEAD_OFFSET)


# --------------------------------------------------------------------------
# Facing direction
# --------------------------------------------------------------------------

def test_facing_azimuth_falls_back_with_fewer_than_two_cameras():
    cameras = [{"label": "00", "center": np.zeros(3), "projection": np.zeros((3, 4))}]
    result = rov.facing_azimuth({"00": (np.zeros(2),) * 3}, cameras, np.array([0.0, 1.0, 0.0]),
                                np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), fallback=42.0)
    assert result == 42.0


def test_facing_azimuth_points_at_the_nose_side_of_the_eye_line():
    up = np.array([0.0, 1.0, 0.0])
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    # face looking along +e1: nose sits ahead of the eye midpoint on that axis
    nose, left_eye, right_eye = np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, -0.03]), np.array([0.0, 0.0, 0.03])

    cameras, face_kps = [], {}
    for label, eye in (("00", np.array([3.0, 0.0, 0.0])), ("01", np.array([0.0, 0.0, 3.0]))):
        projection = make_projection(eye, np.zeros(3))
        cameras.append({"label": label, "center": eye, "projection": projection})
        observed = []
        for point in (nose, left_eye, right_eye):
            homogeneous = projection @ np.append(point, 1.0)
            observed.append(homogeneous[:2] / homogeneous[2])
        face_kps[label] = tuple(observed)

    assert rov.facing_azimuth(face_kps, cameras, up, e1, e2, fallback=999.0) == pytest.approx(0.0, abs=1.0)
