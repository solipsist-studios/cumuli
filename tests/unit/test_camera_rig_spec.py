# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Rig geometry: layouts, subject-relative resolution, and conventions.

The camera poses these functions produce are the ground truth every later
measurement is compared against, so the properties worth pinning are
geometric rather than incidental: cameras look where they were told to,
sit where the spec put them, and convert to the world convention the rest
of the pipeline reads.
"""

import json
import math

import numpy as np
import pytest

import camera_rig_spec as crs


MANIFEST = {
    "subject_bbox": {"min": [-0.5, -0.4, 0.0], "max": [0.5, 0.4, 1.8]},
}


def ring_spec(**overrides):
    spec = {
        "name": "t",
        "layout": "rings",
        "target": [0.0, 0.0, 1.0],
        "rings": [{"count": 8, "radius": 3.0, "height": 1.2}],
        "resolution": [640, 480],
        "intrinsics": {"lens_mm": 35, "sensor_width_mm": 36},
        "eval": {"count": 4},
    }
    spec.update(overrides)
    return spec


# ----------------------------------------------------------------- look-at
def test_look_at_points_camera_minus_z_at_target():
    c2w = crs.look_at_matrix([3.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    forward = -c2w[:3, 2]
    expected = np.array([-1.0, 0.0, 0.0])
    assert np.allclose(forward, expected, atol=1e-9)


def test_look_at_basis_is_orthonormal_and_right_handed():
    c2w = crs.look_at_matrix([2.0, -3.0, 4.0], [0.1, 0.2, 0.9])
    R = c2w[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_look_at_survives_a_straight_down_view():
    """A cage rig's top row looks along world up, where the usual cross
    product degenerates."""
    c2w = crs.look_at_matrix([0.0, 0.0, 5.0], [0.0, 0.0, 0.0])
    R = c2w[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.allclose(-c2w[:3, 2], [0.0, 0.0, -1.0], atol=1e-9)


def test_look_at_rejects_a_camera_on_its_own_target():
    with pytest.raises(crs.RigSpecError):
        crs.look_at_matrix([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])


# ------------------------------------------------------------- conventions
def test_blender_to_opengl_world_maps_z_up_to_y_up():
    """Verified against the March export: a camera at Blender (2, 0, 2)
    appears in transforms_gt.json at (2, 2, 0)."""
    c2w = np.eye(4)
    c2w[:3, 3] = [2.0, 0.0, 2.0]
    assert np.allclose(crs.blender_to_opengl_world(c2w)[:3, 3], [2.0, 2.0, 0.0])


def test_opengl_conversion_preserves_orthonormality():
    c2w = crs.look_at_matrix([1.0, 2.0, 3.0], [0.0, 0.0, 1.0])
    R = crs.blender_to_opengl_world(c2w)[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)


@pytest.mark.parametrize("count,width", [(4, 2), (16, 2), (100, 2), (101, 3), (250, 3)])
def test_label_width_keeps_labels_sortable(count, width):
    labels = crs.make_labels(count)
    assert all(len(label) == width for label in labels)
    assert labels == sorted(labels)


# ------------------------------------------------------------------ layouts
def test_rings_places_the_requested_count_at_the_requested_radius():
    rig = crs.resolve_rig(ring_spec())
    assert len(rig["train"]) == 8
    for cam in rig["train"]:
        horizontal = math.hypot(cam.position[0], cam.position[1])
        assert horizontal == pytest.approx(3.0, abs=1e-9)
        assert cam.position[2] == pytest.approx(1.2, abs=1e-9)


def test_rings_honours_the_azimuth_offset():
    spec = ring_spec(rings=[{"count": 4, "radius": 2.0, "height": 1.0,
                             "azimuth_offset_deg": 45.0}])
    rig = crs.resolve_rig(spec)
    angles = sorted(round(math.degrees(math.atan2(c.position[1], c.position[0])) % 360)
                    for c in rig["train"])
    assert angles == [45, 135, 225, 315]


def test_cage_stays_inside_its_height_band():
    spec = {
        "name": "c", "layout": "cage", "target": [0.0, 0.0, 1.0],
        "cage": {"theta": 8, "phi": 3, "radius": 4.0,
                 "min_height": 0.5, "max_height": 3.0},
        "resolution": [320, 240], "intrinsics": {"lens_mm": 24},
        "eval": {"count": 0},
    }
    rig = crs.resolve_rig(spec)
    assert len(rig["train"]) == 24
    heights = [c.position[2] for c in rig["train"]]
    assert min(heights) >= 0.5 - 1e-9
    assert max(heights) <= 3.0 + 1e-9


def test_cage_rejects_an_empty_height_band():
    spec = {
        "name": "c", "layout": "cage", "target": [0.0, 0.0, 1.0],
        "cage": {"theta": 4, "phi": 2, "radius": 4.0,
                 "min_height": 3.0, "max_height": 1.0},
        "resolution": [320, 240], "intrinsics": {"lens_mm": 24},
    }
    with pytest.raises(crs.RigSpecError, match="height band is empty"):
        crs.resolve_rig(spec)


def test_explicit_layout_uses_the_given_positions():
    spec = {
        "name": "e", "layout": "explicit", "target": [0.0, 0.0, 0.0],
        "cameras": [{"position": [1.0, 0.0, 0.0]},
                    {"position": [0.0, 2.0, 0.0], "look_at": [0.0, 0.0, 1.0]}],
        "resolution": [320, 240], "intrinsics": {"lens_mm": 24},
        "eval": {"count": 0},
    }
    rig = crs.resolve_rig(spec)
    assert np.allclose(rig["train"][0].position, [1.0, 0.0, 0.0])
    assert np.allclose(rig["train"][1].look_at, [0.0, 0.0, 1.0])


def test_every_camera_aims_at_the_target():
    rig = crs.resolve_rig(ring_spec(rings=[
        {"count": 6, "radius": 2.5, "height": 0.8},
        {"count": 6, "radius": 2.5, "height": 2.0}]))
    target = np.asarray(rig["target"])
    for cam in rig["train"]:
        forward = -cam.c2w_blender[:3, 2]
        aim = target - cam.position
        aim = aim / np.linalg.norm(aim)
        assert np.allclose(forward, aim, atol=1e-9)


# ------------------------------------------------- subject-relative values
def test_subject_heights_scales_a_length_by_subject_height():
    spec = ring_spec(target="subject_center",
                     rings=[{"count": 4, "radius": {"subject_heights": 2.0},
                             "height": {"subject_fraction": 0.5}}])
    rig = crs.resolve_rig(spec, MANIFEST)
    target = np.asarray(rig["target"])
    for cam in rig["train"]:
        horizontal = math.hypot(cam.position[0] - target[0],
                                cam.position[1] - target[1])
        assert horizontal == pytest.approx(3.6, abs=1e-9)   # 2.0 * 1.8 m
        assert cam.position[2] == pytest.approx(0.9, abs=1e-9)


def test_subject_center_is_the_bbox_centre_at_mid_height():
    rig = crs.resolve_rig(ring_spec(target="subject_center"), MANIFEST)
    assert np.allclose(rig["target"], [0.0, 0.0, 0.9])


def test_subject_floor_center_sits_on_the_floor():
    rig = crs.resolve_rig(ring_spec(target="subject_floor_center"), MANIFEST)
    assert rig["target"][2] == pytest.approx(0.0, abs=1e-9)


def test_a_subject_relative_spec_follows_a_taller_subject():
    """One spec, two characters. This is what makes rig specs reusable."""
    spec = ring_spec(target="subject_center",
                     rings=[{"count": 4, "radius": {"subject_heights": 2.0},
                             "height": {"subject_fraction": 1.0}}])
    tall = {"subject_bbox": {"min": [0, 0, 0], "max": [1, 1, 2.4]}}
    rig = crs.resolve_rig(spec, tall)
    assert rig["target"][2] == pytest.approx(1.2)
    assert rig["train"][0].position[2] == pytest.approx(2.4)


def test_subject_relative_value_without_a_manifest_is_an_error():
    spec = ring_spec(rings=[{"count": 4, "radius": {"subject_heights": 2.0},
                             "height": 1.0}])
    with pytest.raises(crs.RigSpecError, match="scene manifest"):
        crs.resolve_rig(spec, None)


# --------------------------------------------------------------- eval ring
def test_eval_cameras_fall_between_the_training_cameras():
    """Half a step off the first ring: the hardest place for the model,
    which is where a novel-view score should be taken."""
    rig = crs.resolve_rig(ring_spec())
    train_az = sorted(math.degrees(math.atan2(c.position[1], c.position[0])) % 360
                      for c in rig["train"])
    eval_az = sorted(math.degrees(math.atan2(c.position[1], c.position[0])) % 360
                     for c in rig["eval"])
    assert len(eval_az) == 4
    for angle in eval_az:
        gaps = [abs(angle - t) for t in train_az]
        assert min(gaps) == pytest.approx(22.5, abs=1e-6)


def test_eval_cameras_are_labelled_apart_from_rig_cameras():
    rig = crs.resolve_rig(ring_spec())
    train_labels = {c.label for c in rig["train"]}
    eval_labels = {c.label for c in rig["eval"]}
    assert not (train_labels & eval_labels)
    assert all(label.startswith("e") for label in eval_labels)


def test_eval_count_zero_produces_no_eval_cameras():
    rig = crs.resolve_rig(ring_spec(eval={"count": 0}))
    assert rig["eval"] == []


def test_eval_cameras_have_the_role_recorded():
    rig = crs.resolve_rig(ring_spec())
    assert {c.role for c in rig["train"]} == {"train"}
    assert {c.role for c in rig["eval"]} == {"eval"}


# ------------------------------------------------------------- intrinsics
def test_lens_and_sensor_give_the_expected_focal_length():
    rig = crs.resolve_rig(ring_spec())
    K = rig["calibration"]["camera_matrix"]
    assert K[0, 0] == pytest.approx(35.0 / 36.0 * 640)
    assert K[0, 2] == pytest.approx(320.0)
    assert K[1, 2] == pytest.approx(240.0)


def test_fov_intrinsics_match_the_requested_field():
    spec = ring_spec(intrinsics={"fov_deg": 90.0})
    K = crs.resolve_rig(spec)["calibration"]["camera_matrix"]
    assert K[0, 0] == pytest.approx(320.0)          # (w/2) / tan(45 deg)


def test_a_distorted_model_needs_a_calibration_rather_than_a_focal_length():
    spec = ring_spec(camera_model="OPENCV_FISHEYE")
    with pytest.raises(crs.RigSpecError, match="calibration_pkl"):
        crs.resolve_rig(spec)


def test_calibration_pkl_without_a_loader_is_an_error():
    spec = ring_spec(camera_model="OPENCV_FISHEYE",
                     calibration_pkl="does/not/matter.pkl")
    with pytest.raises(crs.RigSpecError, match="loader"):
        crs.resolve_rig(spec)


def test_pickle_calib_loader_resolves_relative_to_the_spec(tmp_path):
    import pickle

    calib = {"camera_matrix": np.eye(3), "distortion_coefficients": np.zeros(4),
             "image_size": (640, 480), "model": "OPENCV_FISHEYE"}
    (tmp_path / "cal").mkdir()
    with open(tmp_path / "cal" / "c.pkl", "wb") as f:
        pickle.dump(calib, f)
    spec_path = tmp_path / "rig.json"
    spec_path.write_text("{}")
    loader = crs.pickle_calib_loader(spec_path)
    assert loader("cal/c.pkl")["model"] == "OPENCV_FISHEYE"


def test_pickle_calib_loader_reports_where_it_looked(tmp_path):
    spec_path = tmp_path / "rig.json"
    spec_path.write_text("{}")
    loader = crs.pickle_calib_loader(spec_path)
    with pytest.raises(crs.RigSpecError, match="Tried:"):
        loader("nowhere/c.pkl")


# ------------------------------------------------------------- validation
def test_unknown_spec_keys_are_rejected():
    with pytest.raises(crs.RigSpecError, match="unknown rig spec key"):
        crs.validate_spec(ring_spec(nonsense=1))


@pytest.mark.parametrize("key", ["subject_collections", "background_collections",
                                 "markers"])
def test_keys_nothing_reads_are_rejected_rather_than_ignored(key):
    """The collections are fixed names set by prepare_blender_scene.py, and
    nothing places markers. Accepting these keys made a spec that set them
    look configured while changing nothing."""
    with pytest.raises(crs.RigSpecError, match="unknown rig spec key"):
        crs.validate_spec(ring_spec(**{key: []}))


def test_unknown_layout_is_rejected():
    with pytest.raises(crs.RigSpecError, match="layout must be"):
        crs.validate_spec(ring_spec(layout="spiral"))


def test_missing_resolution_is_rejected():
    spec = ring_spec()
    del spec["resolution"]
    with pytest.raises(crs.RigSpecError, match="resolution"):
        crs.validate_spec(spec)


def test_load_spec_reads_and_validates(tmp_path):
    path = tmp_path / "rig.json"
    path.write_text(json.dumps(ring_spec()))
    assert crs.load_spec(path)["layout"] == "rings"


# ----------------------------------------------------------------- lights
def test_lights_sit_between_the_camera_columns():
    spec = ring_spec(lights={"count": 4, "power_w": 50.0, "radius": 5.0})
    rig = crs.resolve_rig(spec)
    assert len(rig["lights"]) == 4
    angles = sorted(round(math.degrees(math.atan2(light["position"][1],
                                                  light["position"][0])) % 360)
                    for light in rig["lights"])
    assert angles == [45, 135, 225, 315]
    assert all(light["energy"] == 50.0 for light in rig["lights"])


def test_no_lights_block_leaves_the_scene_lighting_alone():
    assert crs.resolve_rig(ring_spec())["lights"] == []


# ----------------------------------------------------- property-based checks
def test_hypothesis_look_at_always_aims_at_the_target():
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st

    coord = st.floats(min_value=-50, max_value=50, allow_nan=False,
                      allow_infinity=False)

    @hypothesis.given(px=coord, py=coord, pz=coord)
    @hypothesis.settings(max_examples=200, deadline=None)
    def check(px, py, pz):
        position = np.array([px, py, pz])
        target = np.zeros(3)
        if np.linalg.norm(position - target) < 1e-3:
            return
        c2w = crs.look_at_matrix(position, target)
        forward = -c2w[:3, 2]
        aim = (target - position) / np.linalg.norm(target - position)
        assert np.allclose(forward, aim, atol=1e-6)
        assert np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-6)

    check()


# ------------------------------------------------- staggered and arc rings
def test_heights_cycle_across_evenly_spaced_azimuths():
    """The point of a stagger is that elevation varies WITHOUT disturbing
    the azimuths. Writing it as three separate rings would space each ring
    independently and lose that."""
    spec = ring_spec(rings=[{"count": 12, "radius": 3.0,
                             "heights": [1.0, 1.5, 2.0]}])
    rig = crs.resolve_rig(spec)
    azimuths = [round(math.degrees(math.atan2(c.position[1], c.position[0])) % 360)
                for c in rig["train"]]
    assert sorted(azimuths) == [i * 30 for i in range(12)]
    assert [round(c.position[2], 6) for c in rig["train"]][:6] == \
        [1.0, 1.5, 2.0, 1.0, 1.5, 2.0]


def test_a_single_height_still_works_as_before():
    rig = crs.resolve_rig(ring_spec(rings=[{"count": 4, "radius": 2.0,
                                            "height": 1.3}]))
    assert {round(c.position[2], 6) for c in rig["train"]} == {1.3}


def test_heights_may_be_subject_relative():
    spec = ring_spec(rings=[{"count": 4, "radius": 2.0,
                             "heights": [{"subject_fraction": 0.25},
                                         {"subject_fraction": 0.75}]}])
    rig = crs.resolve_rig(spec, MANIFEST)
    assert sorted({round(c.position[2], 6) for c in rig["train"]}) == [0.45, 1.35]


def test_an_empty_heights_list_is_rejected():
    spec = ring_spec(rings=[{"count": 4, "radius": 2.0, "heights": []}])
    with pytest.raises(crs.RigSpecError, match="heights is empty"):
        crs.resolve_rig(spec)


def test_an_arc_spreads_cameras_inclusive_of_both_ends():
    """A closed ring divides the circle so the last camera does not meet the
    first. An arc must reach both extremes, or an eleven-camera front arc
    stops one step short of its own span."""
    spec = ring_spec(rings=[{"count": 11, "radius": 3.0, "height": 1.0,
                             "azimuth_span_deg": 160,
                             "azimuth_centre_deg": 0.0}])
    rig = crs.resolve_rig(spec)
    az = sorted(round(math.degrees(math.atan2(c.position[1], c.position[0])), 3)
                for c in rig["train"])
    assert az[0] == pytest.approx(-80.0)
    assert az[-1] == pytest.approx(80.0)
    steps = {round(b - a, 3) for a, b in zip(az, az[1:])}
    assert steps == {16.0}


def test_an_arc_centres_on_the_azimuth_it_is_given():
    """The centre is where the subject faces, so this is the number that
    aims the rig at the front rather than the shoulder."""
    spec = ring_spec(rings=[{"count": 5, "radius": 3.0, "height": 1.0,
                             "azimuth_span_deg": 90,
                             "azimuth_centre_deg": -82.0}])
    rig = crs.resolve_rig(spec)
    az = sorted(math.degrees(math.atan2(c.position[1], c.position[0]))
                for c in rig["train"])
    assert (az[0] + az[-1]) / 2 == pytest.approx(-82.0, abs=1e-6)


def test_an_arc_without_a_centre_starts_at_the_offset():
    spec = ring_spec(rings=[{"count": 3, "radius": 3.0, "height": 1.0,
                             "azimuth_span_deg": 90,
                             "azimuth_offset_deg": 10.0}])
    rig = crs.resolve_rig(spec)
    az = sorted(round(math.degrees(math.atan2(c.position[1], c.position[0])), 3)
                for c in rig["train"])
    assert az == [10.0, 55.0, 100.0]


def test_a_centre_without_a_span_is_rejected():
    """A closed ring has no centre, and silently ignoring the key would
    leave the rig rotated somewhere the author did not intend."""
    spec = ring_spec(rings=[{"count": 4, "radius": 3.0, "height": 1.0,
                             "azimuth_centre_deg": 90.0}])
    with pytest.raises(crs.RigSpecError, match="azimuth_centre_deg"):
        crs.resolve_rig(spec)


def test_a_single_camera_ring_sits_at_its_offset():
    """How a lone rear camera is written."""
    spec = ring_spec(rings=[{"count": 1, "radius": 3.0, "height": 1.0,
                             "azimuth_offset_deg": 98.0}])
    rig = crs.resolve_rig(spec)
    assert len(rig["train"]) == 1
    assert math.degrees(math.atan2(rig["train"][0].position[1],
                                   rig["train"][0].position[0])) == \
        pytest.approx(98.0)


def test_a_closed_ring_is_unchanged_by_the_new_options():
    """Regression guard: specs written before arcs and height cycles existed
    must still resolve to exactly the same cameras."""
    rig = crs.resolve_rig(ring_spec(rings=[{"count": 8, "radius": 3.0,
                                            "height": 1.2}]))
    az = sorted(round(math.degrees(math.atan2(c.position[1], c.position[0])) % 360)
                for c in rig["train"])
    assert az == [i * 45 for i in range(8)]
