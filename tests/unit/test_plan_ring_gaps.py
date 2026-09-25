# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""plan_ring_gaps.py: gap-finding geometry is what a wrong answer here costs
the most -- a mis-measured gap sends 4DAnyone at the wrong azimuth for real
GPU time, so the properties worth pinning are the ones a hand derivation can
check independently, not just round-trips through the code itself.
"""

import json
import math

import numpy as np
import pytest

import plan_ring_gaps as prg
import rig_geometry as rg


def azimuths_on_circle(n, start_deg=0.0, span_deg=360.0):
    if span_deg >= 360.0:
        return [start_deg + i * 360.0 / n for i in range(n)]
    step = span_deg / (n - 1) if n > 1 else 0.0
    return [start_deg + i * step for i in range(n)]


# --------------------------------------------------------- round_views_per_layer
def test_round_views_per_layer_divides_4_or_6():
    for n in range(1, 60):
        v = prg.round_views_per_layer(n)
        assert v >= n
        assert v % 4 == 0 or v % 6 == 0


def test_round_views_per_layer_is_minimal():
    # No smaller valid value should also satisfy v >= n.
    for n in range(1, 60):
        v = prg.round_views_per_layer(n)
        for candidate in range(4, v):
            assert not (candidate >= n and (candidate % 4 == 0 or candidate % 6 == 0))


def test_round_views_per_layer_hypothesis():
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st

    @hypothesis.given(n=st.integers(min_value=1, max_value=200))
    @hypothesis.settings(max_examples=200, deadline=None)
    def check(n):
        v = prg.round_views_per_layer(n)
        assert v >= n
        assert v % 4 == 0 or v % 6 == 0

    check()


# --------------------------------------------------------------- find_gap_arcs
def test_full_coverage_is_no_gaps():
    real_az = {"a": 0.0, "b": 90.0, "c": 180.0, "d": 270.0}
    targets = [(f"t{i}", az, 10.0) for i, az in enumerate(azimuths_on_circle(8))]
    gaps = prg.find_gap_arcs(targets, real_az, front_azimuth_deg=0.0, min_separation_deg=50.0)
    assert gaps == []


def test_total_gap_is_one_arc_spanning_everything():
    real_az = {"only": 999.0}  # nowhere near any target
    targets = [(f"t{i}", az, 10.0) for i, az in enumerate(azimuths_on_circle(6))]
    gaps = prg.find_gap_arcs(targets, real_az, front_azimuth_deg=0.0, min_separation_deg=1.0)
    assert len(gaps) == 1
    assert len(gaps[0].target_azimuths_deg) == 6


def test_gap_matches_hand_derivation_on_a_partial_rig():
    """12 cameras evenly spaced from -71 to +149 (matching the real rig this
    was designed against), 16 evenly spaced targets, 20 deg tolerance. Hand
    derivation (see the session that wrote this test): the surviving gap is
    exactly the 4 targets at 180, 202.5, 225, 247.5 degrees, because the
    ring's own 22.5-degree spacing happens to put its neighbours within 20
    degrees of the real rig's boundary cameras on both sides."""
    real_az = {f"cam{i:02d}": -71 + i * 220 / 11 for i in range(12)}
    targets = [(f"t{i:02d}", i * 360 / 16, 10.0) for i in range(16)]
    gaps = prg.find_gap_arcs(targets, real_az, front_azimuth_deg=0.0, min_separation_deg=20.0)
    assert len(gaps) == 1
    got = sorted(round(a, 3) % 360 for a in gaps[0].target_azimuths_deg)
    assert got == [180.0, 202.5, 225.0, 247.5]


def test_gap_wrapping_across_the_seam_is_one_arc_not_two():
    """A gap straddling +/-180 must not fragment into two arcs just because
    the underlying azimuth representation wraps there."""
    real_az = {"front": 0.0}
    targets = [(f"t{i}", az, 10.0) for i, az in enumerate([170.0, 180.0, -170.0, -160.0])]
    gaps = prg.find_gap_arcs(targets, real_az, front_azimuth_deg=0.0, min_separation_deg=5.0)
    assert len(gaps) == 1
    assert len(gaps[0].target_azimuths_deg) == 4


def test_every_gap_target_is_actually_uncovered():
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st

    @hypothesis.given(
        n_real=st.integers(min_value=1, max_value=6),
        n_target=st.integers(min_value=1, max_value=20),
        seed=st.integers(min_value=0, max_value=10_000),
        min_sep=st.floats(min_value=1.0, max_value=60.0, allow_nan=False),
    )
    @hypothesis.settings(max_examples=150, deadline=None)
    def check(n_real, n_target, seed, min_sep):
        rng = np.random.default_rng(seed)
        real_az = {f"r{i}": float(rng.uniform(-180, 180)) for i in range(n_real)}
        targets = [(f"t{i}", float(rng.uniform(-180, 180)), 10.0) for i in range(n_target)]
        gaps = prg.find_gap_arcs(targets, real_az, front_azimuth_deg=0.0, min_separation_deg=min_sep)
        gapped_azs = {round(a, 6) for g in gaps for a in g.target_azimuths_deg}
        for _, az, _ in targets:
            nearest = min(abs(rg.shortest_arc(az, ra)) for ra in real_az.values())
            is_gapped = round(az, 6) in gapped_azs
            if nearest >= min_sep:
                assert is_gapped, (az, nearest, min_sep)
            # A covered target must never appear in a returned gap.
            if nearest < min_sep:
                assert not is_gapped

    check()


def test_gaps_do_not_overlap_and_partition_the_uncovered_targets():
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st

    @hypothesis.given(
        n_real=st.integers(min_value=1, max_value=6),
        n_target=st.integers(min_value=2, max_value=20),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @hypothesis.settings(max_examples=150, deadline=None)
    def check(n_real, n_target, seed):
        rng = np.random.default_rng(seed)
        real_az = {f"r{i}": float(rng.uniform(-180, 180)) for i in range(n_real)}
        targets = [(f"t{i}", float(rng.uniform(-180, 180)), 10.0) for i in range(n_target)]
        gaps = prg.find_gap_arcs(targets, real_az, front_azimuth_deg=0.0, min_separation_deg=15.0)
        seen = []
        for g in gaps:
            for a in g.target_azimuths_deg:
                key = round(a, 6)
                assert key not in seen, "a target azimuth appeared in two gaps"
                seen.append(key)

    check()


def test_front_azimuth_deg_shifts_the_whole_comparison():
    real_az = {"a": 0.0}
    targets = [("t0", 90.0, 10.0)]
    # Target at spec-frame 90 deg, real camera at 0. Uncovered with no shift.
    assert len(prg.find_gap_arcs(targets, real_az, front_azimuth_deg=0.0, min_separation_deg=5.0)) == 1
    # Shifting the spec's frame by -90 deg puts the target exactly on the real camera.
    assert prg.find_gap_arcs(targets, real_az, front_azimuth_deg=-90.0, min_separation_deg=5.0) == []


# -------------------------------------------------------- split_gap_by_nearest_camera
def test_split_gap_shares_a_gap_between_two_bordering_cameras():
    """The motivating case: two real cameras close together at one edge of
    a wide gap (e.g. a stereo pair) must each get their own piece, not have
    the whole gap claimed by whichever one is nearest the centroid. Only
    one pair exists in this fixture, so there is no competing cluster on
    the gap's far side to confound the split."""
    real_az = {"5564": -145.0, "1959": -135.0}
    targets = [(f"t{i}", az, 16.7) for i, az in enumerate(
        [-170, -160, -150, -140, -130, -120, -110, -100])]
    gaps = prg.find_gap_arcs(targets, real_az, front_azimuth_deg=0.0, min_separation_deg=5.0)
    assert len(gaps) == 1  # one contiguous gap before splitting
    split = prg.split_gap_by_nearest_camera(gaps[0], real_az)
    anchors = {prg.choose_anchor_camera(g, real_az) for g in split}
    assert anchors == {"5564", "1959"}, anchors
    # Every original target azimuth is still covered, exactly once.
    covered = sorted(a for g in split for a in g.target_azimuths_deg)
    assert covered == sorted(t[1] for t in targets)


def test_split_gap_is_a_noop_with_one_bordering_camera():
    real_az = {"only": 0.0}
    gap = prg.GapArc(start_deg=90.0, end_deg=150.0, center_deg=120.0, span_deg=60.0,
                     target_azimuths_deg=(90.0, 120.0, 150.0), pitch_deg=10.0)
    split = prg.split_gap_by_nearest_camera(gap, real_az)
    assert split == [gap]


def test_split_gap_partitions_every_target_exactly_once():
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st

    @hypothesis.given(
        n_real=st.integers(min_value=1, max_value=6),
        n_target=st.integers(min_value=1, max_value=20),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @hypothesis.settings(max_examples=150, deadline=None)
    def check(n_real, n_target, seed):
        rng = np.random.default_rng(seed)
        real_az = {f"r{i}": float(rng.uniform(-180, 180)) for i in range(n_real)}
        gap = prg.GapArc(
            start_deg=0.0, end_deg=0.0, center_deg=0.0, span_deg=0.0,
            target_azimuths_deg=tuple(float(rng.uniform(-180, 180)) for _ in range(n_target)),
            pitch_deg=10.0,
        )
        split = prg.split_gap_by_nearest_camera(gap, real_az)
        covered = sorted(round(a, 9) for g in split for a in g.target_azimuths_deg)
        expected = sorted(round(a, 9) for a in gap.target_azimuths_deg)
        assert covered == expected

    check()


# ------------------------------------------------------------ choose_anchor_camera
def test_choose_anchor_camera_returns_a_real_label():
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st

    @hypothesis.given(
        n_real=st.integers(min_value=1, max_value=8),
        center=st.floats(min_value=-180, max_value=180, allow_nan=False),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @hypothesis.settings(max_examples=150, deadline=None)
    def check(n_real, center, seed):
        rng = np.random.default_rng(seed)
        real_az = {f"r{i}": float(rng.uniform(-180, 180)) for i in range(n_real)}
        gap = prg.GapArc(start_deg=center, end_deg=center, center_deg=center,
                         span_deg=0.0, target_azimuths_deg=(center,), pitch_deg=10.0)
        anchor = prg.choose_anchor_camera(gap, real_az)
        assert anchor in real_az

    check()


def test_choose_anchor_camera_picks_the_nearest():
    real_az = {"far": 170.0, "near": 10.0, "mid": 90.0}
    gap = prg.GapArc(start_deg=5.0, end_deg=5.0, center_deg=5.0, span_deg=0.0,
                     target_azimuths_deg=(5.0,), pitch_deg=0.0)
    assert prg.choose_anchor_camera(gap, real_az) == "near"


# --------------------------------------------------------------- target_rig_by_ring
RING_SPEC = {
    "name": "t", "layout": "rings", "target": [0.0, 0.0, 0.0],
    "rings": [
        {"count": 8, "radius": 3.0, "height": 1.0},
        {"count": 4, "radius": 3.0, "height": 2.0, "azimuth_offset_deg": 45.0},
    ],
    "resolution": [640, 480],
    "intrinsics": {"lens_mm": 35, "sensor_width_mm": 36},
    "eval": {"count": 0},
}


def test_target_rig_by_ring_groups_by_ring_index(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(RING_SPEC))
    groups = prg.target_rig_by_ring(spec_path)
    assert set(groups) == {0, 1}
    assert len(groups[0]) == 8
    assert len(groups[1]) == 4


def test_target_rig_by_ring_azimuths_match_offset(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(RING_SPEC))
    groups = prg.target_rig_by_ring(spec_path)
    # atan2 returns (-180, 180], so a plain sort does not put the
    # azimuth_offset_deg=0 camera first (its 225-360 deg peers wrap
    # negative); normalise to [0, 360) before comparing.
    ring0_az = sorted(az % 360 for _, az, _ in groups[0])
    assert ring0_az[0] == pytest.approx(0.0, abs=1e-6)
    ring1_az = sorted(az % 360 for _, az, _ in groups[1])
    assert ring1_az[0] == pytest.approx(45.0, abs=1e-6)


def test_target_rig_by_ring_resolves_subject_relative_spec_without_a_real_manifest(tmp_path):
    """ring16.json-style specs use subject-relative radius/height and a
    subject_center target. This must resolve via the placeholder manifest
    without the caller supplying a real scene manifest -- azimuth doesn't
    depend on subject size (see the module docstring)."""
    spec = {
        "name": "t", "layout": "rings", "target": "subject_center",
        "rings": [{"count": 6, "radius": {"subject_heights": 1.5},
                  "height": {"subject_fraction": 0.5}}],
        "resolution": [640, 480],
        "intrinsics": {"lens_mm": 35, "sensor_width_mm": 36},
        "eval": {"count": 0},
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    groups = prg.target_rig_by_ring(spec_path)
    assert len(groups[0]) == 6
    az = sorted(a % 360 for _, a, _ in groups[0])
    assert az == pytest.approx([0.0, 60.0, 120.0, 180.0, 240.0, 300.0], abs=1e-6)


# --------------------------------------------------------------- real_rig_azimuths
def _opengl_c2w_looking_at_origin(eye, up=np.array([0.0, 1.0, 0.0])):
    eye = np.asarray(eye, dtype=np.float64)
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def _write_transforms(path, azimuths_deg, radius=3.0, height=1.5):
    frames = []
    for i, az_deg in enumerate(azimuths_deg):
        az = math.radians(az_deg)
        eye = np.array([radius * math.cos(az), height, radius * math.sin(az)])
        frames.append({
            "file_path": f"cam{i:02d}/frame_00001",
            "camera_label": f"{i:02d}",
            "transform_matrix": _opengl_c2w_looking_at_origin(eye).tolist(),
            "fl_x": 1000.0, "fl_y": 1000.0, "cx": 500.0, "cy": 400.0,
            "w": 1000, "h": 800,
        })
    path.write_text(json.dumps({"frames": frames}))


def test_real_rig_azimuths_first_camera_is_zero(tmp_path):
    """orbit_basis defines e1 toward the first real camera, so that camera's
    own azimuth must always come out as ~0 regardless of the rig's true
    physical orientation."""
    transforms = tmp_path / "transforms.json"
    _write_transforms(transforms, [-71, -20, 30, 90, 149])
    az, _target, _up = prg.real_rig_azimuths(transforms)
    assert az["00"] == pytest.approx(0.0, abs=1e-6)


def test_real_rig_azimuths_preserve_relative_spacing(tmp_path):
    """orbit_basis's target is the MEAN of camera centres, which only
    coincides with the true circle centre for a full, symmetric ring -- a
    partial arc or lopsided placement biases the mean toward the covered
    side (a real, known limitation of the centroid heuristic, not something
    this test should paper over). Use a full ring, where the premise holds
    by symmetry regardless of starting phase."""
    azimuths = azimuths_on_circle(6, start_deg=15.0)
    transforms_path = tmp_path / "t.json"
    _write_transforms(transforms_path, azimuths)
    az, _target, _up = prg.real_rig_azimuths(transforms_path)
    got = sorted(az.values())
    # Relative spacing survives even though the absolute frame is redefined
    # (e1 toward camera 0); allow a global sign flip (CW vs CCW under the
    # up-vector's arbitrary but consistent orientation).
    diffs = sorted(abs(rg.shortest_arc(got[0], g)) for g in got[1:])
    expected = sorted(abs(rg.shortest_arc(azimuths[0], a)) for a in azimuths[1:])
    assert diffs == pytest.approx(expected, abs=1e-4)


# ----------------------------------------------------------- plan_generation_runs
def test_plan_generation_runs_end_to_end(tmp_path):
    transforms_path = tmp_path / "transforms.json"
    _write_transforms(transforms_path, [-71 + i * 220 / 11 for i in range(12)])
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(RING_SPEC))

    plans = prg.plan_generation_runs(transforms_path, spec_path,
                                     front_azimuth_deg=0.0, min_separation_deg=20.0)
    for p in plans:
        assert p.views_per_layer % 4 == 0 or p.views_per_layer % 6 == 0
        assert len(p.layer_pitches) == 1
        assert 0.0 <= p.yaw_span <= 360.0
        assert -180.0 <= p.start_yaw <= 180.0
        assert p.target_azimuths_deg


def test_plan_generation_runs_finds_nothing_when_spec_matches_real_rig(tmp_path):
    """If the target spec's cameras coincide with the real rig (same
    azimuths, front_azimuth_deg correctly zeroed), there is nothing to
    generate."""
    azimuths = azimuths_on_circle(8)
    transforms_path = tmp_path / "transforms.json"
    _write_transforms(transforms_path, azimuths)
    spec = {
        "name": "t", "layout": "rings", "target": [0.0, 0.0, 0.0],
        "rings": [{"count": 8, "radius": 3.0, "height": 1.0}],
        "resolution": [640, 480],
        "intrinsics": {"lens_mm": 35, "sensor_width_mm": 36},
        "eval": {"count": 0},
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    # front_azimuth_deg=0 happens to line up here because both the rig-spec
    # (azimuth 0 = +X) and real_rig_azimuths (0 = toward the first camera,
    # which was placed at azimuth 0 too) share the same zero by construction.
    plans = prg.plan_generation_runs(transforms_path, spec_path,
                                     front_azimuth_deg=0.0, min_separation_deg=20.0)
    assert plans == []
