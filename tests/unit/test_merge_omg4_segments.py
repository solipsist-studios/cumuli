"""Unit tests for merge_omg4_segments.py."""

import json
import struct

import numpy as np
import pytest

import merge_omg4_segments as mos
from splat4d_io import OMG4_MAGIC, OMG4_V2_FIELDS, OMG4_V2_VERSION

N_BASE = len(OMG4_V2_FIELDS)


def write_segment(path, t_centers, *, opacity=2.0, t_sigma=0.05, velocity=0.0):
    """Minimal valid .omg4 v2 file whose Gaussians sit at the given t_centers."""
    n = len(t_centers)
    arrays = np.zeros((N_BASE, n), dtype=np.float32)
    arrays[mos.T_CENTER_ROW] = t_centers
    arrays[mos.T_SIGMA_ROW] = t_sigma
    arrays[mos.OPACITY_ROW] = opacity
    arrays[mos.VELOCITY_ROWS] = velocity
    with open(path, "wb") as fp:
        fp.write(struct.pack("<IIIIfffI", OMG4_MAGIC, OMG4_V2_VERSION, n, 0, 0.0, 1.0, 30.0, 0))
        fp.write(arrays.tobytes())
    return arrays


def write_manifest(path, segments, fade_zones=None):
    manifest = {"fps": 30.0, "time_range": [0.0, 1.0], "segments": segments}
    if fade_zones is not None:
        manifest["fade_zones"] = fade_zones
    path.write_text(json.dumps(manifest))
    return path


# --------------------------------------------------------------------------
# Field-index derivation
# --------------------------------------------------------------------------

def test_row_indices_match_the_documented_field_order():
    # Derived from splat4d_io's field list rather than hardcoded, so a format
    # change moves them together.
    assert OMG4_V2_FIELDS[mos.T_CENTER_ROW] == "t_center"
    assert OMG4_V2_FIELDS[mos.T_SIGMA_ROW] == "t_sigma"
    assert OMG4_V2_FIELDS[mos.OPACITY_ROW] == "opacity"
    assert OMG4_V2_FIELDS[mos.VELOCITY_ROWS] == ["vx", "vy", "vz"]


# --------------------------------------------------------------------------
# Round trip and partitioning
# --------------------------------------------------------------------------

def test_hard_merge_partitions_on_t_center(tmp_path):
    write_segment(tmp_path / "a.omg4", [0.1, 0.2, 0.9])  # 0.9 belongs to b's slot
    write_segment(tmp_path / "b.omg4", [0.05, 0.6, 0.7])  # 0.05 belongs to a's slot
    manifest = write_manifest(tmp_path / "m.json", [
        {"file": "a.omg4", "start": None, "end": 0.5},
        {"file": "b.omg4", "start": 0.5, "end": None},
    ])
    segments, fades, time_range, fps = mos.load_manifest(manifest, None)
    out = tmp_path / "merged.omg4"
    count = mos.merge_segments(segments, fades, "hard", time_range, fps, out, 0.15, 0.25)

    # each Gaussian is kept by exactly the segment owning its t_center
    assert count == 4
    merged = mos.read_omg4(out, {})
    assert sorted(np.round(merged["arrays"][mos.T_CENTER_ROW], 3)) == [0.1, 0.2, 0.6, 0.7]


def test_merged_output_is_sorted_by_t_center(tmp_path):
    write_segment(tmp_path / "a.omg4", [0.4, 0.1, 0.3])
    manifest = write_manifest(tmp_path / "m.json", [{"file": "a.omg4", "start": None, "end": None}])
    segments, fades, time_range, fps = mos.load_manifest(manifest, None)
    out = tmp_path / "merged.omg4"
    mos.merge_segments(segments, fades, "hard", time_range, fps, out, 0.15, 0.25)

    t_center = mos.read_omg4(out, {})["arrays"][mos.T_CENTER_ROW]
    assert np.all(np.diff(t_center) >= 0)


def test_header_carries_the_manifest_clip_range_not_the_segment_range(tmp_path):
    # Segment headers say [0, 1]; the merged clip must advertise the manifest's range,
    # otherwise playback clamps to whichever window happened to be written first.
    write_segment(tmp_path / "a.omg4", [0.5])
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"fps": 29.97, "time_range": [0.0, 2.5],
                                    "segments": [{"file": "a.omg4", "start": None, "end": None}]}))
    segments, fades, time_range, fps = mos.load_manifest(manifest, None)
    out = tmp_path / "merged.omg4"
    mos.merge_segments(segments, fades, "hard", time_range, fps, out, 0.15, 0.25)

    result = mos.read_omg4(out, {})
    assert (result["tmin"], result["tmax"]) == pytest.approx((0.0, 2.5))
    assert result["fps"] == pytest.approx(29.97)


def test_segment_dir_overrides_the_manifest_location(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    write_segment(staged / "a.omg4", [0.5])
    manifest = write_manifest(tmp_path / "m.json", [{"file": "a.omg4", "start": None, "end": None}])

    segments, _, _, _ = mos.load_manifest(manifest, staged)
    assert segments[0][0] == staged / "a.omg4"


def test_unbounded_ends_become_infinities(tmp_path):
    write_segment(tmp_path / "a.omg4", [0.5])
    manifest = write_manifest(tmp_path / "m.json", [{"file": "a.omg4", "start": None, "end": 0.5}])
    segments, _, _, _ = mos.load_manifest(manifest, None)
    assert segments[0][1] == -np.inf and segments[0][2] == 0.5


def test_truncated_file_is_rejected(tmp_path):
    path = tmp_path / "short.omg4"
    with open(path, "wb") as fp:
        fp.write(struct.pack("<IIIIfffI", OMG4_MAGIC, OMG4_V2_VERSION, 100, 0, 0.0, 1.0, 30.0, 0))
        fp.write(np.zeros((N_BASE, 3), dtype=np.float32).tobytes())  # 3 splats, not 100
    with pytest.raises(ValueError, match="truncated"):
        mos.read_omg4(path, {})


def test_wrong_version_is_rejected(tmp_path):
    path = tmp_path / "v1.omg4"
    with open(path, "wb") as fp:
        fp.write(struct.pack("<IIIIfffI", OMG4_MAGIC, 1, 0, 0, 0.0, 1.0, 30.0, 0))
    with pytest.raises(ValueError, match="v2"):
        mos.read_omg4(path, {})


def test_read_is_memoized_per_path(tmp_path):
    # A wide window split around a narrower one appears at two slots; reading it
    # twice from disk would double the cost of every stitch.
    write_segment(tmp_path / "a.omg4", [0.5])
    cache = {}
    first = mos.read_omg4(tmp_path / "a.omg4", cache)
    second = mos.read_omg4(tmp_path / "a.omg4", cache)
    assert first is second


# --------------------------------------------------------------------------
# Opacity ramps
# --------------------------------------------------------------------------

def test_scale_opacity_by_one_is_identity():
    logit = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    assert mos.scale_opacity(logit, np.ones(3)) == pytest.approx(logit, abs=1e-5)


def test_scale_opacity_halves_alpha_not_logit():
    logit = np.zeros(1, dtype=np.float32)  # alpha = 0.5
    scaled = mos.scale_opacity(logit, np.array([0.5]))
    alpha = 1.0 / (1.0 + np.exp(-scaled))
    assert alpha == pytest.approx(0.25, abs=1e-6)


def test_fade_mode_requires_one_zone_per_seam(tmp_path):
    write_segment(tmp_path / "a.omg4", [0.2])
    write_segment(tmp_path / "b.omg4", [0.8])
    manifest = write_manifest(tmp_path / "m.json", [
        {"file": "a.omg4", "start": None, "end": 0.5},
        {"file": "b.omg4", "start": 0.5, "end": None},
    ], fade_zones=[])
    segments, fades, time_range, fps = mos.load_manifest(manifest, None)
    with pytest.raises(ValueError, match="one fade zone per seam"):
        mos.merge_segments(segments, fades, "fade", time_range, fps,
                           tmp_path / "out.omg4", 0.15, 0.25)


def test_fade_mode_keeps_both_sides_alive_across_the_seam(tmp_path):
    # Gaussians sitting inside the fade zone must appear from BOTH segments --
    # that is the whole point of fade mode, and the "extras" pass is what does it.
    write_segment(tmp_path / "a.omg4", [0.45, 0.55])
    write_segment(tmp_path / "b.omg4", [0.45, 0.55])
    manifest = write_manifest(tmp_path / "m.json", [
        {"file": "a.omg4", "start": None, "end": 0.5},
        {"file": "b.omg4", "start": 0.5, "end": None},
    ], fade_zones=[[0.4, 0.6]])
    segments, fades, time_range, fps = mos.load_manifest(manifest, None)
    out = tmp_path / "merged.omg4"
    count = mos.merge_segments(segments, fades, "fade", time_range, fps, out, 0.15, 0.25)

    # a owns 0.45 and reaches forward to 0.55; b owns 0.55 and reaches back to 0.45
    assert count == 4
    hard_count = mos.merge_segments(segments, [], "hard", time_range, fps,
                                    tmp_path / "hard.omg4", 0.15, 0.25)
    assert hard_count == 2  # fade genuinely costs splats relative to a hard cut


# --------------------------------------------------------------------------
# Temporal sigma clamp
# --------------------------------------------------------------------------

def test_sigma_clamp_shrinks_a_tail_that_would_cross_the_seam():
    arrays = np.zeros((N_BASE, 1), dtype=np.float32)
    arrays[mos.T_CENTER_ROW] = 0.5
    arrays[mos.T_SIGMA_ROW] = 10.0   # absurdly wide: alive across the whole clip
    arrays[mos.OPACITY_ROW] = 4.0    # alpha ~ 0.98, so it stays visible for many sigmas
    clamped = mos.clamp_sigma_to_interval(arrays.copy(), 0.4, 0.6, max_drift=1e9)

    sigma = float(clamped[mos.T_SIGMA_ROW][0])
    assert sigma < 10.0
    # temporal alpha must be below 1/255 by the time it reaches the interval edge
    assert 0.98 * np.exp(-0.5 * (0.1 / sigma) ** 2) < 1 / 255


def test_sigma_clamp_leaves_an_already_narrow_tail_alone():
    arrays = np.zeros((N_BASE, 1), dtype=np.float32)
    arrays[mos.T_CENTER_ROW] = 0.5
    arrays[mos.T_SIGMA_ROW] = 0.001
    arrays[mos.OPACITY_ROW] = 2.0
    clamped = mos.clamp_sigma_to_interval(arrays.copy(), 0.4, 0.6, max_drift=1.0)
    assert float(clamped[mos.T_SIGMA_ROW][0]) == pytest.approx(0.001)


def test_velocity_drift_clamp_is_stricter_for_faster_gaussians():
    def clamped_sigma(speed):
        arrays = np.zeros((N_BASE, 1), dtype=np.float32)
        arrays[mos.T_CENTER_ROW] = 0.5
        arrays[mos.T_SIGMA_ROW] = 10.0
        arrays[mos.OPACITY_ROW] = 2.0
        arrays[mos.VELOCITY_ROWS] = np.array([[speed], [0.0], [0.0]])
        return float(mos.clamp_sigma_to_interval(arrays, -np.inf, np.inf, 0.15)[mos.T_SIGMA_ROW][0])

    # a static Gaussian is unconstrained by drift; doubling speed halves the allowance
    assert clamped_sigma(0.0) == pytest.approx(10.0)
    assert clamped_sigma(2.0) == pytest.approx(clamped_sigma(1.0) / 2, rel=1e-6)


def test_unbounded_interval_applies_no_temporal_clamp():
    arrays = np.zeros((N_BASE, 1), dtype=np.float32)
    arrays[mos.T_CENTER_ROW] = 0.5
    arrays[mos.T_SIGMA_ROW] = 3.0
    arrays[mos.OPACITY_ROW] = 2.0
    clamped = mos.clamp_sigma_to_interval(arrays.copy(), -np.inf, np.inf, max_drift=1e9)
    assert float(clamped[mos.T_SIGMA_ROW][0]) == pytest.approx(3.0)
