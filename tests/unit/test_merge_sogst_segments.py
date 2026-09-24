# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Tests for stitching windowed .sogst models into one clip.

The merge has two failure modes that cost real quality before they were
caught, and both are silent: dropping the higher-order spherical harmonics
because the decoder hands them back under a different name than the packer
wants, and letting each window's splats render across the whole clip
through the tails of their temporal Gaussians. Each has a test here that
fails if the guard is removed.
"""

import json

import numpy as np
import pytest

import merge_sogst_segments as msm


def make_fields(n, t_center, t_sigma=0.1, speed=1.0, with_sh=True, seed=0):
    """Minimal field dictionary in the shape load_segment produces."""
    rng = np.random.default_rng(seed)
    fields = {
        "x": np.zeros(n, np.float32), "y": np.zeros(n, np.float32),
        "z": np.zeros(n, np.float32),
        "vx": np.full(n, speed, np.float32), "vy": np.zeros(n, np.float32),
        "vz": np.zeros(n, np.float32),
        "opacity": np.zeros(n, np.float32),
        "rot_0": np.ones(n, np.float32),
        "rot_1": np.zeros(n, np.float32), "rot_2": np.zeros(n, np.float32),
        "rot_3": np.zeros(n, np.float32),
        "scale_0": np.full(n, -3.0, np.float32),
        "scale_1": np.full(n, -3.0, np.float32),
        "scale_2": np.full(n, -3.0, np.float32),
        "f_dc_0": np.zeros(n, np.float32), "f_dc_1": np.zeros(n, np.float32),
        "f_dc_2": np.zeros(n, np.float32),
        "t_center": np.asarray(t_center, np.float32) * np.ones(n, np.float32),
        "t_sigma": np.asarray(t_sigma, np.float32) * np.ones(n, np.float32),
    }
    if with_sh:
        fields["f_rest"] = rng.standard_normal((n, 45)).astype(np.float32)
    return fields


def segment(fields, t_lo, t_hi):
    return {"path": "w/splat_4d.sogst", "offset": t_lo, "fields": fields,
            "count": len(fields["x"]), "t_lo": t_lo, "t_hi": t_hi, "fps": 24.0}


def test_seams_fall_midway_between_windows():
    segs = [segment(make_fields(4, 0.5), 0.0, 1.25),
            segment(make_fields(4, 1.9), 1.2917, 2.5),
            segment(make_fields(4, 3.1), 2.5417, 3.75)]
    seams = msm.default_seams(segs)
    assert seams == pytest.approx([1.27085, 2.52085], abs=1e-4)


def test_hard_partition_gives_each_instant_one_owner():
    segs = [segment(make_fields(3, 0.5), 0.0, 1.25),
            segment(make_fields(3, 1.9), 1.2917, 2.5)]
    seams = msm.default_seams(segs)
    owners = 0
    for (lo, hi) in msm.slot_bounds(segs, seams):
        owners += int(lo <= 0.5 < hi)
    assert owners == 1


def test_gate_stops_a_splat_rendering_outside_its_slot():
    """A long-lived splat is what streaks across a neighbouring window."""
    fields = make_fields(1, t_center=4.4, t_sigma=8.0)
    alpha_before = np.exp(-0.5 * ((0.0 - 4.4) / 8.0) ** 2)
    assert alpha_before > 0.8                       # renders at t=0 unchecked

    gated = msm.gate_tails(dict(fields), 3.77, np.inf, k=2.0, floor=0.4)
    sigma = float(np.abs(gated["t_sigma"][0]))
    assert sigma < 0.4
    alpha_after = np.exp(-0.5 * ((0.0 - 4.4) / sigma) ** 2)
    assert alpha_after < 1e-9


def test_gate_leaves_a_short_lived_splat_alone():
    fields = make_fields(1, t_center=4.4, t_sigma=0.08)
    gated = msm.gate_tails(dict(fields), 3.77, np.inf, k=2.0, floor=0.4)
    assert float(gated["t_sigma"][0]) == pytest.approx(0.08)


def test_gate_is_off_when_k_is_zero():
    fields = make_fields(1, t_center=4.4, t_sigma=8.0)
    gated = msm.gate_tails(dict(fields), 3.77, np.inf, k=0.0, floor=0.4)
    assert float(gated["t_sigma"][0]) == pytest.approx(8.0)


def test_snapping_time_leaves_every_position_unchanged():
    """Moving t_center must slide xyz by v * delta, or the splat teleports."""
    rng = np.random.default_rng(3)
    n = 500
    merged = make_fields(n, 0.0, with_sh=False)
    merged["t_center"] = rng.uniform(0.0, 5.0, n).astype(np.float32)
    merged["x"] = rng.standard_normal(n).astype(np.float32)
    merged["vx"] = rng.standard_normal(n).astype(np.float32) * 2.0

    before_tc = merged["t_center"].copy()
    before_x = merged["x"].copy()
    before_v = merged["vx"].copy()

    snapped, moved = msm.snap_time_centers(merged)
    assert moved >= 0.0
    for t in (0.0, 1.3, 2.7, 5.0):
        was = before_x + before_v * (t - before_tc)
        now = snapped["x"] + snapped["vx"] * (t - snapped["t_center"])
        assert np.allclose(was, now, atol=1e-4)


def test_merge_keeps_spherical_harmonics():
    """The decoder returns one f_rest block, not f_rest_0..f_rest_44."""
    segs = [segment(make_fields(6, 0.5), 0.0, 1.25),
            segment(make_fields(6, 1.9, seed=1), 1.2917, 2.5)]
    merged, _ = msm.merge(segs, msm.default_seams(segs), "hard", 0.08)
    fields = msm.to_pack_fields(merged)
    assert "f_rest" in fields
    assert fields["f_rest"].shape[1] == 45


def test_fade_halves_opacity_at_the_seam():
    fields = make_fields(1, t_center=1.27, t_sigma=0.05)
    keep, weight = msm.select_fade(segment(fields, 0.0, 1.25), -np.inf, 1.27,
                                   fade=0.2)
    assert bool(keep[0])
    assert float(weight[0]) == pytest.approx(0.5, abs=0.05)


def test_apply_weight_scales_alpha_not_the_logit():
    fields = make_fields(4, 0.5)
    fields["opacity"] = np.full(4, 2.0, np.float32)   # alpha ~0.8808
    out = msm.apply_weight(dict(fields), np.full(4, 0.25))
    alpha = 1.0 / (1.0 + np.exp(-out["opacity"].astype(np.float64)))
    assert alpha == pytest.approx(np.full(4, 0.8808 * 0.25), abs=1e-3)


def test_mismatched_field_sets_are_refused():
    a = make_fields(3, 0.5)
    b = make_fields(3, 1.9, with_sh=False)
    segs = [segment(a, 0.0, 1.25), segment(b, 1.2917, 2.5)]
    with pytest.raises(SystemExit, match="different field sets"):
        msm.merge(segs, msm.default_seams(segs), "hard", 0.08)


def test_plan_supplies_segments_and_seams(tmp_path):
    """A plan already knows the cuts and the time origins, so retyping them
    into --segment is a chance to get one wrong."""
    plan = {
        "fps": 24.0,
        "windows": [
            {"index": 0, "offset_seconds": 0.0, "out_dir": str(tmp_path / "w0")},
            {"index": 1, "offset_seconds": 1.2916667,
             "out_dir": str(tmp_path / "w1")},
        ],
        "seams_seconds": [1.2708333],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    pairs, seams = msm.segments_from_plan(path)
    assert [p for p, _ in pairs] == [str(tmp_path / "w0" / "splat_4d.sogst"),
                                     str(tmp_path / "w1" / "splat_4d.sogst")]
    assert [o for _, o in pairs] == [0.0, 1.2916667]
    assert seams == [1.2708333]


def test_plan_without_out_dirs_is_refused(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"windows": [
        {"index": 0, "offset_seconds": 0.0, "out_dir": None},
        {"index": 1, "offset_seconds": 1.0, "out_dir": str(tmp_path)},
    ]}))
    with pytest.raises(SystemExit, match="no out_dir"):
        msm.segments_from_plan(path)


def test_plan_model_name_is_configurable(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"windows": [
        {"index": 0, "offset_seconds": 0.0, "out_dir": str(tmp_path / "w0")},
    ]}))
    pairs, _ = msm.segments_from_plan(path, model_name="other.sogst")
    assert pairs[0][0].endswith("other.sogst")
