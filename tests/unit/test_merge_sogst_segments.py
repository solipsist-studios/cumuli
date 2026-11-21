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
import sys
import zipfile

import numpy as np
import pytest

import merge_sogst_segments as msm
from sogst_pack import compute_sogst_order


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


# ---------------------------------------------------------------------------
# Degree-2 motion: the raw dt^2 coefficient, spec sections 1, 4.8 and 7
# ---------------------------------------------------------------------------

AXES = (("x", "vx", "ax"), ("y", "vy", "ay"), ("z", "vz", "az"))


def with_accel(fields, scale=1.0, seed=7):
    """Random per-splat accel, row-tagged so misalignment cannot hide."""
    n = len(fields["x"])
    rng = np.random.default_rng(seed)
    out = dict(fields)
    for _, _, acc in AXES:
        out[acc] = (rng.standard_normal(n) * scale).astype(np.float32)
    return out


def moving_degree2(n=500, t_hi=5.0, scale=1.0, seed=3):
    rng = np.random.default_rng(seed)
    fields = with_accel(make_fields(n, 0.0, with_sh=False), scale=scale, seed=seed + 1)
    fields["t_center"] = rng.uniform(0.0, t_hi, n).astype(np.float32)
    for pos, vel, _ in AXES:
        fields[pos] = rng.standard_normal(n).astype(np.float32)
        fields[vel] = (rng.standard_normal(n) * 2.0).astype(np.float32)
    return fields


def trajectory(fields, t):
    """mean(t) as spec section 1 states it, in float64."""
    dt = t - np.asarray(fields["t_center"], np.float64)
    out = []
    for pos, vel, acc in AXES:
        p = np.asarray(fields[pos], np.float64) + np.asarray(fields[vel], np.float64) * dt
        if acc in fields:
            p = p + np.asarray(fields[acc], np.float64) * dt * dt
        out.append(p)
    return np.stack(out, axis=1)


def test_merge_and_to_pack_fields_keep_accel_on_its_own_row():
    """The bug this guards: to_pack_fields kept only SOGST_FIELDS and f_rest,
    so a degree-2 stitch silently came out degree 1."""
    first = with_accel(make_fields(6, 0.5))
    second = with_accel(make_fields(6, 1.9, seed=1), seed=8)
    segs = [segment(first, 0.0, 1.25), segment(second, 1.2917, 2.5)]
    merged, _ = msm.merge(segs, msm.default_seams(segs), "hard", 0.08)
    fields = msm.to_pack_fields(merged)
    for name in ("ax", "ay", "az"):
        np.testing.assert_array_equal(
            fields[name], np.concatenate([first[name], second[name]]))
        assert fields[name].dtype == np.float32


def test_selection_drops_accel_with_its_row_in_both_modes():
    fields = with_accel(make_fields(100, 0.0, with_sh=False))
    fields["t_center"] = np.linspace(0.0, 2.5, 100).astype(np.float32)
    other = with_accel(make_fields(100, 0.0, with_sh=False), seed=9)
    other["t_center"] = np.linspace(0.0, 2.5, 100).astype(np.float32)
    segs = [segment(fields, 0.0, 1.25), segment(other, 1.25, 2.5)]
    for mode in ("hard", "fade"):
        merged, report = msm.merge(segs, [1.25], mode, 0.2)
        n_first = report[0]["splats_kept"]
        assert 0 < n_first < 100
        tc_first = merged["t_center"][:n_first]
        kept = np.isin(fields["t_center"], tc_first)
        assert kept.sum() == n_first
        for name in ("ax", "ay", "az"):
            np.testing.assert_array_equal(merged[name][:n_first], fields[name][kept])


def test_snap_time_preserves_a_degree_2_trajectory_where_velocity_alone_cannot():
    """Slide xyz by velocity alone and the splat leaves its parabola."""
    merged = moving_degree2()
    before = {k: v.copy() for k, v in merged.items()}
    snapped, moved = msm.snap_time_centers(dict(merged))
    assert moved > 0.0
    assert not np.array_equal(snapped["t_center"], before["t_center"])
    # a is unchanged, v takes 2*a*delta
    delta = snapped["t_center"].astype(np.float64) - before["t_center"].astype(np.float64)
    np.testing.assert_array_equal(snapped["ax"], before["ax"])
    np.testing.assert_allclose(
        snapped["vx"], before["vx"].astype(np.float64)
        + 2.0 * before["ax"].astype(np.float64) * delta, atol=1e-5)

    naive = dict(before)
    for pos, vel, _ in AXES:
        naive[pos] = before[pos] + before[vel] * delta.astype(np.float32)
    naive["t_center"] = snapped["t_center"]

    for t in (0.0, 1.3, 2.7, 5.0):
        assert np.abs(trajectory(before, t) - trajectory(snapped, t)).max() < 1e-4
    worst_naive = max(np.abs(trajectory(before, t) - trajectory(naive, t)).max()
                      for t in (0.0, 5.0))
    assert worst_naive > 1e-3                       # the old formula is measurably wrong


def test_snap_time_preserves_the_trajectory_at_very_large_accel():
    merged = moving_degree2(scale=2000.0)
    before = {k: v.copy() for k, v in merged.items()}
    snapped, _ = msm.snap_time_centers(dict(merged))
    for t in (0.0, 1.3, 2.7, 5.0):
        reach = np.abs(trajectory(before, t)).max()
        worst = np.abs(trajectory(before, t) - trajectory(snapped, t)).max()
        assert worst < max(1e-5 * reach, 1e-3)


def test_degree_1_snap_keeps_its_float32_arithmetic_and_grows_no_accel():
    """Degree-1 archives must not change, so this pins the exact arithmetic."""
    merged = make_fields(300, 0.0, with_sh=False)
    rng = np.random.default_rng(11)
    merged["t_center"] = rng.uniform(0.0, 5.0, 300).astype(np.float32)
    merged["x"] = rng.standard_normal(300).astype(np.float32)
    merged["vx"] = rng.standard_normal(300).astype(np.float32)
    before = {k: v.copy() for k, v in merged.items()}

    snapped, _ = msm.snap_time_centers(merged)
    tc = before["t_center"].astype(np.float64)
    codebook, idx = msm.kmeans_1d(tc)
    delta = (codebook[idx] - tc).astype(np.float32)
    np.testing.assert_array_equal(snapped["x"], before["x"] + before["vx"] * delta)
    np.testing.assert_array_equal(snapped["vx"], before["vx"])
    np.testing.assert_array_equal(snapped["t_center"], codebook[idx].astype(np.float32))
    assert "ax" not in snapped
    assert "ax" not in msm.to_pack_fields(snapped)


@pytest.mark.parametrize("flip", [False, True])
def test_mixed_degree_windows_are_refused_rather_than_synthesised(flip):
    a = with_accel(make_fields(3, 0.5))
    b = make_fields(3, 1.9)
    pair = [segment(a, 0.0, 1.25), segment(b, 1.2917, 2.5)]
    if flip:
        pair = pair[::-1]
    with pytest.raises(SystemExit, match="mix motion degrees") as caught:
        msm.merge(pair, [1.27], "hard", 0.08)
    assert "degree 1" in str(caught.value) and "degree 2" in str(caught.value)
    assert "4.8" in str(caught.value)


def test_a_window_with_partial_accel_is_refused_by_name():
    a = with_accel(make_fields(3, 0.5))
    del a["ay"]
    segs = [segment(a, 0.0, 1.25), segment(with_accel(make_fields(3, 1.9)), 1.2917, 2.5)]
    with pytest.raises(SystemExit, match="all-or-nothing"):
        msm.merge(segs, [1.27], "hard", 0.08)


def split16_error(values):
    """The largest decoding error of a split-16 value, from spec section 2.2."""
    t = np.sign(values) * np.log1p(np.abs(values))
    half_step = 0.5 * float(t.max() - t.min()) / 65535.0
    return (1.0 + float(np.abs(values).max())) * half_step


def write_window(path, count, seed, scale=1.0, t_hi=1.25):
    """A degree-2 window on disk, with accel big enough to matter."""
    fields = moving_degree2(n=count, t_hi=t_hi, scale=scale, seed=seed)
    fields["t_sigma"] = np.full(count, 0.15, np.float32)
    fields["opacity"] = np.full(count, 2.0, np.float32)
    msm.pack_sogst(str(path), fields, 0.0, t_hi, 24.0, shn_count=0)
    return path


def test_load_segment_shift_moves_only_t_center(tmp_path):
    """Shifting the time origin leaves dt alone, so nothing else may change,
    accel included, and the shifted trajectory is the same curve."""
    path = write_window(tmp_path / "w.sogst", 400, seed=21)
    still = msm.load_segment(path, 0.0)
    moved = msm.load_segment(path, 3.0)
    assert (moved["t_lo"], moved["t_hi"]) == (3.0, 4.25)
    for name, values in still["fields"].items():
        if name == "t_center":
            np.testing.assert_array_equal(values + np.float32(3.0), moved["fields"][name])
        else:
            np.testing.assert_array_equal(values, moved["fields"][name], err_msg=name)
    for t in (0.0, 0.6, 1.25):
        worst = np.abs(trajectory(still["fields"], t)
                       - trajectory(moved["fields"], t + 3.0)).max()
        assert worst < 1e-4


@pytest.mark.parametrize("scale", [1.0, 200.0])
def test_stitched_archive_decodes_to_the_window_trajectories(tmp_path, scale):
    """End to end through the real packer: decoded rows match the windows.
    With scale=200 the velocity-only snap misses by whole scene units."""
    a = write_window(tmp_path / "a.sogst", 1500, seed=31, scale=scale)
    b = write_window(tmp_path / "b.sogst", 1500, seed=41, scale=scale)
    segs = [msm.load_segment(a, 0.0), msm.load_segment(b, 1.2916667)]
    merged, _ = msm.merge(segs, msm.default_seams(segs), "fade", 0.35)
    source = {k: v.copy() for k, v in merged.items()}
    snapped, _ = msm.snap_time_centers(merged)
    fields = msm.to_pack_fields(snapped)

    order, table = compute_sogst_order(fields, 0.0, 2.5416667)
    out = tmp_path / "stitched.sogst"
    written = msm.pack_sogst(str(out), fields, 0.0, 2.5416667, 24.0, shn_count=0,
                             order_segments=(order, table))
    assert written["motion"]["degree"] == 2
    assert written["accel"]["files"] == ["accel_l.webp", "accel_u.webp"]

    _, decoded = msm.decode_sogst_fields(str(out))
    aligned = {k: v[order] for k, v in source.items()}
    # The 16-bit log coding of x, v and a is the only error left, so the bar
    # comes from the codec (spec section 2.2) rather than from a chosen number.
    for t in (0.0, 0.7, 1.3, 1.9, 2.5):
        dt = np.abs(t - fields["t_center"].astype(np.float64)).max()
        for pos, vel, acc in AXES:
            bar = (split16_error(fields[pos]) + split16_error(fields[vel]) * dt
                   + split16_error(fields[acc]) * dt * dt) * 1.05 + 1e-4
            c = "xyz".index(pos)
            worst = np.abs(trajectory(aligned, t)[:, c] - trajectory(decoded, t)[:, c]).max()
            assert worst < bar, (t, pos, worst, bar)


def test_cli_stitches_degree_2_windows_with_no_extra_flags(tmp_path, monkeypatch):
    a = write_window(tmp_path / "a.sogst", 800, seed=51)
    b = write_window(tmp_path / "b.sogst", 800, seed=61)
    out = tmp_path / "stitched.sogst"
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "merge", "--segment", str(a), "0.0", "--segment", str(b), "1.2916667",
        "--out", str(out), "--report_json", str(report)])
    msm.main()
    with zipfile.ZipFile(out) as zf:
        meta = json.loads(zf.read("meta.json"))
        assert any(n.endswith("/accel_l.webp") for n in zf.namelist())
        assert any(n.endswith("/accel_u.webp") for n in zf.namelist())
    assert meta["motion"]["degree"] == 2
    assert meta["accel"]["files"] == ["accel_l.webp", "accel_u.webp"]
    assert json.loads(report.read_text())["motion_degree"] == 2
    assert "ax" in msm.decode_sogst_fields(str(out))[1]


def test_cli_refuses_mixed_windows(tmp_path, monkeypatch):
    a = write_window(tmp_path / "a.sogst", 200, seed=71)
    plain = moving_degree2(n=200, t_hi=1.25, seed=81)
    for name in ("ax", "ay", "az"):
        del plain[name]
    b = tmp_path / "b.sogst"
    msm.pack_sogst(str(b), plain, 0.0, 1.25, 24.0, shn_count=0)
    out = tmp_path / "stitched.sogst"
    monkeypatch.setattr(sys, "argv", [
        "merge", "--segment", str(a), "0.0", "--segment", str(b), "1.2916667",
        "--out", str(out)])
    with pytest.raises(SystemExit, match="mix motion degrees"):
        msm.main()
    assert not out.exists()


def test_degree_1_windows_stitch_to_degree_1(tmp_path, monkeypatch):
    """No accel in, none out: spec section 4.8 forbids synthesising it."""
    paths = []
    for i, seed in enumerate((91, 101)):
        plain = moving_degree2(n=200, t_hi=1.25, seed=seed)
        for name in ("ax", "ay", "az"):
            del plain[name]
        paths.append(tmp_path / f"{i}.sogst")
        msm.pack_sogst(str(paths[-1]), plain, 0.0, 1.25, 24.0, shn_count=0)
    out = tmp_path / "stitched.sogst"
    monkeypatch.setattr(sys, "argv", [
        "merge", "--segment", str(paths[0]), "0.0", "--segment", str(paths[1]), "1.2916667",
        "--out", str(out)])
    msm.main()
    with zipfile.ZipFile(out) as zf:
        meta = json.loads(zf.read("meta.json"))
        assert not [n for n in zf.namelist() if "accel" in n]
    assert meta["motion"]["degree"] == 1
    assert "accel" not in meta
