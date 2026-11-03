# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Degree-2 motion through the packer: the round trip and the re-anchor.

Getting these wrong is silent. A player that ignores accel still renders the
file, and a re-anchor that forgets a term only bends trajectories, so nothing
errors. The parabola fixture carries genuine second-order motion.
"""

import json
import zipfile

import numpy as np
import pytest

import make_sogst_fixture as fixture
import sogst_pack
from eval_render import decode_sogst_fields
from sogst_pack import (
    compute_sogst_order,
    kmeans_1d,
    motion_degree,
    pack_sogst,
    reanchor_degree2,
    snap_degree2_to_codebook,
    verify_sogst,
)

AXES = (("x", "vx", "ax"), ("y", "vy", "ay"), ("z", "vz", "az"))
SPLIT16 = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]


def scene(count=2048, degree=2, spread=False, seed=None):
    fields, meta = fixture.build_fixture(count=count, degree=degree, include_sh=False,
                                         **({} if seed is None else {"seed": seed}))
    fields = {k: np.asarray(v, np.float32) for k, v in fields.items()}
    if spread and degree == 2:
        # Accel that differs per splat and per axis, with both signs.
        rng = np.random.default_rng(5)
        for name in ("ax", "ay", "az"):
            fields[name] = rng.normal(0.0, 4.0, count).astype(np.float32)
    return fields, meta


def pack(path, fields, meta, **kw):
    return pack_sogst(str(path), fields, meta["time_min"], meta["time_max"], meta["fps"],
                      shn_count=0, **kw)


def trajectory(fields, t):
    """mean(t) exactly as spec section 1 states it, in float64."""
    dt = t - np.asarray(fields["t_center"], np.float64)
    out = []
    for pos, vel, acc in AXES:
        p = np.asarray(fields[pos], np.float64) + np.asarray(fields[vel], np.float64) * dt
        if acc in fields:
            p = p + np.asarray(fields[acc], np.float64) * dt * dt
        out.append(p)
    return np.stack(out, axis=1)


def split16_step(values):
    """Worst decoded error of one split-16 value, from spec section 2.2."""
    t = np.sign(values) * np.log1p(np.abs(values))
    span = max(float(t.max() - t.min()), 1e-12)
    return (1.0 + float(np.abs(values).max())) * (0.5 * span / 65535.0) * 1.01 + 1e-5


def test_motion_degree_comes_from_the_columns():
    fields, _ = scene(16, degree=1)
    assert motion_degree(fields) == 1
    fields, _ = scene(16)
    assert motion_degree(fields) == 2


@pytest.mark.parametrize("missing", ["ax", "ay", "az"])
def test_partial_accel_is_refused(tmp_path, missing):
    """Reading a partial set as degree 1 would silently drop the rest."""
    fields, meta = scene(64)
    del fields[missing]
    with pytest.raises(ValueError, match="all-or-nothing"):
        pack(tmp_path / "x.sogst", fields, meta)


@pytest.mark.parametrize("segment_duration, stream", [(0.1, True), (0.1, False), (0.0, True)])
def test_accel_round_trips_in_every_layout(tmp_path, segment_duration, stream):
    """Streamed, monolithic and unsegmented files all reorder every attribute
    with the same permutation, so a decoded row is still one source splat."""
    fields, meta = scene(spread=True)
    path = tmp_path / "x.sogst"
    written = pack(path, fields, meta, segment_duration=segment_duration, stream=stream)

    assert written["motion"]["degree"] == 2
    assert written["accel"]["files"] == ["accel_l.webp", "accel_u.webp"]
    assert len(written["accel"]["mins"]) == len(written["accel"]["maxs"]) == 3

    with zipfile.ZipFile(path) as zf:
        assert json.loads(zf.read("meta.json"))["motion"]["degree"] == 2
        names = zf.namelist()
    assert any(n.endswith("accel_l.webp") for n in names)
    assert any(n.endswith("accel_u.webp") for n in names)

    order, _ = compute_sogst_order(fields, meta["time_min"], meta["time_max"], segment_duration)
    _, decoded = decode_sogst_fields(str(path))
    for name in SPLIT16:
        source = fields[name][order]
        err = np.abs(np.asarray(decoded[name], np.float64) - source.astype(np.float64))
        assert err.max() <= split16_step(source), name


def test_accel_uses_the_split16_log_encoding_of_the_spec(tmp_path):
    """Decode the accel texture by hand with spec section 2.2's formula."""
    fields, meta = scene(256, spread=True)
    path = tmp_path / "x.sogst"
    written = pack(path, fields, meta, segment_duration=0.0)
    from PIL import Image
    import io
    with zipfile.ZipFile(path) as zf:
        lo = np.asarray(Image.open(io.BytesIO(zf.read("accel_l.webp"))).convert("RGBA"))
        hi = np.asarray(Image.open(io.BytesIO(zf.read("accel_u.webp"))).convert("RGBA"))
    assert (lo[..., 3] == 255).all() and (hi[..., 3] == 255).all()   # spec: alpha is 255
    n = written["count"]
    q = (hi[..., :3].reshape(-1, 3)[:n].astype(np.float64) * 256
         + lo[..., :3].reshape(-1, 3)[:n]) / 65535.0
    mins, maxs = np.array(written["accel"]["mins"]), np.array(written["accel"]["maxs"])
    t = mins + q * (maxs - mins)
    by_hand = np.sign(t) * (np.exp(np.abs(t)) - 1.0)

    order, _ = compute_sogst_order(fields, meta["time_min"], meta["time_max"], 0.0)
    for c, name in enumerate(("ax", "ay", "az")):
        source = fields[name][order]
        assert np.abs(by_hand[:, c] - source).max() <= split16_step(source), name
        # the mins and maxs are of the log transform, not of the raw values
        assert mins[c] == pytest.approx(np.sign(source.min()) * np.log1p(abs(source.min())))
        assert maxs[c] == pytest.approx(np.sign(source.max()) * np.log1p(abs(source.max())))


def test_constant_accel_axis_decodes_to_its_constant(tmp_path):
    """The fixture's ax and az are all zero and its ay is one constant: the
    zero-span case that spec section 2.2 sends through a span of 1.0."""
    fields, meta = scene(128)
    path = tmp_path / "x.sogst"
    pack(path, fields, meta)
    _, decoded = decode_sogst_fields(str(path))
    assert np.asarray(decoded["ax"]) == pytest.approx(0.0, abs=1e-9)
    assert np.asarray(decoded["ay"]) == pytest.approx(0.5 * fixture.GRAVITY, abs=1e-4)


def test_degree_1_input_writes_no_accel(tmp_path):
    """Spec section 4.8: never synthesised."""
    fields, meta = scene(128, degree=1)
    path = tmp_path / "x.sogst"
    written = pack(path, fields, meta)
    assert written["motion"]["degree"] == 1
    assert "accel" not in written
    with zipfile.ZipFile(path) as zf:
        assert not [n for n in zf.namelist() if "accel" in n]
    assert "ax" not in decode_sogst_fields(str(path))[1]


def test_verify_reports_accel(tmp_path):
    fields, meta = scene(256, spread=True)
    path = tmp_path / "x.sogst"
    order = compute_sogst_order(fields, meta["time_min"], meta["time_max"])
    pack(path, fields, meta, order_segments=order)
    report = verify_sogst(str(path), fields, order[0])
    assert {"accel.ax", "accel.ay", "accel.az"} <= set(report)
    assert report["accel.ay"][2] <= split16_step(fields["ay"])


def test_reanchor_matches_the_algebra_by_hand():
    fields = {"t_center": np.array([1.0], np.float32),
              "x": np.array([2.0], np.float32), "vx": np.array([3.0], np.float32),
              "ax": np.array([5.0], np.float32),
              "y": np.zeros(1, np.float32), "vy": np.zeros(1, np.float32),
              "ay": np.zeros(1, np.float32),
              "z": np.zeros(1, np.float32), "vz": np.zeros(1, np.float32),
              "az": np.zeros(1, np.float32)}
    out = reanchor_degree2(fields, np.array([1.5]))
    assert out["t_center"][0] == 1.5
    assert out["x"][0] == pytest.approx(2.0 + 3.0 * 0.5 + 5.0 * 0.25)   # x + v*d + a*d^2
    assert out["vx"][0] == pytest.approx(3.0 + 2.0 * 5.0 * 0.5)         # v + 2*a*d
    assert out["ax"][0] == 5.0                                          # a unchanged
    assert fields["x"][0] == 2.0                                        # input untouched


@pytest.mark.parametrize("scale", [1.0, 500.0])
def test_reanchor_leaves_the_trajectory_alone(scale):
    fields, _ = scene(1024, spread=True)
    for name in ("ax", "ay", "az"):
        fields[name] = fields[name] * scale
    rng = np.random.default_rng(1)
    moved = fields["t_center"] + rng.uniform(-0.2, 0.2, 1024).astype(np.float32)
    out = reanchor_degree2(fields, moved)
    for t in (0.0, 0.73, 2.0):
        worst = np.abs(trajectory(fields, t) - trajectory(out, t)).max()
        assert worst <= 2e-5 * scale


def test_snap_lands_on_the_codebook_the_packer_keeps(tmp_path):
    """The whole point: the re-anchor has to survive the pack. k-means on
    snapped values collapses sparse levels, so the packer must keep them."""
    fields, meta = scene(4096, spread=True)
    snapped, moved = snap_degree2_to_codebook(fields)
    assert moved > 0.0
    assert len(np.unique(snapped["t_center"])) <= 256

    path = tmp_path / "x.sogst"
    written = pack(path, snapped, meta)
    assert len(written["trbf"]["center"]["codebook"]) == 256

    order, _ = compute_sogst_order(snapped, meta["time_min"], meta["time_max"])
    _, decoded = decode_sogst_fields(str(path))
    np.testing.assert_array_equal(
        np.asarray(decoded["t_center"], np.float32), snapped["t_center"][order])
    for t in (0.0, 0.5, 1.3, 2.0):
        err = np.abs(trajectory({k: v[order] for k, v in fields.items()}, t)
                     - trajectory(decoded, t))
        assert err.max() < 0.01                 # split-16 and float32 only


def test_packing_without_a_snap_keeps_positions_as_given(tmp_path):
    """Spec section 10: decoded split-plane fields follow from the input
    alone, because t_center's codebook is implementation-defined. A
    compensation inside the packer would break that comparison."""
    fields, meta = scene(4096, spread=True)
    path = tmp_path / "x.sogst"
    pack(path, fields, meta)
    order, _ = compute_sogst_order(fields, meta["time_min"], meta["time_max"])
    _, decoded = decode_sogst_fields(str(path))
    for name in ("x", "y", "z", "vx", "vy", "vz"):
        source = fields[name][order]
        err = np.abs(np.asarray(decoded[name]) - source)
        assert err.max() <= split16_step(source), name


def test_degree_1_time_codes_are_always_clustered():
    """Keeping levels exactly is degree-2 only, so degree-1 archives stay
    byte-identical to what they were."""
    t = np.repeat(np.linspace(0.0, 1.0, 40), np.arange(1, 41) ** 2).astype(np.float32)
    want_cb, want_idx = kmeans_1d(t)
    cb, idx = sogst_pack.time_center_codes(t, 1)
    np.testing.assert_array_equal(cb, want_cb)
    np.testing.assert_array_equal(idx, want_idx)


def test_degree_2_clusters_when_there_are_too_many_levels():
    t = np.linspace(0.0, 1.0, 1000).astype(np.float32)
    cb, idx = sogst_pack.time_center_codes(t, 2)
    want_cb, want_idx = kmeans_1d(t)
    np.testing.assert_array_equal(cb, want_cb)
    np.testing.assert_array_equal(idx, want_idx)


def test_degree_2_keeps_levels_and_pads_to_a_full_codebook():
    t = np.repeat(np.linspace(0.0, 1.0, 40), np.arange(1, 41) ** 2).astype(np.float32)
    cb, idx = sogst_pack.time_center_codes(t, 2)
    assert len(cb) == 256
    assert (np.diff(cb) > 0).all()
    np.testing.assert_array_equal(cb[idx].astype(np.float32), t)
