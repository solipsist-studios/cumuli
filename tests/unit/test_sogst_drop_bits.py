# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Encoder-side precision reduction of the 16-bit split planes.

Dropping low bits snaps each 16-bit integer to a coarser grid.  The decode
is unchanged (spec section 2.2), so a player cannot tell: these tests pin
that the decoded error stays inside the bound split16_step() predicts, that
zero dropped bits is byte-identical to the old encoder, and that the low
byte really does lose entropy (the point of the change).
"""

import zipfile

import numpy as np
import pytest

import make_sogst_fixture as fixture
from eval_render import decode_sogst_fields
from sogst_pack import compute_sogst_order, pack_sogst, split16, split16_step


def test_zero_drop_bits_is_the_old_quantizer():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(5000, 3))
    mins, maxs = t.min(0), t.max(0)
    span = maxs - mins
    q = np.clip(np.rint((t - mins) / span * 65535.0), 0, 65535).astype(np.uint32)
    lo, hi = split16(t, mins, maxs, 0)
    assert np.array_equal(lo, (q & 0xFF).astype(np.uint8))
    assert np.array_equal(hi, (q >> 8).astype(np.uint8))


@pytest.mark.parametrize("bits", [1, 3, 6, 8])
def test_dropped_bits_are_zero_and_top_stays_in_range(bits):
    rng = np.random.default_rng(1)
    t = rng.normal(size=(5000, 3))
    t[0] = t.max(0)  # a value on the top of the range must not overflow 16 bits
    lo, hi = split16(t, t.min(0), t.max(0), bits)
    q = hi.astype(np.uint32) * 256 + lo
    assert np.all(q % (1 << bits) == 0)
    assert q.max() <= 65535
    assert len(np.unique(lo)) <= 256 >> bits


def test_per_axis_bits():
    rng = np.random.default_rng(2)
    t = rng.normal(size=(4000, 3))
    lo, hi = split16(t, t.min(0), t.max(0), [0, 4, 8])
    q = hi.astype(np.uint32) * 256 + lo
    assert len(np.unique(q[:, 0] % 2)) == 2
    assert np.all(q[:, 1] % 16 == 0)
    assert np.all(q[:, 2] % 256 == 0)


def _pack(tmp_path, name, drop_bits):
    fields, meta = fixture.build_fixture(count=3000, degree=2, include_sh=False, seed=7)
    order_segments = compute_sogst_order(fields, meta["time_min"], meta["time_max"], 0.1)
    path = tmp_path / name
    pack_sogst(str(path), fields, meta["time_min"], meta["time_max"], meta["fps"],
               shn_count=0, order_segments=order_segments, drop_bits=drop_bits)
    return path, fields, order_segments[0]


def test_default_is_byte_identical_to_zero_bits(tmp_path):
    a, _, _ = _pack(tmp_path, "a.sogst", None)
    b, _, _ = _pack(tmp_path, "b.sogst", {"means": 0, "motion": 0, "accel": 0})
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("group, cols", [("means", ("x", "y", "z")),
                                         ("motion", ("vx", "vy", "vz")),
                                         ("accel", ("ax", "ay", "az"))])
def test_decoded_error_within_predicted_bound(tmp_path, group, cols):
    bits = 5
    path, fields, order = _pack(tmp_path, f"{group}.sogst", {group: bits})
    _h, dec = decode_sogst_fields(str(path))
    src = np.stack([fields[c][order] for c in cols], 1).astype(np.float64)
    got = np.stack([dec[c] for c in cols], 1).astype(np.float64)
    bound = 0.5 * split16_step(src, bits)
    # Rounding twice (16-bit, then the coarse grid) adds at most half a
    # 16-bit step to the coarse half step.  The 1% covers the exponential's
    # curvature, which split16_step()'s linear estimate leaves out.
    full = 0.5 * split16_step(src, 0)
    assert np.all(np.abs(got - src) <= 1.01 * (bound + full))


def test_manifest_records_dropped_bits_only_when_set(tmp_path):
    import json
    full, _, _ = _pack(tmp_path, "full.sogst", None)
    coarse, _, _ = _pack(tmp_path, "coarse.sogst", {"motion": 6, "accel": [0, 2, 4]})
    with zipfile.ZipFile(full) as z:
        m = json.loads(z.read("meta.json"))
    assert all("drop_bits" not in m[g] for g in ("means", "motion", "accel"))
    with zipfile.ZipFile(coarse) as z:
        m = json.loads(z.read("meta.json"))
    assert "drop_bits" not in m["means"]
    assert m["motion"]["drop_bits"] == [6, 6, 6]
    assert m["accel"]["drop_bits"] == [0, 2, 4]


def test_low_byte_plane_shrinks(tmp_path):
    full, _, _ = _pack(tmp_path, "full.sogst", None)
    coarse, _, _ = _pack(tmp_path, "coarse.sogst", {"motion": 6})

    def size(path, name):
        with zipfile.ZipFile(path) as z:
            return sum(i.compress_size for i in z.infolist() if i.filename.endswith(name))

    assert size(coarse, "motion_l.webp") < 0.5 * size(full, "motion_l.webp")
    # the stretched range (see _split16_range) moves the high bytes a little
    assert abs(size(coarse, "motion_u.webp") / size(full, "motion_u.webp") - 1) < 0.02
