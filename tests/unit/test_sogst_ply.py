# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Tests for the .sogst interchange PLY (docs/sogst-format.md section 7).

The PLY is a contract between two implementations in two languages, so the
tests that matter here are the ones covering conventions that fail
*silently* when broken: a missing clip scalar, a partially-present optional
block, a column-length mismatch, and the exact column order the encoder
relies on.
"""

import json

import numpy as np
import pytest

import make_sogst_fixture as fixture
import sogst_ply
from sogst_ply import (
    PLY_ACCEL_COLUMNS,
    PLY_BASE_COLUMNS,
    PLY_SH_COLUMNS,
    read_sogst_ply,
    write_sogst_ply,
    write_sogst_sidecar,
)


def make_fields(n=64, degree=1, include_sh=False, seed=3):
    fields, _meta = fixture.build_fixture(
        count=n, degree=degree, include_sh=include_sh, seed=seed)
    return fields


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("include_sh", [False, True])
def test_round_trip_is_bit_exact(tmp_path, degree, include_sh):
    fields = make_fields(degree=degree, include_sh=include_sh)
    path = tmp_path / "scene.ply"

    n = write_sogst_ply(str(path), fields, 0.0, 2.0, 24.0)
    header, back = read_sogst_ply(str(path))

    assert n == header["count"] == 64
    assert header["time_min"] == 0.0
    assert header["time_max"] == 2.0
    assert header["fps"] == 24.0
    assert header["motion_degree"] == degree

    for name in PLY_BASE_COLUMNS:
        expected = np.asarray(fields[name], dtype=np.float32)
        np.testing.assert_array_equal(back[name], expected, err_msg=name)

    assert ("f_rest" in back) is include_sh
    if include_sh:
        np.testing.assert_array_equal(
            back["f_rest"], np.asarray(fields["f_rest"], dtype=np.float32))

    assert all((c in back) is (degree == 2) for c in PLY_ACCEL_COLUMNS)


def test_column_order_matches_the_spec(tmp_path):
    """The encoder reads columns by name, but a reordered header is still a
    contract change -- and the base block's order is what section 7.2 fixes."""
    path = tmp_path / "scene.ply"
    write_sogst_ply(str(path), make_fields(degree=2, include_sh=True), 0.0, 1.0, 30.0)

    header = path.read_bytes().split(b"end_header")[0].decode("ascii")
    names = [ln.split()[-1] for ln in header.splitlines()
             if ln.startswith("property float ")]
    assert names == PLY_BASE_COLUMNS + PLY_SH_COLUMNS + PLY_ACCEL_COLUMNS


def test_header_is_ascii_and_body_is_float32(tmp_path):
    path = tmp_path / "scene.ply"
    write_sogst_ply(str(path), make_fields(n=10, degree=1), 0.0, 1.0, 30.0)

    raw = path.read_bytes()
    head, _, body = raw.partition(b"end_header\n")
    assert head.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert len(body) == 10 * len(PLY_BASE_COLUMNS) * 4


# --------------------------------------------------------------------------
# Clip-level scalars: absence must be an error, never a default
# --------------------------------------------------------------------------

def strip_comment(path, key):
    """Remove one `comment sogst.<key> ...` line from a PLY in place."""
    raw = path.read_bytes()
    head, sep, body = raw.partition(b"end_header\n")
    kept = [ln for ln in head.split(b"\n")
            if not ln.startswith(f"comment sogst.{key} ".encode())]
    path.write_bytes(b"\n".join(kept) + sep + body)


@pytest.mark.parametrize("key", ["time_min", "time_max", "fps"])
def test_missing_clip_scalar_raises(tmp_path, key):
    """A defaulted fps or time range renders perfectly and plays at the wrong
    speed -- no test downstream catches it, so refusing here is the defence."""
    path = tmp_path / "scene.ply"
    write_sogst_ply(str(path), make_fields(), 0.0, 2.0, 24.0)
    strip_comment(path, key)

    with pytest.raises(ValueError, match=key):
        read_sogst_ply(str(path))

    header, _ = read_sogst_ply(str(path), require_scalars=False)
    assert header[key] is None


def test_sidecar_overrides_comments(tmp_path):
    path = tmp_path / "scene.ply"
    write_sogst_ply(str(path), make_fields(), 0.0, 2.0, 24.0)
    write_sogst_sidecar(str(path), 1.0, 3.0, 60.0, cov2d_scale=(1.5, 2.5))

    header, _ = read_sogst_ply(str(path))
    assert (header["time_min"], header["time_max"], header["fps"]) == (1.0, 3.0, 60.0)
    assert header["cov2d_scale"] == (1.5, 2.5)


def test_sidecar_alone_satisfies_the_requirement(tmp_path):
    """Sidecar-only is readable, for toolchains that strip comments."""
    path = tmp_path / "scene.ply"
    write_sogst_ply(str(path), make_fields(), 0.0, 2.0, 24.0)
    for key in ("time_min", "time_max", "fps"):
        strip_comment(path, key)
    write_sogst_sidecar(str(path), 0.0, 2.0, 24.0)

    header, _ = read_sogst_ply(str(path))
    assert header["fps"] == 24.0


def test_cov2d_scale_round_trips_through_comments(tmp_path):
    path = tmp_path / "scene.ply"
    write_sogst_ply(str(path), make_fields(), 0.0, 2.0, 30.0, cov2d_scale=(1.25, 0.75))
    header, _ = read_sogst_ply(str(path))
    assert header["cov2d_scale"] == (1.25, 0.75)


def test_unknown_sogst_comments_are_ignored(tmp_path):
    path = tmp_path / "scene.ply"
    write_sogst_ply(str(path), make_fields(), 0.0, 2.0, 30.0)
    raw = path.read_bytes()
    head, sep, body = raw.partition(b"end_header\n")
    path.write_bytes(head + b"comment sogst.future_key 1 2 3\n" + sep + body)

    header, _ = read_sogst_ply(str(path))
    assert header["fps"] == 30.0


def test_motion_degree_comment_must_agree_with_the_columns(tmp_path):
    path = tmp_path / "scene.ply"
    write_sogst_ply(str(path), make_fields(degree=1), 0.0, 2.0, 30.0)
    write_sogst_sidecar(str(path), 0.0, 2.0, 30.0, motion_degree=2)

    with pytest.raises(ValueError, match="motion_degree"):
        read_sogst_ply(str(path))


# --------------------------------------------------------------------------
# Writer validation: every one of these corrupts silently if it gets through
# --------------------------------------------------------------------------

def test_missing_required_column_raises(tmp_path):
    fields = make_fields()
    del fields["t_sigma"]
    with pytest.raises(ValueError, match="t_sigma"):
        write_sogst_ply(str(tmp_path / "s.ply"), fields, 0.0, 1.0, 30.0)


def test_column_length_mismatch_raises(tmp_path):
    fields = make_fields()
    fields["vy"] = fields["vy"][:-1]
    with pytest.raises(ValueError, match="length mismatch"):
        write_sogst_ply(str(tmp_path / "s.ply"), fields, 0.0, 1.0, 30.0)


@pytest.mark.parametrize("bad", [0.0, -0.1, np.nan])
def test_non_positive_t_sigma_raises(tmp_path, bad):
    """t_sigma is a standard deviation, not a variance, and it divides."""
    fields = make_fields()
    fields["t_sigma"] = np.asarray(fields["t_sigma"]).copy()
    fields["t_sigma"][3] = bad
    with pytest.raises(ValueError, match="t_sigma"):
        write_sogst_ply(str(tmp_path / "s.ply"), fields, 0.0, 1.0, 30.0)


def test_partial_accel_raises(tmp_path):
    fields = make_fields(degree=2)
    del fields["az"]
    with pytest.raises(ValueError, match="all-or-nothing"):
        write_sogst_ply(str(tmp_path / "s.ply"), fields, 0.0, 1.0, 30.0)


def test_partial_f_rest_raises(tmp_path):
    fields = make_fields(include_sh=True)
    f_rest = fields.pop("f_rest")
    for i in range(40):                       # 40 of 45
        fields[f"f_rest_{i}"] = f_rest[:, i]
    with pytest.raises(ValueError, match="all-or-nothing"):
        write_sogst_ply(str(tmp_path / "s.ply"), fields, 0.0, 1.0, 30.0)


def test_wrong_f_rest_width_raises(tmp_path):
    fields = make_fields(include_sh=True)
    fields["f_rest"] = fields["f_rest"][:, :30]
    with pytest.raises(ValueError, match=r"\[N, 45\]"):
        write_sogst_ply(str(tmp_path / "s.ply"), fields, 0.0, 1.0, 30.0)


# --------------------------------------------------------------------------
# Reader validation
# --------------------------------------------------------------------------

def test_non_float_property_is_rejected(tmp_path):
    path = tmp_path / "s.ply"
    write_sogst_ply(str(path), make_fields(n=4), 0.0, 1.0, 30.0)
    raw = path.read_bytes()
    head, sep, body = raw.partition(b"end_header\n")
    path.write_bytes(head + b"property uchar red\n" + sep + body)

    with pytest.raises(ValueError, match="all float32"):
        read_sogst_ply(str(path))


def test_truncated_body_is_rejected(tmp_path):
    path = tmp_path / "s.ply"
    write_sogst_ply(str(path), make_fields(n=8), 0.0, 1.0, 30.0)
    raw = path.read_bytes()
    path.write_bytes(raw[:-64])

    with pytest.raises(ValueError, match="truncated body"):
        read_sogst_ply(str(path))


def test_truncated_header_is_rejected(tmp_path):
    path = tmp_path / "s.ply"
    path.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 4\n")
    with pytest.raises(ValueError, match="truncated PLY header"):
        read_sogst_ply(str(path))


# --------------------------------------------------------------------------
# The fixture itself
# --------------------------------------------------------------------------

def test_fixture_exercises_all_four_quaternion_modes():
    """Section 4.2's smallest-three encoding has four cases; a fixture that
    only hits three lets a wrong mode mapping pass."""
    fields, _ = fixture.build_fixture(count=256, seed=1)
    quat = np.stack([fields[f"rot_{i}"] for i in range(4)], axis=1)
    assert set(np.argmax(np.abs(quat), axis=1).tolist()) == {0, 1, 2, 3}


def test_fixture_motion_is_genuinely_second_order():
    """accel must come from real second-order coefficients, never synthesised
    (section 4.8). The parabola's is the analytic dt^2 term."""
    fields, meta = fixture.build_fixture(count=128, degree=2)
    assert np.allclose(fields["ay"], meta["gravity_dt2_coefficient"])
    assert np.allclose(fields["ax"], 0.0) and np.allclose(fields["az"], 0.0)


def test_fixture_reanchoring_reproduces_the_trajectory():
    """Columns are anchored at each splat's own t_center; a producer that
    re-anchors position without also re-anchoring velocity produces a file
    that is wrong everywhere except t_center, and looks right there."""
    fields, meta = fixture.build_fixture(count=512, degree=2)
    tc = np.asarray(fields["t_center"])
    xyz = np.stack([fields[c] for c in "xyz"], axis=1)
    vel = np.stack([fields[c] for c in ("vx", "vy", "vz")], axis=1)
    acc = np.stack([fields[c] for c in ("ax", "ay", "az")], axis=1)

    # Every splat evaluated back at t = 0 must land on its launch point,
    # which the fixture drew from a tight gaussian about the origin.
    dt = (0.0 - tc)[:, None]
    launch = xyz + vel * dt + acc * dt * dt
    assert np.abs(launch).max() < 0.5
    assert meta["motion_degree"] == 2


def test_fixture_degree_1_omits_accel():
    fields, meta = fixture.build_fixture(count=32, degree=1)
    assert meta["motion_degree"] == 1
    assert not any(c in fields for c in PLY_ACCEL_COLUMNS)


def test_sidecar_payload_shape(tmp_path):
    path = tmp_path / "s.ply"
    write_sogst_ply(str(path), make_fields(), 0.0, 2.0, 30.0)
    sidecar = write_sogst_sidecar(str(path), 0.0, 2.0, 30.0, cov2d_scale=(1.0, 1.0))

    assert sidecar.endswith(sogst_ply.SIDECAR_SUFFIX)
    with open(sidecar) as fp:
        payload = json.load(fp)
    assert payload["version"] == sogst_ply.SOGST_PLY_VERSION
    assert payload["cov2d_scale"] == [1.0, 1.0]
