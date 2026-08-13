# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Tests for the cross-implementation equivalence check.

A comparison tool that cannot fail is worse than no tool: it certifies
divergence. So most of these are negative controls -- deliberate injections
of the exact mistakes docs/sogst-format.md warns a second implementation
about -- asserting that each one is caught, and by name.
"""

import numpy as np
import pytest

import compare_sogst
import make_sogst_fixture as fixture
from sog_pack import pack_v3

PIL = pytest.importorskip("PIL", reason="pack_v3 encodes webp textures")


@pytest.fixture(scope="module")
def source():
    """A small fixture scene, plus its clip scalars."""
    fields, meta = fixture.build_fixture(count=1200, degree=2, include_sh=True, seed=7)
    return fields, meta


@pytest.fixture(scope="module")
def packer(tmp_path_factory, source):
    """Pack a (possibly mutated) copy of the source scene."""
    fields, meta = source
    out_dir = tmp_path_factory.mktemp("sogst")

    def pack(name, mutate=None, **kwargs):
        data = dict(fields)
        if mutate is not None:
            mutate(data)
        path = out_dir / f"{name}.sogst"
        pack_v3(str(path), data, meta["time_min"], meta["time_max"], meta["fps"],
                shn_count=512, **kwargs)
        return str(path)

    return pack


@pytest.fixture(scope="module")
def reference(packer):
    return packer("reference")


def run(a, b, capsys):
    code = compare_sogst.compare(a, b)
    return code, capsys.readouterr().out


# --------------------------------------------------------------------------
# The positive control
# --------------------------------------------------------------------------

def test_identical_input_passes(packer, reference, capsys):
    code, out = run(reference, packer("same"), capsys)
    assert code == 0, out
    assert "PASS" in out


def test_source_ply_against_its_own_pack_passes(tmp_path, source, packer, reference, capsys):
    """A PLY is in producer order and an archive is in packing order; the
    tool must reorder before comparing, or every field reads as wrong."""
    from sogst_ply import write_sogst_ply

    fields, meta = source
    ply = tmp_path / "source.ply"
    write_sogst_ply(str(ply), fields, meta["time_min"], meta["time_max"], meta["fps"])

    code, out = run(str(ply), reference, capsys)
    assert "reordered" in out
    assert code == 0, out


# --------------------------------------------------------------------------
# Negative controls: one per mistake the spec calls out
# --------------------------------------------------------------------------

def test_catches_quaternion_stored_xyzw(packer, reference, capsys):
    """rot_0 is W (section 4.2). Storing xyzw is the classic port bug."""
    def mutate(d):
        d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"] = (
            d["rot_1"], d["rot_2"], d["rot_3"], d["rot_0"])

    code, out = run(reference, packer("quat_xyzw", mutate), capsys)
    assert code == 1
    assert "rotation" in out
    assert "smallest-three mode mapping" in out


def test_catches_velocity_per_frame_instead_of_per_second(packer, reference, capsys):
    """vx/vy/vz are scene units per SECOND (section 7.2)."""
    def mutate(d):
        for c in ("vx", "vy", "vz"):
            d[c] = d[c] / 30.0

    code, out = run(reference, packer("vel_per_frame", mutate), capsys)
    assert code == 1
    assert all(c in out for c in ("vx", "vy", "vz"))


def test_catches_t_sigma_as_variance(packer, reference, capsys):
    """t_sigma is a standard deviation, not a variance (section 1)."""
    def mutate(d):
        d["t_sigma"] = d["t_sigma"] ** 2

    code, out = run(reference, packer("sigma_variance", mutate), capsys)
    assert code == 1
    assert "t_sigma" in out


def test_catches_f_rest_layout_transpose(packer, reference, capsys):
    """f_rest is channel-major, index j*coeffs + k (section 4.7). A
    coefficient-major layout survives an absolute tolerance -- VQ noise is
    the same order -- so it is caught by scale relative to the data spread."""
    def mutate(d):
        d["f_rest"] = d["f_rest"].reshape(-1, 3, 15).transpose(0, 2, 1).reshape(-1, 45)

    code, out = run(reference, packer("sh_transpose", mutate), capsys)
    assert code == 1
    assert "f_rest" in out
    assert "LAYOUT" in out


def test_catches_missing_accel(packer, reference, capsys):
    """accel is all-or-nothing (section 4.8)."""
    def mutate(d):
        for c in ("ax", "ay", "az"):
            d.pop(c)

    code, out = run(reference, packer("no_accel", mutate), capsys)
    assert code == 1
    assert "all-or-nothing" in out


def test_catches_differing_splat_count(packer, reference, capsys):
    def mutate(d):
        for key in list(d):
            d[key] = d[key][:-10]

    code, out = run(reference, packer("short", mutate), capsys)
    assert code == 1
    assert "counts differ" in out


def test_catches_segmentation_mismatch(packer, reference, capsys):
    code, out = run(reference, packer("unsegmented", segment_duration=0), capsys)
    assert code == 1
    assert "segmentation present in only one file" in out


def test_catches_ordering_divergence(packer, reference, capsys):
    """Splat order is part of the format; a shuffled file must be reported
    as an ordering finding, not as 19 unrelated field failures."""
    def mutate(d):
        rng = np.random.default_rng(11)
        perm = rng.permutation(len(d["x"]))
        for key in list(d):
            d[key] = d[key][perm] if d[key].ndim == 1 else d[key][perm, :]

    # Pass a precomputed identity ordering so the shuffle survives packing.
    n = 1200
    identity = (np.arange(n), None)
    code, out = run(reference, packer("shuffled", mutate, order_segments=identity), capsys)
    assert code == 1
    assert "ORDERING DIVERGES" in out


# --------------------------------------------------------------------------
# Tolerance derivation
# --------------------------------------------------------------------------

def test_codebook_tolerance_is_half_the_largest_gap():
    meta = {"scales": {"codebook": [0.0, 1.0, 5.0, 6.0]}}
    assert compare_sogst.codebook_tolerance(meta, ("scales", "codebook")) == 2.0


def test_codebook_tolerance_absent_group_is_none():
    assert compare_sogst.codebook_tolerance({}, ("scales", "codebook")) is None
    assert compare_sogst.codebook_tolerance(None, ("scales", "codebook")) is None


def test_split16_tolerance_grows_with_magnitude():
    """The log transform makes the linear-space step value-dependent, so a
    single constant tolerance would be wrong at one end or the other."""
    meta = {"means": {"mins": [-2.0, 0, 0], "maxs": [2.0, 0, 0]}}
    tol = compare_sogst.split16_tolerance(meta, "means", 0, np.array([0.0, 100.0]))
    assert tol[1] > tol[0]


def test_codebook_fields_are_judged_by_fraction_not_worst_case():
    """Bin assignment at a boundary is implementation-defined, so a handful
    of outliers must not fail a codebook field -- while a systematic error
    still must."""
    err = np.zeros(1000)
    err[:3] = 10.0                                   # 0.3% way past tolerance
    assert compare_sogst.judge("scale_0", err, 1.0, codebook=True)[-1] is True
    assert compare_sogst.judge("x", err, 1.0, codebook=False)[-1] is False

    err[:] = 10.0                                    # systematic
    assert compare_sogst.judge("scale_0", err, 1.0, codebook=True)[-1] is False


def test_is_codebook_field():
    assert compare_sogst.is_codebook_field("scale_1")
    assert compare_sogst.is_codebook_field("f_dc_2")
    assert compare_sogst.is_codebook_field("t_sigma")
    assert not compare_sogst.is_codebook_field("x")
    assert not compare_sogst.is_codebook_field("vx")
