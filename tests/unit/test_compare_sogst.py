# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Tests for the cross-implementation equivalence check.

A comparison tool that cannot fail is worse than no tool: it certifies
divergence. So most of these are negative controls. Each injects one of
the exact mistakes docs/sogst-format.md warns a second implementation
about, and asserts that the mistake is caught, and by name.
"""

import numpy as np
import pytest

import compare_sogst
import make_sogst_fixture as fixture
from sogst_pack import pack_sogst

PIL = pytest.importorskip("PIL", reason="pack_sogst encodes webp textures")


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
        pack_sogst(str(path), data, meta["time_min"], meta["time_max"], meta["fps"],
                shn_count=512, **kwargs)
        return str(path)

    return pack


@pytest.fixture(scope="module")
def reference(packer):
    return packer("reference")


def test_packed_archive_declares_sogst_v1(reference):
    """The container is version 1 and self-identifies. The development-era
    numbering ("3") and the .omg4 magic are gone, not deprecated. Nothing
    was ever released that a reader would need to accept."""
    import json
    import zipfile

    from sogst_io import SOGST_FORMAT_ID, SOGST_VERSION

    with zipfile.ZipFile(reference) as zf:
        assert zf.namelist()[0] == "meta.json", "meta.json must be the first entry"
        meta = json.loads(zf.read("meta.json"))
    assert meta["version"] == SOGST_VERSION == 1
    assert meta["format"] == SOGST_FORMAT_ID == "sogst"


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
    """A PLY is in producer order and an archive is in packing order. The
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
    coefficient-major layout survives an absolute tolerance, because VQ
    noise is the same order. So it is caught by scale relative to the
    data spread."""
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


def true_order(source):
    from sogst_pack import compute_sogst_order

    fields, meta = source
    return compute_sogst_order(fields, meta["time_min"], meta["time_max"], 0.1)


def group_bounds(segments):
    bounds = []
    p0, p1 = segments["persistent"]
    if p1 > p0:
        bounds.append((p0, p1))
    bounds += [tuple(s["range"]) for s in segments["list"] if s["range"][1] > s["range"][0]]
    return bounds


def test_intra_group_shuffle_is_not_a_failure(packer, reference, source, capsys):
    """Morton ordering within a group is a compression heuristic, not part
    of the format (section 5). A player sees [0, P) plus a span of whole
    segments, so any permutation inside a group is unobservable. Two
    conforming encoders differ here routinely (the reference Morton and
    splat-transform's differ in quantizer scale and tie-breaking), so
    failing it would fail correct code."""
    order, segments = true_order(source)
    rng = np.random.default_rng(5)
    shuffled = order.copy()
    for lo, hi in group_bounds(segments):
        shuffled[lo:hi] = rng.permutation(shuffled[lo:hi])

    path = packer("intra_shuffled", order_segments=(shuffled, segments))
    code, out = run(reference, path, capsys)
    assert code == 0, out
    assert "index ranges identical" in out
    assert "informational" in out
    # and it really was a different permutation, not a no-op
    assert "0 of 1,200 splats in a different position" not in out


def test_catches_wrong_group_membership(packer, reference, source, capsys):
    """The same group ranges filled with the wrong splats IS observable: a
    player culling by segment would draw the wrong ones. Realigning within
    groups must not paper over it."""
    order, segments = true_order(source)
    bounds = group_bounds(segments)
    swapped = order.copy()
    # Swap equal-size blocks between two different segments, leaving every
    # range in the table untouched.
    (lo1, hi1), (lo2, hi2) = bounds[1], bounds[2]
    size = min(hi1 - lo1, hi2 - lo2)
    assert size > 0
    swapped[lo1:lo1 + size], swapped[lo2:lo2 + size] = (
        order[lo2:lo2 + size].copy(), order[lo1:lo1 + size].copy())

    path = packer("wrong_members", order_segments=(swapped, segments))
    code, out = run(reference, path, capsys)
    assert code == 1, out
    assert "index ranges identical" in out          # ranges agree...
    assert "GROUP MEMBERSHIP DIVERGES" in out       # ...contents do not


def test_catches_differing_group_ranges(packer, reference, source, capsys):
    """A wrong persistent predicate or wrong bucketing shows up as a
    different range table, and must stop the run rather than produce a
    meaningless field table."""
    order, segments = true_order(source)
    moved = {**segments, "persistent": [0, segments["persistent"][1] + 25]}

    path = packer("shifted_ranges", order_segments=(order, moved))
    code, out = run(reference, path, capsys)
    assert code == 1
    assert "group index ranges differ" in out


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
    of outliers must not fail a codebook field, while a systematic error
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


# --------------------------------------------------------------------------
# Boundary-rounding repair
#
# A coordinate sitting exactly on a quantization boundary rounds one way in a
# float32 encoder and the other in a float64 PLY. That one-step difference
# moves the splat in the lexsort, pairing it against a neighbour, so a
# perfectly conforming archive reports large errors on every field. The repair
# re-pairs those few splats. These tests pin down that it does so WITHOUT
# becoming a tool that always passes.
# --------------------------------------------------------------------------

def _blocks(count=2048, gap=(0.7, 1.3)):
    """The --blocks fixture, which is where boundary ties actually show up:
    its clusters are tight in log space, so adjacent splats land on adjacent
    quantization steps."""
    return fixture.build_fixture(count=count, degree=1, include_sh=False,
                                 seed=3, gap=gap, blocks=True)


def test_blocks_fixture_round_trips(tmp_path, capsys):
    """The block fixture must pass PLY-vs-archive. It did not before the
    repair existed: 2 splats in 8,192 produced six field failures and a
    spurious membership divergence."""
    from sogst_ply import write_sogst_ply

    fields, meta = _blocks()
    ply = tmp_path / "blocks.ply"
    write_sogst_ply(str(ply), fields, meta["time_min"], meta["time_max"], meta["fps"])
    archive = tmp_path / "blocks.sogst"
    pack_sogst(str(archive), fields, meta["time_min"], meta["time_max"], meta["fps"],
               shn_count=0)
    code, out = run(str(ply), str(archive), capsys)
    assert code == 0, out


def test_repair_stands_down_above_the_limit():
    """Past the ceiling the repair does nothing, so a real bug still fails.
    This is the property that keeps the tool honest."""
    n = 1000
    keys_a = np.zeros((n, 3), dtype=np.int64)
    keys_b = keys_a.copy()
    keys_b[:500] += 1                                  # 50%, far past the ceiling
    a = {k: np.zeros(n) for k in ("x", "y", "z")}
    b = {k: np.zeros(n) for k in ("x", "y", "z")}
    _, repaired, mask = compare_sogst.repair_boundary_pairings(
        a, b, keys_a, keys_b, [("g", 0, n)], compare_sogst.ALIGNMENT_REPAIR_LIMIT)
    assert repaired == 0
    assert not mask.any()


def test_repair_will_not_move_a_splat_between_groups():
    """The repair matches only within a group. A splat that genuinely landed
    in the wrong group has no in-group partner, so it stays mismatched and
    the membership check still sees it. The repair cannot launder the one
    error it is closest to."""
    n = 4
    keys_a = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
    keys_b = keys_a.copy()
    keys_b[0] += 1                                     # one splat, one group
    keys_b[3] += 1                                     # one splat, another group
    a = {k: np.arange(n, dtype=float) for k in ("x", "y", "z")}
    b = {k: np.arange(n, dtype=float) for k in ("x", "y", "z")}
    groups = [("g0", 0, 2), ("g1", 2, 4)]
    _, repaired, mask = compare_sogst.repair_boundary_pairings(
        a, b, keys_a, keys_b, groups, 1.0)
    assert repaired == 0, "a lone mismatch in a group has nothing to swap with"
    assert not mask.any()


def test_repair_is_inert_when_keys_agree(packer, reference, capsys):
    """It must not fire on a clean comparison."""
    code, out = run(reference, packer("repair_inert"), capsys)
    assert code == 0, out
    assert "re-paired" not in out


def test_catches_archive_disagreeing_with_its_own_source_ply(tmp_path, capsys):
    """A PLY carries no group table, so the tool derives one. It must compare
    that derived table against the archive's rather than adopt it.
    Otherwise the single most important property of a PLY-vs-archive check
    (did the archive segment its own source the way it claims?) is invisible.

    Reproduces the real defect: t_center values sitting exactly on a bucket
    boundary bucket one way at float64 and the other after a float32 PLY
    round-trip, silently moving a plateau of splats between two adjacent
    segments. This passed before the fix."""
    from sogst_ply import write_sogst_ply

    fields, meta = fixture.build_fixture(count=1500, degree=1, include_sh=False,
                                         seed=5, gap=(0.7, 1.3), blocks=True)
    # Park the folded plateau exactly on the boundary, undoing the nudge.
    # This is what the fixture generator used to do.
    tc = np.asarray(fields["t_center"], np.float64).copy()
    tc[np.isclose(tc, 1.3, atol=2e-3)] = 1.3
    fields["t_center"] = tc

    archive = tmp_path / "boundary.sogst"
    pack_sogst(str(archive), fields, meta["time_min"], meta["time_max"], meta["fps"],
               shn_count=0)                                   # packed from float64
    ply = tmp_path / "boundary.ply"
    write_sogst_ply(str(ply), fields, meta["time_min"], meta["time_max"], meta["fps"])

    code, out = run(str(ply), str(archive), capsys)           # PLY re-reads as float32
    assert code == 1, out
    assert "group index ranges differ" in out, out
