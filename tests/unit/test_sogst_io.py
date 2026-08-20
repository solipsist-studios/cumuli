# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Tests for the container writer's post-write offset verification.

Section 2 requires a writer to verify its computed streaming offsets against
the file it actually wrote. The interesting case is the one that looks like
it needs no attention. When a marker falls on the LAST entry, there is no
following entry to compare against, and the offset points at the start of
the central directory instead. Guarding the check with "is there a next
entry?" therefore skips it precisely when nothing follows, which is every
archive with no shN group.

That is not a hypothetical. The same wrong assumption (that an offset always
points at an entry) appeared independently in four places: this repo's spec
section 6, this writer, and both the checker and the writer verification of
the independent TypeScript encoder. None was a typo. Each came from
reasoning about offsets without asking what happens at the end.
"""

import zipfile

import numpy as np
import pytest

import make_sogst_fixture as fixture
import sogst_io
from sogst_pack import pack_sogst


@pytest.fixture(scope="module")
def scene():
    fields, meta = fixture.build_fixture(count=900, degree=1, include_sh=False, seed=11)
    return fields, meta


@pytest.fixture(scope="module")
def scene_with_sh():
    fields, meta = fixture.build_fixture(count=900, degree=1, include_sh=True, seed=11)
    return fields, meta


def _pack(path, scene, shn_count):
    fields, meta = scene
    return pack_sogst(str(path), fields, meta["time_min"], meta["time_max"],
                      meta["fps"], shn_count=shn_count)


def test_geometry_bytes_lands_on_the_central_directory_without_sh(tmp_path, scene):
    """With no shN group nothing follows the geometry, so the marker is the
    central-directory offset rather than an entry header."""
    path = tmp_path / "nosh.sogst"
    meta = _pack(path, scene, shn_count=0)
    streams = meta.get("streams")
    if not streams:
        pytest.skip("archive not written streamed")
    with zipfile.ZipFile(path) as zf:
        offsets = {i.header_offset for i in zf.infolist()}
        assert streams["geometry_bytes"] == zf.start_dir
        assert streams["geometry_bytes"] not in offsets
        assert streams["reveal_bytes"] in offsets


def test_offset_verification_is_live_on_the_no_sh_path(tmp_path, scene, monkeypatch):
    """The check must actually run when the marker is last, not skip.

    Perturbing the real layout by one byte has to raise. If it does not, the
    verification is passing vacuously on exactly the archives it was added to
    protect.

    **Where the perturbation goes is load-bearing, so do not move it.** It
    shifts the *observed* start_dir at read time, downstream of both the
    fixed-point loop and serialization. Mutating `streams['geometry_bytes']`
    earlier would prove nothing: the loop overwrites both offsets from the real
    entry sizes on every pass, so a pre-loop mutation is erased and the writer
    produces a self-consistent archive that passes. The same trap exists in the
    TypeScript encoder, where the equivalent mutation has to land after
    meta.json is serialized.

    A corollary worth knowing: this test deliberately does not produce a
    corrupt artifact. The file on disk carries the correct offset throughout.
    Only the comparison sees a wrong value. That is what a check-the-checker
    test should do.

    Scope of what the last-marker fix actually bought, measured rather than
    assumed: starving the fixed-point loop to one pass raises on
    *reveal_bytes* even under the old guard, because that marker points at a
    real entry. So a non-converging loop was never the exposure. What was
    invisible is an error specific to geometry_bytes (a wrong
    `geometry_through`, or an off-by-one that only bites on the last entry),
    since that check did not run at all."""
    real = zipfile.ZipFile

    class Shifted(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if self.mode == "r":
                self.start_dir += 1

    monkeypatch.setattr(sogst_io.zipfile, "ZipFile", Shifted)
    with pytest.raises(AssertionError, match="geometry_bytes"):
        _pack(tmp_path / "mutated.sogst", scene, shn_count=0)


def test_offset_verification_still_covers_the_entry_case(tmp_path, scene_with_sh):
    """With shN deferred the marker does point at an entry header, and that
    path must keep working."""
    path = tmp_path / "sh.sogst"
    meta = _pack(path, scene_with_sh, shn_count=256)
    streams = meta.get("streams")
    assert streams and streams.get("sh_deferred"), \
        "expected a deferred shN group -- this test covers the entry-header path"
    with zipfile.ZipFile(path) as zf:
        offsets = {i.header_offset for i in zf.infolist()}
        assert streams["geometry_bytes"] in offsets
        assert streams["geometry_bytes"] != zf.start_dir


def test_archive_is_byte_reproducible(tmp_path, scene):
    """Two packs of the same input must be byte-identical.

    The TypeScript encoder is not. Its ZIP writer stamps DOS timestamps from
    the wall clock, so re-encoding the same input changes 484 bytes across
    121 entries while every payload stays identical. That makes a hash a false
    negative as a same-input check there. This writer had the same defect in
    disguise: zipfile's str-name writestr() stamps the wall clock at DOS
    2-second granularity, so this test only passed when both packs landed in
    one 2-second tick, and flaked on a loaded runner (caught live in CI,
    2026-08-20). The writer now stamps the fixed ZIP epoch, and a regression
    would silently break any tooling that dedupes by hash."""
    a = tmp_path / "a.sogst"
    b = tmp_path / "b.sogst"
    _pack(a, scene, shn_count=0)
    _pack(b, scene, shn_count=0)
    assert a.read_bytes() == b.read_bytes()


def test_sh_vector_quantization_is_deterministic_on_cpu(tmp_path, scene_with_sh,
                                                        monkeypatch):
    """The SH path is byte-reproducible on CPU, and NOT on GPU.

    Clustering is where an encoder most easily stops being deterministic, and
    both implementations of this format are affected. The TypeScript
    encoder's GPU k-means lands on a different local optimum per run: two
    consecutive encodes of one real asset differ in 23 of 191 entries with a
    0.15% size spread, geometry bit-identical. This encoder has the same
    class of defect for a narrower reason. `pack_shn` runs Lloyd in torch,
    and on CUDA the `index_add_` centroid accumulation uses atomics, so
    summation order varies between runs. The resulting ~4e-9 wobble is
    absorbed by quantization in the textures but survives into
    `meta.shN.codebook`, whose changed digit count then shifts
    `reveal_bytes` and `geometry_bytes`.

    Measured: with CUDA available, `meta.json` differs between two packs of
    one input (187 of 256 codebook entries, last few digits). With CUDA
    hidden, every entry is byte-identical. That is what this test pins,
    since it is the part that is a property of the algorithm rather than of
    the hardware.

    Both encoders' GPU archives remain valid. The difference is a clustering
    choice, not a quality one. But it means a hash is not a same-input check
    on a GPU-encoded SH asset from either implementation, and §10's field
    comparison is the check that still holds."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    a = tmp_path / "sh_a.sogst"
    b = tmp_path / "sh_b.sogst"
    _pack(a, scene_with_sh, shn_count=256)
    _pack(b, scene_with_sh, shn_count=256)

    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        names = [i.filename for i in za.infolist()]
        assert [i.filename for i in zb.infolist()] == names
        assert any("shN" in n for n in names), "expected an shN group to be present"
        differing = [n for n in names if za.read(n) != zb.read(n)]
        assert not differing, f"nondeterministic on CPU: {differing}"
