# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Tests for the container writer's post-write offset verification.

Section 2 requires a writer to verify its computed streaming offsets against
the file it actually wrote. The interesting case is the one that looks like
it needs no attention: when a marker falls on the LAST entry there is no
following entry to compare against, and the offset points at the start of the
central directory instead. Guarding the check with "is there a next entry?"
therefore skips it precisely when nothing follows -- which is every archive
with no shN group.

That is not a hypothetical. The same wrong assumption -- that an offset always
points at an entry -- appeared independently in four places: this repo's spec
section 6, this writer, and both the checker and the writer verification of the
independent TypeScript encoder. None was a typo; each came from reasoning about
offsets without asking what happens at the end.
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

    Perturbing the real layout by one byte has to raise; if it does not, the
    verification is passing vacuously on exactly the archives it was added to
    protect."""
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

    The TypeScript encoder is not: its ZIP writer stamps DOS timestamps from
    the wall clock, so re-encoding the same input changes 484 bytes across 121
    entries while every payload stays identical. That makes a hash a false
    negative as a same-input check there. This side has no such excuse, and a
    regression would silently break any tooling that dedupes by hash."""
    a = tmp_path / "a.sogst"
    b = tmp_path / "b.sogst"
    _pack(a, scene, shn_count=0)
    _pack(b, scene, shn_count=0)
    assert a.read_bytes() == b.read_bytes()
