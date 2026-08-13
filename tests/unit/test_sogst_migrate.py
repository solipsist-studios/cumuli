# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Tests for the development-era archive migrator.

The payload of a development-era archive is already exactly a .sogst v1
payload, so migration must rewrite the manifest and touch nothing else.
The one thing that is not a pure metadata edit is the streamed layout:
meta.streams.reveal_bytes and geometry_bytes are absolute offsets, and
meta.json is the first entry, so changing the manifest length shifts every
entry after it. Most of these tests exist for that.
"""

import json
import zipfile

import pytest

import make_sogst_fixture as fixture
import sogst_migrate
from sogst_io import SOGST_FORMAT_ID, SOGST_VERSION
from sogst_pack import pack_sogst

pytest.importorskip("PIL", reason="pack_sogst encodes webp textures")


LEGACY = sogst_migrate.LEGACY_ZIP_VERSION


def make_legacy(path, streamed=True, count=600):
    """Pack a scene, then rewind its manifest to the development-era form.

    Rewinding rather than hand-building keeps the fixture honest: it is a
    real archive, with a real segment table and real stream offsets.
    """
    fields, meta = fixture.build_fixture(count=count, degree=1, include_sh=False, seed=4)
    pack_sogst(str(path), fields, meta["time_min"], meta["time_max"], meta["fps"],
               shn_count=0, segment_duration=0.1 if streamed else 0)

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        payloads = {n: zf.read(n) for n in names}
    manifest = json.loads(payloads.pop("meta.json"))
    manifest["version"] = LEGACY
    manifest.pop("format", None)

    if manifest.get("streams"):
        # Re-emit so the recorded offsets match this shorter manifest,
        # exactly as the development-era writer would have left them.
        entries = [(n, payloads[n]) for n in names if n != "meta.json"]
        reveal_through = next(i for i, (n, _) in enumerate(entries)
                              if n.startswith("seg_000/") and n.endswith("trbf.webp"))
        sogst_migrate.write_sogst_streamed(str(path), manifest, entries, reveal_through)
    else:
        sogst_migrate.write_sogst(str(path), manifest,
                                  {n: p for n, p in payloads.items() if n != "meta.json"})
    return path


def read_meta(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("meta.json"))


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("scene_v3.omg4", "scene.sogst"),
    ("scene-v3.omg4", "scene.sogst"),
    ("scene.omg4", "scene.sogst"),
    ("heidi_hires_full_v3.omg4", "heidi_hires_full.sogst"),
    ("already_v3_named_v3.omg4", "already_v3_named.sogst"),
])
def test_migrated_name(name, expected, tmp_path):
    got = sogst_migrate.migrated_name(str(tmp_path / name))
    assert got == str(tmp_path / expected)


def test_migrated_name_honours_out_dir(tmp_path):
    got = sogst_migrate.migrated_name("/somewhere/scene_v3.omg4", str(tmp_path))
    assert got == str(tmp_path / "scene.sogst")


# --------------------------------------------------------------------------
# Migration proper
# --------------------------------------------------------------------------

@pytest.mark.parametrize("streamed", [True, False])
def test_manifest_is_renumbered(tmp_path, streamed):
    src = make_legacy(tmp_path / "scene_v3.omg4", streamed=streamed)
    assert read_meta(src)["version"] == LEGACY

    out = tmp_path / "scene.sogst"
    sogst_migrate.migrate(str(src), str(out))

    meta = read_meta(out)
    assert meta["version"] == SOGST_VERSION
    assert meta["format"] == SOGST_FORMAT_ID


@pytest.mark.parametrize("streamed", [True, False])
def test_payloads_survive_byte_for_byte(tmp_path, streamed):
    src = make_legacy(tmp_path / "scene_v3.omg4", streamed=streamed)
    out = tmp_path / "scene.sogst"
    sogst_migrate.migrate(str(src), str(out))
    sogst_migrate.verify(str(src), str(out))          # raises on any drift


def test_entry_order_is_preserved(tmp_path):
    """Entry order carries the streaming semantics -- play order, deferred SH
    last -- so sorting it would silently break progressive playback."""
    src = make_legacy(tmp_path / "scene_v3.omg4", streamed=True)
    with zipfile.ZipFile(src) as zf:
        before = zf.namelist()

    out = tmp_path / "scene.sogst"
    sogst_migrate.migrate(str(src), str(out))
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == before


def test_stream_offsets_are_recomputed_not_copied(tmp_path):
    """The manifest grows by the `format` key, so every entry shifts. Copying
    the old offsets would leave them pointing mid-payload."""
    src = make_legacy(tmp_path / "scene_v3.omg4", streamed=True)
    old = read_meta(src)["streams"]

    out = tmp_path / "scene.sogst"
    sogst_migrate.migrate(str(src), str(out))
    new = read_meta(out)["streams"]

    assert new["reveal_bytes"] != old["reveal_bytes"]
    assert new["geometry_bytes"] != old["geometry_bytes"]

    # ...and the new ones must land exactly on an entry boundary. `start_dir`
    # counts as one: this fixture carries no SH, so nothing is deferred and
    # geometry_bytes marks the end of the payload rather than the start of a
    # trailing shN entry.
    with zipfile.ZipFile(out) as zf:
        boundaries = {zf.getinfo(n).header_offset for n in zf.namelist()}
        boundaries.add(zf.start_dir)
    assert new["reveal_bytes"] in boundaries
    assert new["geometry_bytes"] in boundaries


def test_reveal_point_still_covers_the_first_segment(tmp_path):
    """reveal_bytes must still mark the same logical point -- the end of the
    first segment's geometry -- not merely some valid boundary."""
    src = make_legacy(tmp_path / "scene_v3.omg4", streamed=True)
    out = tmp_path / "scene.sogst"
    sogst_migrate.migrate(str(src), str(out))

    meta = read_meta(out)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        at_reveal = next(n for n in names
                         if zf.getinfo(n).header_offset == meta["streams"]["reveal_bytes"])
    # the entry starting at reveal_bytes is the first one AFTER the reveal,
    # so it belongs to the second segment
    assert at_reveal.startswith("seg_001/")


def test_unsegmented_archive_migrates(tmp_path):
    src = make_legacy(tmp_path / "flat_v3.omg4", streamed=False)
    assert "streams" not in read_meta(src)

    out = tmp_path / "flat.sogst"
    info = sogst_migrate.migrate(str(src), str(out))
    assert "whole-clip" in info["layout"]
    assert read_meta(out)["version"] == SOGST_VERSION


def test_already_migrated_archive_is_accepted(tmp_path):
    """Re-running the migrator must be a no-op, not an error -- it will be
    pointed at directories that are partly converted."""
    src = make_legacy(tmp_path / "scene_v3.omg4", streamed=True)
    once = tmp_path / "once.sogst"
    sogst_migrate.migrate(str(src), str(once))

    twice = tmp_path / "twice.sogst"
    sogst_migrate.migrate(str(once), str(twice))
    sogst_migrate.verify(str(once), str(twice))
    assert read_meta(twice)["version"] == SOGST_VERSION


# --------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------

def test_non_zip_is_rejected_with_the_rebake_instruction(tmp_path):
    """The binary containers are gone; the error has to say what to do."""
    binary = tmp_path / "old.omg4"
    binary.write_bytes(b"OMG4" + b"\x00" * 64)

    with pytest.raises(ValueError, match="re-bake"):
        sogst_migrate.migrate(str(binary), str(tmp_path / "out.sogst"))


def test_unexpected_version_is_rejected(tmp_path):
    src = make_legacy(tmp_path / "scene_v3.omg4", streamed=False)
    with zipfile.ZipFile(src) as zf:
        payloads = {n: zf.read(n) for n in zf.namelist()}
    meta = json.loads(payloads.pop("meta.json"))
    meta["version"] = 99
    sogst_migrate.write_sogst(str(src), meta, payloads)

    with pytest.raises(ValueError, match="unexpected meta.version"):
        sogst_migrate.migrate(str(src), str(tmp_path / "out.sogst"))


def test_missing_manifest_is_rejected(tmp_path):
    path = tmp_path / "empty.omg4"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("means_l.webp", b"not really a webp")

    with pytest.raises(ValueError, match="no meta.json"):
        sogst_migrate.migrate(str(path), str(tmp_path / "out.sogst"))
