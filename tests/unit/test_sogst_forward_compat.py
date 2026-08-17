# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Tests for spec section 3.1: unknown manifest keys are ignored.

The rule exists so that a future revision can add OPTIONAL groups under
version 1 without every deployed player hard-rejecting the file. The worked
example in the spec is temporally varying rotation (which TSOG, the
independently developed SOG 4D extension, already has): it would arrive as a
new coefficient group beside `motion`, and a version-1 player must treat it
as it already treats `shN` and `accel` when it does not implement them --
render the rest of the file, identically to the same file without the group.

These tests pin the reference decoder to that contract by decoding a real
archive and a rebuilt copy of it that carries unknown keys and unknown
texture entries, and asserting the decodes are identical. The rebuilt copy
is re-emitted through the normal writers where one applies, so its stream
offsets stay verified rather than stale.
"""

import json
import zipfile

import numpy as np
import pytest

import make_sogst_fixture as fixture
from eval_render import decode_sogst_fields
from sogst_io import write_sogst_streamed
from sogst_pack import pack_sogst

pytest.importorskip("PIL", reason="pack_sogst encodes webp textures")


# A future additive group, as section 3.1 sketches it: manifest keys a
# version-1 player has never heard of.
UNKNOWN_TOP_LEVEL = {
    "rot_motion": {
        "degree": 1,
        "mins": [0.0, 0.0, 0.0],
        "maxs": [1.0, 1.0, 1.0],
        "files": ["rot_motion.webp"],
    },
    "x_experiment": 7,
}
UNKNOWN_IN_GROUP = {"smoothing": "catmull-rom"}


def _pack(path, streamed, count=600):
    fields, meta = fixture.build_fixture(count=count, degree=1,
                                         include_sh=False, seed=9)
    pack_sogst(str(path), fields, meta["time_min"], meta["time_max"],
               meta["fps"], shn_count=0,
               segment_duration=0.1 if streamed else 0)
    return path


def _decode(path):
    header, fields = decode_sogst_fields(str(path))
    return header, fields


def _assert_identical(a_path, b_path):
    header_a, fields_a = _decode(a_path)
    header_b, fields_b = _decode(b_path)
    assert header_b == header_a
    assert set(fields_b) == set(fields_a)
    for name in fields_a:
        assert np.array_equal(fields_a[name], fields_b[name]), name


def test_unknown_manifest_keys_are_ignored_streamed(tmp_path):
    """Streamed archive whose manifest carries unknown top-level and
    in-group keys. Re-emitted through write_sogst_streamed, so the offsets
    are recomputed for the larger manifest and verified -- the archive stays
    conforming, it just says more than this decoder understands."""
    src = _pack(tmp_path / "plain.sogst", streamed=True)

    with zipfile.ZipFile(src) as zf:
        names = zf.namelist()
        payloads = {n: zf.read(n) for n in names}
    manifest = json.loads(payloads.pop("meta.json"))
    manifest.update(UNKNOWN_TOP_LEVEL)
    manifest["motion"].update(UNKNOWN_IN_GROUP)

    entries = [(n, payloads[n]) for n in names if n != "meta.json"]
    reveal_through = next(i for i, (n, _) in enumerate(entries)
                          if n.startswith("seg_000/") and n.endswith("trbf.webp"))
    out = tmp_path / "annotated.sogst"
    write_sogst_streamed(str(out), manifest, entries, reveal_through)

    _assert_identical(src, out)


def test_unknown_manifest_keys_are_ignored_whole_clip(tmp_path):
    """Whole-clip variant, including an unknown texture entry.

    Built with zipfile directly rather than write_sogst: that writer emits
    only the textures its known groups list, which is correct for it but
    would silently drop the unknown entry this test exists to include. The
    entry is a byte-copy of trbf.webp under the unknown group's name -- a
    valid lossless WebP of exactly the covered texel count, which is what a
    real additive group would ship."""
    src = _pack(tmp_path / "plain.sogst", streamed=False)

    with zipfile.ZipFile(src) as zf:
        payloads = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(payloads.pop("meta.json"))
    manifest.update(UNKNOWN_TOP_LEVEL)
    manifest["trbf"].update(UNKNOWN_IN_GROUP)
    payloads["rot_motion.webp"] = payloads["trbf.webp"]

    out = tmp_path / "annotated.sogst"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("meta.json", json.dumps(manifest))
        for name in sorted(payloads):
            zf.writestr(name, payloads[name])

    _assert_identical(src, out)


def test_unknown_texture_entries_are_ignored_streamed(tmp_path):
    """Streamed variant of the unknown-entry case: the additive group's
    texture appears under EVERY group prefix, which is the shape a real
    per-splat attribute has -- every splat carries a value, so every group
    stores its slice.

    **The unknown entry goes FIRST in each group, before every required
    file, and that placement is the test.** The first player implementation
    of section 3.1 gated streamed group completion on a count of buffered
    entries; an unknown entry arriving before the group's last required
    file hit the count early and the decode ran a file short -- while the
    same entry appended after the known files was skipped harmlessly. So a
    fixture that appends the unknown entry passes on a player that is still
    broken. This decoder reads entries by name rather than in byte order,
    so it cannot fail this way -- the placement is here because this test
    is the conformance template, and the template must be the adversarial
    shape."""
    src = _pack(tmp_path / "plain.sogst", streamed=True)

    with zipfile.ZipFile(src) as zf:
        names = zf.namelist()
        payloads = {n: zf.read(n) for n in names}
    manifest = json.loads(payloads.pop("meta.json"))
    manifest.update(UNKNOWN_TOP_LEVEL)

    trbf_of = {n.rsplit("/", 1)[0]: payloads[n]
               for n in names if "/" in n and n.endswith("trbf.webp")}
    entries = []
    seen_prefixes = set()
    for n in names:
        if n == "meta.json":
            continue
        prefix = n.rsplit("/", 1)[0] if "/" in n else None
        if prefix in trbf_of and prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            entries.append((f"{prefix}/rot_motion.webp", trbf_of[prefix]))
        entries.append((n, payloads[n]))
    reveal_through = next(i for i, (n, _) in enumerate(entries)
                          if n.startswith("seg_000/") and n.endswith("trbf.webp"))
    out = tmp_path / "annotated.sogst"
    write_sogst_streamed(str(out), manifest, entries, reveal_through)

    _assert_identical(src, out)
