#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""sogst_migrate.py - modernize development-era ZIP archives to .sogst v1.

During development the ZIP container was numbered "version 3" (the first
two versions were binary layouts that no longer exist) and carried a .omg4
extension.  Nothing was ever released, so the format was renumbered to
version 1 rather than carrying a legacy version number forever.

The payload did not change: the textures, codebooks and segment table in a
development-era archive are already exactly what a .sogst v1 file holds.
So an already-baked asset needs its manifest rewritten, not a re-bake from
the trainer, which would cost a GPU run and re-quantize the data a second
time.

    python sogst_migrate.py scene_v3.omg4                  # -> scene.sogst
    python sogst_migrate.py splats/*.omg4 --out-dir out/
    python sogst_migrate.py scene_v3.omg4 --delete-original

Why this is a script and not a one-line meta.json edit: in the streamed
layout, meta.streams.reveal_bytes and geometry_bytes are absolute byte
offsets into the archive, and meta.json is itself the first entry.  Editing
the manifest changes its length, which shifts every entry after it and
invalidates both offsets.  So the archive is re-emitted through the same
writers a fresh bake uses, which recompute the offsets to a fixed point and
verify them against the written file.

Only ZIP archives are migratable.  The binary containers that shared the
.omg4 extension are gone.  Assets in that form have to be re-baked from a
trainer checkpoint with bake_sogst.py.
"""

import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sogst_io import (  # noqa: E402
    SOGST_EXTENSION,
    SOGST_FORMAT_ID,
    SOGST_VERSION,
    write_sogst,
    write_sogst_streamed,
)

# Development-era container version, as written in meta.json before the
# renumbering.  Accepted as input, never produced.
LEGACY_ZIP_VERSION = 3


def migrated_name(path, out_dir=None):
    """Destination path: drop a `_v3` suffix, force the .sogst extension."""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    for suffix in ('_v3', '-v3'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return os.path.join(out_dir or os.path.dirname(path), stem + SOGST_EXTENSION)


def migrate(in_path, out_path):
    """Rewrite one archive's manifest to .sogst v1. Returns a summary dict."""
    if not zipfile.is_zipfile(in_path):
        raise ValueError(
            f'{in_path}: not a ZIP archive. The development-era binary containers '
            'cannot be migrated -- re-bake from a trainer checkpoint with '
            'bake_sogst.py.')

    with zipfile.ZipFile(in_path) as zf:
        names = zf.namelist()
        if 'meta.json' not in names:
            raise ValueError(f'{in_path}: no meta.json')
        meta = json.loads(zf.read('meta.json'))
        version = meta.get('version')
        if version not in (LEGACY_ZIP_VERSION, SOGST_VERSION):
            raise ValueError(f'{in_path}: unexpected meta.version {version!r} '
                             f'(expected {LEGACY_ZIP_VERSION} or {SOGST_VERSION})')
        # Read every payload up front: the source is closed before the
        # destination is written, so in-place migration is safe.
        payloads = {name: zf.read(name) for name in names if name != 'meta.json'}
        # Entry ORDER carries the streaming semantics (play order, with the
        # deferred SH tail last), so it is preserved exactly, not sorted.
        order = [name for name in names if name != 'meta.json']

    streams = meta.get('streams')
    meta['version'] = SOGST_VERSION
    meta['format'] = SOGST_FORMAT_ID

    if streams:
        # Recover which entry the old reveal point sat at, by matching the
        # recorded byte offset against the source layout, then let the writer
        # recompute the offset for the new manifest length.
        with zipfile.ZipFile(in_path) as zf:
            offsets = {name: zf.getinfo(name).header_offset for name in order}
        old_reveal = streams.get('reveal_bytes')
        reveal_through = len(order) - 1
        if old_reveal is not None:
            # reveal_bytes points at the START of the entry after the reveal.
            following = [i for i, name in enumerate(order) if offsets[name] >= old_reveal]
            reveal_through = (following[0] - 1) if following else len(order) - 1
            if reveal_through < 0:
                raise ValueError(f'{in_path}: reveal_bytes {old_reveal} precedes the '
                                 'first entry; archive looks corrupt')
        entries = [(name, payloads[name]) for name in order]
        write_sogst_streamed(out_path, meta, entries, reveal_through)
        layout = f'streamed, {len(entries)} entries, reveal through {order[reveal_through]}'
    else:
        write_sogst(out_path, meta, payloads)
        layout = f'whole-clip, {len(payloads)} textures'

    return {'count': meta.get('count'), 'layout': layout,
            'was_version': version, 'in_size': os.path.getsize(in_path),
            'out_size': os.path.getsize(out_path)}


def verify(in_path, out_path):
    """Assert the migration preserved every payload byte-for-byte."""
    with zipfile.ZipFile(in_path) as a, zipfile.ZipFile(out_path) as b:
        names_a = [n for n in a.namelist() if n != 'meta.json']
        names_b = [n for n in b.namelist() if n != 'meta.json']
        if names_a != names_b:
            raise AssertionError(f'entry list changed:\n  {names_a}\n  {names_b}')
        for name in names_a:
            if a.read(name) != b.read(name):
                raise AssertionError(f'payload changed for {name}')
        meta_a = json.loads(a.read('meta.json'))
        meta_b = json.loads(b.read('meta.json'))
        if meta_b['version'] != SOGST_VERSION or meta_b['format'] != SOGST_FORMAT_ID:
            raise AssertionError(f'meta not migrated: {meta_b.get("version")!r} '
                                 f'{meta_b.get("format")!r}')
        # Everything except the identity fields and the recomputed stream
        # offsets must be untouched.
        skip = {'version', 'format', 'streams'}
        for key in set(meta_a) | set(meta_b):
            if key not in skip and meta_a.get(key) != meta_b.get(key):
                raise AssertionError(f'meta.{key} changed during migration')
        if 'streams' in meta_b:
            for key, value in meta_b['streams'].items():
                if key in ('reveal_bytes', 'geometry_bytes'):
                    continue          # recomputed by design
                if meta_a.get('streams', {}).get(key) != value:
                    raise AssertionError(f'meta.streams.{key} changed')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+', help='development-era ZIP archives')
    parser.add_argument('--out-dir', default=None,
                        help='write beside the sources by default')
    parser.add_argument('--delete-original', action='store_true',
                        help='remove each source after a verified migration')
    parser.add_argument('--no-verify', action='store_true',
                        help='skip the payload-identity check (not recommended)')
    args = parser.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    failures = 0
    for in_path in args.inputs:
        out_path = migrated_name(in_path, args.out_dir)
        if os.path.abspath(out_path) == os.path.abspath(in_path):
            print(f'SKIP {in_path}: destination is the source')
            continue
        try:
            info = migrate(in_path, out_path)
            if not args.no_verify:
                verify(in_path, out_path)
            delta = info['out_size'] - info['in_size']
            print(f'{os.path.basename(in_path)} -> {os.path.basename(out_path)}  '
                  f'v{info["was_version"]}->v{SOGST_VERSION}, {info["count"]:,} splats, '
                  f'{info["layout"]}, {delta:+,} bytes')
            if args.delete_original:
                os.remove(in_path)
                print(f'  removed {in_path}')
        except Exception as exc:                       # noqa: BLE001 - CLI boundary
            failures += 1
            print(f'FAIL {in_path}: {exc}')

    if failures:
        sys.exit(f'{failures} archive(s) failed to migrate')


if __name__ == '__main__':
    main()
