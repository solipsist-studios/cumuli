# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""sogst_io.py - the .sogst container writer and its shared constants.

`docs/sogst-format.md` is the authoritative specification.  This module is
its reference writer.  Read the spec before changing anything here, and
change the spec first if the change is normative.

A .sogst file is a ZIP archive of lossless WebP attribute textures plus a
meta.json manifest.  Static attributes follow the PlayCanvas SOG v2
conventions exactly, so an existing SOG decoder reconstructs them
unmodified.  The spacetime extension adds per-splat linear motion, a
temporal radial-basis window, an optional second-order motion term, and an
optional temporal segmentation that lets a player cull and stream by time.

Three entry points, all writers:

    build_sogst_meta()     - assemble meta.json
    write_sogst()          - whole-clip archive
    write_sogst_streamed() - per-group texture sets in play order

Quantization itself lives in sogst_pack.py.  The per-splat interchange
format that feeds it is the 4D PLY in sogst_ply.py: the single
intermediate between a bake and the container.

Numpy only: no torch, no PIL (the packer brings PIL for webp encoding).
"""

import json
import os
import zipfile

import numpy as np

# ---------------------------------------------------------------------------
# Format identity
# ---------------------------------------------------------------------------

# Written into meta.json.  A reader identifies a .sogst file by the leading
# ZIP magic, then checks these.  There is deliberately no legacy branch and
# no back-compatibility surface: the development-era binary containers were
# never released, so nothing exists that a reader would have to accept.
SOGST_FORMAT_ID = 'sogst'
SOGST_VERSION = 1
SOGST_EXTENSION = '.sogst'

# Codebook size, and the shN centroid texture widths a decoder accepts (it
# infers the SH band count from the centroids texture width, so the width
# is normative: docs/sogst-format.md section 4.7).
SOGST_CODEBOOK_SIZE = 256
SOGST_SHN_WIDTHS = {1: 192, 2: 512, 3: 960}

# The canonical per-splat field order, shared by the PLY interchange format
# and the packer.  docs/sogst-format.md section 7.2 says what each one means
# and which space it is in.  Several are easy to get wrong: the quaternion
# is w-first, scales are natural-log, opacity is logit, and t_sigma is a
# standard deviation rather than a variance.
SOGST_FIELDS = [
    'x', 'y', 'z',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
    'scale_0', 'scale_1', 'scale_2',
    'opacity',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'vx', 'vy', 'vz',
    't_center',
    't_sigma',
]


# ---------------------------------------------------------------------------
# meta.json
# ---------------------------------------------------------------------------

def build_sogst_meta(
    count: int,
    time_min: float,
    time_max: float,
    fps: float,
    means_mins,
    means_maxs,
    scales_codebook,
    sh0_codebook,
    motion_mins,
    motion_maxs,
    trbf_center_codebook,
    trbf_sigma_codebook,
    shn_count: int = 0,
    shn_bands: int = 0,
    shn_codebook=None,
    motion_degree: int = 1,
    accel_mins=None,
    accel_maxs=None,
    segments=None,
    cov2d_scale=None,
    generator: str = 'volumetric-capture-pipeline',
) -> dict:
    """Assemble the meta.json dictionary (docs/sogst-format.md section 3).

    Codebooks and mins/maxs are accepted as numpy arrays or lists.  They are
    converted to plain Python floats for JSON serialization.  `shn_*` are
    optional: pass shn_count=0 to omit higher-order SH.  `segments` is the
    optional temporal segment table.
    """
    tolist = lambda a: np.asarray(a, dtype=np.float64).tolist()

    meta = {
        'version': SOGST_VERSION,
        'format': SOGST_FORMAT_ID,
        'asset': {'generator': generator},
        'count': int(count),
        'time': {'min': float(time_min), 'max': float(time_max), 'fps': float(fps)},
        'means': {
            'mins': tolist(means_mins),
            'maxs': tolist(means_maxs),
            'files': ['means_l.webp', 'means_u.webp'],
        },
        'scales': {'codebook': tolist(scales_codebook), 'files': ['scales.webp']},
        'quats': {'files': ['quats.webp']},
        'sh0': {'codebook': tolist(sh0_codebook), 'files': ['sh0.webp']},
        'motion': {
            'degree': int(motion_degree),
            'mins': tolist(motion_mins),
            'maxs': tolist(motion_maxs),
            'files': ['motion_l.webp', 'motion_u.webp'],
        },
        'trbf': {
            'center': {'codebook': tolist(trbf_center_codebook)},
            'sigma': {'codebook': tolist(trbf_sigma_codebook)},
            'files': ['trbf.webp'],
        },
    }
    if accel_mins is not None:
        # degree-2 motion: quadratic coefficient (units/sec^2), same
        # log-transform + 16-bit-split scheme as motion
        meta['motion']['degree'] = 2
        meta['accel'] = {
            'mins': tolist(accel_mins),
            'maxs': tolist(accel_maxs),
            'files': ['accel_l.webp', 'accel_u.webp'],
        }
    if shn_count > 0:
        if shn_bands not in SOGST_SHN_WIDTHS:
            raise ValueError(f'shn_bands must be one of {sorted(SOGST_SHN_WIDTHS)}, got {shn_bands}')
        meta['shN'] = {
            'count': int(shn_count),
            'bands': int(shn_bands),
            'codebook': tolist(shn_codebook),
            'files': ['shN_centroids.webp', 'shN_labels.webp'],
        }
    if segments:
        meta['segments'] = segments
    if cov2d_scale is not None:
        # screen-space 2D-covariance compensation a player applies when
        # rasterising.  Absent means [1, 1].
        meta['cov2d_scale'] = [float(cov2d_scale[0]), float(cov2d_scale[1])]
    return meta


# ---------------------------------------------------------------------------
# Archive writers
# ---------------------------------------------------------------------------

def write_sogst(out_path: str, meta: dict, textures: dict) -> None:
    """Write a whole-clip .sogst archive.

    Parameters
    ----------
    out_path : Output .sogst path.
    meta     : meta.json dictionary (from build_sogst_meta()).
    textures : Maps webp filename -> encoded lossless-webp bytes.  Must
               contain every file listed in the meta's `files` entries.

    Entries are ZIP_STORED with no extra fields and no data descriptor.
    Webp payloads are already compressed, and the fixed 30 + len(name)
    local header is what lets a player byte-range into the archive (spec
    section 2).
    """
    expected = set()
    for key in ('means', 'scales', 'quats', 'sh0', 'shN', 'motion', 'accel', 'trbf'):
        expected.update(meta.get(key, {}).get('files', []))
    missing = expected - set(textures)
    if missing:
        raise ValueError(f'write_sogst: missing texture blobs: {sorted(missing)}')

    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr('meta.json', json.dumps(meta))
        for name in sorted(expected):
            zf.writestr(name, textures[name])


def write_sogst_streamed(out_path: str, meta: dict, entries, reveal_through: int) -> None:
    """Write a .sogst archive in the streamed layout.

    `entries` is an ordered list of (name, bytes) written verbatim after
    meta.json: geometry groups in play order (persistent/*, seg_000/*,
    ...), then shN_centroids and the per-group shN_labels when SH is
    deferred (meta.streams.sh_deferred).  A sequential download therefore
    yields decodable groups progressively and adds the view-dependent SH
    last.  `reveal_through` is the index of the last entry a player needs
    before it can reveal the scene (end of the first temporal segment's
    geometry).  The byte offset of that point is stored as
    meta.streams.reveal_bytes so progress bars can fill against the
    reveal, not the whole file.

    Entries are ZIP_STORED with no extra fields, so each entry costs
    exactly 30 + len(name) header bytes.  reveal_bytes is computed
    analytically and verified against the written file.
    """
    def local_size(name: str, data: bytes) -> int:
        return 30 + len(name) + len(data)

    # meta.json contains reveal_bytes / geometry_bytes, whose digit counts
    # feed back into its own size.  Iterate to a fixed point (converges in
    # a few passes).  geometry_bytes marks the end of the last geometry
    # entry (before shN_centroids/labels): players use it with measured
    # bandwidth to hold the playhead until gap-free playback is possible.
    streams = meta.setdefault('streams', {})
    geometry_through = next(
        (i for i in range(len(entries) - 1, -1, -1)
         if '/shN_labels' not in entries[i][0] and entries[i][0] != 'shN_centroids.webp'),
        len(entries) - 1)
    for _ in range(6):
        meta_bytes = json.dumps(meta).encode()
        prev = (streams.get('reveal_bytes'), streams.get('geometry_bytes'))
        pos = local_size('meta.json', meta_bytes)
        for i, (name, data) in enumerate(entries):
            pos += local_size(name, data)
            if i == reveal_through:
                streams['reveal_bytes'] = pos
            if i == geometry_through:
                streams['geometry_bytes'] = pos
        if prev == (streams['reveal_bytes'], streams['geometry_bytes']):
            break

    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr('meta.json', json.dumps(meta))
        for name, data in entries:
            zf.writestr(name, data)

    # Verify the analytic offsets against the real layout. A marker points at
    # the local header of the entry after it.  When it is the last entry, it
    # points at the start of the central directory instead. That second case
    # is the one to be careful about: guarding this loop with
    # `idx + 1 < len(entries)` and stopping there silently skips the check
    # exactly when nothing follows, which is every archive with no shN group.
    # §2 requires a writer to verify these offsets, so skipping is not a
    # lesser check, it is no check.
    #
    # zipfile's start_dir comes from the EOCD record. Do NOT locate the
    # directory by scanning for the PK\x01\x02 signature: a scan finds *a*
    # directory header, not reliably the first one (§6).
    with zipfile.ZipFile(out_path) as zf:
        for key, idx in (('reveal_bytes', reveal_through), ('geometry_bytes', geometry_through)):
            if idx + 1 < len(entries):
                actual = zf.getinfo(entries[idx + 1][0]).header_offset
                where = f'entry {entries[idx + 1][0]!r}'
            else:
                actual = zf.start_dir
                where = 'the central directory'
            if actual != streams[key]:
                raise AssertionError(
                    f'write_sogst_streamed: {key} {streams[key]} != '
                    f'actual offset {actual} at {where} '
                    '(zip writer added extra fields?)')


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def report_output(out_path: str) -> None:
    """Print the output file path and size to stdout."""
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")
