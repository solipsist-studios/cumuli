"""
omg4_repack.py – Repack an existing OMG4 v2 file into the streamable
tiled layout (flags bit 2): splats sorted by visual importance, body
written as self-contained chunked-SoA tiles so viewers can render
progressively from a sequential download.

Usage:
    python omg4_repack.py --input scene.omg4 --output scene_stream.omg4 \
        [--tile-size 65536] [--stream-order importance|shuffle|none] [--verify]

The input must be a standard (non-tiled) v2 file. All header semantics
(SH flag, cov2d_scale word, time range, fps) are preserved.
"""

import argparse
import struct
import sys

import numpy as np

from splat4d_io import (
    OMG4_MAGIC,
    OMG4_V2_VERSION,
    OMG4_V2_FLAG_SH,
    OMG4_V2_FLAG_COV2D,
    OMG4_V2_FLAG_TILED,
    write_omg4_v2_header,
    write_omg4_v2_body_tiled,
    report_output,
)

NUM_BASE_FIELDS = 19
NUM_SH_FIELDS = 45


def read_omg4_v2(path):
    """Read a v2 file. Returns (header dict, list of float32[N] arrays)."""
    with open(path, 'rb') as fp:
        raw = fp.read()

    magic, version, n, flags, time_min, time_max, fps, reserved = struct.unpack_from('<IIIIfffI', raw, 0)
    if magic != OMG4_MAGIC:
        sys.exit(f'ERROR: not an OMG4 file (magic 0x{magic:08x})')
    if version != OMG4_V2_VERSION:
        sys.exit(f'ERROR: expected v2, got version {version}')
    if flags & OMG4_V2_FLAG_TILED:
        sys.exit('ERROR: input is already tiled/streamable')

    num_fields = NUM_BASE_FIELDS + (NUM_SH_FIELDS if flags & OMG4_V2_FLAG_SH else 0)
    header_size = 32
    expected = header_size + num_fields * n * 4
    if len(raw) != expected:
        sys.exit(f'ERROR: file size {len(raw)} != expected {expected} '
                 f'(N={n}, fields={num_fields})')

    arrays = [
        np.frombuffer(raw, dtype=np.float32, count=n, offset=header_size + i * n * 4)
        for i in range(num_fields)
    ]

    cov2d_scale = None
    if flags & OMG4_V2_FLAG_COV2D:
        halves = np.array([reserved & 0xFFFF, (reserved >> 16) & 0xFFFF], dtype=np.uint16).view(np.float16)
        cov2d_scale = (float(halves[0]), float(halves[1]))

    header = {
        'num_splats': n,
        'flags': flags,
        'time_min': time_min,
        'time_max': time_max,
        'fps': fps,
        'cov2d_scale': cov2d_scale,
        'num_fields': num_fields,
    }
    return header, arrays


def read_omg4_v2_tiled(path):
    """Read back a tiled v2 file (for --verify). Returns (header, arrays)."""
    with open(path, 'rb') as fp:
        raw = fp.read()

    magic, version, n, flags, time_min, time_max, fps, reserved = struct.unpack_from('<IIIIfffI', raw, 0)
    assert magic == OMG4_MAGIC and version == OMG4_V2_VERSION
    assert flags & OMG4_V2_FLAG_TILED, 'verify: output is not tiled'
    tile_size, _reserved2 = struct.unpack_from('<II', raw, 32)

    num_fields = NUM_BASE_FIELDS + (NUM_SH_FIELDS if flags & OMG4_V2_FLAG_SH else 0)
    header_size = 40
    assert len(raw) == header_size + num_fields * n * 4, 'verify: tiled size mismatch'

    arrays = [np.empty(n, dtype=np.float32) for _ in range(num_fields)]
    offset = header_size
    for start in range(0, n, tile_size):
        count = min(tile_size, n - start)
        for f in range(num_fields):
            arrays[f][start:start + count] = np.frombuffer(raw, dtype=np.float32, count=count, offset=offset)
            offset += count * 4

    header = {'num_splats': n, 'flags': flags, 'tile_size': tile_size,
              'time_min': time_min, 'time_max': time_max, 'fps': fps}
    return header, arrays


def importance_order(arrays):
    """Descending visual importance: peak opacity x projected area."""
    log_scales = np.stack(arrays[7:10], axis=1)
    opacity_logit = arrays[10]
    alpha = 1.0 / (1.0 + np.exp(-opacity_logit))
    mean_scale = np.exp(log_scales.mean(axis=1))
    return np.argsort(-(alpha * mean_scale ** 2), kind='stable')


def verify_roundtrip(header, arrays, out_path):
    """De-tile the output and assert it holds the same splat multiset."""
    out_header, out_arrays = read_omg4_v2_tiled(out_path)
    assert out_header['num_splats'] == header['num_splats'], 'verify: N mismatch'
    for key in ('time_min', 'time_max', 'fps'):
        assert out_header[key] == header[key], f'verify: header {key} mismatch'
    assert (out_header['flags'] & ~OMG4_V2_FLAG_TILED) == header['flags'], 'verify: flags mismatch'

    def canonical(arrs):
        m = np.stack(arrs, axis=1)
        # lexicographic row sort over all fields (last key = primary)
        order = np.lexsort(tuple(m[:, i] for i in reversed(range(m.shape[1]))))
        return m[order]

    a = canonical(arrays)
    b = canonical(out_arrays)
    assert a.shape == b.shape and np.array_equal(a, b), 'verify: splat multiset differs'
    print('  Verify: OK — output holds the identical splat set')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='Standard (non-tiled) v2 .omg4 file')
    parser.add_argument('--output', required=True, help='Destination streamable .omg4 file')
    parser.add_argument('--tile-size', type=int, default=65536,
                        help='Splats per tile (default: 65536; ~4.8MB/tile without SH, ~16MB with)')
    parser.add_argument('--stream-order', choices=['importance', 'shuffle', 'none'], default='importance',
                        help='Splat ordering: importance (default), random shuffle, or keep input order')
    parser.add_argument('--verify', action='store_true',
                        help='Read the output back and assert it matches the input splat set')
    args = parser.parse_args()

    header, arrays = read_omg4_v2(args.input)
    n = header['num_splats']
    print(f"Read {args.input}: {n:,} splats, {header['num_fields']} fields, "
          f"flags={header['flags']}, cov2d={header['cov2d_scale']}")

    if args.stream_order == 'importance':
        order = importance_order(arrays)
    elif args.stream_order == 'shuffle':
        order = np.random.default_rng(0).permutation(n)
    else:
        order = None

    if order is not None:
        arrays = [a[order] for a in arrays]

    with open(args.output, 'wb') as fp:
        write_omg4_v2_header(
            fp, OMG4_MAGIC, n,
            header['time_min'], header['time_max'], header['fps'],
            flags=header['flags'] & OMG4_V2_FLAG_SH,
            cov2d_scale=header['cov2d_scale'],
            tile_size=args.tile_size,
        )
        write_omg4_v2_body_tiled(fp, arrays, args.tile_size)

    report_output(args.output)

    if args.verify:
        # Re-read the input fresh so verification is independent of the
        # in-memory permutation above.
        in_header, in_arrays = read_omg4_v2(args.input)
        verify_roundtrip(in_header, in_arrays, args.output)


if __name__ == '__main__':
    main()
