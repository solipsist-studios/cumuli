"""
sog_pack.py – Quantize OMG4 v2 splat arrays into the SOG-compressed
version-3 .omg4 container (webp attribute textures + k-means codebooks).

The static-attribute encoding follows the PlayCanvas SOG v2 conventions
exactly (see splat4d_io.py's v3 spec) so the engine's existing SOG decoder
reconstructs them unmodified; motion and trbf extend the scheme with two
additional textures.

Usage (repack an existing v2 file, standard or tiled):
    python sog_pack.py --input scene.omg4 --output scene_v3.omg4 \
        [--shn-count 65536] [--strip-sh] [--webp-method 4] [--verify]

Or call pack_v3() with field arrays directly (used by xz_to_omg4.py).
"""

import argparse
import io
import math
import struct
import sys

import numpy as np
from PIL import Image

from splat4d_io import (
    OMG4_MAGIC,
    OMG4_V2_FLAG_SH,
    OMG4_V2_FLAG_TILED,
    OMG4_V2_FIELDS,
    OMG4_V3_CODEBOOK_SIZE,
    OMG4_V3_SHN_WIDTHS,
    build_v3_meta,
    write_omg4_v3,
    report_output,
)
from omg4_repack import read_omg4_v2, read_omg4_v2_tiled

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def tex_dims(n: int):
    """Near-square texture dimensions holding n row-major texels."""
    w = int(math.ceil(math.sqrt(n)))
    h = int(math.ceil(n / w))
    return w, h


def to_rgba(w: int, h: int, channels, n: int) -> np.ndarray:
    """Build an [h, w, 4] uint8 image from per-channel uint8[n] arrays.

    `channels` is a 4-tuple; entries may be None (filled with 0) or a
    scalar (broadcast).  Padding texels beyond n are zero (never read by
    the decoder, and zeros compress best).
    """
    img = np.zeros((h * w, 4), dtype=np.uint8)
    for c, ch in enumerate(channels):
        if ch is None:
            continue
        if np.isscalar(ch):
            img[:n, c] = ch
        else:
            img[:n, c] = ch
    return img.reshape(h, w, 4)


def encode_webp(rgba: np.ndarray, method: int = 4) -> bytes:
    """Encode an RGBA image as LOSSLESS webp.

    `exact=True` is required: without it libwebp is free to rewrite the
    RGB of fully-transparent pixels, which destroys codebook indices
    stored alongside a zero alpha.
    """
    buf = io.BytesIO()
    Image.fromarray(rgba, 'RGBA').save(
        buf, format='WEBP', lossless=True, quality=100, method=method, exact=True)
    return buf.getvalue()


def decode_webp(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob)).convert('RGBA'))


def log_transform(x: np.ndarray) -> np.ndarray:
    """SOG means transform: sign(x) * ln(1 + |x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


def split16(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray):
    """Normalize [N, C] values over per-column mins/maxs to 16 bits and
    split into (low, high) uint8 arrays."""
    span = np.where(maxs - mins > 0, maxs - mins, 1.0)
    q = np.clip(np.rint((values - mins) / span * 65535.0), 0, 65535).astype(np.uint32)
    return (q & 0xFF).astype(np.uint8), (q >> 8).astype(np.uint8)


def kmeans_1d(values: np.ndarray, k: int = OMG4_V3_CODEBOOK_SIZE, iters: int = 16,
              log_domain: bool = False):
    """1-D k-means (quantile init + Lloyd).  Returns (codebook float64[k],
    indices uint8 shaped like `values`).

    With log_domain=True, clustering runs on ln(values) so precision is
    allocated by relative rather than absolute error (use for strictly
    positive scale-like attributes, e.g. t_sigma); the returned codebook
    is still in linear space, so decoders are unaffected.
    """
    shape = values.shape
    v = np.asarray(values, dtype=np.float64).ravel()
    if log_domain:
        centers, idx = kmeans_1d(np.log(np.maximum(v, 1e-9)), k, iters)
        return np.exp(centers), idx.reshape(shape)
    centers = np.quantile(v, np.linspace(0.0, 1.0, k))
    # collapse duplicate quantiles so every center is distinct
    centers += np.arange(k) * 1e-12
    for _ in range(iters):
        edges = (centers[1:] + centers[:-1]) * 0.5
        idx = np.searchsorted(edges, v)
        sums = np.bincount(idx, weights=v, minlength=k)
        counts = np.bincount(idx, minlength=k)
        nz = counts > 0
        centers[nz] = sums[nz] / counts[nz]
        centers.sort()
    edges = (centers[1:] + centers[:-1]) * 0.5
    idx = np.searchsorted(edges, v).astype(np.uint8)
    return centers, idx.reshape(shape)


def vq_vectors(data: np.ndarray, k: int, iters: int = 8,
               train_size: int = 1 << 18, chunk: int = 2048, seed: int = 0):
    """Vector-quantize [M, D] float32 rows into k centroids (Lloyd, torch,
    GPU when available).  Returns (centroids float32[k, D], labels uint32[M])."""
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = data.shape[0]
    k = min(k, m)
    rng = np.random.default_rng(seed)

    full = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).to(device)
    train = full[torch.from_numpy(rng.choice(m, size=min(train_size, m), replace=False)).to(device)] \
        if m > train_size else full
    centroids = train[torch.from_numpy(rng.choice(train.shape[0], size=k, replace=False)).to(device)].clone()

    def assign(rows):
        labels = torch.empty(rows.shape[0], dtype=torch.long, device=device)
        c_sq = (centroids * centroids).sum(dim=1)
        for s in range(0, rows.shape[0], chunk):
            block = rows[s:s + chunk]
            d = (block * block).sum(dim=1, keepdim=True) + c_sq - 2.0 * block @ centroids.T
            labels[s:s + chunk] = d.argmin(dim=1)
        return labels

    for _ in range(iters):
        labels = assign(train)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, labels, train)
        counts = torch.bincount(labels, minlength=k).unsqueeze(1)
        occupied = counts.squeeze(1) > 0
        centroids[occupied] = sums[occupied] / counts[occupied]
        n_empty = int((~occupied).sum())
        if n_empty:
            refill = train[torch.from_numpy(rng.choice(train.shape[0], size=n_empty, replace=False)).to(device)]
            centroids[~occupied] = refill

    labels = assign(full)
    return centroids.cpu().numpy(), labels.cpu().numpy().astype(np.uint32)


def morton_order(xyz: np.ndarray) -> np.ndarray:
    """Permutation sorting splats by 30-bit Morton code of position —
    the spatial ordering that makes the webp textures compress well."""
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    span = np.where(maxs - mins > 0, maxs - mins, 1.0)
    q = np.clip(((xyz - mins) / span * 1023.0), 0, 1023).astype(np.uint64)

    def spread(v):
        v = (v | (v << 16)) & np.uint64(0x030000FF)
        v = (v | (v << 8)) & np.uint64(0x0300F00F)
        v = (v | (v << 4)) & np.uint64(0x030C30C3)
        v = (v | (v << 2)) & np.uint64(0x09249249)
        return v

    code = spread(q[:, 0]) | (spread(q[:, 1]) << np.uint64(1)) | (spread(q[:, 2]) << np.uint64(2))
    return np.argsort(code, kind='stable')


# ---------------------------------------------------------------------------
# Attribute packers (each returns textures + meta fragments)
# ---------------------------------------------------------------------------

def pack_split16_pair(values3: np.ndarray, w: int, h: int, n: int):
    """Shared means/motion packer: log-transform, per-axis 16-bit split."""
    t = log_transform(values3.astype(np.float64))
    mins, maxs = t.min(axis=0), t.max(axis=0)
    lo, hi = split16(t, mins, maxs)
    tex_l = to_rgba(w, h, (lo[:, 0], lo[:, 1], lo[:, 2], 255), n)
    tex_u = to_rgba(w, h, (hi[:, 0], hi[:, 1], hi[:, 2], 255), n)
    return tex_l, tex_u, mins, maxs


def pack_quats(rot_wxyz: np.ndarray, w: int, h: int, n: int) -> np.ndarray:
    """Smallest-three quaternion encoding, mode in alpha (252 + dropped)."""
    q = rot_wxyz.astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # decoder component order is (x, y, z, w); v2 stores (w, x, y, z)
    xyzw = q[:, [1, 2, 3, 0]]
    dropped = np.argmax(np.abs(xyzw), axis=1)
    sign = np.sign(np.take_along_axis(xyzw, dropped[:, None], axis=1))
    sign[sign == 0] = 1.0
    xyzw *= sign

    # stored component triples per dropped index (see splat4d_io v3 spec):
    #   dropped x -> (w,y,z)  dropped y -> (w,x,z)  dropped z -> (w,x,y)
    #   dropped w -> (x,y,z);  mode byte = 252 + (dropped + 1) % 4
    keep = np.array([[3, 1, 2], [3, 0, 2], [3, 0, 1], [0, 1, 2]])
    stored = np.take_along_axis(xyzw, keep[dropped], axis=1)
    b = np.clip(np.rint((stored / SQRT2 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    mode = (252 + (dropped + 1) % 4).astype(np.uint8)
    return to_rgba(w, h, (b[:, 0], b[:, 1], b[:, 2], mode), n)


def pack_shn(f_rest: np.ndarray, shn_count: int, bands: int = 3):
    """VQ the [N, 45] higher-order SH into centroids + labels textures."""
    coeffs = {1: 3, 2: 8, 3: 15}[bands]
    dims = 3 * coeffs
    centroids, labels = vq_vectors(f_rest[:, :dims], shn_count)
    k = centroids.shape[0]

    codebook, cidx = kmeans_1d(centroids)
    # centroid texture: 64 palette entries per row, each `coeffs` texels
    # wide; texel (u+k, v) RGB = codebook indices of coefficient k for the
    # three color channels (channel-major f_rest layout: j*coeffs + k)
    width = OMG4_V3_SHN_WIDTHS[bands]
    height = int(math.ceil(k / 64))
    cent_img = np.zeros((height, width, 4), dtype=np.uint8)
    cent_img[..., 3] = 255
    entry = np.arange(k)
    u, v = (entry % 64) * coeffs, entry // 64
    for j in range(3):
        for c in range(coeffs):
            cent_img[v, u + c, j] = cidx[:, j * coeffs + c]

    return cent_img, labels, codebook, k


# ---------------------------------------------------------------------------
# Main packing entry point
# ---------------------------------------------------------------------------

def pack_v3(out_path: str, fields: dict, time_min: float, time_max: float,
            fps: float, shn_count: int = 65536, webp_method: int = 4,
            generator: str = 'volumetric-capture-pipeline sog_pack v1',
            segments=None, cov2d_scale=None, reorder: bool = True) -> dict:
    """Quantize v2-style field arrays and write a version-3 .omg4 archive.

    `fields` maps OMG4_V2_FIELDS names to float32[N] arrays, plus optional
    'f_rest' as float32[N, 45].  Returns the meta dict that was written.
    """
    n = len(fields['x'])
    xyz = np.stack([fields['x'], fields['y'], fields['z']], axis=1)

    if reorder:
        order = morton_order(xyz)
        fields = {k: (v[order] if v.ndim == 1 else v[order, :]) for k, v in fields.items()}
        xyz = xyz[order]

    w, h = tex_dims(n)
    textures = {}

    means_l, means_u, m_mins, m_maxs = pack_split16_pair(xyz, w, h, n)
    textures['means_l.webp'] = encode_webp(means_l, webp_method)
    textures['means_u.webp'] = encode_webp(means_u, webp_method)

    rot = np.stack([fields[f'rot_{i}'] for i in range(4)], axis=1)
    textures['quats.webp'] = encode_webp(pack_quats(rot, w, h, n), webp_method)

    scales = np.stack([fields[f'scale_{i}'] for i in range(3)], axis=1)
    scales_cb, scales_idx = kmeans_1d(scales)
    textures['scales.webp'] = encode_webp(
        to_rgba(w, h, (scales_idx[:, 0], scales_idx[:, 1], scales_idx[:, 2], 255), n), webp_method)

    f_dc = np.stack([fields[f'f_dc_{i}'] for i in range(3)], axis=1)
    sh0_cb, sh0_idx = kmeans_1d(f_dc)
    alpha = np.clip(np.rint(255.0 / (1.0 + np.exp(-fields['opacity'].astype(np.float64)))), 0, 255).astype(np.uint8)
    textures['sh0.webp'] = encode_webp(
        to_rgba(w, h, (sh0_idx[:, 0], sh0_idx[:, 1], sh0_idx[:, 2], alpha), n), webp_method)

    vel = np.stack([fields['vx'], fields['vy'], fields['vz']], axis=1)
    motion_l, motion_u, v_mins, v_maxs = pack_split16_pair(vel, w, h, n)
    textures['motion_l.webp'] = encode_webp(motion_l, webp_method)
    textures['motion_u.webp'] = encode_webp(motion_u, webp_method)

    tc_cb, tc_idx = kmeans_1d(fields['t_center'])
    ts_cb, ts_idx = kmeans_1d(fields['t_sigma'], log_domain=True)
    textures['trbf.webp'] = encode_webp(to_rgba(w, h, (tc_idx, ts_idx, None, 255), n), webp_method)

    shn_kwargs = {}
    if fields.get('f_rest') is not None and shn_count > 0:
        cent_img, labels, shn_cb, k = pack_shn(fields['f_rest'], shn_count)
        textures['shN_centroids.webp'] = encode_webp(cent_img, webp_method)
        textures['shN_labels.webp'] = encode_webp(
            to_rgba(w, h, ((labels & 0xFF).astype(np.uint8),
                           (labels >> 8).astype(np.uint8), None, 255), n), webp_method)
        shn_kwargs = {'shn_count': k, 'shn_bands': 3, 'shn_codebook': shn_cb}

    meta = build_v3_meta(
        count=n, time_min=time_min, time_max=time_max, fps=fps,
        means_mins=m_mins, means_maxs=m_maxs,
        scales_codebook=scales_cb, sh0_codebook=sh0_cb,
        motion_mins=v_mins, motion_maxs=v_maxs,
        trbf_center_codebook=tc_cb, trbf_sigma_codebook=ts_cb,
        segments=segments, cov2d_scale=cov2d_scale, generator=generator, **shn_kwargs)

    write_omg4_v3(out_path, meta, textures)
    return meta


# ---------------------------------------------------------------------------
# Verification: decode the archive back and report quantization error
# ---------------------------------------------------------------------------

def verify_v3(v3_path: str, fields: dict):
    """Decode the written archive exactly as the engine decoder would and
    report worst-case reconstruction error per attribute."""
    import json
    import zipfile

    zf = zipfile.ZipFile(v3_path)
    meta = json.loads(zf.read('meta.json'))
    n = meta['count']
    tex = {name: decode_webp(zf.read(name)).reshape(-1, 4)[:n]
           for name in zf.namelist() if name.endswith('.webp')}

    def unsplit16(l, u, mins, maxs, col):
        q = (u[:, col].astype(np.float64) * 256 + l[:, col]) / 65535.0
        t = mins[col] + q * (maxs[col] - mins[col])
        return np.sign(t) * (np.exp(np.abs(t)) - 1.0)

    order = morton_order(np.stack([fields['x'], fields['y'], fields['z']], axis=1))
    src = {k: (v[order] if v.ndim == 1 else v[order, :]) for k, v in fields.items()}

    report = {}

    def add(name, rec, ref):
        err = np.abs(rec - ref)
        report[name] = (float(err.mean()), float(np.percentile(err, 99)), float(err.max()))

    mm = meta['means']
    for c, name in enumerate('xyz'):
        rec = unsplit16(tex['means_l.webp'], tex['means_u.webp'],
                        np.array(mm['mins']), np.array(mm['maxs']), c)
        add(f'means.{name}', rec, src[name])

    scales_cb = np.array(meta['scales']['codebook'])
    for c in range(3):
        add(f'scale_{c}', scales_cb[tex['scales.webp'][:, c]], src[f'scale_{c}'])

    sh0_cb = np.array(meta['sh0']['codebook'])
    for c in range(3):
        add(f'f_dc_{c}', sh0_cb[tex['sh0.webp'][:, c]], src[f'f_dc_{c}'])
    alpha_src = 1.0 / (1.0 + np.exp(-src['opacity'].astype(np.float64)))
    add('opacity(linear)', tex['sh0.webp'][:, 3] / 255.0, alpha_src)

    vm = meta['motion']
    for c, name in enumerate(['vx', 'vy', 'vz']):
        rec = unsplit16(tex['motion_l.webp'], tex['motion_u.webp'],
                        np.array(vm['mins']), np.array(vm['maxs']), c)
        add(f'motion.{name}', rec, src[name])

    tc_cb = np.array(meta['trbf']['center']['codebook'])
    ts_cb = np.array(meta['trbf']['sigma']['codebook'])
    add('t_center', tc_cb[tex['trbf.webp'][:, 0]], src['t_center'])
    add('t_sigma', ts_cb[tex['trbf.webp'][:, 1]], src['t_sigma'])

    print('  Verify (reconstruction error per attribute):')
    print(f"    {'attribute':18s} {'mean':>10s} {'p99':>10s} {'max':>10s}")
    for k, (mean, p99, mx) in report.items():
        print(f'    {k:18s} {mean:10.6f} {p99:10.6f} {mx:10.6f}')
    return report


# ---------------------------------------------------------------------------
# CLI: v2 .omg4 -> v3
# ---------------------------------------------------------------------------

def fields_from_v2(path: str):
    """Read a standard or tiled v2 file into a name->array field dict."""
    with open(path, 'rb') as fp:
        head = fp.read(20)
    flags = struct.unpack_from('<I', head, 12)[0]
    header, arrays = (read_omg4_v2_tiled if flags & OMG4_V2_FLAG_TILED else read_omg4_v2)(path)

    fields = {name: np.asarray(arrays[i]) for i, name in enumerate(OMG4_V2_FIELDS)}
    if header['flags'] & OMG4_V2_FLAG_SH:
        fields['f_rest'] = np.stack(arrays[len(OMG4_V2_FIELDS):], axis=1)
    return header, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='v2 .omg4 file (standard or tiled)')
    parser.add_argument('--output', required=True, help='Destination v3 .omg4 archive')
    parser.add_argument('--shn-count', type=int, default=65536,
                        help='VQ centroid count for higher-order SH (default 65536)')
    parser.add_argument('--strip-sh', action='store_true', help='Drop higher-order SH entirely')
    parser.add_argument('--webp-method', type=int, default=4, choices=range(7),
                        help='libwebp effort 0-6 (default 4; 6 is smallest/slowest)')
    parser.add_argument('--verify', action='store_true',
                        help='Decode the output and report per-attribute quantization error')
    args = parser.parse_args()

    header, fields = fields_from_v2(args.input)
    n = header['num_splats']
    has_sh = 'f_rest' in fields
    if args.strip_sh:
        fields.pop('f_rest', None)
    print(f"Read {args.input}: {n:,} splats, SH={'yes' if has_sh else 'no'}"
          f"{' (stripped)' if args.strip_sh and has_sh else ''}")

    meta = pack_v3(args.output, fields, header['time_min'], header['time_max'],
                   header['fps'], shn_count=args.shn_count, webp_method=args.webp_method,
                   cov2d_scale=header.get('cov2d_scale'))

    report_output(args.output)
    import os
    in_size = os.path.getsize(args.input)
    out_size = os.path.getsize(args.output)
    print(f'  {in_size / 1e6:.1f} MB -> {out_size / 1e6:.1f} MB '
          f'({in_size / out_size:.1f}x, {out_size / n:.1f} bytes/splat)')

    if args.verify:
        verify_v3(args.output, fields)


if __name__ == '__main__':
    main()
