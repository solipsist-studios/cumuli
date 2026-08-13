#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
sog_pack.py – Quantize OMG4 v2 splat arrays into the SOG-compressed
version-3 .omg4 container (webp attribute textures + k-means codebooks).

The static-attribute encoding follows the PlayCanvas SOG v2 conventions
exactly (docs/sogst-format.md is the authoritative spec) so the engine's
existing SOG decoder reconstructs them unmodified; motion and trbf extend
the scheme with two additional textures.

Usage (repack an existing v2 file, standard or tiled):
    python sog_pack.py --input scene.omg4 --output scene.sogst \
        [--shn-count 65536] [--strip-sh] [--webp-method 4] [--verify]

Usage (pack a 4D interchange PLY -- the same input the TypeScript encoder
takes, so packing one PLY both ways is the cross-implementation check):
    python sog_pack.py --input scene.ply --output scene.sogst

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
    OMG4_V2_FLAG_ACCEL,
    OMG4_V2_FIELDS,
    OMG4_V3_CODEBOOK_SIZE,
    OMG4_V3_SHN_WIDTHS,
    build_v3_meta,
    write_omg4_v3,
    write_omg4_v3_streamed,
    report_output,
)
from omg4_repack import read_omg4_v2, read_omg4_v2_tiled

SQRT2 = math.sqrt(2.0)

# Upper bound on meta.segments.list length (spec section 5).  Every segment
# gets an entry whether or not it holds splats, so this bounds meta.json.
MAX_SEGMENTS = 65536


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


def compute_v3_order(fields: dict, time_min: float, time_max: float,
                     segment_duration: float = 0.1,
                     persistent_span_mult: float = 3.0,
                     k_sigma: float = 3.8):
    """Compute the splat ordering and temporal segment table.

    Splats whose active interval [t_center - k*sigma, t_center + k*sigma]
    spans more than persistent_span_mult segment durations are 'persistent'
    and placed first (always drawn).  The rest are bucketed by t_center
    into fixed segments and each bucket is Morton-ordered, so a player can
    draw [0, persistent_end) plus the contiguous index range of segments
    whose coverage overlaps the current time.  All emitted index ranges are
    half-open, [first, last).

    Returns (order, segments) where segments is the meta.json table, or
    (morton order, None) when segment_duration <= 0 (segmentation off).
    """
    xyz = np.stack([fields['x'], fields['y'], fields['z']], axis=1)
    if not segment_duration or segment_duration <= 0:
        return morton_order(xyz), None

    tc = fields['t_center'].astype(np.float64)
    ts = np.abs(fields['t_sigma'].astype(np.float64))
    lo, hi = tc - k_sigma * ts, tc + k_sigma * ts
    persistent = (hi - lo) > persistent_span_mult * segment_duration
    n_seg = max(1, int(math.ceil((time_max - time_min) / max(segment_duration, 1e-6))))
    # meta.segments.list carries one entry per segment whether or not it
    # holds splats, so a tiny duration on a long clip yields a valid file
    # whose meta.json is mostly an empty table.  Spec section 5 caps it.
    if n_seg > MAX_SEGMENTS:
        raise ValueError(
            f'segment_duration {segment_duration}s over a {time_max - time_min}s clip '
            f'needs {n_seg:,} segments, over the {MAX_SEGMENTS:,} limit. Use a longer '
            '--segment-duration (or 0 to disable segmentation).')
    bucket = np.clip(((tc - time_min) / segment_duration).astype(np.int64), 0, n_seg - 1)

    idx_all = np.arange(len(tc))
    p_idx = idx_all[persistent]
    if len(p_idx):
        p_idx = p_idx[morton_order(xyz[p_idx])]
    pieces = [p_idx]
    seg_list = []
    cursor = len(p_idx)
    for s in range(n_seg):
        mask = ~persistent & (bucket == s)
        s_idx = idx_all[mask]
        if len(s_idx):
            s_idx = s_idx[morton_order(xyz[s_idx])]
            t0, t1 = float(lo[mask].min()), float(hi[mask].max())
        else:
            t0 = time_min + s * segment_duration
            t1 = t0 + segment_duration
        pieces.append(s_idx)
        seg_list.append({'t0': t0, 't1': t1, 'range': [cursor, cursor + len(s_idx)]})
        cursor += len(s_idx)

    order = np.concatenate(pieces)
    segments = {
        'duration': float(segment_duration),
        'k_sigma': float(k_sigma),
        # Recorded so the persistent/dynamic split — and therefore the
        # file's whole ordering — can be re-derived from the file alone.
        # Players ignore it.  See docs/sogst-format.md section 5.
        'persistent_span_mult': float(persistent_span_mult),
        'persistent': [0, int(len(p_idx))],
        'list': seg_list,
    }
    return order, segments


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
# Attribute packers
# ---------------------------------------------------------------------------

def compute_split16_planes(values3: np.ndarray):
    """Shared means/motion quantizer: log-transform, per-axis 16-bit split.
    Returns (lo[N,3], hi[N,3], mins, maxs) byte planes."""
    t = log_transform(values3.astype(np.float64))
    mins, maxs = t.min(axis=0), t.max(axis=0)
    lo, hi = split16(t, mins, maxs)
    return lo, hi, mins, maxs


def compute_quat_planes(rot_wxyz: np.ndarray):
    """Smallest-three quaternion encoding.  Returns (bytes[N,3], mode[N])
    where mode = 252 + dropped-component code (texture alpha channel)."""
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
    return b, mode


def pack_shn(f_rest: np.ndarray, shn_count: int, bands: int = 3):
    """VQ the [N, 45] higher-order SH into centroids + labels textures."""
    coeffs = {1: 3, 2: 8, 3: 15}[bands]
    dims = 3 * coeffs
    centroids, labels = vq_vectors(f_rest[:, :dims], shn_count)
    k = centroids.shape[0]

    # NOT DONE, deliberately, with the measurement recorded so nobody has to
    # rediscover it: sorting palette entries by magnitude (and remapping
    # labels to match) makes each centroid-texture row a smooth ramp instead
    # of unrelated entries side by side.  It is exactly neutral on decoded
    # output -- every splat resolves to the same coefficients, and the 1-D
    # codebook below sees the same multiset either way -- so a decoder
    # cannot tell.  But the size effect is conditional, not a win:
    #
    #   heidi spm-native (75,848 splats): -32% shN_centroids, -6.0% file
    #   heidi full      (682,389 splats): +0.4% file
    #
    # shN_centroids is fixed-size at a given palette count while
    # shN_labels grows with the splat count, so the gain only survives when
    # centroids dominate.  Picking the smaller of the two per file would be
    # strictly better, but labels are encoded per group in group_textures(),
    # so measuring a total means encoding every group twice.  Not worth that
    # restructuring for a conditional few percent.
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
            cov2d_scale=None, segment_duration: float = 0.1,
            order_segments=None, stream: bool = True) -> dict:
    """Quantize v2-style field arrays and write a version-3 .omg4 archive.

    `fields` maps OMG4_V2_FIELDS names to float32[N] arrays, plus optional
    'f_rest' as float32[N, 45].  `order_segments` accepts a precomputed
    (order, segments) pair from compute_v3_order() (the CLI shares it with
    verify_v3); otherwise it is computed here.  Returns the written meta.

    With stream=True (and segmentation on), the archive uses the streamed
    layout: one webp texture set per group (persistent, then one per
    temporal segment), written in play order so a sequential download can
    decode and reveal the scene progressively.  Quantization is global
    either way — codebooks, mins/maxs and shN centroids live in meta.json
    and are shared by every group.
    """
    n = len(fields['x'])
    if order_segments is None:
        order_segments = compute_v3_order(fields, time_min, time_max, segment_duration)
    order, segments = order_segments
    fields = {k: (v[order] if v.ndim == 1 else v[order, :]) for k, v in fields.items()}
    xyz = np.stack([fields['x'], fields['y'], fields['z']], axis=1)

    # -- global quantization: per-splat byte planes + shared codebooks -----
    m_lo, m_hi, m_mins, m_maxs = compute_split16_planes(xyz)

    rot = np.stack([fields[f'rot_{i}'] for i in range(4)], axis=1)
    q_b, q_mode = compute_quat_planes(rot)

    scales = np.stack([fields[f'scale_{i}'] for i in range(3)], axis=1)
    scales_cb, scales_idx = kmeans_1d(scales)

    f_dc = np.stack([fields[f'f_dc_{i}'] for i in range(3)], axis=1)
    sh0_cb, sh0_idx = kmeans_1d(f_dc)
    alpha = np.clip(np.rint(255.0 / (1.0 + np.exp(-fields['opacity'].astype(np.float64)))), 0, 255).astype(np.uint8)

    vel = np.stack([fields['vx'], fields['vy'], fields['vz']], axis=1)
    v_lo, v_hi, v_mins, v_maxs = compute_split16_planes(vel)

    accel_kwargs = {}
    a_lo = a_hi = None
    if 'ax' in fields:
        accel = np.stack([fields['ax'], fields['ay'], fields['az']], axis=1)
        a_lo, a_hi, a_mins, a_maxs = compute_split16_planes(accel)
        accel_kwargs = {'accel_mins': a_mins, 'accel_maxs': a_maxs}

    tc_cb, tc_idx = kmeans_1d(fields['t_center'])
    ts_cb, ts_idx = kmeans_1d(fields['t_sigma'], log_domain=True)

    shn_kwargs = {}
    cent_blob = None
    labels = None
    if fields.get('f_rest') is not None and shn_count > 0:
        cent_img, labels, shn_cb, k = pack_shn(fields['f_rest'], shn_count)
        cent_blob = encode_webp(cent_img, webp_method)
        shn_kwargs = {'shn_count': k, 'shn_bands': 3, 'shn_codebook': shn_cb}

    # -- texture emission for an index range [a, b) ------------------------
    def group_textures(a: int, b: int) -> dict:
        w, h = tex_dims(b - a)
        m = b - a
        s = slice(a, b)
        images = {
            'means_l.webp': to_rgba(w, h, (m_lo[s, 0], m_lo[s, 1], m_lo[s, 2], 255), m),
            'means_u.webp': to_rgba(w, h, (m_hi[s, 0], m_hi[s, 1], m_hi[s, 2], 255), m),
            'quats.webp': to_rgba(w, h, (q_b[s, 0], q_b[s, 1], q_b[s, 2], q_mode[s]), m),
            'scales.webp': to_rgba(w, h, (scales_idx[s, 0], scales_idx[s, 1], scales_idx[s, 2], 255), m),
            'sh0.webp': to_rgba(w, h, (sh0_idx[s, 0], sh0_idx[s, 1], sh0_idx[s, 2], alpha[s]), m),
            'motion_l.webp': to_rgba(w, h, (v_lo[s, 0], v_lo[s, 1], v_lo[s, 2], 255), m),
            'motion_u.webp': to_rgba(w, h, (v_hi[s, 0], v_hi[s, 1], v_hi[s, 2], 255), m),
            'trbf.webp': to_rgba(w, h, (tc_idx[s], ts_idx[s], None, 255), m),
        }
        if a_lo is not None:
            images['accel_l.webp'] = to_rgba(w, h, (a_lo[s, 0], a_lo[s, 1], a_lo[s, 2], 255), m)
            images['accel_u.webp'] = to_rgba(w, h, (a_hi[s, 0], a_hi[s, 1], a_hi[s, 2], 255), m)
        if labels is not None:
            images['shN_labels.webp'] = to_rgba(
                w, h, ((labels[s] & 0xFF).astype(np.uint8),
                       (labels[s] >> 8).astype(np.uint8), None, 255), m)
        return {name: encode_webp(img, webp_method) for name, img in images.items()}

    meta = build_v3_meta(
        count=n, time_min=time_min, time_max=time_max, fps=fps,
        means_mins=m_mins, means_maxs=m_maxs,
        scales_codebook=scales_cb, sh0_codebook=sh0_cb,
        motion_mins=v_mins, motion_maxs=v_maxs,
        trbf_center_codebook=tc_cb, trbf_sigma_codebook=ts_cb,
        segments=segments, cov2d_scale=cov2d_scale, generator=generator,
        **accel_kwargs, **shn_kwargs)

    if segments and stream:
        # Streamed layout, SH deferred behind geometry:
        #   [meta | persistent/* | seg_NNN/* | shN_centroids | */shN_labels]
        # The reveal point covers geometry + DC color only: the viewer
        # starts DC-only playback after the first segment and layers the
        # view-dependent SH in as the trailing entries arrive.  Entry
        # names are unchanged — only their position in the archive moves.
        p_end = segments['persistent'][1]
        groups = [('persistent', 0, p_end)]
        prefixes = []
        for i, seg in enumerate(segments['list']):
            a, b = seg['range']
            prefix = f'seg_{i:03d}' if b > a else None
            prefixes.append(prefix)
            if prefix:
                groups.append((prefix, a, b))

        entries = []
        label_entries = []
        reveal_prefix = next((p for p in prefixes if p), None)
        reveal_through = 0
        for prefix, a, b in groups:
            if b <= a and prefix == 'persistent':
                continue
            textures = group_textures(a, b)
            labels_blob = textures.pop('shN_labels.webp', None)
            for name, blob in textures.items():
                entries.append((f'{prefix}/{name}', blob))
            if labels_blob is not None:
                label_entries.append((f'{prefix}/shN_labels.webp', labels_blob))
            if prefix in ('persistent', reveal_prefix):
                reveal_through = len(entries) - 1
        if cent_blob is not None:
            entries.append(('shN_centroids.webp', cent_blob))
        entries.extend(label_entries)

        meta['streams'] = {
            'persistent': 'persistent' if p_end > 0 else None,
            'segments': prefixes,
            'sh_deferred': bool(label_entries),
        }
        write_omg4_v3_streamed(out_path, meta, entries, reveal_through)
    else:
        textures = group_textures(0, n)
        if cent_blob is not None:
            textures['shN_centroids.webp'] = cent_blob
        write_omg4_v3(out_path, meta, textures)
    return meta


# ---------------------------------------------------------------------------
# Verification: decode the archive back and report quantization error
# ---------------------------------------------------------------------------

def verify_v3(v3_path: str, fields: dict, order: np.ndarray):
    """Decode the written archive exactly as the engine decoder would and
    report worst-case reconstruction error per attribute.  `order` must be
    the same permutation pack_v3 applied (from compute_v3_order)."""
    import json
    import zipfile

    zf = zipfile.ZipFile(v3_path)
    meta = json.loads(zf.read('meta.json'))
    n = meta['count']
    if meta.get('streams'):
        # streamed layout: reassemble full-length texel planes from the
        # per-group textures at their index ranges
        groups = []
        if meta['streams']['persistent']:
            groups.append((meta['streams']['persistent'], meta['segments']['persistent']))
        for prefix, seg in zip(meta['streams']['segments'], meta['segments']['list']):
            if prefix:
                groups.append((prefix, seg['range']))
        names = {name.split('/', 1)[1] for name in zf.namelist() if '/' in name}
        tex = {name: np.zeros((n, 4), dtype=np.uint8) for name in names}
        for prefix, (a, b) in groups:
            for name in names:
                tex[name][a:b] = decode_webp(zf.read(f'{prefix}/{name}')).reshape(-1, 4)[:b - a]
    else:
        tex = {name: decode_webp(zf.read(name)).reshape(-1, 4)[:n]
               for name in zf.namelist() if name.endswith('.webp')}

    def unsplit16(l, u, mins, maxs, col):
        q = (u[:, col].astype(np.float64) * 256 + l[:, col]) / 65535.0
        t = mins[col] + q * (maxs[col] - mins[col])
        return np.sign(t) * (np.exp(np.abs(t)) - 1.0)

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

    if meta.get('accel'):
        am = meta['accel']
        for c, name in enumerate(['ax', 'ay', 'az']):
            rec = unsplit16(tex['accel_l.webp'], tex['accel_u.webp'],
                            np.array(am['mins']), np.array(am['maxs']), c)
            add(f'accel.{name}', rec, src[name])

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
    cursor = len(OMG4_V2_FIELDS)
    if header['flags'] & OMG4_V2_FLAG_SH:
        fields['f_rest'] = np.stack(arrays[cursor:cursor + 45], axis=1)
        cursor += 45
    if header['flags'] & OMG4_V2_FLAG_ACCEL:
        for i, name in enumerate(('ax', 'ay', 'az')):
            fields[name] = np.asarray(arrays[cursor + i])
    return header, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True,
                        help='v2 .omg4 file (standard or tiled), or a 4D interchange '
                             '.ply (auto-detected by extension)')
    parser.add_argument('--output', required=True, help='Destination v3 .sogst archive')
    parser.add_argument('--shn-count', type=int, default=65536,
                        help='VQ centroid count for higher-order SH (default 65536)')
    parser.add_argument('--strip-sh', action='store_true', help='Drop higher-order SH entirely')
    parser.add_argument('--webp-method', type=int, default=4, choices=range(7),
                        help='libwebp effort 0-6 (default 4; 6 is smallest/slowest)')
    parser.add_argument('--segment-duration', type=float, default=0.1,
                        help='Temporal segment length in seconds for per-segment culling '
                             '(default 0.1; 0 disables segmentation)')
    parser.add_argument('--monolithic', action='store_true',
                        help='Write whole-clip textures instead of the streamed '
                             'per-segment layout (streamed is the default when '
                             'segmentation is on)')
    parser.add_argument('--verify', action='store_true',
                        help='Decode the output and report per-attribute quantization error')
    args = parser.parse_args()

    if args.input.endswith('.ply'):
        # Interchange PLY -> container. This is the path the TypeScript
        # encoder mirrors, so packing the same PLY both ways is the
        # cross-implementation equivalence check (docs/sogst-format.md
        # section 10).
        from sogst_ply import read_sogst_ply
        header, fields = read_sogst_ply(args.input)
        n = header['count']
    else:
        header, fields = fields_from_v2(args.input)
        n = header['num_splats']
    has_sh = 'f_rest' in fields
    if args.strip_sh:
        fields.pop('f_rest', None)
    print(f"Read {args.input}: {n:,} splats, SH={'yes' if has_sh else 'no'}"
          f"{' (stripped)' if args.strip_sh and has_sh else ''}")

    order_segments = compute_v3_order(fields, header['time_min'], header['time_max'],
                                      args.segment_duration)
    meta = pack_v3(args.output, fields, header['time_min'], header['time_max'],
                   header['fps'], shn_count=args.shn_count, webp_method=args.webp_method,
                   cov2d_scale=header.get('cov2d_scale'), order_segments=order_segments,
                   stream=not args.monolithic)

    segments = order_segments[1]
    if segments:
        p0, p1 = segments['persistent']
        seg_list = segments['list']
        counts = [s['range'][1] - s['range'][0] for s in seg_list]
        # expected drawn fraction: persistent plus the contiguous run of
        # segments whose coverage contains t, sampled across the clip
        drawn = []
        for t in np.linspace(header['time_min'], header['time_max'], 21):
            active = [s for s in seg_list if s['t0'] <= t <= s['t1']]
            dyn = (max(s['range'][1] for s in active) - min(s['range'][0] for s in active)) if active else 0
            drawn.append((p1 - p0 + dyn) / n)
        print(f"  Segments: {len(seg_list)} x {segments['duration']}s, "
              f"persistent {p1 - p0:,} ({(p1 - p0) / n:.1%}), "
              f"per-segment median {int(np.median(counts)):,} splats")
        print(f"  Drawn fraction with culling: mean {np.mean(drawn):.1%}, "
              f"max {np.max(drawn):.1%} (vs 100% without)")

    report_output(args.output)
    import os
    in_size = os.path.getsize(args.input)
    out_size = os.path.getsize(args.output)
    print(f'  {in_size / 1e6:.1f} MB -> {out_size / 1e6:.1f} MB '
          f'({in_size / out_size:.1f}x, {out_size / n:.1f} bytes/splat)')

    if args.verify:
        verify_v3(args.output, fields, order_segments[0])


if __name__ == '__main__':
    main()
