#!/usr/bin/env python3
"""
xz_to_omg4.py  –  Convert an OMG4 compressed 4DGS model (comp.xz) to the
compact .omg4 v2 binary format consumed by the supersplat-viewer.

OMG4 (arXiv 2510.03857) represents a dynamic scene with 4D space-time
Gaussians. Each Gaussian has a 4x4 covariance built from two quaternions
(a 4D rotor) and four scales (three spatial + one temporal). Rendering at
time t "slices" the 4D Gaussian:

    L      = R4(q_l, q_r) @ diag(exp(s_xyz), exp(s_t))
    Sigma  = L @ L^T                       (4x4)
    S11    = Sigma[:3,:3]  S12 = Sigma[:3,3]  Stt = Sigma[3,3]
    Sigma3D(t) = S11 - outer(S12, S12) / Stt      (constant in t)
    mean(t)    = xyz + (S12 / Stt) * (t - t_c)    (linear motion)
    opacity(t) = sigmoid(o) * exp(-0.5 * (t - t_c)^2 / Stt)

Color and (peak) opacity come from small MLPs evaluated on the contracted
position and normalised time. This exporter bakes every per-Gaussian
quantity once (evaluating the MLPs at each Gaussian's own temporal centre,
where it is most visible) and stores the temporal parameters explicitly so
the viewer can evaluate motion and temporal fade per rendered frame on the
GPU. There is no per-frame data: the file covers the full clip continuously.

Requirements: torch numpy dahuffman  (the OMG4 training environment)

Usage:
    python xz_to_omg4.py \
        --input  path/to/comp.xz \
        --output path/to/scene.omg4 \
        --time_min 0.0 --time_max 10.0 --fps 30

time_min/time_max MUST match the `time_duration` the model was trained
with (configs/dynerf/*.yaml in the OMG4 repo: [0.0, 10.0]); both the
temporal-opacity units and the MLP time normalisation depend on it.

.omg4 v2 file format (all values little-endian) — see splat4d_io.py for the
shared spec:
    Header (32 bytes):
        uint32  magic = 0x34474D4F ("OMG4")
        uint32  version = 2
        uint32  numSplats (N)
        uint32  flags = 0 (reserved)
        float32 timeMin, timeMax   (clip time range, seconds)
        float32 fps                (advisory, for UI only)
        uint32  reserved = 0
    Data: 19 SoA float32[N] arrays, in order:
        x y z                      position at t = t_center
        rot_0 rot_1 rot_2 rot_3    quaternion (w,x,y,z) of sliced 3D covariance
        scale_0 scale_1 scale_2    log-space sqrt eigenvalues of sliced 3D cov
        opacity                    logit-space peak opacity
        f_dc_0 f_dc_1 f_dc_2       SH DC coefficients
        vx vy vz                   velocity, scene units / second
        t_center                   temporal centre, seconds
        t_sigma                    temporal std-dev, seconds
"""

import argparse
import lzma
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from splat4d_io import OMG4_MAGIC, OMG4_V2_FIELDS, write_omg4_v2_header, report_output

# Set by main() when --v3 is given: {'shn_count': int, 'webp_method': int}.
# finish_export() then writes the SOG-compressed v3 container instead of
# the raw-float v2 layout (same filters, same field data).
V3_EXPORT_OPTIONS = None

try:
    import dahuffman
except ImportError:
    sys.exit("ERROR: 'dahuffman' package not found. Install it with: pip install dahuffman")


# ---------------------------------------------------------------------------
# SVQ / Huffman decode (mirrors utils/compress_utils.py + decode() in the
# OMG4 repository's scene/gaussian_model.py)
# ---------------------------------------------------------------------------

def huffman_decode(encoded_bytes, huffman_table, count):
    codec = dahuffman.HuffmanCodec(code_table=huffman_table)
    return np.fromiter(codec.decode(encoded_bytes), dtype=np.uint16, count=count)


def decode_all_layers(code_list, index_list, htable_list, count):
    """Decode a full VQ attribute (split into sub-vector slices)."""
    parts = []
    for codes, idx_bytes, htable in zip(code_list, index_list, htable_list):
        labels = huffman_decode(idx_bytes, htable, count)
        codes = np.asarray(codes, dtype=np.float32)
        parts.append(codes[labels])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# MLP forward pass.
#
# The published OMG4 checkpoints are trained with the TorchFallbackMLP in
# scene/gaussian_model.py (used when tiny-cuda-nn is not installed): plain
# nn.Linear layers WITH biases, no padding, hidden width 64, one hidden
# layer. The flat `params` vector is torch's parameters() order:
#   linear0.weight [64, n_in] | linear0.bias [64] |
#   linear1.weight [n_out, 64] | linear1.bias [n_out]
# Hidden activation: ReLU for mlp_cont, LeakyReLU(negative_slope=0.1) for
# mlp_dc / mlp_view / mlp_opacity. No output activation.
# ---------------------------------------------------------------------------

def unpack_linear_mlp(params_f16, n_in, n_hidden, n_out):
    p = np.asarray(params_f16, dtype=np.float32)
    expected = n_hidden * n_in + n_hidden + n_out * n_hidden + n_out
    if p.size != expected:
        raise ValueError(
            f"MLP param count mismatch: got {p.size}, expected {expected} "
            f"for Linear({n_in},{n_hidden})+Linear({n_hidden},{n_out}) with biases. "
            "Was this checkpoint trained with tiny-cuda-nn instead of the torch fallback?")
    o = 0
    w0 = p[o:o + n_hidden * n_in].reshape(n_hidden, n_in); o += n_hidden * n_in
    b0 = p[o:o + n_hidden]; o += n_hidden
    w1 = p[o:o + n_out * n_hidden].reshape(n_out, n_hidden); o += n_out * n_hidden
    b1 = p[o:o + n_out]
    return w0, b0, w1, b1


def mlp_forward(params_f16, x, n_hidden, n_out, hidden_activation):
    w0, b0, w1, b1 = unpack_linear_mlp(params_f16, x.shape[1], n_hidden, n_out)
    h = x @ w0.T + b0
    if hidden_activation == 'relu':
        h = np.maximum(h, 0.0)
    elif hidden_activation == 'leaky_relu':
        h = np.where(h >= 0.0, h, 0.1 * h)   # slope 0.1, matching TorchFallbackMLP
    else:
        raise ValueError(hidden_activation)
    return h @ w1.T + b1


def frequency_encode(x, n_frequencies=16):
    """TorchFallbackMLP ordering: per octave i, [sin(x*2^i*pi) (D dims), cos(...) (D dims)]."""
    out = []
    for i in range(n_frequencies):
        freq = (2.0 ** i) * math.pi
        out.append(np.sin(x * freq))
        out.append(np.cos(x * freq))
    return np.concatenate(out, axis=-1).astype(np.float32)


def contract_to_unisphere(x):
    """Scene contraction with the fixed aabb [-1,1]^3 used by the OMG4 renderer."""
    # aabb is [-1,1] so the normalisation-to-[-1,1] step is the identity
    x = x.copy()
    mag = np.linalg.norm(x, axis=-1, keepdims=True)
    mask = mag[:, 0] > 1
    x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
    return x / 4 + 0.5


# ---------------------------------------------------------------------------
# tiny-cuda-nn MLP forward (used by the FTGS variant checkpoints).
# FullyFusedMLP: no biases, dims padded to multiples of 16, params are
#   W0 [n_hidden, in_padded] | W1 [out_padded, n_hidden]   (row-major f16)
# Frequency encoding order per input dim d: [sin(2^0 pi x_d), cos(2^0 pi x_d),
# sin(2^1 pi x_d), ...]. Hidden LeakyReLU slope is tcnn's 0.01.
# ---------------------------------------------------------------------------

def _pad16(n):
    return ((n + 15) // 16) * 16


def tcnn_mlp_forward(params_f16, x, n_hidden, n_out, hidden_activation):
    p = np.asarray(params_f16, dtype=np.float32)
    n_in = x.shape[1]
    inp, outp = _pad16(n_in), _pad16(n_out)
    expected = n_hidden * inp + outp * n_hidden
    if p.size != expected:
        raise ValueError(f'tcnn MLP param count mismatch: got {p.size}, expected {expected}')
    w0 = p[:n_hidden * inp].reshape(n_hidden, inp)[:, :n_in]
    w1 = p[n_hidden * inp:].reshape(outp, n_hidden)[:n_out]
    h = x @ w0.T
    if hidden_activation == 'relu':
        h = np.maximum(h, 0.0)
    else:
        h = np.where(h >= 0.0, h, 0.01 * h)
    return h @ w1.T


def tcnn_frequency_encode(x, n_frequencies=16):
    N, D = x.shape
    out = np.empty((N, D * 2 * n_frequencies), dtype=np.float32)
    for d in range(D):
        for f in range(n_frequencies):
            a = x[:, d] * (2.0 ** f) * np.pi
            out[:, d * 2 * n_frequencies + 2 * f] = np.sin(a)
            out[:, d * 2 * n_frequencies + 2 * f + 1] = np.cos(a)
    return out


def bad_color_mask(f_dc):
    """Flag splats whose base (view-independent) colour is outside the
    displayable [0,1] range.

    f_dc comes straight out of an MLP with no output activation, so it is
    unbounded. An early version of this exporter CLAMPED these into [0,1]
    instead of dropping them -- that was wrong: clamping a garbage value to
    the nearest valid boundary doesn't recover the true colour, it just
    picks black or white almost arbitrarily depending which side of zero
    the garbage landed on. Scattered across a meaningful fraction of splats
    (observed: ~50% after aggressive SVQ appearance-codebook quantization
    on a complex scene) that produces a fine speckled/leafy noise texture,
    not a fixed image. Pruning is honest about the fact that this splat's
    decoded attributes are unreliable, same treatment as
    garbage_splat_mask() for geometry.
    """
    SH_C0 = 0.28209479177387814
    color = f_dc * SH_C0 + 0.5
    bad = ((color < 0) | (color > 1)).any(axis=1)
    print(f'  Bad-colour filter: dropping {int(bad.sum()):,} / {len(f_dc):,} splats with out-of-range f_dc')
    return ~bad


def clamp_sh_overshoot(f_dc, f_rest, sh_clamp):
    """Scale down each splat's higher SH bands so its colour stays bounded
    from every view direction.

    Splats observed from a narrow cone of training views can have SH
    coefficients that explode when evaluated from novel directions
    (saturated 'firework' spikes). A conservative bound on the band
    contribution is sum(|c_lm| * max|Y_lm|); per splat, if
    base + bound exceeds sh_clamp (in colour units), the f_rest coefficients
    are scaled so it does not. sh_clamp <= 0 disables.
    """
    if f_rest is None or sh_clamp <= 0:
        return f_rest
    # max |Y_lm| over the sphere for bands 1..3 (l=1: 0.489, l=2: up to 0.630, l=3: up to 0.746)
    y_max = np.array([0.489, 0.489, 0.489,
                      0.546, 0.546, 0.630, 0.546, 0.546,
                      0.590, 0.590, 0.457, 0.746, 0.457, 0.590, 0.590], dtype=np.float32)
    SH_C0 = 0.28209479177387814
    base = np.abs(f_dc * SH_C0 + 0.5)                                  # [N,3]
    bound = (np.abs(f_rest) * y_max[None, :, None]).sum(axis=1)        # [N,3]
    excess = (base + bound).max(axis=1)                                # [N]
    scale = np.minimum(1.0, np.maximum(sh_clamp - base.max(axis=1), 0.0) /
                       np.maximum(bound.max(axis=1), 1e-9))
    clamped = int((scale < 1.0).sum())
    print(f'  SH overshoot clamp: attenuated {clamped:,} / {len(scale):,} splats '
          f'(max total excursion was {excess.max():.2f})')
    return f_rest * scale[:, None, None]


# ---------------------------------------------------------------------------
# 4D rotor / covariance math (mirrors utils/general_utils.py)
# ---------------------------------------------------------------------------

def build_rotation_4d(l, r):
    """4D rotation matrices [N,4,4] from left/right quaternions (unnormalised ok)."""
    q_l = l / np.linalg.norm(l, axis=-1, keepdims=True)
    q_r = r / np.linalg.norm(r, axis=-1, keepdims=True)

    a, b, c, d = q_l[:, 0], q_l[:, 1], q_l[:, 2], q_l[:, 3]
    p, q, r_, s = q_r[:, 0], q_r[:, 1], q_r[:, 2], q_r[:, 3]

    N = l.shape[0]
    M_l = np.stack([a, -b, -c, -d,
                    b, a, -d, c,
                    c, d, a, -b,
                    d, -c, b, a], axis=0).reshape(4, 4, N).transpose(2, 0, 1)
    M_r = np.stack([p, q, r_, s,
                    -q, p, -s, r_,
                    -r_, s, p, -q,
                    -s, -r_, q, p], axis=0).reshape(4, 4, N).transpose(2, 0, 1)
    A = M_l @ M_r
    # matches torch.flip(A, [1, 2])
    return A[:, ::-1, ::-1]


def quat_from_rotmat(R):
    """Batch rotation matrix [N,3,3] -> quaternion [N,4] (w,x,y,z), w >= 0."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22

    # Case selection: trace, or the largest diagonal element.
    case = np.where(tr > 0, 0,
           np.where((m00 >= m11) & (m00 >= m22), 1,
           np.where(m11 >= m22, 2, 3)))

    s0 = np.sqrt(np.maximum(tr + 1.0, 0.0)) * 2               # 4*qw
    s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)) * 2  # 4*qx
    s2 = np.sqrt(np.maximum(1.0 - m00 + m11 - m22, 0.0)) * 2  # 4*qy
    s3 = np.sqrt(np.maximum(1.0 - m00 - m11 + m22, 0.0)) * 2  # 4*qz
    eps = 1e-12

    q0 = np.stack([0.25 * s0, (m21 - m12) / (s0 + eps), (m02 - m20) / (s0 + eps), (m10 - m01) / (s0 + eps)], axis=1)
    q1 = np.stack([(m21 - m12) / (s1 + eps), 0.25 * s1, (m01 + m10) / (s1 + eps), (m02 + m20) / (s1 + eps)], axis=1)
    q2 = np.stack([(m02 - m20) / (s2 + eps), (m01 + m10) / (s2 + eps), 0.25 * s2, (m12 + m21) / (s2 + eps)], axis=1)
    q3 = np.stack([(m10 - m01) / (s3 + eps), (m02 + m20) / (s3 + eps), (m12 + m21) / (s3 + eps), 0.25 * s3], axis=1)

    q = np.select([(case == 0)[:, None], (case == 1)[:, None], (case == 2)[:, None]], [q0, q1, q2], q3)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0
    return q.astype(np.float32)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def main_cluster_mask(xyz, opacity_logit, cell_frac=0.03):
    """Keep only the largest connected splat cluster (voxel 26-connectivity).

    Removes isolated floater clusters (mask noise, stray props) that
    reconstruct away from the subject. Cell size is cell_frac of the bbox
    diagonal; clusters are ranked by opacity mass.
    """
    from scipy import ndimage
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    cell = max(diag * cell_frac, 1e-6)
    ijk = np.floor((xyz - lo) / cell).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    grid = np.zeros(dims, dtype=bool)
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    labels, n_lab = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if n_lab <= 1:
        return np.ones(len(xyz), dtype=bool)
    splat_label = labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    w = 1.0 / (1.0 + np.exp(-opacity_logit))
    mass = np.bincount(splat_label, weights=w, minlength=n_lab + 1)
    mass[0] = 0
    keep = splat_label == mass.argmax()
    print(f'  Main-cluster filter: kept {int(keep.sum()):,} / {len(xyz):,} splats '
          f'({n_lab} clusters found)')
    return keep


def garbage_splat_mask(log_scales, max_scale_thresh=0.2, max_aspect_thresh=10.0):
    """Flag splats with grossly oversized, needle-thin covariance.

    A small fraction of splats (observed: ~1.6% of a compressed model, but
    ~96% correlated with also having an out-of-range base colour -- see
    bad_color_mask, which independently drops those) come out of
    SVQ-quantized decode with geometry that was never anything real:
    eigenvalue aspect ratios up to 10^6:1 and a long axis tens of times the
    population median. They are usually also disproportionately HIGH
    opacity, so a handful of them paint a bright streak clear across the
    frame. Kept as a separate check because a few percent of them have an
    in-range colour and would otherwise slip through bad_color_mask alone.
    """
    scale = np.exp(log_scales)
    max_scale = scale.max(axis=1)
    min_scale = np.maximum(scale.min(axis=1), 1e-12)
    aspect = max_scale / min_scale
    bad = (max_scale > max_scale_thresh) & (aspect > max_aspect_thresh)
    print(f'  Garbage-geometry filter: dropping {int(bad.sum()):,} / {len(scale):,} splats '
          f'(max-axis>{max_scale_thresh}, aspect>{max_aspect_thresh}:1)')
    return ~bad


def dark_occluder_mask(f_dc, opacity_logit, log_scales, brightness_thresh=0.05,
                       opacity_thresh=0.7, size_percentile=75):
    """Flag oversized, near-black, high-opacity splats.

    A known Gaussian-splatting failure mode: the optimizer grows dark
    "shadow-filling" splats to explain away self-occlusion/shadowing that's
    only visible from a subset of training cameras -- this capture has a
    lot of that (fast human motion, limbs crossing the body). Those splats
    look correct from the cameras that saw the shadow, but from a novel
    angle they just sit there as opaque black blobs occluding the real
    geometry behind them. Distinguished from legitimate dark material
    (black shoes, hair shadow, the subject's own dark blazer) by ALSO
    requiring an above-typical size: real fine dark detail tends to be
    normal-sized, not oversized. Observed correlating with a modest spatial
    offset from the main "bright, well-defined subject" splat population,
    consistent with being a distinct (if intermixed) failure population
    rather than the subject's true surface.
    """
    SH_C0 = 0.28209479177387814
    color = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)
    brightness = color.mean(axis=1)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))
    scale = np.exp(log_scales)
    max_scale = scale.max(axis=1)
    size_thresh = np.percentile(max_scale, size_percentile)
    bad = (brightness < brightness_thresh) & (opacity > opacity_thresh) & (max_scale > size_thresh)
    print(f'  Dark-occluder filter: dropping {int(bad.sum()):,} / {len(f_dc):,} splats '
          f'(brightness<{brightness_thresh}, opacity>{opacity_thresh}, size>p{size_percentile})')
    return ~bad


def black_floater_mask(f_dc, opacity_logit, log_scales,
                       black_thresh=-0.25, dim_thresh=0.05,
                       alpha_thresh=0.30, aspect_thresh=6.0):
    """Flag the black-floater cloud an SPM fine-tune leaves around a masked
    subject, without deleting its legitimate dark material.

    The SPM export's below-range-DC population does double duty: most of it
    is the subject's own dark clothing (deleting it all is the transparency
    failure bad_color_mask caused), but a subset is detached near-black
    streaks that read as an "evil cloud" against a light viewer background
    (invisible over the black eval background, which is how it survived the
    metric checks). The junk subset is distinguished by being *deeply*
    negative in colour (mean below black_thresh -- no SH band can bring it
    back), or dark (below dim_thresh) while ALSO low-alpha (murk, not
    surface) or spike-elongated. Calibrated on the tatum SPM export, where
    the drop set turned out to be ~95% identical to bad_color_mask's -- the
    old filter had the right suspects; the washout it caused came from the
    sh_clamp interaction, not the deletions. This variant keeps the ~5%
    that are real surface and, unlike bad_color_mask, never touches the
    above-range (bright) side.
    """
    SH_C0 = 0.28209479177387814
    mean_col = (f_dc * SH_C0 + 0.5).mean(axis=1)
    alpha = 1.0 / (1.0 + np.exp(-opacity_logit))
    scale = np.exp(log_scales)
    aspect = scale.max(axis=1) / np.maximum(scale.min(axis=1), 1e-12)
    bad = (mean_col < black_thresh) | (
        (mean_col < dim_thresh) & ((alpha < alpha_thresh) | (aspect > aspect_thresh)))
    print(f'  Black-floater filter: dropping {int(bad.sum()):,} / {len(f_dc):,} splats '
          f'(mean colour<{black_thresh}, or <{dim_thresh} with alpha<{alpha_thresh} '
          f'or aspect>{aspect_thresh}:1)')
    return ~bad


def finish_export(out_path, time_min, time_max, fps, prune_threshold, cov2d_scale,
                  xyz, quat, log_scales, opacity_logit, f_dc, f_rest,
                  velocity, t_center, t_sigma, keep_main_cluster=False, filter_corrupted=True,
                  filter_dark_occluders=False, filter_black_floaters=False,
                  extra_keep_mask=None):
    """Shared tail: prune, diagnostics, write the v2 file.

    filter_corrupted controls the bad_color_mask/garbage_splat_mask checks,
    which were calibrated against SVQ compression's specific failure mode
    (catastrophic MLP extrapolation on badly-quantized inputs -- f_dc values
    in the hundreds). convert_from_checkpoint()'s data doesn't have that
    failure mode (no quantization at all): a bare f_dc outside [0,1] there
    is usually just normal SH representation (the real colour only becomes
    valid once combined with f_rest, which clamp_sh_overshoot already
    accounts for), not corruption. Applying these filters there flagged
    ~34% of splats as "bad" -- clearly miscalibrated for clean data -- so
    convert_from_checkpoint() passes filter_corrupted=False and relies on
    prune_threshold + clamp_sh_overshoot + keep_main_cluster instead.
    """
    N = xyz.shape[0]
    dist = np.abs(t_center - np.clip(t_center, time_min, time_max))
    peak_weight = np.exp(-0.5 * (dist / np.maximum(t_sigma, 1e-9)) ** 2)
    peak_alpha = (1.0 / (1.0 + np.exp(-opacity_logit))) * peak_weight
    keep = peak_alpha >= prune_threshold
    if filter_corrupted:
        keep &= garbage_splat_mask(log_scales)
        keep &= bad_color_mask(f_dc)
    if filter_dark_occluders:
        keep &= dark_occluder_mask(f_dc, opacity_logit, log_scales)
    if filter_black_floaters:
        keep &= black_floater_mask(f_dc, opacity_logit, log_scales)
    if extra_keep_mask is not None:
        n_before = int(keep.sum())
        keep &= extra_keep_mask
        print(f'  External keep-mask (e.g. real-mask consistency): dropping '
              f'{n_before - int(keep.sum()):,} additional splats')
    if keep_main_cluster:
        keep &= main_cluster_mask(xyz, opacity_logit)
    kept = int(keep.sum())
    print(f"  Pruning: dropped {N - kept:,} / {N:,} Gaussians below alpha {prune_threshold:.5f} inside time range")

    arrays = [
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3],
        log_scales[:, 0], log_scales[:, 1], log_scales[:, 2],
        opacity_logit,
        f_dc[:, 0], f_dc[:, 1], f_dc[:, 2],
        velocity[:, 0], velocity[:, 1], velocity[:, 2],
        t_center,
        t_sigma
    ]
    if f_rest is not None:
        # PLY channel-major order: f_rest_{ch*15 + k} = eff_rest[k][channel ch]
        for ch in range(3):
            for k in range(15):
                arrays.append(f_rest[:, k, ch])
    arrays = [np.ascontiguousarray(a[keep], dtype=np.float32) for a in arrays]

    for t in np.linspace(time_min, time_max, 5):
        w = np.exp(-0.5 * ((t_center[keep] - t) / np.maximum(t_sigma[keep], 1e-9)) ** 2)
        alpha = (1.0 / (1.0 + np.exp(-opacity_logit[keep]))) * w
        print(f"    t={t:6.2f}s  active(alpha>1/255): {(alpha > 1 / 255).mean() * 100:5.1f}%")
    vmag = np.linalg.norm(velocity[keep], axis=1)
    print(f"    |velocity| p50/p95/p99: {np.percentile(vmag, [50, 95, 99]).round(4)}")
    print(f"    t_sigma    p50/p95    : {np.percentile(t_sigma[keep], [50, 95]).round(3)}")

    if V3_EXPORT_OPTIONS is not None:
        from sog_pack import pack_v3
        fields = {name: arrays[i] for i, name in enumerate(OMG4_V2_FIELDS)}
        if f_rest is not None:
            fields['f_rest'] = np.stack(arrays[len(OMG4_V2_FIELDS):], axis=1)
        pack_v3(out_path, fields, time_min, time_max, fps,
                cov2d_scale=cov2d_scale, **V3_EXPORT_OPTIONS)
    else:
        flags = 1 if f_rest is not None else 0
        with open(out_path, 'wb') as fp:
            write_omg4_v2_header(fp, OMG4_MAGIC, kept, time_min, time_max, fps, flags,
                                 cov2d_scale=cov2d_scale)
            for a in arrays:
                fp.write(a.tobytes())

    report_output(out_path)


def convert_from_checkpoint(checkpoint_path, out_path, time_min, time_max, fps, prune_threshold,
                            include_sh=True, scale_boost=1.0, sh_clamp=1.5, keep_main_cluster=False,
                            top_k_fraction=1.0, extra_keep_mask_path=None):
    """Export directly from an OMG4 train_scratch.py checkpoint (chkpntNNNN.pth),
    bypassing OMG4's SVQ+MLP compression (train.py) entirely.

    The checkpoint stores full per-splat SH coefficients (gaussian_model.py
    capture()/restore(), gaussian_dim==4 branch) -- no quantization, no MLP
    distillation. Compression's codebooks proved too small for a complex
    scene (dense self-motion, thousands of training views): a large
    fraction of splats got quantized into appearance-feature combinations
    the compact appearance MLPs never saw in training and extrapolated
    into unusable colour/geometry. Reading the checkpoint directly can't
    reproduce that failure mode since there's no quantization step at all.
    Same 4D-covariance-slicing + temporal-SH-fold math as convert(), just
    sourced from real per-splat data instead of a decoded codebook.

    top_k_fraction (0.0-1.0, default 1.0 = keep everything): keep only the
    top fraction of splats ranked by peak visibility (opacity x temporal-
    fade weight at each splat's own t_center). A size/quality knob that
    degrades gracefully -- drops the least-visible splats first -- rather
    than reusing the SVQ pipeline, which corrupts unpredictably instead of
    shrinking gracefully.
    """
    print(f"Loading checkpoint {checkpoint_path} …")
    model_args, _first_iter = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    (_active_sh_degree, xyz_t, features_dc_t, features_rest_t, scaling_raw_t, rotation_t, opacity_t,
     _max_radii2D, _xyz_grad_accum, _t_grad_accum, _denom, _opt_dict, _spatial_lr_scale,
     t_center_t, scaling_t_t, rotation_r_t, _rot_4d, _env_map, _active_sh_degree_t) = model_args

    xyz = to_numpy(xyz_t)
    scaling = to_numpy(scaling_raw_t)                 # [N,3] log
    rotation_l = to_numpy(rotation_t)                 # [N,4]
    rotation_r = to_numpy(rotation_r_t)                # [N,4]
    scaling_tt = to_numpy(scaling_t_t)[:, 0]           # [N] log
    t_center = to_numpy(t_center_t)[:, 0]              # [N]
    opacity_logit = to_numpy(opacity_t)[:, 0]          # [N]
    features_dc = to_numpy(features_dc_t)[:, 0, :]     # [N,3]
    features_rest = to_numpy(features_rest_t)          # [N,47,3] -- same layout as convert()'s view_sh

    N = xyz.shape[0]
    print(f"  {N:,} Gaussians (uncompressed checkpoint) | time range [{time_min}, {time_max}] s")

    extra_keep_mask = None
    if extra_keep_mask_path is not None:
        extra_keep_mask = np.load(extra_keep_mask_path)
        if extra_keep_mask.shape[0] != N:
            raise ValueError(f'extra_keep_mask has {extra_keep_mask.shape[0]} entries, '
                             f'checkpoint has {N} splats -- must be computed against this exact checkpoint')
        n_drop = int((~extra_keep_mask).sum())
        print(f"  External keep-mask ({extra_keep_mask_path}): dropping {n_drop:,} / {N:,} splats "
              f"before covariance slicing")
        xyz, scaling, rotation_l, rotation_r = (xyz[extra_keep_mask], scaling[extra_keep_mask],
                                                 rotation_l[extra_keep_mask], rotation_r[extra_keep_mask])
        scaling_tt, t_center, opacity_logit = (scaling_tt[extra_keep_mask], t_center[extra_keep_mask],
                                               opacity_logit[extra_keep_mask])
        features_dc, features_rest = features_dc[extra_keep_mask], features_rest[extra_keep_mask]
        N = xyz.shape[0]

    print("  Slicing 4D covariance …")
    R4 = build_rotation_4d(rotation_l, rotation_r).astype(np.float64)
    s4 = np.concatenate([np.exp(scaling), np.exp(scaling_tt)[:, None]], axis=1).astype(np.float64)
    L = R4 * s4[:, None, :]
    Sigma = L @ L.transpose(0, 2, 1)
    S11 = Sigma[:, :3, :3]
    S12 = Sigma[:, :3, 3]
    Stt = np.maximum(Sigma[:, 3, 3], 1e-12)
    velocity = (S12 / Stt[:, None]).astype(np.float32)
    t_sigma = np.sqrt(Stt).astype(np.float32)
    Sigma3D = S11 - np.einsum('ni,nj->nij', S12, S12) / Stt[:, None, None]
    eigval, eigvec = np.linalg.eigh(Sigma3D)
    eigval = np.maximum(eigval, 1e-18)
    neg = np.linalg.det(eigvec) < 0
    eigvec[neg, :, 2] *= -1
    quat = quat_from_rotmat(eigvec)
    log_scales = (0.5 * np.log(eigval) + math.log(max(scale_boost, 1e-6))).astype(np.float32)

    # Temporal SH fold at t = t_center (dirs_t = 0 -> both cosine bands = 1),
    # matching eval_shfs_4d()'s coefficient layout (utils/sh_utils.py): index
    # 0 = static DC, 16 = t1*DC, 32 = t2*DC fold into f_dc; the three
    # temporal copies of spatial bands 1..15 fold into f_rest. Identical
    # arithmetic to convert()'s MLP path, just fed real coefficients.
    print("  Folding temporal SH at each splat's own t_center …")
    f_dc = features_dc + features_rest[:, 15, :] + features_rest[:, 31, :]
    f_rest = None
    if include_sh:
        f_rest = (features_rest[:, 0:15, :] + features_rest[:, 16:31, :] + features_rest[:, 32:47, :])
        f_rest = clamp_sh_overshoot(f_dc, f_rest, sh_clamp)

    if top_k_fraction < 1.0:
        dist = np.abs(t_center - np.clip(t_center, time_min, time_max))
        peak_weight = np.exp(-0.5 * (dist / np.maximum(t_sigma, 1e-9)) ** 2)
        peak_alpha = (1.0 / (1.0 + np.exp(-opacity_logit))) * peak_weight
        k = max(1, min(N, int(round(top_k_fraction * N))))
        thresh = np.partition(peak_alpha, N - k)[N - k]
        top_mask = peak_alpha >= thresh
        print(f"  top_k_fraction={top_k_fraction}: keeping {int(top_mask.sum()):,} / {N:,} "
              f"highest-visibility splats")
        xyz, quat, log_scales, opacity_logit = (xyz[top_mask], quat[top_mask],
                                                 log_scales[top_mask], opacity_logit[top_mask])
        f_dc, velocity, t_center, t_sigma = (f_dc[top_mask], velocity[top_mask],
                                             t_center[top_mask], t_sigma[top_mask])
        if f_rest is not None:
            f_rest = f_rest[top_mask]

    finish_export(out_path, time_min, time_max, fps, prune_threshold, None,
                  xyz, quat, log_scales, opacity_logit, f_dc, f_rest,
                  velocity, t_center, t_sigma, keep_main_cluster=keep_main_cluster,
                  filter_corrupted=False, filter_dark_occluders=(extra_keep_mask is None))


def convert_ftgs(save_dict, out_path, time_min, time_max, fps, prune_threshold,
                 include_sh=True, sh_clamp=1.5, keep_main_cluster=False):
    """FTGS-variant checkpoint (OMG4_FTGS): explicit velocity + temporal
    opacity, gsplat-trained (no FoV-sentinel bug -> no compensation), MLPs in
    tiny-cuda-nn layout with a 3-D (xyz-only) frequency-encoded input."""
    means = np.asarray(save_dict['means'], dtype=np.float32)          # [N,3]
    times = np.asarray(save_dict['times'], dtype=np.float32).reshape(-1)  # [N]
    N = means.shape[0]
    print(f"  FTGS checkpoint: {N:,} Gaussians | time range [{time_min}, {time_max}] s")

    log_scales = decode_all_layers(save_dict['scale_code'], save_dict['scale_index'], save_dict['scale_htable'], N)      # [N,3] log
    quats = decode_all_layers(save_dict['rotation_code'], save_dict['rotation_index'], save_dict['rotation_htable'], N)  # [N,4] (w,x,y,z)
    durations = decode_all_layers(save_dict['durations_code'], save_dict['durations_index'], save_dict['durations_htable'], N).reshape(-1)  # log
    velocity = decode_all_layers(save_dict['velocities_code'], save_dict['velocities_index'], save_dict['velocities_htable'], N)  # [N,3]
    appearance = decode_all_layers(save_dict['app_code'], save_dict['app_index'], save_dict['app_htable'], N)            # [N,6]
    t_sigma = np.exp(durations).astype(np.float32)
    # normalise quats; renderer expects w-first which matches gsplat's storage
    quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-9)

    print("  Evaluating appearance MLPs (tcnn layout) …")
    xyz_c = contract_to_unisphere(means.astype(np.float64)).astype(np.float32)
    encoded = tcnn_frequency_encode(xyz_c)                                       # [N,96]
    cont_feat = tcnn_mlp_forward(save_dict['MLP_cont'], encoded, 64, 13, 'relu')  # [N,13]
    space_feat = np.concatenate([cont_feat, appearance[:, 0:3]], axis=1)          # [N,16]
    f_dc = tcnn_mlp_forward(save_dict['MLP_dc'], space_feat, 64, 3, 'leaky')      # [N,3]
    opacity_logit = tcnn_mlp_forward(save_dict['MLP_opacity'], space_feat, 64, 1, 'leaky')[:, 0]

    f_rest = None
    if include_sh:
        view_feat = np.concatenate([cont_feat, appearance[:, 3:6]], axis=1)
        view_sh = tcnn_mlp_forward(save_dict['MLP_sh'], view_feat, 64, 141, 'leaky').reshape(-1, 47, 3)
        # gsplat consumes the first (deg+1)^2 = 16 coefficients of [dc, view]:
        # dc is coeff 0, view[0:15] are the deg 1..3 spatial coefficients.
        f_rest = view_sh[:, 0:15, :].copy()
        f_rest = clamp_sh_overshoot(f_dc, f_rest, sh_clamp)

    finish_export(out_path, time_min, time_max, fps, prune_threshold, None,
                  means, quats, log_scales.astype(np.float32), opacity_logit,
                  f_dc, f_rest, velocity, times, t_sigma,
                  keep_main_cluster=keep_main_cluster)


def convert(xz_path, out_path, time_min, time_max, fps, prune_threshold, include_sh=True,
            scale_boost=1.0, aniso_boost=None, aniso_camera_rotations=None,
            cov2d_scale=None, sh_clamp=1.5, keep_main_cluster=False,
            filter_corrupted=True, filter_black_floaters=False):
    print(f"Loading {xz_path} …")
    with lzma.open(xz_path, "rb") as f:
        save_dict = pickle.load(f)

    if 'means' in save_dict:
        convert_ftgs(save_dict, out_path, time_min, time_max, fps, prune_threshold,
                     include_sh=include_sh, sh_clamp=sh_clamp,
                     keep_main_cluster=keep_main_cluster)
        return

    xyz = to_numpy(save_dict['xyz'])                # [N,3]
    t_center = to_numpy(save_dict['t'])[:, 0]       # [N]
    N = xyz.shape[0]

    print(f"  {N:,} Gaussians | time range [{time_min}, {time_max}] s")

    scaling = decode_all_layers(save_dict['scale_code'], save_dict['scale_index'], save_dict['scale_htable'], N)          # [N,3] log
    rotation_l = decode_all_layers(save_dict['rotation_code'], save_dict['rotation_index'], save_dict['rotation_htable'], N)  # [N,4]
    rotation_r = decode_all_layers(save_dict['rotation_r_code'], save_dict['rotation_r_index'], save_dict['rotation_r_htable'], N)  # [N,4]
    scaling_t = decode_all_layers(save_dict['scaling_t_code'], save_dict['scaling_t_index'], save_dict['scaling_t_htable'], N)[:, 0]  # [N] log
    appearance = decode_all_layers(save_dict['app_code'], save_dict['app_index'], save_dict['app_htable'], N)             # [N,6]
    features_static = appearance[:, 0:3]

    # ── 4D covariance slicing ───────────────────────────────────────────────
    print("  Slicing 4D covariance …")
    R4 = build_rotation_4d(rotation_l, rotation_r).astype(np.float64)         # [N,4,4]
    s4 = np.concatenate([np.exp(scaling), np.exp(scaling_t)[:, None]], axis=1).astype(np.float64)  # [N,4]
    L = R4 * s4[:, None, :]                                                    # R4 @ diag(s)
    Sigma = L @ L.transpose(0, 2, 1)                                           # [N,4,4]

    S11 = Sigma[:, :3, :3]
    S12 = Sigma[:, :3, 3]                                                      # [N,3]
    Stt = np.maximum(Sigma[:, 3, 3], 1e-12)                                    # [N]

    velocity = (S12 / Stt[:, None]).astype(np.float32)                         # units/sec
    t_sigma = np.sqrt(Stt).astype(np.float32)
    Sigma3D = S11 - np.einsum('ni,nj->nij', S12, S12) / Stt[:, None, None]

    # Anisotropic compensation for the FoV-sentinel training bug: inflate each
    # splat's covariance by (kx, ky) along the average training-camera x/y axes
    # (a good approximation when the rig's cameras share one orientation, as in
    # N3V). Exact per-view compensation would be screen-space; this bakes the
    # dominant effect into world space so any renderer benefits.
    if aniso_boost is not None:
        kx, ky = aniso_boost
        cam_rots = np.array(aniso_camera_rotations)                      # [C,3,3] C2W
        # average camera basis via SVD orthonormalisation of the mean rotation
        u, _, vt = np.linalg.svd(cam_rots.mean(axis=0))
        R_avg = u @ vt
        M = R_avg @ np.diag([kx, ky, 1.0]) @ R_avg.T
        Sigma3D = np.einsum('ij,njk,lk->nil', M, Sigma3D, M)

    # Eigendecompose the sliced covariance -> principal axes + scales
    eigval, eigvec = np.linalg.eigh(Sigma3D)                                   # ascending
    eigval = np.maximum(eigval, 1e-18)
    # ensure right-handed rotation before quaternion conversion
    neg = np.linalg.det(eigvec) < 0
    eigvec[neg, :, 2] *= -1
    quat = quat_from_rotmat(eigvec)                                            # (w,x,y,z)
    log_scales = (0.5 * np.log(eigval) + math.log(max(scale_boost, 1e-6))).astype(np.float32)

    # ── MLP appearance at each Gaussian's own temporal centre ───────────────
    print("  Evaluating appearance MLPs …")
    t_eval = np.clip(t_center, time_min, time_max)
    t_norm = (t_eval - time_min) / max(time_max - time_min, 1e-9)

    xyz_c = contract_to_unisphere(xyz.astype(np.float64)).astype(np.float32)
    xyzt = np.concatenate([xyz_c, t_norm[:, None].astype(np.float32)], axis=1)  # [N,4]

    encoded = frequency_encode(xyzt, n_frequencies=16)                          # [N,128]
    cont_feat = mlp_forward(save_dict['MLP_cont'], encoded, 64, 13, 'relu')     # [N,13]
    space_feat = np.concatenate([cont_feat, features_static], axis=1)           # [N,16]

    f_dc = mlp_forward(save_dict['MLP_dc'], space_feat, 64, 3, 'leaky_relu')    # [N,3] SH DC
    opacity_logit = mlp_forward(save_dict['MLP_opacity'], space_feat, 64, 1, 'leaky_relu')[:, 0]  # [N]

    # ── View-dependent SH (optional) ────────────────────────────────────────
    # The reference model evaluates 48 4D-spherindrical-harmonic coefficients:
    # 16 spatial (deg 3) x 3 temporal cosine bands (utils/sh_utils.py
    # eval_shfs_4d). At dir_t = t - t_center = 0 both cosine bands equal 1, so
    # the effective spatial coefficient k collapses to
    #   eff[k] = sh[k] + sh[k+16] + sh[k+32]
    # which is exactly a standard 3-band 3DGS SH set. Coefficient 0 folds into
    # f_dc; coefficients 1..15 become f_rest (PLY channel-major layout).
    f_rest = None
    if include_sh:
        features_view = appearance[:, 3:6]
        view_feat = np.concatenate([cont_feat, features_view], axis=1)          # [N,16]
        view_sh = mlp_forward(save_dict['MLP_sh'], view_feat, 64, 141, 'leaky_relu').reshape(-1, 47, 3)
        # full coeff j (1..47) = view_sh[:, j-1]; temporal fold at t = t_center:
        f_dc = f_dc + view_sh[:, 15, :] + view_sh[:, 31, :]
        # eff[k] for k=1..15: view indices k-1, k+15, k+31
        f_rest = (view_sh[:, 0:15, :] + view_sh[:, 16:31, :] + view_sh[:, 32:47, :])  # [N,15,3]

    if f_rest is not None:
        f_rest = clamp_sh_overshoot(f_dc, f_rest, sh_clamp)

    finish_export(out_path, time_min, time_max, fps, prune_threshold, cov2d_scale,
                  xyz, quat, log_scales, opacity_logit, f_dc, f_rest,
                  velocity, t_center, t_sigma, keep_main_cluster=keep_main_cluster,
                  filter_corrupted=filter_corrupted,
                  filter_black_floaters=filter_black_floaters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert OMG4 comp.xz checkpoint to compact .omg4 v2 for supersplat-viewer')
    parser.add_argument('--input', required=True, help='Path to comp.xz (OMG4 output)')
    parser.add_argument('--output', required=True, help='Destination .omg4 file')
    parser.add_argument('--time_min', type=float, default=0.0,
                        help='Training time_duration min in seconds (default: 0.0)')
    parser.add_argument('--time_max', type=float, default=10.0,
                        help='Training time_duration max in seconds (default: 10.0)')
    parser.add_argument('--fps', type=float, default=30.0, help='Advisory fps for UI (default: 30)')
    parser.add_argument('--prune_threshold', type=float, default=1.0 / 1024,
                        help='Drop Gaussians whose peak alpha inside the time range is below this (default: 1/1024; 0 disables)')
    parser.add_argument('--keep_main_cluster', action='store_true',
                        help='Drop splats outside the largest connected cluster (removes isolated '
                             'floater blobs; intended for masked single-subject captures)')
    parser.add_argument('--sh_clamp', type=float, default=1.5,
                        help='Attenuate higher SH bands per splat so total colour excursion stays below '
                             'this (colour units) from every direction; kills firework artifacts on '
                             'under-observed splats (default: 1.5; 0 disables)')
    parser.add_argument('--no_sh', action='store_true',
                        help='Skip baking the 3-band view-dependent SH coefficients (smaller file, flatter shading)')
    parser.add_argument('--cov2d_scale', type=str, default=None,
                        help='"kx,ky": store a screen-space 2D-covariance scale in the header for the '
                             'viewer to apply per view. Reproduces the reference renderer exactly at '
                             'training-like views but smears at oblique angles; for free-viewpoint '
                             'viewing prefer --aniso_boost, which bakes the compensation statically '
                             'along the training-rig axes.')
    parser.add_argument('--aniso_boost', type=str, default=None,
                        help='"kx,ky": anisotropic scale compensation baked along the average training-camera '
                             'x/y axes (requires --cameras_json). For OMG4 N3V checkpoints use "1.6942,1.2707". '
                             'RECOMMENDED for free-viewpoint viewing of FoV-sentinel-trained checkpoints: '
                             'it un-squashes the world-frame anisotropy the bug baked into the model, '
                             'so quality holds up at oblique angles (unlike --cov2d_scale).')
    parser.add_argument('--cameras_json', type=str, default=None,
                        help='cameras.json from the training output dir (provides camera orientations for --aniso_boost)')
    parser.add_argument('--scale_boost', type=float, default=1.0,
                        help='Multiply all splat scales by this factor. The OMG4 repo trains its 4D-rotor '
                             'models with a camera bug: dataset cameras carry a FoV=-1 sentinel, so the '
                             'rasterizer computes its EWA Jacobian from tan(-0.5) instead of the real focal, '
                             'inflating every splat footprint by W/(2*tan(0.5)*fl_x) horizontally and '
                             'H/(2*tan(0.5)*fl_y) vertically during training. Models fit under that inflation '
                             'look thin/streaky in a geometrically correct renderer. For the N3V/dynerf '
                             'checkpoints use sqrt(1.6942*1.2707) = 1.4672. Models trained with a fixed '
                             'camera (or the FTGS/gsplat variant) need the default 1.0.')
    parser.add_argument('--top_k_fraction', type=float, default=1.0,
                        help='Only used with --input pointing at a checkpoint (chkpntNNNN.pth), not comp.xz. '
                             'Keep only this fraction (0.0-1.0) of splats, ranked by peak visibility '
                             '(opacity x temporal-fade weight at each splat\'s own t_center). 1.0 (default) '
                             'keeps every splat -- full fidelity, largest file. Size/quality knob that '
                             'degrades gracefully by dropping the least-visible splats first, as an '
                             'alternative to OMG4\'s SVQ compression pipeline.')
    parser.add_argument('--extra_keep_mask', type=str, default=None,
                        help='Only used with --input pointing at a checkpoint. Path to a .npy boolean array '
                             '(one entry per splat, in the checkpoint\'s original order) precomputed by a '
                             'separate dataset-specific script -- e.g. a real-camera-mask consistency check '
                             '(project each splat into the real training cameras at its own t_center and '
                             'test against the ground-truth subject segmentation masks; splats that fall '
                             'outside the mask in most views that can see them are floaters, not real '
                             'subject geometry). Applied before covariance slicing.')
    parser.add_argument('--no_filter_corrupted', action='store_true',
                        help='comp.xz path only: skip the bad_color/garbage-geometry corruption '
                             'filters. Those were calibrated for the legacy SVQ pipeline\'s '
                             'catastrophic-extrapolation failure (f_dc in the hundreds); on a '
                             'healthy SPM fine-tune an out-of-range bare f_dc is normal SH '
                             'representation, and the filter deletes ~40% of good splats '
                             '(visible as transparency). Same rationale as the checkpoint path, '
                             'which never applies them.')
    parser.add_argument('--filter_black_floaters', action='store_true',
                        help='comp.xz path only: drop the detached near-black floater cloud an SPM '
                             'fine-tune leaves around a masked subject (deeply-negative mean colour, '
                             'or dark + low-alpha / spike-elongated) while keeping legitimate dark '
                             'material. Pair with --no_filter_corrupted for SPM exports.')
    parser.add_argument('--v3', action='store_true',
                        help='Write the SOG-compressed v3 container (webp textures + codebooks, '
                             '~5-7x smaller) instead of the raw-float v2 layout. Higher-order SH '
                             'is vector-quantized rather than dropped; combine with --no_sh to '
                             'omit it entirely.')
    parser.add_argument('--shn_count', type=int, default=65536,
                        help='v3 only: VQ centroid count for higher-order SH (default: 65536)')
    parser.add_argument('--webp_method', type=int, default=4, choices=range(7),
                        help='v3 only: libwebp effort 0-6 (default: 4; 6 is smallest/slowest)')
    parser.add_argument('--segment_duration', type=float, default=0.1,
                        help='v3 only: temporal segment length in seconds for per-segment '
                             'culling (default: 0.1; 0 disables segmentation)')
    args = parser.parse_args()

    if args.v3:
        V3_EXPORT_OPTIONS = {'shn_count': args.shn_count, 'webp_method': args.webp_method,
                             'segment_duration': args.segment_duration}

    aniso = None
    cam_rots = None
    if args.aniso_boost:
        import json as _json
        aniso = tuple(float(v) for v in args.aniso_boost.split(','))
        if not args.cameras_json:
            sys.exit('--aniso_boost requires --cameras_json')
        cam_rots = [c['rotation'] for c in _json.load(open(args.cameras_json))[:64]]

    cov2d = tuple(float(v) for v in args.cov2d_scale.split(',')) if args.cov2d_scale else None

    if args.input.endswith('.pth'):
        if not (0.0 < args.top_k_fraction <= 1.0):
            sys.exit('--top_k_fraction must be in (0.0, 1.0]')
        convert_from_checkpoint(args.input, args.output, args.time_min, args.time_max, args.fps,
                                args.prune_threshold, include_sh=not args.no_sh,
                                scale_boost=args.scale_boost, sh_clamp=args.sh_clamp,
                                keep_main_cluster=args.keep_main_cluster,
                                top_k_fraction=args.top_k_fraction,
                                extra_keep_mask_path=args.extra_keep_mask)
    else:
        convert(args.input, args.output, args.time_min, args.time_max, args.fps,
                args.prune_threshold, include_sh=not args.no_sh, scale_boost=args.scale_boost,
                aniso_boost=aniso, aniso_camera_rotations=cam_rots, cov2d_scale=cov2d,
                sh_clamp=args.sh_clamp, keep_main_cluster=args.keep_main_cluster,
                filter_corrupted=not args.no_filter_corrupted,
                filter_black_floaters=args.filter_black_floaters)
