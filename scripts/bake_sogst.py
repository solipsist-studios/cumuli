#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
bake_sogst.py  -  Bake a trained 4D interchange PLY into the .sogst container
and/or a filtered interchange PLY.

Input is cumuli-trainer's splat_4d.ply: spacetime Gaussians with a position
at t_center, a velocity, a Gaussian temporal envelope (t_sigma), constant
rotation and scale, and SH colour (docs/sogst-format.md section 7). The
bake post-filters and optionally smooths those splats, then writes a .sogst
archive (--output), a filtered interchange PLY (--emit_ply), or both, from
the same arrays so the two cannot disagree. The PLY carries its own clip
scalars (time_min, time_max, fps), so the bake takes none.

Steps, in order:
    SH truncation (--sh_degree), SH chroma clamp (--sh_chroma_clamp), SH
    overshoot clamp (--sh_clamp), top-k by visibility (--top_k_fraction),
    3D smoothing (--smooth_px), temporal smoothing (--smooth_frames),
    lifetime mask-consistency filter (--mask_filter_root), dark-occluder and
    optional black-floater / main-cluster filters, and the peak-alpha prune.

Usage:
    python bake_sogst.py --input train4d_output/splat_4d.ply \
        --output splat_4d.sogst --mask_filter_root dataset_4dgs \
        --sh_chroma_clamp 0.02 --smooth_px 0.5 --smooth_frames 2
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sogst_io import SOGST_FIELDS, report_output

# Set by main() when --output is given: the packer options for the .sogst
# container, {'shn_count': int, 'webp_method': int, 'segment_duration': float}.
# None means this bake writes only the interchange PLY.
SOGST_EXPORT_OPTIONS = None

# Set by main() when --emit_ply is given: the destination .ply path.
# finish_export() writes the 4D interchange PLY (docs/sogst-format.md
# section 7) from the same post-filter field arrays that feed the container
# writer, so the two cannot disagree.
PLY_EXPORT_PATH = None
PLY_EXPORT_SIDECAR = False

SH_Y_MAX = np.array([0.489, 0.489, 0.489,
                     0.546, 0.546, 0.630, 0.546, 0.546,
                     0.590, 0.590, 0.457, 0.746, 0.457, 0.590, 0.590], dtype=np.float32)


def clamp_sh_chroma(f_rest, limit):
    """Bound the COLOURED part of each splat's view-dependent term, leaving
    its brightness part alone.

    The higher SH bands split into a luminance component (the per-coefficient
    mean over R, G, B) and a chroma component (the rest). With ~10 cameras,
    a splat's colour is pinned down in ~10 directions; between them the
    fitted lobes extrapolate into hues the subject does not have, which
    shows as coloured glints wherever the splat is opaque, worst on dark
    material. Real view-dependence on skin and cloth is mostly brightness
    (sheen, specular), so bounding only the chroma removes the glints and
    keeps the shading. Per splat, the worst-case chroma excursion
    sum(|chroma_lm| * max|Y_lm|) is scaled down to ``limit`` (colour units).

    Heidi (cumuli-trainer, held-out cam05; close-up orbit, glint pixels per
    10k subject pixels): no limit 0.94 glints, 36.08 dB, LPIPS 0.0096;
    limit 0.02 0.14 glints (SH-off floor 0.12), 36.07 dB, LPIPS 0.0096.
    The absolute --sh_clamp at 0.5 reached 0.21 glints only by costing
    0.38 dB. limit <= 0 disables."""
    if f_rest is None or limit <= 0:
        return f_rest
    y_max = SH_Y_MAX[:f_rest.shape[1]]
    lum = f_rest.mean(axis=2, keepdims=True)                           # [N,c,1]
    chroma = f_rest - lum
    bound = (np.abs(chroma) * y_max[None, :, None]).sum(axis=1).max(axis=1)  # [N]
    scale = np.minimum(1.0, limit / np.maximum(bound, 1e-9))
    print(f'  SH chroma clamp: limited {int((scale < 1.0).sum()):,} / {len(scale):,} splats '
          f'(max chroma excursion was {bound.max():.2f})')
    return (lum + chroma * scale[:, None, None]).astype(f_rest.dtype)


def clamp_sh_overshoot(f_dc, f_rest, sh_clamp):
    """Scale down each splat's higher SH bands so its colour stays bounded
    from every view direction.

    Splats observed from a narrow cone of training views can have SH
    coefficients that explode when evaluated from novel directions
    (saturated 'firework' spikes). A conservative bound on the band
    contribution is sum(|c_lm| * max|Y_lm|). Per splat, if
    base + bound exceeds sh_clamp (in colour units), the f_rest coefficients
    are scaled so it does not. sh_clamp <= 0 disables.
    """
    if f_rest is None or sh_clamp <= 0:
        return f_rest
    # a degree-1 or degree-2 block carries 3 or 8 coefficients per channel
    y_max = SH_Y_MAX[:f_rest.shape[1]]
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


def truncate_sh(f_rest, sh_degree):
    """Keep the first sh_degree bands of an [N, coeffs, 3] SH block.

    Bands are nested ((d+1)^2 - 1 coefficients through degree d), so
    dropping the trailing coefficients is exactly a lower-degree model.
    sh_degree None keeps the model's own degree. 0 returns None: DC only.
    Asking for more bands than the model has is refused, since padding with
    zeros would only add bytes that render as nothing."""
    if f_rest is None or sh_degree is None:
        return f_rest
    model_degree = int(round(math.sqrt(f_rest.shape[1] + 1))) - 1
    if sh_degree > model_degree:
        raise ValueError(f'--sh_degree {sh_degree} exceeds the model\'s own SH degree '
                         f'{model_degree}; there are no higher bands to bake')
    if sh_degree == 0:
        print(f'  SH degree: baking DC only (model has degree {model_degree})')
        return None
    if sh_degree < model_degree:
        print(f'  SH degree: baking degree {sh_degree} of the model\'s {model_degree}')
    return f_rest[:, :(sh_degree + 1) ** 2 - 1, :]


def main_cluster_mask(xyz, opacity_logit, cell_frac=0.03):
    """Keep only the largest connected splat cluster (voxel 26-connectivity).

    Removes isolated floater clusters (mask noise, stray props) that
    reconstruct away from the subject. Cell size is cell_frac of the bbox
    diagonal. Clusters are ranked by opacity mass.
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


def dark_occluder_mask(f_dc, opacity_logit, log_scales, brightness_thresh=0.05,
                       opacity_thresh=0.7, size_percentile=75):
    """Flag oversized, near-black, high-opacity splats.

    A known Gaussian-splatting failure mode: the optimizer grows dark
    "shadow-filling" splats to explain away self-occlusion/shadowing that
    is only visible from a subset of training cameras. Fast-motion human
    captures, with limbs crossing the body, have a lot of that. Those splats
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
    negative in colour (mean below black_thresh, where no SH band can bring
    it back), or dark (below dim_thresh) while ALSO low-alpha (murk, not
    surface) or spike-elongated. Calibrated on a subject-capture SPM
    export, where the drop set turned out to be ~95% identical to
    bad_color_mask's. The old filter had the right suspects. The washout it
    caused came from the sh_clamp interaction, not the deletions. This
    variant keeps the ~5%
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


def training_sample_rate(dataset_root, xyz):
    """[N] the finest sampling of each splat by any training camera, in
    pixels per scene unit: max over cameras that see it in frame of
    focal / depth. Splats no camera frames get the median of the rest."""
    import json
    root = os.path.expanduser(dataset_root)
    with open(os.path.join(root, 'transforms_train.json')) as fh:
        t = json.load(fh)
    seen = {}
    for fr in t['frames']:   # a static rig: one pose per camera label
        lab = fr.get('camera_label') or fr['file_path'].split('/')[-2]
        seen.setdefault(lab, fr)
    rate = np.zeros(len(xyz), np.float64)
    for fr in seen.values():
        c2w = np.asarray(fr['transform_matrix'], np.float64).copy()
        c2w[:3, 1:3] *= -1.0                       # OpenGL -> OpenCV
        w2c = np.linalg.inv(c2w)
        cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        fx = float(fr.get('fl_x', t.get('fl_x'))); fy = float(fr.get('fl_y', t.get('fl_y')))
        cx = float(fr.get('cx', t.get('cx'))); cy = float(fr.get('cy', t.get('cy')))
        w = float(fr.get('w', t.get('w'))); h = float(fr.get('h', t.get('h')))
        zz = np.maximum(z, 1e-6)
        u = fx * cam[:, 0] / zz + cx
        v = fy * cam[:, 1] / zz + cy
        ok = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        rate = np.where(ok, np.maximum(rate, fx / zz), rate)
    if (rate > 0).any():
        rate[rate == 0] = np.median(rate[rate > 0])
    else:
        rate[:] = 1.0
    return rate


def smooth_3d(dataset_root, xyz, log_scales, opacity_logit, smooth_px):
    """Low-pass each splat by an isotropic Gaussian of ``smooth_px`` training
    pixels: the 3D smoothing filter of Mip-Splatting (Yu et al. 2024), as a
    bake-time knob rather than a training term.

    Adding sigma^2 I to a covariance keeps its axes, so each scale becomes
    sqrt(s^2 + sigma^2), with sigma = smooth_px / (finest training sampling
    rate of that splat). Peak opacity is scaled by s1 s2 s3 / s1' s2' s3'
    so each splat keeps its integrated density. Thin, noisy splats widen and
    fade the most; large smooth ones barely change. 0 disables.

    It trades sharpness for calm: on a 10-camera rig the fine detail between
    cameras is partly noise, and a softer bake reads as cleaner."""
    if smooth_px <= 0:
        return log_scales, opacity_logit
    rate = training_sample_rate(dataset_root, xyz)
    sigma2 = (smooth_px / rate) ** 2                                    # [N]
    s2 = np.exp(2.0 * log_scales.astype(np.float64))
    new_s2 = s2 + sigma2[:, None]
    new_log_scales = 0.5 * np.log(new_s2)
    ratio = np.exp((log_scales.astype(np.float64) - new_log_scales).sum(axis=1))  # det ratio^(1/2)
    alpha = 1.0 / (1.0 + np.exp(-opacity_logit.astype(np.float64)))
    new_alpha = np.clip(alpha * ratio, 1e-7, 1 - 1e-7)
    print(f'  3D smoothing: {smooth_px:g} training px; median filter width '
          f'{np.median(np.sqrt(sigma2)):.4g} units, median opacity x{np.median(ratio):.3f}')
    return (new_log_scales.astype(np.float32),
            np.log(new_alpha / (1.0 - new_alpha)).astype(np.float32))


def smooth_time(t_sigma, opacity_logit, smooth_frames, fps):
    """Low-pass each splat in time by a Gaussian ``smooth_frames`` frames
    wide: t_sigma becomes sqrt(t_sigma^2 + s^2), and peak opacity scales by
    t_sigma / t_sigma' so each splat keeps its integral over time. Short-
    lived splats fade in and out more gently, which reads as steadier
    playback; long-lived ones barely change. 0 disables.

    Heidi (cumuli-trainer, front-view head frame-to-frame change / held-out
    cam05 LPIPS): 0 -> 6.08 / 0.0097, 2 frames -> 5.74 / 0.0101,
    4 frames -> 5.48 / 0.0108. Spatial --smooth_px 1 -> 5.05 / 0.0120.
    Long smoothing lets a splat drift further along its velocity, so
    expect ghosting well before the image softens."""
    if smooth_frames <= 0:
        return t_sigma, opacity_logit
    s = smooth_frames / float(fps)
    ts = t_sigma.astype(np.float64)
    new_ts = np.sqrt(ts ** 2 + s ** 2)
    alpha = 1.0 / (1.0 + np.exp(-opacity_logit.astype(np.float64)))
    new_alpha = np.clip(alpha * ts / new_ts, 1e-7, 1 - 1e-7)
    print(f'  temporal smoothing: {smooth_frames:g} frames; median lifetime '
          f'{np.median(ts):.3f}s -> {np.median(new_ts):.3f}s')
    return new_ts.astype(np.float32), np.log(new_alpha / (1.0 - new_alpha)).astype(np.float32)


def mask_consistency_keep(dataset_root, xyz, velocity, t_center, t_sigma, opacity_logit,
                          outside_frac=0.5, downscale=4, time_stride=4, k_sigma=2.0,
                          alpha_thresh=0.02):
    """Keep-mask from multi-view silhouette consistency, evaluated across each
    splat's visible lifetime rather than only at its temporal centre.

    Projects every Gaussian into every training camera at several times where
    it actually contributes, and drops those landing OUTSIDE the subject mask
    in more than `outside_frac` of the (time, camera) tests that resolve them.
    The masks come from the training RGBA alpha, so this is a direct geometric
    test against ground truth rather than a density heuristic.

    Note on scope: this removes splats that leave the silhouette. It does
    NOT remove splats sitting at wrong depths INSIDE the silhouette,
    because depth is unobservable from silhouettes alone (measured:
    dropping the flagged set leaves view-aligned streak artifacts
    untouched).

    Do not reach for anisotropy regularisation as the fix for those,
    either. It was built (OMG4 lambda_aniso / aniso_max) and measured on
    one capture: aspect p50 25.8 -> 5.7, p95 4376 -> 10.6, rounder than
    scenes that look correct, yet it costs 1.7 dB and the streaks survive.
    The cause is view sparsity (10-11 frontal cameras), so the remaining
    levers are more views or depth priors during training, not any
    post-hoc geometry edit. Forcing THIS filter as a fix costs 2.1 dB
    (48.60 -> 46.51) for the same reason: with 11 cameras the 2D test
    misclassifies legitimate boundary and partially-occluded geometry.
    Keep it for what it is: removal of genuine escaping junk.
    """
    import json
    from PIL import Image

    root = os.path.expanduser(dataset_root)
    t = json.load(open(os.path.join(root, 'transforms_train.json')))
    cams = {}
    for fr in t['frames']:
        lab = fr.get('camera_label') or fr['file_path'].split('/')[-2]
        cams.setdefault(lab, {'meta': fr, 'frames': {}})['frames'][round(fr['time'], 4)] = fr['file_path']
    labels = sorted(cams)
    times = sorted(cams[labels[0]]['frames'])[::max(1, time_stride)]

    masks = {}
    for lab in labels:
        for tt in times:
            a = Image.open(os.path.join(root, cams[lab]['frames'][tt] + '.png')).split()[3]
            if downscale > 1:
                a = a.resize((a.width // downscale, a.height // downscale), Image.BILINEAR)
            masks[(lab, tt)] = np.asarray(a) > 100

    n = len(xyz)
    alpha = 1.0 / (1.0 + np.exp(-opacity_logit))
    ts = np.maximum(np.abs(t_sigma), 1e-9)
    inside = np.zeros(n, np.int32)
    tested = np.zeros(n, np.int32)
    for tt in times:
        w = alpha * np.exp(-0.5 * ((tt - t_center) / ts) ** 2)
        live = np.where(w > alpha_thresh)[0]
        if not len(live):
            continue
        pts = xyz[live] + velocity[live] * (tt - t_center[live])[:, None]
        for lab in labels:
            m = cams[lab]['meta']
            mask = masks[(lab, tt)]
            c2w = np.asarray(m['transform_matrix'], np.float64).copy()
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            cam = (w2c[:3, :3] @ pts.T + w2c[:3, 3:4]).T
            z = cam[:, 2]
            front = z > 0.05
            zs = np.where(front, z, 1.0)
            u = (m['fl_x'] / downscale) * cam[:, 0] / zs + (m['cx'] / downscale)
            v = (m['fl_y'] / downscale) * cam[:, 1] / zs + (m['cy'] / downscale)
            h_, w_ = mask.shape
            ok = front & (u >= 0) & (u < w_) & (v >= 0) & (v < h_)
            ui = np.clip(u, 0, w_ - 1).astype(np.int32)
            vi = np.clip(v, 0, h_ - 1).astype(np.int32)
            idx = live[ok]
            inside[idx] += mask[vi[ok], ui[ok]]
            tested[idx] += 1
    resolved = tested > 0
    frac_out = np.zeros(n)
    frac_out[resolved] = 1.0 - inside[resolved] / tested[resolved]
    keep = ~(resolved & (frac_out > outside_frac))
    print(f'  Mask-consistency filter: dropping {int((~keep).sum()):,} / {n:,} splats '
          f'outside the subject silhouette in >{outside_frac:.0%} of '
          f'{len(times)} times x {len(labels)} cameras '
          f'({int((~resolved).sum()):,} resolved by no camera, kept)')
    return keep


def finish_export(out_path, time_min, time_max, fps, prune_threshold, cov2d_scale,
                  xyz, quat, log_scales, opacity_logit, f_dc, f_rest,
                  velocity, t_center, t_sigma, keep_main_cluster=False,
                  filter_dark_occluders=False, filter_black_floaters=False,
                  extra_keep_mask=None, accel=None):
    """Shared tail: filters, prune, diagnostics, write the outputs."""
    N = xyz.shape[0]
    dist = np.abs(t_center - np.clip(t_center, time_min, time_max))
    peak_weight = np.exp(-0.5 * (dist / np.maximum(t_sigma, 1e-9)) ** 2)
    peak_alpha = (1.0 / (1.0 + np.exp(-opacity_logit))) * peak_weight
    keep = peak_alpha >= prune_threshold
    if filter_dark_occluders:
        keep &= dark_occluder_mask(f_dc, opacity_logit, log_scales)
    if filter_black_floaters:
        keep &= black_floater_mask(f_dc, opacity_logit, log_scales)
    if extra_keep_mask is not None:
        n_before = int(keep.sum())
        keep &= extra_keep_mask
        print(f'  External keep-mask (for example real-mask consistency): dropping '
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
    n_rest = 0 if f_rest is None else f_rest.shape[1]
    if f_rest is not None:
        # PLY channel-major order: f_rest_{ch*n + k} = eff_rest[k][channel ch],
        # with n = 3, 8 or 15 coefficients for SH degree 1, 2 or 3
        for ch in range(3):
            for k in range(n_rest):
                arrays.append(f_rest[:, k, ch])
    if accel is not None:
        # degree-2 motion: the raw dt^2 coefficient (spec section 4.8)
        for c in range(3):
            arrays.append(accel[:, c])
    arrays = [np.ascontiguousarray(a[keep], dtype=np.float32) for a in arrays]

    for t in np.linspace(time_min, time_max, 5):
        w = np.exp(-0.5 * ((t_center[keep] - t) / np.maximum(t_sigma[keep], 1e-9)) ** 2)
        alpha = (1.0 / (1.0 + np.exp(-opacity_logit[keep]))) * w
        print(f"    t={t:6.2f}s  active(alpha>1/255): {(alpha > 1 / 255).mean() * 100:5.1f}%")
    vmag = np.linalg.norm(velocity[keep], axis=1)
    print(f"    |velocity| p50/p95/p99: {np.percentile(vmag, [50, 95, 99]).round(4)}")
    if accel is not None:
        amag = np.linalg.norm(accel[keep], axis=1)
        print(f"    |accel|    p50/p95/p99: {np.percentile(amag, [50, 95, 99]).round(4)}")
    print(f"    t_sigma    p50/p95    : {np.percentile(t_sigma[keep], [50, 95]).round(3)}")

    # The name -> array view of the pruned data.  Both the .sogst container
    # writer and the interchange-PLY writer consume this same dict, which
    # is the point: the seam is one place, so the two outputs cannot
    # describe different splats.
    fields = {name: arrays[i] for i, name in enumerate(SOGST_FIELDS)}
    cursor = len(SOGST_FIELDS)
    if f_rest is not None:
        fields['f_rest'] = np.stack(arrays[cursor:cursor + 3 * n_rest], axis=1)
        cursor += 3 * n_rest
    if accel is not None:
        for i, name in enumerate(('ax', 'ay', 'az')):
            fields[name] = arrays[cursor + i]

    if PLY_EXPORT_PATH is not None:
        from sogst_ply import write_sogst_ply, write_sogst_sidecar
        n_ply = write_sogst_ply(PLY_EXPORT_PATH, fields, time_min, time_max, fps,
                                cov2d_scale=cov2d_scale,
                                generator='volumetric-capture-pipeline bake_sogst')
        print(f'  Interchange PLY: {n_ply:,} splats -> {PLY_EXPORT_PATH}')
        if PLY_EXPORT_SIDECAR:
            print(f'    sidecar: {write_sogst_sidecar(PLY_EXPORT_PATH, time_min, time_max, fps, cov2d_scale=cov2d_scale, motion_degree=2 if accel is not None else 1)}')

    if SOGST_EXPORT_OPTIONS is not None:
        from sogst_pack import pack_sogst
        pack_sogst(out_path, fields, time_min, time_max, fps,
                   cov2d_scale=cov2d_scale, **SOGST_EXPORT_OPTIONS)
        report_output(out_path)


def convert_from_ply(ply_path, out_path, time_min, time_max, fps, prune_threshold,
                     sh_degree=None, sh_clamp=1.5, keep_main_cluster=False,
                     sh_chroma_clamp=0.0, smooth_px=0.0, smooth_frames=0.0, top_k_fraction=1.0, filter_black_floaters=False,
                     mask_filter_root=None, mask_filter_outside_frac=0.5):
    """Bake a 4D interchange PLY: SH truncation and clamps, top-k, smoothing,
    the mask-consistency filter, then finish_export's filters and writers."""
    from sogst_ply import read_sogst_ply

    print(f"Loading interchange PLY {ply_path} …")
    _header, fields = read_sogst_ply(ply_path)
    col = lambda *names: np.stack([fields[n] for n in names], axis=1).astype(np.float32)
    xyz = col('x', 'y', 'z')
    quat = col('rot_0', 'rot_1', 'rot_2', 'rot_3')
    log_scales = col('scale_0', 'scale_1', 'scale_2')
    opacity_logit = fields['opacity'].astype(np.float32)
    f_dc = col('f_dc_0', 'f_dc_1', 'f_dc_2')
    velocity = col('vx', 'vy', 'vz')
    t_center = fields['t_center'].astype(np.float32)
    t_sigma = fields['t_sigma'].astype(np.float32)
    accel = col('ax', 'ay', 'az') if 'ax' in fields else None
    N = xyz.shape[0]
    print(f"  {N:,} splats")

    f_rest = None
    if 'f_rest' in fields:
        # channel-major [N, 3c] -> [N, c, 3], the layout finish_export writes back
        n_coeffs = fields['f_rest'].shape[1] // 3
        f_rest = fields['f_rest'].reshape(N, 3, n_coeffs).transpose(0, 2, 1).astype(np.float32)
    f_rest = truncate_sh(f_rest, sh_degree)
    if f_rest is not None:
        f_rest = clamp_sh_overshoot(f_dc, clamp_sh_chroma(f_rest, sh_chroma_clamp), sh_clamp)

    if top_k_fraction < 1.0:
        dist = np.abs(t_center - np.clip(t_center, time_min, time_max))
        peak_weight = np.exp(-0.5 * (dist / np.maximum(t_sigma, 1e-9)) ** 2)
        peak_alpha = (1.0 / (1.0 + np.exp(-opacity_logit))) * peak_weight
        k = max(1, min(N, int(round(top_k_fraction * N))))
        thresh = np.partition(peak_alpha, N - k)[N - k]
        top = peak_alpha >= thresh
        print(f"  top_k_fraction={top_k_fraction}: keeping {int(top.sum()):,} / {N:,} "
              f"highest-visibility splats")
        xyz, quat, log_scales, opacity_logit = xyz[top], quat[top], log_scales[top], opacity_logit[top]
        f_dc, velocity, t_center, t_sigma = f_dc[top], velocity[top], t_center[top], t_sigma[top]
        if f_rest is not None:
            f_rest = f_rest[top]
        if accel is not None:
            accel = accel[top]

    if smooth_px > 0:
        log_scales, opacity_logit = smooth_3d(mask_filter_root, xyz, log_scales, opacity_logit, smooth_px)
    if smooth_frames > 0:
        t_sigma, opacity_logit = smooth_time(t_sigma, opacity_logit, smooth_frames, fps)

    mask_keep = None
    if mask_filter_root:
        mask_keep = mask_consistency_keep(mask_filter_root, xyz, velocity, t_center,
                                          t_sigma, opacity_logit,
                                          outside_frac=mask_filter_outside_frac)

    finish_export(out_path, time_min, time_max, fps, prune_threshold, None,
                  xyz, quat, log_scales, opacity_logit, f_dc, f_rest,
                  velocity, t_center, t_sigma, keep_main_cluster=keep_main_cluster,
                  filter_dark_occluders=True,
                  filter_black_floaters=filter_black_floaters,
                  extra_keep_mask=mask_keep, accel=accel)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bake a 4D interchange PLY (cumuli-trainer output) into the .sogst '
                    'container and/or a filtered interchange PLY')
    parser.add_argument('--input', required=True,
                        help='4D interchange .ply (docs/sogst-format.md section 7)')
    parser.add_argument('--output', default=None,
                        help='Destination .sogst archive. Optional when --emit_ply is given.')

    parser.add_argument('--prune_threshold', type=float, default=1.0 / 1024,
                        help='Drop Gaussians whose peak alpha inside the time range is below this (default: 1/1024, 0 disables)')
    parser.add_argument('--keep_main_cluster', action='store_true',
                        help='Drop splats outside the largest connected cluster (removes isolated '
                             'floater blobs, intended for masked single-subject captures)')
    parser.add_argument('--smooth_frames', type=float, default=0.0,
                        help='widen every splat\'s lifetime by a Gaussian '
                             'this many frames wide, conserving its integral over time. Gentler '
                             'fade-in/out, steadier playback; too much ghosts. Default 0: off.')
    parser.add_argument('--smooth_px', type=float, default=0.0,
                        help='blur every splat by an isotropic Gaussian '
                             'this many training-camera pixels wide (Mip-Splatting 3D filter), '
                             'conserving its density. Trades sharpness for a calmer, less noisy '
                             'look. Needs --mask_filter_root for the training cameras. Default 0: off.')
    parser.add_argument('--sh_chroma_clamp', type=float, default=0.0,
                        help='bound the coloured part of each splat\'s '
                             'view-dependent SH term to this many colour units, keeping its '
                             'brightness part. Removes coloured glints from sparse-camera fits '
                             '(see clamp_sh_chroma). The pipeline passes 0.02 for cumuli-trainer '
                             'output. Default 0: off.')
    parser.add_argument('--sh_clamp', type=float, default=1.5,
                        help='Attenuate higher SH bands per splat so total colour excursion stays below '
                             'this (colour units) from every direction. Kills firework artifacts on '
                             'under-observed splats (default: 1.5, 0 disables)')
    parser.add_argument('--sh_degree', type=int, default=None, choices=range(4),
                        help='Spherical-harmonic degree to bake, 0-3: 0 is DC only (view-independent '
                             'colour), 1-3 keep that many view-dependent bands (3, 8 or 15 '
                             'coefficients per channel). Lower degrees give a smaller file and flatter '
                             'shading. Default: the model\'s own degree. Must not exceed it.')
    parser.add_argument('--no_sh', action='store_true',
                        help='Same as --sh_degree 0')
    parser.add_argument('--top_k_fraction', type=float, default=1.0,
                        help='Keep only this fraction (0.0-1.0) of splats, ranked by peak visibility '
                             '(opacity x temporal-fade weight at each splat\'s own t_center). 1.0 (default) '
                             'keeps every splat: full fidelity, largest file. Size/quality knob that '
                             'degrades gracefully by dropping the least-visible splats first.')
    parser.add_argument('--mask_filter_root', type=str, default=None,
                        help='4DGS dataset root (with transforms_train.json '
                             'and RGBA realcams frames). Drops Gaussians projecting OUTSIDE the '
                             'subject silhouette in most (time, camera) tests across their visible '
                             'lifetime, not just at t_center. Removes splats that drift off the '
                             'body. Does NOT fix wrong-depth splats inside the silhouette.')
    parser.add_argument('--mask_filter_outside_frac', type=float, default=0.5,
                        help='Drop a Gaussian when it lands outside the mask in more than this '
                             'fraction of the tests that resolve it (default 0.5)')
    parser.add_argument('--filter_black_floaters', action='store_true',
                        help='Drop a detached near-black floater cloud around a masked subject '
                             '(deeply-negative mean colour, or dark + low-alpha / spike-elongated) '
                             'while keeping legitimate dark material. Off by default.')
    parser.add_argument('--shn_count', type=int, default=65536,
                        help='--output only: VQ centroid count for higher-order SH (default: 65536)')
    parser.add_argument('--webp_method', type=int, default=4, choices=range(7),
                        help='--output only: libwebp effort 0-6 (default: 4, and 6 is smallest/slowest)')
    parser.add_argument('--segment_duration', type=float, default=0.1,
                        help='--output only: temporal segment length in seconds for per-segment '
                             'culling (default: 0.1, 0 disables segmentation)')
    parser.add_argument('--emit_ply', type=str, default=None,
                        help='Write the 4D interchange PLY to this path: the per-splat '
                             'spacetime data an external encoder packs '
                             '(docs/sogst-format.md section 7). Written from the same '
                             'post-filter arrays as --output, so the two agree by '
                             'construction. May be given with or instead of --output.')
    parser.add_argument('--emit_ply_sidecar', action='store_true',
                        help='With --emit_ply, also write the optional <name>.sogst.json '
                             'sidecar. The clip scalars are in the PLY comments either '
                             'way. This is for toolchains that strip them.')
    args = parser.parse_args()

    if not args.input.endswith('.ply'):
        sys.exit('--input must be a 4D interchange .ply (cumuli-trainer splat_4d.ply)')
    if not args.output and not args.emit_ply:
        sys.exit('nothing to write: pass --output (a .sogst archive), --emit_ply '
                 '(the interchange PLY), or both')
    if args.output:
        SOGST_EXPORT_OPTIONS = {'shn_count': args.shn_count, 'webp_method': args.webp_method,
                                'segment_duration': args.segment_duration}
    if args.no_sh:
        if args.sh_degree not in (None, 0):
            sys.exit(f'--no_sh contradicts --sh_degree {args.sh_degree}')
        args.sh_degree = 0
    if args.emit_ply_sidecar and not args.emit_ply:
        sys.exit('--emit_ply_sidecar requires --emit_ply')
    PLY_EXPORT_PATH = args.emit_ply
    PLY_EXPORT_SIDECAR = args.emit_ply_sidecar
    if args.smooth_px > 0 and not args.mask_filter_root:
        sys.exit('--smooth_px needs --mask_filter_root (the dataset with the training cameras)')
    if not (0.0 < args.top_k_fraction <= 1.0):
        sys.exit('--top_k_fraction must be in (0.0, 1.0]')

    from sogst_ply import read_sogst_ply
    header, _ = read_sogst_ply(args.input)
    convert_from_ply(args.input, args.output, header['time_min'], header['time_max'],
                     header['fps'], args.prune_threshold, sh_degree=args.sh_degree,
                     sh_clamp=args.sh_clamp, keep_main_cluster=args.keep_main_cluster,
                     sh_chroma_clamp=args.sh_chroma_clamp, smooth_px=args.smooth_px,
                     smooth_frames=args.smooth_frames, top_k_fraction=args.top_k_fraction,
                     filter_black_floaters=args.filter_black_floaters,
                     mask_filter_root=args.mask_filter_root,
                     mask_filter_outside_frac=args.mask_filter_outside_frac)
