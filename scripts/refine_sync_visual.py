#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 AND MIT
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
#
# The estimator is adapted from Open4DGS-Core prep/sync/visual-epipolar-sync.py
# (https://github.com/ASU-NEM-Labs/open4dgs-core), used under the MIT License:
#
#   Copyright (c) 2026 Open4DGS contributors
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
"""refine_sync_visual.py - each camera's sub-frame capture offset, measured from the pictures.

Audio sync gives every camera a whole-frame offset and a residual under a frame. This measures that residual
visually, for footage whose cameras are posed (Camera poses) but not genlocked:

  1. RoMa matches a camera pair at one frame (only where either camera's mask says subject, when masks exist).
  2. Each matched point is tracked a few frames each way in its own camera (pyramidal Lucas-Kanade, with a
     forward/backward check), giving its image velocity and acceleration in pixels per frame.
  3. A point seen by A at time t_A and by B at t_B lies off its epipolar line by however far it moved in between.
     The signed epipolar (Sampson) residual r0 and how much it changes when the two points are slid half a
     frame along their tracks in opposite directions (de) give the pair's phase for that point:
         delta = -r0 / de      frames, = capture(A) - capture(B)
  4. Phases are combined per image cell (a circular weighted median), and cells with equal weight, so a limb
     with thousands of matches does not outvote the rest; a bootstrap over cells gives the uncertainty.
  5. Each pair's phases over the sampled frames are combined, and a weighted least-squares solve over the
     camera graph gives one offset per camera, the reference camera held at 0.

The phase is only known modulo one frame. With --prior_json (the audio residuals, same convention), every
phase is unwrapped around the prior's prediction, so the result refines the audio solve within half a frame
instead of jumping a whole frame. Static background carries no timing signal and is ignored.

Output (--out): {"convention", "fps", "reference", "verdict", "cameras": {label: {"offset_s", "sigma_s",
"pairs", "status"}}, "pairs": [...], "residual_rms_ms", ...}. PLUS convention: capture = frame time + offset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONVENTION = "capture_time = frame_time + offset_s (seconds; PLUS: a positive offset means the camera fired later)"


# ----------------------------------------------------------------------------- cameras
@dataclass
class Camera:
    label: str
    K: np.ndarray  # 3x3, undistorted pinhole
    R: np.ndarray  # world -> camera, OpenCV
    t: np.ndarray
    width: int
    height: int

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def axis(self) -> np.ndarray:
        return self.R[2]  # optical axis in world coordinates

    def normalize(self, uv: np.ndarray) -> np.ndarray:
        """Pixels -> normalized image coordinates (the images are undistorted pinhole)."""
        return np.stack([(uv[:, 0] - self.K[0, 2]) / self.K[0, 0], (uv[:, 1] - self.K[1, 2]) / self.K[1, 1]], axis=1)


def load_cameras(transforms_path: Path) -> dict[str, Camera]:
    """Cameras of a Camera poses transforms.json (OpenGL c2w, per-frame pinhole intrinsics, camera_label)."""
    data = json.loads(Path(transforms_path).read_text())
    cams = {}
    for f in data["frames"]:
        for k in ("k1", "k2", "p1", "p2", "k3", "k4"):
            if abs(float(f.get(k, 0.0) or 0.0)) > 1e-9:
                raise SystemExit(f"{transforms_path}: camera {f.get('camera_label')} has lens distortion; "
                                 "this works on undistorted images")
        c2w = np.array(f["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1.0  # OpenGL -> OpenCV
        w2c = np.linalg.inv(c2w)
        label = str(f.get("camera_label") or Path(f["file_path"]).stem)
        K = np.array([[f["fl_x"], 0, f["cx"]], [0, f["fl_y"], f["cy"]], [0, 0, 1.0]], dtype=np.float64)
        cams[label] = Camera(label, K, w2c[:3, :3], w2c[:3, 3], int(f["w"]), int(f["h"]))
    return cams


def essential(a: Camera, b: Camera) -> np.ndarray:
    """E with x_b^T E x_a = 0 for normalized coordinates."""
    R = b.R @ a.R.T
    t = b.t - R @ a.t
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return tx @ R


def choose_pairs(cams: dict[str, Camera], per_camera: int, max_angle_deg: float) -> list[tuple[str, str]]:
    """Each camera with its `per_camera` closest neighbours by viewing direction (RoMa needs shared content)."""
    labels = sorted(cams)
    pairs = set()
    for a in labels:
        scored = []
        for b in labels:
            if b == a:
                continue
            cosang = float(np.clip(cams[a].axis @ cams[b].axis, -1, 1))
            angle = math.degrees(math.acos(cosang))
            if angle <= max_angle_deg:
                scored.append((angle, b))
        for _, b in sorted(scored)[:per_camera]:
            pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


# ----------------------------------------------------------------------------- statistics
def wrap(values, period: float = 1.0):
    return (np.asarray(values, dtype=np.float64) + 0.5 * period) % period - 0.5 * period


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def circular_weighted_median(values: np.ndarray, weights: np.ndarray, period: float = 1.0) -> float:
    """Weighted median on a circle: unwrap around the circular mean first, then a plain weighted median."""
    ang = 2 * math.pi * np.asarray(values, dtype=np.float64) / period
    centre = math.atan2(float(np.sum(weights * np.sin(ang))), float(np.sum(weights * np.cos(ang)))) * period / (2 * math.pi)
    return float(wrap(weighted_median(wrap(values - centre, period), weights) + centre, period))


def mad(values: np.ndarray, centre: float | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    centre = float(np.median(values)) if centre is None else centre
    return float(np.median(np.abs(values - centre)))


# ----------------------------------------------------------------------------- one pair at one frame
def signed_sampson_px(a: Camera, b: Camera, E: np.ndarray, uv_a: np.ndarray, uv_b: np.ndarray) -> np.ndarray:
    xa = np.concatenate([a.normalize(uv_a), np.ones((len(uv_a), 1))], axis=1)
    xb = np.concatenate([b.normalize(uv_b), np.ones((len(uv_b), 1))], axis=1)
    Exa, Etxb = xa @ E.T, xb @ E
    e = np.sum(xb * Exa, axis=1)
    denom = np.sqrt(Exa[:, 0] ** 2 + Exa[:, 1] ** 2 + Etxb[:, 0] ** 2 + Etxb[:, 1] ** 2 + 1e-12)
    f = 0.25 * (a.K[0, 0] + a.K[1, 1] + b.K[0, 0] + b.K[1, 1])
    return e / denom * f


def pair_phase(a: Camera, b: Camera, uv_a, uv_b, flow_a, flow_b, accel_a, accel_b, cert, *, size_b,
               prior_frames: float = 0.0, min_epipolar_motion_px: float = 0.5, grid: int = 6, bootstrap: int = 64,
               seed: int = 0) -> dict | None:
    """The phase capture(A) - capture(B), in frames, of one pair at one frame, or None without enough signal.

    flow_* and accel_* are each point's image velocity (px/frame) and acceleration (px/frame^2) in its own camera.
    Phases are unwrapped around `prior_frames`."""
    E = essential(a, b)

    def slide(uv, v, acc, s):
        return uv + v * s + 0.5 * acc * s * s

    r0 = signed_sampson_px(a, b, E, uv_a, uv_b)
    # phase +-0.5: A shown a quarter frame earlier / later, B the other way; one frame apart in total
    r_plus = signed_sampson_px(a, b, E, slide(uv_a, flow_a, accel_a, -0.25), slide(uv_b, flow_b, accel_b, +0.25))
    r_minus = signed_sampson_px(a, b, E, slide(uv_a, flow_a, accel_a, +0.25), slide(uv_b, flow_b, accel_b, -0.25))
    de = r_plus - r_minus
    ok = np.isfinite(r0) & np.isfinite(de) & (np.abs(de) >= min_epipolar_motion_px)
    if ok.sum() < 15:
        return None
    delta = -r0[ok] / de[ok]
    delta = wrap(delta - prior_frames) + prior_frames
    near = np.abs(delta - prior_frames) <= 0.5
    weights = np.maximum(np.asarray(cert, dtype=np.float64)[ok] * np.clip(np.abs(de[ok]), 0, 20), 1e-6)
    w_b, h_b = size_b
    uvb = uv_b[ok]
    gx = np.clip((uvb[:, 0] / w_b * grid).astype(int), 0, grid - 1)
    gy = np.clip((uvb[:, 1] / h_b * grid).astype(int), 0, grid - 1)
    cell_all, weights_all = gy * grid + gx, weights
    cell = cell_all[near]
    delta, weights = delta[near], weights[near]
    cells = np.unique(cell)
    # every point that passed the motion checks, for the pair's fit over all frames (fit_pair): with a pose
    # error a valid point's own phase can sit far from the prior, so the per-point window is not applied there
    points = {"_r0": r0[ok], "_de": de[ok], "_cell": cell_all, "_w": weights_all,
              "median_epipolar_motion_px": float(np.median(np.abs(de[ok])))}
    if len(delta) < 15 or len(cells) < 3:
        return {"phase_frames": None, "se_frames": None, "points": int(len(delta)), "cells": int(len(cells)), **points}
    centred = delta - prior_frames
    per_cell = np.array([circular_weighted_median(centred[cell == c], weights[cell == c]) for c in cells])
    phase = circular_weighted_median(per_cell, np.ones(len(per_cell)))
    rng = np.random.default_rng(seed)
    boots = [circular_weighted_median(per_cell[rng.integers(0, len(per_cell), len(per_cell))], np.ones(len(per_cell)))
             for _ in range(bootstrap)]
    se = max(1.4826 * mad(wrap(np.array(boots) - phase), 0.0), 0.005) if boots else 0.05
    return {"phase_frames": float(phase + prior_frames), "se_frames": float(se), "points": int(len(delta)),
            "cells": int(len(cells)), "scatter_frames": float(mad(wrap(centred - phase), 0.0)), **points}


def fit_pair(rows: list[dict], *, min_frames: int = 3, huber_px: float = 1.0, bootstrap: int = 200, seed: int = 0) -> dict | None:
    """One pair's phase from every frame together, robust to camera-pose error.

    A point's epipolar residual is r0 = b - phase * de: b is what the poses get wrong at that place in the image
    (fixed over time, one value per image cell) and phase * de is what timing adds (de is how much the residual
    changes per frame of phase, which follows the point's motion). Motion changes direction from frame to
    frame, so a fit over several frames tells the two apart; a single frame cannot. Robust (Huber) least
    squares; the uncertainty is a bootstrap over whole frames."""
    rows = [r for r in rows if "_r0" in r and len(r["_r0"])]
    if len(rows) < min_frames:
        return None
    cells = sorted({int(c) for r in rows for c in r["_cell"]})
    index = {c: i for i, c in enumerate(cells)}

    def solve(sample):
        r0 = np.concatenate([rows[i]["_r0"] for i in sample])
        de = np.concatenate([rows[i]["_de"] for i in sample])
        cl = np.concatenate([[index[int(c)] for c in rows[i]["_cell"]] for i in sample]).astype(int)
        w0 = np.concatenate([rows[i]["_w"] / np.sum(rows[i]["_w"]) for i in sample])  # each frame one vote
        X = np.zeros((len(r0), 1 + len(cells)))
        X[:, 0] = -de
        X[np.arange(len(r0)), 1 + cl] = 1.0
        w = w0.copy()
        coef = np.zeros(X.shape[1])
        for _ in range(10):
            sw = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(X * sw[:, None], r0 * sw, rcond=None)
            res = np.abs(r0 - X @ coef)
            w = w0 * np.where(res <= huber_px, 1.0, huber_px / np.maximum(res, 1e-9))
        return float(coef[0]), np.abs(r0 - X @ coef)

    phase, res = solve(range(len(rows)))
    rng = np.random.default_rng(seed)
    boots = [solve(rng.integers(0, len(rows), len(rows)))[0] for _ in range(bootstrap)]
    sigma = max(float(np.std(boots)), 0.005)
    return {"phase_frames": phase, "sigma_frames": sigma, "frames": len(rows),
            "residual_px": float(np.median(res)), "cells": len(cells)}


# ----------------------------------------------------------------------------- the camera graph
#: Frames a pair must be measured at: one frame's phase can be thrown by a single limb's motion.
MIN_FRAMES = 3


def combine_frames(rows: list[dict], prior_frames: float, *, min_frames: int = MIN_FRAMES, max_spread_frames: float = 0.2) -> dict | None:
    """One pair's phase over every frame it was measured at."""
    vals = np.array([r["phase_frames"] for r in rows]) - prior_frames
    se = np.array([max(r["se_frames"], 0.01) for r in rows])
    if len(vals) < min_frames:
        return None
    w = np.minimum(1.0 / np.maximum(se, 0.02) ** 2, 2500.0)
    med = circular_weighted_median(vals, w)
    dev = np.abs(wrap(vals - med))
    spread = max(mad(dev, 0.0), float(np.quantile(dev, 0.8)))
    if spread > max_spread_frames:
        return None  # the frames disagree: no trustworthy phase for this pair
    sigma = max(1.0 / math.sqrt(float(np.sum(w))), 1.4826 * mad(dev, 0.0) / math.sqrt(len(vals)), 0.005)
    return {"phase_frames": float(med + prior_frames), "sigma_frames": float(sigma), "frames": int(len(vals)),
            "spread_frames": float(spread)}


def solve_graph(pairs: dict[tuple[str, str], dict], labels: list[str], reference: str, fps: float,
                outlier_sigmas: float = 4.0) -> dict:
    """Offsets (seconds) with offset[a] - offset[b] = phase(a, b) / fps, the reference held at 0; edges whose
    residual is far beyond their uncertainty are dropped and the solve repeated."""
    index = {l: i for i, l in enumerate(labels)}
    edges = dict(pairs)
    for _ in range(len(pairs) + 1):
        rows, rhs, wts = [], [], []
        for (a, b), e in edges.items():
            row = np.zeros(len(labels))
            row[index[a]], row[index[b]] = 1.0, -1.0
            rows.append(row)
            rhs.append(e["phase_frames"] / fps)
            wts.append(1.0 / (e["sigma_frames"] / fps))
        gauge = np.zeros(len(labels))
        gauge[index[reference]] = 1.0
        A = np.array(rows + [gauge]) if rows else gauge[None]
        y = np.array(rhs + [0.0])
        w = np.array(wts + [1e9])
        x, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        resid = {k: (x[index[k[0]]] - x[index[k[1]]]) - e["phase_frames"] / fps for k, e in edges.items()}
        worst = max(resid, key=lambda k: abs(resid[k]) / (edges[k]["sigma_frames"] / fps), default=None)
        if worst is None or abs(resid[worst]) / (edges[worst]["sigma_frames"] / fps) <= outlier_sigmas or len(edges) <= 1:
            break
        edges.pop(worst)
    # per-camera uncertainty from the normal equations; cameras the edges do not reach are not placed
    reached = {reference}
    changed = True
    while changed:
        changed = False
        for a, b in edges:
            if (a in reached) != (b in reached):
                reached |= {a, b}
                changed = True
    N = (A * w[:, None]).T @ (A * w[:, None])
    cov = np.linalg.pinv(N)
    degree = {l: sum(1 for k in edges if l in k) for l in labels}
    cameras = {}
    for l in labels:
        sigma = float(math.sqrt(max(cov[index[l], index[l]], 0.0)))
        if l == reference:
            status = "reference"
        elif l not in reached:
            status = "not placed"
        elif degree[l] < 2:
            status = "weak"
        else:
            status = "ok"
        cameras[l] = {"offset_s": float(x[index[l]]) if l in reached else None, "sigma_s": sigma if l in reached else None,
                      "pairs": degree[l], "status": status}
    res = np.array([resid[k] for k in edges]) if edges else np.zeros(0)
    return {"cameras": cameras, "used_pairs": sorted(f"{a}-{b}" for a, b in edges),
            "dropped_pairs": sorted(f"{a}-{b}" for a, b in pairs if (a, b) not in edges),
            "residual_rms_ms": float(np.sqrt(np.mean(res ** 2)) * 1e3) if len(res) else None}


def verdict(cameras: dict, fps: float, max_sigma_frames: float = 0.1) -> str:
    """good: every camera placed, joined by two or more pairs and known to within max_sigma_frames;
    partial: some are; failed: none besides the reference."""
    others = [c for c in cameras.values() if c["status"] != "reference"]
    good = [c for c in others if c["status"] == "ok" and c["sigma_s"] is not None and c["sigma_s"] * fps <= max_sigma_frames]
    if others and len(good) == len(others):
        return "good"
    return "partial" if good else "failed"


# ----------------------------------------------------------------------------- images, tracks, matches
def load_gray(path: Path):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def track(frames: list[Path], centre: int, pts: np.ndarray, *, half: int = 2, fb_px: float = 1.5, win: int = 21):
    """Velocity and acceleration (px/frame, px/frame^2) of `pts` at frames[centre], from Lucas-Kanade tracks to
    centre +-k (the shortest k whose end-to-end move is informative, up to `half`). Returns (vel, acc, valid)."""
    import cv2

    n = len(pts)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    params = dict(winSize=(win, win), maxLevel=4, criteria=criteria)
    g0 = load_gray(frames[centre])
    h, w = g0.shape
    positions, valid = {0: pts.astype(np.float32)}, {0: np.ones(n, bool)}
    for direction in (-1, 1):
        prev, prev_pts, prev_ok = g0, positions[0].copy(), valid[0].copy()
        for step in range(1, half + 1):
            j = centre + direction * step
            if not 0 <= j < len(frames):
                break
            nxt = load_gray(frames[j])
            idx = np.flatnonzero(prev_ok)
            new = prev_pts.copy()
            ok = np.zeros(n, bool)
            if len(idx):
                fwd, st, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, prev_pts[idx].reshape(-1, 1, 2), None, **params)
                fwd = fwd.reshape(-1, 2)
                inside = (st.reshape(-1) > 0) & (fwd[:, 0] >= 0) & (fwd[:, 0] < w) & (fwd[:, 1] >= 0) & (fwd[:, 1] < h)
                keep = idx[inside]
                new[keep] = fwd[inside]
                if len(keep):
                    back, bst, _ = cv2.calcOpticalFlowPyrLK(nxt, prev, new[keep].reshape(-1, 1, 2), None, **params)
                    err = np.linalg.norm(back.reshape(-1, 2) - prev_pts[keep], axis=1)
                    ok[keep] = (bst.reshape(-1) > 0) & (err <= fb_px)
            positions[direction * step], valid[direction * step] = new, ok
            prev, prev_pts, prev_ok = nxt, new, ok
    vel = np.zeros((n, 2), np.float64)
    acc = np.zeros((n, 2), np.float64)
    done = np.zeros(n, bool)
    for k in range(1, half + 1):
        if -k not in positions or k not in positions:
            break
        ok = valid[-k] & valid[k] & ~done
        move = np.linalg.norm(positions[k] - positions[-k], axis=1)
        # the shortest baseline that moved at least 2 px, or the longest there is
        last = k == half or not (-(k + 1) in positions and (k + 1) in positions)
        take = ok & ((move >= 2.0) | last)
        pm, pp, p0 = positions[-k], positions[k], positions[0]
        vel[take] = (pp[take] - pm[take]) / (2.0 * k)
        acc[take] = (pp[take] - 2.0 * p0[take] + pm[take]) / float(k * k)
        done |= take
    return vel, acc, done


class Matcher:
    """RoMa v2 (default) or v1, loaded once. Weights come from TORCH_HOME (the models folder)."""

    def __init__(self, version: str, device: str = "cuda", samples: int = 5000):
        self.version, self.samples = version, samples
        if version == "v2":
            from romav2 import RoMaV2
            self.model = RoMaV2()
        elif version == "v1":
            from romatch import roma_outdoor
            self.model = roma_outdoor(device=device)
        else:
            raise SystemExit(f"--roma must be v1 or v2, not {version!r}")

    def __call__(self, path_a: Path, path_b: Path, box_a=None, box_b=None, min_certainty: float = 0.2):
        """Matches between two images, each optionally cropped to a box (x0, y0, x1, y1) first so the samples
        fall on what the box holds; coordinates are in the full images either way."""
        import tempfile

        import torch
        from PIL import Image

        def prepared(path, box, work):
            with Image.open(path) as im:
                im = im.convert("RGB")
                if box is None:
                    return str(path), (0, 0), im.size
                crop = im.crop(tuple(int(v) for v in box))
                out = Path(work) / f"{Path(path).stem}_{len(list(Path(work).iterdir()))}.png"
                crop.save(out)
                return str(out), (int(box[0]), int(box[1])), crop.size

        with tempfile.TemporaryDirectory() as work:
            pa, oa, (wa, ha) = prepared(path_a, box_a, work)
            pb, ob, (wb, hb) = prepared(path_b, box_b, work)
            with torch.inference_mode():
                if self.version == "v2":
                    preds = self.model.match(pa, pb)
                    matches, cert, _, _ = self.model.sample(preds, self.samples)
                else:
                    warp, certainty = self.model.match(pa, pb)
                    matches, cert = self.model.sample(warp, certainty, num=self.samples)
                ka, kb = self.model.to_pixel_coordinates(matches, ha, wa, hb, wb)
        cert = cert.float().cpu().numpy().reshape(-1)
        keep = cert >= min_certainty
        ka = ka.cpu().numpy()[keep].astype(np.float64) + np.array(oa, dtype=np.float64)
        kb = kb.cpu().numpy()[keep].astype(np.float64) + np.array(ob, dtype=np.float64)
        return ka, kb, cert[keep]


def subject_box(mask_path: Path | None, pad: float = 0.1, min_side: int = 64):
    """The mask's bounding box, padded by `pad` of its size, or None (no mask, or nothing in it)."""
    if mask_path is None or not mask_path.is_file():
        return None
    import cv2

    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.nonzero(m > 127)
    if len(xs) == 0:
        return None
    h, w = m.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = max(pad * (x1 - x0), min_side / 2), max(pad * (y1 - y0), min_side / 2)
    return (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))


def inside_mask(mask_path: Path | None, uv: np.ndarray) -> np.ndarray:
    if mask_path is None or not mask_path.is_file():
        return np.ones(len(uv), bool)
    import cv2

    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, m.shape[1] - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, m.shape[0] - 1)
    return m[y, x] > 127


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transforms", required=True, help="Camera poses transforms.json (undistorted pinhole cameras)")
    ap.add_argument("--frames_root", required=True, help="folder of frame_NNNN/ folders with <image_dir>/<label>.png")
    ap.add_argument("--image_dir", default="images_flat")
    ap.add_argument("--mask_dir", default="fmasks", help="per-frame masks <mask_dir>/<label>.png; matches are kept "
                                                       "where either camera's mask says subject ('' for none)")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--reference", required=True, help="label of the reference camera (offset 0)")
    ap.add_argument("--prior_json", default=None, help='{"cameras": {label: {"offset_s": s}}}: unwrap around these')
    ap.add_argument("--roma", default="v2", choices=("v2", "v1"))
    ap.add_argument("--samples", type=int, default=5000, help="RoMa matches drawn per pair and frame")
    ap.add_argument("--max_frames", type=int, default=12, help="frames measured, spread over the clip")
    ap.add_argument("--track_half", type=int, default=2, help="frames tracked each way")
    ap.add_argument("--pairs_per_camera", type=int, default=4)
    ap.add_argument("--max_pair_angle", type=float, default=75.0, help="degrees between optical axes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cams = load_cameras(Path(args.transforms))
    labels = sorted(cams)
    if args.reference not in cams:
        raise SystemExit(f"reference camera {args.reference} is not in {args.transforms}")
    frame_dirs = sorted(p for p in Path(args.frames_root).glob("frame_*") if p.is_dir())
    half = int(args.track_half)
    if len(frame_dirs) < 3:
        raise SystemExit(f"{args.frames_root}: tracking needs at least 3 frames, found {len(frame_dirs)}")
    half = min(half, (len(frame_dirs) - 1) // 2)
    usable = list(range(half, len(frame_dirs) - half))
    picks = sorted({usable[int(round(i))] for i in np.linspace(0, len(usable) - 1, min(args.max_frames, len(usable)))})
    if len(picks) < MIN_FRAMES:
        raise SystemExit(f"{len(frame_dirs)} frames leave {len(picks)} to measure (each needs {half} on either side "
                         f"for tracking); at least {MIN_FRAMES} are needed for a trustworthy phase")
    prior = {}
    if args.prior_json:
        prior = {str(k): float(v.get("offset_s") or 0.0)
                 for k, v in (json.loads(Path(args.prior_json).read_text()).get("cameras") or {}).items()}
    pairs = choose_pairs(cams, args.pairs_per_camera, args.max_pair_angle)
    print(f"{len(labels)} cameras, {len(pairs)} pairs, frames {[frame_dirs[i].name for i in picks]}, RoMa {args.roma}", flush=True)

    matcher = Matcher(args.roma, samples=args.samples)
    image = lambda i, l: frame_dirs[i] / args.image_dir / f"{l}.png"  # noqa: E731
    mask = lambda i, l: (frame_dirs[i] / args.mask_dir / f"{l}.png") if args.mask_dir else None  # noqa: E731
    measured: dict[tuple[str, str], list[dict]] = {}
    total = len(pairs) * len(picks)
    done = 0
    for fi in picks:
        for a, b in pairs:
            done += 1
            ca, cb = cams[a], cams[b]
            prior_frames = (prior.get(a, 0.0) - prior.get(b, 0.0)) * args.fps
            uv_a, uv_b, cert = matcher(image(fi, a), image(fi, b), subject_box(mask(fi, a)), subject_box(mask(fi, b)))
            counts = {"matches": int(len(uv_a))}
            keep = inside_mask(mask(fi, a), uv_a) | inside_mask(mask(fi, b), uv_b)
            uv_a, uv_b, cert = uv_a[keep], uv_b[keep], cert[keep]
            counts["subject"] = int(len(uv_a))
            row = None
            if len(uv_a) >= 20:
                seq_a = [d / args.image_dir / f"{a}.png" for d in frame_dirs]
                seq_b = [d / args.image_dir / f"{b}.png" for d in frame_dirs]
                va, aa, oka = track(seq_a, fi, uv_a, half=half)
                vb, ab, okb = track(seq_b, fi, uv_b, half=half)
                moving = oka & okb & (np.maximum(np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)) >= 0.5)
                counts["tracked"] = int((oka & okb).sum())
                counts["moving"] = int(moving.sum())
                if moving.sum() >= 20:
                    row = pair_phase(ca, cb, uv_a[moving], uv_b[moving], va[moving], vb[moving], aa[moving], ab[moving],
                                     cert[moving], size_b=(cb.width, cb.height), prior_frames=prior_frames,
                                     seed=fi * 7919 + done)
            if row is not None:
                row["frame"] = frame_dirs[fi].name
                measured.setdefault((a, b), []).append(row)
            if row is None:
                status = "no signal"
            elif row["phase_frames"] is None:
                status = f"{len(row['_r0'])} moving points, no single-frame phase"
            else:
                status = f"single frame {row['phase_frames']:+.3f} f +- {row['se_frames']:.3f} ({len(row['_r0'])} moving points)"
            status += "  [" + ", ".join(f"{k} {v}" for k, v in counts.items()) + "]"
            print(f"[{done}/{total}] {frame_dirs[fi].name} {a}-{b}: {status}", flush=True)

    combined = {}
    for (a, b), rows in measured.items():
        c = fit_pair(rows, seed=hash((a, b)) % 10007)
        if c is not None:
            combined[(a, b)] = c
            print(f"pair {a}-{b}: {c['phase_frames']:+.3f} f +- {c['sigma_frames']:.3f} over {c['frames']} frames "
                  f"(residual {c['residual_px']:.2f} px after the pose term)", flush=True)
    solved = solve_graph(combined, labels, args.reference, args.fps)
    public = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}  # noqa: E731
    report = {"convention": CONVENTION, "fps": args.fps, "reference": args.reference, "roma": args.roma,
              "frames": [frame_dirs[i].name for i in picks], "verdict": verdict(solved["cameras"], args.fps),
              **solved,
              "pairs": [{"a": a, "b": b, **c, "measurements": [public(r) for r in measured.get((a, b), [])]}
                        for (a, b), c in sorted(combined.items())],
              "unmeasured_pairs": sorted(f"{a}-{b}" for a, b in pairs if (a, b) not in combined),
              "prior": prior or None}
    tmp = Path(args.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, args.out)
    for l, c in report["cameras"].items():
        off = "" if c["offset_s"] is None else f"{c['offset_s'] * 1e3:+.2f} ms +- {c['sigma_s'] * 1e3:.2f}"
        was = f" (audio {prior[l] * 1e3:+.2f} ms)" if l in prior else ""
        print(f"camera {l}: {c['status']} {off}{was}")
    print(f"verdict: {report['verdict']}; residual rms {report['residual_rms_ms']} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
