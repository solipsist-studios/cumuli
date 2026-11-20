#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""frame_motion.py - how fast the subject moves at each frame, and how much more training each frame should get.

For every camera, dense optical flow (OpenCV DIS) between neighbouring frames, at a reduced size, inside the
subject's masks: a frame's speed in that camera is a high percentile of the flow there (the fast parts of the
body, not its average), in full-resolution pixels per frame, the mean of the flow into and out of the frame. A
frame's speed is the median over cameras, so one camera's bad flow does not decide it.

Frames much faster than the clip's typical frame get a visit weight above 1: training visits them that many times
as often. Fast motion is what a fixed number of visits per frame fits worst (the model has to place the same
Gaussians far apart from one frame to the next), so it gets more of them.

    weight = clip(speed / median_speed, 1, max_boost)   for frames above fast_ratio * median_speed, else 1

Output (--out): {"median_speed_px", "fast_ratio", "max_boost", "percentile",
                 "frames": {frame_folder_name: {"speed_px", "weight", "cameras"}}}
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def load_gray(path: Path, scale: float):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img


def load_mask(path: Path, shape):
    import cv2

    if path is None or not path.is_file():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 127


def flow_speed(prev, nxt, mask, percentile: float) -> float | None:
    """A high percentile of the optical-flow magnitude inside the mask, in the images' pixels per frame."""
    import cv2

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(prev, nxt, None)
    mag = np.linalg.norm(flow, axis=2)
    values = mag[mask] if mask is not None else mag.ravel()
    if values.size < 50:
        return None
    return float(np.percentile(values, percentile))


def camera_speeds(frames: list[Path], label: str, image_dir: str, mask_dir: str | None, scale: float, percentile: float):
    """Per frame, this camera's subject speed in full-resolution pixels per frame (None where it cannot say)."""
    grays = [load_gray(d / image_dir / f"{label}.png", scale) for d in frames]
    masks = []
    for d, g in zip(frames, grays):
        masks.append(None if g is None or not mask_dir else load_mask(d / mask_dir / f"{label}.png", g.shape))
    between = []
    for i in range(len(frames) - 1):
        a, b = grays[i], grays[i + 1]
        if a is None or b is None:
            between.append(None)
            continue
        m = masks[i] if masks[i] is not None else masks[i + 1]
        if masks[i] is not None and masks[i + 1] is not None:
            m = masks[i] | masks[i + 1]
        s = flow_speed(a, b, m, percentile)
        between.append(None if s is None else s / scale)
    out = []
    for i in range(len(frames)):
        near = [v for v in (between[i - 1] if i > 0 else None, between[i] if i < len(between) else None) if v is not None]
        out.append(float(np.mean(near)) if near else None)
    return out


def weights(speeds: dict[str, float], fast_ratio: float, max_boost: float) -> tuple[float, dict[str, float]]:
    """(median speed, {frame: weight}) by the rule in the module docstring."""
    known = [s for s in speeds.values() if s is not None]
    median = float(np.median(known)) if known else 0.0
    out = {}
    for name, s in speeds.items():
        if s is None or median <= 0 or s < fast_ratio * median:
            out[name] = 1.0
        else:
            out[name] = float(min(max(s / median, 1.0), max_boost))
    return median, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames_root", required=True, help="folder of frame_NNNN/ (links are followed)")
    ap.add_argument("--image_dir", default="images_flat")
    ap.add_argument("--mask_dir", default="fmasks_clean", help="'' to measure the whole picture")
    ap.add_argument("--max_side", type=int, default=960, help="flow is computed at this longest side")
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--fast_ratio", type=float, default=1.5, help="a frame this many times the median speed is fast")
    ap.add_argument("--max_boost", type=float, default=3.0, help="most visits a fast frame gets, relative to a normal one")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    frames = sorted(p for p in Path(args.frames_root).glob("frame_*") if p.is_dir())
    if len(frames) < 2:
        raise SystemExit(f"{args.frames_root}: need at least two frames")
    labels = sorted(p.stem for p in (frames[0] / args.image_dir).glob("*.png"))
    first = load_gray(frames[0] / args.image_dir / f"{labels[0]}.png", 1.0)
    scale = min(1.0, args.max_side / max(first.shape))
    print(f"{len(frames)} frames, {len(labels)} cameras, flow at {scale:.3f} of full size", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        per_camera = dict(zip(labels, pool.map(lambda l: camera_speeds(frames, l, args.image_dir, args.mask_dir or None,
                                                                         scale, args.percentile), labels)))
    names = [Path(os.path.realpath(d)).name for d in frames]  # windows link frame_i to the clip's frame_NNNN
    speeds = {}
    for i, name in enumerate(names):
        vals = [per_camera[l][i] for l in labels if per_camera[l][i] is not None]
        speeds[name] = float(np.median(vals)) if vals else None
    median, w = weights(speeds, args.fast_ratio, args.max_boost)
    report = {"median_speed_px": median, "fast_ratio": args.fast_ratio, "max_boost": args.max_boost,
              "percentile": args.percentile,
              "frames": {n: {"speed_px": speeds[n], "weight": w[n], "cameras": sum(per_camera[l][i] is not None for l in labels)}
                         for i, n in enumerate(names)}}
    tmp = Path(args.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, args.out)
    fast = [n for n in names if w[n] > 1]
    print(f"median speed {median:.1f} px/frame; {len(fast)} fast frame(s): "
          + ", ".join(f"{n} {speeds[n]:.1f} px x{w[n]:.2f}" for n in fast), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
