#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
color_adjust.py - per-camera color correction without RawTherapee.

Three methods, chosen per camera or for all cameras in a settings file:

  adjust  exposure (stops), temperature, contrast, saturation
  lut     a .cube 3D LUT (trilinear)
  pp3     a RawTherapee profile, through color_correct.py

The adjustments are defined once, here, as numpy code on float RGB in 0..1, and
Studio's browser preview repeats the same arithmetic in JavaScript; the test
vectors in tests/unit/test_color_adjust.py are what keeps the two agreeing.

Order, for sRGB-encoded input:
  1. decode to linear light
  2. exposure: multiply by 2**exposure
  3. temperature: multiply red by 1 + 0.15*t and blue by 1 - 0.15*t
  4. encode back to sRGB, clip to 0..1
  5. contrast: (x - 0.5) * (1 + contrast) + 0.5
  6. saturation: Y + (x - Y) * (1 + saturation), Y = 0.2126 R + 0.7152 G + 0.0722 B
  7. clip to 0..1

Settings file:
  {"method": "adjust", "adjust": {"exposure": 0.3, ...},
   "lut": "/path/to/look.cube", "pp3_dir": "/path/to/profiles",
   "cameras": {"0001": {"method": "adjust", "adjust": {...}}}}      (optional per-camera overrides)

Usage:
    python3 color_adjust.py --settings settings.json --in_dir frames/ --out_dir corrected/ [--image_ext .png]
Cameras are matched by filename stem, ignoring a leading "undistorted_".
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ADJUST_KEYS = ("exposure", "temperature", "contrast", "saturation")
TEMPERATURE_GAIN = 0.15


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def adjust(rgb, exposure=0.0, temperature=0.0, contrast=0.0, saturation=0.0):
    """rgb: float array [..., 3] in 0..1 (sRGB encoded). Returns the same shape."""
    x = srgb_to_linear(np.asarray(rgb, np.float64))
    x = x * (2.0 ** exposure)
    gain = np.array([1 + TEMPERATURE_GAIN * temperature, 1.0, 1 - TEMPERATURE_GAIN * temperature])
    x = np.clip(linear_to_srgb(x * gain), 0.0, 1.0)
    x = (x - 0.5) * (1 + contrast) + 0.5
    luma = (x * np.array([0.2126, 0.7152, 0.0722])).sum(axis=-1, keepdims=True)
    x = luma + (x - luma) * (1 + saturation)
    return np.clip(x, 0.0, 1.0)


def read_cube(path):
    """(size, table) from a .cube file. table[b, g, r] is the output RGB for input (r, g, b)."""
    size, rows, lo, hi = None, [], np.zeros(3), np.ones(3)
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        key = line.split()[0]
        if key == "LUT_3D_SIZE":
            size = int(line.split()[1])
        elif key == "DOMAIN_MIN":
            lo = np.array([float(v) for v in line.split()[1:4]])
        elif key == "DOMAIN_MAX":
            hi = np.array([float(v) for v in line.split()[1:4]])
        elif key in ("TITLE", "LUT_1D_SIZE") or not (key[0].isdigit() or key[0] in "-+."):
            if key == "LUT_1D_SIZE":
                raise ValueError("1D LUTs are not supported; use a 3D .cube file.")
        else:
            rows.append([float(v) for v in line.split()[:3]])
    if size is None or len(rows) != size ** 3:
        raise ValueError(f"{path}: expected LUT_3D_SIZE and size**3 rows, got size {size} and {len(rows)} rows.")
    return np.array(rows, np.float64).reshape(size, size, size, 3), lo, hi


def apply_lut(rgb, table, lo=None, hi=None):
    """Trilinear lookup. rgb: float [..., 3] in 0..1."""
    n = table.shape[0]
    lo = np.zeros(3) if lo is None else lo
    hi = np.ones(3) if hi is None else hi
    x = np.clip((np.asarray(rgb, np.float64) - lo) / (hi - lo), 0.0, 1.0) * (n - 1)
    i0 = np.minimum(np.floor(x).astype(int), n - 2)
    f = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    out = 0.0
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                w = (fr if dr else 1 - fr) * (fg if dg else 1 - fg) * (fb if db else 1 - fb)
                out = out + w * table[b0 + db, g0 + dg, r0 + dr]
    return np.clip(out, 0.0, 1.0)


def camera_settings(settings, camera):
    merged = {k: v for k, v in settings.items() if k != "cameras"}
    merged.update((settings.get("cameras") or {}).get(camera, {}))
    return merged


def correct_image(array8, cfg, lut_cache):
    """uint8 [h, w, 3] in, uint8 out."""
    rgb = array8.astype(np.float64) / 255.0
    method = cfg.get("method", "none")
    if method == "adjust":
        params = {k: float(cfg.get("adjust", {}).get(k, 0.0)) for k in ADJUST_KEYS}
        rgb = adjust(rgb, **params)
    elif method == "lut":
        path = cfg.get("lut")
        if path not in lut_cache:
            lut_cache[path] = read_cube(path)
        rgb = apply_lut(rgb, *lut_cache[path])
    elif method != "none":
        raise ValueError(f"Unknown method {method!r}.")
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def camera_of(path):
    stem = Path(path).stem
    return stem[len("undistorted_"):] if stem.startswith("undistorted_") else stem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--in_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--image_ext", default=".png")
    args = parser.parse_args()

    settings = json.loads(args.settings.read_text())
    files = sorted(args.in_dir.glob(f"*{args.image_ext}"))
    if not files:
        sys.exit(f"Error: no {args.image_ext} files in {args.in_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lut_cache = {}
    for n, path in enumerate(files):
        cfg = camera_settings(settings, camera_of(path))
        if cfg.get("method") == "pp3":
            from color_correct import apply_pp3, find_pp3, resolve_rawtherapee_cmd

            pp3 = find_pp3(Path(cfg["pp3_dir"]), camera_of(path))
            if pp3 is None:
                sys.exit(f"Error: no .pp3 for camera {camera_of(path)} in {cfg['pp3_dir']}")
            apply_pp3(resolve_rawtherapee_cmd(cfg.get("rawtherapee_cmd")), path, pp3, args.out_dir / path.name)
        else:
            with Image.open(path) as image:
                out = correct_image(np.asarray(image.convert("RGB")), cfg, lut_cache)
            Image.fromarray(out).save(args.out_dir / path.name)
        print(f"PROGRESS {(n + 1) / len(files):.3f} {path.name}", flush=True)


if __name__ == "__main__":
    main()
