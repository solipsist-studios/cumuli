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
    python3 color_adjust.py --settings settings.json --in_dir frames/ --out_dir corrected/ [--image_ext .png|.webp] [--image_quality 1-100]
Cameras are matched by filename stem, ignoring a leading "undistorted_".
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from image_formats import ImageQualityError, add_quality_argument, check_image_quality, save_image

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


def adjust_uint8(array8, exposure=0.0, temperature=0.0, contrast=0.0, saturation=0.0, strip_rows=256):
    """`adjust` for 8-bit images, at a fraction of the cost: the same arithmetic, uint8 [h, w, 3] in and out.

    Exposure, temperature and contrast act on each channel value alone, so all 256 possible values are
    pushed through `adjust`'s steps once (exactly, in float64) and the image is only looked up. Saturation
    mixes channels, so it runs on strips of rows in float32 to keep memory flat.
    """
    levels = np.arange(256, dtype=np.float64) / 255.0
    gain = np.array([1 + TEMPERATURE_GAIN * temperature, 1.0, 1 - TEMPERATURE_GAIN * temperature])
    linear = srgb_to_linear(levels)[:, None] * (2.0 ** exposure) * gain[None, :]          # [256, 3]
    table = (np.clip(linear_to_srgb(linear), 0.0, 1.0) - 0.5) * (1 + contrast) + 0.5       # [256, 3]
    out = np.empty_like(array8)
    weights = np.array([0.2126, 0.7152, 0.0722], np.float32)
    table32 = table.astype(np.float32)
    for top in range(0, array8.shape[0], strip_rows):
        strip = array8[top:top + strip_rows]
        x = np.stack([table32[strip[..., c], c] for c in range(3)], axis=-1)
        if saturation != 0.0:
            luma = x @ weights
            x = luma[..., None] + (x - luma[..., None]) * np.float32(1 + saturation)
        np.clip(x, 0.0, 1.0, out=x)
        x *= 255.0
        np.rint(x, out=x)
        out[top:top + strip_rows] = x.astype(np.uint8)
    return out


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
    method = cfg.get("method", "none")
    if method == "adjust":
        return adjust_uint8(array8, **{k: float(cfg.get("adjust", {}).get(k, 0.0)) for k in ADJUST_KEYS})
    rgb = array8.astype(np.float64) / 255.0
    if method == "lut":
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


def correct_file(path, out_path, cfg, lut_cache):
    """Correct one image file and write it (PNG at a light compression level: lossless, several times faster to write)."""
    if cfg.get("method") == "pp3":
        from color_correct import apply_pp3, find_pp3, resolve_rawtherapee_cmd

        pp3 = find_pp3(Path(cfg["pp3_dir"]), camera_of(path))
        if pp3 is None:
            raise SystemExit(f"Error: no .pp3 for camera {camera_of(path)} in {cfg['pp3_dir']}")
        rt_cmd = resolve_rawtherapee_cmd(cfg.get("rawtherapee_cmd"))
        if out_path.suffix.lower() == ".png":
            apply_pp3(rt_cmd, path, pp3, out_path)
            return
        # RawTherapee always writes PNG whatever the extension says, so render to a
        # PNG beside the output and re-encode it in the requested format.
        rendered = out_path.with_name(out_path.name + ".rt.png")
        try:
            apply_pp3(rt_cmd, path, pp3, rendered)
            with Image.open(rendered) as image:
                save_image(image.convert("RGB"), out_path, cfg.get("image_quality"))
        finally:
            rendered.unlink(missing_ok=True)
        return
    with Image.open(path) as image:
        out = correct_image(np.asarray(image.convert("RGB")), cfg, lut_cache)
    # PNG at a light compression level; .webp lossless unless an image_quality is set.
    save_image(Image.fromarray(out), out_path, cfg.get("image_quality"), png_compress_level=1)


_LUTS = {}


def _worker(path, out_path, cfg):
    # An output left by an earlier run must not pass for this one's if the correction fails.
    out_path.unlink(missing_ok=True)
    correct_file(Path(path), Path(out_path), cfg, _LUTS)
    return Path(path).name


def _digest(path, cache):
    """SHA-1 of a file's contents, read once."""
    path = str(path)
    if path not in cache:
        cache[path] = hashlib.sha1(Path(path).read_bytes()).hexdigest()
    return cache[path]


def correction_signature(cfg, camera, cache):
    """What this camera's output depends on besides its input: the correction, and the contents of any LUT or profile it uses."""
    parts = [json.dumps({k: v for k, v in cfg.items() if k != "cameras"}, sort_keys=True)]
    if cfg.get("method") == "lut" and cfg.get("lut"):
        parts.append(_digest(cfg["lut"], cache))
    if cfg.get("method") == "pp3" and cfg.get("pp3_dir"):
        from color_correct import find_pp3
        profile = find_pp3(Path(cfg["pp3_dir"]), camera)
        if profile is not None:
            parts.append(_digest(profile, cache))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def load_manifest(path):
    try:
        return json.loads(Path(path).read_text()).get("done", {}) if path else {}
    except (OSError, ValueError):
        return {}


def save_manifest(path, done):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps({"version": 1, "done": done}))
    tmp.replace(path)


def default_workers(settings) -> int:
    """Images corrected at once. A RawTherapee profile is a process per image that rarely fills a core by itself,
    so it gets one per core (up to 24, each holding a frame); the in-process methods are capped at 12."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    methods = {settings.get("method")} | {c.get("method") for c in (settings.get("cameras") or {}).values()}
    return max(1, min(cores, 24 if "pp3" in methods else 12))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--in_dir", required=True, type=Path, nargs="+", help="One or more folders of frames; each is paired with the --out_dir at the same position.")
    parser.add_argument("--out_dir", required=True, type=Path, nargs="+")
    parser.add_argument("--image_ext", default=".png",
                        help="Extension of the input frames; the corrected frames keep it. .webp output is lossless unless --image_quality is set.")
    add_quality_argument(parser)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="A file that records the images already corrected. With it, a run carries on from an earlier one: "
                             "an image whose input, correction, LUT and profile are unchanged, and whose output is there, is not done again.")
    parser.add_argument("--workers", type=int, default=0, help="Images corrected at once. 0 (the default) picks from the method: see default_workers.")
    args = parser.parse_args()
    if len(args.in_dir) != len(args.out_dir):
        sys.exit("Error: --in_dir and --out_dir need the same number of folders.")
    try:
        check_image_quality(args.image_ext, args.image_quality)
    except ImageQualityError as e:
        sys.exit(f"Error: --image_quality: {e}")

    settings = json.loads(args.settings.read_text())
    previous = load_manifest(args.manifest)
    digests, jobs, done_entries, skipped = {}, [], {}, 0
    for in_dir, out_dir in zip(args.in_dir, args.out_dir):
        files = sorted(in_dir.glob(f"*{args.image_ext}"))
        if not files:
            sys.exit(f"Error: no {args.image_ext} files in {in_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            camera = camera_of(path)
            cfg = camera_settings(settings, camera)
            if args.image_quality is not None:   # also enters the manifest signature
                cfg = dict(cfg, image_quality=args.image_quality)
            out_path = out_dir / path.name
            stat = path.stat()
            entry = {"sig": correction_signature(cfg, camera, digests), "in": [stat.st_size, stat.st_mtime_ns]}
            key = f"{out_dir.name}/{path.name}"
            if args.manifest and previous.get(key) == entry and out_path.is_file():
                done_entries[key] = entry
                skipped += 1
            else:
                jobs.append((path, out_path, cfg, key, entry))
    total = len(jobs) + skipped
    if skipped:
        print(f"Carrying on: {skipped} of {total} images are already corrected.", flush=True)
        print(f"PROGRESS {skipped / total:.3f} resumed", flush=True)
    if args.manifest:   # what is recorded is only what is finished: a partial output is not in it
        save_manifest(args.manifest, done_entries)
    if not jobs:
        return
    # One pool across every folder, so there is no wait for the slowest image of each one.
    workers = max(1, min(len(jobs), args.workers or default_workers(settings)))

    def finished(done, key, entry, name):
        done_entries[key] = entry
        if args.manifest and done % 12 == 0:
            save_manifest(args.manifest, done_entries)
        print(f"PROGRESS {(skipped + done) / total:.3f} {name}", flush=True)

    try:
        if workers == 1:
            for done, (path, out_path, cfg, key, entry) in enumerate(jobs, 1):
                _worker(path, out_path, cfg)
                finished(done, key, entry, path.name)
            return
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(_worker, path, out_path, cfg): (path, key, entry) for path, out_path, cfg, key, entry in jobs}
            for done, future in enumerate(as_completed(pending), 1):
                future.result()
                path, key, entry = pending[future]
                finished(done, key, entry, path.name)
    finally:
        if args.manifest:
            save_manifest(args.manifest, done_entries)


if __name__ == "__main__":
    main()
