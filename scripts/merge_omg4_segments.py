#!/usr/bin/env python3
"""
merge_omg4_segments.py

Stitch several .omg4 v2 window models -- same world, different time ranges --
into one continuous clip, partitioning Gaussians by their temporal centre at
the seams. The last step of the windowed render-and-repair chain (see
docs/render_and_repair.md).

Why windows at all: one 4D model spread over a whole clip has to split its
splat budget across every instant, and fast motion visibly under-resolves.
Training a model per time window gives each window an undiluted budget; this
script puts them back together.

Two modes:

  * hard -- each Gaussian belongs to exactly one segment, by a cut on
    t_center. Cheapest and the usual choice.
  * fade -- adjacent segments overlap across a fade zone per seam, with
    linear opacity ramps so one hands off to the next. Costs splats (both
    sides are present through the zone) and needs "fade_zones" in the
    manifest.

Two corrections are applied that the segment files never needed on their own,
because each was written with a header clamping playback to its own window:

  * TEMPORAL SIGMA CLAMP. On a merged timeline a segment's wide-sigma
    Gaussians stay faintly alive far outside their slot and drift along their
    velocity vectors, rendering as glinting particles flying off the subject.
    Each Gaussian's t_sigma is clamped so its temporal alpha falls below 1/255
    before it leaves its interval, and further clamped so total visible drift
    stays under --max_drift. Wide sigma is harmless for static content
    (velocity ~ 0) and fatal combined with velocity.

  * SEAM TAIL EXTENSION (hard mode). Clamping tails to die exactly AT the seam
    made the whole image dip through black there -- at the seam instant
    nothing from either side was still visible. Membership stays a hard cut,
    but the clamp interval is widened by --seam_tail_ext in both directions so
    both sides stay alive through the transition; the drift clamp
    independently keeps particles dead.

conda env: none (numpy only).

Usage:
    python3 merge_omg4_segments.py \\
        --manifest /path/to/segments.json \\
        --mode hard \\
        --out /path/to/clip_stitched.omg4 \\
        [--segment_dir /path/to/segments] [--max_drift 0.15] [--seam_tail_ext 0.25]

Manifest (JSON). "start"/"end" are the segment's t_center slot in seconds;
null means unbounded, which the outermost segments want because trained
t_centers drift outside the clip window and their temporal tails still do
useful in-window work. "fade_zones" is required only for --mode fade and must
have exactly one [t0, t1] entry per seam:

    {
      "fps": 29.97,
      "time_range": [0.0, 2.969636],
      "segments": [
        {"file": "window_1.omg4", "start": null,     "end": 0.483817},
        {"file": "window_2.omg4", "start": 0.483817, "end": 0.984318},
        {"file": "window_3.omg4", "start": 0.984318, "end": null}
      ],
      "fade_zones": [[0.450, 0.484], [0.951, 0.985]]
    }

Output:
    --out, one .omg4 v2 file, Gaussians ordered by t_center, carrying the
    manifest's clip-wide time range and fps in its header.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

from splat4d_io import OMG4_V2_FIELDS, OMG4_V2_FLAG_SH, OMG4_V2_VERSION

HEADER = struct.Struct("<IIIIfffI")  # magic, version, count, flags, tmin, tmax, fps, reserved
N_SH_ARRAYS = 45  # SH arrays trailing the base SoA fields when the SH flag is set

T_CENTER_ROW = OMG4_V2_FIELDS.index("t_center")
T_SIGMA_ROW = OMG4_V2_FIELDS.index("t_sigma")
OPACITY_ROW = OMG4_V2_FIELDS.index("opacity")
VELOCITY_ROWS = slice(OMG4_V2_FIELDS.index("vx"), OMG4_V2_FIELDS.index("vz") + 1)

MIN_FADE_WEIGHT = 1e-4  # floor on the ramp so a faded Gaussian's logit stays finite


def read_omg4(path: Path, cache: dict) -> dict:
    """Read one .omg4 v2 file into a (n_arrays, N) float32 matrix.

    Memoized: a segment file may occupy more than one slot (a wide window split
    around a narrower one inserted into its middle), and should be read once."""
    key = str(path)
    if key in cache:
        return cache[key]
    with open(path, "rb") as fp:
        magic, version, count, flags, tmin, tmax, fps, _reserved = HEADER.unpack(fp.read(HEADER.size))
        if version != OMG4_V2_VERSION:
            raise ValueError(f"{path}: expected .omg4 v{OMG4_V2_VERSION}, got v{version}")
        n_arrays = len(OMG4_V2_FIELDS) + (N_SH_ARRAYS if flags & OMG4_V2_FLAG_SH else 0)
        data = np.fromfile(fp, dtype=np.float32, count=n_arrays * count)
    if data.size != n_arrays * count:
        raise ValueError(f"{path}: truncated -- expected {n_arrays * count} floats, read {data.size}")
    result = {"arrays": data.reshape(n_arrays, count), "flags": flags, "magic": magic,
              "tmin": tmin, "tmax": tmax, "fps": fps}
    cache[key] = result
    return result


def write_omg4(path: Path, arrays: np.ndarray, flags: int, magic: int,
               time_range: tuple, fps: float) -> None:
    with open(path, "wb") as fp:
        fp.write(HEADER.pack(magic, OMG4_V2_VERSION, arrays.shape[1], flags,
                             time_range[0], time_range[1], fps, 0))
        fp.write(np.ascontiguousarray(arrays, dtype=np.float32).tobytes())


def scale_opacity(logit: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Multiply a stored opacity logit by a linear weight, in alpha space."""
    alpha = 1.0 / (1.0 + np.exp(-logit))
    faded = np.clip(alpha * weight, 1e-6, 1 - 1e-6)
    return np.log(faded / (1.0 - faded)).astype(np.float32)


def clamp_sigma_to_interval(arrays: np.ndarray, lo: float, hi: float,
                            max_drift: float) -> np.ndarray:
    """Clamp t_sigma so temporal alpha decays below 1/255 before leaving
    [lo, hi], and so velocity-driven drift while visible stays under
    max_drift. See the module docstring for why both are needed."""
    t_center, sigma = arrays[T_CENTER_ROW], arrays[T_SIGMA_ROW]
    alpha = 1.0 / (1.0 + np.exp(-arrays[OPACITY_ROW]))
    # k = how many sigmas out a Gaussian of this peak alpha stays above 1/255
    k = np.sqrt(2.0 * np.maximum(np.log(np.maximum(alpha, 1e-6) * 255.0), 0.5))

    limit = np.full_like(sigma, np.inf)
    if np.isfinite(hi):
        limit = np.minimum(limit, np.maximum(hi - t_center, 1e-4) / k)
    if np.isfinite(lo):
        limit = np.minimum(limit, np.maximum(t_center - lo, 1e-4) / k)
    speed = np.linalg.norm(arrays[VELOCITY_ROWS], axis=0)
    limit = np.minimum(limit, max_drift / (k * np.maximum(speed, 1e-6)))

    arrays[T_SIGMA_ROW] = np.minimum(sigma, limit).astype(np.float32)
    return arrays


def load_manifest(manifest_path: Path, segment_dir: Path | None) -> tuple:
    """(segments, fade_zones, time_range, fps). Each segment is
    (path, lo, hi) with unbounded ends as -inf/+inf."""
    manifest = json.loads(manifest_path.read_text())
    root = segment_dir or manifest_path.parent

    segments = []
    for entry in manifest["segments"]:
        path = Path(entry["file"])
        if not path.is_absolute():
            path = root / path
        lo = -np.inf if entry.get("start") is None else float(entry["start"])
        hi = np.inf if entry.get("end") is None else float(entry["end"])
        segments.append((path, lo, hi))
    if not segments:
        raise ValueError(f"{manifest_path}: no segments")

    fade_zones = [(float(a), float(b)) for a, b in manifest.get("fade_zones", [])]
    time_range = tuple(manifest.get("time_range", (0.0, 0.0)))
    return segments, fade_zones, time_range, float(manifest.get("fps", 30.0))


def merge_segments(segments: list, fade_zones: list, mode: str, time_range: tuple, fps: float,
                   out_path: Path, max_drift: float, seam_tail_ext: float) -> int:
    """Merge and write. Returns the merged Gaussian count."""
    cache = {}
    loaded = [read_omg4(path, cache) for path, _, _ in segments]
    flags = loaded[0]["flags"]
    if any(seg["flags"] != flags for seg in loaded):
        raise ValueError("segments disagree on the SH flag -- export them all with or without SH")

    n = len(segments)
    if mode == "fade" and len(fade_zones) != n - 1:
        raise ValueError(f"fade mode needs one fade zone per seam: "
                         f"{n} segments require {n - 1}, manifest has {len(fade_zones)}")

    # Clamp interval per segment. In fade mode a segment must survive from the start
    # of its left handoff to the end of its right one; in hard mode its own slot,
    # widened at interior seams so neither side blinks out at the transition.
    if mode == "fade":
        intervals = [(-np.inf if i == 0 else fade_zones[i - 1][0],
                      np.inf if i == n - 1 else fade_zones[i][1]) for i in range(n)]
    else:
        intervals = [(lo if np.isinf(lo) else lo - seam_tail_ext,
                      hi if np.isinf(hi) else hi + seam_tail_ext) for _, lo, hi in segments]

    pieces = []
    for i, (segment, (_, lo, hi)) in enumerate(zip(loaded, segments)):
        t_center = segment["arrays"][T_CENTER_ROW]
        in_slot = (t_center >= lo) & (t_center < hi)
        arrays = segment["arrays"][:, in_slot].copy()

        if mode == "fade" and arrays.shape[1]:
            weight = np.ones(arrays.shape[1], dtype=np.float64)
            own = arrays[T_CENTER_ROW]
            if i < n - 1:  # ramp down across the handoff to the next segment
                z0, z1 = fade_zones[i]
                weight *= 1.0 - np.clip((own - z0) / (z1 - z0), 0.0, 1.0)
            if i > 0:      # ramp up across the handoff from the previous one
                z0, z1 = fade_zones[i - 1]
                weight *= np.clip((own - z0) / (z1 - z0), 0.0, 1.0)
            arrays[OPACITY_ROW] = scale_opacity(arrays[OPACITY_ROW], np.maximum(weight, MIN_FADE_WEIGHT))

        pieces.append(clamp_sigma_to_interval(arrays, *intervals[i], max_drift))

    if mode == "fade":
        # Each segment also reaches into the fade zones it participates in but does not
        # own: forward into its right zone (ramping out) and back into its left zone
        # (ramping in), so both sides are present across every handoff.
        for i, (segment, (_, lo, hi)) in enumerate(zip(loaded, segments)):
            t_center = segment["arrays"][T_CENTER_ROW]
            participation = []
            if i > 0:
                participation.append((fade_zones[i - 1], "in"))
            if i < n - 1:
                participation.append((fade_zones[i], "out"))
            for (z0, z1), direction in participation:
                selected = (t_center >= z0) & (t_center < z1) & ~((t_center >= lo) & (t_center < hi))
                if not selected.any():
                    continue
                arrays = segment["arrays"][:, selected].copy()
                ramp = np.clip((arrays[T_CENTER_ROW] - z0) / (z1 - z0), 0.0, 1.0)
                weight = ramp if direction == "in" else 1.0 - ramp
                arrays[OPACITY_ROW] = scale_opacity(arrays[OPACITY_ROW], np.maximum(weight, MIN_FADE_WEIGHT))
                pieces.append(clamp_sigma_to_interval(arrays, z0, z1, max_drift))

    merged = np.concatenate(pieces, axis=1)
    merged = merged[:, np.argsort(merged[T_CENTER_ROW], kind="stable")]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_omg4(out_path, merged, flags, loaded[0]["magic"], time_range, fps)

    for i, (path, lo, hi) in enumerate(segments):
        print(f"  segment {i} ({path.name}): slot [{lo:.3f}, {hi:.3f}) "
              f"contributed {pieces[i].shape[1]:,}")
    t_center = merged[T_CENTER_ROW]
    print(f"wrote {out_path}: {merged.shape[1]:,} splats, "
          f"t_center [{t_center.min():.3f}, {t_center.max():.3f}]")
    return merged.shape[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="segment manifest JSON (see above)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=("hard", "fade"), default="hard",
                    help="hard: one segment owns each Gaussian (default). fade: overlap "
                         "adjacent segments with opacity ramps, needs manifest fade_zones")
    ap.add_argument("--segment_dir", type=Path, default=None,
                    help="directory holding the segment files, overriding the manifest's own "
                         "directory -- lets one manifest stitch a staged variant set "
                         "(e.g. pruned or compressed exports)")
    ap.add_argument("--max_drift", type=float, default=0.15,
                    help="world units a Gaussian may drift while visible (default 0.15)")
    ap.add_argument("--seam_tail_ext", type=float, default=0.25,
                    help="seconds each hard-mode clamp interval is widened past its seam "
                         "so neither side blinks out at the transition (default 0.25)")
    args = ap.parse_args()

    try:
        segments, fade_zones, time_range, fps = load_manifest(args.manifest, args.segment_dir)
        missing = [str(p) for p, _, _ in segments if not p.exists()]
        if missing:
            print("Error: segment file(s) not found:\n  " + "\n  ".join(missing), file=sys.stderr)
            return 1
        merge_segments(segments, fade_zones, args.mode, time_range, fps,
                       args.out, args.max_drift, args.seam_tail_ext)
    except (OSError, ValueError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
