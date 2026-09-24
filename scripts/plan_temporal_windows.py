#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
plan_temporal_windows.py - choose where to cut a long clip into 4DGS windows.

A long take reconstructs better as several short models than as one wide
fit, and merge_sogst_segments.py stitches them back. Cutting at uniform
frame intervals is the fixed-GOP of video coding, so the obvious question is
whether content should choose the cuts instead. Measured on a 5.00 s take
from a 12-camera ring, the answer is more specific than "yes":

  Quality follows window LENGTH, not window content. Across nine runs,
  LPIPS = 0.00641 + 2.214e-5 * frames at R2 0.980; adding a motion term
  takes R2 to 0.9805 with a NEGATIVE coefficient. The four windows settle
  it: at a fixed ~30 frames their motion content varies 6.3x while their
  scores stay inside 0.00691 to 0.00721, the busiest window scoring best.

  Total cost does not depend on the boundaries at all. Splats per window fit
  341k + 508k * that window's share of clip motion at R2 0.950, and motion
  shares sum to one however the clip is cut, so the total is N * 341k + 508k:
  set by the window COUNT alone.

  Seam cost DOES depend on placement. The LPIPS bump within +-3 frames of
  each seam was +13.4%, +7.3% and +3.6% over the clip median, against motion
  at those seams at the 95th, 54th and 24th percentile. The uniform split
  put its first seam at the busiest instant in the clip.

So the useful reframing is that the interesting question is not where
prediction breaks down, it is where the cut is invisible. Two things are
worth choosing: how many windows, which trades quality against storage on a
closed-form curve with a real optimum, and where the seams fall, which is
where the residual error lives.

This script is written as the general form rather than as those conclusions.
It fits its coefficients (--fit_from) and solves a dynamic program over
boundaries. On material where motion DOES drive per-window quality the fit
picks that up and the plan moves the boundaries; on the material measured so
far the same program returns near-equal lengths with the seams pulled onto
quiet instants. The reduction is a result, not a hard-coded rule.

Usage:
    python3 scripts/plan_temporal_windows.py --run ~/runs/ring12_5s --report
    python3 scripts/plan_temporal_windows.py --run ~/runs/ring12_5s \\
        --windows 4 --out window_plan.json
    python3 scripts/plan_temporal_windows.py --run ~/runs/ring12_5s \\
        --target_lpips 0.0078 --out window_plan.json
"""

import argparse
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from image_formats import SUPPORTED_IMAGE_EXTS  # noqa: E402

# Fitted on ~/Dev/datasets/synth_runs, 2026-09-09, over nine ring12 runs: the
# duration ladder (dur_12, dur_24, dur_48, dur_96), the full 121-frame fit
# (ring12_5s) and the four uniform windows (win_0..win_3). Reproduce with
#   --fit_from <R>/dur_12:0 <R>/dur_24:0 <R>/dur_48:0 <R>/dur_96:0 \
#              <R>/ring12_5s:0 <R>/win_0:0 <R>/win_1:31 <R>/win_2:61 <R>/win_3:91
# Quality R2 0.9808, splats R2 0.9268. Recompute on new material; these are a
# starting point, not a law of nature.
DEFAULT_COEFFICIENTS = {
    "quality_intercept": 0.0064036738,  # LPIPS at zero frames
    "quality_per_frame": 2.3066813e-05,  # LPIPS added per frame of window length
    # Fitted at -9.9e-05 against an intercept of 6.4e-03, so under 2% of the
    # score across the whole range: window content does not measurably drive
    # per-window quality on this material. Kept in the model rather than
    # dropped, so material where it DOES matter refits to a real value.
    "quality_per_motion": -9.9085059e-05,
    "splats_fixed": 181454.5,           # per window, whatever it covers
    "splats_per_motion": 1149900.0,     # over the whole clip, split by share
    # Seam terms are in units of the NORMALISED, smoothed signal (mean 1),
    # the same scale plan_windows sees, so they must be refit whenever the
    # smoothing width changes. Fitted so the three measured seams sum to the
    # 0.00068 gap between the windows pooled 0.00710 and the stitched 0.00778,
    # split by their measured bumps of 13.4, 7.3 and 3.6 percent.
    "seam_intercept": 7.4338e-05,       # LPIPS on the clip mean, per seam
    "seam_per_motion": 1.3669e-04,      # per unit of normalised step motion
    "source": "ring12_5s ladder and windows, 2026-09-09",
}

SH_C0 = 0.28209479177387814


@dataclass
class Coefficients:
    """How window length, motion and seams turn into quality and cost."""

    quality_intercept: float
    quality_per_frame: float
    quality_per_motion: float
    splats_fixed: float
    splats_per_motion: float
    seam_intercept: float
    seam_per_motion: float
    source: str

    @classmethod
    def defaults(cls):
        return cls(**DEFAULT_COEFFICIENTS)

    def window_quality(self, frames, motion_share):
        return (self.quality_intercept
                + self.quality_per_frame * frames
                + self.quality_per_motion * motion_share)

    def seam_cost(self, motion_at_seam):
        return self.seam_intercept + self.seam_per_motion * motion_at_seam

    def splats(self, n_windows):
        return n_windows * self.splats_fixed + self.splats_per_motion


# ---------------------------------------------------------------- signals

def _frame_dirs(root):
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("frame_"))


def _camera_labels(frame_dir):
    images = frame_dir / "images_flat"
    return sorted(p.stem for p in images.iterdir()
                  if p.suffix.lower() in SUPPORTED_IMAGE_EXTS)


def pick_cameras(labels, count):
    """Spread `count` cameras evenly through the rig rather than taking the
    first few, which on a ring would all look from one side."""
    if count >= len(labels):
        return list(labels)
    idx = np.linspace(0, len(labels), count, endpoint=False)
    return [labels[int(i)] for i in idx]


def image_motion_signal(run_dir, cameras=4, width=240, height=210):
    """Mean absolute luminance change between consecutive frames, masked.

    This is the frame-difference an encoder measures, and it needs nothing
    but the training images, so a real capture can plan the same way a
    synthetic run does. On the 5 s take it correlated +0.969 with armature
    joint speed and +0.716 with the single wide model's per-frame LPIPS.
    """
    root = Path(run_dir) / "flipbook_src"
    frames = _frame_dirs(root)
    if len(frames) < 2:
        raise SystemExit(f"{root} holds {len(frames)} frames, need at least 2")
    labels = pick_cameras(_camera_labels(frames[0]), cameras)

    signal = np.zeros(len(frames) - 1)
    for label in labels:
        previous = None
        for i, frame in enumerate(frames):
            image = _load_gray(frame / "images_flat", label, width, height)
            mask = _load_gray(frame / "fmasks_clean", label, width, height)
            if mask is not None:
                image = image * (mask / 255.0)
            if previous is not None:
                signal[i - 1] += float(np.abs(image - previous).mean())
            previous = image
    return signal / len(labels)


def _load_gray(directory, label, width, height):
    from PIL import Image

    for ext in SUPPORTED_IMAGE_EXTS:
        path = directory / f"{label}{ext}"
        if path.is_file():
            return np.asarray(
                Image.open(path).convert("L").resize((width, height)),
                dtype=np.float32)
    return None


def joint_motion_signal(run_dir):
    """Summed 3D joint displacement per frame step.

    Synthetic runs carry exact armature positions in render/joints; captures
    carry triangulated keypoints in poses_3d. Both are the same measurement,
    and both are noise-free next to the image signal, but neither sees hair
    or cloth, which the image signal does."""
    run_dir = Path(run_dir)
    for sub, key in (("render/joints", "joints"), ("poses_3d", None)):
        directory = run_dir / sub
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("*.json"))
        if len(files) < 2:
            continue
        points = [_joint_positions(f, key) for f in files]
        stacked = np.stack(points)
        return np.linalg.norm(np.diff(stacked, axis=0), axis=2).sum(axis=1)
    raise SystemExit(f"no render/joints or poses_3d under {run_dir}")


def _joint_positions(path, key):
    data = json.loads(path.read_text())
    if key and key in data:
        joints = data[key]
        return np.array([joints[name]["head"] for name in sorted(joints)])
    for candidate in ("keypoints_3d", "points", "keypoints"):
        if candidate in data:
            return np.asarray(data[candidate], dtype=float)[:, :3]
    raise SystemExit(f"{path} carries no recognisable 3D points")


def error_motion_signal(eval_json, fps=24.0):
    """Per-frame error of a model already trained over the whole clip.

    This is two-pass planning: spend one cheap wide fit, then cut where it
    actually struggled. It measures the thing being optimised rather than a
    proxy for it, at the cost of a first pass."""
    views = json.loads(Path(eval_json).read_text())["views"]
    times = np.array([v["time"] for v in views])
    lpips = np.array([v["lpips"] for v in views])
    index = np.rint(times * fps).astype(int)
    frames = np.arange(index.min(), index.max() + 1)
    per_frame = np.array([lpips[index == f].mean() if (index == f).any() else np.nan
                          for f in frames])
    per_frame = _fill_gaps(per_frame)
    return 0.5 * (per_frame[:-1] + per_frame[1:])


def _fill_gaps(values):
    missing = np.isnan(values)
    if missing.any():
        idx = np.arange(len(values))
        values = values.copy()
        values[missing] = np.interp(idx[missing], idx[~missing], values[~missing])
    return values


def per_frame_lpips(eval_json, fps=24.0):
    """Per-frame mean LPIPS from an eval report, gaps interpolated."""
    views = json.loads(Path(eval_json).read_text())["views"]
    times = np.array([v["time"] for v in views])
    lpips = np.array([v["lpips"] for v in views])
    index = np.rint(times * fps).astype(int)
    frames = np.arange(index.min(), index.max() + 1)
    per_frame = np.array([lpips[index == f].mean() if (index == f).any() else np.nan
                          for f in frames])
    return _fill_gaps(per_frame)


def fit_seam_coefficients(stitched_eval, pooled_lpips, seam_frames, signal,
                          fps=24.0, half_width=3, smoothing=5):
    """Fit the per-seam cost from a stitched model against its own windows.

    The gap between a stitched file's score and the pooled score of the
    windows it was built from IS the total seam cost. Splitting that total by
    each seam's local bump over the clip median attributes it, and regressing
    those shares on the motion at each seam gives a cost that a plan can
    minimise. Attribution rather than direct measurement, because a seam's
    influence reaches further than any window you would choose to measure it
    in: on the 5 s take a +-3 frame window captured only about a sixth of the
    total, so the shares are trustworthy while the absolute widths are not."""
    per_frame = per_frame_lpips(stitched_eval, fps)
    median = float(np.median(per_frame))
    seam_frames = list(seam_frames)
    bumps = []
    for b in seam_frames:
        lo, hi = max(0, b - half_width), min(len(per_frame), b + half_width + 1)
        bumps.append(float(per_frame[lo:hi].mean() / median - 1.0))
    bumps = np.array(bumps)
    if bumps.sum() <= 0:
        raise SystemExit("no seam showed a bump over the clip median, so there "
                         "is nothing to attribute")
    total = float(per_frame.mean()) - float(pooled_lpips)
    costs = total * bumps / bumps.sum()

    density = smooth(normalise(signal), smoothing)
    motion = np.array([density[b - 1] for b in seam_frames])
    design = np.column_stack([np.ones(len(motion)), motion])
    (s0, s1), *_ = np.linalg.lstsq(design, costs, rcond=None)
    return float(s0), float(s1), {"bumps": bumps.tolist(),
                                  "costs": costs.tolist(),
                                  "motion": motion.tolist(),
                                  "total": total}


def normalise(signal):
    """Scale a signal to mean 1, so seam coefficients are comparable between
    an image difference in grey levels and a joint displacement in metres."""
    signal = np.asarray(signal, dtype=np.float64)
    total = signal.mean()
    return signal / total if total > 0 else np.ones_like(signal)


def smooth(signal, window=5):
    """Boxcar, so a single noisy frame cannot claim to be the quiet spot."""
    if window <= 1:
        return np.asarray(signal, dtype=np.float64)
    kernel = np.ones(int(window)) / float(window)
    return np.convolve(np.asarray(signal, dtype=np.float64), kernel, mode="same")


# ---------------------------------------------------------------- fitting

def fit_coefficients(runs, signal, offsets):
    """Refit quality, cost and seam coefficients from finished runs.

    `runs` are directories holding experiment.json, eval_4d.json and
    splat_4d.sogst. `signal` is the parent clip's step signal and `offsets`
    gives each run's first frame within it, which cannot be read from
    experiment.json: a window trained from symlinked frames records the
    parent's --frame_start, not its own, so every window of a clip claims the
    same start. Pass them explicitly (path:offset on the command line)."""
    frames, motion, lpips, splats = [], [], [], []
    signal = np.asarray(signal, dtype=np.float64)
    total_motion = signal.sum()
    for run, offset in zip(runs, offsets):
        run = Path(run)
        report = json.loads((run / "eval_4d.json").read_text())
        experiment = json.loads((run / "experiment.json").read_text())
        count = int(experiment["frames"]["count"])
        frames.append(count)
        window_motion = signal[offset:offset + count - 1].sum()
        motion.append(window_motion / total_motion if total_motion > 0 else 0.0)
        lpips.append(float(np.mean([v["lpips"] for v in report["views"]])))
        meta = json.loads(zipfile.ZipFile(run / "splat_4d.sogst").read("meta.json"))
        splats.append(float(meta["count"]))

    frames = np.array(frames, dtype=float)
    motion = np.array(motion, dtype=float)
    lpips = np.array(lpips, dtype=float)
    splats = np.array(splats, dtype=float)

    design = np.column_stack([np.ones(len(frames)), frames, motion])
    (q0, q1, q2), *_ = np.linalg.lstsq(design, lpips, rcond=None)
    cost_design = np.column_stack([np.ones(len(motion)), motion])
    (s0, s1), *_ = np.linalg.lstsq(cost_design, splats, rcond=None)

    coef = Coefficients.defaults()
    coef.quality_intercept = float(q0)
    coef.quality_per_frame = float(q1)
    coef.quality_per_motion = float(q2)
    coef.splats_fixed = float(s0)
    coef.splats_per_motion = float(s1)
    coef.source = f"refit from {len(runs)} runs"
    return coef, {"r2_quality": _r2(lpips, design @ [q0, q1, q2]),
                  "r2_splats": _r2(splats, cost_design @ [s0, s1]),
                  "frames": frames.tolist(), "motion_share": motion.tolist(),
                  "lpips": lpips.tolist(), "splats": splats.tolist()}


def _r2(actual, predicted):
    residual = ((actual - predicted) ** 2).sum()
    total = ((actual - actual.mean()) ** 2).sum()
    return float(1.0 - residual / total) if total > 0 else float("nan")


# ---------------------------------------------------------------- planning

@dataclass
class Plan:
    boundaries: list      # frame index of each window start, plus the end
    cost: float
    pooled_lpips: float
    stitched_lpips: float
    splats: float
    seam_motion: list


def plan_windows(signal, n_windows, coef=None, min_frames=8, max_frames=None,
                 seam_smoothing=5):
    """Dynamic program over cut points, minimising predicted stitched error.

    Cost is the frame-weighted mean of each window's predicted quality plus a
    seam cost at each interior boundary. Written as a DP rather than a
    formula because the seam term makes the objective non-separable in window
    length: with seams free the best plan is not always equal lengths, and on
    material whose fitted motion coefficient is non-zero it is not even
    close."""
    coef = coef or Coefficients.defaults()
    density = normalise(signal)
    total_frames = len(density) + 1
    seam_density = smooth(density, seam_smoothing)
    cumulative = np.concatenate([[0.0], np.cumsum(density)])
    total_motion = cumulative[-1]

    if n_windows < 1:
        raise ValueError("n_windows must be at least 1")
    if n_windows * min_frames > total_frames:
        raise ValueError(f"{n_windows} windows of at least {min_frames} frames "
                         f"do not fit in {total_frames}")
    max_frames = max_frames or total_frames

    def window_cost(start, end):
        n = end - start
        share = (cumulative[end - 1] - cumulative[start]) / total_motion \
            if total_motion > 0 else 0.0
        return (n / total_frames) * coef.window_quality(n, share)

    # best[w][f]: cheapest cost of covering frames [0, f) with w windows.
    infinity = float("inf")
    best = np.full((n_windows + 1, total_frames + 1), infinity)
    back = np.zeros((n_windows + 1, total_frames + 1), dtype=int)
    best[0][0] = 0.0
    for w in range(1, n_windows + 1):
        for end in range(min_frames, total_frames + 1):
            lower = max(0, end - max_frames)
            for start in range(lower, end - min_frames + 1):
                previous = best[w - 1][start]
                if previous == infinity:
                    continue
                seam = 0.0 if start == 0 else coef.seam_cost(seam_density[start - 1])
                candidate = previous + seam + window_cost(start, end)
                if candidate < best[w][end]:
                    best[w][end] = candidate
                    back[w][end] = start
    if best[n_windows][total_frames] == infinity:
        raise ValueError("no split satisfies the length constraints")

    boundaries = [total_frames]
    frame, w = total_frames, n_windows
    while w > 0:
        frame = int(back[w][frame])
        boundaries.append(frame)
        w -= 1
    boundaries = sorted(boundaries)

    return _describe(boundaries, density, cumulative, seam_density, coef,
                     total_frames, total_motion)


def _describe(boundaries, density, cumulative, seam_density, coef,
              total_frames, total_motion):
    pooled = 0.0
    for start, end in zip(boundaries, boundaries[1:]):
        n = end - start
        share = (cumulative[end - 1] - cumulative[start]) / total_motion \
            if total_motion > 0 else 0.0
        pooled += (n / total_frames) * coef.window_quality(n, share)
    seam_motion = [float(seam_density[b - 1]) for b in boundaries[1:-1]]
    seams = sum(coef.seam_cost(m) for m in seam_motion)
    n_windows = len(boundaries) - 1
    return Plan(boundaries=boundaries, cost=pooled + seams, pooled_lpips=pooled,
                stitched_lpips=pooled + seams, splats=coef.splats(n_windows),
                seam_motion=seam_motion)


def choose_window_count(signal, coef=None, min_frames=8, max_frames=None,
                        target_lpips=None, splat_budget=None, max_windows=12,
                        seam_smoothing=5):
    """Plan at every feasible window count and pick one.

    With no target, the best stitched score wins, which is where shorter
    windows stop paying for the seams they add. With a target, the cheapest
    plan that reaches it wins, because past that point extra windows buy
    quality nobody asked for at 341k splats each."""
    coef = coef or Coefficients.defaults()
    total_frames = len(signal) + 1
    plans = {}
    for n in range(1, max_windows + 1):
        if n * min_frames > total_frames:
            break
        if splat_budget and coef.splats(n) > splat_budget:
            break
        plans[n] = plan_windows(signal, n, coef, min_frames, max_frames,
                                seam_smoothing)
    if not plans:
        raise ValueError("no window count satisfies the constraints")
    if target_lpips:
        reaching = [n for n, p in plans.items() if p.stitched_lpips <= target_lpips]
        if reaching:
            return min(reaching), plans
        return min(plans, key=lambda n: plans[n].stitched_lpips), plans
    return min(plans, key=lambda n: plans[n].stitched_lpips), plans


# ---------------------------------------------------------------- output

def build_plan_json(plan, run_dir, source_frame_start, fps, signal_name,
                    coef, out_dir_template):
    windows = []
    for i, (start, end) in enumerate(zip(plan.boundaries, plan.boundaries[1:])):
        windows.append({
            "index": i,
            "frame_start": int(source_frame_start + start),
            "frame_count": int(end - start),
            "local_frame_start": int(start),
            "offset_seconds": round(start / fps, 7),
            "out_dir": out_dir_template.format(index=i) if out_dir_template else None,
        })
    return {
        "run": str(run_dir),
        "fps": fps,
        "signal": signal_name,
        "coefficients": asdict(coef),
        "windows": windows,
        # The cut sits BETWEEN the last frame of one window and the first
        # of the next, so the seam time is the midpoint, (b - 0.5) / fps.
        # merge_sogst_segments.default_seams derives the same value from the
        # windows' coverage; carrying it here lets the merger use the plan
        # directly and land in exactly the same place.
        "seams_seconds": [round((b - 0.5) / fps, 7) for b in plan.boundaries[1:-1]],
        "seam_motion": [round(m, 5) for m in plan.seam_motion],
        "predicted": {
            "pooled_lpips": round(plan.pooled_lpips, 6),
            "stitched_lpips": round(plan.stitched_lpips, 6),
            "splats": int(round(plan.splats)),
        },
    }


def rate_distortion_table(signal, coef=None, min_frames=8, max_windows=8,
                          seam_smoothing=5):
    coef = coef or Coefficients.defaults()
    rows = []
    for n in range(1, max_windows + 1):
        if n * min_frames > len(signal) + 1:
            break
        plan = plan_windows(signal, n, coef, min_frames, None, seam_smoothing)
        rows.append((n, plan.pooled_lpips, plan.stitched_lpips, plan.splats,
                     [b for b in plan.boundaries[1:-1]]))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="a finished run directory holding flipbook_src/")
    ap.add_argument("--signal", choices=["images", "joints", "error"],
                    default="images")
    ap.add_argument("--signal_cameras", type=int, default=4,
                    help="cameras to average the image signal over, spread "
                         "evenly around the rig (default 4)")
    ap.add_argument("--eval_json", default=None,
                    help="--signal error: the prior wide fit's eval_4d.json "
                         "(default: <run>/eval_4d.json)")
    ap.add_argument("--boundaries", default=None,
                    help="comma-separated interior cut frames, forcing a split "
                         "instead of solving for one. This is how a control "
                         "arm is run: the same plan format, the same merger, "
                         "the same predictions, only the cuts chosen by hand")
    ap.add_argument("--windows", type=int, default=None,
                    help="fixed window count; omit to let the plan choose")
    ap.add_argument("--target_lpips", type=float, default=None,
                    help="choose the cheapest window count reaching this "
                         "predicted stitched LPIPS")
    ap.add_argument("--splat_budget", type=float, default=None)
    ap.add_argument("--min_frames", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--seam_smoothing", type=int, default=5,
                    help="boxcar width in frames for the seam signal, so one "
                         "noisy frame cannot claim to be the quiet spot")
    ap.add_argument("--fit_from", nargs="+", default=None,
                    help="refit the coefficients from finished runs instead of "
                         "using the measured defaults. Each entry is a run "
                         "directory, optionally 'path:first_frame' giving where "
                         "that run starts within this clip; a window trained "
                         "from symlinked frames records the parent's "
                         "--frame_start, so its own offset cannot be read back")
    ap.add_argument("--fit_seams_from", default=None,
                    help="a stitched model's eval_4d.json, to refit the seam "
                         "cost against the windows it was built from")
    ap.add_argument("--fit_seams_pooled", type=float, default=None,
                    help="those windows' pooled LPIPS; the gap to the stitched "
                         "score is the total seam cost")
    ap.add_argument("--fit_seams_at", default=None,
                    help="comma-separated frame indices of that file's seams")
    ap.add_argument("--fps", type=float, default=None,
                    help="default: the run's own frames.fps")
    ap.add_argument("--out", default=None, help="write window_plan.json here")
    ap.add_argument("--out_dir_template", default=None,
                    help="per-window output directory, e.g. "
                         "'~/runs/planned/win_{index}'")
    ap.add_argument("--signal_npy", default=None,
                    help="cache the computed signal here, or load it if present")
    ap.add_argument("--report", action="store_true",
                    help="print the rate-distortion table over window counts")
    args = ap.parse_args()

    run = Path(args.run).expanduser()
    experiment = json.loads((run / "experiment.json").read_text())
    fps = args.fps or float(experiment["frames"].get("fps") or 24.0)
    source_start = int(experiment["frames"]["start"])

    cache = Path(args.signal_npy).expanduser() if args.signal_npy else None
    if cache and cache.is_file():
        signal = np.load(cache)
        print(f"loaded {args.signal} signal from {cache}")
    else:
        if args.signal == "images":
            signal = image_motion_signal(run, cameras=args.signal_cameras)
        elif args.signal == "joints":
            signal = joint_motion_signal(run)
        else:
            signal = error_motion_signal(args.eval_json or (run / "eval_4d.json"), fps)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, signal)
    signal = normalise(signal)

    coef = Coefficients.defaults()
    if args.fit_from:
        runs, offsets = [], []
        for entry in args.fit_from:
            path, _, offset = entry.rpartition(":")
            if path and offset.isdigit():
                runs.append(str(Path(path).expanduser()))
                offsets.append(int(offset))
            else:
                run_path = Path(entry).expanduser()
                other = json.loads((run_path / "experiment.json").read_text())
                runs.append(str(run_path))
                offsets.append(int(other["frames"]["start"]) - source_start)
        coef, quality = fit_coefficients(runs, signal, offsets)
        print(f"refit from {len(runs)} runs: quality R2 {quality['r2_quality']:.4f}, "
              f"splats R2 {quality['r2_splats']:.4f}")
        print(f"  LPIPS = {coef.quality_intercept:.5f} "
              f"+ {coef.quality_per_frame:.4e} x frames "
              f"+ {coef.quality_per_motion:.4e} x motion share")
        print(f"  splats = {coef.splats_fixed/1e3:.0f}k per window "
              f"+ {coef.splats_per_motion/1e3:.0f}k x motion share")

    if args.fit_seams_from:
        if args.fit_seams_pooled is None or not args.fit_seams_at:
            raise SystemExit("--fit_seams_from needs --fit_seams_pooled and "
                             "--fit_seams_at")
        seam_frames = [int(x) for x in args.fit_seams_at.split(",")]
        s0, s1, detail = fit_seam_coefficients(
            args.fit_seams_from, args.fit_seams_pooled, seam_frames, signal,
            fps, smoothing=args.seam_smoothing)
        coef.seam_intercept, coef.seam_per_motion = s0, s1
        coef.source += " + seams refit"
        print(f"refit seams from {len(seam_frames)} measured seams, total cost "
              f"{detail['total']:.5f} LPIPS")
        print(f"  seam cost = {s0:.4e} + {s1:.4e} x motion at the seam")

    frames = len(signal) + 1
    print(f"{run.name}: {frames} frames at {fps:g} fps, "
          f"{args.signal} signal, dynamic range "
          f"{signal.max()/max(signal.min(), 1e-9):.1f}x")

    if args.report:
        print()
        print(f"{'windows':>8}{'pooled':>10}{'stitched':>10}{'splats':>11}"
              f"  boundaries")
        for n, pooled, stitched, splats, boundaries in rate_distortion_table(
                signal, coef, args.min_frames, seam_smoothing=args.seam_smoothing):
            marker = "" if n > 1 else "   (one model, no seams)"
            print(f"{n:>8}{pooled:>10.5f}{stitched:>10.5f}{splats/1e6:>10.2f}M"
                  f"  {boundaries}{marker}")
        print()

    if args.boundaries:
        interior = [int(x) for x in args.boundaries.split(",") if x.strip()]
        boundaries = [0] + interior + [frames]
        if sorted(boundaries) != boundaries or len(set(boundaries)) != len(boundaries):
            raise SystemExit(f"--boundaries must be increasing and inside "
                             f"(0, {frames})")
        density = normalise(signal)
        plan = _describe(boundaries, density,
                         np.concatenate([[0.0], np.cumsum(density)]),
                         smooth(density, args.seam_smoothing), coef,
                         frames, density.sum())
        chosen = len(boundaries) - 1
        print(f"forced {chosen} windows at {interior}")
    elif args.windows:
        plan = plan_windows(signal, args.windows, coef, args.min_frames,
                            args.max_frames, args.seam_smoothing)
        chosen = args.windows
    else:
        chosen, plans = choose_window_count(
            signal, coef, args.min_frames, args.max_frames, args.target_lpips,
            args.splat_budget, seam_smoothing=args.seam_smoothing)
        plan = plans[chosen]
        why = (f"cheapest reaching {args.target_lpips}" if args.target_lpips
               else "best predicted stitched score")
        print(f"chose {chosen} windows ({why})")

    print(f"plan: {chosen} windows, lengths "
          f"{[b - a for a, b in zip(plan.boundaries, plan.boundaries[1:])]}")
    print(f"  seams at frames {plan.boundaries[1:-1]}, "
          f"motion there {[round(m, 3) for m in plan.seam_motion]} "
          f"(clip mean is 1.000)")
    print(f"  predicted pooled {plan.pooled_lpips:.5f}, "
          f"stitched {plan.stitched_lpips:.5f}, {plan.splats/1e6:.2f}M splats")

    document = build_plan_json(plan, run, source_start, fps, args.signal, coef,
                               args.out_dir_template)
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=2))
        print(f"  wrote {out}")
    else:
        print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
