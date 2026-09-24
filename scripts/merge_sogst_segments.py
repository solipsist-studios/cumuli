#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
merge_sogst_segments.py - stitch windowed .sogst models into one clip.

A long clip reconstructs better as several short models than as one wide
fit. Measured on a 5.00 s take from a 12-camera ring, four ~1.25 s windows
scored LPIPS 0.00710 against 0.00899 for a single model, a 21% improvement
for 1.5x the storage, while raising the splat budget or tightening the
initial temporal sigma within one model bought under 3%. Windows are
therefore the lever worth using, and this turns them back into the single
asset a player wants.

Each window is trained standalone with its own time origin, so merging is
five steps:

  1. Shift every splat's t_center into global time. A splat's position is
     xyz + v * (t - t_center); substituting t_local = t_global - offset
     shows only t_center moves. Velocity and sigma are unchanged.
  2. Partition on t_center at the seams, so exactly one window owns each
     instant.
  3. Gate each splat's temporal tail (see gate_tails). Partitioning alone is
     not enough: a splat still renders wherever its temporal Gaussian is
     non-zero, and position extrapolates linearly, so a window's splats
     streak across every other window's frames. Skipping this step scored
     LPIPS 0.0164 against 0.0079 with it, on a stitch whose windows scored
     0.0071 apiece.
  4. Snap t_center onto the packer's codebook and slide xyz to match (see
     snap_time_centers), so the archive's 256 time codes cost no position
     error over the wider merged range.
  5. Concatenate and repack as one archive over the global time range.

Measured on the 5.00 s take: the stitched file scores LPIPS 0.00789 and
PSNR 39.22 dB, against 0.00899 and 39.21 dB for a single wide fit and
0.00710 and 39.50 dB for the four windows scored separately. Stitching
therefore keeps most of windowing's gain in one asset. The residue is
concentrated within about 0.15 s of a seam, where adjacent windows
disagree; views a second away from any seam score 0.0076.

Two seam treatments, following merge_omg4_segments.py in 4dgs-utils, which
did the same job for the older .omg4 format:

  hard  each window keeps only splats whose t_center lies in its slot. One
        model is responsible for any instant. Simple, and the seam can show
        if adjacent windows disagree about geometry.
  fade  windows overlap across a zone at each seam, with linear opacity
        ramps (outgoing 1 to 0, incoming 0 to 1) applied to the opacity
        logit. Costs splats but hides a disagreement rather than cutting
        to it, and measured better on both metrics, so it is the default.
        The two metrics disagree about the width: LPIPS is best at 0.35 s
        and worse by 1.00 s, while PSNR climbs all the way to 1.00 s, which
        is a wide fade blurring across the seam. 0.35 s takes the LPIPS
        side of that.

Usage:
    python3 scripts/merge_sogst_segments.py \\
        --segment win_0/splat_4d.sogst 0.0 \\
        --segment win_1/splat_4d.sogst 1.291667 \\
        --segment win_2/splat_4d.sogst 2.541667 \\
        --segment win_3/splat_4d.sogst 3.791667 \\
        --out stitched.sogst --mode fade --fade 0.35

Offsets are each window's start in global seconds: the source frame index
its local frame 0 came from, divided by fps.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_render import decode_sogst_fields  # noqa: E402
from sogst_io import SOGST_FIELDS  # noqa: E402
from sogst_pack import kmeans_1d, pack_sogst  # noqa: E402

# Opacity logit for a splat ramped fully off. sigmoid(-30) is ~1e-13, which
# the packer's prune threshold discards rather than storing a dead splat.
LOGIT_OFF = -30.0


def load_segment(path, offset):
    """Decode one window and move it into global time."""
    header, fields = decode_sogst_fields(str(path))
    fields = {k: np.asarray(v, dtype=np.float32).copy() for k, v in fields.items()}
    fields["t_center"] = fields["t_center"] + np.float32(offset)
    return {
        "path": str(path),
        "offset": float(offset),
        "fields": fields,
        "count": len(fields["x"]),
        "t_lo": float(header["time_min"]) + offset,
        "t_hi": float(header["time_max"]) + offset,
        "fps": float(header.get("fps") or 0.0),
    }


def segments_from_plan(path, model_name="splat_4d.sogst"):
    """Read window paths, offsets and seam times out of a window plan.

    plan_temporal_windows.py already knows where the cuts are and what each
    window's time origin is, so retyping them into --segment is a chance to
    get one wrong. Each window's `out_dir` is where run_window_plan.py put
    it; the model inside is `model_name`."""
    plan = json.loads(Path(path).read_text())
    pairs, missing = [], []
    for window in plan["windows"]:
        out_dir = window.get("out_dir")
        if not out_dir:
            missing.append(window["index"])
            continue
        model = Path(out_dir).expanduser() / model_name
        pairs.append((str(model), float(window["offset_seconds"])))
    if missing:
        raise SystemExit(
            f"windows {missing} in {path} carry no out_dir, so the plan does "
            "not say where their models are. Re-plan with --out_dir_template, "
            "or pass --segment explicitly.")
    return pairs, [float(t) for t in plan.get("seams_seconds", [])]


def default_seams(segments):
    """Cut midway through the gap between adjacent windows.

    Windows trained on adjacent but disjoint frame ranges leave a one-frame
    gap between the end of one and the start of the next, and the midpoint
    is the only cut that favours neither."""
    return [0.5 * (a["t_hi"] + b["t_lo"])
            for a, b in zip(segments, segments[1:])]


def slot_bounds(segments, seams):
    lows = [-np.inf] + list(seams)
    highs = list(seams) + [np.inf]
    return list(zip(lows, highs))


def select_hard(segment, lo, hi):
    tc = segment["fields"]["t_center"]
    return (tc >= lo) & (tc < hi), None


def select_fade(segment, lo, hi, fade):
    """Keep the slot plus half a fade zone either side, ramping opacity.

    A splat inside an overlap zone is kept by BOTH neighbours, each at
    partial opacity, so the pair sums to roughly one rather than two."""
    tc = segment["fields"]["t_center"]
    half = 0.5 * fade
    lo_edge = lo - half if np.isfinite(lo) else lo
    hi_edge = hi + half if np.isfinite(hi) else hi
    keep = (tc >= lo_edge) & (tc < hi_edge)

    weight = np.ones_like(tc, dtype=np.float64)
    if np.isfinite(lo) and fade > 0:                  # ramping in
        w = np.clip((tc - (lo - half)) / fade, 0.0, 1.0)
        weight = np.minimum(weight, w)
    if np.isfinite(hi) and fade > 0:                  # ramping out
        w = np.clip(((hi + half) - tc) / fade, 0.0, 1.0)
        weight = np.minimum(weight, w)
    return keep, weight


def gate_tails(fields, lo, hi, k, floor, cap_abs=0.0):
    """Cap each splat's temporal sigma so it cannot render outside its slot.

    A splat's position is xyz + v * (t - t_center), so evaluating one far
    outside the window it trained on extrapolates it across the room. Alpha
    there is small, but a faint splat in the wrong place is a streak, and
    that is what a perceptual metric punishes: the first stitch of a 5 s
    take rendered every window's far tail into every other window's frames
    and scored LPIPS 0.0164 against the windows' own 0.0071.

    The cap is per splat, tied to how far its centre sits from the nearest
    seam: sigma <= max(distance, floor) / k. A splat deep inside its slot
    keeps a long life; one near a seam is capped at floor/k, dimming into
    the seam while the neighbouring window's splats dim out of it, which is
    the crossfade the seam wants anyway. With k=3 a splat's alpha at its own
    slot edge is under 1.1% of peak.
    """
    if k <= 0 and cap_abs <= 0:
        return fields
    tc = fields["t_center"].astype(np.float64)
    d = np.full(tc.shape, np.inf)
    if np.isfinite(lo):
        d = np.minimum(d, np.abs(tc - lo))
    if np.isfinite(hi):
        d = np.minimum(d, np.abs(hi - tc))
    cap = np.maximum(d, floor) / k if k > 0 else np.full(tc.shape, np.inf)
    if cap_abs > 0:
        cap = np.minimum(cap, cap_abs)
    ts = fields["t_sigma"]
    sign = np.where(ts < 0, -1.0, 1.0)          # sigma is used as |sigma|
    fields["t_sigma"] = (sign * np.minimum(np.abs(ts), cap)).astype(np.float32)
    return fields


def apply_weight(fields, weight):
    """Scale alpha by `weight` in logit space.

    sigmoid(logit) * w == sigmoid(logit + log(w/(1-w*sigmoid))) has no tidy
    closed form, so scale the probability directly and convert back, which
    is exact."""
    alpha = 1.0 / (1.0 + np.exp(-fields["opacity"].astype(np.float64)))
    scaled = np.clip(alpha * weight, 1e-13, 1.0 - 1e-7)
    fields["opacity"] = np.log(scaled / (1.0 - scaled)).astype(np.float32)
    return fields


def merge(segments, seams, mode, fade, tail_k=2.0, tail_floor=0.4, tail_cap=0.0):
    """Concatenated field arrays plus a per-segment report."""
    bounds = slot_bounds(segments, seams)
    kept, report = [], []
    for seg, (lo, hi) in zip(segments, bounds):
        if mode == "hard":
            mask, weight = select_hard(seg, lo, hi)
        else:
            mask, weight = select_fade(seg, lo, hi, fade)
        fields = {k: (v[mask] if v.ndim == 1 else v[mask, :])
                  for k, v in seg["fields"].items()}
        if weight is not None:
            fields = apply_weight(fields, weight[mask])
        before = np.abs(fields["t_sigma"]).astype(np.float64)
        fields = gate_tails(fields, lo, hi, tail_k, tail_floor, tail_cap)
        capped = int((np.abs(fields["t_sigma"]) < before - 1e-9).sum())
        kept.append(fields)
        report.append({
            "path": seg["path"], "offset": seg["offset"],
            "covers": [seg["t_lo"], seg["t_hi"]],
            "slot": [None if not np.isfinite(lo) else lo,
                     None if not np.isfinite(hi) else hi],
            "splats_in": seg["count"], "splats_kept": int(mask.sum()),
            "sigma_capped": capped,
        })

    keys = set(kept[0])
    for f in kept[1:]:
        if set(f) != keys:
            raise SystemExit(
                "segments carry different field sets, so they were baked "
                "with different settings and cannot be concatenated: "
                f"{sorted(keys ^ set(f))}")
    merged = {k: np.concatenate([f[k] for f in kept], axis=0) for k in keys}
    return merged, report


def snap_time_centers(merged):
    """Move every t_center onto the codebook the packer will use, and slide
    xyz to match, so quantizing time costs no position error.

    The archive stores t_center as an index into a 256-entry codebook fit
    over the whole file. Four merged windows spread those 256 codes across
    four times the range, which measured 0.005 s of error against 0.001 s
    for a window alone. Position is xyz + v * (t - t_center), so that error
    displaces a splat by |v| * 0.005 s: a median 3.9% of the splat's own
    size and 33% at the 95th percentile.

    Rewriting xyz by v * (t_center_new - t_center_old) leaves the position
    at every instant exactly where it was, so the snap is free. Only the
    alpha envelope shifts, by well under a typical t_sigma.
    """
    tc = merged["t_center"].astype(np.float64)
    codebook, idx = kmeans_1d(tc)
    snapped = codebook[idx]
    delta = (snapped - tc).astype(np.float32)
    for axis, vel in (("x", "vx"), ("y", "vy"), ("z", "vz")):
        merged[axis] = (merged[axis] + merged[vel] * delta).astype(np.float32)
    merged["t_center"] = snapped.astype(np.float32)
    return merged, float(np.abs(delta).mean())


def to_pack_fields(merged):
    """Field arrays in the shape pack_sogst wants.

    decode_sogst_fields returns the higher-order spherical harmonics as a
    single 'f_rest' array of [N, 45], NOT as f_rest_0..f_rest_44. Looking
    for the split names finds nothing, silently drops every view-dependent
    coefficient, and produces a flat-shaded model: that mistake cost 0.0186
    LPIPS against 0.0071 on the first attempt at this merge. Carry the block
    through, and refuse to write a model that lost SH the inputs had."""
    fields = {k: merged[k].astype(np.float32) for k in SOGST_FIELDS}
    if "f_rest" in merged:
        fields["f_rest"] = np.asarray(merged["f_rest"], dtype=np.float32)
    else:
        split = sorted((k for k in merged if k.startswith("f_rest_")),
                       key=lambda k: int(k.split("_")[-1]))
        if split:
            fields["f_rest"] = np.stack([merged[k] for k in split],
                                        axis=1).astype(np.float32)
    return fields


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", nargs=2, action="append", default=None,
                    metavar=("SOGST", "OFFSET_SECONDS"),
                    help="a window and its start in global time; repeat, in order")
    ap.add_argument("--plan", default=None,
                    help="a window_plan.json from plan_temporal_windows.py, "
                         "supplying the segments, their offsets and the seams "
                         "instead of repeating --segment")
    ap.add_argument("--plan_model", default="splat_4d.sogst",
                    help="model filename inside each window's out_dir "
                         "(--plan only, default splat_4d.sogst)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["hard", "fade"], default="fade")
    ap.add_argument("--fade", type=float, default=0.35,
                    help="fade zone width in seconds (--mode fade). The default "
                         "0.35 was measured: LPIPS is best there and worse by "
                         "1 s, while PSNR keeps improving out to 1 s because a "
                         "wide fade blurs across the seam")
    ap.add_argument("--tail_k", type=float, default=2.0,
                    help="cap each splat's temporal sigma at (distance from its "
                         "centre to the nearest seam) / TAIL_K, so it cannot "
                         "render outside its slot; 0 disables the cap")
    ap.add_argument("--tail_floor", type=float, default=0.4,
                    help="floor on that distance in seconds, so splats sitting "
                         "on a seam keep a usable life; default 0.4")
    ap.add_argument("--no_snap_time", action="store_true",
                    help="skip moving t_center onto the packer's codebook; the "
                         "snap is exact for position, so this is a debug switch")
    ap.add_argument("--tail_cap", type=float, default=0.0,
                    help="an absolute ceiling in seconds on temporal sigma, "
                         "applied on top of the seam-distance rule; 0 disables")
    ap.add_argument("--seams", default=None,
                    help="comma-separated seam times, overriding the midpoints "
                         "between adjacent windows")
    ap.add_argument("--fps", type=float, default=None,
                    help="output frame rate (default: the first segment's)")
    ap.add_argument("--report_json", default=None)
    args = ap.parse_args()

    if not args.segment and not args.plan:
        raise SystemExit("give --segment entries or a --plan")
    plan_seams = []
    if args.plan:
        pairs, plan_seams = segments_from_plan(args.plan, args.plan_model)
        if args.segment:
            raise SystemExit("--plan and --segment both given; use one")
    else:
        pairs = [(p, float(o)) for p, o in args.segment]
    segments = [load_segment(p, float(o)) for p, o in pairs]
    if len(segments) < 2:
        raise SystemExit("give at least two segments to merge")
    order = np.argsort([s["t_lo"] for s in segments])
    segments = [segments[i] for i in order]

    if args.seams:
        seams = [float(x) for x in args.seams.split(",")]
    elif plan_seams:
        seams = plan_seams
    else:
        seams = default_seams(segments)
    if len(seams) != len(segments) - 1:
        raise SystemExit(f"{len(segments)} segments need {len(segments)-1} seams, "
                         f"got {len(seams)}")

    merged, report = merge(segments, seams, args.mode, args.fade,
                           args.tail_k, args.tail_floor, args.tail_cap)
    if not args.no_snap_time:
        merged, moved = snap_time_centers(merged)
        print(f"  snapped t_center to the packer's codebook, mean move "
              f"{moved*1000:.2f} ms, positions compensated")
    fields = to_pack_fields(merged)
    # Guard the failure mode above: if the inputs carried view-dependent SH,
    # the output must too, or the merge quietly flattens the model.
    inputs_have_sh = any("f_rest" in s["fields"] or
                         any(k.startswith("f_rest_") for k in s["fields"])
                         for s in segments)
    if inputs_have_sh and "f_rest" not in fields:
        raise SystemExit(
            "the segments carry higher-order spherical harmonics but the "
            "merged fields do not: the merge would write a flat-shaded model. "
            "This is a bug in to_pack_fields, not in the inputs.")
    t_min = min(s["t_lo"] for s in segments)
    t_max = max(s["t_hi"] for s in segments)
    fps = args.fps or next((s["fps"] for s in segments if s["fps"]), 24.0)

    print(f"merging {len(segments)} segments, mode {args.mode}")
    for r in report:
        slot = ("(-inf" if r["slot"][0] is None else f"[{r['slot'][0]:.4f}")
        slot += ", inf)" if r["slot"][1] is None else f", {r['slot'][1]:.4f})"
        print(f"  {Path(r['path']).parent.name:<12} covers "
              f"[{r['covers'][0]:.4f}, {r['covers'][1]:.4f}]  slot {slot:<22} "
              f"kept {r['splats_kept']:>9,} of {r['splats_in']:>9,}  "
              f"sigma capped on {r['sigma_capped']:>8,}")
    print(f"  seams: {', '.join(f'{s:.4f}' for s in seams)}")
    print(f"  merged {len(fields['x']):,} splats over [{t_min:.3f}, {t_max:.3f}] s "
          f"at {fps:g} fps")

    meta = pack_sogst(args.out, fields, t_min, t_max, fps)
    size = Path(args.out).stat().st_size / 1e6
    print(f"  wrote {args.out} ({size:.1f} MB, {meta['count']:,} splats after packing)")

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(
            {"mode": args.mode, "fade": args.fade, "seams": seams,
             "tail_k": args.tail_k, "tail_floor": args.tail_floor,
             "tail_cap": args.tail_cap,
             "time": [t_min, t_max], "fps": fps,
             "splats_out": int(meta["count"]), "megabytes": size,
             "segments": report}, indent=2))


if __name__ == "__main__":
    main()
