#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
compare_experiments.py - tabulate synthetic pipeline runs side by side.

Reads the experiment.json each run of run_synthetic_pipeline.py writes and
prints one row per run, so a set of camera configurations can be read as a
table rather than by opening files one at a time.

Rows are ordered by LPIPS, best first. On a masked subject most of the
frame is empty background that every model renders perfectly, so PSNR is
dominated by pixels no camera configuration affects and compresses the
differences that matter. PSNR and SSIM are still shown, and a run whose
PSNR ranking disagrees with its LPIPS ranking is flagged, because that
disagreement is worth looking at rather than averaging away.

Comparisons are only meaningful between runs that scored the same views.
Runs whose eval camera count or frame window differs from the first row are
marked, rather than silently ranked against it.

Usage:
    python3 scripts/compare_experiments.py ~/runs/ariana_*
    python3 scripts/compare_experiments.py --json ~/runs/*/experiment.json
"""

import argparse
import json
import sys
from pathlib import Path


def find_records(paths):
    """Accept run directories, experiment.json files, or a mix."""
    records = []
    for raw in paths:
        path = Path(raw).expanduser()
        candidates = []
        if path.is_dir():
            direct = path / "experiment.json"
            candidates = [direct] if direct.is_file() else sorted(
                path.glob("*/experiment.json"))
        elif path.is_file():
            candidates = [path]
        if not candidates:
            print(f"  skipping {path}: no experiment.json found", file=sys.stderr)
            continue
        for cand in candidates:
            try:
                records.append((cand, json.loads(cand.read_text())))
            except ValueError as e:
                print(f"  skipping {cand}: {e}", file=sys.stderr)
    return records


def rig_summary(record):
    spec = record.get("rig_spec") or {}
    layout = spec.get("layout", "?")
    if layout == "rings":
        counts = "+".join(str(r.get("count", "?")) for r in spec.get("rings", []))
        return f"rings {counts}"
    if layout == "cage":
        cage = spec.get("cage", {})
        return f"cage {cage.get('theta', '?')}x{cage.get('phi', '?')}"
    if layout == "explicit":
        return f"explicit {len(spec.get('cameras', []))}"
    return layout


def row_for(path, record):
    evaluation = record.get("eval") or {}
    poses = record.get("poses", "gt")
    pose_error = None
    scores = record.get("pose_scores")
    if scores:
        for source in scores.get("sources", []):
            if source["source"] == ("refined" if poses == "refined" else "hloc"):
                pose_error = source["position_error_m"]["median"] * 1000.0
    frames = record.get("frames") or {}
    render = record.get("render") or {}
    return {
        "run": record.get("run") or path.parent.name,
        "rig": rig_summary(record),
        "cameras": render.get("train_cameras"),
        "eval_views": evaluation.get("views"),
        "frames": frames.get("count"),
        "poses": poses,
        "lpips": evaluation.get("lpips"),
        "psnr_db": evaluation.get("psnr_db"),
        "ssim": evaluation.get("ssim"),
        "pose_err_mm": pose_error,
        "render_s": render.get("seconds"),
        "path": str(path),
    }


def comparable(rows):
    """Flag rows that did not score the same thing as the first row."""
    if not rows:
        return rows
    base = rows[0]
    for row in rows:
        row["comparable"] = (row["eval_views"] == base["eval_views"]
                             and row["frames"] == base["frames"])
    return rows


def rank_disagreements(rows):
    """Runs whose PSNR order contradicts their LPIPS order."""
    scored = [r for r in rows if r["lpips"] is not None and r["psnr_db"] is not None]
    by_lpips = sorted(scored, key=lambda r: r["lpips"])
    by_psnr = sorted(scored, key=lambda r: -r["psnr_db"])
    lpips_rank = {r["run"]: i for i, r in enumerate(by_lpips)}
    psnr_rank = {r["run"]: i for i, r in enumerate(by_psnr)}
    return [r["run"] for r in scored
            if lpips_rank[r["run"]] != psnr_rank[r["run"]]]


def fmt(value, spec, dash="-"):
    return dash if value is None else format(value, spec)


def print_table(rows):
    header = (f"{'run':<24} {'rig':<14} {'cams':>5} {'eval':>5} {'frm':>4} "
              f"{'poses':<8} {'LPIPS':>8} {'PSNR dB':>8} {'SSIM':>7} "
              f"{'pose mm':>8}")
    print(header)
    print("-" * len(header))
    for row in rows:
        mark = "" if row.get("comparable", True) else "  <- different eval setup"
        print(f"{row['run'][:24]:<24} {row['rig'][:14]:<14} "
              f"{fmt(row['cameras'], 'd'):>5} {fmt(row['eval_views'], 'd'):>5} "
              f"{fmt(row['frames'], 'd'):>4} {row['poses']:<8} "
              f"{fmt(row['lpips'], '.4f'):>8} {fmt(row['psnr_db'], '.2f'):>8} "
              f"{fmt(row['ssim'], '.4f'):>7} {fmt(row['pose_err_mm'], '.1f'):>8}"
              f"{mark}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="Run directories, or experiment.json files")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="Emit the table as JSON instead of text")
    ap.add_argument("--sort", choices=["lpips", "psnr", "ssim", "run"],
                    default="lpips")
    args = ap.parse_args()

    records = find_records(args.paths)
    if not records:
        sys.exit("no experiment.json files found")
    rows = [row_for(path, record) for path, record in records]

    keys = {
        "lpips": lambda r: (r["lpips"] is None, r["lpips"] or 0.0),
        "psnr": lambda r: (r["psnr_db"] is None, -(r["psnr_db"] or 0.0)),
        "ssim": lambda r: (r["ssim"] is None, -(r["ssim"] or 0.0)),
        "run": lambda r: r["run"],
    }
    rows.sort(key=keys[args.sort])
    rows = comparable(rows)

    if args.as_json:
        print(json.dumps(rows, indent=2))
        return

    print_table(rows)
    unscored = [r["run"] for r in rows if r["lpips"] is None]
    if unscored:
        print(f"\n{len(unscored)} run(s) carry no eval scores yet: "
              f"{', '.join(unscored[:6])}")
    disagree = rank_disagreements(rows)
    if disagree:
        print(f"\nPSNR and LPIPS rank these runs differently: "
              f"{', '.join(disagree)}. LPIPS is the ordering above; the "
              "disagreement usually means the runs differ in background "
              "pixels rather than in subject quality.")
    if any(not r.get("comparable", True) for r in rows):
        print("\nRows marked above scored a different number of eval views or "
              "frames than the first row, so their scores are not directly "
              "comparable with it.")


if __name__ == "__main__":
    main()
