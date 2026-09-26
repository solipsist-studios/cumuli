#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""make_sync_review.py - a trimmed, synced 2x2 review video per take, plus the
human sign-off that gates everything downstream of sync.

WHY
---
Audio cross-correlation (compute_sync_offsets.py) can silently lock onto a
wrong lag, and on outdoor race footage it routinely reports LOW CONFIDENCE.
A single sync-grid still (make_sync_grid.py) shows one instant; at 400fps a
reviewer needs motion to judge a few-frame error. This renders every camera
trimmed to the wall-clock interval they all share, stacked in one grid, each
tile labelled with its camera, offset and the SYNCED frame number (identical
across tiles by construction), slowed down so a one-frame (2.5 ms) error is
visible as a wheel or pedal out of phase.

GATE
----
Each render registers its take in <review_json> (default: sync_review.json
next to the take's sync dir) as "pending", recording the offsets it rendered.
A reviewer flips a take to "approved" (or "rejected", with a note, or edits
the offsets and re-renders). Downstream stages call `require_approved()`,
which refuses a take that is not approved or whose sync_offsets.json changed
after the approval, so an approval never outlives the offsets it saw.

    python3 make_sync_review.py approve  <review_json> T08 [--note "..."]
    python3 make_sync_review.py reject   <review_json> T08 --note "1959 ~3 frames late"
    python3 make_sync_review.py status   <review_json>

OFFSETS
-------
Sign convention is compute_sync_offsets.py's: a positive offset_seconds means
that camera started LATER than the reference, so reference time t sits at
(t - offset) in that camera's clip. frame_offset / fps is used when present
(hand-tuned files edit frame_offset), else offset_seconds.

Usage:
    python3 make_sync_review.py render <take_movies_dir> <sync_offsets.json> <out.mp4> \
        [--take T08] [--review_json path] [--slowdown 10] [--max_seconds 0] [--tile_width 640]

--max_seconds 0 renders the whole shared interval. Output is H.264 (NVENC when
available) at source_fps/slowdown.

conda env: cumuli (ffprobe/ffmpeg with drawtext + xstack on PATH).
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _probe(path: Path) -> tuple[float, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    num, den = (int(x) for x in info["streams"][0]["r_frame_rate"].split("/"))
    return num / den, float(info["format"]["duration"])


def _load_review(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"takes": {}}


def _save_review(path: Path, review: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(review, indent=2))


def require_approved(review_json: Path, take: str, sync_json: Path) -> None:
    """Raise unless `take` is approved against the current `sync_json` contents."""
    entry = _load_review(review_json)["takes"].get(take)
    if entry is None:
        raise SystemExit(f"{take}: no sync review registered in {review_json}; run make_sync_review.py render")
    if entry["status"] != "approved":
        raise SystemExit(f"{take}: sync review is '{entry['status']}', not approved ({review_json})")
    if entry["sync_sha"] != _file_sha(sync_json):
        raise SystemExit(f"{take}: {sync_json} changed after approval; re-render and re-review")


def render(args) -> int:
    sync = json.loads(args.sync_json.read_text())
    offsets = sync["offsets"]
    cams = []
    for name, data in sorted(offsets.items()):
        path = args.movies_dir / name
        fps, duration = _probe(path)
        if data.get("frame_offset") is not None:
            off = data["frame_offset"] / fps
        else:
            off = float(data.get("offset_seconds", 0.0))
        cams.append({"name": name, "path": path, "fps": fps, "duration": duration, "off": off,
                     "frame_offset": data.get("frame_offset"),
                     "low_conf": bool(data.get("low_confidence") or data.get("flags"))})
    if len(cams) > 4:
        raise SystemExit("make_sync_review renders at most 4 cameras (2x2)")

    fps = cams[0]["fps"]
    # Shared interval on the reference clock: camera c's clip covers [off_c, off_c + dur_c].
    t0 = max(c["off"] for c in cams)
    t1 = min(c["off"] + c["duration"] for c in cams)
    if t1 - t0 <= 0:
        raise SystemExit(f"No shared interval under these offsets ({t0:.3f}..{t1:.3f}s)")
    length = t1 - t0 if args.max_seconds <= 0 else min(t1 - t0, args.max_seconds)

    tw = args.tile_width
    th = round(tw * 9 / 16 / 2) * 2
    out_fps = fps / args.slowdown
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    filters = []
    for i, c in enumerate(cams):
        cmd += ["-ss", f"{t0 - c['off']:.6f}", "-t", f"{length:.6f}", "-i", str(c["path"])]
        label = f"{Path(c['name']).stem}  off {c['frame_offset'] if c['frame_offset'] is not None else round(c['off'] * fps):+d}f"
        filters.append(
            f"[{i}:v]setpts=PTS-STARTPTS,scale={tw}:{th},"
            f"drawtext=fontfile={FONT}:text='{label}':x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,"
            f"drawtext=fontfile={FONT}:text='f %{{n}}':x=8:y=h-34:fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.6"
            f"[v{i}]")
    n = len(cams)
    layout = ["0_0", "w0_0", "0_h0", "w0_h0"][:n]
    stack_in = "".join(f"[v{i}]" for i in range(n))
    if n == 1:
        filters.append(f"[v0]setpts=N/({out_fps}*TB)[out]")
    else:
        filters.append(f"{stack_in}xstack=inputs={n}:layout={'|'.join(layout)}:fill=black,"
                       f"setpts=N/({out_fps}*TB)[out]")
    encoder = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"] if _has_nvenc() else \
              ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-r", f"{out_fps:g}",
            *encoder, "-pix_fmt", "yuv420p", str(args.out)]
    subprocess.run(cmd, check=True)

    take = args.take or args.movies_dir.name
    review_json = args.review_json or args.sync_json.parent.parent / "sync_review.json"
    review = _load_review(review_json)
    review["takes"][take] = {
        "status": "pending",
        "video": str(args.out.resolve()),
        "sync_json": str(args.sync_json.resolve()),
        "sync_sha": _file_sha(args.sync_json),
        "offsets_frames": {Path(c["name"]).stem: round(c["off"] * fps) for c in cams},
        "shared_seconds": round(t1 - t0, 3),
        "rendered_seconds": round(length, 3),
        "slowdown": args.slowdown,
        "rendered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "",
    }
    _save_review(review_json, review)
    print(f"{take}: {length:.2f}s shared -> {args.out} ({args.slowdown}x slow, {length * args.slowdown:.0f}s playback); "
          f"registered pending in {review_json}")
    return 0


def _has_nvenc() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "h264_nvenc" in enc


def set_status(args, status: str) -> int:
    review = _load_review(args.review_json)
    entry = review["takes"].get(args.take)
    if entry is None:
        raise SystemExit(f"{args.take} not registered in {args.review_json}")
    if status == "approved" and entry["sync_sha"] != _file_sha(Path(entry["sync_json"])):
        raise SystemExit(f"{args.take}: offsets changed since the render; re-render before approving")
    entry["status"] = status
    entry["note"] = args.note or entry.get("note", "")
    entry["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save_review(args.review_json, review)
    print(f"{args.take}: {status}")
    return 0


def status(args) -> int:
    for take, e in sorted(_load_review(args.review_json)["takes"].items()):
        print(f"{take}  {e['status']:<9} {e['shared_seconds']:>6.2f}s  offsets {e['offsets_frames']}  {e.get('note', '')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("movies_dir", type=Path)
    r.add_argument("sync_json", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--take")
    r.add_argument("--review_json", type=Path)
    r.add_argument("--slowdown", type=float, default=10.0)
    r.add_argument("--max_seconds", type=float, default=0.0)
    r.add_argument("--tile_width", type=int, default=640)
    for name in ("approve", "reject"):
        p = sub.add_parser(name)
        p.add_argument("review_json", type=Path)
        p.add_argument("take")
        p.add_argument("--note", default="")
    s = sub.add_parser("status")
    s.add_argument("review_json", type=Path)
    args = ap.parse_args()
    if args.cmd == "render":
        return render(args)
    if args.cmd in ("approve", "reject"):
        return set_status(args, "approved" if args.cmd == "approve" else "rejected")
    return status(args)


if __name__ == "__main__":
    sys.exit(main())
