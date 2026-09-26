#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""index_takes.py - pair one session's clips across cameras into takes.

WHY
---
A capture day leaves one folder per camera (``<camera_id>/GX01NNNN.MP4``) with
unequal clip counts: cameras miss takes, stop early, or start a take a few
seconds late. Clip numbers are per-camera counters and collide across
cameras, so a clip is only identified by ``<camera_id>/<file>``. Before sync
(compute_sync_offsets.py) can run, each take's clips have to be grouped.

WHICH CLOCK
-----------
The GoPro TMCD ``timecode`` tag is NOT usable for this. On HERO13 400fps
footage the per-camera timecode difference drifted from ~4 s to ~10 s over one
session (it does not advance at wall-clock rate in high-frame-rate modes).
The container ``creation_time`` is wall-clock with 1 s resolution, and on the
2026-09-26 session all cameras agreed to within ~3 s. Clips are clustered on
it: a clip joins the current take when its [start, start+duration] interval
overlaps the interval every clip already in the take shares by at least
--min_overlap seconds (a late-starting camera still belongs to the take as
long as sync has footage to work with). Sub-second alignment is left to audio
cross-correlation.

WHAT IT WRITES
--------------
    <out_dir>/takes.json          every take: start time, per-camera clip,
                                  duration, frames, fps, and the wall-clock
                                  interval all its cameras share
    <out_dir>/takes/<take>/<camera_id>.mp4
                                  symlinks, one dir per take, laid out as
                                  compute_sync_offsets.py's movies dir
                                  (only with --link, only takes with at least
                                  --min_cameras cameras)

Filters: --min_fps drops clips below a frame rate (e.g. 300 to keep only the
400fps session and ignore 120fps clips in the same folders); --date keeps only
clips whose creation_time starts with that YYYY-MM-DD.

Usage:
    python3 index_takes.py /media/Exos/datasets/2026_uci_grandprix \
        /media/Exos/datasets/2026_uci_grandprix/pipeline_run2 \
        --cameras 1959 2390 2442 5564 --date 2026-09-26 --min_fps 300 --link

conda env: cumuli (stdlib + ffprobe on PATH).
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from image_formats import SUPPORTED_VIDEO_EXTS


def probe(path: Path) -> dict:
    """Return the fields index_takes needs from one clip via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height,r_frame_rate,nb_frames:format=duration:format_tags=creation_time",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in video["r_frame_rate"].split("/"))
    created = info["format"].get("tags", {}).get("creation_time")
    if created is None:
        raise ValueError(f"{path}: no creation_time tag")
    return {
        "creation_time": created,
        "start": datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp(),
        "duration": float(info["format"]["duration"]),
        "fps": num / den,
        "frames": int(video.get("nb_frames", 0)),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def shared_interval(members: list[dict]) -> tuple[float, float]:
    """Wall-clock interval every clip in ``members`` covers (may be empty)."""
    return (max(c["start"] for c in members),
            min(c["start"] + c["duration"] for c in members))


def cluster(clips: list[dict], min_overlap: float) -> list[list[dict]]:
    """Greedy overlap clustering; a camera appears at most once per take."""
    takes: list[list[dict]] = []
    for clip in sorted(clips, key=lambda c: c["start"]):
        current = takes[-1] if takes else None
        if current is not None:
            _, end = shared_interval(current)
            overlap = end - clip["start"]
        if (current is not None
                and overlap >= min_overlap
                and all(c["camera"] != clip["camera"] for c in current)):
            current.append(clip)
        else:
            takes.append([clip])
    return takes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path, help="dir holding one sub-folder per camera")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--cameras", nargs="+", required=True, help="camera folder names")
    ap.add_argument("--date", help="keep only clips whose creation_time starts with YYYY-MM-DD")
    ap.add_argument("--min_fps", type=float, default=0.0)
    ap.add_argument("--min_overlap", type=float, default=3.0,
                    help="min seconds a clip must overlap the take's shared interval to join it (default 3)")
    ap.add_argument("--min_cameras", type=int, default=2, help="min cameras for --link dirs (default 2)")
    ap.add_argument("--link", action="store_true", help="write takes/<take>/<camera>.mp4 symlinks")
    args = ap.parse_args()

    clips = []
    for cam in args.cameras:
        for path in sorted((args.session_dir / cam).iterdir()):
            if path.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
                continue
            meta = probe(path)
            if args.date and not meta["creation_time"].startswith(args.date):
                continue
            if meta["fps"] < args.min_fps:
                continue
            clips.append({"camera": cam, "file": path.name, "path": str(path.resolve()), **meta})
    if not clips:
        print("No clips matched the filters.", file=sys.stderr)
        return 1

    takes_out = []
    for i, members in enumerate(cluster(clips, args.min_overlap)):
        name = f"T{i:02d}"
        shared_start, shared_end = shared_interval(members)
        takes_out.append({
            "take": name,
            "creation_time": members[0]["creation_time"],
            "n_cameras": len(members),
            "missing_cameras": [c for c in args.cameras if c not in {m["camera"] for m in members}],
            # creation_time has 1 s resolution: treat as a coarse estimate only.
            "shared_seconds_estimate": round(max(0.0, shared_end - shared_start), 1),
            "clips": {m["camera"]: {k: m[k] for k in
                                    ("file", "path", "creation_time", "duration", "frames", "fps",
                                     "width", "height", "has_audio")}
                      for m in sorted(members, key=lambda m: m["camera"])},
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "takes.json").write_text(json.dumps({"takes": takes_out}, indent=2))

    print(f"{len(clips)} clips -> {len(takes_out)} takes ({args.out_dir / 'takes.json'})")
    for t in takes_out:
        cams = " ".join(f"{cam}/{c['file'][:-4]}" for cam, c in t["clips"].items())
        miss = f"  missing {','.join(t['missing_cameras'])}" if t["missing_cameras"] else ""
        print(f"  {t['take']} {t['creation_time'][11:19]}Z  {t['n_cameras']} cams  "
              f"~{t['shared_seconds_estimate']:>4.1f}s shared  {cams}{miss}")

    if args.link:
        for t in takes_out:
            if t["n_cameras"] < args.min_cameras:
                continue
            take_dir = args.out_dir / "takes" / t["take"]
            take_dir.mkdir(parents=True, exist_ok=True)
            for cam, c in t["clips"].items():
                link = take_dir / f"{cam}{Path(c['file']).suffix.lower()}"
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(c["path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
