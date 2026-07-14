#!/usr/bin/env python3
"""
extract_synced_frames.py

Extracts one frame (or a window of consecutive frames) from each camera at
the SAME aligned moment in time, using the frame offsets computed by
compute_sync_offsets.py (or corrected by skeleton_sync_search.py).
Used both to visually verify sync (via make_sync_grid.py) and to pull
the actual production frame you'll run the rest of the pipeline on.

Optional per-camera color correction: pass --pp3_dir pointing at a
directory of RawTherapee .pp3 profiles. See color_correct.py for how
profiles are matched to cameras and applied. Color-corrected output is
PNG (downstream stages should pass --image_ext .png).

Multi-frame window: --window N extracts N consecutive frames per camera
into out_dir/f0/, out_dir/f1/, ... out_dir/f{N-1}/, each subdirectory
being a normal single-instant frames dir (same <camera>.jpg naming). Use
this to produce candidate instants for skeleton_sync_search.py and
multi-instant keypoints for refine_poses_with_keypoints.py -- each
f{k}/ dir goes through undistort_frames.py and predict_keypoints_2d.py
independently.

Usage:
    python3 extract_synced_frames.py /path/to/movies/dir /path/to/sync_offsets.json /path/to/output/dir [seconds_into_clip] \\
        [--window N] [--pp3_dir /path/to/thumbs] [--rawtherapee_cmd "..."] [--output_ext .jpg|.jpeg|.png|.webp]

    seconds_into_clip (optional, default 2.0) -- how far into the
    reference camera's clip to grab the frame from.

Output:
    window=1: output_dir/0001.jpg, output_dir/0002.jpg, ...
    window=N: output_dir/f0/0001.jpg ... output_dir/f{N-1}/0001.jpg, ...
    (.png instead of .jpg when --pp3_dir is used; override with --output_ext,
    though --pp3_dir output can only be .png -- RawTherapee always writes PNG
    bytes here regardless of the requested extension)
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from color_correct import apply_pp3, find_pp3, resolve_rawtherapee_cmd
from image_formats import SUPPORTED_IMAGE_EXTS


def extract_frames(video_path: Path, timestamp_sec: float, count: int, out_dir: Path, ext: str):
    """Extract `count` consecutive frames starting at timestamp_sec.
    Returns the list of written paths (f0, f1, ... order)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{video_path.stem}_%02d{ext}"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{timestamp_sec:.6f}",
        "-i", str(video_path),
        "-frames:v", str(count),
        "-q:v", "2",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    paths = [out_dir / f"{video_path.stem}_{k:02d}{ext}" for k in range(1, count + 1)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f"ffmpeg wrote {count - len(missing)}/{count} frames from {video_path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("movies_dir", type=Path)
    parser.add_argument("offsets_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("seconds_into_clip", type=float, nargs="?", default=2.0)
    parser.add_argument("--window", type=int, default=1,
                        help="Number of consecutive frames to extract per camera (default 1). "
                             "N>1 writes into per-instant subdirs f0/..f{N-1}/.")
    parser.add_argument("--pp3_dir", type=Path, default=None,
                        help="Directory of per-camera RawTherapee .pp3 profiles; enables color correction (PNG output).")
    parser.add_argument("--rawtherapee_cmd", default=None,
                        help="Override the rawtherapee-cli invocation (default: rawtherapee-cli on PATH, else flatpak).")
    parser.add_argument("--output_ext", choices=SUPPORTED_IMAGE_EXTS, default=None,
                        help="Output image extension (default: .png with --pp3_dir, else .jpg).")
    args = parser.parse_args()

    if not args.movies_dir.is_dir():
        print(f"Error: {args.movies_dir} is not a directory")
        sys.exit(1)
    if not args.offsets_path.is_file():
        print(f"Error: {args.offsets_path} not found")
        sys.exit(1)

    rt_cmd = None
    if args.pp3_dir is not None:
        if not args.pp3_dir.is_dir():
            print(f"Error: --pp3_dir {args.pp3_dir} is not a directory")
            sys.exit(1)
        rt_cmd = resolve_rawtherapee_cmd(args.rawtherapee_cmd)
        print(f"Color correction enabled: {' '.join(rt_cmd)} (~6s/frame)")

    if rt_cmd and args.output_ext not in (None, ".png"):
        print("Error: --output_ext other than .png isn't supported with --pp3_dir "
              "(RawTherapee color correction always writes PNG bytes here).")
        sys.exit(1)
    out_ext = args.output_ext or (".png" if rt_cmd else ".jpg")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.offsets_path) as f:
        sync_data = json.load(f)

    offsets = sync_data["offsets"]
    reference_camera = sync_data["reference_camera"]

    print(f"Reference camera: {reference_camera}")
    print(f"Extracting {args.window} frame(s) at {args.seconds_into_clip:.2f}s (reference camera time) for all cameras...\n")

    for name, data in offsets.items():
        video_path = args.movies_dir / name
        if not video_path.is_file():
            print(f"  WARNING: {video_path} not found, skipping")
            continue

        if "error" in data:
            print(f"  WARNING: {name} has no sync data (failed in compute_sync_offsets.py: {data['error']}), skipping")
            continue

        fps = data["fps"]
        frame_offset = data["frame_offset"]

        # Each camera's clip starts frame_offset frames LATER than the
        # reference. So to land on the same real-world moment, we seek
        # to (seconds_into_clip - frame_offset/fps) into THIS camera's
        # own clip.
        seek_time = args.seconds_into_clip - (frame_offset / fps)

        if seek_time < 0:
            print(f"  WARNING: {name} starts too late to extract a frame at "
                  f"{args.seconds_into_clip:.2f}s reference time (would need negative seek). "
                  f"Try a larger seconds_into_clip value. Skipping.")
            continue

        pp3 = find_pp3(args.pp3_dir, video_path.stem) if rt_cmd else None
        if rt_cmd and pp3 is None:
            print(f"  WARNING: no .pp3 profile matches camera {video_path.stem!r}; writing uncorrected frames.")

        try:
            with tempfile.TemporaryDirectory() as tmp:
                raw_paths = extract_frames(video_path, seek_time, args.window, Path(tmp), ".png" if pp3 else out_ext)
                for k, raw_path in enumerate(raw_paths):
                    if args.window == 1:
                        out_path = args.output_dir / f"{video_path.stem}{out_ext}"
                    else:
                        out_path = args.output_dir / f"f{k}" / f"{video_path.stem}{out_ext}"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                    if pp3 is not None:
                        apply_pp3(rt_cmd, raw_path, pp3, out_path)
                    else:
                        shutil.move(str(raw_path), str(out_path))
            cc = f" + pp3 {pp3.name}" if pp3 else ""
            print(f"  {name}: seeking to {seek_time:.3f}s, {args.window} frame(s){cc}")
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"  ERROR extracting frames from {name}: {e}")

    print(f"\nDone. Extracted frames are in {args.output_dir}")
    if args.window == 1:
        print("Open them side by side -- if sync is correct, all frames should")
        print("show the same instant of action/pose across every camera.")
    else:
        print(f"Per-instant subdirs f0/..f{args.window - 1}/ -- run undistort_frames.py + "
              "predict_keypoints_2d.py on each to feed skeleton_sync_search.py / "
              "refine_poses_with_keypoints.py.")


if __name__ == "__main__":
    main()
