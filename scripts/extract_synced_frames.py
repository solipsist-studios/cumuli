#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
extract_synced_frames.py

Extracts one frame (or a window of consecutive frames) from each camera at
the SAME aligned moment in time, using the frame offsets computed by
compute_sync_offsets.py. Used both to visually verify sync (via
make_sync_grid.py) and to pull the actual production frame you'll run
the rest of the pipeline on.

Optional per-camera color correction: pass --pp3_dir pointing at a
directory of RawTherapee .pp3 profiles. See color_correct.py for how
profiles are matched to cameras and applied. Color-corrected output is
PNG (downstream stages should pass --image_ext .png).

Multi-frame window: --window N extracts N consecutive frames per camera
into out_dir/f0/, out_dir/f1/, ... out_dir/f{N-1}/, each subdirectory
being a normal single-instant frames dir (same <camera>.jpg naming). Use
this to produce multi-instant candidates for run_pose_refinement.py --
each f{k}/ dir goes through undistort_frames.py and
predict_keypoints_2d.py independently.

Usage:
    python3 extract_synced_frames.py /path/to/movies/dir /path/to/sync_offsets.json /path/to/output/dir [seconds_into_clip] \\
        [--window N] [--pp3_dir /path/to/thumbs] [--rawtherapee_cmd "..."] [--output_ext .jpg|.jpeg|.png|.webp] [--bit_depth 8|16] [--jobs N]

    seconds_into_clip (optional, default 2.0) -- how far into the
    reference camera's clip to grab the frame from.

Output:
    window=1: output_dir/0001.jpg, output_dir/0002.jpg, ...
    window=N: output_dir/f0/0001.jpg ... output_dir/f{N-1}/0001.jpg, ...
    (.png instead of .jpg when --pp3_dir is used; override with --output_ext.
    --pp3_dir takes .png or .webp: RawTherapee always writes PNG, which .webp
    converts to LOSSLESS WebP and then removes the PNG. Colour-stage output is
    never lossy, so there is no quality flag here.)
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from color_correct import apply_pp3, find_pp3, resolve_rawtherapee_cmd
from image_formats import SUPPORTED_IMAGE_EXTS, ImageFormatError, add_bit_depth_argument, check_bit_depth, save_image


def ffmpeg_quality_args(ext: str):
    """Codec-appropriate near-lossless flags. ffmpeg's -q:v scale is
    codec-dependent -- low values mean high quality for mjpeg (.jpg/.jpeg),
    but libwebp interprets -q:v as a 0-100 quality score where low means
    LOW quality, so the same flag can't be reused across extensions."""
    if ext in (".jpg", ".jpeg"):
        return ["-q:v", "2"]
    if ext == ".webp":
        return ["-lossless", "1"]
    return []  # .png is already lossless, no quality flag needed


def ffmpeg_format_args(ext: str, bit_depth: int = 8):
    """Pixel format and PNG effort. A 10-bit source (yuv420p10le) is written as a 16-bit PNG
    (rgb48be, 69 MB at 5312x4648) unless told otherwise, and every later stage reads 8 bits, so
    PNG and WebP are written as 8-bit RGB by default: files about 4x smaller and a much faster
    encode. ``bit_depth=16`` (PNG only) keeps the 16-bit RGB, for a strong colour grade that
    follows. PNG compression level 1 is ~3x faster than ffmpeg's default for files about 10%
    larger. A .jpg keeps ffmpeg's own choice."""
    if ext == ".png":
        return ["-pix_fmt", "rgb48be" if bit_depth == 16 else "rgb24", "-compression_level", "1"]
    if ext == ".webp":
        return ["-pix_fmt", "rgb24"]
    return []


def extract_frames(video_path: Path, timestamp_sec: float, count: int, out_dir: Path, ext: str, bit_depth: int = 8):
    """Extract `count` consecutive frames starting at timestamp_sec.
    Returns the list of written paths (f0, f1, ... order)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{video_path.stem}_%02d{ext}"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{timestamp_sec:.6f}",
        "-i", str(video_path),
        "-frames:v", str(count),
        # For ".webp" ffmpeg 6.1 picks the animated encoder (libwebp_anim) by default, which
        # buffers every frame into ONE file whatever the %02d pattern says; naming libwebp,
        # the still-image encoder, writes one file per frame.
        *(["-c:v", "libwebp"] if ext == ".webp" else []),
        *ffmpeg_format_args(ext, bit_depth),
        *ffmpeg_quality_args(ext),
        # Name the muxer too, so the output pattern is never taken as a single animated file.
        "-f", "image2",
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
    add_bit_depth_argument(
        parser, help="Bits per channel of the written frames (default 8). 16 keeps the source's 16-bit RGB "
                     "(ffmpeg rgb48be) and is for .png only: about 4x the disk and a slower encode, only useful "
                     "when a strong colour grade follows. Not available with --pp3_dir.")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Cameras extracted at once (default: the number of cameras, at most 8). Each is one "
                             "mostly single-threaded ffmpeg run holding a frame in memory.")
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

    if rt_cmd and args.output_ext not in (None, ".png", ".webp"):
        print("Error: --output_ext other than .png or .webp isn't supported with --pp3_dir "
              "(RawTherapee always writes PNG; .webp converts it to lossless WebP).")
        sys.exit(1)
    out_ext = args.output_ext or (".png" if rt_cmd else ".jpg")
    try:
        check_bit_depth(out_ext, args.bit_depth)
    except ImageFormatError as e:
        print(f"Error: --bit_depth: {e}")
        sys.exit(1)
    if args.bit_depth == 16 and rt_cmd:
        print("Error: --bit_depth 16 isn't supported with --pp3_dir (RawTherapee is run for 8-bit PNG output here). "
              "Extract 16-bit frames, then colour-correct them with color_adjust.py.")
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(args.offsets_path) as f:
            sync_data = json.load(f)
        offsets = sync_data["offsets"]
        reference_camera = sync_data["reference_camera"]
    except json.JSONDecodeError as e:
        print(f"Error: {args.offsets_path} is not valid JSON: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: {args.offsets_path} is missing expected key {e}")
        sys.exit(1)

    stems = {}
    for name in offsets:
        stems.setdefault((args.movies_dir / name).stem, []).append(name)
    collisions = {stem: names for stem, names in stems.items() if len(names) > 1}
    if collisions:
        print("Error: multiple camera files share the same output filename stem "
              "(they would silently overwrite each other's extracted frame):")
        for stem, names in collisions.items():
            print(f"  {stem!r}: {', '.join(names)}")
        sys.exit(1)

    print(f"Reference camera: {reference_camera}")
    print(f"Extracting {args.window} frame(s) at {args.seconds_into_clip:.2f}s (reference camera time) for all cameras...\n")

    def process_camera(name, data):
        """Extract one camera's frame(s). Returns (ok, log lines); the caller prints a camera's lines together."""
        video_path = args.movies_dir / name
        if not video_path.is_file():
            return False, [f"  WARNING: {video_path} not found, skipping"]

        if "error" in data:
            return False, [f"  WARNING: {name} has no sync data (failed in compute_sync_offsets.py: {data['error']}), skipping"]

        fps = data["fps"]
        frame_offset = data["frame_offset"]

        # Each camera's clip starts frame_offset frames LATER than the
        # reference. So to land on the same real-world moment, we seek
        # to (seconds_into_clip - frame_offset/fps) into THIS camera's
        # own clip.
        seek_time = args.seconds_into_clip - (frame_offset / fps)

        if seek_time < 0:
            return False, [f"  WARNING: {name} starts too late to extract a frame at "
                           f"{args.seconds_into_clip:.2f}s reference time (would need negative seek). "
                           f"Try a larger seconds_into_clip value. Skipping."]

        log = []
        pp3 = find_pp3(args.pp3_dir, video_path.stem) if rt_cmd else None
        if rt_cmd and pp3 is None:
            log.append(f"  WARNING: no .pp3 profile matches camera {video_path.stem!r}; writing uncorrected frames.")

        try:
            with tempfile.TemporaryDirectory() as tmp:
                raw_paths = extract_frames(video_path, seek_time, args.window, Path(tmp), ".png" if pp3 else out_ext,
                                           **({"bit_depth": 16} if args.bit_depth == 16 else {}))
                for k, raw_path in enumerate(raw_paths):
                    if args.window == 1:
                        out_path = args.output_dir / f"{video_path.stem}{out_ext}"
                    else:
                        out_path = args.output_dir / f"f{k}" / f"{video_path.stem}{out_ext}"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                    if pp3 is not None and out_ext == ".png":
                        apply_pp3(rt_cmd, raw_path, pp3, out_path)
                    elif pp3 is not None:
                        # RawTherapee writes PNG whatever the extension: render, then re-encode lossless.
                        rendered = out_path.with_name(out_path.stem + ".rt.png")
                        try:
                            apply_pp3(rt_cmd, raw_path, pp3, rendered)
                            with Image.open(rendered) as im:
                                save_image(im, out_path)
                        finally:
                            rendered.unlink(missing_ok=True)
                    else:
                        shutil.move(str(raw_path), str(out_path))
            cc = f" + pp3 {pp3.name}" if pp3 else ""
            log.append(f"  {name}: seeking to {seek_time:.3f}s, {args.window} frame(s){cc}")
            return True, log
        except (subprocess.CalledProcessError, RuntimeError) as e:
            return False, log + [f"  ERROR extracting frames from {name}: {e}"]

    # The cameras are independent ffmpeg runs and one encode is mostly single-threaded (a 5K PNG is
    # seconds on one core), so several at a time is close to linear. Results are reported in camera
    # order, each camera's lines together.
    jobs = max(1, min(args.jobs or min(len(offsets), 8), max(len(offsets), 1)))
    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [(name, pool.submit(process_camera, name, data)) for name, data in offsets.items()]
        for name, future in futures:
            ok, lines = future.result()
            for line in lines:
                print(line, flush=True)
            if not ok:
                failed.append(name)

    print(f"\nDone. Extracted frames are in {args.output_dir}")
    if failed:
        print(f"Failed/skipped cameras: {failed}")
        sys.exit(1)
    if args.window == 1:
        print("Open them side by side -- if sync is correct, all frames should")
        print("show the same instant of action/pose across every camera.")
    else:
        print(f"Per-instant subdirs f0/..f{args.window - 1}/ -- run undistort_frames.py + "
              "predict_keypoints_2d.py on each to feed run_pose_refinement.py.")


if __name__ == "__main__":
    main()
