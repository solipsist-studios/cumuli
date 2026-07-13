#!/usr/bin/env python3
"""
01_compute_sync_offsets.py

Computes sync offsets across multiple camera clips using audio
cross-correlation (real signal matching), instead of relying on GoPro
internal timecode metadata (which can drift independently per camera
unless camera-to-camera time sync was explicitly used during capture).

Requires: numpy, scipy, ffmpeg (for audio extraction)

Usage:
    python3 01_compute_sync_offsets.py /path/to/movies/dir /path/to/output/dir [reference_camera_filename]

    reference_camera_filename (optional) -- which camera's audio to use
    as the reference everyone else is aligned to. If not given, the
    first clip found alphabetically is used.

Output:
    sync_offsets.json in the output directory:
    {
      "reference_camera": "...",
      "reference_fps": ...,
      "offsets": {
        "0001.mp4": {"timecode": null, "fps": ..., "frame_offset": ...},
        ...
      },
      "method": "audio_cross_correlation"
    }

Note: after generating sync_offsets.json, it is common to hand-tune a few
cameras' frame_offset by a couple of frames based on visual inspection
with 03_make_sync_grid.py, then save the result as sync_offsets_v2.json,
sync_offsets_v3.json, etc. Point 02_extract_synced_frames.py at whichever
version is your current best.

This script's own live-computed output has not itself been validated
unverified end-to-end -- every real run so far has used a precomputed,
hand-verified sync_offsets file (via 21_run_unified_pipeline.py's
--initial_sync_json) instead of trusting this script's raw output
directly. Always inspect the sync grid before trusting the result here.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate


SAMPLE_RATE = 16000  # downsample target -- plenty for sync correlation, much faster than 48kHz
CORRELATION_WINDOW_SECONDS = 30  # how much audio find_offset_seconds() actually compares
EXTRACT_SECONDS = 45  # correlation window + margin for slow-starting cameras; caps ffmpeg decode time
CONFIDENCE_THRESHOLD = 8  # below this, the correlation peak isn't clearly above the noise floor


def extract_audio_array(video_path: Path, tmp_dir: Path):
    """Extract mono audio from a video file as a numpy array at SAMPLE_RATE.

    Only decodes the first EXTRACT_SECONDS -- that's all find_offset_seconds()
    uses, and skipping it avoids paying full-length ffmpeg decode time on
    every clip.
    """
    wav_path = tmp_dir / f"{video_path.stem}.wav"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-t", str(EXTRACT_SECONDS),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)
    rate, data = wavfile.read(wav_path)
    wav_path.unlink()
    return data.astype(np.float32)


def get_fps(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    num, den = (int(x) for x in result.stdout.strip().split("/"))
    return num / den


def find_offset_seconds(reference_audio: np.ndarray, target_audio: np.ndarray, sample_rate: int):
    """
    Cross-correlate target_audio against reference_audio. Returns
    (offset_seconds, confidence) where confidence is the peak correlation
    value normalized against the average -- low confidence means the
    audio was too quiet/repetitive for a reliable match and the result
    should be treated with suspicion.

    Positive offset means target started LATER than reference (target
    needs to be trimmed forward by that many seconds to align).
    """
    max_samples = sample_rate * CORRELATION_WINDOW_SECONDS
    ref = reference_audio[:max_samples]
    tgt = target_audio[:max_samples]

    correlation = correlate(tgt, ref, mode="full")
    peak_idx = np.argmax(correlation)
    lag = peak_idx - (len(ref) - 1)

    peak_value = correlation[peak_idx]
    mean_abs = np.mean(np.abs(correlation)) + 1e-9
    confidence = float(peak_value / mean_abs)

    offset_seconds = lag / sample_rate
    return offset_seconds, confidence


def main():
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: python3 01_compute_sync_offsets.py "
            "/path/to/movies/dir /path/to/output/dir [reference_camera_filename]"
        )
        sys.exit(1)

    movies_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    forced_reference = sys.argv[3] if len(sys.argv) == 4 else None

    if not movies_dir.is_dir():
        print(f"Error: {movies_dir} is not a directory")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(movies_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"Error: no .mp4 files found in {movies_dir}")
        sys.exit(1)

    if forced_reference:
        ref_path = movies_dir / forced_reference
        if not ref_path.is_file():
            print(f"Error: reference camera {forced_reference} not found in {movies_dir}")
            sys.exit(1)
    else:
        ref_path = mp4_files[0]

    print(f"Reference camera: {ref_path.name}")
    print(f"Found {len(mp4_files)} clips. Extracting and correlating audio (this may take a bit)...\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        print(f"  Extracting reference audio from {ref_path.name}...")
        try:
            reference_audio = extract_audio_array(ref_path, tmp_dir)
            reference_fps = get_fps(ref_path)
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"Error: could not read reference camera {ref_path.name}: {e}")
            sys.exit(1)

        offsets = {}
        for video_path in mp4_files:
            if video_path == ref_path:
                offsets[video_path.name] = {
                    "timecode": None,
                    "fps": reference_fps,
                    "frame_offset": 0,
                    "offset_seconds": 0.0,
                }
                print(f"  {video_path.name}: frame_offset = 0 (reference)")
                continue

            print(f"  Extracting and correlating {video_path.name}...")
            try:
                fps = get_fps(video_path)
                target_audio = extract_audio_array(video_path, tmp_dir)
                offset_seconds, confidence = find_offset_seconds(reference_audio, target_audio, SAMPLE_RATE)
            except (subprocess.CalledProcessError, ValueError) as e:
                print(f"    -> FAILED: {e}  <-- skipping, fix and re-run for this camera")
                offsets[video_path.name] = {"error": str(e)}
                continue

            frame_offset = round(offset_seconds * fps)

            confidence_flag = "" if confidence > CONFIDENCE_THRESHOLD else "  <-- LOW CONFIDENCE, verify visually"

            offsets[video_path.name] = {
                "timecode": None,
                "fps": fps,
                "frame_offset": frame_offset,
                "offset_seconds": round(offset_seconds, 4),
                "confidence": round(confidence, 2),
            }
            print(f"    -> frame_offset = {frame_offset:+d} frames ({offset_seconds:+.3f} sec)"
                  f"  confidence={confidence:.2f}{confidence_flag}")

    output = {
        "reference_camera": ref_path.name,
        "reference_fps": reference_fps,
        "offsets": offsets,
        "method": "audio_cross_correlation",
        "note": (
            "frame_offset is relative to reference_camera's start, computed "
            "via audio cross-correlation (not internal timecode). A positive "
            "value means that camera started LATER and should be trimmed by "
            "that many frames at the start to align with the reference."
        ),
    }

    out_path = output_dir / "sync_offsets.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nWrote offsets to {out_path}")


if __name__ == "__main__":
    main()
