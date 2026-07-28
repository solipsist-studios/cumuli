#!/usr/bin/env python3
"""
compute_sync_offsets.py

Computes sync offsets across multiple camera clips using audio
cross-correlation (real signal matching), instead of relying on GoPro
internal timecode metadata (which can drift independently per camera
unless camera-to-camera time sync was explicitly used during capture).

Requires: numpy, scipy, ffmpeg (for audio extraction)

Usage:
    python3 compute_sync_offsets.py /path/to/movies/dir /path/to/output/dir [reference_camera_filename]

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
      "method": "audio_envelope_cross_correlation"
    }

Sign convention (matches extract_synced_frames.py): a positive
frame_offset / offset_seconds means that camera started recording LATER
than the reference, so a real-world moment at t seconds into the
reference clip sits at (t - offset_seconds) into that camera's clip.

How it works: each clip's audio is reduced to its band-passed loudness
envelope, every camera pair is aligned by normalized cross-correlation
(with parabolic sub-sample peak refinement), and per-camera offsets are
solved jointly by least squares over all pairwise measurements --
averaging out per-pair noise and letting cycle-consistency residuals
expose any camera whose alignment disagrees with the rest of the rig.

Each camera reports two reliability numbers:
  match_quality -- normalized correlation at the chosen lag (0..1); low
      values mean the two envelopes don't actually look alike when
      aligned (wrong content, silence, or no shared audio).
  peak_ratio -- best peak vs. the best competing peak elsewhere; values
      near 1 mean the audio is repetitive/ambiguous (e.g. music) and a
      competing lag explains it almost as well.
A camera failing either threshold is flagged LOW CONFIDENCE and, if its
pairwise measurements are unusable, falls back to its direct
correlation against the reference.

Note: after generating sync_offsets.json, it is common to hand-tune a few
cameras' frame_offset by a couple of frames based on visual inspection
with make_sync_grid.py, then save the result as sync_offsets_v2.json,
sync_offsets_v3.json, etc. Point extract_synced_frames.py at whichever
version is your current best.

Validation status: the live output of this build was checked per-camera
against the hand-verified sync_offsets_v5.json for the 260521-105422
capture (and cross-checked against subject-motion correlation computed
directly on the video frames). Earlier builds returned offsets with an
inverted sign -- if comparing against JSONs produced before the fix,
expect historical small offsets to be sign-flipped.
"""

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, correlate, hilbert, sosfiltfilt

from image_formats import SUPPORTED_VIDEO_EXTS


SAMPLE_RATE = 16000  # downsample target -- plenty for sync correlation, much faster than 48kHz
CORRELATION_WINDOW_SECONDS = 30  # how much audio find_offset_seconds() actually compares
EXTRACT_SECONDS = 45  # correlation window + margin for slow-starting cameras; caps ffmpeg decode time

BANDPASS_HZ = (300.0, 4000.0)  # speech/transient band; drops rumble, hiss, and mains hum
ENVELOPE_LOWPASS_HZ = 200.0  # loudness-envelope bandwidth used for correlation
PEAK_EXCLUSION_SECONDS = 0.2  # a competing peak must be at least this far away to count as ambiguity
MIN_OVERLAP_FRACTION = 0.5  # only consider lags where at least half the shorter clip overlaps

MIN_MATCH_QUALITY = 0.4  # real matched captures score ~0.6-0.8; unrelated audio scores ~0.1
MIN_PEAK_RATIO = 1.3  # repetitive audio (music) scores ~1.0; clean matches score >2


@dataclass(frozen=True)
class OffsetEstimate:
    offset_seconds: float
    match_quality: float
    peak_ratio: float

    @property
    def is_confident(self) -> bool:
        return (self.match_quality >= MIN_MATCH_QUALITY
                and self.peak_ratio >= MIN_PEAK_RATIO)


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


def preprocess_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Reduce raw audio to a mean-removed loudness envelope for correlation.

    Correlating envelopes instead of raw waveforms is robust to per-camera
    coloration, reverb, and mic distance, which decorrelate the fine
    waveform structure between cameras while leaving loudness dynamics
    intact.
    """
    x = np.asarray(audio, dtype=np.float64)
    x = x[: sample_rate * CORRELATION_WINDOW_SECONDS]
    if len(x) < sample_rate // 4 or not np.any(x):
        return np.zeros(len(x))
    x = x - x.mean()
    high = min(BANDPASS_HZ[1], sample_rate / 2 - 1)
    sos = butter(4, [BANDPASS_HZ[0], high], "bandpass", fs=sample_rate, output="sos")
    banded = sosfiltfilt(sos, x)
    envelope = np.abs(hilbert(banded))
    sos_lp = butter(2, ENVELOPE_LOWPASS_HZ, "lowpass", fs=sample_rate, output="sos")
    envelope = sosfiltfilt(sos_lp, envelope)
    return envelope - envelope.mean()


def _ncc_profile(ref_env: np.ndarray, tgt_env: np.ndarray):
    """Normalized cross-correlation of two envelopes at every lag with
    sufficient overlap. Returns (lags_in_samples, ncc_values); lags without
    enough overlap are -inf.
    """
    c = correlate(tgt_env, ref_env, mode="full")
    n_ref, n_tgt = len(ref_env), len(tgt_env)
    lags = np.arange(-(n_ref - 1), n_tgt)

    # Per-lag energies of the overlapping segments, via cumulative sums:
    # at lag L the comparison is ref[a0:a1] vs tgt[a0+L:a1+L].
    ref_sq = np.concatenate([[0.0], np.cumsum(ref_env * ref_env)])
    tgt_sq = np.concatenate([[0.0], np.cumsum(tgt_env * tgt_env)])
    a0 = np.maximum(0, -lags)
    a1 = np.minimum(n_ref, n_tgt - lags)
    overlap = np.maximum(a1 - a0, 0)
    valid = overlap >= max(2, int(MIN_OVERLAP_FRACTION * min(n_ref, n_tgt)))
    a0v = np.where(valid, a0, 0)
    a1v = np.where(valid, a1, 0)
    b0v = np.where(valid, a0 + lags, 0)
    b1v = np.where(valid, a1 + lags, 0)
    energy = np.sqrt((ref_sq[a1v] - ref_sq[a0v]) * (tgt_sq[b1v] - tgt_sq[b0v]))

    ncc = np.full(len(lags), -np.inf)
    ok = valid & (energy > 1e-12)
    ncc[ok] = c[ok] / energy[ok]
    return lags, ncc


def correlate_envelopes(ref_env: np.ndarray, tgt_env: np.ndarray,
                        sample_rate: int) -> OffsetEstimate:
    """Align two preprocessed envelopes. See find_offset_seconds for the
    sign convention.
    """
    lags, ncc = _ncc_profile(ref_env, tgt_env)
    peak_idx = int(np.argmax(ncc))
    peak_value = float(ncc[peak_idx])
    if not np.isfinite(peak_value):
        return OffsetEstimate(0.0, 0.0, 1.0)

    # Parabolic sub-sample refinement around the peak.
    delta = 0.0
    if 0 < peak_idx < len(ncc) - 1:
        y0, y1, y2 = ncc[peak_idx - 1], ncc[peak_idx], ncc[peak_idx + 1]
        if np.isfinite(y0) and np.isfinite(y2):
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-15:
                delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))

    # correlate(tgt, ref) peaks at positive lag when the target started
    # EARLIER than the reference, so negate to get the documented
    # "positive = started later" convention.
    offset_seconds = -(lags[peak_idx] + delta) / sample_rate

    exclusion = int(PEAK_EXCLUSION_SECONDS * sample_rate)
    competing = ncc.copy()
    competing[max(0, peak_idx - exclusion): peak_idx + exclusion + 1] = -np.inf
    runner_up = float(np.max(competing))
    if peak_value <= 0:
        peak_ratio = 1.0
    elif runner_up <= 0 or not np.isfinite(runner_up):
        peak_ratio = np.inf
    else:
        peak_ratio = peak_value / runner_up
    return OffsetEstimate(offset_seconds, peak_value, peak_ratio)


def find_offset_seconds(reference_audio: np.ndarray, target_audio: np.ndarray,
                        sample_rate: int) -> OffsetEstimate:
    """Estimate how much later the target camera started than the reference,
    by normalized cross-correlation of the two clips' loudness envelopes.

    Positive offset_seconds means the target started LATER than the
    reference: a real-world moment at t seconds into the reference clip is
    at (t - offset_seconds) into the target clip.
    """
    ref_env = preprocess_audio(reference_audio, sample_rate)
    tgt_env = preprocess_audio(target_audio, sample_rate)
    return correlate_envelopes(ref_env, tgt_env, sample_rate)


def solve_pairwise_offsets(envelopes: dict, reference: str, sample_rate: int) -> dict:
    """Estimate every camera's offset from the reference by least squares
    over all confident pairwise correlations, instead of trusting each
    camera's single direct correlation against the reference.

    Returns {name: {"estimate": OffsetEstimate, "solver": ...,
    "max_residual_seconds": ...}} for every non-reference camera.
    "pairwise_lsq" cameras were solved jointly; "reference_fallback"
    cameras had no confident pairwise path to the reference, so their
    direct (unconfident) measurement is reported as-is.
    """
    names = list(envelopes)
    pair_estimates = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair_estimates[(a, b)] = correlate_envelopes(
                envelopes[a], envelopes[b], sample_rate)

    accepted = {pair: est for pair, est in pair_estimates.items() if est.is_confident}

    # Cameras reachable from the reference through confident pairs can be
    # solved jointly; anything else falls back to its direct measurement.
    connected = {reference}
    frontier = [reference]
    while frontier:
        cur = frontier.pop()
        for a, b in accepted:
            other = b if a == cur else a if b == cur else None
            if other is not None and other not in connected:
                connected.add(other)
                frontier.append(other)

    solved = list(connected)
    index = {name: i for i, name in enumerate(solved)}
    solve_pairs = [pair for pair in accepted if pair[0] in connected and pair[1] in connected]

    starts = {reference: 0.0}
    residuals = {name: 0.0 for name in solved}
    if len(solved) > 1 and solve_pairs:
        # start[b] - start[a] = measured offset, anchored at start[reference] = 0.
        rows = np.zeros((len(solve_pairs), len(solved)))
        rhs = np.zeros(len(solve_pairs))
        for r, (a, b) in enumerate(solve_pairs):
            rows[r, index[b]] = 1.0
            rows[r, index[a]] = -1.0
            rhs[r] = accepted[(a, b)].offset_seconds
        free = [i for i, name in enumerate(solved) if name != reference]
        solution, *_ = np.linalg.lstsq(rows[:, free], rhs, rcond=None)
        for i, col in enumerate(free):
            starts[solved[col]] = float(solution[i])
        fit = rows[:, free] @ solution - rhs
        for r, (a, b) in enumerate(solve_pairs):
            residuals[a] = max(residuals[a], abs(float(fit[r])))
            residuals[b] = max(residuals[b], abs(float(fit[r])))

    def direct(name):
        a, b = (reference, name) if (reference, name) in pair_estimates else (name, reference)
        est = pair_estimates[(a, b)]
        if a == name:  # measured the other way around
            est = OffsetEstimate(-est.offset_seconds, est.match_quality, est.peak_ratio)
        return est

    results = {}
    for name in names:
        if name == reference:
            continue
        direct_est = direct(name)
        if name in connected:
            # Report the jointly-solved offset with the reliability of this
            # camera's weakest accepted measurement.
            own_pairs = [est for pair, est in accepted.items() if name in pair]
            results[name] = {
                "estimate": OffsetEstimate(
                    starts[name],
                    min(est.match_quality for est in own_pairs),
                    min(est.peak_ratio for est in own_pairs),
                ),
                "solver": "pairwise_lsq",
                "max_residual_seconds": residuals[name],
            }
        else:
            results[name] = {
                "estimate": direct_est,
                "solver": "reference_fallback",
                "max_residual_seconds": None,
            }
    return results


def main():
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: python3 compute_sync_offsets.py "
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

    mp4_files = sorted(p for p in movies_dir.iterdir() if p.suffix.lower() in SUPPORTED_VIDEO_EXTS)
    if not mp4_files:
        print(f"Error: no video files found in {movies_dir} "
              f"(supported: {', '.join(SUPPORTED_VIDEO_EXTS)}, case-insensitive)")
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

    envelopes = {}
    fps_by_name = {}
    errors = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for video_path in mp4_files:
            print(f"  Extracting audio from {video_path.name}...")
            try:
                fps_by_name[video_path.name] = get_fps(video_path)
                audio = extract_audio_array(video_path, tmp_dir)
                envelopes[video_path.name] = preprocess_audio(audio, SAMPLE_RATE)
            except (subprocess.CalledProcessError, ValueError) as e:
                if video_path == ref_path:
                    print(f"Error: could not read reference camera {ref_path.name}: {e}")
                    sys.exit(1)
                print(f"    -> FAILED: {e}  <-- skipping, fix and re-run for this camera")
                errors[video_path.name] = str(e)

    reference_fps = fps_by_name[ref_path.name]
    print("\n  Correlating all camera pairs and solving jointly...")
    solutions = solve_pairwise_offsets(envelopes, ref_path.name, SAMPLE_RATE)

    offsets = {
        ref_path.name: {
            "timecode": None,
            "fps": reference_fps,
            "frame_offset": 0,
            "offset_seconds": 0.0,
        }
    }
    print(f"  {ref_path.name}: frame_offset = 0 (reference)")
    for video_path in mp4_files:
        name = video_path.name
        if name == ref_path.name:
            continue
        if name in errors:
            offsets[name] = {"error": errors[name]}
            continue
        sol = solutions[name]
        est = sol["estimate"]
        fps = fps_by_name[name]
        frame_offset = round(est.offset_seconds * fps)
        reasons = []
        if not est.is_confident:
            reasons.append("LOW CONFIDENCE, verify visually")
        if sol["solver"] == "reference_fallback":
            reasons.append("no confident pairwise path, verify visually")
        flag = "  <-- " + "; ".join(reasons) if reasons else ""
        offsets[name] = {
            "timecode": None,
            "fps": fps,
            "frame_offset": frame_offset,
            "offset_seconds": round(est.offset_seconds, 4),
            "match_quality": round(est.match_quality, 3),
            "peak_ratio": round(min(est.peak_ratio, 999.0), 2),
            "low_confidence": not est.is_confident,
            "solver": sol["solver"],
        }
        if sol["max_residual_seconds"] is not None:
            offsets[name]["max_residual_seconds"] = round(sol["max_residual_seconds"], 4)
        print(f"    {name}: frame_offset = {frame_offset:+d} frames ({est.offset_seconds:+.3f} sec)"
              f"  match_quality={est.match_quality:.2f} peak_ratio={est.peak_ratio:.2f}{flag}")

    output = {
        "reference_camera": ref_path.name,
        "reference_fps": reference_fps,
        "offsets": offsets,
        "method": "audio_envelope_cross_correlation",
        "solver": "pairwise_least_squares",
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
