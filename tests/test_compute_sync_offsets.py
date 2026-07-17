import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np

import compute_sync_offsets as cso

SR = cso.SAMPLE_RATE


# --------------------------------------------------------------------------
# synthetic audio fixtures -- pure numpy, no ffmpeg or real video needed
# --------------------------------------------------------------------------

def transient_master(seconds=8.0, seed=0):
    """Wall-clock audio with distinctive transients: clicks over low noise,
    the kind of content sync correlation is supposed to lock onto."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    audio = 0.05 * rng.standard_normal(n)
    for click_t in rng.uniform(0.5, seconds - 0.5, int(seconds * 2)):
        idx = int(click_t * SR)
        burst = np.hanning(80) * np.sin(2 * np.pi * 2000 * np.arange(80) / SR) * 3
        audio[idx:idx + 80] += burst
    return audio


def periodic_master(seconds=8.0, period=0.3, seed=1):
    """Music-like audio: a strictly repeating beat that offers naive
    correlation an equally good lock one period away from the truth."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    audio = 0.02 * rng.standard_normal(n)
    beat = np.hanning(160) * np.sin(2 * np.pi * 800 * np.arange(160) / SR)
    t = period
    while t < seconds - 0.1:
        idx = int(t * SR)
        audio[idx:idx + 160] += beat
        t += period
    return audio


def camera_clip(master, start_seconds, clip_seconds=3.0):
    """Simulate a camera that started recording at start_seconds wall time."""
    a = int(start_seconds * SR)
    return master[a:a + int(clip_seconds * SR)].astype(np.float32)


# --------------------------------------------------------------------------
# find_offset_seconds -- sign convention and accuracy
# --------------------------------------------------------------------------

def test_positive_offset_when_target_starts_later():
    # Regression test for the historical sign inversion: a camera that
    # started LATER than the reference must get a POSITIVE offset
    # (extract_synced_frames.py seeks to t - offset/fps in that clip).
    master = transient_master()
    ref = camera_clip(master, 1.0)
    tgt = camera_clip(master, 1.2)
    est = cso.find_offset_seconds(ref, tgt, SR)
    assert est.offset_seconds == pytest.approx(0.2, abs=0.005)
    assert est.is_confident


def test_negative_offset_when_target_starts_earlier():
    master = transient_master()
    ref = camera_clip(master, 1.0)
    tgt = camera_clip(master, 0.65)
    est = cso.find_offset_seconds(ref, tgt, SR)
    assert est.offset_seconds == pytest.approx(-0.35, abs=0.005)
    assert est.is_confident


def test_zero_offset():
    master = transient_master()
    ref = camera_clip(master, 1.0)
    tgt = camera_clip(master, 1.0)
    est = cso.find_offset_seconds(ref, tgt, SR)
    assert est.offset_seconds == pytest.approx(0.0, abs=0.005)
    assert est.is_confident


def test_subframe_offset_resolved():
    # A shift that is not a whole number of 30fps frames (0.05s = 1.5
    # frames) must come back accurate to well under a frame.
    master = transient_master(seed=3)
    ref = camera_clip(master, 1.0)
    tgt = camera_clip(master, 1.05)
    est = cso.find_offset_seconds(ref, tgt, SR)
    assert est.offset_seconds == pytest.approx(0.05, abs=0.005)


def test_repetitive_music_is_flagged_ambiguous():
    # Strictly periodic content: whatever lag wins, a competing lag one
    # period away is nearly as good, so the estimate must NOT be confident.
    master = periodic_master()
    ref = camera_clip(master, 1.0)
    tgt = camera_clip(master, 1.3)  # exactly one beat period later
    est = cso.find_offset_seconds(ref, tgt, SR)
    assert est.peak_ratio < cso.MIN_PEAK_RATIO
    assert not est.is_confident


def test_unrelated_audio_is_flagged_no_match():
    # Two cameras that heard completely different things: no lag makes the
    # envelopes genuinely alike, so match_quality must be low.
    rng = np.random.default_rng(7)
    ref = rng.standard_normal(3 * SR).astype(np.float32)
    tgt = rng.standard_normal(3 * SR).astype(np.float32)
    est = cso.find_offset_seconds(ref, tgt, SR)
    assert est.match_quality < cso.MIN_MATCH_QUALITY
    assert not est.is_confident


def test_silent_audio_does_not_crash_or_pretend_confidence():
    ref = camera_clip(transient_master(), 1.0)
    tgt = np.zeros(3 * SR, dtype=np.float32)
    est = cso.find_offset_seconds(ref, tgt, SR)
    assert np.isfinite(est.offset_seconds)
    assert not est.is_confident


def test_preprocess_audio_of_silence_is_all_zeros():
    env = cso.preprocess_audio(np.zeros(3 * SR, dtype=np.float32), SR)
    assert not np.any(env)
    assert np.all(np.isfinite(env))


# --------------------------------------------------------------------------
# solve_pairwise_offsets -- joint least-squares solve and fallback
# --------------------------------------------------------------------------

def test_pairwise_solve_recovers_injected_offsets():
    master = transient_master(seed=5)
    starts = {"0001.mp4": 1.0, "0002.mp4": 1.15, "0003.mp4": 0.8, "0004.mp4": 1.0}
    envelopes = {
        name: cso.preprocess_audio(camera_clip(master, s), SR)
        for name, s in starts.items()
    }
    solutions = cso.solve_pairwise_offsets(envelopes, "0001.mp4", SR)
    for name, start in starts.items():
        if name == "0001.mp4":
            continue
        sol = solutions[name]
        assert sol["solver"] == "pairwise_lsq"
        assert sol["estimate"].offset_seconds == pytest.approx(start - 1.0, abs=0.005)
        assert sol["estimate"].is_confident
        assert sol["max_residual_seconds"] < 0.01


def test_pairwise_solve_falls_back_for_unmatchable_camera():
    master = transient_master(seed=6)
    rng = np.random.default_rng(8)
    envelopes = {
        "0001.mp4": cso.preprocess_audio(camera_clip(master, 1.0), SR),
        "0002.mp4": cso.preprocess_audio(camera_clip(master, 1.2), SR),
        # this camera heard something else entirely
        "0003.mp4": cso.preprocess_audio(rng.standard_normal(3 * SR), SR),
    }
    solutions = cso.solve_pairwise_offsets(envelopes, "0001.mp4", SR)
    good = solutions["0002.mp4"]
    assert good["solver"] == "pairwise_lsq"
    assert good["estimate"].offset_seconds == pytest.approx(0.2, abs=0.005)
    bad = solutions["0003.mp4"]
    assert bad["solver"] == "reference_fallback"
    assert not bad["estimate"].is_confident
