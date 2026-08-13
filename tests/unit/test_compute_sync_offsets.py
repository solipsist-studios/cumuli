# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

import json
import subprocess
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

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


@given(
    true_offset=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=100_000),
)
@settings(max_examples=40, deadline=None)
def test_find_offset_seconds_recovers_any_injected_offset(true_offset, seed):
    # The 3 tests above (positive/negative/zero) each prove the sign
    # convention at ONE hand-picked offset value on ONE fixed audio seed --
    # they can't rule out a bug that only shows up at other offsets or
    # other audio content. This sweeps both across many randomized
    # combinations to check the recovery-accuracy property holds in
    # general, not just at the specific points already hand-tested.
    # clip_seconds=6.0 with |true_offset|<=2.0 keeps overlap well above
    # find_offset_seconds' MIN_OVERLAP_FRACTION=0.5 requirement.
    master = transient_master(seconds=16.0, seed=seed)
    ref = camera_clip(master, 5.0, clip_seconds=6.0)
    tgt = camera_clip(master, 5.0 + true_offset, clip_seconds=6.0)
    est = cso.find_offset_seconds(ref, tgt, SR)
    assert est.offset_seconds == pytest.approx(true_offset, abs=0.02)
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


def test_correlate_envelopes_peak_value_non_positive_gives_floor_peak_ratio():
    # Every valid lag has a negative correlation (constant envelopes of
    # opposite sign never agree at any shift) -- peak_value <= 0, so a
    # ratio would be meaningless. Must hard-floor to 1.0 (the "completely
    # unconfident" value), not divide by something nonsensical.
    ref_env = np.full(6, 1.0)
    tgt_env = np.full(6, -1.0)
    est = cso.correlate_envelopes(ref_env, tgt_env, sample_rate=10)
    assert est.peak_ratio == 1.0
    assert not est.is_confident


def test_correlate_envelopes_no_competing_peak_gives_infinite_peak_ratio():
    # A tiny envelope self-matched against itself: relative to the exclusion
    # window (derived from PEAK_EXCLUSION_SECONDS * sample_rate), there's no
    # other lag left to compare against once the winning lag's neighborhood
    # is excluded -- runner_up stays at the -inf sentinel. This is the
    # "perfectly clean, uncontested match" case, capped to 999.0 by main()
    # before writing JSON (see test_main_caps_infinite_peak_ratio_to_keep_json_valid).
    rng = np.random.default_rng(1)
    env = rng.standard_normal(6)
    est = cso.correlate_envelopes(env, env.copy(), sample_rate=1000)
    assert est.peak_ratio == float("inf")
    assert est.is_confident


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


# --------------------------------------------------------------------------
# main() -- CLI arg wiring, reference selection, output JSON, error handling.
# get_fps/extract_audio_array are monkeypatched (they shell out to
# ffprobe/ffmpeg); these tests exercise main()'s own decisions on top of the
# already-tested preprocess_audio/solve_pairwise_offsets internals.
# --------------------------------------------------------------------------

def make_movies(tmp_path, names):
    d = tmp_path / "movies"
    d.mkdir()
    for n in names:
        (d / n).write_bytes(b"")
    return d


def patch_audio(monkeypatch, starts, fps_by_name=None, master=None, fail=()):
    master = master if master is not None else transient_master(seed=9)
    fps_by_name = fps_by_name or {name: 30.0 for name in starts}

    def fake_get_fps(video_path):
        return fps_by_name[video_path.name]

    def fake_extract_audio_array(video_path, tmp_dir):
        if video_path.name in fail:
            raise subprocess.CalledProcessError(1, "ffmpeg")
        return camera_clip(master, starts[video_path.name])

    monkeypatch.setattr(cso, "get_fps", fake_get_fps)
    monkeypatch.setattr(cso, "extract_audio_array", fake_extract_audio_array)


def test_main_usage_exits_1_on_wrong_arg_count(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "/only/one/arg"])
    with pytest.raises(SystemExit) as exc_info:
        cso.main()
    assert exc_info.value.code == 1
    assert "Usage" in capsys.readouterr().out


def test_main_single_camera_run_produces_reference_only_output(tmp_path, monkeypatch):
    # A one-camera "rig" has no one to correlate against -- must not crash,
    # and the output should just be the reference entry at frame_offset 0.
    movies = make_movies(tmp_path, ["0001.mp4"])
    patch_audio(monkeypatch, {"0001.mp4": 1.0})
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(out_dir)])
    cso.main()

    output = json.loads((out_dir / "sync_offsets.json").read_text())
    assert output["reference_camera"] == "0001.mp4"
    assert set(output["offsets"]) == {"0001.mp4"}
    assert output["offsets"]["0001.mp4"]["frame_offset"] == 0


def test_main_propagates_low_confidence_flag_into_written_json(tmp_path, monkeypatch):
    # The internals-level test_repetitive_music_is_flagged_ambiguous already
    # proves find_offset_seconds itself flags ambiguous audio -- this proves
    # that flag actually reaches the final JSON file through main(), not
    # just the internal OffsetEstimate object.
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4"])
    master = periodic_master()
    patch_audio(monkeypatch, {"0001.mp4": 1.0, "0002.mp4": 1.3}, master=master)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(out_dir)])
    cso.main()

    output = json.loads((out_dir / "sync_offsets.json").read_text())
    tgt = output["offsets"]["0002.mp4"]
    assert tgt["low_confidence"] is True
    assert "solver" in tgt


def test_main_reference_fallback_prints_both_warnings_together(tmp_path, monkeypatch, capsys):
    # Regression test: sol["solver"] == "reference_fallback" can only ever
    # occur when the camera's direct correlation to the reference is itself
    # not confident (see test_pairwise_solve_falls_back_for_unmatchable_camera
    # -- "bad" there is both reference_fallback AND not is_confident). Before
    # this fix, the two warnings were an if/elif, so the reference_fallback
    # message was unreachable dead code -- "not confident" always won the
    # branch first. Now both print together.
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4", "0003.mp4"])
    master = transient_master(seed=6)
    rng = np.random.default_rng(8)

    def fake_get_fps(video_path):
        return 30.0

    def fake_extract_audio_array(video_path, tmp_dir):
        if video_path.name == "0003.mp4":
            return rng.standard_normal(3 * SR)  # heard something else entirely
        starts = {"0001.mp4": 1.0, "0002.mp4": 1.2}
        return camera_clip(master, starts[video_path.name])

    monkeypatch.setattr(cso, "get_fps", fake_get_fps)
    monkeypatch.setattr(cso, "extract_audio_array", fake_extract_audio_array)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(out_dir)])
    cso.main()

    output = json.loads((out_dir / "sync_offsets.json").read_text())
    assert output["offsets"]["0003.mp4"]["solver"] == "reference_fallback"
    assert output["offsets"]["0003.mp4"]["low_confidence"] is True

    out = capsys.readouterr().out
    assert "LOW CONFIDENCE" in out
    assert "no confident pairwise path" in out


def test_main_caps_infinite_peak_ratio_to_keep_json_valid(tmp_path, monkeypatch):
    # correlate_envelopes can legitimately return peak_ratio=inf (a
    # perfectly clean match with no competing peak at all) -- json.dump
    # would otherwise write the non-standard "Infinity" token, which most
    # JSON parsers (including strict ones) reject. main() is supposed to
    # cap it at 999.0 before writing; this proves that cap actually
    # reaches the file and the file is genuinely valid, re-parseable JSON.
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4"])
    patch_audio(monkeypatch, {"0001.mp4": 1.0, "0002.mp4": 1.0})
    monkeypatch.setattr(cso, "correlate_envelopes",
                        lambda ref, tgt, sr: cso.OffsetEstimate(0.05, 0.9, float("inf")))
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(out_dir)])
    cso.main()

    raw_text = (out_dir / "sync_offsets.json").read_text()
    assert "Infinity" not in raw_text  # would make the file non-standard JSON
    output = json.loads(raw_text)  # re-parse from scratch, proving validity
    assert output["offsets"]["0002.mp4"]["peak_ratio"] == 999.0


def test_main_errors_when_movies_dir_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", str(tmp_path / "nope"), str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exc_info:
        cso.main()
    assert exc_info.value.code == 1
    assert "is not a directory" in capsys.readouterr().out


def test_main_errors_when_no_mp4_files(tmp_path, monkeypatch, capsys):
    movies = tmp_path / "movies"
    movies.mkdir()
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exc_info:
        cso.main()
    assert exc_info.value.code == 1
    assert "no video files" in capsys.readouterr().out


def test_main_finds_uppercase_mp4_extension(tmp_path, monkeypatch):
    # Regression test: real GoPro cameras default to uppercase .MP4 naming
    # (e.g. GX010001.MP4) -- a case-sensitive "*.mp4" glob silently found
    # zero cameras for real unmodified GoPro footage.
    movies = make_movies(tmp_path, ["GX010001.MP4", "GX010002.MP4"])
    patch_audio(monkeypatch, {"GX010001.MP4": 1.0, "GX010002.MP4": 1.0})
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(out_dir)])
    cso.main()

    output = json.loads((out_dir / "sync_offsets.json").read_text())
    assert output["reference_camera"] == "GX010001.MP4"
    assert set(output["offsets"]) == {"GX010001.MP4", "GX010002.MP4"}


def test_main_finds_other_supported_video_extensions(tmp_path, monkeypatch):
    # .mov/.mkv/.avi are all declared supported (SUPPORTED_VIDEO_EXTS) --
    # the old hardcoded "*.mp4" glob silently ignored all of them.
    movies = make_movies(tmp_path, ["0001.mov", "0002.MKV"])
    patch_audio(monkeypatch, {"0001.mov": 1.0, "0002.MKV": 1.0})
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(out_dir)])
    cso.main()

    output = json.loads((out_dir / "sync_offsets.json").read_text())
    assert set(output["offsets"]) == {"0001.mov", "0002.MKV"}


def test_main_errors_when_forced_reference_not_found(tmp_path, monkeypatch, capsys):
    movies = make_movies(tmp_path, ["0001.mp4"])
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(tmp_path / "out"), "9999.mp4"])
    with pytest.raises(SystemExit) as exc_info:
        cso.main()
    assert exc_info.value.code == 1
    assert "not found" in capsys.readouterr().out


def test_main_uses_first_alphabetical_as_default_reference(tmp_path, monkeypatch):
    movies = make_movies(tmp_path, ["0003.mp4", "0001.mp4", "0002.mp4"])
    starts = {"0001.mp4": 1.0, "0002.mp4": 1.1, "0003.mp4": 1.0}
    patch_audio(monkeypatch, starts)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(out_dir)])
    cso.main()

    output = json.loads((out_dir / "sync_offsets.json").read_text())
    assert output["reference_camera"] == "0001.mp4"


def test_main_uses_forced_reference_when_given(tmp_path, monkeypatch):
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4"])
    starts = {"0001.mp4": 1.0, "0002.mp4": 1.1}
    patch_audio(monkeypatch, starts)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(out_dir), "0002.mp4"])
    cso.main()

    output = json.loads((out_dir / "sync_offsets.json").read_text())
    assert output["reference_camera"] == "0002.mp4"
    assert output["offsets"]["0002.mp4"]["frame_offset"] == 0


def test_main_writes_expected_fields_and_correct_frame_offset_sign(tmp_path, monkeypatch):
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4"])
    starts = {"0001.mp4": 1.0, "0002.mp4": 1.2}  # 0002 started 0.2s LATER
    patch_audio(monkeypatch, starts)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(out_dir)])
    cso.main()

    output = json.loads((out_dir / "sync_offsets.json").read_text())
    assert output["method"] == "audio_envelope_cross_correlation"
    ref = output["offsets"]["0001.mp4"]
    assert ref["frame_offset"] == 0
    tgt = output["offsets"]["0002.mp4"]
    assert tgt["frame_offset"] == round(0.2 * 30.0)
    assert tgt["offset_seconds"] == pytest.approx(0.2, abs=0.01)
    assert tgt["low_confidence"] is False


def test_main_records_error_for_non_reference_camera_audio_failure(tmp_path, monkeypatch, capsys):
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4"])
    starts = {"0001.mp4": 1.0, "0002.mp4": 1.1}
    patch_audio(monkeypatch, starts, fail={"0002.mp4"})
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(out_dir)])
    cso.main()  # must not raise -- non-reference audio failures are recorded, not fatal

    output = json.loads((out_dir / "sync_offsets.json").read_text())
    assert "error" in output["offsets"]["0002.mp4"]
    assert "FAILED" in capsys.readouterr().out


def test_main_exits_when_reference_camera_audio_fails(tmp_path, monkeypatch, capsys):
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4"])
    starts = {"0001.mp4": 1.0, "0002.mp4": 1.1}
    patch_audio(monkeypatch, starts, fail={"0001.mp4"})  # reference itself fails
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exc_info:
        cso.main()
    assert exc_info.value.code == 1
    assert "reference camera" in capsys.readouterr().out
