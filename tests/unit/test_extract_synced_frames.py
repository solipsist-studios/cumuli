import json
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("PIL")
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

import extract_synced_frames as esf


# --------------------------------------------------------------------------
# ffmpeg_quality_args -- pure function, codec-dependent quality flag
# --------------------------------------------------------------------------

def test_ffmpeg_quality_args_jpg():
    assert esf.ffmpeg_quality_args(".jpg") == ["-q:v", "2"]


def test_ffmpeg_quality_args_jpeg():
    assert esf.ffmpeg_quality_args(".jpeg") == ["-q:v", "2"]


def test_ffmpeg_quality_args_webp():
    assert esf.ffmpeg_quality_args(".webp") == ["-lossless", "1"]


def test_ffmpeg_quality_args_png_has_no_flags():
    # .png is already lossless -- must not get the .jpg/.webp flags.
    assert esf.ffmpeg_quality_args(".png") == []


# --------------------------------------------------------------------------
# main() -- camera skip/fail wiring, seek-time sign convention, --window
# fan-out, --output_ext defaulting, --pp3_dir wiring. extract_frames() and
# apply_pp3() are monkeypatched (they shell out to ffmpeg/rawtherapee-cli);
# these tests only exercise main()'s own decisions.
# --------------------------------------------------------------------------

def make_movies(tmp_path, names):
    d = tmp_path / "movies"
    d.mkdir()
    for n in names:
        (d / n).write_bytes(b"")
    return d


def write_offsets(tmp_path, offsets, reference="0001.mp4"):
    path = tmp_path / "sync_offsets.json"
    path.write_text(json.dumps({"reference_camera": reference, "offsets": offsets}))
    return path


def patch_extract_frames(monkeypatch, calls):
    def fake_extract_frames(video_path, timestamp_sec, count, out_dir, ext):
        calls.append({"name": video_path.name, "seek": timestamp_sec, "count": count, "ext": ext})
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for k in range(1, count + 1):
            p = out_dir / f"{video_path.stem}_{k:02d}{ext}"
            p.write_bytes(b"")
            paths.append(p)
        return paths
    monkeypatch.setattr(esf, "extract_frames", fake_extract_frames)


GOOD_ENTRY = {"fps": 30.0, "frame_offset": 3}


def test_main_errors_when_movies_dir_missing(tmp_path, monkeypatch, capsys):
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    monkeypatch.setattr(sys, "argv", [
        "prog", str(tmp_path / "nope"), str(offsets_path), str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as exc_info:
        esf.main()
    assert exc_info.value.code == 1
    assert "is not a directory" in capsys.readouterr().out


def test_main_errors_when_offsets_path_missing(tmp_path, monkeypatch, capsys):
    movies = make_movies(tmp_path, ["0001.mp4"])
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(tmp_path / "nope.json"), str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as exc_info:
        esf.main()
    assert exc_info.value.code == 1
    assert "not found" in capsys.readouterr().out


def test_main_errors_cleanly_on_offsets_missing_expected_key(tmp_path, monkeypatch, capsys):
    # Regression test: sync_offsets.json without "offsets"/"reference_camera"
    # used to crash with a raw KeyError traceback instead of a clean message.
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = tmp_path / "sync_offsets.json"
    offsets_path.write_text(json.dumps({"reference_camera": "0001.mp4"}))  # no "offsets" key
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as exc_info:
        esf.main()
    assert exc_info.value.code == 1
    assert "missing expected key" in capsys.readouterr().out


def test_main_errors_cleanly_on_invalid_json(tmp_path, monkeypatch, capsys):
    # Regression test: malformed JSON used to crash with a raw
    # JSONDecodeError traceback instead of a clean message.
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = tmp_path / "sync_offsets.json"
    offsets_path.write_text("not valid json {{{")
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as exc_info:
        esf.main()
    assert exc_info.value.code == 1
    assert "not valid JSON" in capsys.readouterr().out


def test_main_errors_when_pp3_dir_missing(tmp_path, monkeypatch, capsys):
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(tmp_path / "out"),
        "--pp3_dir", str(tmp_path / "no_such_pp3_dir"),
    ])
    with pytest.raises(SystemExit) as exc_info:
        esf.main()
    assert exc_info.value.code == 1
    assert "--pp3_dir" in capsys.readouterr().out


def test_main_errors_on_output_ext_incompatible_with_pp3(tmp_path, monkeypatch, capsys):
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    pp3_dir = tmp_path / "pp3"
    pp3_dir.mkdir()
    monkeypatch.setattr(esf, "resolve_rawtherapee_cmd", lambda override=None: ["rawtherapee-cli"])
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(tmp_path / "out"),
        "--pp3_dir", str(pp3_dir), "--output_ext", ".jpg",
    ])
    with pytest.raises(SystemExit) as exc_info:
        esf.main()
    assert exc_info.value.code == 1
    assert "isn't supported with --pp3_dir" in capsys.readouterr().out


def test_main_seek_time_uses_documented_sign_convention(tmp_path, monkeypatch):
    # frame_offset=3 means this camera started LATER, so we must seek
    # EARLIER into its own clip: seconds_into_clip - frame_offset/fps.
    # Uses a non-default seconds_into_clip (5.0, default is 2.0) so a bug
    # that ignored the CLI value and used the hardcoded default couldn't
    # pass this test by coincidence.
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": {"fps": 30.0, "frame_offset": 3}}, reference="0001.mp4")
    calls = []
    patch_extract_frames(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(tmp_path / "out"), "5.0",
    ])
    esf.main()
    assert len(calls) == 1
    assert calls[0]["seek"] == pytest.approx(5.0 - 3 / 30.0)


REALISTIC_FPS = [23.976, 24.0, 25.0, 29.97, 30.0, 47.95, 48.0, 50.0, 59.94, 60.0]


@given(
    seconds_into_clip=st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False),
    frame_offset=st.integers(min_value=-200, max_value=200),
    fps=st.sampled_from(REALISTIC_FPS),
)
@settings(max_examples=60, deadline=None)
def test_main_seek_time_formula_holds_for_any_valid_combination(seconds_into_clip, frame_offset, fps):
    # test_main_seek_time_uses_documented_sign_convention already pins the
    # exact sign semantics against one independently hand-derived value.
    # This test's job is different: prove the CLI correctly THREADS
    # seconds_into_clip/frame_offset/fps through to that formula for any
    # realistic combination (any GoPro frame rate, any offset magnitude/
    # sign, any requested clip time) -- not just the one example already
    # hand-verified. It deliberately reuses the same formula as the
    # implementation as the expected value, since independent correctness
    # of the formula itself is what the sign-convention test already
    # covers; this one covers generalization across the input space.
    expected_seek = seconds_into_clip - frame_offset / fps
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        movies = make_movies(tmp_path, ["0001.mp4"])
        offsets_path = write_offsets(tmp_path, {"0001.mp4": {"fps": fps, "frame_offset": frame_offset}})
        out_dir = tmp_path / "out"
        calls = []
        with pytest.MonkeyPatch.context() as mp:
            patch_extract_frames(mp, calls)
            mp.setattr(sys, "argv", [
                "prog", str(movies), str(offsets_path), str(out_dir), str(seconds_into_clip),
            ])
            if expected_seek < 0:
                with pytest.raises(SystemExit):
                    esf.main()
                assert calls == []
            else:
                esf.main()
                assert len(calls) == 1
                assert calls[0]["seek"] == pytest.approx(expected_seek, abs=1e-6)


def test_main_seconds_into_clip_defaults_to_2_0_when_omitted(tmp_path, monkeypatch):
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": {"fps": 30.0, "frame_offset": 0}})
    calls = []
    patch_extract_frames(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(offsets_path), str(tmp_path / "out")])
    esf.main()
    assert calls[0]["seek"] == pytest.approx(2.0)


def test_main_window_zero_extracts_nothing_for_that_camera(tmp_path, monkeypatch):
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    calls = []
    patch_extract_frames(monkeypatch, calls)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(out_dir), "--window", "0",
    ])
    esf.main()  # should not raise -- 0 frames requested, 0 frames written, no error
    assert calls[0]["count"] == 0
    assert not any(out_dir.iterdir())


def test_main_skips_camera_with_negative_seek_and_exits_1(tmp_path, monkeypatch, capsys):
    # frame_offset large enough that seconds_into_clip - offset/fps < 0.
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4"])
    offsets_path = write_offsets(tmp_path, {
        "0001.mp4": {"fps": 30.0, "frame_offset": 0},
        "0002.mp4": {"fps": 30.0, "frame_offset": 100},
    }, reference="0001.mp4")
    calls = []
    patch_extract_frames(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(tmp_path / "out"), "1.0",
    ])
    with pytest.raises(SystemExit) as exc_info:
        esf.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "negative seek" in out
    assert {c["name"] for c in calls} == {"0001.mp4"}  # 0002 must not have been extracted


def test_main_skips_camera_with_missing_video_file(tmp_path, monkeypatch, capsys):
    movies = make_movies(tmp_path, ["0001.mp4"])  # 0002.mp4 not actually present
    offsets_path = write_offsets(tmp_path, {
        "0001.mp4": GOOD_ENTRY, "0002.mp4": GOOD_ENTRY,
    })
    calls = []
    patch_extract_frames(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit):
        esf.main()
    assert "0002.mp4 not found" in capsys.readouterr().out
    assert {c["name"] for c in calls} == {"0001.mp4"}


def test_main_skips_camera_with_error_stub_entry(tmp_path, monkeypatch, capsys):
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4"])
    offsets_path = write_offsets(tmp_path, {
        "0001.mp4": GOOD_ENTRY, "0002.mp4": {"error": "ffmpeg exited 1"},
    })
    calls = []
    patch_extract_frames(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit):
        esf.main()
    assert "no sync data" in capsys.readouterr().out
    assert {c["name"] for c in calls} == {"0001.mp4"}


def test_main_all_cameras_succeed_does_not_exit(tmp_path, monkeypatch, capsys):
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY, "0002.mp4": GOOD_ENTRY})
    calls = []
    patch_extract_frames(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(tmp_path / "out"),
    ])
    esf.main()  # should not raise
    assert "same instant of action" in capsys.readouterr().out


def test_main_window_writes_per_instant_subdirs(tmp_path, monkeypatch):
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    calls = []
    patch_extract_frames(monkeypatch, calls)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(out_dir), "--window", "3",
    ])
    esf.main()
    assert calls[0]["count"] == 3
    for k in range(3):
        assert (out_dir / f"f{k}" / "0001.jpg").is_file()


def test_main_output_ext_defaults_to_jpg_without_pp3(tmp_path, monkeypatch):
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    calls = []
    patch_extract_frames(monkeypatch, calls)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(offsets_path), str(out_dir)])
    esf.main()
    assert (out_dir / "0001.jpg").is_file()


def test_main_output_ext_explicit_override_without_pp3(tmp_path, monkeypatch):
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    calls = []
    patch_extract_frames(monkeypatch, calls)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(out_dir), "--output_ext", ".webp",
    ])
    esf.main()
    assert (out_dir / "0001.webp").is_file()


def test_main_output_ext_png_explicitly_allowed_with_pp3(tmp_path, monkeypatch):
    # .png is the one --output_ext value that's compatible with --pp3_dir
    # (RawTherapee always writes PNG bytes here) -- must NOT be rejected.
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    pp3_dir = tmp_path / "pp3"
    pp3_dir.mkdir()
    calls = []
    patch_extract_frames(monkeypatch, calls)
    monkeypatch.setattr(esf, "resolve_rawtherapee_cmd", lambda override=None: ["rawtherapee-cli"])
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(out_dir),
        "--pp3_dir", str(pp3_dir), "--output_ext", ".png",
    ])
    esf.main()  # should not raise
    assert (out_dir / "0001.png").is_file()


def test_main_extraction_exception_marks_camera_failed_and_exits_1(tmp_path, monkeypatch, capsys):
    # extract_frames() itself can raise (ffmpeg wrote a partial set of
    # frames -> RuntimeError) after all upfront validation already passed.
    # That must be caught per-camera, not crash the whole run, and other
    # cameras must still be processed.
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY, "0002.mp4": GOOD_ENTRY})
    calls = []

    def flaky_extract_frames(video_path, timestamp_sec, count, out_dir, ext):
        calls.append(video_path.name)
        if video_path.name == "0002.mp4":
            raise RuntimeError("ffmpeg wrote 0/1 frames")
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{video_path.stem}_01{ext}"
        p.write_bytes(b"")
        return [p]
    monkeypatch.setattr(esf, "extract_frames", flaky_extract_frames)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(offsets_path), str(out_dir)])
    with pytest.raises(SystemExit) as exc_info:
        esf.main()

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "ERROR extracting frames from 0002.mp4" in out
    assert "Failed/skipped cameras" in out and "0002.mp4" in out
    assert calls == ["0001.mp4", "0002.mp4"]  # 0001 still got processed
    assert (out_dir / "0001.jpg").is_file()
    assert not (out_dir / "0002.jpg").exists()


def test_main_window_greater_than_1_prints_multiframe_followup(tmp_path, monkeypatch, capsys):
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    calls = []
    patch_extract_frames(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(tmp_path / "out"), "--window", "3",
    ])
    esf.main()
    assert "Per-instant subdirs f0/..f2/" in capsys.readouterr().out


def test_main_errors_on_same_stem_different_extension_collision(tmp_path, monkeypatch, capsys):
    # Two distinct camera files that share a filename stem but differ in
    # extension (e.g. "0001.mp4" and "0001.MOV" -- plausible if a directory
    # has both a primary capture and a re-encoded backup, or after the
    # case-insensitive/multi-extension discovery fix in
    # compute_sync_offsets.py made it more likely for both to be
    # discovered as separate cameras) would otherwise silently overwrite
    # each other's output frame with no warning. Must be rejected up front,
    # before any extraction happens, rather than silently corrupting output.
    movies = make_movies(tmp_path, ["0001.mp4", "0001.MOV"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY, "0001.MOV": GOOD_ENTRY})
    calls = []
    patch_extract_frames(monkeypatch, calls)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(offsets_path), str(out_dir)])

    with pytest.raises(SystemExit) as exc_info:
        esf.main()

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "same output filename stem" in out
    assert "0001.mp4" in out and "0001.MOV" in out
    assert calls == []  # must fail BEFORE extracting anything
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_main_no_collision_when_stems_are_distinct(tmp_path, monkeypatch):
    # Sanity check: the collision detector must not have false positives
    # for the normal case of distinctly-named cameras.
    movies = make_movies(tmp_path, ["0001.mp4", "0002.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY, "0002.mp4": GOOD_ENTRY})
    calls = []
    patch_extract_frames(monkeypatch, calls)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prog", str(movies), str(offsets_path), str(out_dir)])

    esf.main()  # should not raise

    assert len(calls) == 2


def test_main_pp3_applies_color_correction_and_forces_png(tmp_path, monkeypatch):
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    pp3_dir = tmp_path / "pp3"
    pp3_dir.mkdir()
    pp3_path = pp3_dir / "0001.mp4.pp3"
    pp3_path.write_text("")

    calls = []
    patch_extract_frames(monkeypatch, calls)
    monkeypatch.setattr(esf, "resolve_rawtherapee_cmd", lambda override=None: ["rawtherapee-cli"])
    apply_calls = []

    def fake_apply_pp3(rt_cmd, image_path, pp3_path_arg, out_path):
        apply_calls.append((rt_cmd, pp3_path_arg, out_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
    monkeypatch.setattr(esf, "apply_pp3", fake_apply_pp3)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(out_dir), "--pp3_dir", str(pp3_dir),
    ])
    esf.main()

    assert len(apply_calls) == 1
    assert apply_calls[0][1] == pp3_path
    assert apply_calls[0][2] == out_dir / "0001.png"


def test_main_pp3_no_matching_profile_warns_and_writes_uncorrected(tmp_path, monkeypatch, capsys):
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    pp3_dir = tmp_path / "pp3"
    pp3_dir.mkdir()  # no .pp3 files inside -- nothing will match

    calls = []
    patch_extract_frames(monkeypatch, calls)
    monkeypatch.setattr(esf, "resolve_rawtherapee_cmd", lambda override=None: ["rawtherapee-cli"])
    monkeypatch.setattr(esf, "apply_pp3", lambda *a, **k: pytest.fail("apply_pp3 must not be called"))

    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(out_dir), "--pp3_dir", str(pp3_dir),
    ])
    esf.main()

    assert "no .pp3 profile matches" in capsys.readouterr().out
    assert (out_dir / "0001.png").is_file()  # still written, just uncorrected


def test_main_rawtherapee_cmd_override_is_passed_through(tmp_path, monkeypatch):
    movies = make_movies(tmp_path, ["0001.mp4"])
    offsets_path = write_offsets(tmp_path, {"0001.mp4": GOOD_ENTRY})
    pp3_dir = tmp_path / "pp3"
    pp3_dir.mkdir()
    (pp3_dir / "0001.mp4.pp3").write_text("")

    calls = []
    patch_extract_frames(monkeypatch, calls)
    resolve_calls = []
    monkeypatch.setattr(esf, "resolve_rawtherapee_cmd", lambda override=None: (resolve_calls.append(override), ["custom-rt"])[1])

    def fake_apply_pp3(rt_cmd, image_path, pp3_path_arg, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
    monkeypatch.setattr(esf, "apply_pp3", fake_apply_pp3)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", str(movies), str(offsets_path), str(out_dir),
        "--pp3_dir", str(pp3_dir), "--rawtherapee_cmd", "custom-rt --flag",
    ])
    esf.main()
    assert resolve_calls == ["custom-rt --flag"]
