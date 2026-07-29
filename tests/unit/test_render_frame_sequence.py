import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import render_frame_sequence as rfs
import run_unified_pipeline as unified


# --------------------------------------------------------------------------
# resolve_sync_json
# --------------------------------------------------------------------------

def test_resolve_sync_json_uses_resolved_path_when_present(tmp_path):
    calib_run_dir = tmp_path
    (calib_run_dir / "resolved_sync_json.txt").write_text(str(tmp_path / "hand_tuned.json"))
    assert rfs.resolve_sync_json(calib_run_dir) == tmp_path / "hand_tuned.json"


def test_resolve_sync_json_falls_back_to_default_path(tmp_path):
    assert rfs.resolve_sync_json(tmp_path) == tmp_path / "sync_offsets.json"


# --------------------------------------------------------------------------
# compute_or_reuse_calibration
# --------------------------------------------------------------------------

def test_compute_or_reuse_calibration_reuses_existing_transforms(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    L = unified.build_layout(out_dir)
    L["transforms_refined"].parent.mkdir(parents=True)
    L["transforms_refined"].write_text("{}")
    (out_dir / "resolved_sync_json.txt").write_text(str(out_dir / "my_sync.json"))

    calls = []
    monkeypatch.setattr(unified, "stage_sync", lambda *a, **k: calls.append("stage_sync"))
    monkeypatch.setattr(unified, "stage_production", lambda *a, **k: calls.append("stage_production"))
    monkeypatch.setattr(unified, "stage_poses", lambda *a, **k: calls.append("stage_poses"))

    args = NS(out_dir=out_dir, calib_time=None, start_time="1.5s")
    transforms, sync_json = rfs.compute_or_reuse_calibration(args, ".jpg")

    assert calls == []  # nothing recomputed
    assert transforms == L["transforms_refined"]
    assert sync_json == out_dir / "my_sync.json"


def test_compute_or_reuse_calibration_computes_when_nothing_cached(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    calls = []

    def fake_stage_sync(args, L, image_ext):
        calls.append("stage_sync")
        return L["sync_offsets"]
    monkeypatch.setattr(unified, "stage_sync", fake_stage_sync)
    monkeypatch.setattr(unified, "stage_production",
                         lambda *a, **k: calls.append("stage_production"))
    monkeypatch.setattr(unified, "stage_poses", lambda *a, **k: calls.append("stage_poses"))

    args = NS(out_dir=out_dir, calib_time=None, start_time="1.5s")
    transforms, sync_json = rfs.compute_or_reuse_calibration(args, ".jpg")

    assert calls == ["stage_sync", "stage_production", "stage_poses"]
    L = unified.build_layout(out_dir)
    assert transforms == L["transforms_refined"]
    assert sync_json == L["sync_offsets"]
    assert (out_dir / "resolved_sync_json.txt").read_text() == str(L["sync_offsets"])


def test_compute_or_reuse_calibration_uses_calib_time_over_start_time(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    captured = {}

    def fake_stage_sync(args, L, image_ext):
        captured["target_time_s"] = args.target_time_s
        return L["sync_offsets"]
    monkeypatch.setattr(unified, "stage_sync", fake_stage_sync)
    monkeypatch.setattr(unified, "stage_production", lambda *a, **k: None)
    monkeypatch.setattr(unified, "stage_poses", lambda *a, **k: None)

    args = NS(out_dir=out_dir, calib_time="2.5s", start_time="1.0s")
    rfs.compute_or_reuse_calibration(args, ".jpg")
    assert captured["target_time_s"] == pytest.approx(2.5)


def test_compute_or_reuse_calibration_falls_back_to_start_time_when_no_calib_time(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    captured = {}

    def fake_stage_sync(args, L, image_ext):
        captured["target_time_s"] = args.target_time_s
        return L["sync_offsets"]
    monkeypatch.setattr(unified, "stage_sync", fake_stage_sync)
    monkeypatch.setattr(unified, "stage_production", lambda *a, **k: None)
    monkeypatch.setattr(unified, "stage_poses", lambda *a, **k: None)

    args = NS(out_dir=out_dir, calib_time=None, start_time="1.0s")
    rfs.compute_or_reuse_calibration(args, ".jpg")
    assert captured["target_time_s"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# resolve_target_times
# --------------------------------------------------------------------------

def test_resolve_target_times_generates_uniform_range():
    args = NS(start_time="1.0s", stop_time="1.2s", fps=10.0)
    times = rfs.resolve_target_times(args)
    assert times == pytest.approx([1.0, 1.1, 1.2])


def test_resolve_target_times_single_instant_when_start_equals_stop():
    args = NS(start_time="1.5s", stop_time="1.5s", fps=30.0)
    assert rfs.resolve_target_times(args) == pytest.approx([1.5])


def test_resolve_target_times_errors_when_fps_not_positive(monkeypatch):
    args = NS(start_time="1.0s", stop_time="1.2s", fps=0.0)
    with pytest.raises(SystemExit) as exc:
        rfs.resolve_target_times(args)
    assert exc.value.code == 1


def test_resolve_target_times_errors_when_stop_before_start(monkeypatch):
    args = NS(start_time="1.2s", stop_time="1.0s", fps=10.0)
    with pytest.raises(SystemExit) as exc:
        rfs.resolve_target_times(args)
    assert exc.value.code == 1


# --------------------------------------------------------------------------
# build_parser -- required args
# --------------------------------------------------------------------------

_BASE_CLI = ["--video_dir", "/v", "--calib_dir", "/c", "--out_dir", "/o",
             "--start_time", "1s", "--stop_time", "2s", "--fps", "30"]


@pytest.mark.parametrize("missing", ["--video_dir", "--calib_dir", "--out_dir",
                                     "--start_time", "--stop_time", "--fps"])
def test_build_parser_errors_when_required_arg_missing(missing):
    argv = list(_BASE_CLI)
    idx = argv.index(missing)
    del argv[idx:idx + 2]
    parser = rfs.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 2


# --------------------------------------------------------------------------
# main() -- CLI-level validation and per-frame wiring
# --------------------------------------------------------------------------

@pytest.fixture
def rig(tmp_path):
    video_dir = tmp_path / "videos"
    calib_dir = tmp_path / "calib"
    out_dir = tmp_path / "out"
    video_dir.mkdir()
    calib_dir.mkdir()
    (video_dir / "0001.mp4").write_bytes(b"")
    return {"video_dir": video_dir, "calib_dir": calib_dir, "out_dir": out_dir}


def base_argv(rig, **extra):
    argv = [
        "prog",
        "--video_dir", str(rig["video_dir"]),
        "--calib_dir", str(rig["calib_dir"]),
        "--out_dir", str(rig["out_dir"]),
        "--start_time", "1.0s", "--stop_time", "1.0s", "--fps", "30",
    ]
    for k, v in extra.items():
        argv += [f"--{k}", str(v)]
    return argv


def test_main_errors_when_video_dir_not_a_directory(rig, monkeypatch):
    argv = base_argv(rig)
    argv[argv.index(str(rig["video_dir"]))] = str(rig["out_dir"] / "nope")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        rfs.main()
    assert exc.value.code == 1


def test_main_errors_when_calib_dir_not_a_directory(rig, monkeypatch):
    argv = base_argv(rig)
    argv[argv.index(str(rig["calib_dir"]))] = str(rig["out_dir"] / "nope")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        rfs.main()
    assert exc.value.code == 1


def test_main_errors_when_target_pkl_dir_not_a_directory(rig, monkeypatch):
    # Regression test: this script used to accept --target_pkl_dir and thread
    # it straight into stage_production() with no upfront validation at all,
    # unlike run_unified_pipeline.py's own identical flag -- a bad path would
    # only surface deep inside undistort_frames.py, after calibration already ran.
    # Calibration/frame-loop stages are stubbed out so the *only* way this can
    # exit 1 is the --target_pkl_dir check itself, not an unrelated real call.
    monkeypatch.setattr(rfs, "compute_or_reuse_calibration",
                         lambda *a, **k: (Path("/t.json"), Path("/s.json")))
    monkeypatch.setattr(unified, "stage_production", lambda *a, **k: None)
    monkeypatch.setattr(rfs.hloc_mod, "restructure_flat_to_percam", lambda *a, **k: None)
    monkeypatch.setattr(unified, "stage_masks", lambda *a, **k: None)
    monkeypatch.setattr(unified, "stage_branch_direct", lambda *a, **k: None)

    monkeypatch.setattr(sys, "argv",
                         base_argv(rig, target_pkl_dir=str(rig["out_dir"] / "nope")))
    with pytest.raises(SystemExit) as exc:
        rfs.main()
    assert exc.value.code == 1


def test_main_errors_when_calib_run_dir_missing_transforms(rig, monkeypatch):
    calib_run_dir = rig["out_dir"].parent / "other_run"
    calib_run_dir.mkdir()
    monkeypatch.setattr(sys, "argv", base_argv(rig, calib_run_dir=str(calib_run_dir)))
    with pytest.raises(SystemExit) as exc:
        rfs.main()
    assert exc.value.code == 1


def test_main_errors_when_calib_run_dir_missing_sync_json(rig, monkeypatch):
    calib_run_dir = rig["out_dir"].parent / "other_run"
    calib_run_dir.mkdir()
    (calib_run_dir / "transforms_refined.json").write_text("{}")
    # no sync_offsets.json / resolved_sync_json.txt present
    monkeypatch.setattr(sys, "argv", base_argv(rig, calib_run_dir=str(calib_run_dir)))
    with pytest.raises(SystemExit) as exc:
        rfs.main()
    assert exc.value.code == 1


def test_main_calib_run_dir_reuses_calibration_and_skips_compute(rig, monkeypatch):
    calib_run_dir = rig["out_dir"].parent / "other_run"
    calib_run_dir.mkdir()
    (calib_run_dir / "transforms_refined.json").write_text("{}")
    (calib_run_dir / "sync_offsets.json").write_text("{}")

    calib_calls = []
    monkeypatch.setattr(rfs, "compute_or_reuse_calibration",
                         lambda *a, **k: calib_calls.append("called"))
    frame_calls = []
    monkeypatch.setattr(unified, "stage_production", lambda *a, **k: frame_calls.append("stage_production"))
    monkeypatch.setattr(rfs.hloc_mod, "restructure_flat_to_percam", lambda *a, **k: None)
    monkeypatch.setattr(unified, "stage_masks", lambda *a, **k: frame_calls.append("stage_masks"))
    monkeypatch.setattr(unified, "stage_branch_direct",
                         lambda *a, **k: frame_calls.append("stage_branch_direct"))

    monkeypatch.setattr(sys, "argv", base_argv(rig, calib_run_dir=str(calib_run_dir)))
    rfs.main()

    assert calib_calls == []  # compute_or_reuse_calibration never invoked
    assert frame_calls == ["stage_production", "stage_masks", "stage_branch_direct"]


def test_main_errors_when_multiframe_sfm_script_missing_and_no_calib_run_dir(rig, monkeypatch):
    monkeypatch.setattr(sys, "argv",
                         base_argv(rig, multiframe_sfm_script="/definitely/not/real.py"))
    with pytest.raises(SystemExit) as exc:
        rfs.main()
    assert exc.value.code == 1


def test_main_errors_when_initial_sync_json_given_but_missing(rig, monkeypatch):
    monkeypatch.setattr(sys, "argv",
                         base_argv(rig, initial_sync_json=str(rig["out_dir"] / "nope.json")))
    with pytest.raises(SystemExit) as exc:
        rfs.main()
    assert exc.value.code == 1


def test_main_errors_when_no_videos_found(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()  # empty
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    argv = ["prog", "--video_dir", str(video_dir), "--calib_dir", str(calib_dir),
            "--out_dir", str(tmp_path / "out"), "--start_time", "1s", "--stop_time", "1s", "--fps", "30"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(rfs, "compute_or_reuse_calibration",
                         lambda *a, **k: (tmp_path / "t.json", tmp_path / "s.json"))
    with pytest.raises(SystemExit) as exc:
        rfs.main()
    assert exc.value.code == 1


def test_main_calibration_failure_stops_before_any_frame(rig, monkeypatch):
    def fake_calib(*a, **k):
        raise unified.StageError("calibration boom")
    monkeypatch.setattr(rfs, "compute_or_reuse_calibration", fake_calib)
    frame_calls = []
    monkeypatch.setattr(unified, "stage_production", lambda *a, **k: frame_calls.append("stage_production"))
    monkeypatch.setattr(sys, "argv", base_argv(rig))

    with pytest.raises(SystemExit) as exc:
        rfs.main()
    assert exc.value.code == 1
    assert frame_calls == []


def test_main_runs_full_per_frame_pipeline_in_order_for_each_resolved_time(rig, monkeypatch):
    monkeypatch.setattr(rfs, "compute_or_reuse_calibration",
                         lambda *a, **k: (Path("/transforms_refined.json"), Path("/sync.json")))
    calls = []
    monkeypatch.setattr(unified, "stage_production", lambda *a, **k: calls.append("stage_production"))
    monkeypatch.setattr(rfs.hloc_mod, "restructure_flat_to_percam",
                         lambda *a, **k: calls.append("restructure_flat_to_percam"))
    monkeypatch.setattr(unified, "stage_masks", lambda *a, **k: calls.append("stage_masks"))
    monkeypatch.setattr(unified, "stage_branch_direct", lambda *a, **k: calls.append("stage_branch_direct"))

    monkeypatch.setattr(sys, "argv", base_argv(rig, start_time="1.0s", stop_time="1.1s", fps="10"))
    rfs.main()

    one_frame = ["stage_production", "restructure_flat_to_percam", "stage_masks", "stage_branch_direct"]
    assert calls == one_frame * 2  # two resolved times: 1.0s, 1.1s


def test_main_sets_transforms_refined_and_per_frame_run_name(rig, monkeypatch):
    monkeypatch.setattr(rfs, "compute_or_reuse_calibration",
                         lambda *a, **k: (Path("/fixed_transforms.json"), Path("/sync.json")))
    captured_L = []

    def fake_stage_production(args, L, image_ext, sync_json):
        captured_L.append((L["transforms_refined"], args.run_name))
    monkeypatch.setattr(unified, "stage_production", fake_stage_production)
    monkeypatch.setattr(rfs.hloc_mod, "restructure_flat_to_percam", lambda *a, **k: None)
    monkeypatch.setattr(unified, "stage_masks", lambda *a, **k: None)
    monkeypatch.setattr(unified, "stage_branch_direct", lambda *a, **k: None)

    monkeypatch.setattr(sys, "argv", base_argv(rig, run_name="myrun"))
    rfs.main()

    transforms_used, run_name_used = captured_L[0]
    assert transforms_used == Path("/fixed_transforms.json")
    assert run_name_used == "myrun_frame_0000"


def test_main_frame_failure_stops_remaining_frames(rig, monkeypatch):
    monkeypatch.setattr(rfs, "compute_or_reuse_calibration",
                         lambda *a, **k: (Path("/transforms_refined.json"), Path("/sync.json")))
    calls = []
    production_call_count = []

    def failing_stage_production(args, L, image_ext, sync_json):
        production_call_count.append(1)
        calls.append("stage_production")
        if len(production_call_count) == 2:  # fail on the second frame
            raise unified.StageError("frame boom")
    monkeypatch.setattr(unified, "stage_production", failing_stage_production)
    monkeypatch.setattr(rfs.hloc_mod, "restructure_flat_to_percam", lambda *a, **k: None)
    monkeypatch.setattr(unified, "stage_masks", lambda *a, **k: calls.append("stage_masks"))
    monkeypatch.setattr(unified, "stage_branch_direct", lambda *a, **k: calls.append("stage_branch_direct"))

    monkeypatch.setattr(sys, "argv", base_argv(rig, start_time="1.0s", stop_time="1.2s", fps="10"))
    with pytest.raises(SystemExit) as exc:
        rfs.main()
    assert exc.value.code == 1
    # 3 resolved times (1.0, 1.1, 1.2s); stops right after the 2nd frame's
    # stage_production failure -- the 3rd frame must never start.
    assert calls == ["stage_production", "stage_masks", "stage_branch_direct", "stage_production"]
