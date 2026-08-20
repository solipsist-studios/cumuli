# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import run_unified_pipeline as unified


# --------------------------------------------------------------------------
# validate_stage_output.py wiring: main() calls it (via run_script) after
# every stage unless --no_validate is given, and a validation failure must
# stop the pipeline exactly like any other StageError. run_script itself is
# monkeypatched here (not the real subprocess/conda call) since this is
# testing the wiring, not validate_stage_output.py's own logic (see
# tests/test_validate_stage_output.py for that, with synthetic tmp_path
# fixtures).
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
        "--target_time", "1.5s",
        "--no_validate",
    ]
    for k, v in extra.items():
        argv += [f"--{k}", str(v)]
    return argv


def patch_all_stages(monkeypatch, calls):
    def make_stage(name):
        def _stage(*args, **kwargs):
            calls.append(name)
            if name == "stage_sync":
                return args[1]["sync_offsets"]
        return _stage

    for name in ("stage_sync", "stage_production", "stage_poses", "stage_masks",
                 "stage_dataset4d", "stage_train4d"):
        monkeypatch.setattr(unified, name, make_stage(name))


def test_main_validates_every_stage_by_default(rig, monkeypatch):
    stage_calls, validate_calls = [], []
    patch_all_stages(monkeypatch, stage_calls)

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        if script_name == "validate_stage_output.py":
            validate_calls.append(args[args.index("--stage") + 1])
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    argv = [
        "prog", "--video_dir", str(rig["video_dir"]), "--calib_dir", str(rig["calib_dir"]),
        "--out_dir", str(rig["out_dir"]), "--target_time", "1.5s",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    unified.main()

    assert validate_calls == ["sync", "production", "poses", "masks", "dataset4d", "train4d"]


def test_main_start_from_stage_does_not_validate_skipped_stages(rig, monkeypatch):
    # Regression test: validating a stage that was SKIPPED (not just resumed
    # past) is wrong when a later stage's own side effects have already
    # mutated that earlier stage's on-disk output in a prior run (e.g.
    # stage_poses' run_hloc() call restructuring production_undistorted/'s
    # flat layout into per-camera subdirs) -- caught by an actual smoke test
    # where --start_from_stage masks falsely failed re-validating 'production'.
    stage_calls, validate_calls = [], []
    patch_all_stages(monkeypatch, stage_calls)
    rig["out_dir"].mkdir(parents=True)

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        if script_name == "validate_stage_output.py":
            validate_calls.append(args[args.index("--stage") + 1])
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    argv = [
        "prog", "--video_dir", str(rig["video_dir"]), "--calib_dir", str(rig["calib_dir"]),
        "--out_dir", str(rig["out_dir"]), "--target_time", "1.5s",
        "--start_from_stage", "poses",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    unified.main()

    assert stage_calls == ["stage_poses", "stage_masks", "stage_dataset4d", "stage_train4d"]
    assert validate_calls == ["poses", "masks", "dataset4d", "train4d"]


def test_main_no_validate_skips_validation(rig, monkeypatch):
    patch_all_stages(monkeypatch, [])
    validate_calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        if script_name == "validate_stage_output.py":
            validate_calls.append(args)
    monkeypatch.setattr(unified, "run_script", fake_run_script)
    monkeypatch.setattr(sys, "argv", base_argv(rig))  # base_argv already passes --no_validate

    unified.main()

    assert validate_calls == []


def test_main_validation_failure_stops_pipeline(rig, monkeypatch):
    stage_calls = []
    patch_all_stages(monkeypatch, stage_calls)

    def failing_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        if script_name == "validate_stage_output.py" and "sync" in args:
            raise unified.StageError("validate_stage_output.py (sync) failed (exit 1)")
    monkeypatch.setattr(unified, "run_script", failing_run_script)

    argv = [
        "prog", "--video_dir", str(rig["video_dir"]), "--calib_dir", str(rig["calib_dir"]),
        "--out_dir", str(rig["out_dir"]), "--target_time", "1.5s",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        unified.main()

    assert exc_info.value.code == 1
    # production must never run -- the pipeline stopped right after sync's
    # (failed) validation, before starting the next stage.
    assert "stage_production" not in stage_calls


# --------------------------------------------------------------------------
# resolve_conda -- shutil.which first, then well-known install paths
# --------------------------------------------------------------------------

def test_resolve_conda_uses_shutil_which_first(monkeypatch):
    monkeypatch.setattr(unified.shutil, "which", lambda name: "/usr/bin/conda")
    assert unified.resolve_conda() == "/usr/bin/conda"


def test_resolve_conda_falls_back_to_known_candidate_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(unified.shutil, "which", lambda name: None)
    fake_conda = tmp_path / "conda"
    fake_conda.touch()
    monkeypatch.setattr(unified, "_CONDA_CANDIDATES", [tmp_path / "nope", fake_conda])
    assert unified.resolve_conda() == str(fake_conda)


def test_resolve_conda_raises_when_nothing_found(monkeypatch, tmp_path):
    monkeypatch.setattr(unified.shutil, "which", lambda name: None)
    monkeypatch.setattr(unified, "_CONDA_CANDIDATES", [tmp_path / "nope1", tmp_path / "nope2"])
    with pytest.raises(unified.StageError, match="Could not locate"):
        unified.resolve_conda()


# --------------------------------------------------------------------------
# run_script -- the one function every stage funnels through
# --------------------------------------------------------------------------

class FakeCompletedProcess:
    def __init__(self, returncode):
        self.returncode = returncode


def test_run_script_missing_script_raises_stage_error(tmp_path, monkeypatch):
    monkeypatch.setattr(unified, "SCRIPTS_DIR", tmp_path)
    with pytest.raises(unified.StageError, match="not found"):
        unified.run_script("nonexistent.py", [])


def test_run_script_without_conda_env_uses_sys_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(unified, "SCRIPTS_DIR", tmp_path)
    (tmp_path / "foo.py").touch()
    captured = {}

    def fake_run(cmd, cwd=None, env=None):
        captured.update(cmd=cmd, cwd=cwd, env=env)
        return FakeCompletedProcess(0)
    monkeypatch.setattr(unified.subprocess, "run", fake_run)

    unified.run_script("foo.py", ["--a", 1])
    assert captured["cmd"] == [sys.executable, str(tmp_path / "foo.py"), "--a", "1"]


def test_run_script_with_conda_env_wraps_command(tmp_path, monkeypatch):
    monkeypatch.setattr(unified, "SCRIPTS_DIR", tmp_path)
    (tmp_path / "foo.py").touch()
    monkeypatch.setattr(unified, "resolve_conda", lambda: "/opt/conda/bin/conda")
    captured = {}

    def fake_run(cmd, cwd=None, env=None):
        captured["cmd"] = cmd
        return FakeCompletedProcess(0)
    monkeypatch.setattr(unified.subprocess, "run", fake_run)

    unified.run_script("foo.py", [], conda_env="hloc")
    assert captured["cmd"] == ["/opt/conda/bin/conda", "run", "-n", "hloc", "--no-capture-output",
                                "python3", str(tmp_path / "foo.py")]


def test_run_script_passes_fresh_env_copy_not_mutating_process_env(tmp_path, monkeypatch):
    monkeypatch.setattr(unified, "SCRIPTS_DIR", tmp_path)
    (tmp_path / "foo.py").touch()
    monkeypatch.setenv("SOME_MARKER_VAR", "original")
    captured = {}

    def fake_run(cmd, cwd=None, env=None):
        captured["env"] = dict(env)  # snapshot before the simulated child mutates its copy
        env["SOME_MARKER_VAR"] = "mutated_by_child"
        return FakeCompletedProcess(0)
    monkeypatch.setattr(unified.subprocess, "run", fake_run)

    unified.run_script("foo.py", [], extra_env={"NEW_VAR": "1"})
    assert captured["env"]["SOME_MARKER_VAR"] == "original"
    assert captured["env"]["NEW_VAR"] == "1"
    assert os.environ["SOME_MARKER_VAR"] == "original"  # this process's own env is untouched


def test_run_script_nonzero_exit_raises_stage_error(tmp_path, monkeypatch):
    monkeypatch.setattr(unified, "SCRIPTS_DIR", tmp_path)
    (tmp_path / "foo.py").touch()
    monkeypatch.setattr(unified.subprocess, "run", lambda *a, **k: FakeCompletedProcess(7))
    with pytest.raises(unified.StageError, match=r"failed \(exit 7\)"):
        unified.run_script("foo.py", [], label="foo label")


def test_run_script_zero_exit_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(unified, "SCRIPTS_DIR", tmp_path)
    (tmp_path / "foo.py").touch()
    monkeypatch.setattr(unified.subprocess, "run", lambda *a, **k: FakeCompletedProcess(0))
    unified.run_script("foo.py", [])  # must not raise


# --------------------------------------------------------------------------
# check_expected
# --------------------------------------------------------------------------

def test_check_expected_warns_when_path_missing(tmp_path, capsys):
    unified.check_expected(tmp_path / "missing", "sync")
    assert "does not exist" in capsys.readouterr().out


def test_check_expected_silent_when_path_exists(tmp_path, capsys):
    p = tmp_path / "exists"
    p.touch()
    unified.check_expected(p, "sync")
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# parse_target_time
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("1.5", 1.5), ("1.5s", 1.5), ("1500ms", 1.5), ("2", 2.0), ("2S", 2.0), ("2000MS", 2.0),
])
def test_parse_target_time(value, expected):
    assert unified.parse_target_time(value) == pytest.approx(expected)


# --------------------------------------------------------------------------
# discover_cameras
# --------------------------------------------------------------------------

def test_discover_cameras_finds_and_sorts_videos_case_insensitively(tmp_path):
    for name in ["0003.mp4", "0001.MP4", "0002.mov"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    videos = unified.discover_cameras(tmp_path)
    assert [v.name for v in videos] == ["0001.MP4", "0002.mov", "0003.mp4"]


def test_discover_cameras_raises_when_none_found(tmp_path):
    with pytest.raises(unified.StageError, match="No videos found"):
        unified.discover_cameras(tmp_path)


# --------------------------------------------------------------------------
# build_layout -- pure path derivation
# --------------------------------------------------------------------------

def test_build_layout_derives_paths_under_out_dir(tmp_path):
    L = unified.build_layout(tmp_path)
    assert L["sync_offsets"] == tmp_path / "sync_offsets.json"
    assert L["flat_transforms"] == tmp_path / "transforms.json"
    assert L["dataset4d"] == tmp_path / "dataset_4dgs"
    assert L["sogst_out"] == tmp_path / "splat_4d.sogst"


# --------------------------------------------------------------------------
# undistort_args / keypoint_args -- optional-flag wiring
# --------------------------------------------------------------------------

def test_undistort_args_without_target_pkl_dir():
    args = NS(calib_dir=Path("/calib"), target_pkl_dir=None)
    a = unified.undistort_args(args, Path("/frames"), Path("/out"), Path("/pkl"), ".jpg")
    assert "--target_pkl_dir" not in a


def test_undistort_args_with_target_pkl_dir():
    args = NS(calib_dir=Path("/calib"), target_pkl_dir=Path("/target"))
    a = unified.undistort_args(args, Path("/frames"), Path("/out"), Path("/pkl"), ".jpg")
    assert a[-2:] == ["--target_pkl_dir", Path("/target")]


def test_keypoint_args_without_checkpoint_root():
    args = NS(sapiens_checkpoint_root=None, sapiens_model_size="1b")
    a = unified.keypoint_args(args, Path("/img"), Path("/kp2d"), Path("/fmask"))
    assert "--sapiens_checkpoint_root" not in a


def test_keypoint_args_with_checkpoint_root():
    args = NS(sapiens_checkpoint_root=Path("/ckpt"), sapiens_model_size="1b")
    a = unified.keypoint_args(args, Path("/img"), Path("/kp2d"), Path("/fmask"))
    assert a[-2:] == ["--sapiens_checkpoint_root", Path("/ckpt")]


def test_keypoint_args_omits_model_size_at_default_1b():
    args = NS(sapiens_checkpoint_root=None, sapiens_model_size="1b")
    a = unified.keypoint_args(args, Path("/img"), Path("/kp2d"), Path("/fmask"))
    assert "--sapiens_model_size" not in a


def test_keypoint_args_forwards_non_default_model_size():
    # Real CI failure (2026-07-30): the actual GitHub runner has only
    # ~7.75GB RAM, and the 1b checkpoint alone peaks at 6.5-7.6GB --
    # --sapiens_model_size lets CPU mode use a smaller checkpoint instead
    # (0.4b measured at 4.1GB peak on the same real fixture).
    args = NS(sapiens_checkpoint_root=None, sapiens_model_size="0.4b")
    a = unified.keypoint_args(args, Path("/img"), Path("/kp2d"), Path("/fmask"))
    assert a[-2:] == ["--sapiens_model_size", "0.4b"]


# --------------------------------------------------------------------------
# run_hloc
# --------------------------------------------------------------------------

def test_run_hloc_builds_correct_argv_and_conda_env(monkeypatch, tmp_path):
    captured = {}

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        captured.update(script_name=script_name, args=[str(a) for a in args],
                         conda_env=conda_env, label=label)
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    args = NS(hloc_env="cumuli", multiframe_sfm_script=Path("/mfs.py"), hloc_feature_type="aliked",
              hloc_resize_max=2048, hloc_max_keypoints=4096)
    out = unified.run_hloc(args, Path("/undist"), Path("/pkl"), tmp_path / "outputs", ".jpg", "mylabel")

    assert captured["script_name"] == "run_hloc.py"
    assert captured["conda_env"] == "cumuli"
    assert captured["label"] == "mylabel"
    assert "aliked" in captured["args"]
    assert "2048" in captured["args"]
    assert "4096" in captured["args"]
    assert out == tmp_path / "outputs" / "transforms_multiframe.json"


# --------------------------------------------------------------------------
# prepare_candidate_window -- per-instant pipeline + on-disk reuse
# --------------------------------------------------------------------------

def _window_args(tmp_path, pp3_dir=None):
    return NS(hloc_env="cumuli", diffuman4d_env="cumuli", video_dir=tmp_path / "videos", pp3_dir=pp3_dir, target_time_s=1.5,
              generic_env="queen", sapiens_env="sapiens2", sapiens_checkpoint_root=None,
              sapiens_model_size="1b", calib_dir=Path("/calib"), target_pkl_dir=None)


def test_prepare_candidate_window_runs_full_per_frame_pipeline_in_order(monkeypatch, tmp_path):
    calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        calls.append(script_name)
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    poses2d_dir = tmp_path / "poses2d"
    dirs = unified.prepare_candidate_window(
        _window_args(tmp_path), {}, tmp_path / "sync.json", 2, ".jpg",
        tmp_path / "raw", tmp_path / "undist", tmp_path / "pkl",
        tmp_path / "fmasks", tmp_path / "kp2d", poses2d_dir, tag="test",
    )
    assert calls == [
        "extract_synced_frames.py",
        "undistort_frames.py", "generate_masks.py", "predict_keypoints_2d.py", "split_keypoints_per_camera.py",
        "undistort_frames.py", "generate_masks.py", "predict_keypoints_2d.py", "split_keypoints_per_camera.py",
    ]
    assert dirs == [poses2d_dir / "f0", poses2d_dir / "f1"]


def test_prepare_candidate_window_reuses_already_processed_frame(monkeypatch, tmp_path):
    calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        calls.append(script_name)
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    # window=1 uses the flat layout (no f0/ subdir) -- matches
    # extract_synced_frames.py's own --window==1 special case, see
    # _instant_dirs's docstring for the real bug this guards against.
    poses2d_dir = tmp_path / "poses2d"
    poses2d_dir.mkdir(parents=True)
    (poses2d_dir / "Camera_0000.json").write_text("{}")

    dirs = unified.prepare_candidate_window(
        _window_args(tmp_path), {}, tmp_path / "sync.json", 1, ".jpg",
        tmp_path / "raw", tmp_path / "undist", tmp_path / "pkl",
        tmp_path / "fmasks", tmp_path / "kp2d", poses2d_dir, tag="test",
    )
    # extract_synced_frames.py always runs; the per-frame pipeline is
    # skipped entirely for the already-processed frame
    assert calls == ["extract_synced_frames.py"]
    assert dirs == [poses2d_dir]


def test_prepare_candidate_window_window_1_uses_flat_layout_not_f0_subdir(monkeypatch, tmp_path):
    """Real bug found running this locally (2026-07-30): --sync_window 1
    always assumed the f0/../f{N-1}/ subdir layout that extract_synced_frames.py
    only produces for window>1 -- window==1 writes flat instead, so the
    orchestrator looked for a dir that was never created and crashed."""
    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        pass
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    dirs = unified.prepare_candidate_window(
        _window_args(tmp_path), {}, tmp_path / "sync.json", 1, ".jpg",
        tmp_path / "raw", tmp_path / "undist", tmp_path / "pkl",
        tmp_path / "fmasks", tmp_path / "kp2d", tmp_path / "poses2d", tag="test",
    )
    assert dirs == [tmp_path / "poses2d"]


def test_prepare_candidate_window_forwards_pp3_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        if script_name == "extract_synced_frames.py":
            captured["args"] = [str(a) for a in args]
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    unified.prepare_candidate_window(
        _window_args(tmp_path, pp3_dir=tmp_path / "pp3"), {}, tmp_path / "sync.json", 0, ".jpg",
        tmp_path / "raw", tmp_path / "undist", tmp_path / "pkl",
        tmp_path / "fmasks", tmp_path / "kp2d", tmp_path / "poses2d", tag="test",
    )
    assert "--pp3_dir" in captured["args"]


def test_prepare_candidate_window_omits_pp3_dir_when_not_given(monkeypatch, tmp_path):
    captured = {}

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        if script_name == "extract_synced_frames.py":
            captured["args"] = [str(a) for a in args]
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    unified.prepare_candidate_window(
        _window_args(tmp_path, pp3_dir=None), {}, tmp_path / "sync.json", 0, ".jpg",
        tmp_path / "raw", tmp_path / "undist", tmp_path / "pkl",
        tmp_path / "fmasks", tmp_path / "kp2d", tmp_path / "poses2d", tag="test",
    )
    assert "--pp3_dir" not in captured["args"]


# --------------------------------------------------------------------------
# stage_sync
# --------------------------------------------------------------------------

def test_stage_sync_uses_initial_sync_json_when_given(monkeypatch, tmp_path):
    calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        calls.append(script_name)
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    initial = tmp_path / "initial_sync.json"
    initial.write_text("{}")
    args = NS(initial_sync_json=initial, video_dir=tmp_path / "videos", out_dir=out_dir,
              target_time_s=1.5, generic_env="queen", pp3_dir=None)
    L = unified.build_layout(out_dir)

    result = unified.stage_sync(args, L, ".jpg")
    assert "compute_sync_offsets.py" not in calls
    assert result == L["sync_offsets"]
    assert L["sync_offsets"].read_text() == "{}"


def test_stage_sync_computes_offsets_when_no_initial_json(monkeypatch, tmp_path):
    calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        calls.append(script_name)
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = NS(initial_sync_json=None, video_dir=tmp_path / "videos", out_dir=out_dir,
              target_time_s=1.5, generic_env="queen", pp3_dir=None)
    L = unified.build_layout(out_dir)

    unified.stage_sync(args, L, ".jpg")
    assert calls == ["compute_sync_offsets.py", "extract_synced_frames.py", "make_sync_grid.py"]


# --------------------------------------------------------------------------
# stage_production
# --------------------------------------------------------------------------

def test_stage_production_forwards_pp3_dir_and_calls_undistort(monkeypatch, tmp_path):
    calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        calls.append((script_name, [str(a) for a in args]))
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    args = NS(video_dir=tmp_path / "videos", pp3_dir=tmp_path / "pp3", target_time_s=1.5,
              generic_env="queen", calib_dir=Path("/calib"), target_pkl_dir=None)
    L = unified.build_layout(tmp_path / "out")
    unified.stage_production(args, L, ".jpg", tmp_path / "sync.json")

    names = [c[0] for c in calls]
    assert names == ["extract_synced_frames.py", "undistort_frames.py"]
    assert "--pp3_dir" in calls[0][1]


# --------------------------------------------------------------------------
# stage_poses
# --------------------------------------------------------------------------

def _poses_args(sync_window=2):
    return NS(hloc_env="cumuli", diffuman4d_env="cumuli", video_dir=Path("/videos"), pp3_dir=None, target_time_s=1.5, sync_window=sync_window,
              generic_env="queen", sapiens_env="sapiens2", sapiens_checkpoint_root=None,
              sapiens_model_size="1b", calib_dir=Path("/calib"), target_pkl_dir=None,
              multiframe_sfm_script=Path("/mfs.py"), hloc_feature_type="superpoint",
              hloc_resize_max=4096, hloc_max_keypoints=8192)


def test_stage_poses_runs_report_only_before_real_optimization(monkeypatch, tmp_path):
    calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        calls.append((script_name, [str(a) for a in args]))
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    L = unified.build_layout(tmp_path / "out")
    unified.stage_poses(_poses_args(), L, ".jpg", tmp_path / "sync.json")

    refine_calls = [c for c in calls if c[0] == "run_pose_refinement.py"]
    assert len(refine_calls) == 2
    assert "--report_only" in refine_calls[0][1]
    assert "--report_only" not in refine_calls[1][1]
    assert "run_hloc.py" in [c[0] for c in calls]


def test_stage_poses_reuses_cached_candidate_window(monkeypatch, tmp_path):
    calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        calls.append(script_name)
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    L = unified.build_layout(tmp_path / "out")
    f0 = L["cand_corr_poses2d"] / "f0"
    f0.mkdir(parents=True)
    (f0 / "Camera_0000.json").write_text("{}")

    unified.stage_poses(_poses_args(sync_window=1), L, ".jpg", tmp_path / "sync.json")
    assert "extract_synced_frames.py" not in calls


def test_stage_poses_reuses_cached_hloc_output(monkeypatch, tmp_path):
    calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        calls.append(script_name)
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    L = unified.build_layout(tmp_path / "out")
    f0 = L["cand_corr_poses2d"] / "f0"
    f0.mkdir(parents=True)
    (f0 / "Camera_0000.json").write_text("{}")
    L["hloc_final"].mkdir(parents=True)
    (L["hloc_final"] / "transforms_multiframe.json").write_text("{}")

    unified.stage_poses(_poses_args(sync_window=1), L, ".jpg", tmp_path / "sync.json")
    assert "run_hloc.py" not in calls


# --------------------------------------------------------------------------
# stage_masks
# --------------------------------------------------------------------------

def _masks_args():
    return NS(hloc_env="cumuli", diffuman4d_env="cumuli", generic_env="queen", sapiens_env="sapiens2", sapiens_checkpoint_root=None,
              sapiens_model_size="1b", triangulate_env="queen")


def test_stage_masks_runs_everything_when_nothing_cached(monkeypatch, tmp_path):
    calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        calls.append(script_name)
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    L = unified.build_layout(tmp_path / "out")
    unified.stage_masks(_masks_args(), L, ".jpg")

    assert calls == [
        "build_flat_dataset.py", "generate_masks.py", "predict_keypoints_2d.py",
        "split_keypoints_per_camera.py", "clean_masks.py", "triangulate_and_project_keypoints.py",
    ]


def test_stage_masks_reuses_all_cached_steps_but_always_triangulates(monkeypatch, tmp_path):
    calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        calls.append(script_name)
    monkeypatch.setattr(unified, "run_script", fake_run_script)

    L = unified.build_layout(tmp_path / "out")
    L["flat_transforms"].parent.mkdir(parents=True, exist_ok=True)
    L["flat_transforms"].write_text("{}")
    L["flat_images"].mkdir(parents=True)
    (L["flat_images"] / "00.png").write_bytes(b"")
    L["flat_fmasks"].mkdir(parents=True)
    (L["flat_fmasks"] / "00.png").write_bytes(b"")
    L["flat_poses2d"].mkdir(parents=True)
    (L["flat_poses2d"] / "00.json").write_text("{}")
    L["flat_fmasks_clean"].mkdir(parents=True)
    (L["flat_fmasks_clean"] / "00.png").write_bytes(b"")

    unified.stage_masks(_masks_args(), L, ".jpg")
    assert calls == ["triangulate_and_project_keypoints.py"]


# --------------------------------------------------------------------------
# build_parser -- required args
# --------------------------------------------------------------------------

_BASE_CLI = ["--video_dir", "/v", "--calib_dir", "/c", "--out_dir", "/o", "--target_time", "1s"]


@pytest.mark.parametrize("missing", ["--video_dir", "--calib_dir", "--out_dir", "--target_time"])
def test_build_parser_errors_when_required_arg_missing(missing):
    argv = list(_BASE_CLI)
    idx = argv.index(missing)
    del argv[idx:idx + 2]
    parser = unified.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 2


def test_build_parser_rejects_invalid_start_from_stage_choice():
    parser = unified.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(_BASE_CLI + ["--start_from_stage", "not_a_real_stage"])
    assert exc.value.code == 2


# --------------------------------------------------------------------------
# apply_config_defaults
# --------------------------------------------------------------------------

def test_apply_config_defaults_overrides_configurable_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"generic_env": "custom_env", "trainer_env": "omg4b"}))
    parser = unified.build_parser()
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path)])

    unified.apply_config_defaults(parser)
    args = parser.parse_args(_BASE_CLI)
    assert args.generic_env == "custom_env"
    assert args.trainer_env == "omg4b"


def test_apply_config_defaults_explicit_cli_flag_wins_over_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"generic_env": "custom_env"}))
    parser = unified.build_parser()
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path)])

    unified.apply_config_defaults(parser)
    args = parser.parse_args(_BASE_CLI + ["--generic_env", "explicit_env"])
    assert args.generic_env == "explicit_env"


def test_apply_config_defaults_rejects_unknown_key(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"not_a_real_key": "x"}))
    parser = unified.build_parser()
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path)])

    with pytest.raises(SystemExit) as exc:
        unified.apply_config_defaults(parser)
    assert exc.value.code == 1
    assert "unrecognized key" in capsys.readouterr().out


def test_apply_config_defaults_invalid_json_errors_cleanly(tmp_path, monkeypatch, capsys):
    # Regression test: this used to raise a raw json.JSONDecodeError traceback.
    config_path = tmp_path / "config.json"
    config_path.write_text("{not valid json")
    parser = unified.build_parser()
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path)])

    with pytest.raises(SystemExit) as exc:
        unified.apply_config_defaults(parser)
    assert exc.value.code == 1
    assert "not valid JSON" in capsys.readouterr().out


def test_apply_config_defaults_missing_config_file_errors_cleanly(tmp_path, monkeypatch):
    parser = unified.build_parser()
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(tmp_path / "nope.json")])
    with pytest.raises(SystemExit) as exc:
        unified.apply_config_defaults(parser)
    assert exc.value.code == 1


def test_apply_config_defaults_is_noop_when_no_config_given(monkeypatch):
    parser = unified.build_parser()
    monkeypatch.setattr(sys, "argv", ["prog"] + _BASE_CLI)
    unified.apply_config_defaults(parser)  # must not raise
    args = parser.parse_args(_BASE_CLI)
    assert args.generic_env == "cumuli"  # untouched built-in default


# --------------------------------------------------------------------------
# main() -- top-level validation, resume logic, and failure propagation
# --------------------------------------------------------------------------

def test_main_errors_when_video_dir_not_a_directory(tmp_path, monkeypatch):
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    argv = ["prog", "--video_dir", str(tmp_path / "nope"), "--calib_dir", str(calib_dir),
            "--out_dir", str(tmp_path / "out"), "--target_time", "1s"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        unified.main()
    assert exc.value.code == 1


def test_main_errors_when_calib_dir_not_a_directory(rig, monkeypatch):
    argv = ["prog", "--video_dir", str(rig["video_dir"]), "--calib_dir", str(rig["out_dir"] / "nope"),
            "--out_dir", str(rig["out_dir"]), "--target_time", "1s"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        unified.main()
    assert exc.value.code == 1


def test_main_errors_when_target_pkl_dir_given_but_not_a_directory(rig, monkeypatch):
    monkeypatch.setattr(sys, "argv", base_argv(rig, target_pkl_dir=str(rig["out_dir"] / "nope")))
    with pytest.raises(SystemExit) as exc:
        unified.main()
    assert exc.value.code == 1


def test_main_errors_when_initial_sync_json_given_but_missing(rig, monkeypatch):
    monkeypatch.setattr(sys, "argv", base_argv(rig, initial_sync_json=str(rig["out_dir"] / "nope.json")))
    with pytest.raises(SystemExit) as exc:
        unified.main()
    assert exc.value.code == 1


def test_main_errors_when_multiframe_sfm_script_missing(rig, monkeypatch):
    monkeypatch.setattr(sys, "argv",
                         base_argv(rig, multiframe_sfm_script="/definitely/not/a/real/path.py"))
    with pytest.raises(SystemExit) as exc:
        unified.main()
    assert exc.value.code == 1


def test_main_errors_when_no_videos_found(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()  # empty -- no video files
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    argv = ["prog", "--video_dir", str(video_dir), "--calib_dir", str(calib_dir),
            "--out_dir", str(tmp_path / "out"), "--target_time", "1s", "--no_validate"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        unified.main()
    assert exc.value.code == 1


def test_main_errors_when_stop_after_stage_precedes_start_from_stage(rig, monkeypatch):
    # Regression test: --start_from_stage masks --stop_after_stage production
    # (stop earlier than start) used to silently run NOTHING while printing a
    # "STOPPED after stage 'production'" banner and exiting 0 -- looked like a
    # successful partial run but not a single stage, or even validation, ever
    # executed.
    stage_calls = []
    patch_all_stages(monkeypatch, stage_calls)
    monkeypatch.setattr(unified, "run_script", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", base_argv(
        rig, start_from_stage="masks", stop_after_stage="production"))

    with pytest.raises(SystemExit) as exc:
        unified.main()
    assert exc.value.code == 1
    assert stage_calls == []


def test_main_stop_after_stage_equal_to_start_from_stage_runs_that_one_stage(rig, monkeypatch):
    # Boundary case: stop == start (not stop < start) must still work --
    # only the equality-based "stop before start" ordering is invalid.
    stage_calls = []
    patch_all_stages(monkeypatch, stage_calls)
    monkeypatch.setattr(unified, "run_script", lambda *a, **k: None)
    rig["out_dir"].mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", base_argv(
        rig, start_from_stage="production", stop_after_stage="production"))

    unified.main()
    assert stage_calls == ["stage_production"]


def test_main_stop_after_first_stage_halts_immediately(rig, monkeypatch):
    stage_calls = []
    patch_all_stages(monkeypatch, stage_calls)
    monkeypatch.setattr(unified, "run_script", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", base_argv(rig, stop_after_stage="sync"))

    unified.main()
    assert stage_calls == ["stage_sync"]


def test_main_start_from_last_stage_skips_every_earlier_stage(rig, monkeypatch, capsys):
    stage_calls = []
    patch_all_stages(monkeypatch, stage_calls)
    monkeypatch.setattr(unified, "run_script", lambda *a, **k: None)
    rig["out_dir"].mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", base_argv(rig, start_from_stage="train4d"))

    unified.main()
    assert stage_calls == ["stage_train4d"]
    out = capsys.readouterr().out
    for key in ("sync", "production", "poses", "masks", "dataset4d"):
        assert f"Skipping stage '{key}'" in out


def test_main_stop_after_stage_halts_before_next_stage(rig, monkeypatch):
    stage_calls = []
    patch_all_stages(monkeypatch, stage_calls)
    monkeypatch.setattr(unified, "run_script", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", base_argv(rig, stop_after_stage="production"))

    unified.main()
    assert stage_calls == ["stage_sync", "stage_production"]


@pytest.mark.parametrize("stop_stage,expected_calls", [
    ("poses", ["stage_sync", "stage_production", "stage_poses"]),
    ("masks", ["stage_sync", "stage_production", "stage_poses", "stage_masks"]),
])
def test_main_stop_after_stage_halts_at_each_remaining_stage(rig, monkeypatch, stop_stage, expected_calls):
    stage_calls = []
    patch_all_stages(monkeypatch, stage_calls)
    monkeypatch.setattr(unified, "run_script", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", base_argv(rig, stop_after_stage=stop_stage))

    unified.main()
    assert stage_calls == expected_calls


def test_main_stage_failure_stops_pipeline_before_later_stages(rig, monkeypatch):
    calls = []

    def make_stage(name):
        def _stage(*a, **k):
            calls.append(name)
            if name == "stage_production":
                raise unified.StageError("boom")
            if name == "stage_sync":
                return a[1]["sync_offsets"]
        return _stage
    for name in ("stage_sync", "stage_production", "stage_poses", "stage_masks",
                 "stage_dataset4d", "stage_train4d"):
        monkeypatch.setattr(unified, name, make_stage(name))
    monkeypatch.setattr(unified, "run_script", lambda *a, **k: None)  # validate_stage_output no-ops
    monkeypatch.setattr(sys, "argv", base_argv(rig))

    with pytest.raises(SystemExit) as exc:
        unified.main()
    assert exc.value.code == 1
    assert calls == ["stage_sync", "stage_production"]


def test_main_resume_without_resolved_sync_json_falls_back_to_default_path(rig, monkeypatch):
    stage_calls = []

    def make_stage(name):
        def _stage(*a, **k):
            stage_calls.append((name, a))
        return _stage
    for name in ("stage_production", "stage_poses", "stage_masks", "stage_train4d"):
        monkeypatch.setattr(unified, name, make_stage(name))
    monkeypatch.setattr(unified, "run_script", lambda *a, **k: None)
    rig["out_dir"].mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", base_argv(rig, start_from_stage="production"))

    unified.main()

    # stage_production(args, L, image_ext, sync_json) -- sync_json is the 4th
    # positional arg. It must fall back to the default L["sync_offsets"] path
    # since resolved_sync_json.txt was never written (no 'sync' stage ran in
    # this resumed session).
    _, prod_args = stage_calls[0]
    assert prod_args[3] == unified.build_layout(rig["out_dir"])["sync_offsets"]


# --------------------------------------------------------------------------
# stage_dataset4d / stage_train4d -- the 4D training path
# --------------------------------------------------------------------------

def _dataset4d_args(tmp_path, window=2, eval_camera=None, holdouts=()):
    return NS(diffuman4d_env="cumuli", video_dir=tmp_path / "videos", target_time_s=1.5, pp3_dir=None,
              train_window=window, train_fps=30.0, dataset_downscale=1,
              dataset_jobs=8, eval_camera=eval_camera,
              holdout_cameras=list(holdouts), generic_env="queen",
              sapiens_env="sapiens2", sapiens_checkpoint_root=None,
              sapiens_model_size="1b", calib_dir=tmp_path / "calib",
              target_pkl_dir=None, hull_min_views=9)


def _write_flat_transforms(L, n_cams=3):
    L["flat_transforms"].parent.mkdir(parents=True, exist_ok=True)
    L["flat_transforms"].write_text(json.dumps(
        {"frames": [{"camera_label": f"{i:02d}"} for i in range(n_cams)]}))


def _capture_runs(monkeypatch):
    calls = []

    def fake_run_script(script_name, args, conda_env=None, cwd=None, extra_env=None, label=None):
        calls.append({"script": str(script_name), "args": [str(a) for a in args],
                      "conda_env": conda_env, "cwd": cwd, "extra_env": extra_env})
    monkeypatch.setattr(unified, "run_script", fake_run_script)
    return calls


def test_stage_dataset4d_runs_per_frame_chain_then_builder(monkeypatch, tmp_path):
    calls = _capture_runs(monkeypatch)
    L = unified.build_layout(tmp_path / "out")
    _write_flat_transforms(L)
    unified.stage_dataset4d(_dataset4d_args(tmp_path, window=2), L, ".png",
                            tmp_path / "sync.json")

    scripts = [Path(c["script"]).name for c in calls]
    assert scripts[0] == "extract_synced_frames.py"
    # Per frame: undistort, assemble, masks, keypoints, split, clean.
    per_frame = ["undistort_frames.py", "assemble_flipbook_frame.py",
                 "generate_masks.py", "predict_keypoints_2d.py",
                 "split_keypoints_per_camera.py", "clean_masks.py"]
    assert scripts[1:13] == per_frame * 2
    assert scripts[13] == "build_flipbook_4dgs_dataset.py"

    extract = calls[0]["args"]
    assert "--window" in extract and extract[extract.index("--window") + 1] == "2"
    assert "--output_ext" in extract and extract[extract.index("--output_ext") + 1] == ".png"


def test_stage_dataset4d_skips_processed_frames(monkeypatch, tmp_path):
    calls = _capture_runs(monkeypatch)
    L = unified.build_layout(tmp_path / "out")
    _write_flat_transforms(L)
    done = L["flipbook_src"] / "frame_0000" / "fmasks_clean"
    done.mkdir(parents=True)
    (done / "00.png").write_bytes(b"")

    unified.stage_dataset4d(_dataset4d_args(tmp_path, window=2), L, ".png",
                            tmp_path / "sync.json")
    scripts = [Path(c["script"]).name for c in calls]
    # Frame 0 is reused: exactly one per-frame chain runs (frame 1).
    assert scripts.count("clean_masks.py") == 1


def test_stage_dataset4d_forwards_eval_and_mate_holdouts(monkeypatch, tmp_path):
    calls = _capture_runs(monkeypatch)
    L = unified.build_layout(tmp_path / "out")
    _write_flat_transforms(L)
    unified.stage_dataset4d(
        _dataset4d_args(tmp_path, window=1, eval_camera="05", holdouts=["05", "06"]),
        L, ".png", tmp_path / "sync.json")

    build = next(c for c in calls if Path(c["script"]).name == "build_flipbook_4dgs_dataset.py")
    a = build["args"]
    assert a[a.index("--test_cameras") + 1] == "05"
    # The eval camera itself is stripped from the holdout list; only the
    # mate remains excluded-but-unscored.
    assert a[a.index("--holdout_cameras") + 1] == "06"
    assert a.count("05") == 1


def test_stage_dataset4d_omits_eval_flags_without_eval_camera(monkeypatch, tmp_path):
    calls = _capture_runs(monkeypatch)
    L = unified.build_layout(tmp_path / "out")
    _write_flat_transforms(L)
    unified.stage_dataset4d(_dataset4d_args(tmp_path, window=1), L, ".png",
                            tmp_path / "sync.json")
    build = next(c for c in calls if Path(c["script"]).name == "build_flipbook_4dgs_dataset.py")
    assert "--test_cameras" not in build["args"]
    assert "--holdout_cameras" not in build["args"]


def test_stage_dataset4d_clamps_hull_min_views_to_rig_size(monkeypatch, tmp_path):
    calls = _capture_runs(monkeypatch)
    L = unified.build_layout(tmp_path / "out")
    _write_flat_transforms(L, n_cams=3)
    unified.stage_dataset4d(_dataset4d_args(tmp_path, window=1), L, ".png",
                            tmp_path / "sync.json")
    build = next(c for c in calls if Path(c["script"]).name == "build_flipbook_4dgs_dataset.py")
    a = build["args"]
    # The builder default of 9 exceeds a 3-camera rig: the hull would be
    # empty by construction (caught live in CPU CI, 2026-08-20).
    assert a[a.index("--hull_min_views") + 1] == "3"


def test_stage_dataset4d_keeps_hull_min_views_on_a_full_rig(monkeypatch, tmp_path):
    calls = _capture_runs(monkeypatch)
    L = unified.build_layout(tmp_path / "out")
    _write_flat_transforms(L, n_cams=11)
    unified.stage_dataset4d(_dataset4d_args(tmp_path, window=1), L, ".png",
                            tmp_path / "sync.json")
    build = next(c for c in calls if Path(c["script"]).name == "build_flipbook_4dgs_dataset.py")
    a = build["args"]
    assert a[a.index("--hull_min_views") + 1] == "9"


def _train4d_args(tmp_path, iters=200, t_init_div=100, trainer_config=None,
                  eval_camera="05", skip_eval=False):
    repo = tmp_path / "omg4repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "train_scratch.py").write_text("# stub")
    return NS(trainer_repo=repo, trainer_env="omg4", total_train_iters=iters,
              train_window=8, train_fps=30.0, t_init_div=t_init_div,
              num_pts=100000, batch_size4d=2, densify_until_iter=150,
              densify_until_num_points=1000, trainer_config=trainer_config,
              eval_camera=eval_camera, eval_every=10, skip_eval=skip_eval)


def test_stage_train4d_raises_without_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(unified, "gpu_available", lambda: False)
    with pytest.raises(unified.StageError, match="CUDA GPU"):
        unified.stage_train4d(_train4d_args(tmp_path), unified.build_layout(tmp_path / "out"))


def test_stage_train4d_generates_config_and_trains(monkeypatch, tmp_path):
    monkeypatch.setattr(unified, "gpu_available", lambda: True)
    calls = _capture_runs(monkeypatch)
    args = _train4d_args(tmp_path, iters=200)
    L = unified.build_layout(tmp_path / "out")
    L["train4d_config"].parent.mkdir(parents=True, exist_ok=True)

    # Training "succeeds": the fake run_script drops the checkpoint the
    # stage asserts on afterwards.
    real_capture = unified.run_script

    def run_and_checkpoint(script_name, a, **kw):
        real_capture(script_name, a, **kw)
        if Path(str(script_name)).name == "train_scratch.py":
            L["train4d_model"].mkdir(parents=True, exist_ok=True)
            (L["train4d_model"] / "chkpnt200.pth").write_bytes(b"x")
    monkeypatch.setattr(unified, "run_script", run_and_checkpoint)

    unified.stage_train4d(args, L)

    config = L["train4d_config"].read_text()
    assert "iterations: 200" in config
    assert f'source_path: "{L["dataset4d"].resolve()}"' in config
    # (window - 1) / fps = 7/30
    assert "time_duration: [0.0, 0.233333]" in config

    train = next(c for c in calls if Path(c["script"]).name == "train_scratch.py")
    assert train["conda_env"] == "omg4"
    assert str(train["cwd"]) == str(args.trainer_repo)
    assert train["extra_env"] == {"GS4D_T_INIT_DIV": "100"}
    a = train["args"]
    assert a[a.index("--save_iterations") + 1] == "200"
    assert a[a.index("--test_iterations") + 1] == "200"

    bake = next(c for c in calls if Path(c["script"]).name == "bake_sogst.py")
    assert "--mask_filter_root" in bake["args"]
    assert any(Path(c["script"]).name == "eval_render.py" for c in calls)


def test_stage_train4d_t_init_div_zero_leaves_env_unset(monkeypatch, tmp_path):
    monkeypatch.setattr(unified, "gpu_available", lambda: True)
    calls = _capture_runs(monkeypatch)
    args = _train4d_args(tmp_path, iters=7, t_init_div=0, skip_eval=True)
    L = unified.build_layout(tmp_path / "out")
    L["train4d_model"].mkdir(parents=True, exist_ok=True)
    (L["train4d_model"] / "chkpnt7.pth").write_bytes(b"x")
    L["train4d_config"].parent.mkdir(parents=True, exist_ok=True)

    unified.stage_train4d(args, L)
    # Training was skipped (checkpoint pre-existing), so only bake ran; the
    # env-var rule is observable on the generated config path instead: no
    # train call happened, and no call carries GS4D_T_INIT_DIV.
    assert all(not (c["extra_env"] or {}).get("GS4D_T_INIT_DIV") for c in calls)
    assert not any(Path(c["script"]).name == "eval_render.py" for c in calls)


def test_stage_train4d_trainer_config_bypasses_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(unified, "gpu_available", lambda: True)
    calls = _capture_runs(monkeypatch)
    own_config = tmp_path / "mine.yaml"
    own_config.write_text("gaussian_dim: 4\n")
    args = _train4d_args(tmp_path, iters=7, trainer_config=own_config, skip_eval=True)
    L = unified.build_layout(tmp_path / "out")
    L["train4d_model"].mkdir(parents=True, exist_ok=True)
    (L["train4d_model"] / "chkpnt7.pth").write_bytes(b"x")

    unified.stage_train4d(args, L)
    assert not L["train4d_config"].exists()


def test_stage_train4d_missing_trainer_repo_is_a_stage_error(monkeypatch, tmp_path):
    monkeypatch.setattr(unified, "gpu_available", lambda: True)
    args = _train4d_args(tmp_path)
    (args.trainer_repo / "train_scratch.py").unlink()
    with pytest.raises(unified.StageError, match="trainer entry point"):
        unified.stage_train4d(args, unified.build_layout(tmp_path / "out"))
