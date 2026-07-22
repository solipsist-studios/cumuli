import sys

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

    for name in ("stage_sync", "stage_production", "stage_poses", "stage_masks", "stage_branch_direct"):
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

    assert validate_calls == ["sync", "production", "poses", "masks", "branch"]


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

    assert stage_calls == ["stage_poses", "stage_masks", "stage_branch_direct"]
    assert validate_calls == ["poses", "masks", "branch"]


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
