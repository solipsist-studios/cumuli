# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

import subprocess
import sys

import pytest

import run_pose_refinement as rpr


# --------------------------------------------------------------------------
# merge_instants -- pure file-reorganization logic, no mocking
# --------------------------------------------------------------------------

def write_kp_json(kp2d_dir, cam, name="000000.json", content="{}"):
    d = kp2d_dir / cam
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(content)
    return d


def test_merge_instants_basic_multi_instant_multi_camera(tmp_path):
    f0 = tmp_path / "f0"
    f1 = tmp_path / "f1"
    write_kp_json(f0, "00", content="instant0-cam00")
    write_kp_json(f0, "01", content="instant0-cam01")
    write_kp_json(f1, "00", content="instant1-cam00")
    write_kp_json(f1, "01", content="instant1-cam01")

    merged = tmp_path / "merged"
    labels = rpr.merge_instants([f0, f1], merged)

    assert labels == {"00", "01"}
    assert (merged / "00" / "000000.json").read_text() == "instant0-cam00"
    assert (merged / "00" / "000001.json").read_text() == "instant1-cam00"
    assert (merged / "01" / "000000.json").read_text() == "instant0-cam01"
    assert (merged / "01" / "000001.json").read_text() == "instant1-cam01"


def test_merge_instants_recreates_existing_merged_dir(tmp_path):
    # A stale merged_dir from a prior run must be wiped, not accumulated into.
    merged = tmp_path / "merged"
    (merged / "99").mkdir(parents=True)
    (merged / "99" / "000000.json").write_text("stale leftover")

    f0 = tmp_path / "f0"
    write_kp_json(f0, "00")
    rpr.merge_instants([f0], merged)

    assert not (merged / "99").exists()
    assert (merged / "00" / "000000.json").is_file()


def test_merge_instants_warns_when_no_camera_subdirs_in_instant(tmp_path, capsys):
    f0 = tmp_path / "f0"
    f0.mkdir(parents=True)  # exists but has no per-camera subdirs at all
    merged = tmp_path / "merged"

    labels = rpr.merge_instants([f0], merged)

    assert labels == set()
    assert f"WARNING: no per-camera subdirs found in {f0}" in capsys.readouterr().out


def test_merge_instants_warns_when_camera_subdir_has_no_json(tmp_path, capsys):
    # Regression test: a per-camera subdir that exists but has zero *.json
    # files (e.g. a partial prior run) used to be dropped with NO warning at
    # all, unlike the sibling "no per-camera subdirs found" case above.
    f0 = tmp_path / "f0"
    (f0 / "00").mkdir(parents=True)  # exists, but empty -- no json inside
    write_kp_json(f0, "01")
    merged = tmp_path / "merged"

    labels = rpr.merge_instants([f0], merged)

    assert labels == {"01"}
    out = capsys.readouterr().out
    assert f"no keypoints json found in {f0 / '00'}" in out
    assert not (merged / "00").exists()
    assert (merged / "01" / "000000.json").is_file()


def test_merge_instants_multiple_jsons_in_cam_dir_takes_first_sorted(tmp_path):
    f0 = tmp_path / "f0"
    cam_dir = f0 / "00"
    cam_dir.mkdir(parents=True)
    (cam_dir / "b.json").write_text("second")
    (cam_dir / "a.json").write_text("first")
    merged = tmp_path / "merged"

    rpr.merge_instants([f0], merged)

    assert (merged / "00" / "000000.json").read_text() == "first"


def test_merge_instants_ignores_non_directory_entries_in_kp2d_dir(tmp_path):
    f0 = tmp_path / "f0"
    f0.mkdir(parents=True)
    (f0 / "stray_file.txt").write_text("not a camera dir")
    write_kp_json(f0, "00")
    merged = tmp_path / "merged"

    labels = rpr.merge_instants([f0], merged)
    assert labels == {"00"}


def test_merge_instants_camera_present_in_some_instants_not_others(tmp_path):
    f0 = tmp_path / "f0"
    f1 = tmp_path / "f1"
    write_kp_json(f0, "00")
    write_kp_json(f0, "01")
    write_kp_json(f1, "00")  # camera 01 missing from this instant entirely
    merged = tmp_path / "merged"

    labels = rpr.merge_instants([f0, f1], merged)

    assert labels == {"00", "01"}
    assert (merged / "00" / "000000.json").is_file()
    assert (merged / "00" / "000001.json").is_file()
    assert (merged / "01" / "000000.json").is_file()
    assert not (merged / "01" / "000001.json").exists()


# --------------------------------------------------------------------------
# main() -- CLI wiring. subprocess.run is monkeypatched (it shells out to
# refine_poses_with_keypoints.py, tested separately); merge_instants() and
# everything around the subprocess boundary is real.
# --------------------------------------------------------------------------

def base_argv(transforms, kp2d_dirs, out_transforms, extra=None):
    argv = [
        "prog",
        "--transforms", str(transforms),
        "--kp2d_dirs", ",".join(str(d) for d in kp2d_dirs),
        "--out_transforms", str(out_transforms),
    ]
    return argv + (extra or [])


def make_instant_dirs(tmp_path, n, cams=("00", "01")):
    dirs = []
    for k in range(n):
        d = tmp_path / f"f{k}"
        for cam in cams:
            write_kp_json(d, cam)
        dirs.append(d)
    return dirs


def test_main_errors_when_refine_script_missing(tmp_path, monkeypatch, capsys):
    kp2d_dirs = make_instant_dirs(tmp_path, 1)
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "t.json", kp2d_dirs, tmp_path / "out" / "t.json",
        ["--refine_script", str(tmp_path / "does_not_exist.py")],
    ))
    with pytest.raises(SystemExit) as exc_info:
        rpr.main()
    assert exc_info.value.code == 1
    assert "not found" in capsys.readouterr().out


def test_main_errors_when_kp2d_dir_missing(tmp_path, monkeypatch, capsys):
    refine_script = tmp_path / "refine.py"
    refine_script.write_text("")
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "t.json", [tmp_path / "nope"], tmp_path / "out" / "t.json",
        ["--refine_script", str(refine_script)],
    ))
    with pytest.raises(SystemExit) as exc_info:
        rpr.main()
    assert exc_info.value.code == 1
    assert "is not a directory" in capsys.readouterr().out


def test_main_errors_when_kp2d_dirs_contains_empty_segment(tmp_path, monkeypatch, capsys):
    # Regression test: a trailing/double comma in --kp2d_dirs used to
    # silently resolve Path("") to the cwd, which .is_dir() reports as True
    # -- passing validation and treating an unrelated directory as real
    # keypoints input instead of erroring.
    refine_script = tmp_path / "refine.py"
    refine_script.write_text("")
    kp2d_dirs = make_instant_dirs(tmp_path, 1)
    argv = base_argv(tmp_path / "t.json", kp2d_dirs, tmp_path / "out" / "t.json",
                      ["--refine_script", str(refine_script)])
    argv[argv.index("--kp2d_dirs") + 1] += ","  # inject a trailing comma
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        rpr.main()
    assert exc_info.value.code == 1
    assert "contains an empty path" in capsys.readouterr().out


def test_main_errors_when_merged_dir_is_a_file(tmp_path, monkeypatch, capsys):
    # Regression test: --merged_dir already existing as a file used to
    # crash merge_instants() with a raw NotADirectoryError from shutil.rmtree.
    refine_script = tmp_path / "refine.py"
    refine_script.write_text("")
    kp2d_dirs = make_instant_dirs(tmp_path, 1)
    merged_as_file = tmp_path / "merged_as_file"
    merged_as_file.write_text("oops")
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "t.json", kp2d_dirs, tmp_path / "out" / "t.json",
        ["--refine_script", str(refine_script), "--merged_dir", str(merged_as_file)],
    ))
    with pytest.raises(SystemExit) as exc_info:
        rpr.main()
    assert exc_info.value.code == 1
    assert "exists and is not a directory" in capsys.readouterr().out


def patch_subprocess(monkeypatch, calls, returncode=0):
    def fake_run(cmd):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode)
    monkeypatch.setattr(rpr.subprocess, "run", fake_run)


def run_main_ok(monkeypatch, tmp_path, n_instants=5, extra=None, cams=("00", "01")):
    refine_script = tmp_path / "refine.py"
    refine_script.write_text("")
    kp2d_dirs = make_instant_dirs(tmp_path, n_instants, cams=cams)
    out_transforms = tmp_path / "out" / "transforms_refined.json"
    argv = base_argv(tmp_path / "t.json", kp2d_dirs, out_transforms,
                      ["--refine_script", str(refine_script)] + (extra or []))
    monkeypatch.setattr(sys, "argv", argv)
    return out_transforms


def test_main_warns_when_fewer_than_5_instants(tmp_path, monkeypatch, capsys):
    calls = []
    patch_subprocess(monkeypatch, calls)
    run_main_ok(monkeypatch, tmp_path, n_instants=2)
    with pytest.raises(SystemExit):
        rpr.main()
    assert "only 2 time instant(s)" in capsys.readouterr().out


def test_main_no_warning_when_5_or_more_instants(tmp_path, monkeypatch, capsys):
    calls = []
    patch_subprocess(monkeypatch, calls)
    run_main_ok(monkeypatch, tmp_path, n_instants=5)
    with pytest.raises(SystemExit):
        rpr.main()
    assert "time instant(s) given" not in capsys.readouterr().out


def test_main_merged_dir_defaults_alongside_out_transforms(tmp_path, monkeypatch):
    calls = []
    patch_subprocess(monkeypatch, calls)
    out_transforms = run_main_ok(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        rpr.main()
    expected_merged = out_transforms.parent / "poses_2d_merged"
    assert expected_merged.is_dir()
    cmd = calls[0]
    assert cmd[cmd.index("--kp2d") + 1] == str(expected_merged)


def test_main_explicit_merged_dir_override(tmp_path, monkeypatch):
    calls = []
    patch_subprocess(monkeypatch, calls)
    custom_merged = tmp_path / "custom_merged"
    run_main_ok(monkeypatch, tmp_path, extra=["--merged_dir", str(custom_merged)])
    with pytest.raises(SystemExit):
        rpr.main()
    assert custom_merged.is_dir()
    cmd = calls[0]
    assert cmd[cmd.index("--kp2d") + 1] == str(custom_merged)


def test_main_cmd_construction_all_flags(tmp_path, monkeypatch):
    calls = []
    patch_subprocess(monkeypatch, calls)
    transforms = tmp_path / "t.json"
    out_transforms = run_main_ok(monkeypatch, tmp_path, extra=[
        "--score_thr", "0.7", "--min_views", "4", "--huber_px", "10.0",
        "--outlier_px", "200.0", "--max_iters", "500",
    ])
    with pytest.raises(SystemExit):
        rpr.main()

    cmd = calls[0]
    assert cmd[0] == sys.executable
    assert cmd[cmd.index("--transforms") + 1] == str(transforms)
    assert cmd[cmd.index("--out_transforms") + 1] == str(out_transforms)
    assert cmd[cmd.index("--score_thr") + 1] == "0.7"
    assert cmd[cmd.index("--min_views") + 1] == "4"
    assert cmd[cmd.index("--huber_px") + 1] == "10.0"
    assert cmd[cmd.index("--outlier_px") + 1] == "200.0"
    assert cmd[cmd.index("--max_iters") + 1] == "500"
    assert "--report_only" not in cmd


def test_main_report_only_flag_appended_when_given(tmp_path, monkeypatch):
    calls = []
    patch_subprocess(monkeypatch, calls)
    run_main_ok(monkeypatch, tmp_path, extra=["--report_only"])
    with pytest.raises(SystemExit):
        rpr.main()
    assert "--report_only" in calls[0]


@pytest.mark.parametrize("returncode", [0, 1, 137])
def test_main_propagates_subprocess_returncode_exactly(tmp_path, monkeypatch, returncode):
    calls = []
    patch_subprocess(monkeypatch, calls, returncode=returncode)
    run_main_ok(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        rpr.main()
    assert exc_info.value.code == returncode


@pytest.mark.parametrize("missing_flag", ["--transforms", "--kp2d_dirs", "--out_transforms"])
def test_main_errors_when_required_arg_missing(tmp_path, monkeypatch, missing_flag):
    kp2d_dirs = make_instant_dirs(tmp_path, 1)
    argv = base_argv(tmp_path / "t.json", kp2d_dirs, tmp_path / "out" / "t.json")
    idx = argv.index(missing_flag)
    argv = argv[:idx] + argv[idx + 2:]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        rpr.main()
    assert exc_info.value.code == 2
