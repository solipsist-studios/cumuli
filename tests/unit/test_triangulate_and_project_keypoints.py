# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

import subprocess
import sys
from pathlib import Path

import pytest

import triangulate_and_project_keypoints as tpk


def base_argv(camera_path, kp2d_dir, out_kp3d_dir, extra=None):
    argv = [
        "prog",
        "--camera_path", str(camera_path),
        "--kp2d_dir", str(kp2d_dir),
        "--out_kp3d_dir", str(out_kp3d_dir),
    ]
    return argv + (extra or [])


def test_main_errors_when_script_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tpk, "SCRIPT", tmp_path / "does_not_exist.py")
    monkeypatch.setattr(sys, "argv", base_argv(tmp_path / "t.json", tmp_path / "kp2d", tmp_path / "kp3d"))
    with pytest.raises(SystemExit) as exc_info:
        tpk.main()
    assert exc_info.value.code == 1
    assert "Diffuman4D submodule checked out" in capsys.readouterr().out


@pytest.mark.parametrize("returncode", [0, 1, 137])
def test_main_propagates_subprocess_returncode_exactly(tmp_path, monkeypatch, returncode):
    monkeypatch.setattr(tpk, "SCRIPT", tmp_path / "triangulate_skeleton.py")
    tpk.SCRIPT.write_text("")

    def fake_run(cmd, cwd=None):
        return subprocess.CompletedProcess(cmd, returncode)
    monkeypatch.setattr(tpk.subprocess, "run", fake_run)

    monkeypatch.setattr(sys, "argv", base_argv(tmp_path / "t.json", tmp_path / "kp2d", tmp_path / "kp3d"))
    with pytest.raises(SystemExit) as exc_info:
        tpk.main()
    assert exc_info.value.code == returncode


def test_main_cmd_construction_without_out_pcd_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tpk, "SCRIPT", tmp_path / "triangulate_skeleton.py")
    tpk.SCRIPT.write_text("")

    calls = []
    captured_tmp_dirs = []
    def fake_run(cmd, cwd=None):
        calls.append((cmd, cwd))
        proj_dir = cmd[cmd.index("--out_kp2d_proj_dir") + 1]
        captured_tmp_dirs.append(proj_dir)
        assert Path(proj_dir).is_dir()  # tempdir must exist DURING the subprocess call
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(tpk.subprocess, "run", fake_run)

    camera_path, kp2d_dir, out_kp3d_dir = tmp_path / "t.json", tmp_path / "kp2d", tmp_path / "kp3d"
    monkeypatch.setattr(sys, "argv", base_argv(camera_path, kp2d_dir, out_kp3d_dir))
    with pytest.raises(SystemExit) as exc_info:
        tpk.main()
    assert exc_info.value.code == 0

    assert len(calls) == 1
    cmd, cwd = calls[0]
    assert cmd[0] == sys.executable
    assert cmd[1] == str(tpk.SCRIPT)
    assert cmd[cmd.index("--camera_path") + 1] == str(camera_path)
    assert cmd[cmd.index("--kp2d_dir") + 1] == str(kp2d_dir)
    assert cmd[cmd.index("--out_kp3d_dir") + 1] == str(out_kp3d_dir)
    assert "--spa_labels_proj_range=[0,1,1]" in cmd
    assert "--out_pcd_dir" not in cmd
    assert cwd == str(tpk.DIFFUMAN4D_ROOT)

    # the TemporaryDirectory must be cleaned up once main() has returned
    assert not Path(captured_tmp_dirs[0]).exists()


def test_main_cmd_includes_out_pcd_dir_when_given(tmp_path, monkeypatch):
    monkeypatch.setattr(tpk, "SCRIPT", tmp_path / "triangulate_skeleton.py")
    tpk.SCRIPT.write_text("")

    calls = []
    def fake_run(cmd, cwd=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(tpk.subprocess, "run", fake_run)

    out_pcd_dir = tmp_path / "pcd"
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "t.json", tmp_path / "kp2d", tmp_path / "kp3d", ["--out_pcd_dir", str(out_pcd_dir)],
    ))
    with pytest.raises(SystemExit):
        tpk.main()

    cmd = calls[0]
    assert cmd[cmd.index("--out_pcd_dir") + 1] == str(out_pcd_dir)


@pytest.mark.parametrize("missing_flag", ["--camera_path", "--kp2d_dir", "--out_kp3d_dir"])
def test_main_errors_when_required_arg_missing(tmp_path, monkeypatch, missing_flag):
    argv = base_argv(tmp_path / "t.json", tmp_path / "kp2d", tmp_path / "kp3d")
    idx = argv.index(missing_flag)
    argv = argv[:idx] + argv[idx + 2:]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        tpk.main()
    assert exc_info.value.code == 2
