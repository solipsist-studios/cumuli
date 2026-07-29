import subprocess
import sys

import pytest

import generate_masks as gm


def base_argv(images_dir, out_fmasks_dir, extra=None):
    argv = ["prog", "--images_dir", str(images_dir), "--out_fmasks_dir", str(out_fmasks_dir)]
    return argv + (extra or [])


def test_main_errors_when_script_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gm, "SCRIPT", tmp_path / "does_not_exist.py")
    monkeypatch.setattr(sys, "argv", base_argv(tmp_path / "images", tmp_path / "out"))
    with pytest.raises(SystemExit) as exc_info:
        gm.main()
    assert exc_info.value.code == 1
    assert "Diffuman4D submodule checked out" in capsys.readouterr().out


@pytest.mark.parametrize("returncode", [0, 1, 137])
def test_main_propagates_subprocess_returncode_exactly(tmp_path, monkeypatch, returncode):
    monkeypatch.setattr(gm, "SCRIPT", tmp_path / "remove_background.py")
    gm.SCRIPT.write_text("")

    def fake_run(cmd, cwd=None):
        return subprocess.CompletedProcess(cmd, returncode)
    monkeypatch.setattr(gm.subprocess, "run", fake_run)

    monkeypatch.setattr(sys, "argv", base_argv(tmp_path / "images", tmp_path / "out"))
    with pytest.raises(SystemExit) as exc_info:
        gm.main()
    assert exc_info.value.code == returncode


def test_main_creates_out_fmasks_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "SCRIPT", tmp_path / "remove_background.py")
    gm.SCRIPT.write_text("")

    def fake_run(cmd, cwd=None):
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(gm.subprocess, "run", fake_run)

    out_fmasks_dir = tmp_path / "nested" / "out"
    monkeypatch.setattr(sys, "argv", base_argv(tmp_path / "images", out_fmasks_dir))
    with pytest.raises(SystemExit) as exc_info:
        gm.main()
    assert exc_info.value.code == 0
    assert out_fmasks_dir.is_dir()


def test_main_cmd_and_cwd_construction_default_image_ext(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "SCRIPT", tmp_path / "remove_background.py")
    gm.SCRIPT.write_text("")

    calls = []
    def fake_run(cmd, cwd=None):
        calls.append((cmd, cwd))
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(gm.subprocess, "run", fake_run)

    images_dir = tmp_path / "images"
    out_fmasks_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(images_dir, out_fmasks_dir))
    with pytest.raises(SystemExit):
        gm.main()

    assert len(calls) == 1
    cmd, cwd = calls[0]
    assert cmd == [sys.executable, str(gm.SCRIPT), str(images_dir), str(out_fmasks_dir), "--image_ext", ".png"]
    assert cwd == str(gm.DIFFUMAN4D_ROOT)


def test_main_explicit_image_ext_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "SCRIPT", tmp_path / "remove_background.py")
    gm.SCRIPT.write_text("")

    calls = []
    def fake_run(cmd, cwd=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(gm.subprocess, "run", fake_run)

    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "images", tmp_path / "out", ["--image_ext", ".jpg"],
    ))
    with pytest.raises(SystemExit):
        gm.main()
    assert calls[0][calls[0].index("--image_ext") + 1] == ".jpg"


def test_main_rejects_unsupported_image_ext(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "images", tmp_path / "out", ["--image_ext", ".bmp"],
    ))
    with pytest.raises(SystemExit) as exc_info:
        gm.main()
    assert exc_info.value.code == 2  # argparse's own usage-error exit code


@pytest.mark.parametrize("missing_flag", ["--images_dir", "--out_fmasks_dir"])
def test_main_errors_when_required_arg_missing(tmp_path, monkeypatch, missing_flag):
    argv = base_argv(tmp_path / "images", tmp_path / "out")
    idx = argv.index(missing_flag)
    argv = argv[:idx] + argv[idx + 2:]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        gm.main()
    assert exc_info.value.code == 2
