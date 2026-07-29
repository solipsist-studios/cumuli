# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

import subprocess

import pytest

import color_correct as cc


# --------------------------------------------------------------------------
# resolve_rawtherapee_cmd
# --------------------------------------------------------------------------

def test_resolve_rawtherapee_cmd_override_is_split_on_whitespace():
    assert cc.resolve_rawtherapee_cmd("custom-rt --flag value") == ["custom-rt", "--flag", "value"]


def test_resolve_rawtherapee_cmd_finds_cli_on_path(monkeypatch):
    monkeypatch.setattr(cc.shutil, "which", lambda name: "/usr/bin/rawtherapee-cli" if name == "rawtherapee-cli" else None)
    assert cc.resolve_rawtherapee_cmd() == ["rawtherapee-cli"]


def test_resolve_rawtherapee_cmd_falls_back_to_flatpak(monkeypatch):
    def fake_which(name):
        return "/usr/bin/flatpak" if name == "flatpak" else None
    monkeypatch.setattr(cc.shutil, "which", fake_which)
    assert cc.resolve_rawtherapee_cmd() == [
        "flatpak", "run", "--command=rawtherapee-cli", "com.rawtherapee.RawTherapee",
    ]


def test_resolve_rawtherapee_cmd_exits_when_neither_found(monkeypatch, capsys):
    monkeypatch.setattr(cc.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as exc_info:
        cc.resolve_rawtherapee_cmd()
    assert exc_info.value.code == 1
    assert "not found" in capsys.readouterr().out


def test_resolve_rawtherapee_cmd_prefers_cli_over_flatpak_when_both_present(monkeypatch):
    monkeypatch.setattr(cc.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert cc.resolve_rawtherapee_cmd() == ["rawtherapee-cli"]


# --------------------------------------------------------------------------
# find_pp3
# --------------------------------------------------------------------------

def test_find_pp3_matches_by_substring_in_filename(tmp_path):
    (tmp_path / "0001.mp4.thumb.jpg.pp3").write_text("")
    result = cc.find_pp3(tmp_path, "0001.mp4")
    assert result == tmp_path / "0001.mp4.thumb.jpg.pp3"


def test_find_pp3_returns_none_when_no_match(tmp_path):
    (tmp_path / "0002.mp4.thumb.jpg.pp3").write_text("")
    assert cc.find_pp3(tmp_path, "0001.mp4") is None


def test_find_pp3_searches_nested_subdirectories(tmp_path):
    nested = tmp_path / "camera_profiles"
    nested.mkdir()
    (nested / "0001.mp4.pp3").write_text("")
    assert cc.find_pp3(tmp_path, "0001.mp4") == nested / "0001.mp4.pp3"


def test_find_pp3_multiple_matches_returns_first_alphabetically(tmp_path):
    (tmp_path / "b_0001.mp4.pp3").write_text("")
    (tmp_path / "a_0001.mp4.pp3").write_text("")
    assert cc.find_pp3(tmp_path, "0001.mp4") == tmp_path / "a_0001.mp4.pp3"


def test_find_pp3_ignores_non_pp3_files(tmp_path):
    (tmp_path / "0001.mp4.jpg").write_text("")
    assert cc.find_pp3(tmp_path, "0001.mp4") is None


# --------------------------------------------------------------------------
# apply_pp3
# --------------------------------------------------------------------------

def test_apply_pp3_builds_expected_command_and_succeeds(tmp_path, monkeypatch):
    image_path = tmp_path / "0001.jpg"
    pp3_path = tmp_path / "0001.pp3"
    out_path = tmp_path / "out.png"
    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        out_path.write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(cc.subprocess, "run", fake_run)

    cc.apply_pp3(["rawtherapee-cli"], image_path, pp3_path, out_path)

    assert calls[0] == [
        "rawtherapee-cli", "-o", str(out_path), "-p", str(pp3_path), "-n", "-Y", "-c", str(image_path),
    ]


def test_apply_pp3_raises_runtime_error_when_output_not_written(tmp_path, monkeypatch):
    image_path = tmp_path / "0001.jpg"
    pp3_path = tmp_path / "0001.pp3"
    out_path = tmp_path / "out.png"  # deliberately never created by the fake

    def fake_run(cmd, capture_output=True, text=True):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="rawtherapee-cli: profile load failed")
    monkeypatch.setattr(cc.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="profile load failed"):
        cc.apply_pp3(["rawtherapee-cli"], image_path, pp3_path, out_path)


def test_apply_pp3_multi_word_rt_cmd_is_prefixed_whole(tmp_path, monkeypatch):
    # rt_cmd from a flatpak resolve is itself a multi-element list -- must be
    # prepended as-is, not re-split or truncated to its first element.
    image_path = tmp_path / "0001.jpg"
    pp3_path = tmp_path / "0001.pp3"
    out_path = tmp_path / "out.png"
    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        out_path.write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(cc.subprocess, "run", fake_run)

    rt_cmd = ["flatpak", "run", "--command=rawtherapee-cli", "com.rawtherapee.RawTherapee"]
    cc.apply_pp3(rt_cmd, image_path, pp3_path, out_path)

    assert calls[0][:4] == rt_cmd
