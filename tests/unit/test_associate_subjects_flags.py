# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""associate_subjects.py: the gates are command-line flags, not constants or environment variables."""

import json
import sys

import pytest

import associate_subjects as asub


def test_every_gate_has_a_flag_with_the_old_default(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["a", "--help"])
    with pytest.raises(SystemExit):
        asub.main()
    text = capsys.readouterr().out
    for flag in ("--score_thr", "--min_common", "--max_aspect", "--epipolar_px", "--dark_ratio", "--size_ratio", "--edge_margin_px"):
        assert flag in text


def test_the_environment_no_longer_changes_the_gate(monkeypatch):
    monkeypatch.setenv("EDGE_PX", "99")
    import importlib

    importlib.reload(asub)
    assert asub.EDGE_PX == 14.0
