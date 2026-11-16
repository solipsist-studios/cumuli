# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""stitch_window_plys.py: raw window PLYs in, one clip PLY out."""

import json
import sys

import numpy as np

import stitch_window_plys as swp
from cumuli_core.ply import read_sogst_ply, write_sogst_ply
from test_merge_sogst_segments import make_fields


def write_window(path, n, t_center, duration):
    write_sogst_ply(str(path), make_fields(n, t_center, t_sigma=0.05), 0.0, duration, 24.0)


def test_two_windows_become_one_clip_over_global_time(tmp_path, monkeypatch):
    a, b, out, report = tmp_path / "a.ply", tmp_path / "b.ply", tmp_path / "out.ply", tmp_path / "r.json"
    write_window(a, 200, 0.5, 1.0)
    write_window(b, 200, 0.5, 1.0)
    monkeypatch.setattr(sys, "argv", ["stitch", "--window", str(b), "2.0", "--window", str(a), "0.0",
                                      "--out", str(out), "--report_json", str(report)])
    swp.main()
    header, fields = read_sogst_ply(str(out))
    assert float(header["time_min"]) == 0.0 and float(header["time_max"]) == 3.0
    centers = np.asarray(fields["t_center"])
    assert centers.min() >= 0.0 and centers.max() <= 3.0
    # Each window's splats sit in its own slot: the second window's were shifted by 2 s.
    assert (centers > 1.5).any() and (centers < 1.5).any()
    data = json.loads(report.read_text())
    assert [w["start_s"] for w in data["windows"]] == [0.0, 2.0]  # sorted by start
    assert data["splats"] == len(centers)


def test_a_missing_window_is_a_clean_error(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(sys, "argv", ["stitch", "--window", str(tmp_path / "none.ply"), "0", "--out", str(tmp_path / "o.ply")])
    with pytest.raises(SystemExit, match="not found"):
        swp.main()
