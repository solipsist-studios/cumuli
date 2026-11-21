# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""fit_rate_target.py: the size search, with the GPU contribution score faked.

The score is the only GPU step, so these tests replace it and check what the
script decides on its own: that it lands between the aim and the target, that
it keeps the highest-scoring splats, and that a target already met prunes
nothing.
"""

import json
import sys

import numpy as np
import pytest

import fit_rate_target
import make_sogst_fixture as fixture
import cumuli_core_path  # noqa: F401
from cumuli_core.ply import read_sogst_ply, write_sogst_ply


@pytest.fixture
def scene(tmp_path, monkeypatch):
    fields, meta = fixture.build_fixture(count=6000, degree=2, include_sh=False, seed=3)
    ply = tmp_path / "in.ply"
    write_sogst_ply(str(ply), fields, meta["time_min"], meta["time_max"], meta["fps"])
    tf = tmp_path / "transforms_train.json"
    tf.write_text(json.dumps({"frames": [{
        "file_path": "a", "time": 0.0, "fl_x": 500.0, "fl_y": 500.0, "cx": 50.0, "cy": 50.0,
        "w": 100, "h": 100, "transform_matrix": np.eye(4).tolist()}]}))
    # fake contribution: splat i scores i, so the kept set must be the top indices
    monkeypatch.setattr(fit_rate_target, "contribution",
                        lambda fields, header, cams: np.arange(len(fields["x"]), dtype=np.float64))
    return ply, tf, tmp_path


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["fit_rate_target.py", *argv])
    fit_rate_target.main()


def test_lands_inside_the_target_and_keeps_the_top(scene, monkeypatch):
    ply, tf, tmp = scene
    full = 6000
    out = tmp / "fit"
    # aim at roughly half the full-size archive
    run(monkeypatch, "--input", str(ply), "--transforms", str(tf), "--target_mb", "0.06",
        "--out", str(out), "--shn_count", "0")
    report = json.loads((tmp / "fit.json").read_text())
    assert report["bytes"] <= report["target_bytes"]
    assert report["bytes"] >= 0.97 * 0.985 * report["target_bytes"] or len(report["packs"]) == 6
    assert 0 < report["count_out"] < full
    _h, kept = read_sogst_ply(str(out) + ".ply")
    _h, src = read_sogst_ply(str(ply))
    k = report["count_out"]
    assert len(kept["x"]) == k
    # the fake score ranks by index, so the kept splats are the last k, in order
    assert np.array_equal(kept["x"], src["x"][full - k:])


def test_target_already_met_prunes_nothing(scene, monkeypatch):
    ply, tf, tmp = scene
    run(monkeypatch, "--input", str(ply), "--transforms", str(tf), "--target_mb", "100",
        "--out", str(tmp / "big"), "--shn_count", "0")
    report = json.loads((tmp / "big.json").read_text())
    assert report["count_out"] == report["count_in"]
    assert len(report["packs"]) == 1
