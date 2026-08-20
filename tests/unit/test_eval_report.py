# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Schema tests for eval_render.build_report, the --report_json payload.

build_report is deliberately torch-free so the report schema the GPU
integration workflow consumes (eval_4d.json) is pinned by a test that runs
in the CPU-only unit suite.
"""

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from eval_render import build_report

CONFIG = {"model": "a.sogst", "transforms": "t.json", "gt_dir": "gt",
          "every": 10, "downscale": 2.0}


def view(name, psnr, ssim, lp, t=0.0):
    return {"name": name, "time": t, "psnr_db": psnr, "ssim": ssim, "lpips": lp}


def test_report_means_and_shape():
    views = [view("f1", 30.0, 0.9, 0.10), view("f2", 32.0, 0.8, 0.20)]
    report = build_report(views, CONFIG)

    assert set(report) == {"mean", "views", "config"}
    assert report["mean"] == {"psnr_db": 31.0, "ssim": pytest.approx(0.85),
                              "lpips": pytest.approx(0.15)}
    assert report["views"] == views
    assert report["config"] == CONFIG


def test_report_empty_views_yields_null_means():
    report = build_report([], CONFIG)

    assert report["mean"] == {"psnr_db": None, "ssim": None, "lpips": None}
    assert report["views"] == []


def test_report_is_json_serializable():
    report = build_report([view("f1", 30.0, 0.9, 0.10)], CONFIG)

    round_tripped = json.loads(json.dumps(report))
    assert round_tripped == report
