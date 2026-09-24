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


# --------------------------------------------------------------------------
# Camera loading. Two silent mis-scoring bugs lived here: image size was read
# only from the top level of the transforms file, so a dataset with
# per-camera intrinsics could not trip the size check, and the time-unit
# heuristic invented a scale when every sampled view shared one timestamp.
# Both were found by an end-to-end run whose eval scored 22 dB below the
# trainer's own number on the same cameras.
# --------------------------------------------------------------------------
from eval_render import load_cameras  # noqa: E402


def per_frame_transforms(tmp_path, n=2, w=960, h=840, fl=853.33, times=(0.0, 0.5)):
    """A transforms file shaped like build_flipbook_4dgs_dataset.py's: every
    camera carries its own intrinsics, and there is no top-level block."""
    frames = []
    for i in range(n):
        frames.append({
            "file_path": f"evalcams/came{i:02d}/came{i:02d}_frame_00001",
            "camera_label": f"e{i:02d}",
            "time": times[i % len(times)],
            "fl_x": fl, "fl_y": fl, "cx": w / 2, "cy": h / 2, "w": w, "h": h,
            "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3],
                                 [0, 0, 0, 1]],
        })
    path = tmp_path / "transforms_test.json"
    path.write_text(json.dumps({"camera_model": "OPENCV", "frames": frames}))
    return path


def test_image_size_is_read_per_frame_when_there_is_no_global_block(tmp_path):
    cams = load_cameras(per_frame_transforms(tmp_path), downscale=1, every=1)
    assert [c["w"] for c in cams] == [960, 960]
    assert [c["h"] for c in cams] == [840, 840]


def test_downscale_applies_to_the_per_frame_size_too(tmp_path):
    """Without this the size check cannot fire, and a wrong --downscale
    renders at the wrong scale against correctly sized ground truth."""
    cams = load_cameras(per_frame_transforms(tmp_path), downscale=2, every=1)
    assert cams[0]["w"] == 480 and cams[0]["h"] == 420
    assert cams[0]["K"][0, 0] == pytest.approx(853.33 / 2)


def test_a_global_intrinsics_block_still_works(tmp_path):
    """The n3v-style layout: one intrinsics block for every camera."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "w": 1024, "h": 768, "fl_x": 800.0, "fl_y": 800.0,
        "cx": 512.0, "cy": 384.0,
        "frames": [{"file_path": "a", "time": 0.0,
                    "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0],
                                         [0, 0, 1, 3], [0, 0, 0, 1]]}]}))
    cams = load_cameras(path, downscale=2, every=1)
    assert cams[0]["w"] == 512 and cams[0]["h"] == 384
    assert cams[0]["K"][0, 0] == pytest.approx(400.0)


def test_every_selects_a_subset(tmp_path):
    cams = load_cameras(per_frame_transforms(tmp_path, n=6), downscale=1, every=3)
    assert len(cams) == 2


def test_camera_names_come_from_the_file_path_basename(tmp_path):
    """Ground truth is found by this name, so it has to be unique per view."""
    cams = load_cameras(per_frame_transforms(tmp_path), downscale=1, every=1)
    assert cams[0]["name"] == "came00_frame_00001"
    assert cams[1]["name"] == "came01_frame_00001"
