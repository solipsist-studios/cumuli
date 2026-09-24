# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""The separate eval-camera ring merged into a 4D dataset.

Holding out a rig camera scores whatever view that camera happens to have,
which moves when the rig changes. A fixed eval ring is what makes two
camera configurations comparable, so the properties worth pinning are that
its views never train, that their ground-truth filenames stay unique, and
that they line up with the training timestamps.
"""

import json

import numpy as np
import pytest

import build_flipbook_4dgs_dataset as builder


def write_flipbook(root, labels, n_frames, size=(8, 6)):
    """A minimal flipbook: static rig, one grey image and matte per camera."""
    from PIL import Image

    w, h = size
    frames = []
    for label in labels:
        c2w = np.eye(4)
        c2w[:3, 3] = [float(int(label[-2:])), 0.0, 3.0]
        frames.append({
            "camera_label": label,
            "file_path": f"images_flat/{label}.png",
            "transform_matrix": c2w.tolist(),
            "fl_x": 500.0, "fl_y": 500.0, "cx": w / 2, "cy": h / 2,
            "w": w, "h": h, "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        })
    payload = {"camera_model": "OPENCV", "frames": frames}
    for i in range(n_frames):
        frame_dir = root / f"frame_{i:04d}"
        (frame_dir / "images_flat").mkdir(parents=True, exist_ok=True)
        (frame_dir / "fmasks_clean").mkdir(parents=True, exist_ok=True)
        (frame_dir / "transforms.json").write_text(json.dumps(payload))
        for label in labels:
            Image.fromarray(np.full((h, w, 3), 100 + i, dtype=np.uint8)).save(
                frame_dir / "images_flat" / f"{label}.png")
            Image.fromarray(np.full((h, w), 255, dtype=np.uint8)).save(
                frame_dir / "fmasks_clean" / f"{label}.png")
    return root


# --------------------------------------------------------------- basenames
def test_eval_basenames_are_unique_across_cameras_and_frames():
    """eval_render.py finds ground truth by basename alone, so two cameras
    sharing a frame number would silently score one against the other."""
    names = {builder.eval_basename(label, i)
             for label in ("e00", "e01", "e02") for i in range(3)}
    assert len(names) == 9


def test_eval_basename_carries_the_one_based_frame_number():
    assert builder.eval_basename("e00", 0) == "came00_frame_00001"


# --------------------------------------------------------------- eval root
def test_no_eval_root_gives_an_empty_rig():
    assert builder.load_eval_root(None, 3, [0, 1, 2]) == ({}, [])


def test_eval_root_loads_the_frames_the_training_side_kept(tmp_path):
    write_flipbook(tmp_path / "eval", ["e00", "e01"], 4)
    rig, dirs = builder.load_eval_root(tmp_path / "eval", 4, [0, 2])
    assert sorted(rig) == ["e00", "e01"]
    assert [d.name for d in dirs] == ["frame_0000", "frame_0002"]


def test_eval_root_frame_count_must_match_the_source(tmp_path):
    """The two flipbooks come from one render of one clip; a mismatch means
    something was rendered or deleted separately."""
    write_flipbook(tmp_path / "eval", ["e00"], 2)
    with pytest.raises(SystemExit, match="frame directories"):
        builder.load_eval_root(tmp_path / "eval", 5, [0, 1])


def test_eval_root_without_frames_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no frame_"):
        builder.load_eval_root(tmp_path / "empty", 1, [0])


# ------------------------------------------------------------ end to end
def run_builder(monkeypatch, argv):
    import sys

    monkeypatch.setattr(sys, "argv", argv)
    builder.main()


def test_eval_cameras_are_scored_but_never_trained(tmp_path, monkeypatch):
    train = tmp_path / "flipbook_src"
    evals = tmp_path / "eval_src"
    out = tmp_path / "dataset"
    write_flipbook(train, ["00", "01", "02"], 2)
    write_flipbook(evals, ["e00", "e01"], 2)

    run_builder(monkeypatch, [
        "build_flipbook_4dgs_dataset.py",
        "--flipbook_root", str(train), "--eval_root", str(evals),
        "--out", str(out), "--fps", "24", "--downscale", "1",
        "--hull_min_views", "1", "--jobs", "2",
        "--init_bbox=-1,-1,-1,1,1,1",
    ])

    train_json = json.loads((out / "transforms_train.json").read_text())
    test_json = json.loads((out / "transforms_test.json").read_text())
    train_labels = {f["camera_label"] for f in train_json["frames"]}
    test_labels = {f["camera_label"] for f in test_json["frames"]}

    assert train_labels == {"00", "01", "02"}       # every rig camera trains
    assert test_labels == {"e00", "e01"}            # only the eval ring scores
    assert not (train_labels & test_labels)

    # Ground truth exists for each scored view, under its unique basename.
    gt = sorted(p.name for p in (out / "eval_gt_flat").glob("*.png"))
    assert gt == ["came00_frame_00001.png", "came00_frame_00002.png",
                  "came01_frame_00001.png", "came01_frame_00002.png"]
    for entry in test_json["frames"]:
        assert (out / (entry["file_path"] + ".png")).is_file()
        basename = entry["file_path"].rsplit("/", 1)[-1]
        assert (out / "eval_gt_flat" / (basename + ".png")).is_file()


def test_without_eval_root_the_real_capture_path_is_unchanged(tmp_path, monkeypatch):
    """The flag is additive: with it absent, a real capture builds exactly
    as before, test views falling back to a training camera."""
    train = tmp_path / "flipbook_src"
    out = tmp_path / "dataset"
    write_flipbook(train, ["00", "01"], 2)

    run_builder(monkeypatch, [
        "build_flipbook_4dgs_dataset.py",
        "--flipbook_root", str(train), "--out", str(out),
        "--fps", "24", "--downscale", "1", "--hull_min_views", "1",
        "--jobs", "2", "--init_bbox=-1,-1,-1,1,1,1",
    ])

    assert not (out / "evalcams").exists()
    assert not (out / "eval_gt_flat").exists()
    test_json = json.loads((out / "transforms_test.json").read_text())
    assert {f["camera_label"] for f in test_json["frames"]} == {"00"}


def test_explicit_bbox_skips_the_search(tmp_path, monkeypatch, capsys):
    """A subject small next to the camera spread makes rejection sampling
    find almost nothing; the synthetic path knows the extent already."""
    train = tmp_path / "flipbook_src"
    write_flipbook(train, ["00", "01"], 1)
    run_builder(monkeypatch, [
        "build_flipbook_4dgs_dataset.py",
        "--flipbook_root", str(train), "--out", str(tmp_path / "d"),
        "--fps", "24", "--downscale", "1", "--hull_min_views", "1",
        "--jobs", "1", "--init_bbox", "0,0,0,1,1,1",
    ])
    assert "hull bbox (given)" in capsys.readouterr().out


def test_a_malformed_bbox_is_rejected(tmp_path, monkeypatch):
    train = tmp_path / "flipbook_src"
    write_flipbook(train, ["00"], 1)
    with pytest.raises(SystemExit, match="6 numbers"):
        run_builder(monkeypatch, [
            "build_flipbook_4dgs_dataset.py",
            "--flipbook_root", str(train), "--out", str(tmp_path / "d"),
            "--init_bbox", "0,0,0,1",
        ])
