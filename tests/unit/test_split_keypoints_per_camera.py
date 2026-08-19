# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

import json
import sys

import pytest

import split_keypoints_per_camera as skpc


def write_kp2d(kp2d_flat_dir, images_dir_name, camera_label, keypoints=None, scores=None, extra=None):
    keypoints = keypoints if keypoints is not None else [[1.0, 2.0], [3.0, 4.0]]
    scores = scores if scores is not None else [0.9, 0.8]
    data = {"instance_info": [{"keypoints": keypoints, "keypoint_scores": scores, **(extra or {})}]}
    cam_dir = kp2d_flat_dir / images_dir_name
    cam_dir.mkdir(parents=True, exist_ok=True)
    path = cam_dir / f"{camera_label}.json"
    path.write_text(json.dumps(data))
    return path


def test_main_errors_when_no_json_files_found(tmp_path, monkeypatch):
    kp2d_flat_dir = tmp_path / "flat"
    kp2d_flat_dir.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "prog", "--kp2d_flat_dir", str(kp2d_flat_dir), "--out_dir", str(tmp_path / "out"),
    ])
    with pytest.raises(FileNotFoundError, match="No per-camera keypoint JSONs"):
        skpc.main()


def test_main_writes_expected_layout_with_default_tem_label(tmp_path, monkeypatch):
    kp2d_flat_dir = tmp_path / "flat"
    write_kp2d(kp2d_flat_dir, "images_flat", "00")
    write_kp2d(kp2d_flat_dir, "images_flat", "01")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--kp2d_flat_dir", str(kp2d_flat_dir), "--out_dir", str(out_dir),
    ])
    skpc.main()

    assert (out_dir / "00" / "000000.json").is_file()
    assert (out_dir / "01" / "000000.json").is_file()


def test_main_explicit_tem_label(tmp_path, monkeypatch):
    kp2d_flat_dir = tmp_path / "flat"
    write_kp2d(kp2d_flat_dir, "images_flat", "00")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--kp2d_flat_dir", str(kp2d_flat_dir), "--out_dir", str(out_dir),
        "--tem_label", "000005",
    ])
    skpc.main()
    assert (out_dir / "00" / "000005.json").is_file()
    assert not (out_dir / "00" / "000000.json").exists()


def test_main_adds_keypoint_depths_when_missing(tmp_path, monkeypatch):
    kp2d_flat_dir = tmp_path / "flat"
    write_kp2d(kp2d_flat_dir, "images_flat", "00", keypoints=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--kp2d_flat_dir", str(kp2d_flat_dir), "--out_dir", str(out_dir),
    ])
    skpc.main()

    with open(out_dir / "00" / "000000.json") as f:
        written = json.load(f)
    depths = written["instance_info"][0]["keypoint_depths"]
    assert depths == [0.0, 0.0, 0.0]  # one per keypoint, defaulted


def test_main_preserves_existing_keypoint_depths(tmp_path, monkeypatch):
    kp2d_flat_dir = tmp_path / "flat"
    write_kp2d(kp2d_flat_dir, "images_flat", "00", extra={"keypoint_depths": [1.5, 2.5]})
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--kp2d_flat_dir", str(kp2d_flat_dir), "--out_dir", str(out_dir),
    ])
    skpc.main()

    with open(out_dir / "00" / "000000.json") as f:
        written = json.load(f)
    assert written["instance_info"][0]["keypoint_depths"] == [1.5, 2.5]  # not overwritten


def test_main_skips_camera_with_empty_instance_info(tmp_path, monkeypatch, capsys):
    # Regression test: an empty instance_info list (no detected instance
    # for that camera/instant -- plausible from the coco_wholebody133 path,
    # which writes files directly without predict_keypoints_2d.py's own
    # skip-if-empty guard) crashed the whole batch with a raw IndexError.
    kp2d_flat_dir = tmp_path / "flat"
    (kp2d_flat_dir / "images_flat").mkdir(parents=True)
    (kp2d_flat_dir / "images_flat" / "00.json").write_text(json.dumps({"instance_info": []}))
    write_kp2d(kp2d_flat_dir, "images_flat", "01")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--kp2d_flat_dir", str(kp2d_flat_dir), "--out_dir", str(out_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        skpc.main()  # must not crash with an uncaught IndexError
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "WARNING: skipping 00" in out
    assert "Skipped cameras: ['00']" in out
    assert (out_dir / "01" / "000000.json").is_file()  # good camera still processed
    assert not (out_dir / "00").exists()


def test_main_skips_camera_with_missing_instance_info_key(tmp_path, monkeypatch, capsys):
    kp2d_flat_dir = tmp_path / "flat"
    (kp2d_flat_dir / "images_flat").mkdir(parents=True)
    (kp2d_flat_dir / "images_flat" / "00.json").write_text(json.dumps({}))
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--kp2d_flat_dir", str(kp2d_flat_dir), "--out_dir", str(out_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        skpc.main()
    assert exc_info.value.code == 1
    assert "WARNING: skipping 00" in capsys.readouterr().out


def test_main_skips_camera_with_invalid_json(tmp_path, monkeypatch, capsys):
    kp2d_flat_dir = tmp_path / "flat"
    (kp2d_flat_dir / "images_flat").mkdir(parents=True)
    (kp2d_flat_dir / "images_flat" / "00.json").write_text("not valid json {{{")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--kp2d_flat_dir", str(kp2d_flat_dir), "--out_dir", str(out_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        skpc.main()
    assert exc_info.value.code == 1
    assert "WARNING: skipping 00" in capsys.readouterr().out


def test_main_uses_json_stem_as_camera_label_not_parent_images_dir_name(tmp_path, monkeypatch):
    # Regression-style check: label comes from src_path.stem (the json
    # filename), not the parent images_dir_name -- multiple images_dir_name
    # subdirs with differently-named camera json files must each land under
    # their own label.
    kp2d_flat_dir = tmp_path / "flat"
    write_kp2d(kp2d_flat_dir, "images_flat_a", "00")
    write_kp2d(kp2d_flat_dir, "images_flat_b", "01")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--kp2d_flat_dir", str(kp2d_flat_dir), "--out_dir", str(out_dir),
    ])
    skpc.main()
    assert (out_dir / "00" / "000000.json").is_file()
    assert (out_dir / "01" / "000000.json").is_file()
