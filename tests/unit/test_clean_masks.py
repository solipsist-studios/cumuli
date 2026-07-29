import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

import clean_masks as cm


def write_mask(path, w=20, h=20, val=255, fg_box=(5, 5, 15, 15)):
    arr = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = fg_box
    arr[y0:y1, x0:x1] = val
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def write_kp(path, keypoints, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"instance_info": [{"keypoints": keypoints, "keypoint_scores": scores}]}))


def base_argv(fmasks_dir, kp2d_dir, out_dir, extra=None):
    argv = ["prog", "--fmasks_dir", str(fmasks_dir), "--kp2d_dir", str(kp2d_dir), "--out_dir", str(out_dir)]
    return argv + (extra or [])


# --------------------------------------------------------------------------
# find_keypoints_json -- both supported directory layouts
# --------------------------------------------------------------------------

def test_find_keypoints_json_split_keypoints_per_camera_layout(tmp_path):
    (tmp_path / "00").mkdir()
    (tmp_path / "00" / "000000.json").write_text("{}")
    result = cm.find_keypoints_json(tmp_path, "00")
    assert result == tmp_path / "00" / "000000.json"


def test_find_keypoints_json_predict_keypoints_2d_layout(tmp_path):
    (tmp_path / "images_flat" / "00.json").parent.mkdir(parents=True)
    (tmp_path / "images_flat" / "00.json").write_text("{}")
    result = cm.find_keypoints_json(tmp_path, "00")
    assert result == tmp_path / "images_flat" / "00.json"


def test_find_keypoints_json_no_match_returns_none(tmp_path):
    assert cm.find_keypoints_json(tmp_path, "00") is None


def test_find_keypoints_json_nonexistent_dir_returns_none(tmp_path):
    assert cm.find_keypoints_json(tmp_path / "nonexistent", "00") is None


def test_find_keypoints_json_camera_dir_exists_but_empty_falls_back_to_rglob(tmp_path):
    (tmp_path / "00").mkdir()  # exists as a dir but has no *.json directly inside
    (tmp_path / "nested" / "00.json").parent.mkdir(parents=True)
    (tmp_path / "nested" / "00.json").write_text("{}")
    result = cm.find_keypoints_json(tmp_path, "00")
    assert result == tmp_path / "nested" / "00.json"


def test_find_keypoints_json_multiple_rglob_matches_picks_first_alphabetically(tmp_path):
    (tmp_path / "b" / "00.json").parent.mkdir(parents=True)
    (tmp_path / "b" / "00.json").write_text("{}")
    (tmp_path / "a" / "00.json").parent.mkdir(parents=True)
    (tmp_path / "a" / "00.json").write_text("{}")
    result = cm.find_keypoints_json(tmp_path, "00")
    assert result == tmp_path / "a" / "00.json"


# --------------------------------------------------------------------------
# load_keypoints -- score_thr filtering + malformed-input behavior (raises;
# main()'s caller is responsible for catching and falling back)
# --------------------------------------------------------------------------

def test_load_keypoints_filters_by_score_threshold(tmp_path):
    p = tmp_path / "kp.json"
    write_kp(p, [[1, 1], [2, 2], [3, 3]], [0.9, 0.4, 0.5])
    kpts = cm.load_keypoints(p, score_thr=0.5)
    assert kpts.tolist() == [[1, 1], [3, 3]]  # 0.4 excluded, 0.5 included (boundary is inclusive)


def test_load_keypoints_missing_instance_info_key_raises(tmp_path):
    p = tmp_path / "kp.json"
    p.write_text(json.dumps({}))
    with pytest.raises(KeyError):
        cm.load_keypoints(p, 0.5)


def test_load_keypoints_empty_instance_info_raises(tmp_path):
    p = tmp_path / "kp.json"
    p.write_text(json.dumps({"instance_info": []}))
    with pytest.raises(IndexError):
        cm.load_keypoints(p, 0.5)


def test_load_keypoints_missing_keypoints_key_raises(tmp_path):
    p = tmp_path / "kp.json"
    p.write_text(json.dumps({"instance_info": [{"keypoint_scores": [0.9]}]}))
    with pytest.raises(KeyError):
        cm.load_keypoints(p, 0.5)


# --------------------------------------------------------------------------
# coverage -- keypoint-in-mask fraction
# --------------------------------------------------------------------------

def test_coverage_empty_keypoints_returns_zero():
    mask_bool = np.ones((10, 10), dtype=bool)
    assert cm.coverage(mask_bool, np.zeros((0, 2))) == 0.0


def test_coverage_all_inside_returns_one():
    mask_bool = np.ones((10, 10), dtype=bool)
    kpts = np.array([[1, 1], [5, 5], [8, 8]])
    assert cm.coverage(mask_bool, kpts) == 1.0


def test_coverage_partial():
    mask_bool = np.zeros((10, 10), dtype=bool)
    mask_bool[:, :5] = True  # left half is foreground
    kpts = np.array([[1, 1], [8, 1]])  # one inside, one outside
    assert cm.coverage(mask_bool, kpts) == 0.5


def test_coverage_clips_out_of_bounds_keypoints():
    mask_bool = np.ones((10, 10), dtype=bool)
    kpts = np.array([[-5, -5], [999, 999]])  # both clip into bounds, both land on foreground
    assert cm.coverage(mask_bool, kpts) == 1.0


# --------------------------------------------------------------------------
# filter_components -- connected-component keypoint-hit filtering
# --------------------------------------------------------------------------

def test_filter_components_no_components_returns_unchanged():
    mask_bool = np.zeros((10, 10), dtype=bool)
    cleaned, kept, total = cm.filter_components(mask_bool, np.zeros((0, 2)), min_hits=1)
    assert kept == 0 and total == 0
    assert not cleaned.any()


def test_filter_components_keeps_component_hit_by_enough_keypoints():
    mask_bool = np.zeros((10, 10), dtype=bool)
    mask_bool[2:5, 2:5] = True
    kpts = np.array([[3, 3]])
    cleaned, kept, total = cm.filter_components(mask_bool, kpts, min_hits=1)
    assert total == 1 and kept == 1
    assert cleaned[3, 3]


def test_filter_components_drops_component_with_too_few_hits():
    mask_bool = np.zeros((10, 10), dtype=bool)
    mask_bool[2:5, 2:5] = True
    kpts = np.array([[3, 3]])  # only 1 hit
    cleaned, kept, total = cm.filter_components(mask_bool, kpts, min_hits=2)
    assert total == 1 and kept == 0
    assert not cleaned.any()


def test_filter_components_two_separate_components_one_kept_one_dropped():
    mask_bool = np.zeros((20, 20), dtype=bool)
    mask_bool[2:5, 2:5] = True     # component A -- will be hit
    mask_bool[15:18, 15:18] = True  # component B -- bystander, no hits
    kpts = np.array([[3, 3]])
    cleaned, kept, total = cm.filter_components(mask_bool, kpts, min_hits=1)
    assert total == 2 and kept == 1
    assert cleaned[3, 3] and not cleaned[16, 16]


def test_filter_components_clips_out_of_bounds_keypoints():
    mask_bool = np.zeros((10, 10), dtype=bool)
    mask_bool[0:3, 0:3] = True
    kpts = np.array([[-100, -100]])  # clips to (0, 0), inside the component
    cleaned, kept, total = cm.filter_components(mask_bool, kpts, min_hits=1)
    assert kept == 1


# --------------------------------------------------------------------------
# keypoint_crop_box -- margin math + boundary clamping
# --------------------------------------------------------------------------

def test_keypoint_crop_box_applies_margin():
    kpts = np.array([[10, 10], [20, 20]])
    box = cm.keypoint_crop_box(kpts, w=100, h=100, margin=0.5)
    # bbox is 10x10, margin 0.5 -> 5px each side
    assert box == (5, 5, 25, 25)


def test_keypoint_crop_box_clamped_to_image_bounds():
    kpts = np.array([[2, 2], [5, 5]])
    box = cm.keypoint_crop_box(kpts, w=10, h=10, margin=2.0)  # huge margin
    x0, y0, x1, y1 = box
    assert x0 >= 0 and y0 >= 0 and x1 <= 10 and y1 <= 10


def test_keypoint_crop_box_single_point_degenerates_without_crashing():
    kpts = np.array([[5, 5]])
    box = cm.keypoint_crop_box(kpts, w=100, h=100, margin=0.15)
    assert box == (5, 5, 5, 5)  # zero-size box, but no crash


# --------------------------------------------------------------------------
# run_birefnet_on_crops -- subprocess.run is monkeypatched (real BiRefNet
# model, not runnable here); crop saving/cmd construction/output parsing
# around it is real.
# --------------------------------------------------------------------------

def test_run_birefnet_on_crops_errors_when_script_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cm, "REMOVE_BG_SCRIPT", tmp_path / "does_not_exist.py")
    with pytest.raises(SystemExit) as exc_info:
        cm.run_birefnet_on_crops({}, ".png")
    assert exc_info.value.code == 1
    assert "Diffuman4D submodule checked out" in capsys.readouterr().out


def test_run_birefnet_on_crops_subprocess_failure_returns_empty_dict(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cm, "REMOVE_BG_SCRIPT", tmp_path / "remove_background.py")
    cm.REMOVE_BG_SCRIPT.write_text("")
    image_path = tmp_path / "00.jpg"
    Image.new("RGB", (50, 50)).save(image_path)

    def fake_run(cmd, cwd=None):
        return subprocess.CompletedProcess(cmd, 1)
    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    result = cm.run_birefnet_on_crops({"00": (image_path, (0, 0, 10, 10))}, ".png")
    assert result == {}
    assert "BiRefNet retry failed" in capsys.readouterr().out


def test_run_birefnet_on_crops_success_crops_and_parses_output(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "REMOVE_BG_SCRIPT", tmp_path / "remove_background.py")
    cm.REMOVE_BG_SCRIPT.write_text("")
    image_path = tmp_path / "00.jpg"
    Image.new("RGB", (50, 50)).save(image_path)

    # run_birefnet_on_crops uses its own tempfile.TemporaryDirectory for masks_dir,
    # so intercept via the cmd's masks_dir argument instead of a fixed path.
    def fake_run(cmd, cwd=None):
        masks_dir = Path(cmd[3])
        masks_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((10, 10), 255, dtype=np.uint8)).save(masks_dir / "00.png")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    result = cm.run_birefnet_on_crops({"00": (image_path, (0, 0, 10, 10))}, ".png")
    assert "00" in result
    crop_mask, box = result["00"]
    assert box == (0, 0, 10, 10)
    assert crop_mask.shape == (10, 10)
    assert (crop_mask == 255).all()


def test_run_birefnet_on_crops_no_output_mask_warns_and_omits_camera(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cm, "REMOVE_BG_SCRIPT", tmp_path / "remove_background.py")
    cm.REMOVE_BG_SCRIPT.write_text("")
    image_path = tmp_path / "00.jpg"
    Image.new("RGB", (50, 50)).save(image_path)

    def fake_run(cmd, cwd=None):
        return subprocess.CompletedProcess(cmd, 0)  # succeeds, writes nothing
    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    result = cm.run_birefnet_on_crops({"00": (image_path, (0, 0, 10, 10))}, ".png")
    assert result == {}
    assert "WARNING: retry produced no mask for 00" in capsys.readouterr().out


# --------------------------------------------------------------------------
# main() -- full pipeline wiring, fault isolation, regression tests for all
# 7 bugs found via real (unmocked) execution against broken inputs.
# --------------------------------------------------------------------------

def test_main_errors_when_fmasks_dir_not_a_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", base_argv(tmp_path / "nope", tmp_path / "kp2d", tmp_path / "out"))
    with pytest.raises(SystemExit) as exc_info:
        cm.main()
    assert exc_info.value.code == 1
    assert "is not a directory" in capsys.readouterr().out


def test_main_errors_when_no_masks_found(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    fmasks_dir.mkdir()
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, tmp_path / "kp2d", tmp_path / "out"))
    with pytest.raises(SystemExit) as exc_info:
        cm.main()
    assert exc_info.value.code == 1
    assert "no masks found" in capsys.readouterr().out


def test_main_errors_when_out_dir_is_a_regular_file(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png")
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[10, 10]], [0.9])
    out_dir = tmp_path / "out_as_file"
    out_dir.write_text("not a directory")
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir))
    with pytest.raises(SystemExit) as exc_info:
        cm.main()
    assert exc_info.value.code == 1
    assert "already exists and is not a directory" in capsys.readouterr().out


def test_main_no_keypoints_json_copies_raw_mask_unfiltered(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png")
    kp2d_dir = tmp_path / "kp2d"
    kp2d_dir.mkdir()
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir))
    cm.main()
    assert (out_dir / "00.png").is_file()
    assert "no keypoints JSON found" in capsys.readouterr().out
    # not in the report -- this camera bypassed filtering entirely
    with open(out_dir / "cleanup_report.json") as f:
        report = json.load(f)
    assert "00" not in report


def test_main_malformed_keypoints_json_falls_back_to_raw_copy_not_crash(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png")
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [], [])
    (kp2d_dir / "00" / "000000.json").write_text(json.dumps({"instance_info": []}))  # empty -> IndexError internally
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir))
    cm.main()  # must not raise
    assert (out_dir / "00.png").is_file()
    assert "malformed keypoints JSON" in capsys.readouterr().out


def test_main_corrupted_mask_file_skipped_good_camera_survives(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    fmasks_dir.mkdir()
    (fmasks_dir / "00.png").write_bytes(b"not a real png")
    write_mask(fmasks_dir / "01.png")
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[10, 10]], [0.9])
    write_kp(kp2d_dir / "01" / "000000.json", [[10, 10]], [0.9])
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir))
    cm.main()  # must not raise
    out = capsys.readouterr().out
    assert "could not read mask file" in out
    assert not (out_dir / "00.png").exists()
    assert (out_dir / "01.png").is_file()


def test_main_filters_bystander_component_and_writes_report(tmp_path, monkeypatch):
    fmasks_dir = tmp_path / "fm"
    # foreground has two blobs: subject (hit by keypoint) + bystander (not hit)
    arr = np.zeros((30, 30), dtype=np.uint8)
    arr[2:6, 2:6] = 255     # subject blob
    arr[20:25, 20:25] = 255  # bystander blob
    fmasks_dir.mkdir()
    Image.fromarray(arr).save(fmasks_dir / "00.png")
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[3, 3]], [0.9])
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir))
    cm.main()

    cleaned = np.asarray(Image.open(out_dir / "00.png"))
    assert cleaned[3, 3] == 255      # subject kept
    assert cleaned[22, 22] == 0      # bystander removed

    with open(out_dir / "cleanup_report.json") as f:
        report = json.load(f)
    assert report["00"]["components_total"] == 2
    assert report["00"]["components_kept"] == 1
    assert report["00"]["retried"] is False


def test_main_mask_emptied_flag_printed_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    # Regression test: an emptied mask used to still exit 0 -- a script
    # reporting success while producing a known-blank, unusable mask. Now
    # exits 1 so a caller (run_unified_pipeline.py, CI) can catch it
    # automatically instead of relying on someone reading stdout.
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", fg_box=(2, 2, 6, 6))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[3, 3]], [0.9])  # 1 hit only
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(
        fmasks_dir, kp2d_dir, out_dir, ["--min_hits", "5"],  # more hits than available
    ))
    with pytest.raises(SystemExit) as exc_info:
        cm.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "MASK EMPTIED" in out
    assert "ERROR: 1 camera(s) have a fully blank cleaned mask: ['00']" in out
    assert (out_dir / "00.png").is_file()  # report + output still written before the hard failure


def test_main_rejects_min_hits_zero(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png")
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[10, 10]], [0.9])
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir, ["--min_hits", "0"]))
    with pytest.raises(SystemExit) as exc_info:
        cm.main()
    assert exc_info.value.code == 1
    assert "--min_hits must be >= 1" in capsys.readouterr().out


def test_main_low_coverage_flagged_even_without_retry(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    # foreground only in a tiny corner; keypoint far away -> low coverage
    write_mask(fmasks_dir / "00.png", w=50, h=50, fg_box=(0, 0, 5, 5))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[2, 2], [40, 40]], [0.9, 0.9])
    out_dir = tmp_path / "out"
    # no --retry -- must still flag LOW COVERAGE in pass 2's report/print
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir, ["--min_hits", "1"]))
    cm.main()
    out = capsys.readouterr().out
    assert "LOW COVERAGE" in out


def test_main_retry_not_attempted_without_retry_flag(tmp_path, monkeypatch):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", w=50, h=50, fg_box=(0, 0, 5, 5))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[2, 2], [40, 40]], [0.9, 0.9])
    out_dir = tmp_path / "out"

    called = []
    monkeypatch.setattr(cm, "run_birefnet_on_crops", lambda *a, **k: called.append(1) or {})
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir))  # no --retry
    cm.main()
    assert called == []


def test_main_retry_skipped_when_zero_confident_keypoints(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", w=50, h=50, fg_box=(0, 0, 5, 5))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[40, 40]], [0.1])  # below default score_thr
    images_dir = tmp_path / "images"
    write_mask(images_dir / "00.jpg")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(
        fmasks_dir, kp2d_dir, out_dir, ["--retry", "--images_dir", str(images_dir)],
    ))
    # zero confident keypoints -> retry skipped -> pass 2 has nothing to hit
    # the one component -> mask ends up emptied -> exits 1 (see the dedicated
    # MASK EMPTIED regression test above for that behavior itself).
    with pytest.raises(SystemExit) as exc_info:
        cm.main()
    assert exc_info.value.code == 1
    assert "cannot compute a retry crop region" in capsys.readouterr().out


def test_main_retry_skipped_when_images_dir_not_given(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", w=50, h=50, fg_box=(0, 0, 5, 5))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[2, 2], [40, 40]], [0.9, 0.9])
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir, ["--retry"]))
    cm.main()
    assert "--images_dir not given" in capsys.readouterr().out


def test_main_retry_skipped_when_no_image_found(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", w=50, h=50, fg_box=(0, 0, 5, 5))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[2, 2], [40, 40]], [0.9, 0.9])
    images_dir = tmp_path / "images"
    images_dir.mkdir()  # empty -- no 00.* image
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(
        fmasks_dir, kp2d_dir, out_dir, ["--retry", "--images_dir", str(images_dir)],
    ))
    cm.main()
    assert "no image found in" in capsys.readouterr().out


def test_main_retry_invokes_birefnet_and_merges_result(tmp_path, monkeypatch):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", w=50, h=50, fg_box=(0, 0, 5, 5))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[2, 2], [40, 40]], [0.9, 0.9])
    images_dir = tmp_path / "images"
    write_mask(images_dir / "00.jpg")
    out_dir = tmp_path / "out"

    def fake_run_birefnet(crops, image_ext):
        assert "00" in crops
        image_path, box = crops["00"]
        x0, y0, x1, y1 = box
        return {"00": (np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8), box)}
    monkeypatch.setattr(cm, "run_birefnet_on_crops", fake_run_birefnet)

    monkeypatch.setattr(sys, "argv", base_argv(
        fmasks_dir, kp2d_dir, out_dir, ["--retry", "--images_dir", str(images_dir)],
    ))
    cm.main()

    with open(out_dir / "cleanup_report.json") as f:
        report = json.load(f)
    assert report["00"]["retried"] is True
    assert report["00"]["coverage_after"] == 1.0  # merged mask now covers the far keypoint too


def test_main_retry_merge_shape_mismatch_skips_merge_not_crash(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", w=50, h=50, fg_box=(0, 0, 5, 5))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[2, 2], [40, 40]], [0.9, 0.9])
    images_dir = tmp_path / "images"
    write_mask(images_dir / "00.jpg")
    out_dir = tmp_path / "out"

    def fake_run_birefnet_mismatch(crops, image_ext):
        return {"00": (np.full((3, 3), 255, dtype=np.uint8), crops["00"][1])}  # wrong shape
    monkeypatch.setattr(cm, "run_birefnet_on_crops", fake_run_birefnet_mismatch)

    monkeypatch.setattr(sys, "argv", base_argv(
        fmasks_dir, kp2d_dir, out_dir, ["--retry", "--images_dir", str(images_dir)],
    ))
    cm.main()  # must not raise
    assert "doesn't match" in capsys.readouterr().out
    assert (out_dir / "00.png").is_file()  # still produced output despite skipped merge


# --------------------------------------------------------------------------
# Wiring checks: main()'s default CLI values must actually reach the real
# math functions, not just when explicitly overridden.
# --------------------------------------------------------------------------

def test_main_default_score_thr_reaches_load_keypoints(tmp_path, monkeypatch):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png")
    kp2d_dir = tmp_path / "kp2d"
    # one keypoint just above 0.5 (default score_thr), one just below
    write_kp(kp2d_dir / "00" / "000000.json", [[10, 10], [1, 1]], [0.51, 0.49])
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir))  # no --score_thr override
    cm.main()
    with open(out_dir / "cleanup_report.json") as f:
        report = json.load(f)
    assert report["00"]["coverage_before"] == 1.0  # only the 0.51 keypoint counted, and it's inside


def test_main_default_min_hits_reaches_filter_components(tmp_path, monkeypatch):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", fg_box=(2, 2, 6, 6))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[3, 3]], [0.9])  # exactly 1 hit
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir))  # default min_hits=1
    cm.main()
    with open(out_dir / "cleanup_report.json") as f:
        report = json.load(f)
    assert report["00"]["components_kept"] == 1  # default of 1 is satisfied by exactly 1 hit


def test_main_default_warn_coverage_reaches_flagging(tmp_path, monkeypatch, capsys):
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", w=50, h=50, fg_box=(0, 0, 5, 5))
    kp2d_dir = tmp_path / "kp2d"
    # coverage 0.5 -- below default warn_coverage=0.9, above a hypothetical lower custom value
    write_kp(kp2d_dir / "00" / "000000.json", [[2, 2], [40, 40]], [0.9, 0.9])
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(fmasks_dir, kp2d_dir, out_dir))  # default warn_coverage
    cm.main()
    assert "LOW COVERAGE" in capsys.readouterr().out


def test_main_explicit_score_thr_override_reaches_load_keypoints(tmp_path, monkeypatch):
    # Re-audit gap closed: only the DEFAULT score_thr reaching main() was
    # tested above -- an explicit CLI override was never proven to reach
    # load_keypoints() through main() at all.
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png")  # fg at (5,5,15,15)
    kp2d_dir = tmp_path / "kp2d"
    # [10,10] inside fg, score 0.6; [1,1] outside fg, score 0.9
    write_kp(kp2d_dir / "00" / "000000.json", [[10, 10], [1, 1]], [0.6, 0.9])
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(
        fmasks_dir, kp2d_dir, out_dir, ["--score_thr", "0.7"],  # excludes the 0.6-score, inside-fg keypoint
    ))
    # excluding that keypoint leaves zero confident hits on the only
    # component -> MASK EMPTIED -> exit 1 (correct side effect, not a bug;
    # see test_main_mask_emptied_flag_printed_and_exits_nonzero for that
    # behavior in isolation).
    with pytest.raises(SystemExit) as exc_info:
        cm.main()
    assert exc_info.value.code == 1
    with open(out_dir / "cleanup_report.json") as f:
        report = json.load(f)
    assert report["00"]["coverage_before"] == 0.0  # only the 0.9-score, outside-fg keypoint counted


def test_main_explicit_warn_coverage_override_changes_flagging(tmp_path, monkeypatch, capsys):
    # Re-audit gap closed: only the DEFAULT warn_coverage reaching main()
    # was tested above -- an explicit override was never proven to actually
    # move the flagging threshold.
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", w=50, h=50, fg_box=(0, 0, 5, 5))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[2, 2], [40, 40]], [0.9, 0.9])  # actual coverage 0.5
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(
        fmasks_dir, kp2d_dir, out_dir, ["--min_hits", "1", "--warn_coverage", "0.3"],  # 0.5 > 0.3 -- should NOT flag
    ))
    cm.main()
    assert "LOW COVERAGE" not in capsys.readouterr().out


def test_main_explicit_crop_margin_override_reaches_keypoint_crop_box(tmp_path, monkeypatch):
    # Re-audit gap closed: --crop_margin had NO main()-level wiring test at
    # all (only the pure keypoint_crop_box() function was tested directly).
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", w=50, h=50, fg_box=(0, 0, 5, 5))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[10, 10], [40, 40]], [0.9, 0.9])
    images_dir = tmp_path / "images"
    write_mask(images_dir / "00.jpg")

    captured_boxes = {}
    def fake_run_birefnet(crops, image_ext):
        for cam, (image_path, box) in crops.items():
            captured_boxes[cam] = box
            x0, y0, x1, y1 = box
        return {cam: (np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8), box) for cam, (_, box) in crops.items()}
    monkeypatch.setattr(cm, "run_birefnet_on_crops", fake_run_birefnet)

    out_dir_default = tmp_path / "out_default"
    monkeypatch.setattr(sys, "argv", base_argv(
        fmasks_dir, kp2d_dir, out_dir_default, ["--retry", "--images_dir", str(images_dir)],
    ))
    cm.main()
    box_default = captured_boxes["00"]

    captured_boxes.clear()
    out_dir_wide = tmp_path / "out_wide"
    monkeypatch.setattr(sys, "argv", base_argv(
        fmasks_dir, kp2d_dir, out_dir_wide,
        ["--retry", "--images_dir", str(images_dir), "--crop_margin", "5.0"],
    ))
    cm.main()
    box_wide = captured_boxes["00"]

    # a much larger margin must produce a strictly larger (or equally
    # clamped-to-bounds) crop box -- confirms args.crop_margin actually
    # reaches keypoint_crop_box() through main(), not just the default
    x0d, y0d, x1d, y1d = box_default
    x0w, y0w, x1w, y1w = box_wide
    assert (x1w - x0w) >= (x1d - x0d)
    assert (y1w - y0w) >= (y1d - y0d)
    assert box_wide != box_default


def test_main_explicit_image_ext_override_reaches_retry_crops(tmp_path, monkeypatch):
    # Re-audit gap closed: --image_ext had no main()-level wiring test
    # confirming it reaches run_birefnet_on_crops (only its own default was
    # exercised incidentally by other retry tests, never asserted on).
    fmasks_dir = tmp_path / "fm"
    write_mask(fmasks_dir / "00.png", w=50, h=50, fg_box=(0, 0, 5, 5))
    kp2d_dir = tmp_path / "kp2d"
    write_kp(kp2d_dir / "00" / "000000.json", [[10, 10], [40, 40]], [0.9, 0.9])
    images_dir = tmp_path / "images"
    write_mask(images_dir / "00.jpg")

    captured_exts = []
    def fake_run_birefnet(crops, image_ext):
        captured_exts.append(image_ext)
        out = {}
        for cam, (_, box) in crops.items():
            x0, y0, x1, y1 = box
            out[cam] = (np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8), box)
        return out
    monkeypatch.setattr(cm, "run_birefnet_on_crops", fake_run_birefnet)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(
        fmasks_dir, kp2d_dir, out_dir, ["--retry", "--images_dir", str(images_dir), "--image_ext", ".jpg"],
    ))
    cm.main()
    assert captured_exts == [".jpg"]  # not the default ".png"


# --------------------------------------------------------------------------
# Property-based sweeps for the real math: coverage() and filter_components()
# --------------------------------------------------------------------------

@given(
    w=st.integers(min_value=5, max_value=50),
    h=st.integers(min_value=5, max_value=50),
    n_inside=st.integers(min_value=0, max_value=10),
    n_outside=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=40, deadline=None)
def test_coverage_fraction_holds_for_any_split_of_inside_outside_keypoints(w, h, n_inside, n_outside):
    mask_bool = np.zeros((h, w), dtype=bool)
    mid = w // 2
    if mid < 1:
        return
    mask_bool[:, :mid] = True  # left half is foreground

    inside_pts = [[float(x % mid), float(y % h)] for x, y in zip(range(n_inside), range(1, n_inside + 1))]
    outside_pts = [[float(mid + (x % max(w - mid, 1))), float(y % h)] for x, y in zip(range(n_outside), range(1, n_outside + 1))]
    kpts = np.array(inside_pts + outside_pts) if (inside_pts + outside_pts) else np.zeros((0, 2))

    cov = cm.coverage(mask_bool, kpts)
    total = n_inside + n_outside
    if total == 0:
        assert cov == 0.0
    else:
        expected = n_inside / total
        assert abs(cov - expected) < 1e-9


@given(min_hits=st.integers(min_value=1, max_value=5), hits=st.integers(min_value=0, max_value=10))
@settings(max_examples=40, deadline=None)
def test_filter_components_keeps_component_iff_hits_meet_min_hits(min_hits, hits):
    mask_bool = np.zeros((30, 30), dtype=bool)
    mask_bool[2:8, 2:8] = True  # single component
    kpts = np.array([[3.0, 3.0]] * hits) if hits else np.zeros((0, 2))
    _, kept, total = cm.filter_components(mask_bool, kpts, min_hits=min_hits)
    assert total == 1
    assert kept == (1 if hits >= min_hits else 0)
