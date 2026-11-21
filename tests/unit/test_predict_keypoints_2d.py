# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

import json
import subprocess
import sys
from pathlib import Path

import pytest

import predict_keypoints_2d as pk2d


def base_argv(images_dir, out_kp2d_dir, fmasks_dir, extra=None):
    argv = [
        "prog",
        "--images_dir", str(images_dir),
        "--out_kp2d_dir", str(out_kp2d_dir),
        "--fmasks_dir", str(fmasks_dir),
    ]
    return argv + (extra or [])


def make_dirs(tmp_path):
    images_dir = tmp_path / "images_flat"
    out_kp2d_dir = tmp_path / "kp2d"
    fmasks_dir = tmp_path / "fmasks_flat"
    images_dir.mkdir()
    fmasks_dir.mkdir()
    return images_dir, out_kp2d_dir, fmasks_dir


# --------------------------------------------------------------------------
# main() wiring: only one model now (goliath308) -- prove main() always
# calls run_goliath308, and --sapiens_checkpoint_root takes precedence over
# the env var (with the env var as fallback, and None when neither is set).
# --------------------------------------------------------------------------

def test_main_always_calls_run_goliath308(tmp_path, monkeypatch):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    calls = []
    monkeypatch.setattr(pk2d, "run_goliath308", lambda args, ckpt_root: calls.append(ckpt_root))
    monkeypatch.delenv("SAPIENS_CHECKPOINT_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", base_argv(images_dir, out_kp2d_dir, fmasks_dir))
    pk2d.main()
    assert calls == [None]


def test_main_ckpt_root_explicit_arg_overrides_env_var(tmp_path, monkeypatch):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    calls = []
    monkeypatch.setattr(pk2d, "run_goliath308", lambda args, ckpt_root: calls.append(ckpt_root))
    monkeypatch.setenv("SAPIENS_CHECKPOINT_ROOT", "/from/env")
    monkeypatch.setattr(sys, "argv", base_argv(
        images_dir, out_kp2d_dir, fmasks_dir, ["--sapiens_checkpoint_root", "/from/arg"],
    ))
    pk2d.main()
    assert calls == ["/from/arg"]


def test_main_ckpt_root_falls_back_to_env_var_when_arg_absent(tmp_path, monkeypatch):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    calls = []
    monkeypatch.setattr(pk2d, "run_goliath308", lambda args, ckpt_root: calls.append(ckpt_root))
    monkeypatch.setenv("SAPIENS_CHECKPOINT_ROOT", "/from/env")
    monkeypatch.setattr(sys, "argv", base_argv(images_dir, out_kp2d_dir, fmasks_dir))
    pk2d.main()
    assert calls == ["/from/env"]


def test_main_ckpt_root_none_when_neither_arg_nor_env_set(tmp_path, monkeypatch):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    calls = []
    monkeypatch.setattr(pk2d, "run_goliath308", lambda args, ckpt_root: calls.append(ckpt_root))
    monkeypatch.delenv("SAPIENS_CHECKPOINT_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", base_argv(images_dir, out_kp2d_dir, fmasks_dir))
    pk2d.main()
    assert calls == [None]


@pytest.mark.parametrize("missing_flag", ["--images_dir", "--out_kp2d_dir", "--fmasks_dir"])
def test_main_errors_when_required_arg_missing(tmp_path, monkeypatch, missing_flag):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    argv = base_argv(images_dir, out_kp2d_dir, fmasks_dir)
    idx = argv.index(missing_flag)
    argv = argv[:idx] + argv[idx + 2:]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        pk2d.main()
    assert exc_info.value.code == 2  # argparse's own usage-error exit code


def test_main_no_longer_accepts_model_flag(tmp_path, monkeypatch):
    # Regression guard: --model/--gpu_ids/--num_workers/coco_wholebody133
    # support was removed entirely -- only goliath308 exists now.
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    monkeypatch.setattr(sys, "argv", base_argv(
        images_dir, out_kp2d_dir, fmasks_dir, ["--model", "goliath308"],
    ))
    with pytest.raises(SystemExit) as exc_info:
        pk2d.main()
    assert exc_info.value.code == 2
    assert not hasattr(pk2d, "run_coco_wholebody133")
    assert not hasattr(pk2d, "LITE_VIS_POSE_SCRIPT")


# --------------------------------------------------------------------------
# run_goliath308 -- subprocess.run and resolve_model_files/default_device
# are monkeypatched (they shell out to the vendored Sapiens2 vis_pose.py
# and download checkpoints, not runnable here); everything around that
# boundary (cmd construction, exit-code propagation, calling
# split_combined_predictions) is real.
# --------------------------------------------------------------------------

class Args:
    def __init__(self, images_dir, out_kp2d_dir, fmasks_dir, sapiens_model_size="1b"):
        self.images_dir = images_dir
        self.out_kp2d_dir = out_kp2d_dir
        self.fmasks_dir = fmasks_dir
        self.sapiens_model_size = sapiens_model_size


def write_combined_predictions(out_kp2d_dir, cameras=("00", "01")):
    out_kp2d_dir.mkdir(parents=True, exist_ok=True)
    predictions_json = out_kp2d_dir / f"{out_kp2d_dir.name}_predictions.json"
    frames = [
        {
            "image_name": f"{cam}.jpg",
            "instances": [{"keypoints": [1.0, 2.0], "keypoint_scores": [0.9]}],
        }
        for cam in cameras
    ]
    predictions_json.write_text(json.dumps({"frames": frames}))
    return predictions_json


@pytest.fixture
def fake_models(monkeypatch):
    """Stub checkpoint resolution; records the (root, size) it was asked for."""
    asked = []

    def resolve(ckpt_root, size):
        asked.append((ckpt_root, size))
        return (ckpt_root / "pose" / f"sapiens2_{size}_pose.safetensors",
                Path(f"/sapiens/pose/configs/sapiens2_{size}.py"),
                str(ckpt_root / "detector" / "detr-resnet-101-dc5"))
    monkeypatch.setattr(pk2d, "resolve_model_files", resolve)
    monkeypatch.setattr(pk2d, "default_device", lambda: "cpu")
    return asked


def test_vis_pose_script_ships_in_repo():
    assert pk2d.VIS_POSE_SCRIPT.is_file()
    assert (pk2d.VIS_POSE_SCRIPT.parent / "pose_render_utils.py").is_file()
    assert (pk2d.VIS_POSE_SCRIPT.parent / "LICENSE.md").is_file()


def test_run_goliath308_errors_without_checkpoint_root(tmp_path, capsys):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        pk2d.run_goliath308(Args(images_dir, out_kp2d_dir, fmasks_dir), None)
    assert exc_info.value.code == 1
    assert "SAPIENS_CHECKPOINT_ROOT" in capsys.readouterr().out


def test_run_goliath308_propagates_nonzero_subprocess_exit_code(tmp_path, monkeypatch, fake_models):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    monkeypatch.setattr(pk2d.subprocess, "run",
                        lambda cmd: subprocess.CompletedProcess(cmd, 7, stdout="", stderr="boom"))
    with pytest.raises(SystemExit) as exc_info:
        pk2d.run_goliath308(Args(images_dir, out_kp2d_dir, fmasks_dir), "/ckpt")
    assert exc_info.value.code == 7


def test_run_goliath308_errors_when_combined_predictions_file_missing(tmp_path, monkeypatch, capsys,
                                                                     fake_models):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    monkeypatch.setattr(pk2d.subprocess, "run", lambda cmd: subprocess.CompletedProcess(cmd, 0))
    with pytest.raises(SystemExit) as exc_info:
        pk2d.run_goliath308(Args(images_dir, out_kp2d_dir, fmasks_dir), "/ckpt")
    assert exc_info.value.code == 1
    assert "expected combined predictions file not found" in capsys.readouterr().out


def test_run_goliath308_success_splits_combined_predictions_into_per_camera_files(tmp_path, monkeypatch,
                                                                                  fake_models):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)

    def fake_run(cmd):
        write_combined_predictions(out_kp2d_dir, cameras=("00", "01"))
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(pk2d.subprocess, "run", fake_run)

    pk2d.run_goliath308(Args(images_dir, out_kp2d_dir, fmasks_dir), "/ckpt")

    assert (out_kp2d_dir / images_dir.name / "00.json").is_file()
    assert (out_kp2d_dir / images_dir.name / "01.json").is_file()
    with open(out_kp2d_dir / images_dir.name / "00.json") as f:
        written = json.load(f)
    assert written == {"instance_info": [{"keypoints": [1.0, 2.0], "keypoint_scores": [0.9]}]}


def test_run_goliath308_cmd_construction(tmp_path, monkeypatch, fake_models):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        write_combined_predictions(out_kp2d_dir, cameras=())
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(pk2d.subprocess, "run", fake_run)

    pk2d.run_goliath308(Args(images_dir, out_kp2d_dir, fmasks_dir), "/ckpt")

    assert calls == [[
        sys.executable, str(pk2d.VIS_POSE_SCRIPT),
        "/ckpt/detector/detr-resnet-101-dc5", "/sapiens/pose/configs/sapiens2_1b.py",
        "/ckpt/pose/sapiens2_1b_pose.safetensors",
        "--input", str(images_dir), "--output", str(out_kp2d_dir),
        "--fmasks-dir", str(fmasks_dir),
        "--kpt-thr", "0.3", "--max-bone-sigma", "0.0", "--max-bilateral-ratio", "3.5",
        "--device", "cpu",
    ]]
    assert fake_models == [(Path("/ckpt"), "1b")]


def test_run_goliath308_forwards_smaller_model_size(tmp_path, monkeypatch, fake_models):
    # Real CI failure (2026-07-30): the actual GitHub runner has only
    # ~7.75GB RAM, and the 1b checkpoint alone peaks at 6.5-7.6GB -- CPU
    # mode uses a smaller checkpoint instead (0.4b measured at 4.1GB peak
    # on the real fixture).
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)

    def fake_run(cmd):
        write_combined_predictions(out_kp2d_dir, cameras=())
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(pk2d.subprocess, "run", fake_run)

    pk2d.run_goliath308(Args(images_dir, out_kp2d_dir, fmasks_dir, sapiens_model_size="0.4b"), "/ckpt/root")
    assert fake_models == [(Path("/ckpt/root"), "0.4b")]


# --------------------------------------------------------------------------
# Model file resolution -- the Hugging Face downloads are monkeypatched.
# --------------------------------------------------------------------------

def test_ensure_pose_checkpoint_keeps_existing_file(tmp_path, monkeypatch):
    # Returns before importing huggingface_hub, so this also runs (and
    # proves no download is attempted) where that package is absent.
    ckpt = tmp_path / "pose" / "sapiens2_1b_pose.safetensors"
    ckpt.parent.mkdir()
    ckpt.write_text("weights")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # any import would raise
    assert pk2d.ensure_pose_checkpoint(ckpt, "1b") == ckpt


def test_ensure_pose_checkpoint_downloads_matching_repo(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    cached = tmp_path / "hf_cache" / "sapiens2_0.4b_pose.safetensors"
    cached.parent.mkdir()
    cached.write_text("weights")
    asked = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, filename: asked.append((repo_id, filename)) or str(cached))
    monkeypatch.delenv("SAPIENS_HF_REPO", raising=False)
    ckpt = tmp_path / "root" / "pose" / "sapiens2_0.4b_pose.safetensors"
    assert pk2d.ensure_pose_checkpoint(ckpt, "0.4b") == ckpt
    assert ckpt.read_text() == "weights"
    assert asked == [("facebook/sapiens2-pose-0.4b", "sapiens2_0.4b_pose.safetensors")]


def test_ensure_detector_falls_back_to_repo_id_when_download_fails(tmp_path, monkeypatch, capsys):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    def boom(**kw):
        raise OSError("offline")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    monkeypatch.delenv("SAPIENS_DETECTOR_HF_REPO", raising=False)
    assert pk2d.ensure_detector(tmp_path / "detector") == "facebook/detr-resnet-101-dc5"
    assert "falling back" in capsys.readouterr().out


def test_ensure_detector_uses_existing_snapshot(tmp_path):
    detector = tmp_path / "detector"
    detector.mkdir()
    (detector / "config.json").write_text("{}")
    assert pk2d.ensure_detector(detector) == str(detector)


# --------------------------------------------------------------------------
# split_combined_predictions -- pure file/JSON logic, no mocking. Real
# (unmocked) execution against both well-formed and deliberately broken
# inputs.
# --------------------------------------------------------------------------

def test_split_normal_multi_frame_writes_per_camera_files(tmp_path):
    predictions_json = tmp_path / "predictions.json"
    predictions_json.write_text(json.dumps({"frames": [
        {"image_name": "00.jpg", "instances": [{"keypoints": [1, 2], "keypoint_scores": [0.9]}]},
        {"image_name": "01.jpg", "instances": [{"keypoints": [3, 4], "keypoint_scores": [0.8]}]},
    ]}))
    out_kp2d_dir = tmp_path / "out"
    pk2d.split_combined_predictions(predictions_json, out_kp2d_dir, "images_flat")

    with open(out_kp2d_dir / "images_flat" / "00.json") as f:
        assert json.load(f) == {"instance_info": [{"keypoints": [1, 2], "keypoint_scores": [0.9]}]}
    with open(out_kp2d_dir / "images_flat" / "01.json") as f:
        assert json.load(f) == {"instance_info": [{"keypoints": [3, 4], "keypoint_scores": [0.8]}]}


def test_split_label_taken_from_image_name_stem_ignoring_directory_prefix(tmp_path):
    predictions_json = tmp_path / "predictions.json"
    predictions_json.write_text(json.dumps({"frames": [
        {"image_name": "some/nested/path/00.jpg", "instances": [{"keypoints": [1], "keypoint_scores": [0.9]}]},
    ]}))
    out_kp2d_dir = tmp_path / "out"
    pk2d.split_combined_predictions(predictions_json, out_kp2d_dir, "images_flat")
    assert (out_kp2d_dir / "images_flat" / "00.json").is_file()


def test_split_empty_instances_list_warns_and_skips(tmp_path, capsys):
    predictions_json = tmp_path / "predictions.json"
    predictions_json.write_text(json.dumps({"frames": [
        {"image_name": "00.jpg", "instances": []},
        {"image_name": "01.jpg", "instances": [{"keypoints": [1], "keypoint_scores": [0.9]}]},
    ]}))
    out_kp2d_dir = tmp_path / "out"
    pk2d.split_combined_predictions(predictions_json, out_kp2d_dir, "images_flat")

    out = capsys.readouterr().out
    assert "WARNING: no detected instance for 00.jpg" in out
    assert not (out_kp2d_dir / "images_flat" / "00.json").exists()
    assert (out_kp2d_dir / "images_flat" / "01.json").is_file()
    assert "Split 1 camera(s)" in out


def test_split_missing_instances_key_treated_same_as_empty(tmp_path, capsys):
    predictions_json = tmp_path / "predictions.json"
    predictions_json.write_text(json.dumps({"frames": [
        {"image_name": "00.jpg"},  # no "instances" key at all
    ]}))
    out_kp2d_dir = tmp_path / "out"
    pk2d.split_combined_predictions(predictions_json, out_kp2d_dir, "images_flat")
    assert "WARNING: no detected instance for 00.jpg" in capsys.readouterr().out
    assert not (out_kp2d_dir / "images_flat").exists() or not list((out_kp2d_dir / "images_flat").iterdir())


def test_split_multiple_instances_takes_first_only(tmp_path):
    predictions_json = tmp_path / "predictions.json"
    predictions_json.write_text(json.dumps({"frames": [
        {"image_name": "00.jpg", "instances": [
            {"keypoints": [1, 1], "keypoint_scores": [0.1]},
            {"keypoints": [9, 9], "keypoint_scores": [0.9]},
        ]},
    ]}))
    out_kp2d_dir = tmp_path / "out"
    pk2d.split_combined_predictions(predictions_json, out_kp2d_dir, "images_flat")
    with open(out_kp2d_dir / "images_flat" / "00.json") as f:
        written = json.load(f)
    assert written["instance_info"][0]["keypoints"] == [1, 1]  # first instance, not the higher-scoring second


def test_split_dest_dir_uses_images_dir_name_not_out_kp2d_dir_name(tmp_path):
    predictions_json = tmp_path / "combined_predictions.json"
    predictions_json.write_text(json.dumps({"frames": [
        {"image_name": "00.jpg", "instances": [{"keypoints": [1], "keypoint_scores": [0.9]}]},
    ]}))
    out_kp2d_dir = tmp_path / "poses_2d_flat"
    pk2d.split_combined_predictions(predictions_json, out_kp2d_dir, "images_flat_a")
    assert (out_kp2d_dir / "images_flat_a" / "00.json").is_file()
    assert not (out_kp2d_dir / "poses_2d_flat").exists()


# --- regression tests: whole-file corruption -> clean error, not a raw
# traceback (same bug class already fixed in extract_synced_frames.py) ---

def test_split_errors_cleanly_on_invalid_json(tmp_path, capsys):
    predictions_json = tmp_path / "bad.json"
    predictions_json.write_text("not valid json {{{")
    with pytest.raises(SystemExit) as exc_info:
        pk2d.split_combined_predictions(predictions_json, tmp_path / "out", "images_flat")
    assert exc_info.value.code == 1
    assert "is not valid JSON" in capsys.readouterr().out


def test_split_errors_cleanly_on_missing_frames_key(tmp_path, capsys):
    predictions_json = tmp_path / "noframes.json"
    predictions_json.write_text(json.dumps({"not_frames": []}))
    with pytest.raises(SystemExit) as exc_info:
        pk2d.split_combined_predictions(predictions_json, tmp_path / "out", "images_flat")
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "is missing expected key" in out
    assert "'frames'" in out


# --- regression tests: per-frame corruption -> warn and skip that frame,
# batch continues (matching the pre-existing empty-instances skip pattern) ---

def test_split_frame_missing_image_name_warns_and_skips(tmp_path, capsys):
    predictions_json = tmp_path / "p.json"
    predictions_json.write_text(json.dumps({"frames": [
        {"instances": [{"keypoints": [1], "keypoint_scores": [0.9]}]},  # no image_name
        {"image_name": "01.jpg", "instances": [{"keypoints": [1], "keypoint_scores": [0.9]}]},
    ]}))
    out_kp2d_dir = tmp_path / "out"
    pk2d.split_combined_predictions(predictions_json, out_kp2d_dir, "images_flat")  # must not raise
    out = capsys.readouterr().out
    assert "WARNING: malformed frame entry (missing key 'image_name')" in out
    assert (out_kp2d_dir / "images_flat" / "01.json").is_file()  # good frame still processed


def test_split_instance_missing_keypoint_scores_warns_and_skips(tmp_path, capsys):
    predictions_json = tmp_path / "p.json"
    predictions_json.write_text(json.dumps({"frames": [
        {"image_name": "00.jpg", "instances": [{"keypoints": [1]}]},  # no keypoint_scores
        {"image_name": "01.jpg", "instances": [{"keypoints": [1], "keypoint_scores": [0.9]}]},
    ]}))
    out_kp2d_dir = tmp_path / "out"
    pk2d.split_combined_predictions(predictions_json, out_kp2d_dir, "images_flat")  # must not raise
    out = capsys.readouterr().out
    assert "WARNING: malformed frame entry (missing key 'keypoint_scores')" in out
    assert not (out_kp2d_dir / "images_flat" / "00.json").exists()
    assert (out_kp2d_dir / "images_flat" / "01.json").is_file()


def test_split_instance_missing_keypoints_warns_and_skips(tmp_path, capsys):
    predictions_json = tmp_path / "p.json"
    predictions_json.write_text(json.dumps({"frames": [
        {"image_name": "00.jpg", "instances": [{"keypoint_scores": [0.9]}]},  # no keypoints
    ]}))
    out_kp2d_dir = tmp_path / "out"
    pk2d.split_combined_predictions(predictions_json, out_kp2d_dir, "images_flat")
    assert "WARNING: malformed frame entry (missing key 'keypoints')" in capsys.readouterr().out
