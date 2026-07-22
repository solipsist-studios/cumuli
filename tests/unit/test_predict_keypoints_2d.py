import json
import subprocess
import sys

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
# run_goliath308 -- subprocess.run is monkeypatched (it shells out to the
# Diffuman4D Sapiens2 preprocess script, not runnable in this environment);
# everything around that boundary (path checks, env dict, cmd construction,
# exit-code propagation, calling split_combined_predictions) is real.
# --------------------------------------------------------------------------

class Args:
    def __init__(self, images_dir, out_kp2d_dir, fmasks_dir):
        self.images_dir = images_dir
        self.out_kp2d_dir = out_kp2d_dir
        self.fmasks_dir = fmasks_dir


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


def test_run_goliath308_errors_when_predict_keypoints_script_missing(tmp_path, monkeypatch, capsys):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    monkeypatch.setattr(pk2d, "PREDICT_KEYPOINTS_SCRIPT", tmp_path / "does_not_exist.py")
    args = Args(images_dir, out_kp2d_dir, fmasks_dir)
    with pytest.raises(SystemExit) as exc_info:
        pk2d.run_goliath308(args, None)
    assert exc_info.value.code == 1
    assert "hloc_validation branch" in capsys.readouterr().out


def test_run_goliath308_propagates_nonzero_subprocess_exit_code(tmp_path, monkeypatch):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    monkeypatch.setattr(pk2d, "PREDICT_KEYPOINTS_SCRIPT", tmp_path / "predict_keypoints.py")
    pk2d.PREDICT_KEYPOINTS_SCRIPT.write_text("")  # just needs .is_file() to be True

    def fake_run(cmd, cwd=None, env=None):
        return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="boom")
    monkeypatch.setattr(pk2d.subprocess, "run", fake_run)

    args = Args(images_dir, out_kp2d_dir, fmasks_dir)
    with pytest.raises(SystemExit) as exc_info:
        pk2d.run_goliath308(args, None)
    assert exc_info.value.code == 7


def test_run_goliath308_errors_when_combined_predictions_file_missing(tmp_path, monkeypatch, capsys):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    monkeypatch.setattr(pk2d, "PREDICT_KEYPOINTS_SCRIPT", tmp_path / "predict_keypoints.py")
    pk2d.PREDICT_KEYPOINTS_SCRIPT.write_text("")

    def fake_run(cmd, cwd=None, env=None):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")  # succeeds but writes nothing
    monkeypatch.setattr(pk2d.subprocess, "run", fake_run)

    args = Args(images_dir, out_kp2d_dir, fmasks_dir)
    with pytest.raises(SystemExit) as exc_info:
        pk2d.run_goliath308(args, None)
    assert exc_info.value.code == 1
    assert "expected combined predictions file not found" in capsys.readouterr().out


def test_run_goliath308_success_splits_combined_predictions_into_per_camera_files(tmp_path, monkeypatch):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    monkeypatch.setattr(pk2d, "PREDICT_KEYPOINTS_SCRIPT", tmp_path / "predict_keypoints.py")
    pk2d.PREDICT_KEYPOINTS_SCRIPT.write_text("")

    def fake_run(cmd, cwd=None, env=None):
        # simulate the external script's own output side effect
        write_combined_predictions(out_kp2d_dir, cameras=("00", "01"))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(pk2d.subprocess, "run", fake_run)

    args = Args(images_dir, out_kp2d_dir, fmasks_dir)
    pk2d.run_goliath308(args, None)

    assert (out_kp2d_dir / images_dir.name / "00.json").is_file()
    assert (out_kp2d_dir / images_dir.name / "01.json").is_file()
    with open(out_kp2d_dir / images_dir.name / "00.json") as f:
        written = json.load(f)
    assert written == {"instance_info": [{"keypoints": [1.0, 2.0], "keypoint_scores": [0.9]}]}


def test_run_goliath308_cmd_and_cwd_construction(tmp_path, monkeypatch):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    monkeypatch.setattr(pk2d, "PREDICT_KEYPOINTS_SCRIPT", tmp_path / "predict_keypoints.py")
    pk2d.PREDICT_KEYPOINTS_SCRIPT.write_text("")

    calls = []
    def fake_run(cmd, cwd=None, env=None):
        calls.append((cmd, cwd, env))
        write_combined_predictions(out_kp2d_dir, cameras=())
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(pk2d.subprocess, "run", fake_run)

    args = Args(images_dir, out_kp2d_dir, fmasks_dir)
    pk2d.run_goliath308(args, None)

    assert len(calls) == 1
    cmd, cwd, env = calls[0]
    assert cmd == [sys.executable, str(pk2d.PREDICT_KEYPOINTS_SCRIPT),
                    str(images_dir), str(out_kp2d_dir), "--fmasks_dir", str(fmasks_dir)]
    assert cwd == str(pk2d.DIFFUMAN4D_ROOT / "scripts" / "preprocess")
    assert "SAPIENS_CHECKPOINT_ROOT" not in env  # ckpt_root was None -- must not inject a bogus env entry


def test_run_goliath308_sets_env_var_when_ckpt_root_given(tmp_path, monkeypatch):
    images_dir, out_kp2d_dir, fmasks_dir = make_dirs(tmp_path)
    monkeypatch.setattr(pk2d, "PREDICT_KEYPOINTS_SCRIPT", tmp_path / "predict_keypoints.py")
    pk2d.PREDICT_KEYPOINTS_SCRIPT.write_text("")
    monkeypatch.delenv("SAPIENS_CHECKPOINT_ROOT", raising=False)

    calls = []
    def fake_run(cmd, cwd=None, env=None):
        calls.append(env)
        write_combined_predictions(out_kp2d_dir, cameras=())
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(pk2d.subprocess, "run", fake_run)

    args = Args(images_dir, out_kp2d_dir, fmasks_dir)
    pk2d.run_goliath308(args, "/ckpt/root")

    assert calls[0]["SAPIENS_CHECKPOINT_ROOT"] == "/ckpt/root"
    # base os.environ must still be present -- confirms env is a copy, not a fresh dict
    import os
    assert calls[0]["PATH"] == os.environ["PATH"]


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
