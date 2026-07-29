import json
import pickle
import subprocess
import sys

import pytest

import run_hloc as rh


# --------------------------------------------------------------------------
# restructure_flat_to_percam -- real filesystem restructuring, no mocking
# --------------------------------------------------------------------------

def test_restructure_moves_flat_images_into_percam_folders(tmp_path):
    (tmp_path / "0001.jpg").write_text("cam1")
    (tmp_path / "0002.jpg").write_text("cam2")

    camera_ids = rh.restructure_flat_to_percam(tmp_path, ".jpg")

    assert sorted(camera_ids) == ["0001", "0002"]
    assert (tmp_path / "Camera_0001" / "0000.jpg").read_text() == "cam1"
    assert (tmp_path / "Camera_0002" / "0000.jpg").read_text() == "cam2"
    assert not (tmp_path / "0001.jpg").exists()


def test_restructure_strips_undistorted_prefix(tmp_path):
    # offline_undistort.py's own output naming convention prefixes files
    # with "undistorted_" -- the camera id must not include that prefix.
    (tmp_path / "undistorted_0001.jpg").write_text("cam1")

    camera_ids = rh.restructure_flat_to_percam(tmp_path, ".jpg")

    assert camera_ids == ["0001"]
    assert (tmp_path / "Camera_0001" / "0000.jpg").is_file()


def test_restructure_overwrites_stale_target(tmp_path):
    # Regression test: a stale Camera_0001/0000.jpg left over from a PRIOR
    # run used to silently win over this run's fresh frame (the old code
    # only renamed into place "if not target.exists()") -- HLOC would then
    # run pose estimation on stale data with zero indication anything was
    # wrong.
    (tmp_path / "Camera_0001").mkdir()
    (tmp_path / "Camera_0001" / "0000.jpg").write_text("STALE")
    (tmp_path / "0001.jpg").write_text("FRESH")

    rh.restructure_flat_to_percam(tmp_path, ".jpg")

    assert (tmp_path / "Camera_0001" / "0000.jpg").read_text() == "FRESH"
    assert not (tmp_path / "0001.jpg").exists()


def test_restructure_returns_empty_list_when_no_matching_images(tmp_path):
    (tmp_path / "0001.png").write_text("wrong ext")
    assert rh.restructure_flat_to_percam(tmp_path, ".jpg") == []


def test_restructure_ignores_non_matching_extension(tmp_path):
    (tmp_path / "0001.jpg").write_text("real")
    (tmp_path / "notes.txt").write_text("stray file")

    camera_ids = rh.restructure_flat_to_percam(tmp_path, ".jpg")

    assert camera_ids == ["0001"]
    assert (tmp_path / "notes.txt").is_file()  # untouched


# --------------------------------------------------------------------------
# build_init_transforms -- real pkl reads, no mocking
# --------------------------------------------------------------------------

def write_calib_pkl(path, w=5312, h=4872, fx=1800.0, fy=1801.0, cx=2656.0, cy=2436.0):
    with open(path, "wb") as f:
        pickle.dump({
            "camera_matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "image_size": (w, h),
        }, f)


def test_build_init_transforms_reads_real_calibration(tmp_path):
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    write_calib_pkl(pkl_dir / "Camera_0001.pkl", w=100, h=200, fx=10.0, fy=11.0, cx=50.0, cy=100.0)
    out_path = tmp_path / "init.json"

    rh.build_init_transforms(pkl_dir, ["0001"], out_path)

    data = json.loads(out_path.read_text())
    fr = data["frames"][0]
    assert fr["camera_label"] == "Camera_0001"
    assert fr["w"] == 100 and fr["h"] == 200
    assert fr["fl_x"] == 10.0 and fr["fl_y"] == 11.0
    assert fr["cx"] == 50.0 and fr["cy"] == 100.0


def test_build_init_transforms_uses_fallback_when_pkl_missing(tmp_path, capsys):
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    out_path = tmp_path / "init.json"

    rh.build_init_transforms(pkl_dir, ["0001"], out_path)

    data = json.loads(out_path.read_text())
    fr = data["frames"][0]
    assert fr["w"] == 5568 and fr["h"] == 4872  # module-level fallback defaults
    assert "no calibration pkl for 0001" in capsys.readouterr().out


def test_build_init_transforms_warns_and_falls_back_on_corrupted_pkl(tmp_path, capsys):
    # Regression test: an unpicklable/corrupted calibration pkl used to
    # crash the ENTIRE init_transforms build with a raw UnpicklingError,
    # taking down every other camera's entry along with it.
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    (pkl_dir / "Camera_0001.pkl").write_bytes(b"not a real pickle")
    write_calib_pkl(pkl_dir / "Camera_0002.pkl")
    out_path = tmp_path / "init.json"

    rh.build_init_transforms(pkl_dir, ["0001", "0002"], out_path)  # must not raise

    data = json.loads(out_path.read_text())
    assert len(data["frames"]) == 2
    frames_by_label = {fr["camera_label"]: fr for fr in data["frames"]}
    assert frames_by_label["Camera_0001"]["w"] == 5568  # fallback used
    assert frames_by_label["Camera_0002"]["w"] == 5312  # real calibration used
    assert "could not read calibration pkl for 0001" in capsys.readouterr().out


def test_build_init_transforms_warns_and_falls_back_on_pkl_missing_expected_key(tmp_path, capsys):
    # Regression test: a pkl that unpickles fine but doesn't have the
    # expected dict shape (e.g. a stray file from a different pipeline
    # version) crashed with a raw KeyError.
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    with open(pkl_dir / "Camera_0001.pkl", "wb") as f:
        pickle.dump({"unrelated_key": 1}, f)
    out_path = tmp_path / "init.json"

    rh.build_init_transforms(pkl_dir, ["0001"], out_path)  # must not raise

    data = json.loads(out_path.read_text())
    assert data["frames"][0]["w"] == 5568  # fallback used
    assert "could not read calibration pkl for 0001" in capsys.readouterr().out


def test_build_init_transforms_writes_identity_transform_matrix(tmp_path):
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    out_path = tmp_path / "init.json"
    rh.build_init_transforms(pkl_dir, ["0001"], out_path)

    data = json.loads(out_path.read_text())
    assert data["frames"][0]["transform_matrix"] == [
        [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    ]


def test_build_init_transforms_camera_label_and_file_path_format(tmp_path):
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    out_path = tmp_path / "init.json"
    rh.build_init_transforms(pkl_dir, ["0007"], out_path)

    fr = json.loads(out_path.read_text())["frames"][0]
    assert fr["camera_label"] == "Camera_0007"
    assert fr["file_path"] == "images/Camera_0007/0000.jpg"


def test_build_init_transforms_empty_camera_ids_writes_empty_frames_list(tmp_path):
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    out_path = tmp_path / "init.json"
    rh.build_init_transforms(pkl_dir, [], out_path)

    data = json.loads(out_path.read_text())
    assert data == {"camera_model": "PINHOLE", "frames": []}


# --------------------------------------------------------------------------
# main() -- CLI wiring. subprocess.run is monkeypatched (it shells out to
# multiframe_sfm.py, a real HLOC pipeline not runnable here); restructure/
# build_init_transforms and everything around the subprocess boundary is real.
# --------------------------------------------------------------------------

def base_argv(undistorted_dir, undistorted_pkl_dir, outputs_dir, extra=None):
    argv = [
        "prog",
        "--undistorted_dir", str(undistorted_dir),
        "--undistorted_pkl_dir", str(undistorted_pkl_dir),
        "--outputs_dir", str(outputs_dir),
    ]
    return argv + (extra or [])


def make_rig(tmp_path, n_cams=2):
    undistorted_dir = tmp_path / "undist"
    pkl_dir = tmp_path / "pkls"
    undistorted_dir.mkdir()
    pkl_dir.mkdir()
    for i in range(1, n_cams + 1):
        cam_id = f"{i:04d}"
        (undistorted_dir / f"{cam_id}.jpg").write_text("frame")
        write_calib_pkl(pkl_dir / f"Camera_{cam_id}.pkl")
    return undistorted_dir, pkl_dir


def patch_subprocess(monkeypatch, calls, returncode=0):
    def fake_run(cmd, env=None):
        calls.append((cmd, env))
        return subprocess.CompletedProcess(cmd, returncode)
    monkeypatch.setattr(rh.subprocess, "run", fake_run)


def test_main_errors_when_multiframe_sfm_script_missing(tmp_path, monkeypatch, capsys):
    undistorted_dir, pkl_dir = make_rig(tmp_path)
    monkeypatch.setattr(sys, "argv", base_argv(
        undistorted_dir, pkl_dir, tmp_path / "out",
        ["--multiframe_sfm_script", str(tmp_path / "does_not_exist.py")],
    ))
    with pytest.raises(SystemExit) as exc_info:
        rh.main()
    assert exc_info.value.code == 1
    assert "not found" in capsys.readouterr().out


def test_main_errors_when_no_images_found(tmp_path, monkeypatch, capsys):
    # Regression test: an empty/wrong-extension --undistorted_dir used to
    # silently proceed with zero cameras (an empty init_transforms.json and
    # a doomed multiframe_sfm.py invocation) instead of failing cleanly.
    undistorted_dir = tmp_path / "undist"
    pkl_dir = tmp_path / "pkls"
    undistorted_dir.mkdir()
    pkl_dir.mkdir()
    sfm_script = tmp_path / "multiframe_sfm.py"
    sfm_script.write_text("")
    monkeypatch.setattr(sys, "argv", base_argv(
        undistorted_dir, pkl_dir, tmp_path / "out",
        ["--multiframe_sfm_script", str(sfm_script)],
    ))
    with pytest.raises(SystemExit) as exc_info:
        rh.main()
    assert exc_info.value.code == 1
    assert "no .jpg files found" in capsys.readouterr().out


def test_main_skip_setup_skips_restructure_and_init_transforms(tmp_path, monkeypatch):
    undistorted_dir = tmp_path / "undist"
    pkl_dir = tmp_path / "pkls"
    undistorted_dir.mkdir()
    pkl_dir.mkdir()
    sfm_script = tmp_path / "multiframe_sfm.py"
    sfm_script.write_text("")
    calls = []
    patch_subprocess(monkeypatch, calls)
    outputs_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(
        undistorted_dir, pkl_dir, outputs_dir,
        ["--multiframe_sfm_script", str(sfm_script), "--skip_setup"],
    ))
    rh.main()

    assert not (outputs_dir.parent / "init_transforms.json").exists()
    assert len(calls) == 1


def test_main_cmd_construction_basic(tmp_path, monkeypatch):
    undistorted_dir, pkl_dir = make_rig(tmp_path)
    sfm_script = tmp_path / "multiframe_sfm.py"
    sfm_script.write_text("")
    calls = []
    patch_subprocess(monkeypatch, calls)
    outputs_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(
        undistorted_dir, pkl_dir, outputs_dir, ["--multiframe_sfm_script", str(sfm_script)],
    ))
    rh.main()

    assert len(calls) == 1
    cmd, env = calls[0]
    assert cmd[0] == sys.executable
    assert cmd[cmd.index("--frames_root") + 1] == str(undistorted_dir)
    assert cmd[cmd.index("--init_transforms") + 1] == str(outputs_dir.parent / "init_transforms.json")
    assert cmd[cmd.index("--undistorted_calibration_dir") + 1] == str(pkl_dir)
    assert cmd[cmd.index("--outputs_dir") + 1] == str(outputs_dir)
    assert cmd[cmd.index("--num_timestamps") + 1] == "1"
    assert cmd[cmd.index("--feature_type") + 1] == "superpoint"
    assert "--resize_max" not in cmd
    assert "--max_keypoints" not in cmd


def test_main_resize_max_included_when_given(tmp_path, monkeypatch):
    undistorted_dir, pkl_dir = make_rig(tmp_path)
    sfm_script = tmp_path / "multiframe_sfm.py"
    sfm_script.write_text("")
    calls = []
    patch_subprocess(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", base_argv(
        undistorted_dir, pkl_dir, tmp_path / "out",
        ["--multiframe_sfm_script", str(sfm_script), "--resize_max", "4000"],
    ))
    rh.main()
    cmd, _ = calls[0]
    assert cmd[cmd.index("--resize_max") + 1] == "4000"


def test_main_max_keypoints_included_when_given(tmp_path, monkeypatch):
    undistorted_dir, pkl_dir = make_rig(tmp_path)
    sfm_script = tmp_path / "multiframe_sfm.py"
    sfm_script.write_text("")
    calls = []
    patch_subprocess(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", base_argv(
        undistorted_dir, pkl_dir, tmp_path / "out",
        ["--multiframe_sfm_script", str(sfm_script), "--max_keypoints", "16384"],
    ))
    rh.main()
    cmd, _ = calls[0]
    assert cmd[cmd.index("--max_keypoints") + 1] == "16384"


def test_main_rejects_invalid_feature_type(tmp_path, monkeypatch):
    undistorted_dir, pkl_dir = make_rig(tmp_path)
    monkeypatch.setattr(sys, "argv", base_argv(
        undistorted_dir, pkl_dir, tmp_path / "out", ["--feature_type", "bogus"],
    ))
    with pytest.raises(SystemExit) as exc_info:
        rh.main()
    assert exc_info.value.code == 2


def test_main_env_vars_set_for_subprocess(tmp_path, monkeypatch):
    undistorted_dir, pkl_dir = make_rig(tmp_path)
    sfm_script = tmp_path / "multiframe_sfm.py"
    sfm_script.write_text("")
    calls = []
    patch_subprocess(monkeypatch, calls)
    monkeypatch.setenv("SOME_EXISTING_VAR", "keep_me")
    monkeypatch.setattr(sys, "argv", base_argv(
        undistorted_dir, pkl_dir, tmp_path / "out", ["--multiframe_sfm_script", str(sfm_script)],
    ))
    rh.main()
    _, env = calls[0]
    assert env["__NV_PRIME_RENDER_OFFLOAD"] == "1"
    assert env["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["SOME_EXISTING_VAR"] == "keep_me"  # base os.environ preserved, not replaced


def test_main_returns_normally_on_subprocess_success(tmp_path, monkeypatch, capsys):
    undistorted_dir, pkl_dir = make_rig(tmp_path)
    sfm_script = tmp_path / "multiframe_sfm.py"
    sfm_script.write_text("")
    calls = []
    patch_subprocess(monkeypatch, calls, returncode=0)
    outputs_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base_argv(
        undistorted_dir, pkl_dir, outputs_dir, ["--multiframe_sfm_script", str(sfm_script)],
    ))
    rh.main()  # must NOT raise SystemExit on success
    assert f"Done. Real camera poses at: {outputs_dir / 'transforms_multiframe.json'}" in capsys.readouterr().out


@pytest.mark.parametrize("returncode", [1, 137])
def test_main_propagates_nonzero_subprocess_returncode(tmp_path, monkeypatch, returncode):
    undistorted_dir, pkl_dir = make_rig(tmp_path)
    sfm_script = tmp_path / "multiframe_sfm.py"
    sfm_script.write_text("")
    calls = []
    patch_subprocess(monkeypatch, calls, returncode=returncode)
    monkeypatch.setattr(sys, "argv", base_argv(
        undistorted_dir, pkl_dir, tmp_path / "out", ["--multiframe_sfm_script", str(sfm_script)],
    ))
    with pytest.raises(SystemExit) as exc_info:
        rh.main()
    assert exc_info.value.code == returncode


def test_main_init_transforms_path_derived_from_outputs_dir_parent(tmp_path, monkeypatch):
    undistorted_dir, pkl_dir = make_rig(tmp_path)
    sfm_script = tmp_path / "multiframe_sfm.py"
    sfm_script.write_text("")
    calls = []
    patch_subprocess(monkeypatch, calls)
    outputs_dir = tmp_path / "run1" / "solipsist_out"
    monkeypatch.setattr(sys, "argv", base_argv(
        undistorted_dir, pkl_dir, outputs_dir, ["--multiframe_sfm_script", str(sfm_script)],
    ))
    rh.main()
    assert (tmp_path / "run1" / "init_transforms.json").is_file()


@pytest.mark.parametrize("missing_flag", ["--undistorted_dir", "--undistorted_pkl_dir", "--outputs_dir"])
def test_main_errors_when_required_arg_missing(tmp_path, monkeypatch, missing_flag):
    undistorted_dir, pkl_dir = make_rig(tmp_path)
    argv = base_argv(undistorted_dir, pkl_dir, tmp_path / "out")
    idx = argv.index(missing_flag)
    argv = argv[:idx] + argv[idx + 2:]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        rh.main()
    assert exc_info.value.code == 2
