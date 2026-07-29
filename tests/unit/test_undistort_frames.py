import os
import pickle
import subprocess
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL")
cv2 = pytest.importorskip("cv2")

import numpy as np

import undistort_frames as uf

RUNNING_AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0
SKIP_IF_ROOT = pytest.mark.skipif(
    RUNNING_AS_ROOT, reason="root ignores chmod(0o000); permission-denial can't be simulated"
)


# --------------------------------------------------------------------------
# resolve_calib_path -- pure logic, naming-convention fallback
# --------------------------------------------------------------------------

def test_resolve_calib_path_prefers_documented_convention_when_both_exist(tmp_path):
    (tmp_path / "Camera_0001.pkl").write_text("")
    (tmp_path / "cam1_calibration_data.pkl").write_text("")
    assert uf.resolve_calib_path(tmp_path, "0001") == tmp_path / "Camera_0001.pkl"


def test_resolve_calib_path_falls_back_to_legacy_naming(tmp_path):
    (tmp_path / "cam1_calibration_data.pkl").write_text("")
    assert uf.resolve_calib_path(tmp_path, "0001") == tmp_path / "cam1_calibration_data.pkl"


def test_resolve_calib_path_returns_documented_path_when_neither_exists(tmp_path):
    # For a clean "no calibration file at ..." error message downstream.
    assert uf.resolve_calib_path(tmp_path, "0001") == tmp_path / "Camera_0001.pkl"


def test_resolve_calib_path_handles_non_integer_camera_id(tmp_path):
    # cam_id that can't be int()-parsed must not crash building the legacy
    # candidate -- just skip it and fall back to the documented path.
    result = uf.resolve_calib_path(tmp_path, "left")
    assert result == tmp_path / "Camera_left.pkl"


# --------------------------------------------------------------------------
# validate_and_rescale -- pure logic, calibration/media resolution reconciliation
# --------------------------------------------------------------------------

def make_calib(fx=800.0, fy=800.0, cx=960.0, cy=540.0, image_size=(1920, 1080)):
    return {
        "camera_matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
        "image_size": image_size,
    }


def test_validate_and_rescale_rejects_wrong_matrix_shape():
    # Regression test: a malformed camera_matrix shape used to pass
    # through silently (numpy accepts any shape into np.asarray) and go on
    # to produce garbage/corrupted output downstream while main() still
    # reported success.
    calib = make_calib(image_size=(200, 150))
    calib["camera_matrix"] = [[800.0, 0.0], [0.0, 800.0]]  # 2x2, not 3x3
    with pytest.raises(ValueError, match="expected \\(3, 3\\)"):
        uf.validate_and_rescale(calib, 200, 150, "0001", allow_rescale=True)


def test_validate_and_rescale_rejects_nan_matrix():
    # Regression test: NaN in camera_matrix used to pass through with NO
    # error and no warning at all when image_size matched exactly -- the
    # most dangerous variant, since main() reported success and wrote a
    # real-looking but silently corrupted (all-black) output frame.
    calib = make_calib(image_size=(200, 150))
    calib["camera_matrix"] = [[float("nan"), 0.0, 100.0], [0.0, 800.0, 75.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="NaN/Inf"):
        uf.validate_and_rescale(calib, 200, 150, "0001", allow_rescale=True)


def test_validate_and_rescale_rejects_inf_matrix():
    calib = make_calib(image_size=(200, 150))
    calib["camera_matrix"] = [[float("inf"), 0.0, 100.0], [0.0, 800.0, 75.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="NaN/Inf"):
        uf.validate_and_rescale(calib, 200, 150, "0001", allow_rescale=True)


def test_validate_and_rescale_rejects_non_positive_focal_length():
    calib = make_calib(fx=-800.0, image_size=(200, 150))
    with pytest.raises(ValueError, match="non-positive focal length"):
        uf.validate_and_rescale(calib, 200, 150, "0001", allow_rescale=True)


def test_validate_and_rescale_rejects_zero_focal_length():
    calib = make_calib(fy=0.0, image_size=(200, 150))
    with pytest.raises(ValueError, match="non-positive focal length"):
        uf.validate_and_rescale(calib, 200, 150, "0001", allow_rescale=True)


def test_main_produces_no_output_for_nan_calibration_instead_of_silent_corruption(tmp_path, monkeypatch, capsys):
    # End-to-end regression test for the exact scenario found via manual
    # real-execution testing: a NaN camera_matrix must fail loudly and
    # produce NO output file, not report success with a corrupted one.
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    calib_path = calib_dir / "Camera_0001.pkl"
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_path, "wb") as f:
        pickle.dump({
            "camera_matrix": [[float("nan"), 0.0, 100.0], [0.0, 800.0, 75.0], [0.0, 0.0, 1.0]],
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
            "image_size": (200, 150),
        }, f)
    patch_undistort_script(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    assert "NaN/Inf" in capsys.readouterr().out
    assert not (out_dir / "0001.jpg").exists()


def test_validate_and_rescale_passes_when_image_size_matches():
    calib = make_calib(image_size=(1920, 1080))
    result, rescaled = uf.validate_and_rescale(calib, 1920, 1080, "0001", allow_rescale=True)
    assert rescaled is False
    assert result["camera_matrix"] == calib["camera_matrix"]


def test_validate_and_rescale_rescales_uniform_mismatch_with_correct_math():
    calib = make_calib(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, image_size=(1920, 1080))
    result, rescaled = uf.validate_and_rescale(calib, 960, 540, "0001", allow_rescale=True)
    assert rescaled is True
    K = np.asarray(result["camera_matrix"])
    # rw = rh = 0.5 -- fx, cx, fy, cy must all scale by exactly that factor.
    assert K[0, 0] == pytest.approx(500.0)
    assert K[0, 2] == pytest.approx(480.0)
    assert K[1, 1] == pytest.approx(500.0)
    assert K[1, 2] == pytest.approx(270.0)
    assert result["image_size"] == (960, 540)


def test_validate_and_rescale_rejects_uniform_mismatch_with_no_rescale():
    calib = make_calib(image_size=(1920, 1080))
    with pytest.raises(ValueError, match="no_rescale"):
        uf.validate_and_rescale(calib, 960, 540, "0001", allow_rescale=False)


def test_validate_and_rescale_rejects_non_uniform_mismatch_regardless_of_allow_rescale():
    calib = make_calib(image_size=(1920, 1080))
    with pytest.raises(ValueError, match="non-uniform scale"):
        uf.validate_and_rescale(calib, 960, 1080, "0001", allow_rescale=True)  # only width halved


def test_validate_and_rescale_heuristic_warns_on_offcenter_principal_point(capsys):
    calib = make_calib(cx=960.0, cy=540.0, image_size=(None, None))
    result, rescaled = uf.validate_and_rescale(calib, 640, 480, "0001", allow_rescale=True)
    assert rescaled is False  # heuristic branch never rescales, only warns
    assert "far from the media" in capsys.readouterr().out


def test_validate_and_rescale_heuristic_silent_when_centered(capsys):
    calib = make_calib(cx=960.0, cy=540.0, image_size=None)
    result, rescaled = uf.validate_and_rescale(calib, 1920, 1080, "0001", allow_rescale=True)
    assert capsys.readouterr().out == ""


def test_validate_and_rescale_treats_partial_none_image_size_as_missing(capsys):
    # (1920, None) must fall into the heuristic branch, not crash trying to
    # int()-compare a None.
    calib = make_calib(cx=960.0, cy=540.0, image_size=(1920, None))
    result, rescaled = uf.validate_and_rescale(calib, 640, 480, "0001", allow_rescale=True)
    assert rescaled is False
    assert "far from the media" in capsys.readouterr().out


# --------------------------------------------------------------------------
# single_warp_undistort -- real cv2 remap math, not mocked
# --------------------------------------------------------------------------

def write_test_frame(path, w=200, h=150):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (60, 120, 200)
    cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 255, 255), -1)
    cv2.imwrite(str(path), img)


def test_single_warp_undistort_rejects_nan_target_matrix(tmp_path):
    # Regression test: the TARGET pkl (single-warp mode) never goes
    # through validate_and_rescale() at all -- a NaN K_t used to pass
    # through silently and produce a real-looking but structurally
    # destroyed output frame while reporting success. Traced with a real
    # checkerboard-pattern frame: the white rectangle was completely gone
    # from the output, replaced by a uniform background color everywhere.
    frame_path = tmp_path / "frame.jpg"
    write_test_frame(frame_path, w=200, h=150)
    calib = make_calib(image_size=(200, 150))
    target = make_calib(image_size=(100, 80))
    target["camera_matrix"] = [[float("nan"), 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="target camera_matrix contains NaN/Inf"):
        uf.single_warp_undistort(frame_path, calib, target, "OPENCV", tmp_path / "out.jpg")


def test_single_warp_undistort_rejects_wrong_shape_target_matrix(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    write_test_frame(frame_path, w=200, h=150)
    calib = make_calib(image_size=(200, 150))
    target = make_calib(image_size=(100, 80))
    target["camera_matrix"] = [[100.0, 0.0], [0.0, 100.0]]  # 2x2, not 3x3
    with pytest.raises(ValueError, match="target camera_matrix has shape"):
        uf.single_warp_undistort(frame_path, calib, target, "OPENCV", tmp_path / "out.jpg")


def test_single_warp_undistort_rejects_non_positive_focal_length_target_matrix(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    write_test_frame(frame_path, w=200, h=150)
    calib = make_calib(image_size=(200, 150))
    target = make_calib(fx=-100.0, image_size=(100, 80))
    with pytest.raises(ValueError, match="target camera_matrix has non-positive focal length"):
        uf.single_warp_undistort(frame_path, calib, target, "OPENCV", tmp_path / "out.jpg")


def test_main_single_warp_produces_no_output_for_nan_target_matrix(tmp_path, monkeypatch, capsys):
    # End-to-end confirmation through main(), matching the source-matrix
    # regression test above.
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    with open(target_dir / "Camera_0001.pkl", "wb") as f:
        pickle.dump({
            "camera_matrix": [[float("nan"), 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
            "image_size": (100, 80),
        }, f)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--target_pkl_dir", str(target_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    assert "target camera_matrix contains NaN/Inf" in capsys.readouterr().out
    assert not (out_dir / "0001.jpg").exists()


def test_single_warp_undistort_raises_on_unreadable_frame(tmp_path):
    calib = make_calib()
    target = make_calib()
    with pytest.raises(ValueError, match="could not read"):
        uf.single_warp_undistort(tmp_path / "nope.jpg", calib, target, "OPENCV", tmp_path / "out.jpg")


@SKIP_IF_ROOT
def test_single_warp_undistort_raises_on_write_failure(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    write_test_frame(frame_path)
    calib = make_calib()
    target = make_calib()
    noperm_dir = tmp_path / "noperm"
    noperm_dir.mkdir()
    noperm_dir.chmod(0o000)
    try:
        with pytest.raises(ValueError, match="could not write"):
            uf.single_warp_undistort(frame_path, calib, target, "OPENCV", noperm_dir / "out.jpg")
    finally:
        noperm_dir.chmod(0o755)


def test_single_warp_undistort_fisheye_model_produces_target_sized_output(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    write_test_frame(frame_path, w=200, h=150)
    calib = make_calib(image_size=(200, 150))
    calib["distortion_coefficients"] = [0.05, -0.01, 0.0, 0.0]  # real, mild fisheye distortion
    target = make_calib(fx=100.0, fy=100.0, cx=50.0, cy=40.0, image_size=(100, 80))
    out_path = tmp_path / "out.jpg"

    (t_w, t_h), K_t = uf.single_warp_undistort(frame_path, calib, target, "OPENCV_FISHEYE", out_path)

    assert (t_w, t_h) == (100, 80)
    assert out_path.is_file()
    out_img = cv2.imread(str(out_path))
    assert out_img.shape[:2] == (80, 100)  # (height, width)


def test_single_warp_undistort_opencv_model_produces_target_sized_output(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    write_test_frame(frame_path, w=200, h=150)
    calib = make_calib(image_size=(200, 150))
    calib["distortion_coefficients"] = [0.02, 0.0, 0.0, 0.0, 0.0]
    target = make_calib(fx=90.0, fy=90.0, cx=60.0, cy=45.0, image_size=(120, 90))
    out_path = tmp_path / "out.jpg"

    (t_w, t_h), K_t = uf.single_warp_undistort(frame_path, calib, target, "OPENCV", out_path)

    assert (t_w, t_h) == (120, 90)
    out_img = cv2.imread(str(out_path))
    assert out_img.shape[:2] == (90, 120)


def test_single_warp_undistort_identity_remap_preserves_image_when_zero_distortion_and_matching_k(tmp_path):
    # Real correctness check, not just "didn't crash": zero distortion +
    # source K == target K + same image_size must be a near-identity remap.
    frame_path = tmp_path / "frame.jpg"
    write_test_frame(frame_path, w=200, h=150)
    same_calib = make_calib(fx=300.0, fy=300.0, cx=100.0, cy=75.0, image_size=(200, 150))
    out_path = tmp_path / "out.jpg"

    uf.single_warp_undistort(frame_path, same_calib, same_calib, "OPENCV", out_path)

    original = cv2.imread(str(frame_path)).astype(np.int16)
    result = cv2.imread(str(out_path)).astype(np.int16)
    # LANCZOS4 interpolation on an identity map isn't bit-exact, but should
    # be very close everywhere -- a real undistortion bug (wrong K used,
    # axes swapped, etc.) would produce a much larger mean difference.
    mean_abs_diff = np.mean(np.abs(original - result))
    assert mean_abs_diff < 5.0


# --------------------------------------------------------------------------
# main() -- CLI wiring for both modes. UNDISTORT_SCRIPT is monkeypatched to
# a fake existing file so wrapper-mode tests don't need the real
# deps/camera-calibration submodule; subprocess.run is monkeypatched so
# offline_undistort.py itself is never actually invoked.
# --------------------------------------------------------------------------

def write_calib_pkl(path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(make_calib(**kwargs), f)


def write_frame(path, w=200, h=150):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_test_frame(path, w=w, h=h)


def base_dirs(tmp_path):
    frames_dir = tmp_path / "frames"
    calib_dir = tmp_path / "calib"
    out_dir = tmp_path / "out"
    out_pkl_dir = tmp_path / "out_pkls"
    frames_dir.mkdir()
    calib_dir.mkdir()
    return frames_dir, calib_dir, out_dir, out_pkl_dir


def patch_undistort_script(monkeypatch, tmp_path, exists=True):
    fake_script = tmp_path / "offline_undistort.py"
    if exists:
        fake_script.write_text("")
    monkeypatch.setattr(uf, "UNDISTORT_SCRIPT", fake_script)


def test_main_errors_when_undistort_script_missing_in_wrapper_mode(tmp_path, monkeypatch, capsys):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    patch_undistort_script(monkeypatch, tmp_path, exists=False)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    assert "camera-calibration submodule" in capsys.readouterr().out


def test_main_single_warp_mode_does_not_require_undistort_script(tmp_path, monkeypatch):
    # The UNDISTORT_SCRIPT check is `target_pkl_dir is None and not exists`
    # -- proves the short-circuit is correct: single-warp mode must work
    # even when the camera-calibration submodule isn't checked out at all,
    # since it never invokes offline_undistort.py.
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path, exists=False)  # submodule "not checked out"
    target_dir = tmp_path / "target"
    write_calib_pkl(target_dir / "Camera_0001.pkl", fx=100.0, fy=100.0, cx=50.0, cy=40.0, image_size=(100, 80))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--target_pkl_dir", str(target_dir),
    ])
    uf.main()  # should not raise
    assert (out_dir / "0001.jpg").is_file()


def test_main_errors_when_no_frames_found(tmp_path, monkeypatch, capsys):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    patch_undistort_script(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    assert "no .jpg files found" in capsys.readouterr().out


def test_main_image_ext_override_changes_glob_and_output_name(tmp_path, monkeypatch):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.png")
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)
    target_dir = tmp_path / "target"
    write_calib_pkl(target_dir / "Camera_0001.pkl", fx=100.0, fy=100.0, cx=50.0, cy=40.0, image_size=(100, 80))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--image_ext", ".png", "--target_pkl_dir", str(target_dir),
    ])
    uf.main()
    assert (out_dir / "0001.png").is_file()


def test_main_skips_camera_with_corrupted_frame_file(tmp_path, monkeypatch, capsys):
    # Regression test: a corrupted/unreadable frame crashed the whole batch
    # with a raw PIL.UnidentifiedImageError, because Image.open(frame_path)
    # (used to read media_w/media_h before any calibration handling) had no
    # try/except around it at all -- unlike every other per-camera failure
    # mode in this script.
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    (frames_dir / "0001.jpg").write_bytes(b"not a real image")
    write_frame(frames_dir / "0002.jpg", w=200, h=150)
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    write_calib_pkl(calib_dir / "Camera_0002.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)

    calls = []
    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(uf.subprocess, "run", fake_run)

    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()  # must not crash with an uncaught PIL.UnidentifiedImageError

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "ERROR 0001: could not read" in out
    assert len(calls) == 1  # 0002 still got processed


def test_main_resolves_legacy_calibration_naming_end_to_end(tmp_path, monkeypatch):
    # Proves main() actually calls resolve_calib_path() for BOTH the source
    # calibration and the single-warp target pkl -- not a hardcoded
    # "Camera_<id>.pkl" path construction that happens to match the
    # isolated resolve_calib_path() unit tests by coincidence.
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    with open(calib_dir / "cam1_calibration_data.pkl", "wb") as f:
        pickle.dump(make_calib(image_size=(200, 150)), f)
    patch_undistort_script(monkeypatch, tmp_path)
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    with open(target_dir / "cam1_calibration_data.pkl", "wb") as f:
        pickle.dump(make_calib(fx=100.0, fy=100.0, cx=50.0, cy=40.0, image_size=(100, 80)), f)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--target_pkl_dir", str(target_dir),
    ])
    uf.main()  # should not raise -- both legacy-named files must be found
    assert (out_dir / "0001.jpg").is_file()


def test_main_single_warp_mode_model_default_threads_through(tmp_path, monkeypatch):
    # Proves main() passes the parsed --model value (default OPENCV_FISHEYE)
    # into single_warp_undistort() rather than a hardcoded string that
    # happens to match every other test's explicit value.
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)
    target_dir = tmp_path / "target"
    write_calib_pkl(target_dir / "Camera_0001.pkl", image_size=(100, 80))

    received = {}
    real_fn = uf.single_warp_undistort
    def spy(frame_path, calib, target, model, out_path):
        received["model"] = model
        return real_fn(frame_path, calib, target, model, out_path)
    monkeypatch.setattr(uf, "single_warp_undistort", spy)

    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--target_pkl_dir", str(target_dir),
    ])  # --model omitted -- must default to OPENCV_FISHEYE
    uf.main()
    assert received["model"] == "OPENCV_FISHEYE"


def test_main_wrapper_mode_model_default_in_subprocess_cmd(tmp_path, monkeypatch):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)

    calls = []
    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(uf.subprocess, "run", fake_run)

    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])  # --model omitted
    uf.main()
    cmd = calls[0]
    assert cmd[cmd.index("--model") + 1] == "OPENCV_FISHEYE"


def test_main_skips_camera_with_missing_calibration(tmp_path, monkeypatch, capsys):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg")
    write_frame(frames_dir / "0002.jpg")
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    # 0002 has no calibration file at all
    patch_undistort_script(monkeypatch, tmp_path)
    target_dir = tmp_path / "target"
    write_calib_pkl(target_dir / "Camera_0001.pkl", fx=100.0, fy=100.0, cx=50.0, cy=40.0, image_size=(100, 80))
    write_calib_pkl(target_dir / "Camera_0002.pkl", fx=100.0, fy=100.0, cx=50.0, cy=40.0, image_size=(100, 80))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--target_pkl_dir", str(target_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "SKIP 0002" in out
    assert (out_dir / "0001.jpg").is_file()
    assert not (out_dir / "0002.jpg").exists()


def test_main_skips_camera_when_calibration_validation_raises(tmp_path, monkeypatch, capsys):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    # Non-uniform mismatch: calibration says 200x100, media is 200x150.
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 100))
    patch_undistort_script(monkeypatch, tmp_path)
    target_dir = tmp_path / "target"
    write_calib_pkl(target_dir / "Camera_0001.pkl", image_size=(100, 80))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--target_pkl_dir", str(target_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    assert "ERROR 0001" in capsys.readouterr().out


def test_main_skips_camera_with_non_dict_calibration(tmp_path, monkeypatch, capsys):
    # Regression test: a calibration pkl that unpickles successfully but
    # isn't the expected dict shape at all (e.g. a stray file from a
    # different pipeline version) raised a raw TypeError -- not covered by
    # the (ValueError, OSError, pickle.PickleError, EOFError) fix, since
    # this is a structural problem, not a read/parse problem.
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    with open(calib_dir / "Camera_0001.pkl", "wb") as f:
        pickle.dump([1, 2, 3], f)
    patch_undistort_script(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()  # must not crash with an uncaught TypeError
    assert exc_info.value.code == 1
    assert "ERROR 0001" in capsys.readouterr().out


def test_main_skips_camera_with_calibration_missing_camera_matrix_key(tmp_path, monkeypatch, capsys):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    with open(calib_dir / "Camera_0001.pkl", "wb") as f:
        pickle.dump({"image_size": (200, 150)}, f)  # no "camera_matrix" key
    patch_undistort_script(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()  # must not crash with an uncaught KeyError
    assert exc_info.value.code == 1
    assert "ERROR 0001" in capsys.readouterr().out


def test_main_errors_cleanly_when_out_dir_is_a_file(tmp_path, monkeypatch, capsys):
    # Regression test: --out_dir already existing as a regular file crashed
    # with a raw FileExistsError from mkdir(), before any per-camera
    # processing even started.
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg")
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)
    out_dir.write_text("")  # a plain file at the path --out_dir expects to be a directory
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    assert "exists and is not a directory" in capsys.readouterr().out


def test_main_errors_cleanly_when_out_pkl_dir_is_a_file(tmp_path, monkeypatch, capsys):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg")
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)
    out_pkl_dir.write_text("")
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    assert "exists and is not a directory" in capsys.readouterr().out


def test_main_single_warp_mode_target_missing_camera_matrix_already_handled(tmp_path, monkeypatch, capsys):
    # NOT a bug -- confirms the single-warp branch's existing broad
    # `except Exception` already covers a malformed target pkl gracefully,
    # so no fix was needed here (unlike the source calibration path above).
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    with open(target_dir / "Camera_0001.pkl", "wb") as f:
        pickle.dump({"image_size": (100, 80)}, f)  # no "camera_matrix" key
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--target_pkl_dir", str(target_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()  # should not raise an uncaught exception
    assert exc_info.value.code == 1
    assert "ERROR 0001" in capsys.readouterr().out


def test_main_skips_camera_with_corrupted_calibration_file(tmp_path, monkeypatch, capsys):
    # Regression test: a corrupted/truncated calibration pkl used to crash
    # the whole batch with a raw pickle.UnpicklingError traceback, because
    # the try/except around load_pkl() only caught ValueError (which is
    # only what validate_and_rescale raises, not what a bad file raises).
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    write_frame(frames_dir / "0002.jpg", w=200, h=150)
    (calib_dir / "Camera_0001.pkl").write_bytes(b"not a pickle file, just garbage bytes")
    write_calib_pkl(calib_dir / "Camera_0002.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)

    calls = []
    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(uf.subprocess, "run", fake_run)

    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()  # must not crash with an uncaught pickle.UnpicklingError

    assert exc_info.value.code == 1
    assert "ERROR 0001" in capsys.readouterr().out
    assert len(calls) == 1  # 0002 still got processed


@SKIP_IF_ROOT
def test_main_skips_camera_with_permission_denied_calibration_file(tmp_path, monkeypatch, capsys):
    # Same bug class as above, different cause: PermissionError is an
    # OSError, not a ValueError.
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    calib_path = calib_dir / "Camera_0001.pkl"
    write_calib_pkl(calib_path, image_size=(200, 150))
    calib_path.chmod(0o000)
    patch_undistort_script(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    try:
        with pytest.raises(SystemExit) as exc_info:
            uf.main()
        assert exc_info.value.code == 1
        assert "ERROR 0001" in capsys.readouterr().out
    finally:
        calib_path.chmod(0o644)


def test_main_single_warp_mode_skips_camera_with_missing_target_pkl(tmp_path, monkeypatch, capsys):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg")
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)
    target_dir = tmp_path / "target"
    target_dir.mkdir()  # empty -- no target pkl for 0001
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--target_pkl_dir", str(target_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    assert "SKIP 0001: no target pkl" in capsys.readouterr().out


def test_main_single_warp_mode_writes_zero_distortion_calib_pkl(tmp_path, monkeypatch):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)
    target_dir = tmp_path / "target"
    write_calib_pkl(target_dir / "Camera_0001.pkl", fx=100.0, fy=100.0, cx=50.0, cy=40.0, image_size=(100, 80))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--target_pkl_dir", str(target_dir),
    ])
    uf.main()

    with open(out_pkl_dir / "Camera_0001.pkl", "rb") as f:
        written = pickle.load(f)
    assert written["image_size"] == (100, 80)
    assert np.all(np.asarray(written["distortion_coefficients"]) == 0.0)
    assert written["model"] == "OPENCV"


def test_main_single_warp_mode_catches_any_exception_from_undistort(tmp_path, monkeypatch, capsys):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg")
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)
    target_dir = tmp_path / "target"
    write_calib_pkl(target_dir / "Camera_0001.pkl", image_size=(100, 80))

    def raise_runtime_error(*a, **kw):
        raise RuntimeError("cv2 blew up")
    monkeypatch.setattr(uf, "single_warp_undistort", raise_runtime_error)

    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--target_pkl_dir", str(target_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    assert "ERROR 0001: cv2 blew up" in capsys.readouterr().out


def test_main_wrapper_mode_invokes_subprocess_with_correct_args(tmp_path, monkeypatch):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)

    calls = []
    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(uf.subprocess, "run", fake_run)

    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir), "--model", "OPENCV",
    ])
    uf.main()

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[1] == str(uf.UNDISTORT_SCRIPT)
    assert "-o" in cmd
    out_arg = cmd[cmd.index("-o") + 1]
    assert out_arg == str(out_dir / "0001.jpg")  # full file path with suffix, not a bare directory
    assert cmd[cmd.index("--model") + 1] == "OPENCV"
    assert cmd[cmd.index("-c") + 1] == str(calib_dir / "Camera_0001.pkl")  # not rescaled -- original path used


def test_main_wrapper_mode_uses_rescaled_temp_pkl_when_rescaled(tmp_path, monkeypatch):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    # Calibration made at 400x300 but media is 200x150 -- uniform 0.5 rescale.
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(400, 300))
    patch_undistort_script(monkeypatch, tmp_path)

    calls = []
    read_back = {}
    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        # The temp pkl only exists inside main()'s tempfile.TemporaryDirectory()
        # context, which is torn down before main() returns -- read it now.
        calib_arg = cmd[cmd.index("-c") + 1]
        with open(calib_arg, "rb") as f:
            read_back["calib"] = pickle.load(f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(uf.subprocess, "run", fake_run)

    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    uf.main()

    cmd = calls[0]
    calib_arg = cmd[cmd.index("-c") + 1]
    assert calib_arg != str(calib_dir / "Camera_0001.pkl")  # a temp pkl, not the original
    assert read_back["calib"]["image_size"] == (200, 150)  # rescaled, not the original 400x300


def test_main_wrapper_mode_subprocess_failure_marks_camera_failed(tmp_path, monkeypatch, capsys):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)

    def fake_run(cmd, capture_output=True, text=True):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="offline_undistort.py blew up")
    monkeypatch.setattr(uf.subprocess, "run", fake_run)

    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    assert "offline_undistort.py blew up" in capsys.readouterr().out


def test_main_no_rescale_flag_threads_through_to_validation(tmp_path, monkeypatch, capsys):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(400, 300))  # uniform mismatch
    patch_undistort_script(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir), "--no_rescale",
    ])
    with pytest.raises(SystemExit) as exc_info:
        uf.main()
    assert exc_info.value.code == 1
    assert "no_rescale" in capsys.readouterr().out


def test_main_all_cameras_succeed_exits_cleanly(tmp_path, monkeypatch):
    frames_dir, calib_dir, out_dir, out_pkl_dir = base_dirs(tmp_path)
    write_frame(frames_dir / "0001.jpg", w=200, h=150)
    write_calib_pkl(calib_dir / "Camera_0001.pkl", image_size=(200, 150))
    patch_undistort_script(monkeypatch, tmp_path)
    target_dir = tmp_path / "target"
    write_calib_pkl(target_dir / "Camera_0001.pkl", fx=100.0, fy=100.0, cx=50.0, cy=40.0, image_size=(100, 80))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--frames_dir", str(frames_dir), "--calib_dir", str(calib_dir),
        "--out_dir", str(out_dir), "--out_pkl_dir", str(out_pkl_dir),
        "--target_pkl_dir", str(target_dir),
    ])
    uf.main()  # should not raise
