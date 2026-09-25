# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""The CPU half of the render step: mattes, composites, and transforms.

This logic lives outside Blender precisely so it can be tested on small
synthetic images. What matters is that the matte is the render's own alpha
untouched, that the composite is the ordinary over operator (Blender writes
straight alpha, measured), and that the transforms it emits are readable by
build_flipbook_4dgs_dataset.py without an adapter.
"""

import json

import numpy as np
import pytest
from PIL import Image

import render_blender_rig as rbr
from build_flipbook_4dgs_dataset import load_flipbook_rig


def write_rgba(path, rgb, alpha, size=(8, 6)):
    w, h = size
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = rgb
    arr[..., 3] = alpha
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGBA").save(path)
    return arr


def write_rgb(path, rgb, size=(8, 6)):
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[...] = rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)
    return arr


# ------------------------------------------------------------ matte split
def test_split_writes_rgb_and_the_alpha_matte_unchanged(tmp_path):
    src = tmp_path / "subject.png"
    write_rgba(src, (200, 100, 50), 128)
    image = tmp_path / "images_flat" / "00.png"
    mask = tmp_path / "fmasks_clean" / "00.png"
    rbr.split_rgba(src, image, mask)

    with Image.open(image) as im:
        assert im.mode == "RGB"
        rgb = np.asarray(im)
    with Image.open(mask) as im:
        assert im.mode == "L"
        matte = np.asarray(im)
    assert np.all(rgb == np.array([200, 100, 50]))
    assert np.all(matte == 128)


def test_split_preserves_a_soft_matte_rather_than_thresholding_it(tmp_path):
    """Hair and fabric edges live in the fractional values. Rounding them
    to a binary mask is exactly the silhouette damage the ground-truth path
    exists to avoid."""
    src = tmp_path / "subject.png"
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[..., :3] = 255
    arr[..., 3] = np.array([[0, 40, 90, 255]] * 4, dtype=np.uint8)
    Image.fromarray(arr, mode="RGBA").save(src)

    rbr.split_rgba(src, tmp_path / "i.png", tmp_path / "m.png")
    with Image.open(tmp_path / "m.png") as im:
        matte = np.asarray(im)
    assert sorted(np.unique(matte).tolist()) == [0, 40, 90, 255]


# ------------------------------------------------------------- composite
@pytest.mark.parametrize("alpha,expected", [
    (0, 40),        # fully transparent: the plate survives
    (255, 200),     # fully opaque: the subject wins
    (128, 120),     # half: straight-alpha blend of 200 over 40
])
def test_composite_is_the_over_operator_on_straight_alpha(tmp_path, alpha, expected):
    subject = tmp_path / "s.png"
    plate = tmp_path / "p.png"
    write_rgba(subject, (200, 200, 200), alpha)
    write_rgb(plate, (40, 40, 40))
    out = tmp_path / "c.png"
    rbr.composite_over(subject, plate, out)
    with Image.open(out) as im:
        got = np.asarray(im)
    assert abs(int(got[0, 0, 0]) - expected) <= 1


def test_composite_rejects_a_plate_of_the_wrong_size(tmp_path):
    subject = tmp_path / "s.png"
    plate = tmp_path / "p.png"
    write_rgba(subject, (10, 10, 10), 255, size=(8, 6))
    write_rgb(plate, (20, 20, 20), size=(4, 3))
    with pytest.raises(SystemExit, match="plate"):
        rbr.composite_over(subject, plate, tmp_path / "c.png")


# ------------------------------------------------------------ transforms
def camera_entry(label, position):
    c2w = np.eye(4)
    c2w[:3, 3] = position
    return {"label": label, "name": f"Camera_{label}", "role": "train",
            "c2w_opengl": c2w.tolist(), "c2w_blender": c2w.tolist(),
            "position": list(position)}


def test_frame_transforms_are_readable_by_the_dataset_builder(tmp_path):
    """The contract that lets a rendered rig join the pipeline where a real
    capture does: no adapter, no rewrite."""
    cameras = [camera_entry("00", (1.0, 0.0, 0.0)),
               camera_entry("01", (0.0, 1.0, 0.0))]
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    intrinsics = {"00": (K, (640, 480)), "01": (K, (640, 480))}
    payload = rbr.frame_transforms(cameras, intrinsics)

    for name in ("frame_0000", "frame_0001"):
        frame_dir = tmp_path / name
        frame_dir.mkdir()
        (frame_dir / "transforms.json").write_text(json.dumps(payload))
    rig = load_flipbook_rig([tmp_path / "frame_0000", tmp_path / "frame_0001"])

    assert sorted(rig) == ["00", "01"]
    assert rig["00"]["intr"] == (500.0, 500.0, 320.0, 240.0)
    assert rig["00"]["w"] == 640 and rig["00"]["h"] == 480


def test_frame_transforms_declare_zero_distortion(tmp_path):
    """The builder refuses a flipbook that still carries distortion, so the
    renderer has to state plainly that its output is undistorted."""
    payload = rbr.frame_transforms(
        [camera_entry("00", (1.0, 0.0, 0.0))],
        {"00": (np.eye(3), (10, 10))})
    entry = payload["frames"][0]
    assert entry["k1"] == entry["k2"] == entry["p1"] == entry["p2"] == 0.0
    assert entry["file_path"] == "images_flat/00.png"


# ------------------------------------------------------------ calibration
def test_calibration_pkls_are_written_with_the_readers_own_numpy(tmp_path):
    """Blender bundles a different numpy, so pickles are produced here and
    not there. A pkl the pipeline cannot unpickle fails deep inside HLOC."""
    import pickle

    cameras = [{
        "label": "00", "role": "train",
        "intrinsics": {"camera_matrix": [[500.0, 0, 320.0], [0, 500.0, 240.0],
                                         [0, 0, 1.0]],
                       "distortion_coefficients": [0.1, 0.2, 0.3, 0.4],
                       "render_size": [640, 480], "model": "OPENCV_FISHEYE"},
    }]
    written = rbr.write_calibration_pkls(cameras, tmp_path / "calib")
    assert set(written) == {"00"}

    with open(tmp_path / "calib" / "Camera_00.pkl", "rb") as f:
        loaded = pickle.load(f)
    assert loaded["model"] == "OPENCV_FISHEYE"
    assert loaded["image_size"] == (640, 480)
    assert np.allclose(loaded["distortion_coefficients"], [0.1, 0.2, 0.3, 0.4])
    # offline_undistort.py reads these keys by name.
    for key in ("camera_matrix", "distortion_coefficients", "image_size", "model"):
        assert key in loaded


def test_undistorted_target_removes_distortion_and_keeps_the_size():
    calib = {"camera_matrix": np.array([[900.0, 0, 960.0], [0, 900.0, 540.0],
                                        [0, 0, 1.0]]),
             "distortion_coefficients": np.array([0.03, 0.06, -0.04, 0.008]),
             "image_size": (1920, 1080), "model": "OPENCV_FISHEYE"}
    target = rbr.undistorted_target(calib, balance=0.0)
    assert target["model"] == "OPENCV"
    assert target["image_size"] == (1920, 1080)
    assert np.allclose(target["distortion_coefficients"], 0.0)
    assert target["camera_matrix"][0, 0] > 0


def test_forwarded_render_args_are_rebuilt_from_the_namespace(tmp_path):
    """Rebuilt rather than filtered out of sys.argv, so a value that
    happens to equal a flag name cannot corrupt the command line."""
    from types import SimpleNamespace

    args = SimpleNamespace(
        rig_spec="rig.json", frame_start=100, frame_count=48, frame_step=1,
        samples=64, engine="CYCLES", device="OPTIX", manifest=None,
        action="--samples", no_denoise=True, view_transform=None,
        background_plates=True, skip_existing=False, index_offset=61)
    out = rbr.forward_args(args, tmp_path / "render")
    assert out[out.index("--action") + 1] == "--samples"
    assert "--no_denoise" in out and "--background_plates" in out
    assert "--skip_existing" not in out
    # Concurrent shards each need their own base index or they overwrite
    # each other's frame_NNNN directories.
    assert out[out.index("--index_offset") + 1] == "61" 


# ------------------------------------------------------- shard coverage
def test_post_step_refuses_to_silently_skip_rendered_frames(tmp_path):
    """Concurrent instances each write rig_resolved.json, so the last to
    start leaves one describing only its own shard. Post-processing the
    subset and reporting success is worse than failing: it produced a
    61-frame flipbook from a 121-frame render once."""
    render = tmp_path / "render"
    for i in range(6):
        (render / "subject" / f"frame_{i:04d}").mkdir(parents=True)
    resolved = {"frames": [{"index": i, "scene_frame": 100 + i} for i in range(3)]}
    with pytest.raises(SystemExit, match="silently skipped"):
        rbr.check_frame_coverage(render, resolved)


def test_post_step_accepts_a_complete_render(tmp_path):
    render = tmp_path / "render"
    for i in range(4):
        (render / "subject" / f"frame_{i:04d}").mkdir(parents=True)
    resolved = {"frames": [{"index": i, "scene_frame": 100 + i} for i in range(4)]}
    rbr.check_frame_coverage(render, resolved)          # must not raise


def test_coverage_check_tolerates_metadata_listing_extra_frames(tmp_path):
    """--rig_only writes the full range before anything is rendered, which
    is a legitimate intermediate state and not an error here."""
    render = tmp_path / "render"
    (render / "subject" / "frame_0000").mkdir(parents=True)
    resolved = {"frames": [{"index": i, "scene_frame": 100 + i} for i in range(9)]}
    rbr.check_frame_coverage(render, resolved)


# ---------------------------------------------------------- spec calibration
def test_spec_calibration_resolves_a_repo_relative_pkl(tmp_path):
    """The shipped fisheye spec names its pkl relative to the repo root, and
    the driver hands Blender that calibration as JSON."""
    spec = rbr.SCRIPT_DIR.parent / "configs" / "rigs" / "ring16_gopro_fisheye.json"
    out = rbr.resolve_spec_calibration(spec, tmp_path / "render")
    payload = json.loads(out.read_text())
    assert payload["model"] == "OPENCV_FISHEYE"
    assert payload["image_size"] == [3840, 3360]
    assert len(payload["distortion_coefficients"]) == 4


def test_spec_calibration_is_skipped_for_a_pinhole_spec(tmp_path):
    spec = tmp_path / "rig.json"
    spec.write_text(json.dumps({"intrinsics": {"lens_mm": 35}}))
    assert rbr.resolve_spec_calibration(spec, tmp_path / "render") is None


def test_a_missing_spec_calibration_is_one_clear_error(tmp_path):
    spec = tmp_path / "rig.json"
    spec.write_text(json.dumps({"calibration_pkl": "nope.pkl"}))
    with pytest.raises(SystemExit, match="nope.pkl"):
        rbr.resolve_spec_calibration(spec, tmp_path / "render")
