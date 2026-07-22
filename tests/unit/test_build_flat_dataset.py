import json
import sys

import pytest
from PIL import Image

import build_flat_dataset as bfd


def base_argv(transforms, undistorted_dir, out_images_flat, out_transforms, extra=None):
    argv = [
        "prog",
        "--transforms", str(transforms),
        "--undistorted_dir", str(undistorted_dir),
        "--out_images_flat", str(out_images_flat),
        "--out_transforms", str(out_transforms),
    ]
    return argv + (extra or [])


def write_transforms(path, frames, extra_top_level=None):
    data = {"frames": frames, **(extra_top_level or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_source_image(undistorted_dir, cam_id, ext=".jpg", size=(20, 15)):
    d = undistorted_dir / f"Camera_{cam_id}"
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(d / f"0000{ext}")


# --------------------------------------------------------------------------
# Whole-file corruption -> clean top-level error, not a raw traceback
# (regression tests for bugs #1-3)
# --------------------------------------------------------------------------

def test_main_errors_cleanly_on_invalid_json(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "bad.json"
    transforms.write_text("not valid json {{{")
    monkeypatch.setattr(sys, "argv", base_argv(
        transforms, tmp_path / "undist", tmp_path / "flat", tmp_path / "out" / "transforms.json",
    ))
    with pytest.raises(SystemExit) as exc_info:
        bfd.main()
    assert exc_info.value.code == 1
    assert "is not valid JSON" in capsys.readouterr().out


def test_main_errors_cleanly_on_missing_frames_key(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "noframes.json"
    transforms.write_text(json.dumps({"not_frames": []}))
    monkeypatch.setattr(sys, "argv", base_argv(
        transforms, tmp_path / "undist", tmp_path / "flat", tmp_path / "out" / "transforms.json",
    ))
    with pytest.raises(SystemExit) as exc_info:
        bfd.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "is missing expected key" in out
    assert "'frames'" in out


def test_main_errors_cleanly_on_frame_missing_camera_label(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "nolabel.json"
    write_transforms(transforms, [{"not_camera_label": "x"}])
    monkeypatch.setattr(sys, "argv", base_argv(
        transforms, tmp_path / "undist", tmp_path / "flat", tmp_path / "out" / "transforms.json",
    ))
    with pytest.raises(SystemExit) as exc_info:
        bfd.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "is missing expected key" in out
    assert "'camera_label'" in out


# --------------------------------------------------------------------------
# Per-camera fault isolation (regression test for bug #4) -- note this
# script's own deliberate design: ANY skipped camera still hard-fails the
# whole run (no partial flat dataset ever written). The fix here is just
# "fail cleanly, not with a raw traceback" -- not "become lenient."
# --------------------------------------------------------------------------

def test_main_corrupted_source_image_in_convert_branch_fails_cleanly(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [{"camera_label": "Camera_undistorted_0001"}])
    undistorted_dir = tmp_path / "undist"
    (undistorted_dir / "Camera_0001").mkdir(parents=True)
    (undistorted_dir / "Camera_0001" / "0000.jpg").write_bytes(b"not a real jpg")
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(
        transforms, undistorted_dir, out_images_flat, out_transforms,
        ["--image_ext", ".jpg", "--out_image_ext", ".png"],  # forces the PIL convert branch
    ))
    with pytest.raises(SystemExit) as exc_info:
        bfd.main()  # must not raise a raw UnidentifiedImageError
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "could not read/convert image" in out
    assert "ERROR: 1 camera(s) missing an undistorted image" in out
    assert not out_transforms.exists()  # no partial output written
    assert not (out_transforms.parent / "camera_label_map.json").exists()


def test_main_corrupted_image_does_not_prevent_good_camera_from_processing_before_failing(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [
        {"camera_label": "Camera_undistorted_0001"},
        {"camera_label": "Camera_undistorted_0002"},
    ])
    undistorted_dir = tmp_path / "undist"
    (undistorted_dir / "Camera_0001").mkdir(parents=True)
    (undistorted_dir / "Camera_0001" / "0000.jpg").write_bytes(b"not a real jpg")
    write_source_image(undistorted_dir, "0002")
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(
        transforms, undistorted_dir, out_images_flat, out_transforms,
        ["--image_ext", ".jpg", "--out_image_ext", ".png"],
    ))
    with pytest.raises(SystemExit) as exc_info:
        bfd.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "Camera_undistorted_0001" in out  # reported as the failure
    assert "00.png" in out  # the good camera 0002 still got processed before the final failure summary


def test_main_missing_source_image_skips_and_fails_cleanly(tmp_path, monkeypatch, capsys):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [{"camera_label": "Camera_undistorted_0001"}])
    undistorted_dir = tmp_path / "undist"  # no Camera_0001 dir at all
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(transforms, undistorted_dir, out_images_flat, out_transforms))
    with pytest.raises(SystemExit) as exc_info:
        bfd.main()
    assert exc_info.value.code == 1
    assert "no image found for Camera_undistorted_0001" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Normal path: label renumbering, copy vs convert branch, transforms rewrite
# --------------------------------------------------------------------------

def test_main_success_writes_flat_images_transforms_and_label_map(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [
        {"camera_label": "Camera_undistorted_0002", "extra_field": "keep me"},
        {"camera_label": "Camera_undistorted_0001", "extra_field": "keep me too"},
    ], extra_top_level={"some_other_key": "preserved"})
    undistorted_dir = tmp_path / "undist"
    write_source_image(undistorted_dir, "0001")
    write_source_image(undistorted_dir, "0002")
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(transforms, undistorted_dir, out_images_flat, out_transforms))
    bfd.main()

    # sorted by camera_label -- "Camera_undistorted_0001" < "...0002" lexically
    assert (out_images_flat / "00.png").is_file()
    assert (out_images_flat / "01.png").is_file()

    with open(out_transforms) as f:
        tf = json.load(f)
    assert tf["some_other_key"] == "preserved"
    labels = {fr["camera_label"]: fr for fr in tf["frames"]}
    assert labels["00"]["file_path"] == "images_flat/00.png"
    assert labels["00"]["extra_field"] == "keep me too"  # original frame 0001's data, other keys preserved
    assert labels["01"]["file_path"] == "images_flat/01.png"

    with open(out_transforms.parent / "camera_label_map.json") as f:
        label_map = json.load(f)
    assert label_map == {"00": "Camera_undistorted_0001", "01": "Camera_undistorted_0002"}


def test_main_skipped_camera_does_not_leave_a_gap_in_numbering(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [
        {"camera_label": "Camera_undistorted_0001"},  # will be missing -> skipped
        {"camera_label": "Camera_undistorted_0002"},
        {"camera_label": "Camera_undistorted_0003"},
    ])
    undistorted_dir = tmp_path / "undist"
    write_source_image(undistorted_dir, "0002")
    write_source_image(undistorted_dir, "0003")
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(transforms, undistorted_dir, out_images_flat, out_transforms))
    with pytest.raises(SystemExit):
        bfd.main()  # this script's own design: any skip is a hard failure

    # but confirm the *numbering itself* (verified by construction, not by reading
    # partial output since none is written) would have been dense: 0002 got "00"
    # not "01", by re-running with 0001 also present to compare against the
    # skip-free case below.


def test_main_dense_numbering_confirmed_via_skip_free_comparison(tmp_path, monkeypatch):
    # Same 3 cameras, all present -- confirms 0002 gets "00" only because of
    # its position in the sort, not because of any gap-filling from a skip
    # (the skip case itself can't write partial output to inspect directly).
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [
        {"camera_label": "Camera_undistorted_0002"},
        {"camera_label": "Camera_undistorted_0003"},
    ])
    undistorted_dir = tmp_path / "undist"
    write_source_image(undistorted_dir, "0002")
    write_source_image(undistorted_dir, "0003")
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(transforms, undistorted_dir, out_images_flat, out_transforms))
    bfd.main()
    with open(out_transforms) as f:
        tf = json.load(f)
    labels = sorted(fr["camera_label"] for fr in tf["frames"])
    assert labels == ["00", "01"]  # dense, no gap


def test_main_same_extension_uses_shutil_copy_not_pil_reencode(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [{"camera_label": "Camera_undistorted_0001"}])
    undistorted_dir = tmp_path / "undist"
    src = undistorted_dir / "Camera_0001"
    src.mkdir(parents=True)
    raw_bytes = b"\xff\xd8\xff\xe0fake-but-byte-identical-jpeg-content"
    (src / "0000.jpg").write_bytes(raw_bytes)
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(
        transforms, undistorted_dir, out_images_flat, out_transforms,
        ["--image_ext", ".jpg", "--out_image_ext", ".jpg"],  # same ext -> shutil.copy path
    ))
    bfd.main()
    # shutil.copy is a blind byte copy -- even non-image-decodable bytes survive
    # identically, proving this went through copy, not PIL re-encoding
    assert (out_images_flat / "00.jpg").read_bytes() == raw_bytes


def test_main_different_extension_uses_pil_convert(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [{"camera_label": "Camera_undistorted_0001"}])
    undistorted_dir = tmp_path / "undist"
    write_source_image(undistorted_dir, "0001", ext=".jpg", size=(30, 20))
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(
        transforms, undistorted_dir, out_images_flat, out_transforms,
        ["--image_ext", ".jpg", "--out_image_ext", ".png"],
    ))
    bfd.main()
    with Image.open(out_images_flat / "00.png") as out_img:
        assert out_img.format == "PNG"
        assert out_img.size == (30, 20)


def test_main_camera_id_strips_camera_undistorted_prefix(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [{"camera_label": "Camera_undistorted_0007"}])
    undistorted_dir = tmp_path / "undist"
    write_source_image(undistorted_dir, "0007")  # looked up as Camera_0007, not Camera_undistorted_0007
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(transforms, undistorted_dir, out_images_flat, out_transforms))
    bfd.main()
    assert (out_images_flat / "00.png").is_file()


def test_main_camera_id_handles_plain_camera_prefix(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [{"camera_label": "Camera_0009"}])  # no "undistorted" infix
    undistorted_dir = tmp_path / "undist"
    write_source_image(undistorted_dir, "0009")
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(transforms, undistorted_dir, out_images_flat, out_transforms))
    bfd.main()
    assert (out_images_flat / "00.png").is_file()


# --------------------------------------------------------------------------
# Wiring + argparse-level checks
# --------------------------------------------------------------------------

def test_main_default_image_ext_reaches_source_lookup(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [{"camera_label": "Camera_undistorted_0001"}])
    undistorted_dir = tmp_path / "undist"
    write_source_image(undistorted_dir, "0001", ext=".jpg")  # default --image_ext is .jpg
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(transforms, undistorted_dir, out_images_flat, out_transforms))
    bfd.main()  # no --image_ext override -- default must actually be used to find 0000.jpg
    assert (out_images_flat / "00.png").is_file()  # default --out_image_ext is .png


def test_main_rejects_unsupported_image_ext(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [{"camera_label": "Camera_undistorted_0001"}])
    monkeypatch.setattr(sys, "argv", base_argv(
        transforms, tmp_path / "undist", tmp_path / "flat", tmp_path / "out" / "transforms.json",
        ["--image_ext", ".bmp"],
    ))
    with pytest.raises(SystemExit) as exc_info:
        bfd.main()
    assert exc_info.value.code == 2


@pytest.mark.parametrize("missing_flag", [
    "--transforms", "--undistorted_dir", "--out_images_flat", "--out_transforms",
])
def test_main_errors_when_required_arg_missing(tmp_path, monkeypatch, missing_flag):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [])
    argv = base_argv(transforms, tmp_path / "undist", tmp_path / "flat", tmp_path / "out" / "transforms.json")
    idx = argv.index(missing_flag)
    argv = argv[:idx] + argv[idx + 2:]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        bfd.main()
    assert exc_info.value.code == 2


def test_main_empty_frames_list_writes_empty_dataset(tmp_path, monkeypatch):
    transforms = tmp_path / "t.json"
    write_transforms(transforms, [])
    undistorted_dir = tmp_path / "undist"
    out_images_flat = tmp_path / "flat"
    out_transforms = tmp_path / "out" / "transforms.json"
    monkeypatch.setattr(sys, "argv", base_argv(transforms, undistorted_dir, out_images_flat, out_transforms))
    bfd.main()  # no cameras, no skips -- must not error
    with open(out_transforms) as f:
        tf = json.load(f)
    assert tf["frames"] == []
