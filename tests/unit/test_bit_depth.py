# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""16-bit frames as an option. Default stays 8-bit everywhere; 16 exists for PNG only, before the colour step
(extract, undistort) and as the colour output; nothing after that reads more than 8 bits."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")

import build_flat_dataset as bfd
import color_adjust as ca
import extract_synced_frames as esf
import frame_motion
import image_formats as fmt
import undistort_frames as uf


def wide(h=24, w=32, seed=0):
    """uint16 RGB whose low bytes are not zero, so 8-bit handling is visible."""
    rng = np.random.default_rng(seed)
    return rng.integers(256, 65280, (h, w, 3), dtype=np.uint16) | 0x0033


def write16(path, rgb):
    assert cv2.imwrite(str(path), np.ascontiguousarray(rgb[..., ::-1]))
    assert fmt.png_bit_depth(path) == 16
    return path


def read_any(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR + cv2.IMREAD_ANYDEPTH)
    return np.ascontiguousarray(bgr[..., ::-1])


# ------------------------------------------------------------ helpers
def test_png_bit_depth_reads_the_header(tmp_path):
    write16(tmp_path / "a.png", wide())
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / "b.png")
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / "c.webp", lossless=True)
    (tmp_path / "d.png").write_bytes(b"not a png")
    assert [fmt.png_bit_depth(tmp_path / n) for n in ("a.png", "b.png", "c.webp", "d.png", "missing.png")] == [16, 8, 8, 8, 8]


def test_16_bit_is_for_png_only():
    assert fmt.check_bit_depth(".png", 16) == 16
    assert fmt.check_bit_depth(".webp", 8) == 8 and fmt.check_bit_depth(".jpg", None) is None
    for ext in (".webp", ".jpg", ".jpeg"):
        with pytest.raises(fmt.ImageFormatError, match="png only"):
            fmt.check_bit_depth(ext, 16)
    with pytest.raises(fmt.ImageFormatError):
        fmt.check_bit_depth(".png", 12)


# ------------------------------------------------------------ extract_synced_frames
def test_ffmpeg_format_args_bit_depth():
    assert esf.ffmpeg_format_args(".png") == ["-pix_fmt", "rgb24", "-compression_level", "1"]
    assert esf.ffmpeg_format_args(".png", 16) == ["-pix_fmt", "rgb48be", "-compression_level", "1"]
    assert esf.ffmpeg_format_args(".webp", 8) == ["-pix_fmt", "rgb24"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_extract_keeps_16_bit_only_when_asked(tmp_path):
    clip = tmp_path / "0001.mkv"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=64x48:rate=10:duration=1,noise=alls=25:allf=t",
                    "-pix_fmt", "yuv420p10le", "-c:v", "ffv1", str(clip)], check=True)
    (default,) = esf.extract_frames(clip, 0.0, 1, tmp_path / "a", ".png")
    (deep,) = esf.extract_frames(clip, 0.0, 1, tmp_path / "b", ".png", bit_depth=16)
    assert fmt.png_bit_depth(default) == 8 and fmt.png_bit_depth(deep) == 16
    low_bytes = read_any(deep) & 0xFF
    assert len(np.unique(low_bytes)) > 16                 # real extra precision, not 8-bit scaled up


def fake_movies(tmp_path):
    movies = tmp_path / "movies"
    movies.mkdir()
    (movies / "0001.mp4").write_bytes(b"")
    offsets = tmp_path / "o.json"
    offsets.write_text(json.dumps({"reference_camera": "0001.mp4",
                                   "offsets": {"0001.mp4": {"fps": 30.0, "frame_offset": 3}}}))
    return movies, offsets


@pytest.mark.parametrize("extra", [[], ["--output_ext", ".webp"], ["--output_ext", ".jpg"]])
def test_extract_rejects_16_bit_for_anything_but_png(tmp_path, monkeypatch, capsys, extra):
    movies, offsets = fake_movies(tmp_path)
    monkeypatch.setattr(sys, "argv", ["p", str(movies), str(offsets), str(tmp_path / "out"), "--bit_depth", "16", *extra])
    with pytest.raises(SystemExit) as e:
        esf.main()
    assert e.value.code == 1 and "--bit_depth" in capsys.readouterr().out


def test_extract_rejects_16_bit_with_a_pp3_profile(tmp_path, monkeypatch, capsys):
    movies, offsets = fake_movies(tmp_path)
    (tmp_path / "pp3").mkdir()
    monkeypatch.setattr(esf, "resolve_rawtherapee_cmd", lambda override=None: ["rawtherapee-cli"])
    monkeypatch.setattr(sys, "argv", ["p", str(movies), str(offsets), str(tmp_path / "out"), "--pp3_dir",
                                      str(tmp_path / "pp3"), "--output_ext", ".png", "--bit_depth", "16"])
    with pytest.raises(SystemExit) as e:
        esf.main()
    assert e.value.code == 1 and "pp3" in capsys.readouterr().out


def test_extract_passes_the_depth_only_when_16(tmp_path, monkeypatch):
    movies, offsets = fake_movies(tmp_path)
    seen = []

    def fake_extract(video_path, seek, count, out_dir, ext, **kw):
        seen.append(kw)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{video_path.stem}_01{ext}"
        p.write_bytes(b"")
        return [p]

    monkeypatch.setattr(esf, "extract_frames", fake_extract)
    for extra, expected in (([], {}), (["--output_ext", ".png", "--bit_depth", "16"], {"bit_depth": 16})):
        seen.clear()
        monkeypatch.setattr(sys, "argv", ["p", str(movies), str(offsets), str(tmp_path / "out"), *extra])
        esf.main()
        assert seen == [expected]


# ------------------------------------------------------------ undistort_frames
def identity_calibs():
    K = [[100.0, 0.0, 16.0], [0.0, 100.0, 12.0], [0.0, 0.0, 1.0]]
    calib = {"camera_matrix": K, "distortion_coefficients": [0.0, 0.0, 0.0, 0.0], "image_size": (32, 24)}
    return calib, dict(calib)


def test_undistort_keeps_16_bit_through_the_remap(tmp_path):
    frame = write16(tmp_path / "f.png", wide())
    calib, target = identity_calibs()
    uf.single_warp_undistort(frame, calib, target, "OPENCV", tmp_path / "out16.png", bit_depth=16)
    out = read_any(tmp_path / "out16.png")
    assert fmt.png_bit_depth(tmp_path / "out16.png") == 16
    assert np.abs(out.astype(int) - wide().astype(int)).max() <= 1
    assert (out & 0xFF).any()                                   # values the 8-bit path cannot represent
    uf.single_warp_undistort(frame, calib, target, "OPENCV", tmp_path / "out8.png")
    assert fmt.png_bit_depth(tmp_path / "out8.png") == 8        # the default is unchanged


def test_undistort_reads_a_16_bit_file_as_uint16_only_when_asked(tmp_path):
    frame = write16(tmp_path / "f.png", wide())
    assert uf.read_bgr(frame).dtype == np.uint8
    assert uf.read_bgr(frame, 8).dtype == np.uint8
    assert uf.read_bgr(frame, 16).dtype == np.uint16
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / "e.png")
    assert uf.read_bgr(tmp_path / "e.png", 16).dtype == np.uint8   # an 8-bit file stays 8-bit


def test_undistort_writes_a_uint16_array_as_a_16_bit_png(tmp_path):
    assert uf.write_image(tmp_path / "w.png", wide())
    assert fmt.png_bit_depth(tmp_path / "w.png") == 16


def test_undistort_rejects_16_bit_for_non_png_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["u", "--frames_dir", str(tmp_path), "--calib_dir", str(tmp_path),
                                      "--out_dir", str(tmp_path / "o"), "--out_pkl_dir", str(tmp_path / "p"),
                                      "--out_image_ext", ".webp", "--bit_depth", "16"])
    with pytest.raises(SystemExit):
        uf.main()
    assert "--bit_depth" in capsys.readouterr().out


# ------------------------------------------------------------ color_adjust
def run_adjust(tmp_path, monkeypatch, settings, *extra, src=None, name="undistorted_0001.png"):
    src_dir, out = tmp_path / "in", tmp_path / "out"
    src_dir.mkdir(exist_ok=True)
    if src is not None:
        write16(src_dir / name, src) if src.dtype == np.uint16 else Image.fromarray(src).save(src_dir / name)
    (tmp_path / "s.json").write_text(json.dumps(settings))
    monkeypatch.setattr(sys, "argv", ["c", "--settings", str(tmp_path / "s.json"), "--in_dir", str(src_dir),
                                      "--out_dir", str(out), "--workers", "1", *extra])
    ca.main()
    return out / name


def test_16_in_8_out_rounds_it_does_not_truncate(tmp_path, monkeypatch):
    src = np.full((4, 4, 3), 511, np.uint16)        # 511 / 257 = 1.988: rounds to 2, truncates to 1
    out = run_adjust(tmp_path, monkeypatch, {"method": "none"}, "--out_bit_depth", "8", src=src)
    assert fmt.png_bit_depth(out) == 8
    assert (np.asarray(Image.open(out)) == 2).all()


def test_16_in_is_kept_at_16_by_default_and_none_is_exact(tmp_path, monkeypatch):
    src = wide()
    out = run_adjust(tmp_path, monkeypatch, {"method": "none"}, src=src)
    assert fmt.png_bit_depth(out) == 16 and (read_any(out) == src).all()


def test_16_in_16_out_keeps_precision_through_an_adjustment(tmp_path, monkeypatch):
    src = wide()
    settings = {"method": "adjust", "adjust": {"exposure": 0.5, "contrast": 0.2}}
    out = run_adjust(tmp_path, monkeypatch, settings, src=src)
    expected = np.rint(ca.adjust(src.astype(np.float64) / 65535.0, exposure=0.5, contrast=0.2) * 65535.0)
    got = read_any(out)
    assert fmt.png_bit_depth(out) == 16 and np.abs(got.astype(float) - expected).max() <= 1
    assert len(np.unique(got & 0xFF)) > 64                       # sub-8-bit gradations survived
    eight = np.rint(expected / 257.0)
    assert not (np.rint(got / 257.0) != eight).mean() > 0.01    # and it agrees with the 8-bit result to a level


def test_8_in_16_out_is_the_8_bit_value_at_16_bits(tmp_path, monkeypatch):
    src = np.random.default_rng(3).integers(0, 256, (6, 6, 3), dtype=np.uint8)
    out = run_adjust(tmp_path, monkeypatch, {"method": "none"}, "--out_bit_depth", "16", src=src)
    assert fmt.png_bit_depth(out) == 16 and (read_any(out) == src.astype(np.uint16) * 257).all()


def test_8_in_8_out_is_unchanged(tmp_path, monkeypatch):
    src = np.random.default_rng(4).integers(0, 256, (6, 6, 3), dtype=np.uint8)
    out = run_adjust(tmp_path, monkeypatch, {"method": "adjust", "adjust": {"exposure": 1.0}}, src=src)
    expected = ca.adjust_uint8(src, exposure=1.0)
    assert fmt.png_bit_depth(out) == 8 and (np.asarray(Image.open(out)) == expected).all()


def test_out_bit_depth_16_is_an_error_for_webp(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_adjust(tmp_path, monkeypatch, {"method": "none"}, "--image_ext", ".webp", "--out_bit_depth", "16",
                   src=np.zeros((4, 4, 3), np.uint8), name="undistorted_0001.webp")


def test_changing_out_bit_depth_invalidates_the_corrected_copy(tmp_path, monkeypatch):
    src = wide()
    manifest = ["--manifest", str(tmp_path / "m.json")]
    out = run_adjust(tmp_path, monkeypatch, {"method": "none"}, *manifest, src=src)
    assert fmt.png_bit_depth(out) == 16
    ca.main()                                                     # same arguments: carried on, not redone
    assert fmt.png_bit_depth(out) == 16
    run_adjust(tmp_path, monkeypatch, {"method": "none"}, *manifest, "--out_bit_depth", "8")
    assert fmt.png_bit_depth(out) == 8


def test_a_pp3_profile_refuses_16_bit(tmp_path):
    src = write16(tmp_path / "undistorted_0001.png", wide())
    with pytest.raises(SystemExit, match="16-bit"):
        ca.correct_file(src, tmp_path / "o.png", {"method": "pp3", "pp3_dir": str(tmp_path)}, {})


# ------------------------------------------------------------ readers take a 16-bit PNG and give 8 bits
def test_build_flat_dataset_turns_a_16_bit_png_into_an_8_bit_one(tmp_path, monkeypatch):
    und = tmp_path / "und"
    src = wide(seed=5)
    (und / "Camera_0001").mkdir(parents=True)
    write16(und / "Camera_0001" / "0000.png", src)
    tf = tmp_path / "t.json"
    tf.write_text(json.dumps({"frames": [{"camera_label": "Camera_undistorted_0001"}]}))
    out = tmp_path / "flat"
    monkeypatch.setattr(sys, "argv", ["p", "--transforms", str(tf), "--undistorted_dir", str(und), "--image_ext", ".png",
                                      "--out_images_flat", str(out), "--out_transforms", str(tmp_path / "o.json")])
    bfd.main()
    assert fmt.png_bit_depth(out / "00.png") == 8
    got = np.asarray(Image.open(out / "00.png"))
    assert np.abs(got.astype(int) - np.rint(src / 257.0).astype(int)).max() <= 0   # rounded, not truncated


def test_the_other_readers_take_a_16_bit_png(tmp_path):
    import build_4dgs_dataset as b4

    img = write16(tmp_path / "i.png", wide())
    mask = tmp_path / "m.png"
    Image.fromarray(np.full((24, 32), 255, np.uint8)).save(mask)
    assert frame_motion.load_gray(img, 1.0).dtype == np.uint8                 # cv2 grayscale read
    assert cv2.imread(str(img), cv2.IMREAD_COLOR).dtype == np.uint8           # hloc / sapiens / refine_sync_visual read
    b4.convert_image(img, mask, tmp_path / "rgba.png", 1)                      # the RGBA training-image writer
    with Image.open(tmp_path / "rgba.png") as im:
        assert im.mode == "RGBA" and np.asarray(im).dtype == np.uint8
    with Image.open(img) as im:                                                # PIL readers: convert("RGB"), "L"
        assert np.asarray(im.convert("RGB")).dtype == np.uint8 and np.asarray(im.convert("L")).dtype == np.uint8
