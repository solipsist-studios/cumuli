# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Frame image formats: lossless WebP by default, an optional lossy WebP quality, per script.

The properties worth pinning: a .webp with no quality is pixel-exact, a quality makes a smaller
lossy file whose alpha is still exact, a quality with .png is an error, scored ground truth never
leaves lossless PNG, and a non-PNG training frame says so in its transforms file_path.
"""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import build_colmap_sparse
import build_flat_dataset as bfd
import build_flipbook_4dgs_dataset as builder
import color_adjust as ca
import extract_synced_frames as esf
import run_hloc
import undistort_frames as uf


def noise(h, w, seed):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def half_matte(h, w):
    m = np.zeros((h, w), np.uint8)
    m[:, : w // 2] = 255
    return m


def write_flipbook(root, labels, n_frames, size=(64, 48), image_ext=".png"):
    """Static rig, a noisy image per camera and a matte covering the left half."""
    w, h = size
    frames = []
    for label in labels:
        c2w = np.eye(4)
        c2w[:3, 3] = [float(int(label[-2:])), 0.0, 3.0]
        frames.append({
            "camera_label": label, "file_path": f"images_flat/{label}{image_ext}",
            "transform_matrix": c2w.tolist(),
            "fl_x": 500.0, "fl_y": 500.0, "cx": w / 2, "cy": h / 2, "w": w, "h": h,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        })
    for i in range(n_frames):
        frame_dir = root / f"frame_{i:04d}"
        (frame_dir / "images_flat").mkdir(parents=True, exist_ok=True)
        (frame_dir / "fmasks_clean").mkdir(parents=True, exist_ok=True)
        (frame_dir / "transforms.json").write_text(json.dumps({"camera_model": "OPENCV", "frames": frames}))
        for k, label in enumerate(labels):
            Image.fromarray(noise(h, w, 10 * i + k)).save(
                frame_dir / "images_flat" / f"{label}{image_ext}", **({"lossless": True} if image_ext == ".webp" else {}))
            Image.fromarray(half_matte(h, w)).save(frame_dir / "fmasks_clean" / f"{label}.png")
    return root


def build(monkeypatch, tmp_path, name, *extra, labels=("00", "01", "02"), image_ext=".png"):
    src = tmp_path / "src"
    if not src.exists():
        write_flipbook(src, list(labels), 2, image_ext=image_ext)
    out = tmp_path / name
    monkeypatch.setattr(sys, "argv", [
        "build_flipbook_4dgs_dataset.py", "--flipbook_root", str(src), "--out", str(out),
        "--fps", "24", "--downscale", "1", "--hull_min_views", "1", "--jobs", "2",
        "--init_bbox=-1,-1,-1,1,1,1", *extra])
    builder.main()
    return out


def rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"))


# ------------------------------------------------------------ flipbook builder
def test_flipbook_default_is_unchanged_png_with_extensionless_paths(tmp_path, monkeypatch):
    out = build(monkeypatch, tmp_path, "d")
    rows = json.loads((out / "transforms_train.json").read_text())["frames"]
    assert all("." not in r["file_path"].rsplit("/", 1)[-1] for r in rows)
    assert (out / "realcams" / "cam00" / "frame_00001.png").is_file()


def test_flipbook_webp_is_lossless_and_names_its_extension(tmp_path, monkeypatch):
    png = build(monkeypatch, tmp_path, "png")
    webp = build(monkeypatch, tmp_path, "webp", "--out_image_ext", ".webp")
    rows = json.loads((webp / "transforms_train.json").read_text())["frames"]
    assert rows and all(r["file_path"].endswith(".webp") for r in rows)
    for r in rows:
        assert (webp / r["file_path"]).is_file()
        a = rgba(png / r["file_path"].replace(".webp", ".png"))
        b = rgba(webp / r["file_path"])
        assert (a[..., 3] == b[..., 3]).all()
        visible = a[..., 3] > 0
        assert (a[visible] == b[visible]).all()


def test_flipbook_quality_is_lossy_smaller_and_keeps_alpha(tmp_path, monkeypatch):
    lossless = build(monkeypatch, tmp_path, "l", "--out_image_ext", ".webp")
    lossy = build(monkeypatch, tmp_path, "q", "--out_image_ext", ".webp", "--out_image_quality", "40")
    one = "realcams/cam00/frame_00001.webp"
    assert (lossy / one).stat().st_size < (lossless / one).stat().st_size
    a, b = rgba(lossless / one), rgba(lossy / one)
    assert (a[..., 3] == b[..., 3]).all()
    visible = a[..., 3] > 0
    assert not (a[visible] == b[visible]).all()


def test_flipbook_scored_cameras_stay_lossless_png(tmp_path, monkeypatch):
    out = build(monkeypatch, tmp_path, "d", "--out_image_ext", ".webp", "--out_image_quality", "40",
                "--test_cameras", "01")
    test_rows = json.loads((out / "transforms_test.json").read_text())["frames"]
    assert test_rows and all("." not in r["file_path"].rsplit("/", 1)[-1] for r in test_rows)
    assert (out / "realcams" / "cam01" / "frame_00001.png").is_file()
    assert not list((out / "realcams" / "cam01").glob("*.webp"))
    assert len(list((out / "eval_gt_flat").glob("*.png"))) == 2
    train_rows = json.loads((out / "transforms_train.json").read_text())["frames"]
    assert {r["camera_label"] for r in train_rows} == {"00", "02"}
    assert all(r["file_path"].endswith(".webp") for r in train_rows)


def test_flipbook_quality_with_png_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="image quality"):
        build(monkeypatch, tmp_path, "d", "--out_image_quality", "80")


def test_flipbook_reads_webp_source_frames(tmp_path, monkeypatch):
    out = build(monkeypatch, tmp_path, "d", "--image_ext", ".webp", "--dedupe", image_ext=".webp")
    assert (out / "realcams" / "cam00" / "frame_00001.png").is_file()


def test_flipbook_jpeg_output_is_not_offered(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        build(monkeypatch, tmp_path, "d", "--out_image_ext", ".jpg")


# ------------------------------------------------------------ build_flat_dataset
def flat_run(monkeypatch, tmp_path, *extra, src_ext=".png"):
    und = tmp_path / "und"
    d = und / "Camera_0001"
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noise(48, 64, 3)).save(d / f"0000{src_ext}")
    tf = tmp_path / "t.json"
    tf.write_text(json.dumps({"frames": [{"camera_label": "Camera_undistorted_0001"}]}))
    out = tmp_path / "flat"
    monkeypatch.setattr(sys, "argv", ["p", "--transforms", str(tf), "--undistorted_dir", str(und),
                                      "--out_images_flat", str(out), "--out_transforms", str(tmp_path / "o.json"),
                                      "--image_ext", src_ext, *extra])
    bfd.main()
    return out, d / f"0000{src_ext}"


def test_flat_png_to_webp_is_lossless(tmp_path, monkeypatch):
    out, src = flat_run(monkeypatch, tmp_path, "--out_image_ext", ".webp")
    assert (np.asarray(Image.open(out / "00.webp").convert("RGB")) == np.asarray(Image.open(src).convert("RGB"))).all()


def test_flat_quality_reencodes_even_when_the_extension_matches(tmp_path, monkeypatch):
    out, src = flat_run(monkeypatch, tmp_path, "--out_image_ext", ".webp", "--out_image_quality", "30", src_ext=".webp")
    assert (out / "00.webp").read_bytes() != src.read_bytes()
    assert (out / "00.webp").stat().st_size < src.stat().st_size


def test_flat_matching_extension_without_quality_still_copies(tmp_path, monkeypatch):
    out, src = flat_run(monkeypatch, tmp_path, "--out_image_ext", ".png")
    assert (out / "00.png").read_bytes() == src.read_bytes()


def test_flat_quality_with_png_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        flat_run(monkeypatch, tmp_path, "--out_image_quality", "80")


# ------------------------------------------------------------ color_adjust
def adjust_run(monkeypatch, tmp_path, *extra, ext=".webp"):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir(exist_ok=True)
    Image.fromarray(noise(48, 64, 5)).save(src / f"undistorted_0001{ext}", **({"lossless": True} if ext == ".webp" else {}))
    (tmp_path / "s.json").write_text(json.dumps({"method": "none"}))
    monkeypatch.setattr(sys, "argv", ["c", "--settings", str(tmp_path / "s.json"), "--in_dir", str(src),
                                      "--out_dir", str(out), "--image_ext", ext, "--workers", "1", *extra])
    ca.main()
    return src / f"undistorted_0001{ext}", out / f"undistorted_0001{ext}"


def test_color_adjust_webp_is_lossless(tmp_path, monkeypatch):
    src, out = adjust_run(monkeypatch, tmp_path)
    assert (np.asarray(Image.open(out).convert("RGB")) == np.asarray(Image.open(src).convert("RGB"))).all()


def test_color_adjust_quality_is_lossy_and_smaller(tmp_path, monkeypatch):
    src, out = adjust_run(monkeypatch, tmp_path, "--image_quality", "30")
    assert out.stat().st_size < src.stat().st_size
    assert not (np.asarray(Image.open(out).convert("RGB")) == np.asarray(Image.open(src).convert("RGB"))).all()


def test_color_adjust_quality_with_png_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        adjust_run(monkeypatch, tmp_path, "--image_quality", "30", ext=".png")


# ------------------------------------------------------------ undistort_frames
def test_undistort_write_image_webp_is_lossless_by_default(tmp_path):
    cv2 = pytest.importorskip("cv2")
    img = noise(40, 56, 7)
    assert uf.write_image(tmp_path / "a.webp", img)
    assert (cv2.imread(str(tmp_path / "a.webp")) == img).all()
    assert uf.write_image(tmp_path / "q.webp", img, 30)
    assert (tmp_path / "q.webp").stat().st_size < (tmp_path / "a.webp").stat().st_size


def test_undistort_quality_with_png_output_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["u", "--frames_dir", str(tmp_path), "--calib_dir", str(tmp_path),
                                      "--out_dir", str(tmp_path / "o"), "--out_pkl_dir", str(tmp_path / "p"),
                                      "--out_image_ext", ".png", "--out_image_quality", "50"])
    with pytest.raises(SystemExit):
        uf.main()


# ------------------------------------------------------------ build_colmap_sparse, run_hloc
def test_colmap_bake_webp_lossless_and_quality(tmp_path):
    img, mask = tmp_path / "i.png", tmp_path / "m.png"
    Image.fromarray(noise(48, 64, 2)).save(img)
    Image.fromarray(half_matte(48, 64)).save(mask)
    build_colmap_sparse.bake_rgba(img, mask, tmp_path / "a.webp")
    build_colmap_sparse.bake_rgba(img, mask, tmp_path / "b.webp", 30)
    a, b = rgba(tmp_path / "a.webp"), rgba(tmp_path / "b.webp")
    assert (a[..., 3] == b[..., 3]).all() and (a[..., 3] == half_matte(48, 64)).all()
    assert (tmp_path / "b.webp").stat().st_size < (tmp_path / "a.webp").stat().st_size
    assert (a[..., :3][a[..., 3] > 0] == np.asarray(Image.open(img).convert("RGB"))[a[..., 3] > 0]).all()


def test_run_hloc_init_transforms_name_the_real_extension(tmp_path):
    out = tmp_path / "init.json"
    run_hloc.build_init_transforms(tmp_path, ["0001"], out, image_ext=".webp")
    assert json.loads(out.read_text())["frames"][0]["file_path"].endswith("/0000.webp")
    run_hloc.build_init_transforms(tmp_path, ["0001"], out)
    assert json.loads(out.read_text())["frames"][0]["file_path"].endswith("/0000.jpg")


# ------------------------------------------------------------ frame_motion
def test_frame_motion_reads_webp_frames(tmp_path):
    import frame_motion
    from test_frame_motion import write_clip

    write_clip(tmp_path / "clip", [0, 2, 2, 8, 8, 2, 2])
    expected = tmp_path / "png.json"
    assert frame_motion.main(["--frames_root", str(tmp_path / "clip"), "--out", str(expected), "--jobs", "1"]) == 0
    for p in (tmp_path / "clip").glob("frame_*/images_flat/*.png"):
        Image.open(p).save(p.with_suffix(".webp"), lossless=True)
        p.unlink()
    got = tmp_path / "webp.json"
    assert frame_motion.main(["--frames_root", str(tmp_path / "clip"), "--out", str(got), "--jobs", "1",
                              "--image_ext", ".webp"]) == 0
    assert json.loads(got.read_text())["frames"] == json.loads(expected.read_text())["frames"]


# ------------------------------------------------------------ extract_synced_frames
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_webp_window_writes_one_lossless_file_per_frame(tmp_path):
    clip = tmp_path / "0001.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1",
                    "-c:v", "mpeg4", str(clip)], check=True)
    paths = esf.extract_frames(clip, 0.0, 3, tmp_path / "out", ".webp")
    assert [p.name for p in paths] == ["0001_01.webp", "0001_02.webp", "0001_03.webp"]
    assert len(list((tmp_path / "out").glob("*.webp"))) == 3     # not one animated file
    for p in paths:
        data = p.read_bytes()
        assert data[:4] == b"RIFF" and b"VP8L" in data[:32]     # the lossless WebP chunk
        with Image.open(p) as im:
            assert im.n_frames == 1


def test_pp3_with_webp_converts_rawtherapees_png_to_lossless_webp(tmp_path, monkeypatch):
    movies = tmp_path / "movies"
    movies.mkdir()
    (movies / "0001.mp4").write_bytes(b"")
    offsets = tmp_path / "o.json"
    offsets.write_text(json.dumps({"reference_camera": "0001.mp4",
                                   "offsets": {"0001.mp4": {"fps": 30.0, "frame_offset": 3}}}))
    pp3_dir = tmp_path / "pp3"
    pp3_dir.mkdir()
    (pp3_dir / "0001.mp4.pp3").write_text("")
    corrected = noise(48, 64, 9)

    def fake_extract(video_path, seek, count, out_dir, ext):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{video_path.stem}_01{ext}"
        p.write_bytes(b"")
        return [p]

    def fake_apply_pp3(rt_cmd, image_path, pp3, out_path):
        Image.fromarray(corrected).save(out_path, format="PNG")    # RawTherapee writes PNG whatever the name

    monkeypatch.setattr(esf, "extract_frames", fake_extract)
    monkeypatch.setattr(esf, "resolve_rawtherapee_cmd", lambda override=None: ["rawtherapee-cli"])
    monkeypatch.setattr(esf, "apply_pp3", fake_apply_pp3)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["p", str(movies), str(offsets), str(out), "--pp3_dir", str(pp3_dir),
                                      "--output_ext", ".webp"])
    esf.main()
    assert sorted(p.name for p in out.iterdir()) == ["0001.webp"]            # the intermediate PNG is gone
    assert b"VP8L" in (out / "0001.webp").read_bytes()[:32]
    with Image.open(out / "0001.webp") as im:
        assert (np.asarray(im.convert("RGB")) == corrected).all()


# ------------------------------------------------------------ extraction speed
def png_bit_depth(path):
    return path.read_bytes()[24]      # the IHDR bit depth: 8 for rgb24, 16 for rgb48


def test_ffmpeg_format_args_per_extension():
    assert esf.ffmpeg_format_args(".png") == ["-pix_fmt", "rgb24", "-compression_level", "1"]
    assert esf.ffmpeg_format_args(".webp") == ["-pix_fmt", "rgb24"]
    assert esf.ffmpeg_format_args(".jpg") == [] == esf.ffmpeg_format_args(".jpeg")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_10_bit_source_is_written_as_a_smaller_8_bit_png(tmp_path):
    """ffmpeg writes a yuv420p10le source as a 16-bit PNG unless told otherwise; every later stage
    reads 8 bits, so the extra bits only cost disk and encode time."""
    clip = tmp_path / "0001.mkv"
    # Sensor-like noise: a clean test pattern compresses so well that the 16-bit cost hides.
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=10:duration=1,noise=alls=25:allf=t",
                    "-pix_fmt", "yuv420p10le", "-c:v", "ffv1", str(clip)], check=True)
    before = tmp_path / "before.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-frames:v", "1", str(before)], check=True)
    (after,) = esf.extract_frames(clip, 0.0, 1, tmp_path / "out", ".png")
    assert png_bit_depth(before) == 16 and png_bit_depth(after) == 8
    print(f"16-bit PNG {before.stat().st_size} bytes -> 8-bit level-1 PNG {after.stat().st_size} bytes "
          f"({before.stat().st_size / after.stat().st_size:.1f}x smaller)")
    assert after.stat().st_size < before.stat().st_size / 1.5
    with Image.open(after) as im:
        assert im.mode == "RGB" and im.size == (320, 240)


def test_png_extraction_asks_for_rgb24_and_compression_level_1(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(esf.subprocess, "run", lambda cmd, check: commands.append(cmd))
    with pytest.raises(RuntimeError):                  # nothing is written, so the frame count check fires
        esf.extract_frames(tmp_path / "0001.mp4", 0.0, 1, tmp_path / "o", ".png")
    cmd = commands[0]
    assert cmd[cmd.index("-pix_fmt") + 1] == "rgb24" and cmd[cmd.index("-compression_level") + 1] == "1"
    commands.clear()
    with pytest.raises(RuntimeError):
        esf.extract_frames(tmp_path / "0001.mp4", 0.0, 1, tmp_path / "o", ".jpg")
    assert "-pix_fmt" not in commands[0] and "-compression_level" not in commands[0]


def run_cameras(tmp_path, monkeypatch, n, *extra):
    """Run main() over n fake cameras; returns the most extractions seen running at once."""
    import threading
    import time

    movies = tmp_path / "movies"
    movies.mkdir()
    entries = {}
    for k in range(1, n + 1):
        (movies / f"{k:04d}.mp4").write_bytes(b"")
        entries[f"{k:04d}.mp4"] = {"fps": 30.0, "frame_offset": 3}
    offsets = tmp_path / "o.json"
    offsets.write_text(json.dumps({"reference_camera": "0001.mp4", "offsets": entries}))
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def fake_extract(video_path, seek, count, out_dir, ext):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.15)
        with lock:
            state["now"] -= 1
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{video_path.stem}_01{ext}"
        p.write_bytes(b"")
        return [p]

    monkeypatch.setattr(esf, "extract_frames", fake_extract)
    monkeypatch.setattr(sys, "argv", ["p", str(movies), str(offsets), str(tmp_path / "out"), *extra])
    esf.main()
    return state["peak"], sorted(p.name for p in (tmp_path / "out").iterdir())


def test_cameras_are_extracted_concurrently_by_default(tmp_path, monkeypatch):
    peak, files = run_cameras(tmp_path, monkeypatch, 4)
    assert peak >= 2 and files == ["0001.jpg", "0002.jpg", "0003.jpg", "0004.jpg"]


def test_jobs_one_extracts_one_camera_at_a_time(tmp_path, monkeypatch):
    peak, files = run_cameras(tmp_path, monkeypatch, 3, "--jobs", "1")
    assert peak == 1 and len(files) == 3


def test_jobs_caps_the_concurrency(tmp_path, monkeypatch):
    peak, _ = run_cameras(tmp_path, monkeypatch, 6, "--jobs", "2")
    assert peak == 2


def test_a_failed_camera_still_exits_1_with_the_others_written(tmp_path, monkeypatch, capsys):
    (tmp_path / "movies").mkdir()
    (tmp_path / "movies" / "0001.mp4").write_bytes(b"")           # 0002.mp4 is in the offsets but missing
    offsets = tmp_path / "o.json"
    offsets.write_text(json.dumps({"reference_camera": "0001.mp4", "offsets": {
        "0001.mp4": {"fps": 30.0, "frame_offset": 3}, "0002.mp4": {"fps": 30.0, "frame_offset": 3}}}))

    def fake_extract(video_path, seek, count, out_dir, ext):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{video_path.stem}_01{ext}"
        p.write_bytes(b"")
        return [p]

    monkeypatch.setattr(esf, "extract_frames", fake_extract)
    monkeypatch.setattr(sys, "argv", ["p", str(tmp_path / "movies"), str(offsets), str(tmp_path / "out")])
    with pytest.raises(SystemExit) as e:
        esf.main()
    assert e.value.code == 1
    assert "Failed/skipped cameras: ['0002.mp4']" in capsys.readouterr().out
    assert (tmp_path / "out" / "0001.jpg").is_file()


# ------------------------------------------------------------ WebP encode cost
def test_lossless_webp_defaults_to_a_fast_encoder_method():
    assert fmt_kwargs(".webp")["method"] == 2
    assert fmt_kwargs(".webp", webp_method=0)["method"] == 0
    assert fmt_kwargs(".webp", quality=50)["method"] == 4


def fmt_kwargs(ext, **kw):
    import image_formats

    return image_formats.pil_save_kwargs(ext, **kw)


def test_undistort_writes_lossless_webp_with_pil_method_0_not_cv2(tmp_path, monkeypatch):
    seen = {}
    real = uf.save_image

    def spy(image, path, quality=None, png_compress_level=None, webp_method=None):
        seen.update(quality=quality, webp_method=webp_method)
        return real(image, path, quality, png_compress_level, webp_method)

    monkeypatch.setattr(uf, "save_image", spy)
    img = noise(40, 56, 4)
    assert uf.write_image(tmp_path / "a.webp", img)
    assert seen == {"quality": None, "webp_method": 0}
    # BGR in, RGB out: the channel order survives the PIL path
    with Image.open(tmp_path / "a.webp") as im:
        assert (np.asarray(im.convert("RGB")) == img[..., ::-1]).all()
    assert uf.write_image(tmp_path / "q.webp", img, 40)
    assert seen == {"quality": 40, "webp_method": None}


def test_flat_dataset_converts_in_parallel_and_keeps_label_order(tmp_path, monkeypatch):
    und = tmp_path / "und"
    frames = []
    for k in range(1, 6):
        d = und / f"Camera_{k:04d}"
        d.mkdir(parents=True)
        Image.fromarray(noise(24, 32, k)).save(d / "0000.png")
        frames.append({"camera_label": f"Camera_undistorted_{k:04d}"})
    tf = tmp_path / "t.json"
    tf.write_text(json.dumps({"frames": frames}))
    out = tmp_path / "flat"
    monkeypatch.setattr(sys, "argv", ["p", "--transforms", str(tf), "--undistorted_dir", str(und), "--image_ext", ".png",
                                      "--out_images_flat", str(out), "--out_transforms", str(tmp_path / "o.json"),
                                      "--out_image_ext", ".webp", "--jobs", "4"])
    bfd.main()
    assert sorted(p.name for p in out.iterdir()) == [f"{k:02d}.webp" for k in range(5)]   # no temp files left
    for k in range(5):
        with Image.open(out / f"{k:02d}.webp") as im:
            assert (np.asarray(im.convert("RGB")) == noise(24, 32, k + 1)).all()          # label k is camera k+1
