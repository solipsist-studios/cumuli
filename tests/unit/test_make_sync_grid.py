import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image

import make_sync_grid as msg


def write_image(path, w, h):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h)).save(path)


LABEL_HEIGHT = 30
COLS = 4


def test_main_usage_exits_1_on_too_few_args(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "/only/one/arg"])
    with pytest.raises(SystemExit) as exc_info:
        msg.main()
    assert exc_info.value.code == 1
    assert "Usage" in capsys.readouterr().out


def test_main_usage_exits_1_on_too_many_args(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "a", "b", "c", "d"])
    with pytest.raises(SystemExit) as exc_info:
        msg.main()
    assert exc_info.value.code == 1
    assert "Usage" in capsys.readouterr().out


def test_main_errors_when_frames_dir_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", str(tmp_path / "nope"), str(tmp_path / "grid.jpg")])
    with pytest.raises(SystemExit) as exc_info:
        msg.main()
    assert exc_info.value.code == 1
    assert "is not a directory" in capsys.readouterr().out


def test_main_errors_when_no_images_found(tmp_path, monkeypatch, capsys):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(tmp_path / "grid.jpg")])
    with pytest.raises(SystemExit) as exc_info:
        msg.main()
    assert exc_info.value.code == 1
    assert "no" in capsys.readouterr().out


def test_main_errors_cleanly_on_zero_thumb_width(tmp_path, monkeypatch, capsys):
    # Regression test: without an explicit upfront check, thumb_width=0
    # makes EVERY image's resize() raise ValueError, which the unreadable-
    # image handler would misreport as "could not be read as an image" --
    # misattributing the actual problem (the thumb_width argument) to the
    # image files.
    frames_dir = tmp_path / "frames"
    write_image(frames_dir / "0001.jpg", w=640, h=480)
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(tmp_path / "grid.jpg"), "0"])
    with pytest.raises(SystemExit) as exc_info:
        msg.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "thumb_width must be > 0" in out
    assert "could not be read" not in out  # must not misattribute to the image


def test_main_errors_cleanly_on_negative_thumb_width(tmp_path, monkeypatch, capsys):
    frames_dir = tmp_path / "frames"
    write_image(frames_dir / "0001.jpg", w=640, h=480)
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(tmp_path / "grid.jpg"), "-10"])
    with pytest.raises(SystemExit) as exc_info:
        msg.main()
    assert exc_info.value.code == 1
    assert "thumb_width must be > 0" in capsys.readouterr().out


def test_main_skips_zero_width_source_image_without_crashing(tmp_path, monkeypatch, capsys):
    # A source image PIL can open but that reports width=0 (a malformed
    # header some decoder let through) would divide-by-zero in the aspect
    # ratio computation -- ZeroDivisionError isn't an OSError/ValueError,
    # so the existing unreadable-image handler wouldn't have caught it
    # without also catching ZeroDivisionError explicitly.
    frames_dir = tmp_path / "frames"
    write_image(frames_dir / "0001.jpg", w=640, h=480)  # one good image
    (frames_dir / "0002.jpg").write_bytes(b"placeholder")  # path stands in for the mock below

    real_open = Image.open

    def fake_open(path, *a, **kw):
        if Path(path).name == "0002.jpg":
            class ZeroWidthImage:
                width = 0
                height = 100
            return ZeroWidthImage()
        return real_open(path, *a, **kw)

    monkeypatch.setattr(msg.Image, "open", fake_open)
    out_path = tmp_path / "grid.jpg"
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(out_path)])

    with pytest.raises(SystemExit) as exc_info:
        msg.main()  # must not crash with an uncaught ZeroDivisionError

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "0002.jpg could not be read" in out
    assert out_path.is_file()  # grid still built from the one good image


def test_main_skips_unreadable_image_and_continues(tmp_path, monkeypatch, capsys):
    # Regression test: a corrupt/unreadable image used to crash the whole
    # grid with a raw PIL.UnidentifiedImageError traceback instead of
    # skipping just that one image and still building a grid of the rest.
    frames_dir = tmp_path / "frames"
    write_image(frames_dir / "0001.jpg", w=640, h=480)
    (frames_dir / "0002.jpg").write_text("not an image")
    out_path = tmp_path / "grid.jpg"
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(out_path)])

    with pytest.raises(SystemExit) as exc_info:
        msg.main()

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "WARNING: 0002.jpg could not be read" in out
    assert "Skipped unreadable images" in out and "0002.jpg" in out
    assert out_path.is_file()  # grid was still built from the readable image


def test_main_errors_when_all_images_unreadable(tmp_path, monkeypatch, capsys):
    frames_dir = tmp_path / "frames"
    (frames_dir / "0001.jpg").parent.mkdir(parents=True, exist_ok=True)
    (frames_dir / "0001.jpg").write_text("not an image")
    out_path = tmp_path / "grid.jpg"
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(out_path)])

    with pytest.raises(SystemExit) as exc_info:
        msg.main()

    assert exc_info.value.code == 1
    assert "no readable images remained" in capsys.readouterr().out
    assert not out_path.exists()


def test_main_default_thumb_width_and_grid_layout_for_5_frames(tmp_path, monkeypatch):
    # 5 frames -> 4 cols, 2 rows at the default thumb_width=480.
    frames_dir = tmp_path / "frames"
    for i in range(5):
        write_image(frames_dir / f"{i:04d}.jpg", w=1920, h=1080)
    out_path = tmp_path / "grid.jpg"
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(out_path)])
    msg.main()

    thumb_h = int(480 * 1080 / 1920)
    cell_h = thumb_h + LABEL_HEIGHT
    with Image.open(out_path) as grid:
        assert grid.size == (COLS * 480, 2 * cell_h)


def test_main_explicit_thumb_width_override(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    for i in range(2):
        write_image(frames_dir / f"{i:04d}.jpg", w=1000, h=500)
    out_path = tmp_path / "grid.jpg"
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(out_path), "100"])
    msg.main()

    thumb_h = int(100 * 500 / 1000)
    cell_h = thumb_h + LABEL_HEIGHT
    with Image.open(out_path) as grid:
        # 2 frames still fits in a single row of the 4-col grid.
        assert grid.size == (COLS * 100, 1 * cell_h)


def test_main_single_frame_grid_is_one_cell(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    write_image(frames_dir / "0001.jpg", w=640, h=480)
    out_path = tmp_path / "grid.jpg"
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(out_path)])
    msg.main()

    thumb_h = int(480 * 480 / 640)
    with Image.open(out_path) as grid:
        assert grid.size == (COLS * 480, thumb_h + LABEL_HEIGHT)


def test_main_row_count_grows_past_4_cameras(tmp_path, monkeypatch):
    # 9 frames -> ceil(9/4) = 3 rows.
    frames_dir = tmp_path / "frames"
    for i in range(9):
        write_image(frames_dir / f"{i:04d}.jpg", w=640, h=480)
    out_path = tmp_path / "grid.jpg"
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(out_path)])
    msg.main()

    thumb_h = int(480 * 480 / 640)
    cell_h = thumb_h + LABEL_HEIGHT
    with Image.open(out_path) as grid:
        assert grid.size == (COLS * 480, 3 * cell_h)


def test_main_picks_up_mixed_supported_extensions(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    write_image(frames_dir / "0001.jpg", w=640, h=480)
    write_image(frames_dir / "0002.png", w=640, h=480)
    out_path = tmp_path / "grid.jpg"
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(out_path)])
    msg.main()  # should not raise -- both extensions must be picked up

    thumb_h = int(480 * 480 / 640)
    with Image.open(out_path) as grid:
        assert grid.size == (COLS * 480, thumb_h + LABEL_HEIGHT)


def test_main_cell_height_driven_by_tallest_thumbnail(tmp_path, monkeypatch):
    # Mixed aspect ratios in the same batch -- cell height must be tall
    # enough for the tallest thumbnail, not just the first one found.
    frames_dir = tmp_path / "frames"
    write_image(frames_dir / "0001.jpg", w=1920, h=1080)  # wide, short thumb
    write_image(frames_dir / "0002.jpg", w=480, h=1920)   # narrow, tall thumb
    out_path = tmp_path / "grid.jpg"
    monkeypatch.setattr(sys, "argv", ["prog", str(frames_dir), str(out_path)])
    msg.main()

    tall_thumb_h = int(480 * 1920 / 480)
    cell_h = tall_thumb_h + LABEL_HEIGHT
    with Image.open(out_path) as grid:
        assert grid.size == (COLS * 480, cell_h)
