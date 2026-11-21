# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import image_formats as fmt  # noqa: E402


def noisy_rgba(h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    a[..., 3] = np.where(rng.random((h, w)) > 0.4, 255, 0)
    return a


def test_webp_default_is_lossless_roundtrip(tmp_path):
    arr = noisy_rgba()
    p = tmp_path / "a.webp"
    fmt.save_image(Image.fromarray(arr), p)
    back = np.asarray(Image.open(p).convert("RGBA"))
    assert (back[..., 3] == arr[..., 3]).all()
    visible = arr[..., 3] > 0
    assert (back[visible] == arr[visible]).all()


def test_webp_quality_is_lossy_and_smaller(tmp_path):
    arr = noisy_rgba(96, 128)
    lossless, lossy = tmp_path / "l.webp", tmp_path / "q.webp"
    fmt.save_image(Image.fromarray(arr), lossless)
    fmt.save_image(Image.fromarray(arr), lossy, quality=60)
    assert lossy.stat().st_size < lossless.stat().st_size
    back = np.asarray(Image.open(lossy).convert("RGBA"))
    visible = arr[..., 3] > 0
    assert not (back[visible] == arr[visible]).all()


def test_alpha_survives_lossy_webp(tmp_path):
    arr = noisy_rgba()
    p = tmp_path / "q.webp"
    fmt.save_image(Image.fromarray(arr), p, quality=50)
    assert (np.asarray(Image.open(p).convert("RGBA"))[..., 3] == arr[..., 3]).all()


def test_png_keeps_compress_level():
    assert fmt.pil_save_kwargs(".png", None, 1) == {"compress_level": 1}
    assert fmt.pil_save_kwargs(".PNG") == {}


@pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg"])
def test_quality_is_an_error_off_webp(ext):
    with pytest.raises(fmt.ImageQualityError):
        fmt.check_image_quality(ext, 90)


@pytest.mark.parametrize("q", [0, 101, -3, 2.5, True])
def test_quality_range_checked(q):
    with pytest.raises(fmt.ImageQualityError):
        fmt.check_image_quality(".webp", q)


def test_no_quality_never_lossy():
    assert fmt.pil_save_kwargs(".webp")["lossless"] is True
    assert fmt.pil_save_kwargs(".webp", 80)["lossless"] is False


def test_cv2_params_lossless_by_default():
    cv2 = pytest.importorskip("cv2")
    assert fmt.cv2_write_params(".webp") == [cv2.IMWRITE_WEBP_QUALITY, 101]
    assert fmt.cv2_write_params(".webp", 70) == [cv2.IMWRITE_WEBP_QUALITY, 70]
    arr = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".webp", arr, fmt.cv2_write_params(".webp"))
    assert ok and (cv2.imdecode(buf, cv2.IMREAD_COLOR) == arr).all()
