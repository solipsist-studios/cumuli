# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

import numpy as np
import pytest
from PIL import Image

import birefnet_masks as bm


def mask_with_blobs(blobs, size=(40, 40)):
    """blobs: [(x0, y0, x1, y1, value)] filled rectangles on a zero mask."""
    m = np.zeros(size, dtype=np.uint8)
    for x0, y0, x1, y1, v in blobs:
        m[y0:y1, x0:x1] = v
    return Image.fromarray(m, mode="L")


# --------------------------------------------------------------------------
# keep_top_subject_bboxes -- pure numpy/OpenCV
# --------------------------------------------------------------------------

def test_keep_top_subject_keeps_most_confident_component():
    fmask = mask_with_blobs([(2, 2, 10, 10, 250), (20, 20, 30, 30, 90)])
    out = np.asarray(bm.keep_top_subject_bboxes(fmask, 1))
    assert out[5, 5] == 250
    assert out[25, 25] == 0


def test_keep_top_subject_ranks_by_confidence_sum_not_peak():
    # a large dim blob outweighs a small bright one
    fmask = mask_with_blobs([(0, 0, 4, 4, 255), (10, 10, 39, 39, 60)])
    out = np.asarray(bm.keep_top_subject_bboxes(fmask, 1))
    assert out[20, 20] == 60
    assert out[1, 1] == 0


def test_keep_top_subject_zero_subjects_blanks_mask():
    out = np.asarray(bm.keep_top_subject_bboxes(mask_with_blobs([(2, 2, 10, 10, 200)]), 0))
    assert not out.any()


def test_keep_top_subject_no_foreground_passes_through():
    fmask = mask_with_blobs([])
    assert bm.keep_top_subject_bboxes(fmask, 2) is fmask


# --------------------------------------------------------------------------
# remove_background -- model loading and inference are monkeypatched
# --------------------------------------------------------------------------

@pytest.fixture
def fake_model(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("tqdm")
    batches = []
    monkeypatch.setattr(bm, "load_model", lambda name, device: object())

    def extract(images, model, sema):
        batches.append(len(images))
        return [mask_with_blobs([(1, 1, 5, 5, 200), (10, 10, 14, 14, 100)], size=im.size[::-1])
                for im in images]
    monkeypatch.setattr(bm, "extract_masks", extract)
    return batches


def test_remove_background_mirrors_layout_and_batches(tmp_path, fake_model):
    images = tmp_path / "images"
    for rel in ("00.png", "01.png", "f1/02.png"):
        (images / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 16)).save(images / rel)
    (images / "notes.txt").write_text("not an image")

    n = bm.remove_background(images, tmp_path / "masks", image_ext=".png", batch_size=2)

    assert n == 3
    assert sorted(fake_model) == [1, 2]
    for rel in ("00.png", "01.png", "f1/02.png"):
        with Image.open(tmp_path / "masks" / rel) as m:
            assert m.mode == "L" and m.size == (20, 16)


def test_remove_background_max_subjects_filters(tmp_path, fake_model):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (20, 16)).save(images / "00.png")
    bm.remove_background(images, tmp_path / "masks", image_ext=".png", max_subjects=1)
    m = np.asarray(Image.open(tmp_path / "masks" / "00.png"))
    assert m[2, 2] == 200 and m[12, 12] == 0


def test_remove_background_no_images_warns(tmp_path, capsys, fake_model):
    (tmp_path / "images").mkdir()
    assert bm.remove_background(tmp_path / "images", tmp_path / "masks", image_ext=".png") == 0
    assert "no *.png images" in capsys.readouterr().out
