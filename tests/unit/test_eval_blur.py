# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""eval_render.blur_and_shift: the blur a render has against its photo, told
apart from a render that is sharp but in the wrong place (misregistration,
for example a camera rendered at the wrong time)."""

import pytest

torch = pytest.importorskip("torch")

from eval_render import _gaussian_blur, blur_and_shift, summarize_blur


def subject(h=64, w=80, seed=0):
    """A textured blob on black, like a masked ground-truth frame."""
    g = torch.Generator().manual_seed(seed)
    img = torch.zeros(h, w, 3)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    inside = ((yy - h / 2) / (h / 3)) ** 2 + ((xx - w / 2) / (w / 3)) ** 2 < 1
    img[inside] = torch.rand(int(inside.sum()), 3, generator=g) * 0.8 + 0.1
    return img


def test_a_render_equal_to_its_photo_has_no_blur_and_no_shift():
    photo = subject()
    out = blur_and_shift(photo, photo.clone())
    assert out["blur_px"] == 0.0 and out["aligned_blur_px"] == 0.0 and out["shift_px"] == [0, 0]
    assert out["error"] == pytest.approx(0.0, abs=1e-7)


def test_a_blurred_render_measures_its_blur():
    photo = subject()
    out = blur_and_shift(photo, _gaussian_blur(photo, 1.5))
    assert out["blur_px"] == pytest.approx(1.5, abs=0.25)
    assert out["shift_px"] == [0, 0]


def test_a_displaced_render_is_misregistration_not_blur():
    photo = subject()
    # render pixel (x, y) shows photo pixel (x + 2, y - 1)
    render = torch.roll(photo, shifts=(1, -2), dims=(0, 1))
    out = blur_and_shift(photo, render)
    assert out["blur_px"] > 0.0          # without shifting, a blurred photo matches better than the sharp one
    assert out["shift_px"] == [2, -1]    # the shift explains it
    assert out["aligned_blur_px"] == 0.0
    assert out["aligned_error"] < out["error"]


def test_an_empty_view_gives_nothing_and_the_summary_skips_it():
    empty = torch.zeros(32, 32, 3)
    assert blur_and_shift(empty, empty) is None
    rows = [None, {"blur_px": 1.0, "aligned_blur_px": 0.5, "shift_px": [0, 0]},
            {"blur_px": 2.0, "aligned_blur_px": 0.0, "shift_px": [3, 4]}]
    s = summarize_blur(rows)
    assert s == {"views": 2, "blur_px": 1.5, "aligned_blur_px": 0.25, "shift_px": 2.5, "shifted_views": 1}
    assert summarize_blur([None]) is None
