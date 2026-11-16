# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""color_adjust.py: the adjustments, the LUT, and the vectors Studio's browser preview is checked against."""

import json
import sys

import numpy as np
import pytest
from PIL import Image

import color_adjust as ca

VECTORS = [
    ((0.5, 0.5, 0.5), {}),
    ((0.2, 0.4, 0.6), {"exposure": 0.5}),
    ((0.2, 0.4, 0.6), {"temperature": 0.6}),
    ((0.2, 0.4, 0.6), {"contrast": 0.4}),
    ((0.2, 0.4, 0.6), {"saturation": -0.5}),
    ((0.9, 0.1, 0.3), {"exposure": -0.7, "temperature": -0.3, "contrast": 0.2, "saturation": 0.4}),
    ((0.0, 1.0, 0.5), {"exposure": 2.0, "contrast": -0.6}),
]


def test_identity_leaves_pixels_alone():
    rgb = np.random.default_rng(0).random((4, 4, 3))
    assert np.allclose(ca.adjust(rgb), rgb, atol=1e-9)


def test_exposure_is_stops_in_linear_light():
    expected = float(ca.linear_to_srgb(ca.srgb_to_linear(0.5) * 2))
    assert ca.adjust(np.array([0.5, 0.5, 0.5]), exposure=1.0) == pytest.approx([expected] * 3, abs=1e-9)


def test_saturation_minus_one_is_grey_and_temperature_warms():
    grey = ca.adjust(np.array([0.9, 0.2, 0.4]), saturation=-1.0)
    assert grey[0] == pytest.approx(grey[1]) == pytest.approx(grey[2])
    warm = ca.adjust(np.array([0.5, 0.5, 0.5]), temperature=1.0)
    assert warm[0] > 0.5 > warm[2]


def test_vectors_for_the_browser_preview():
    """Prints nothing; tools/check_color.mjs in Studio reads the same list. Kept here so a change to the math changes the vectors."""
    out = [{"in": list(rgb), "params": params, "out": ca.adjust(np.array(rgb), **params).tolist()} for rgb, params in VECTORS]
    assert all(0.0 <= c <= 1.0 for row in out for c in row["out"])
    assert json.loads(json.dumps(out))[0]["out"] == pytest.approx([0.5, 0.5, 0.5], abs=1e-9)


def write_cube(path, size=3, fn=lambda r, g, b: (r, g, b)):
    lines = ["TITLE \"t\"", f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(" ".join(f"{v:.6f}" for v in fn(r / (size - 1), g / (size - 1), b / (size - 1))))
    path.write_text("\n".join(lines))


def test_identity_cube_changes_nothing_and_a_swap_cube_swaps(tmp_path):
    write_cube(tmp_path / "id.cube")
    table, lo, hi = ca.read_cube(tmp_path / "id.cube")
    rgb = np.random.default_rng(1).random((5, 5, 3))
    assert np.allclose(ca.apply_lut(rgb, table, lo, hi), rgb, atol=1e-9)
    write_cube(tmp_path / "swap.cube", 5, lambda r, g, b: (b, g, r))
    table, lo, hi = ca.read_cube(tmp_path / "swap.cube")
    assert np.allclose(ca.apply_lut(np.array([0.2, 0.5, 0.9]), table, lo, hi), [0.9, 0.5, 0.2], atol=1e-9)


def test_bad_cubes_are_refused(tmp_path):
    (tmp_path / "a.cube").write_text("LUT_3D_SIZE 2\n0 0 0\n")
    with pytest.raises(ValueError):
        ca.read_cube(tmp_path / "a.cube")
    (tmp_path / "b.cube").write_text("LUT_1D_SIZE 4\n")
    with pytest.raises(ValueError, match="1D"):
        ca.read_cube(tmp_path / "b.cube")


def test_cli_applies_per_camera_overrides(tmp_path, monkeypatch):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for name in ("undistorted_0001.png", "undistorted_0002.png"):
        Image.new("RGB", (4, 4), (100, 100, 100)).save(src / name)
    settings = tmp_path / "s.json"
    settings.write_text(json.dumps({"method": "adjust", "adjust": {"exposure": 1.0}, "cameras": {"0002": {"method": "none"}}}))
    monkeypatch.setattr(sys, "argv", ["c", "--settings", str(settings), "--in_dir", str(src), "--out_dir", str(out)])
    ca.main()
    assert np.asarray(Image.open(out / "undistorted_0001.png"))[0, 0, 0] > 100
    assert np.asarray(Image.open(out / "undistorted_0002.png"))[0, 0, 0] == 100
