# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""frame_motion.py: each frame's subject speed from optical flow inside the masks, and the extra visits fast
frames get."""

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import frame_motion


def write_clip(root, shifts, labels=("00", "01"), size=(320, 240)):
    """A textured subject that moves shifts[i] pixels right between frame i-1 and i, on a still background."""
    w, h = size
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 1.5)
    x = 60.0
    for i, s in enumerate(shifts):
        x += s
        d = root / f"frame_{i:04d}"
        (d / "images_flat").mkdir(parents=True)
        (d / "fmasks_clean").mkdir(parents=True)
        img = np.full((h, w), 90, np.uint8)
        mask = np.zeros((h, w), np.uint8)
        x0 = int(round(x))
        img[60:180, x0:x0 + 80] = np.roll(texture, x0, axis=1)[60:180, x0:x0 + 80]
        mask[60:180, x0:x0 + 80] = 255
        for label in labels:
            cv2.imwrite(str(d / "images_flat" / f"{label}.png"), img)
            cv2.imwrite(str(d / "fmasks_clean" / f"{label}.png"), mask)


def test_fast_frames_get_more_visits_and_slow_ones_none(tmp_path):
    shifts = [0, 2, 2, 2, 2, 2, 8, 8, 8, 2, 2, 2, 2, 2]
    write_clip(tmp_path / "clip", shifts)
    out = tmp_path / "motion.json"
    assert frame_motion.main(["--frames_root", str(tmp_path / "clip"), "--out", str(out), "--jobs", "2"]) == 0
    report = json.loads(out.read_text())
    frames = report["frames"]
    assert report["median_speed_px"] == pytest.approx(2.0, abs=0.6)
    assert frames["frame_0007"]["speed_px"] == pytest.approx(8.0, abs=1.5)  # full-resolution pixels per frame
    fast = {n for n, v in frames.items() if v["weight"] > 1}
    assert {"frame_0006", "frame_0007", "frame_0008"} <= fast
    assert not fast & {"frame_0002", "frame_0003", "frame_0011", "frame_0012"}
    assert max(v["weight"] for v in frames.values()) <= report["max_boost"]


def test_weights_rule():
    median, w = frame_motion.weights({"a": 2.0, "b": 2.0, "c": 2.2, "d": 4.0, "e": 20.0, "f": None}, 1.5, 3.0)
    assert median == 2.2  # f has no speed and is not counted
    assert w == {"a": 1.0, "b": 1.0, "c": 1.0, "d": pytest.approx(4.0 / 2.2), "e": 3.0, "f": 1.0}
