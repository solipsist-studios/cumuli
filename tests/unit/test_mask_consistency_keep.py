# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""bake_sogst.mask_consistency_keep on cropped rigs.

Two properties of per-frame subject crops: the principal point moves from frame
to frame, and a camera can lack a timestamp that another camera has.
"""

import json

import numpy as np
from PIL import Image

import bake_sogst

SIZE = 32


def _write_frame(root, cam, idx, cx, cy):
    """RGBA frame whose alpha is a block centred on this frame's principal point."""
    alpha = np.zeros((SIZE, SIZE), np.uint8)
    alpha[cy - 3:cy + 3, cx - 3:cx + 3] = 255
    rgba = np.zeros((SIZE, SIZE, 4), np.uint8)
    rgba[..., 3] = alpha
    rel = f"realcams/{cam}/frame_{idx:05d}"
    (root / "realcams" / cam).mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(root / (rel + ".png"))
    return {"file_path": rel, "time": idx * 0.1, "transform_matrix": np.eye(4).tolist(),
            "fl_x": 10.0, "fl_y": 10.0, "cx": float(cx), "cy": float(cy), "w": SIZE, "h": SIZE}


def _dataset(tmp_path, frames):
    """frames: {camera: [(frame index, cx, cy), ...]}"""
    records = [_write_frame(tmp_path, cam, i, cx, cy) for cam, rows in frames.items() for i, cx, cy in rows]
    (tmp_path / "transforms_train.json").write_text(json.dumps({"frames": records}))
    return str(tmp_path)


def _keep(root, **kw):
    # one static splat on the optical axis: an OpenGL pose looks down -z, so in front is z < 0
    xyz = np.array([[0.0, 0.0, -2.0]])
    zero = np.zeros((1, 3))
    return bake_sogst.mask_consistency_keep(root, xyz, zero, np.array([0.1]), np.array([5.0]),
                                            np.array([4.0]), downscale=1, time_stride=1, **kw)


def test_per_frame_principal_point(tmp_path):
    # The mask follows a principal point that moves every frame, as a tracked crop does.
    # Reading cx/cy from the camera's first frame would project later frames off the mask.
    rows = [(0, 16, 16), (1, 8, 20), (2, 24, 12)]
    root = _dataset(tmp_path, {"camA": rows, "camB": rows})
    assert _keep(root, outside_frac=0.5).tolist() == [True]


def test_camera_missing_a_timestamp(tmp_path):
    root = _dataset(tmp_path, {"camA": [(0, 16, 16), (1, 16, 16), (2, 16, 16)],
                               "camB": [(0, 16, 16), (2, 16, 16)]})
    assert _keep(root, outside_frac=0.5).tolist() == [True]


def test_splat_outside_the_silhouette_is_dropped(tmp_path):
    # the mask sits away from where the splat projects, in every camera
    rows = [(0, 16, 16), (1, 16, 16)]
    root = _dataset(tmp_path, {"camA": rows, "camB": rows})
    xyz = np.array([[1.5, 0.0, -2.0]])  # projects to u = 16 + 10 * 1.5 / 2 = 23.5: off the block
    zero = np.zeros((1, 3))
    keep = bake_sogst.mask_consistency_keep(root, xyz, zero, np.array([0.05]), np.array([5.0]),
                                            np.array([4.0]), downscale=1, time_stride=1)
    assert keep.tolist() == [False]
