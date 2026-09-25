# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Convert a 4DAnyone result into an OMG4 (D-NeRF-style) training dataset.

WHAT IT CONSUMES
----------------
A 4DAnyone result directory (cameras.json + videos/dense/NN.mp4) and its GVHMR
motion result (motion.json + motion.safetensors). Both live under the 4DAnyone
data root; see that repo's docs/output.md.

WHAT IT PRODUCES
----------------
    out_dir/transforms_train.json   one entry per camera per frame
    out_dir/genNN/frame_NNNNN.png   extracted frames
    out_dir/points3d.ply            SMPL-X vertices as the init point cloud,
                                    WITH a per-point `time` field

CONVENTIONS, WRITTEN DOWN BECAUSE THEY BITE
-------------------------------------------
* 4DAnyone cameras are OpenCV (x right, y down, z forward) camera_to_world.
  The OMG4 reader expects OpenGL/Blender c2w and converts by negating the Y/Z
  axis columns, so this script negates those columns going in -- the involution
  lands the reader back on the OpenCV matrix.
* The world frame is 4DAnyone's canonical human world (y up, origin at the
  subject's initial root on the ground), NOT the capture rig's world. A dataset
  from this script is self-consistent for standalone training; mixing with real
  rig cameras needs a rig-alignment similarity first.
* The init cloud is the deferred "time-stamped init" idea made real: SMPL-X
  vertices sampled per frame, each point carrying its source frame's time.
  fetchPly in the trainer reads `time` and create_from_pcd uses it verbatim,
  replacing the random time assignment. Without points3d.ply the trainer
  silently substitutes a grey cube (see split_4d_dataset.py).
* 4DAnyone backgrounds are a hallucinated grey studio. This script does NOT
  mask; train either with a matting pass (BiRefNet) added on top, or accept the
  studio as scene content for a first bring-up.

conda env: cumuli (smplx + safetensors added alongside the torch pin this
env already carries; see environment.yml).

Usage:
    conda run -n cumuli python scripts/fdanyone_to_omg4.py \\
        --result_dir  ~/Dev/github/4DAnyone/data/fdanyone/<clip> \\
        --motion_dir  ~/Dev/github/4DAnyone/data/gvhmr/results/<clip> \\
        --smplx_model ~/Dev/models/4DAnyone/body_models/smplx/SMPLX_NEUTRAL.npz \\
        --out_dir     /media/IronWolf/Datasets/fdanyone_omg4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def opencv_c2w_to_opengl(c2w: np.ndarray) -> np.ndarray:
    out = np.array(c2w, dtype=np.float64)
    out[:3, 1:3] *= -1
    return out


def extract_frames(video: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    existing = len(list(dest.glob("frame_*.png")))
    if existing > 0:
        return existing
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         str(dest / "frame_%05d.png")],
        check=True)
    return len(list(dest.glob("frame_*.png")))


def smplx_init_cloud(motion_dir: Path, smplx_model: Path, stride: int,
                     verts_per_frame: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """SMPL-X vertices across the clip, each stamped with its frame time."""
    import torch
    from safetensors.torch import load_file

    import smplx

    meta = json.loads((motion_dir / "motion.json").read_text())
    times = np.asarray(meta["frame_timestamps_sec"], dtype=np.float64)
    tensors = load_file(str(motion_dir / "motion.safetensors"))

    model = smplx.SMPLX(
        model_path=str(smplx_model.parent), gender="neutral", use_pca=False,
        num_betas=10, ext="npz")
    frames = range(0, len(times), stride)
    rng = np.random.default_rng(seed)
    points, point_times = [], []
    with torch.no_grad():
        for i in frames:
            out = model(
                betas=tensors["smpl_params_global.betas"][i:i + 1],
                body_pose=tensors["smpl_params_global.body_pose"][i:i + 1],
                global_orient=tensors["smpl_params_global.global_orient"][i:i + 1],
                transl=tensors["smpl_params_global.transl"][i:i + 1])
            verts = out.vertices[0].numpy()
            keep = rng.choice(len(verts), size=min(verts_per_frame, len(verts)),
                              replace=False)
            points.append(verts[keep])
            point_times.append(np.full(len(keep), times[i]))
    return np.concatenate(points), np.concatenate(point_times)


def write_ply_with_time(path: Path, xyz: np.ndarray, times: np.ndarray,
                        rgb: tuple[int, int, int] = (80, 90, 110)) -> None:
    """Ascii-free binary ply with the optional per-vertex time field."""
    from plyfile import PlyData, PlyElement

    n = len(xyz)
    data = np.empty(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                              ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                              ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                              ("time", "f4")])
    data["x"], data["y"], data["z"] = xyz.T.astype(np.float32)
    data["nx"] = data["ny"] = data["nz"] = 0.0
    data["red"], data["green"], data["blue"] = rgb
    data["time"] = times.astype(np.float32)
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result_dir", required=True, type=Path)
    ap.add_argument("--motion_dir", required=True, type=Path)
    ap.add_argument("--smplx_model", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--init_stride", type=int, default=4,
                    help="sample the SMPL-X mesh every Nth frame for the init cloud")
    ap.add_argument("--init_verts_per_frame", type=int, default=3000)
    args = ap.parse_args()

    cams = json.loads((args.result_dir / "cameras.json").read_text())
    meta = json.loads((args.motion_dir / "motion.json").read_text())
    times = meta["frame_timestamps_sec"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_out = []
    for cam in cams["cameras"]:
        cid = cam["camera_id"]
        tag = f"gen{cid:02d}"
        video = args.result_dir / "videos" / "dense" / f"{cid:02d}.mp4"
        if not video.exists():
            print(f"WARNING: {video} missing, skipping camera {cid}", file=sys.stderr)
            continue
        n = extract_frames(video, args.out_dir / tag)
        c2w = opencv_c2w_to_opengl(np.asarray(cam["camera_to_world"]))
        K = np.asarray(cam["K"])
        for f in range(1, n + 1):
            frames_out.append({
                "file_path": f"{tag}/frame_{f:05d}",
                "transform_matrix": c2w.tolist(),
                "fl_x": float(K[0, 0]), "fl_y": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "w": int(cam["image_width"]), "h": int(cam["image_height"]),
                "time": float(times[min(f - 1, len(times) - 1)]),
                "camera_label": tag,
            })
        print(f"{tag}: {n} frames")

    (args.out_dir / "transforms_train.json").write_text(json.dumps({
        "w": int(cams["cameras"][0]["image_width"]),
        "h": int(cams["cameras"][0]["image_height"]),
        "frames": frames_out}, indent=1))

    xyz, ptimes = smplx_init_cloud(args.motion_dir, args.smplx_model,
                                   args.init_stride, args.init_verts_per_frame)
    write_ply_with_time(args.out_dir / "points3d.ply", xyz, ptimes)
    print(f"wrote {len(frames_out)} view entries and a {len(xyz):,}-point "
          f"time-stamped init cloud to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
