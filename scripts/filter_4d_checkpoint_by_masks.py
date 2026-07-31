#!/usr/bin/env python3
"""
filter_4d_checkpoint_by_masks.py

Multi-view mask-consistency filter for a trained 4D checkpoint -- the temporal
counterpart of filter_splat_by_masks.py. Runs after the refit train, before
xz_to_omg4.py, and emits the boolean keep-mask that exporter's
--extra_keep_mask consumes.

Why a separate script: filter_splat_by_masks.py tests a static .ply against one
mask per camera. A 4D checkpoint's Gaussians each carry their own temporal
centre, so each one must be tested against the masks of the FRAME IT LIVES AT
-- projecting every Gaussian into every frame's mask would fail everything that
moves. Each Gaussian is projected at its own t_center into every real camera,
against that camera's mask at the nearest captured frame.

This is a direct geometric test against ground truth, not a heuristic: a
Gaussian landing outside the subject silhouette in most cameras that resolve
it is not subject geometry. Measured on a real refit, Gaussians independently
flagged by a brightness/opacity/size heuristic had a median 83% outside-mask
rate against 0% for the general population -- a clean separation.

Same limitation as the static version: Gaussians hiding BEHIND the subject
inside the silhouette frustum project inside the mask everywhere and are not
caught. Gaussians no camera resolves are kept rather than guessed at.

conda env: omg4 (torch, to read the checkpoint) + numpy + PIL.

Usage:
    python3 filter_4d_checkpoint_by_masks.py \\
        --checkpoint /path/to/output/<model>/chkpnt30000.pth \\
        --transforms /path/to/dataset_4dgs/transforms_train.json \\
        --sequence_root /path/to/pipeline_run \\
        --out_npy /path/to/keep_mask.npy \\
        --fps 29.97 [--n_frames 90] \\
        [--masks_subdir fmasks_clean] [--outside_frac_thresh 0.5]

Output:
    out_npy, a boolean array aligned with the checkpoint's original Gaussian
    order, ready for:
        xz_to_omg4.py --extra_keep_mask <out_npy>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from image_formats import SUPPORTED_IMAGE_EXTS

MASK_FOREGROUND_THRESHOLD = 127  # 8-bit grayscale mask: values above this count as foreground
MIN_CAMERA_Z = 0.01              # reject projections closer than this to the camera plane


def opengl_c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """nerfstudio/OpenGL c2w -> COLMAP-convention w2c, matching the conversion
    documented in build_colmap_sparse.py (negate Y/Z axis columns, invert)."""
    m = c2w.copy()
    m[:3, 1:3] *= -1
    return np.linalg.inv(m)


def unique_cameras(transforms_path: Path) -> list:
    """One entry per physical camera. The rig is static, so any frame of a given
    camera carries that camera's pose and intrinsics."""
    data = json.loads(transforms_path.read_text())
    cameras = {}
    for frame in data["frames"]:
        label = str(frame.get("camera_label") or Path(frame["file_path"]).parent.name)
        if label in cameras:
            continue
        cameras[label] = {
            "label": label,
            "w2c": opengl_c2w_to_w2c(np.array(frame["transform_matrix"], dtype=np.float64)),
            "fl_x": frame["fl_x"], "fl_y": frame["fl_y"],
            "cx": frame["cx"], "cy": frame["cy"],
        }
    return [cameras[label] for label in sorted(cameras)]


def find_mask(masks_dir: Path, camera_label: str) -> Path | None:
    for ext in SUPPORTED_IMAGE_EXTS:
        candidate = masks_dir / f"{camera_label}{ext}"
        if candidate.exists():
            return candidate
    return None


def frame_mask_dirs(sequence_root: Path, masks_subdir: str) -> list:
    """Per-frame mask directories in frame order (render_frame_sequence.py
    layout: <sequence_root>/frame_NNNN/<masks_subdir>/<camera_label>.png)."""
    return [d / masks_subdir for d in sorted(p for p in sequence_root.iterdir() if p.is_dir())
            if (d / masks_subdir).is_dir()]


def load_checkpoint_geometry(checkpoint_path: Path) -> tuple:
    """(xyz, t_center, opacity) from a 4D checkpoint. The tuple layout mirrors
    xz_to_omg4.convert_from_checkpoint (xz_to_omg4.py:496), which is the
    authority on this format; only the three fields needed here are unpacked."""
    import torch

    from xz_to_omg4 import to_numpy

    model_args, _first_iter = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    xyz_t, opacity_t, t_center_t = model_args[1], model_args[6], model_args[13]
    xyz = to_numpy(xyz_t)
    t_center = to_numpy(t_center_t)[:, 0]
    opacity = 1.0 / (1.0 + np.exp(-to_numpy(opacity_t)[:, 0].astype(np.float64)))
    return xyz, t_center, opacity


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path, help="4D trainer checkpoint (chkpntNNNNN.pth)")
    ap.add_argument("--transforms", required=True, type=Path,
                    help="the training dataset's transforms_train.json -- supplies real camera poses "
                         "and intrinsics (one frame per camera is used; the rig is static)")
    ap.add_argument("--sequence_root", required=True, type=Path,
                    help="per-frame run root holding frame_NNNN/<masks_subdir>/<camera_label>.png")
    ap.add_argument("--out_npy", required=True, type=Path)
    ap.add_argument("--fps", type=float, required=True, help="capture frame rate, to map t_center to a frame")
    ap.add_argument("--n_frames", type=int, default=None,
                    help="clamp frame indices to this many frames (default: however many "
                         "mask directories --sequence_root contains)")
    ap.add_argument("--masks_subdir", default="fmasks_clean",
                    help="mask directory name inside each frame dir (default fmasks_clean)")
    ap.add_argument("--outside_frac_thresh", type=float, default=0.5,
                    help="drop a Gaussian if at least this fraction of resolving views "
                         "put it outside the mask (default 0.5)")
    ap.add_argument("--opacity_check_thresh", type=float, default=0.1,
                    help="only test Gaussians with sigmoid(opacity) above this (default 0.1)")
    args = ap.parse_args()

    try:
        cameras = unique_cameras(args.transforms)
        mask_dirs = frame_mask_dirs(args.sequence_root, args.masks_subdir)
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not mask_dirs:
        print(f"Error: no frame_*/{args.masks_subdir}/ directories under {args.sequence_root}",
              file=sys.stderr)
        return 1
    n_frames = args.n_frames or len(mask_dirs)
    print(f"{len(cameras)} cameras, {len(mask_dirs)} frame mask directories, clamping to {n_frames} frames")

    xyz, t_center, opacity = load_checkpoint_geometry(args.checkpoint)
    n = len(xyz)
    keep = np.ones(n, dtype=bool)
    idxs = np.where((opacity > args.opacity_check_thresh) & np.isfinite(xyz).all(axis=1))[0]
    pts = xyz[idxs].astype(np.float64)
    frames_of = np.clip(np.round(t_center[idxs] * args.fps).astype(int), 0, n_frames - 1)
    print(f"{n:,} Gaussians; testing {len(idxs):,} with opacity > {args.opacity_check_thresh}")

    outside = np.zeros(len(idxs), dtype=np.int32)
    checked = np.zeros(len(idxs), dtype=np.int32)
    views_used = 0
    for camera in cameras:
        cam_pts = (camera["w2c"][:3, :3] @ pts.T).T + camera["w2c"][:3, 3][None, :]
        z = cam_pts[:, 2]
        px = camera["fl_x"] * cam_pts[:, 0] / np.maximum(z, 1e-9) + camera["cx"]
        py = camera["fl_y"] * cam_pts[:, 1] / np.maximum(z, 1e-9) + camera["cy"]
        depth_ok = z > MIN_CAMERA_Z

        for frame_number in np.unique(frames_of):
            if frame_number >= len(mask_dirs):
                continue
            mask_path = find_mask(mask_dirs[frame_number], camera["label"])
            if mask_path is None:
                continue
            mask = np.asarray(Image.open(mask_path)) > MASK_FOREGROUND_THRESHOLD
            if mask.ndim == 3:
                mask = mask[..., 0]
            height, width = mask.shape

            selected = np.where(depth_ok & (frames_of == frame_number)
                                & (px >= 0) & (px < width) & (py >= 0) & (py < height))[0]
            if not len(selected):
                continue
            inside = mask[py[selected].astype(np.int32), px[selected].astype(np.int32)]
            checked[selected] += 1
            outside[selected[~inside]] += 1
            views_used += 1
        print(f"  camera {camera['label']} done")

    if views_used == 0:
        print("Error: no camera/frame pair had a usable mask; nothing to test against", file=sys.stderr)
        return 1

    resolved = checked > 0
    frac = np.zeros(len(idxs))
    frac[resolved] = outside[resolved] / checked[resolved]
    drop = resolved & (frac >= args.outside_frac_thresh)
    keep[idxs[drop]] = False
    print(f"mask test: {views_used} camera/frame views, {int(resolved.sum()):,} Gaussians resolved, "
          f"{int(drop.sum()):,} dropped (outside-fraction >= {args.outside_frac_thresh})")
    print(f"keep: {int(keep.sum()):,} / {n:,} ({100 * keep.sum() / n:.1f}%)")

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, keep)
    print(f"wrote {args.out_npy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
