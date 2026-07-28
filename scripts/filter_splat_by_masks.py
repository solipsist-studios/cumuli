#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
filter_splat_by_masks.py

Multi-view mask-consistency filter for a trained splat: projects every
Gaussian's center into every camera and drops Gaussians that land OUTSIDE
the subject mask in most of the cameras that can see them. Runs after
train_brush.py (or any trainer that emits a standard 3DGS .ply), using the
same cleaned masks the training consumed.

Why this exists: masked training does not prevent floaters. Real runs
accumulate substantial non-subject junk that alpha supervision never
removes -- warm-started per-frame sequences have measured 30-56% of
splats failing this check, and even clean cold-start runs carry a few
percent. The junk both occludes the subject from novel views and corrupts
anything computed from the splat (e.g. a subject centroid used to aim
novel-view render cameras). Unlike density/outlier heuristics (which
misclassify embedded high-opacity blobs) this is a direct geometric test
against ground truth: a Gaussian that projects outside the subject
silhouette in most views that resolve it is not subject geometry.

Known limitation: splats hiding BEHIND the subject inside the silhouette
frustum project inside the mask in most views and are NOT caught (depth
is unobservable from silhouettes alone). If downstream consumers compute
statistics from the splat (centroids etc.), bound them spatially as well
-- see --subject_radius / --subject_anchor_ply.

Only Gaussians with opacity above --opacity_check_thresh are tested
(below that they are visually negligible either way), and only cameras
whose frustum actually contains the projection count toward the vote;
Gaussians no camera resolves are kept rather than guessed at.

conda env: none (numpy + PIL + plyfile).

Usage:
    python3 filter_splat_by_masks.py \\
        --splat_ply /path/to/export_30000.ply \\
        --transforms /path/to/transforms_refined.json \\
        --masks_dir /path/to/fmasks_clean \\
        --out_ply /path/to/export_30000_maskfilt.ply \\
        [--outside_frac_thresh 0.5] [--opacity_check_thresh 0.1] \\
        [--subject_anchor_ply /path/to/poses_pcd_fullres/<tem>.ply \\
         --subject_radius 3.0]

Output:
    out_ply with the failing Gaussians removed (all PLY fields preserved),
    plus a printed keep/drop summary. With --report_only, prints the
    summary without writing anything.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
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


def find_mask(masks_dir: Path, camera_label: str) -> Path | None:
    for ext in SUPPORTED_IMAGE_EXTS:
        p = masks_dir / f"{camera_label}{ext}"
        if p.exists():
            return p
    return None


def load_subject_anchor(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1)
    return np.median(pts, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splat_ply", required=True, type=Path)
    ap.add_argument("--transforms", required=True, type=Path,
                    help="nerfstudio transforms.json with per-frame intrinsics and camera_label")
    ap.add_argument("--masks_dir", required=True, type=Path,
                    help="cleaned masks, one <camera_label>.<ext> per camera (clean_masks.py output)")
    ap.add_argument("--out_ply", type=Path, default=None,
                    help="output path (default: <splat_ply stem>_maskfilt.ply alongside the input)")
    ap.add_argument("--outside_frac_thresh", type=float, default=0.5,
                    help="drop a Gaussian if at least this fraction of resolving cameras "
                         "put it outside the mask (default 0.5)")
    ap.add_argument("--opacity_check_thresh", type=float, default=0.1,
                    help="only test Gaussians with sigmoid(opacity) above this (default 0.1)")
    ap.add_argument("--subject_anchor_ply", type=Path, default=None,
                    help="optional subject point cloud (e.g. poses_pcd_fullres/<tem>.ply); "
                         "with --subject_radius, also drops Gaussians farther than the "
                         "radius from its median -- catches behind-the-subject junk the "
                         "silhouette test cannot see")
    ap.add_argument("--subject_radius", type=float, default=None,
                    help="radius (world units) around the subject anchor; requires --subject_anchor_ply")
    ap.add_argument("--report_only", action="store_true",
                    help="print the keep/drop summary without writing the output ply")
    args = ap.parse_args()

    if (args.subject_radius is None) != (args.subject_anchor_ply is None):
        ap.error("--subject_anchor_ply and --subject_radius must be used together")

    transforms = json.loads(args.transforms.read_text())
    frames = transforms["frames"]

    ply = PlyData.read(str(args.splat_ply))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
    n = len(xyz)

    keep = np.ones(n, dtype=bool)
    check = (opacity > args.opacity_check_thresh) & np.isfinite(xyz).all(axis=1)
    idxs = np.where(check)[0]
    pts = xyz[idxs]
    print(f"{n:,} Gaussians; testing {len(idxs):,} with opacity > {args.opacity_check_thresh}")

    outside = np.zeros(len(idxs), dtype=np.int32)
    checked = np.zeros(len(idxs), dtype=np.int32)
    cameras_used = 0
    for fr in frames:
        label = str(fr.get("camera_label", ""))
        mask_path = find_mask(args.masks_dir, label)
        if mask_path is None:
            print(f"  camera {label}: no mask in {args.masks_dir}, skipped")
            continue
        mask = np.asarray(Image.open(mask_path)) > MASK_FOREGROUND_THRESHOLD
        if mask.ndim == 3:
            mask = mask[..., 0]
        h, w = mask.shape

        w2c = opengl_c2w_to_w2c(np.array(fr["transform_matrix"], dtype=np.float64))
        cam_pts = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3][None, :]
        z = cam_pts[:, 2]
        px = fr["fl_x"] * cam_pts[:, 0] / np.maximum(z, 1e-9) + fr["cx"]
        py = fr["fl_y"] * cam_pts[:, 1] / np.maximum(z, 1e-9) + fr["cy"]
        in_frustum = (z > MIN_CAMERA_Z) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        sel = np.where(in_frustum)[0]
        inside = mask[py[sel].astype(np.int32), px[sel].astype(np.int32)]
        checked[sel] += 1
        outside[sel[~inside]] += 1
        cameras_used += 1

    if cameras_used == 0:
        print("ERROR: no camera had a usable mask; nothing to test against", file=sys.stderr)
        return 1

    resolved = checked > 0
    frac = np.zeros(len(idxs))
    frac[resolved] = outside[resolved] / checked[resolved]
    drop = resolved & (frac >= args.outside_frac_thresh)
    keep[idxs[drop]] = False
    print(f"mask test: {cameras_used} cameras, {int(resolved.sum()):,} Gaussians resolved, "
          f"{int(drop.sum()):,} dropped (outside-fraction >= {args.outside_frac_thresh})")

    if args.subject_anchor_ply is not None:
        anchor = load_subject_anchor(args.subject_anchor_ply)
        far = np.linalg.norm(xyz - anchor[None, :], axis=1) > args.subject_radius
        n_far = int((keep & far).sum())
        keep &= ~far
        print(f"subject-radius test: anchor {np.round(anchor, 3).tolist()}, "
              f"radius {args.subject_radius}: {n_far:,} additional Gaussians dropped")

    print(f"keep: {int(keep.sum()):,} / {n:,} ({100 * keep.sum() / n:.1f}%)")

    if args.report_only:
        return 0

    out_ply = args.out_ply or args.splat_ply.with_name(args.splat_ply.stem + "_maskfilt.ply")
    filtered = v.data[keep]
    PlyData([PlyElement.describe(filtered, "vertex")], text=False).write(str(out_ply))
    print(f"wrote {out_ply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
