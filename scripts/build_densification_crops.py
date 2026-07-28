#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
build_densification_crops.py

Appends keypoint-guided head (and optionally arm) crop views to an existing
COLMAP training set (build_colmap_sparse.py output): each crop is the same
real photo at a tighter framing, expressed as an extra PINHOLE camera with a
principal-point-shifted copy of its parent camera's intrinsics -- same focal,
cx/cy offset by the crop origin. Real pixels, zero hallucination.

Why: with a small rig (~12 cameras), the face occupies so few training
pixels that the splat's head reconstructs as a soft, barely-identifiable
blob, and fast-moving extended limbs come out truncated -- there aren't
enough rays concentrating supervision there. Adding dedicated high-zoom
crop views of the head/arms as extra training cameras concentrates
supervision on exactly that geometry. On real runs this was the single
biggest identity improvement measured (head went from wispy artifact to a
solid recognizable face), and arm crops restored limbs that silhouette
supervision alone had truncated. Post-hoc cleanup cannot do this: filters
only remove Gaussians, they cannot restore missing geometry.

Crop geometry (validated values, tuned on real captures):
  head: centered on keypoints 0-4 (nose/eyes/ears), side = clip(4.5 x
        keypoint span, --head_min_side, --head_max_side), shifted up 45%
        of the side so hair isn't cropped.
  arms: one crop per arm over the shoulder/elbow/wrist chain, side =
        clip(1.6 x span, --arm_min_side, --arm_max_side).
Keypoint indices assume the goliath308 layout (the pipeline standard;
see CLAUDE.md) -- head 0-4, left arm 5/7/62, right arm 6/8/41.

Masks are baked into the crop's alpha exactly like build_colmap_sparse.py
does for the body views (RGB zeroed where fully transparent -- see
bake_rgba there for the halo rationale). Crops whose mask coverage is
below --min_mask_px are skipped (subject part not visible in that view).

The script APPENDS to sparse/0/cameras.txt and images.txt. It refuses to
run if crop entries are already present (delete <crops_subdir> and re-run
build_colmap_sparse.py for a clean slate) so a re-run cannot silently
double the crop views.

conda env: none (numpy + PIL).

Usage:
    python3 build_densification_crops.py \\
        --dataset_dir /path/to/train_set \\
        --transforms /path/to/transforms.json \\
        --images_dir /path/to/images_flat \\
        --masks_dir /path/to/fmasks_clean \\
        --kp2d_dir /path/to/poses_2d \\
        [--tem_label 000000] [--no_arms]

Output:
    dataset_dir/<crops_subdir>/head_<label>.png (+ larm_/rarm_) RGBA crops,
    appended entries in dataset_dir/sparse/0/{cameras,images}.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from image_formats import SUPPORTED_IMAGE_EXTS

# goliath308 keypoint indices (sapiens keypoints308 _base_ config, dataset "goliath")
HEAD_KPS = [0, 1, 2, 3, 4]        # nose, eyes, ears
LEFT_ARM_KPS = [5, 7, 62]         # left_shoulder, left_elbow, left_wrist
RIGHT_ARM_KPS = [6, 8, 41]        # right_shoulder, right_elbow, right_wrist

KP_CONFIDENCE_THRESHOLD = 0.3
HEAD_SPAN_SCALE = 4.5
HEAD_TOP_BIAS = 0.45              # fraction of side to shift the crop up (keeps hair)
ARM_SPAN_SCALE = 1.6
MASK_FOREGROUND_THRESHOLD = 127


def opengl_c2w_to_colmap_w2c(c2w: np.ndarray):
    """Same conversion as build_colmap_sparse.py (matches Brush's loader)."""
    c2w_cv = c2w.copy()
    c2w_cv[:3, 1] *= -1
    c2w_cv[:3, 2] *= -1
    w2c = np.linalg.inv(c2w_cv)
    return w2c[:3, :3], w2c[:3, 3]


def load_keypoints(kp2d_dir: Path, camera_label: str, tem_label: str):
    path = kp2d_dir / camera_label / f"{tem_label}.json"
    if not path.exists():
        return None, None
    data = json.loads(path.read_text())
    inst = data["instance_info"][0]
    return np.asarray(inst["keypoints"], dtype=np.float64), np.asarray(inst["keypoint_scores"], dtype=np.float64)


def find_image(images_dir: Path, camera_label: str) -> Path | None:
    for ext in SUPPORTED_IMAGE_EXTS:
        p = images_dir / f"{camera_label}{ext}"
        if p.exists():
            return p
    return None


def crop_box(pts: np.ndarray, span_scale: float, min_side: int, max_side: int,
             top_bias: float, img_w: int, img_h: int, min_span: float):
    cx, cy = pts.mean(axis=0)
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), min_span)
    side = int(np.clip(span_scale * span, min_side, max_side))
    left = int(round(cx - side / 2))
    top = int(round(cy - side * (0.5 if top_bias == 0 else top_bias)))
    left = max(0, min(left, img_w - side))
    top = max(0, min(top, img_h - side))
    return left, top, side


def write_crop_rgba(img_arr: np.ndarray, mask_arr: np.ndarray, box, out_path: Path,
                    min_mask_px: int) -> bool:
    left, top, side = box
    crop = img_arr[top:top + side, left:left + side]
    mcrop = mask_arr[top:top + side, left:left + side]
    if (mcrop > MASK_FOREGROUND_THRESHOLD).sum() < min_mask_px:
        return False
    crop = crop.copy()
    crop[mcrop == 0] = 0  # same halo-avoidance as build_colmap_sparse.bake_rgba
    rgba = np.dstack([crop, mcrop])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True, type=Path,
                    help="build_colmap_sparse.py output root (contains sparse/0)")
    ap.add_argument("--transforms", required=True, type=Path)
    ap.add_argument("--images_dir", required=True, type=Path,
                    help="full-resolution source images, <camera_label>.<ext>")
    ap.add_argument("--masks_dir", required=True, type=Path,
                    help="cleaned masks (<camera_label>.png, clean_masks.py output)")
    ap.add_argument("--kp2d_dir", required=True, type=Path,
                    help="per-camera keypoints, poses_2d/<camera_label>/<tem_label>.json layout")
    ap.add_argument("--tem_label", default="000000")
    ap.add_argument("--crops_subdir", default="crops_rgba")
    ap.add_argument("--no_arms", action="store_true", help="head crops only")
    ap.add_argument("--head_min_side", type=int, default=300)
    ap.add_argument("--head_max_side", type=int, default=560)
    ap.add_argument("--arm_min_side", type=int, default=200)
    ap.add_argument("--arm_max_side", type=int, default=700)
    ap.add_argument("--min_mask_px", type=int, default=500,
                    help="skip a crop when fewer foreground mask pixels than this fall inside it")
    args = ap.parse_args()

    sparse_dir = args.dataset_dir / "sparse" / "0"
    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"
    if not cameras_txt.exists() or not images_txt.exists():
        print(f"ERROR: {sparse_dir} is not a built dataset (run build_colmap_sparse.py first)", file=sys.stderr)
        return 1
    if f"{args.crops_subdir}/" in images_txt.read_text():
        print(f"ERROR: {images_txt} already references {args.crops_subdir}/ -- refusing to append "
              f"duplicates. Rebuild the dataset for a clean slate.", file=sys.stderr)
        return 1

    cam_lines = [l for l in cameras_txt.read_text().splitlines() if l.strip() and not l.startswith("#")]
    img_lines = [l for l in images_txt.read_text().splitlines() if l.strip() and not l.startswith("#")]
    next_cam_id = max(int(l.split()[0]) for l in cam_lines) + 1
    # images.txt alternates pose line / points2D line; pose lines have >= 10 fields
    next_img_id = max(int(l.split()[0]) for l in img_lines if len(l.split()) >= 10) + 1

    transforms = json.loads(args.transforms.read_text())
    frames = sorted(transforms["frames"], key=lambda fr: fr["camera_label"])

    arm_specs = [] if args.no_arms else [("larm", LEFT_ARM_KPS), ("rarm", RIGHT_ARM_KPS)]
    new_cam_lines, new_img_lines = [], []
    n_head = n_arm = n_skipped = 0

    for fr in frames:
        label = fr["camera_label"]
        kps, scores = load_keypoints(args.kp2d_dir, label, args.tem_label)
        if kps is None:
            print(f"  camera {label}: no keypoints, skipped")
            n_skipped += 1
            continue
        image_path = find_image(args.images_dir, label)
        mask_path = args.masks_dir / f"{label}.png"
        if image_path is None or not mask_path.exists():
            print(f"  camera {label}: missing image or mask, skipped")
            n_skipped += 1
            continue
        img_arr = np.asarray(Image.open(image_path).convert("RGB"))
        mask_arr = np.asarray(Image.open(mask_path).convert("L"))
        img_h, img_w = img_arr.shape[:2]

        R, t = opengl_c2w_to_colmap_w2c(np.array(fr["transform_matrix"]))
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()

        crops = []
        conf = scores[HEAD_KPS] > KP_CONFIDENCE_THRESHOLD
        if conf.any():
            pts = kps[HEAD_KPS][conf]
            box = crop_box(pts, HEAD_SPAN_SCALE, args.head_min_side, args.head_max_side,
                           HEAD_TOP_BIAS, img_w, img_h, min_span=40)
            crops.append(("head", box))
        for arm_name, arm_kps in arm_specs:
            conf = scores[arm_kps] > KP_CONFIDENCE_THRESHOLD
            if conf.sum() < 2:
                continue
            pts = kps[arm_kps][conf]
            box = crop_box(pts, ARM_SPAN_SCALE, args.arm_min_side, args.arm_max_side,
                           0, img_w, img_h, min_span=60)
            crops.append((arm_name, box))

        for kind, box in crops:
            name = f"{kind}_{label}.png"
            out_path = args.dataset_dir / args.crops_subdir / name
            if not write_crop_rgba(img_arr, mask_arr, box, out_path,
                                   args.min_mask_px if kind == "head" else args.min_mask_px * 3 // 5):
                continue
            left, top, side = box
            new_cam_lines.append(
                f"{next_cam_id} PINHOLE {side} {side} {fr['fl_x']} {fr['fl_y']} "
                f"{fr['cx'] - left} {fr['cy'] - top}")
            new_img_lines.append(
                f"{next_img_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} "
                f"{next_cam_id} {args.crops_subdir}/{name}")
            next_cam_id += 1
            next_img_id += 1
            if kind == "head":
                n_head += 1
            else:
                n_arm += 1

    if not new_cam_lines:
        print("ERROR: no crops produced (no confident keypoints anywhere?)", file=sys.stderr)
        return 1

    with open(cameras_txt, "a") as f:
        f.write("\n".join(new_cam_lines) + "\n")
    with open(images_txt, "a") as f:
        for line in new_img_lines:
            f.write(line + "\n\n")

    print(f"Appended {n_head} head + {n_arm} arm crop views "
          f"({n_skipped} cameras skipped) to {sparse_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
