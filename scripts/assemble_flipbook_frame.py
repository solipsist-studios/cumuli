#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
assemble_flipbook_frame.py - Lay out ONE extracted instant as a flipbook
frame dir for build_flipbook_4dgs_dataset.py.

The 4D training path needs a frame-major layout: frame_NNNN/ dirs, each
carrying transforms.json + images_flat/<label>.png. The unified pipeline
produces those pieces for its single production instant. This script maps
one additional instant into that shape:

  - Undistorted per-camera images arrive named by physical camera id
    (<camera_id>.png). The 2-digit label convention comes from the poses
    stage. camera_label_map.json translates between the two. Each image
    is hardlinked (copy fallback) to images_flat/<label>.png.
  - The rig is static, so the production instant's flat transforms.json
    applies to every instant verbatim. Its file_path entries are already
    "images_flat/<label>.png", relative to the frame dir. It is copied
    in unchanged.

Masks are NOT handled here. The orchestrator runs the standard mask chain
(generate_masks -> keypoints -> split -> clean_masks) on each frame dir
after this relayout, so the frame gains fmasks_clean/ the same way the
production instant does.

Usage:
    python assemble_flipbook_frame.py \\
        --undist_dir <run>/train_seq_undistorted/f3 \\
        --label_map <run>/camera_label_map.json \\
        --flat_transforms <run>/transforms.json \\
        --out_frame_dir <run>/flipbook_src/frame_0003 \\
        --image_ext .png
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def load_label_map(path: Path) -> dict:
    """camera_label_map.json maps 2-digit label -> the original HLOC
    camera_label (for example "Camera_undistorted_0001"). Returns
    {camera_id: label}, deriving the physical camera id with the same
    prefix-stripping rule build_flat_dataset.py uses to create the map."""
    raw = json.loads(path.read_text())
    label_by_id = {}
    for label, old_label in raw.items():
        cam_id = str(old_label).replace("Camera_undistorted_", "").replace("Camera_", "")
        label_by_id[cam_id] = label
    return label_by_id


def place(src: Path, dst: Path):
    """Hardlink src to dst, falling back to a copy across filesystems."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--undist_dir", required=True, type=Path,
                    help="one instant's undistorted images, named <camera_id><ext>")
    ap.add_argument("--label_map", required=True, type=Path,
                    help="camera_label_map.json from the poses stage")
    ap.add_argument("--flat_transforms", required=True, type=Path,
                    help="the production instant's flat transforms.json (static rig)")
    ap.add_argument("--out_frame_dir", required=True, type=Path)
    ap.add_argument("--image_ext", default=".png")
    args = ap.parse_args()

    label_by_id = load_label_map(args.label_map)
    transforms = json.loads(args.flat_transforms.read_text())
    wanted_labels = set()
    for frame in transforms.get("frames", []):
        label = frame.get("camera_label")
        if label is None:
            fp = frame.get("file_path", "")
            label = Path(fp).stem
        wanted_labels.add(str(label))

    images_out = args.out_frame_dir / "images_flat"
    placed = {}
    for src in sorted(args.undist_dir.iterdir()):
        if src.suffix.lower() != args.image_ext.lower():
            continue
        cam_id = src.stem
        label = label_by_id.get(cam_id)
        if label is None:
            print(f"[warn] {src.name}: camera id {cam_id!r} not in label map -- skipped",
                  file=sys.stderr)
            continue
        place(src, images_out / f"{label}{args.image_ext}")
        placed[label] = src.name

    missing = sorted(wanted_labels - set(placed))
    if missing:
        raise SystemExit(
            f"{args.undist_dir}: transforms expects labels {missing} but no "
            f"matching camera image was placed. The rig must be complete in "
            f"every instant -- a camera missing from one frame poisons the "
            f"whole training window.")
    if not placed:
        raise SystemExit(f"{args.undist_dir}: no images matched {args.image_ext}")

    args.out_frame_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.flat_transforms, args.out_frame_dir / "transforms.json")
    print(f"frame assembled: {len(placed)} cameras -> {args.out_frame_dir}")


if __name__ == "__main__":
    main()
