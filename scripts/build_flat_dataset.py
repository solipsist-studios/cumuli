#!/usr/bin/env python3
"""
build_flat_dataset.py

Converts HLOC's output (transforms_multiframe.json + per-camera
Camera_<id>/0000.<ext> folders from run_hloc.py) into the flat,
sequentially-2-digit-labeled layout that Diffuman4D's own preprocessing
scripts (remove_background.py, predict_keypoints.py) and the diffusion
model itself expect.

IMPORTANT: Diffuman4D's camera_parser.py looks cameras up in transforms.json
by the literal `camera_label` string, and triangulate_skeleton.py's
--spa_labels/--spa_label_range args format labels as zero-padded 2-digit
numbers ("00", "01", ... "47") -- matching its own DNA-Rendering
convention. So camera_label must be a plain 2-digit index, NOT the raw
"Camera_undistorted_0001" style label HLOC produces. This script is the
single place that mapping happens; every later stage inherits the 2-digit
labels from here.

Usage:
    python3 build_flat_dataset.py \\
        --transforms /path/to/solipsist_out/transforms_multiframe.json \\
        --undistorted_dir /path/to/undistorted \\
        --out_images_flat /path/to/images_flat \\
        --out_transforms /path/to/transforms.json \\
        [--image_ext .jpg] [--out_image_ext .png]

Output:
    out_images_flat/00.png, 01.png, ... (real cameras, sorted by original
        camera id, sequential 2-digit label)
    out_transforms -- copy of the input transforms.json with camera_label
        rewritten to "00".."NN" and file_path pointing at the flat images
        (images_flat/<label><out_image_ext>), for use by every later stage.
    A camera_label_map.json alongside out_transforms recording the
    original HLOC camera_label each 2-digit index came from, so you can
    always trace a label back to a physical camera.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from image_formats import SUPPORTED_IMAGE_EXTS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transforms", required=True, type=Path)
    parser.add_argument("--undistorted_dir", required=True, type=Path,
                         help="Per-camera dir from run_hloc.py, e.g. undistorted/Camera_0001/0000.jpg")
    parser.add_argument("--out_images_flat", required=True, type=Path)
    parser.add_argument("--out_transforms", required=True, type=Path)
    parser.add_argument("--image_ext", default=".jpg", choices=SUPPORTED_IMAGE_EXTS,
                         help="Extension of source frames under undistorted_dir")
    parser.add_argument("--out_image_ext", default=".png", choices=SUPPORTED_IMAGE_EXTS,
                         help="Extension to save flat images as")
    args = parser.parse_args()

    try:
        with open(args.transforms) as f:
            tf = json.load(f)
        frames = sorted(tf["frames"], key=lambda fr: fr["camera_label"])
    except json.JSONDecodeError as e:
        print(f"Error: {args.transforms} is not valid JSON: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: {args.transforms} is missing expected key {e}")
        sys.exit(1)

    args.out_images_flat.mkdir(parents=True, exist_ok=True)
    args.out_transforms.parent.mkdir(parents=True, exist_ok=True)

    new_frames = []
    label_map = {}
    skipped = []

    for frame in frames:
        old_label = frame["camera_label"]  # e.g. "Camera_undistorted_0001"
        cam_id = old_label.replace("Camera_undistorted_", "").replace("Camera_", "")

        src_img = args.undistorted_dir / f"Camera_{cam_id}" / f"0000{args.image_ext}"
        if not src_img.is_file():
            print(f"  WARNING: no image found for {old_label} at {src_img}, skipping")
            skipped.append(old_label)
            continue

        # Label from a dense counter of cameras actually written, not the
        # pre-filter loop index -- downstream Diffuman4D tools consume
        # --spa_label_range as a contiguous "00".."NN" range, so a skipped
        # camera must not leave a gap in the numbering.
        new_label = f"{len(new_frames):02d}"
        dst_img = args.out_images_flat / f"{new_label}{args.out_image_ext}"

        try:
            if args.out_image_ext.lower() == args.image_ext.lower():
                shutil.copy(src_img, dst_img)
            else:
                Image.open(src_img).convert("RGB").save(dst_img)
        except (OSError, UnidentifiedImageError) as e:
            print(f"  WARNING: could not read/convert image for {old_label} at {src_img} ({e}), skipping")
            skipped.append(old_label)
            continue

        label_map[new_label] = old_label
        print(f"  {old_label} (cam {cam_id}) -> {new_label}{args.out_image_ext}")

        new_frame = dict(frame)
        new_frame["camera_label"] = new_label
        new_frame["file_path"] = f"images_flat/{new_label}{args.out_image_ext}"
        new_frames.append(new_frame)

    if skipped:
        print(f"\nERROR: {len(skipped)} camera(s) missing an undistorted image, skipped: {skipped}")
        print("Not writing out_transforms/label_map -- a resumed run must not silently "
              "reuse a partial-camera flat dataset.")
        sys.exit(1)

    tf["frames"] = new_frames
    with open(args.out_transforms, "w") as f:
        json.dump(tf, f, indent=2)

    label_map_path = args.out_transforms.parent / "camera_label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nDone. {len(new_frames)} cameras written.")
    print(f"Flat images: {args.out_images_flat}")
    print(f"Transforms:  {args.out_transforms}")
    print(f"Label map:   {label_map_path}")


if __name__ == "__main__":
    main()
