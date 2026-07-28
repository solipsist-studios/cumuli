#!/usr/bin/env python3
"""
make_sync_grid.py

Combines the individual sync-check frame .jpg files into a single grid
image so you can visually compare all cameras at once, and confirm
extract_synced_frames.py pulled the same real-world moment from every
camera before moving on.

Usage:
    python3 make_sync_grid.py /path/to/sync_check_frames /path/to/output_grid.jpg [thumb_width]

    thumb_width (optional, default 480) -- width in pixels to resize each
    frame to before placing in the grid.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from image_formats import SUPPORTED_IMAGE_EXTS


def main():
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: python3 make_sync_grid.py "
            "/path/to/sync_check_frames /path/to/output_grid.jpg [thumb_width]"
        )
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    thumb_width = int(sys.argv[3]) if len(sys.argv) == 4 else 480

    if thumb_width <= 0:
        # Without this check, a degenerate thumb_width makes every image's
        # own resize() raise ValueError -- caught below as "could not be
        # read as an image", which would misattribute the real problem
        # (the thumb_width argument) to the image files themselves.
        print(f"Error: thumb_width must be > 0, got {thumb_width}")
        sys.exit(1)

    if not frames_dir.is_dir():
        print(f"Error: {frames_dir} is not a directory")
        sys.exit(1)

    image_files = sorted(
        p for ext in SUPPORTED_IMAGE_EXTS for p in frames_dir.glob(f"*{ext}")
    )
    if not image_files:
        print(f"Error: no {'/'.join(SUPPORTED_IMAGE_EXTS)} files found in {frames_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} frames. Building grid...")

    thumbs = []
    failed = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            aspect = img.height / img.width
            thumb_height = int(thumb_width * aspect)
            thumb = img.convert("RGB").resize((thumb_width, thumb_height), Image.LANCZOS)
        except (OSError, ValueError, ZeroDivisionError) as e:
            print(f"  WARNING: {img_path.name} could not be read as an image ({e}), skipping")
            failed.append(img_path.name)
            continue
        thumbs.append((img_path.stem, thumb))

    if not thumbs:
        print("Error: no readable images remained after skipping unreadable files")
        sys.exit(1)

    n = len(thumbs)
    cols = 4
    rows = (n + cols - 1) // cols

    label_height = 30
    cell_w = thumb_width
    cell_h = max(thumb.height for _, thumb in thumbs) + label_height

    grid_img = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for idx, (name, thumb) in enumerate(thumbs):
        col = idx % cols
        row = idx // cols
        x = col * cell_w
        y = row * cell_h

        draw.text((x + 8, y + 5), name, fill=(255, 255, 255), font=font)
        grid_img.paste(thumb, (x, y + label_height))

    grid_img.save(output_path, quality=90)
    print(f"Saved grid image to {output_path}")
    if failed:
        print(f"Skipped unreadable images: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
