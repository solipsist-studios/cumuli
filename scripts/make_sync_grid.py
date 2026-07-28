#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

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
    for img_path in image_files:
        img = Image.open(img_path)
        aspect = img.height / img.width
        thumb_height = int(thumb_width * aspect)
        thumb = img.convert("RGB").resize((thumb_width, thumb_height), Image.LANCZOS)
        thumbs.append((img_path.stem, thumb))

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


if __name__ == "__main__":
    main()
