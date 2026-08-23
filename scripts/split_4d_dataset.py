#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""split_4d_dataset.py - a 4D training set with cameras genuinely held out.

Writes a `transforms_train.json` with named cameras removed and a
`transforms_test.json` containing exactly those cameras, so the trainer's own
eval and any downstream novel-view score are measured on views the model has
never seen.

Why this exists: every measurement of novel-view repair on this project has so
far been taken at a camera that was in the training set. That makes the render
already good at the probe -- around 27 dB on the trained model, and a pure VAE
round trip beats every generative setting -- so the experiment cannot show
whether a repair helps in the regime it targets, which is directions the rig
never covered. A held-out probe is the precondition for any of those numbers
meaning anything.

Hold out the probe's whole stereo pair. Rigs built from stereo pairs put their
two halves under a degree apart, so excluding one half leaves its twin supplying
almost the same view and the holdout is not held out in any useful sense.

Images are not copied. The reader resolves each frame's `file_path` relative to
the dataset root, so the per-camera image directories are symlinked and the split
costs kilobytes rather than gigabytes.

conda env: cumuli (stdlib + numpy).

Usage:
    python scripts/split_4d_dataset.py \\
        --transforms /path/to/omg4_full4d/transforms_train.json \\
        --out_dir /path/to/heldout_dataset \\
        --exclude cam03 --exclude cam04 \\
        --test_camera cam04 \\
        [--frames 1-90]

Output:
    out_dir/transforms_train.json   every camera except --exclude
    out_dir/transforms_test.json    only --test_camera
    out_dir/<camera>/               symlinks to the source image directories
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from rig_geometry import camera_label_from_path, trailing_number


def resolve_source(root: Path, file_path: str) -> Path | None:
    """The image behind an extensionless transforms `file_path`."""
    direct = root / file_path
    if direct.is_file():
        return direct.resolve()
    for extension in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = root / f"{file_path}{extension}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def label_of(entry: dict) -> str:
    """The camera a transforms entry belongs to."""
    return str(entry.get("camera_label", "")) or camera_label_from_path(
        str(entry.get("file_path", "")))


def write_rgba(source: Path, destination: Path, background_level: int = 8) -> None:
    """Copy an image with its subject silhouette in the alpha channel.

    bake_sogst's lifetime mask-consistency filter reads its masks from the
    training frames' alpha, so a dataset of RGB-on-black frames gives it nothing
    to test against. Captures whose background is already removed carry the
    silhouette implicitly, and this makes it explicit. The threshold is small but
    non-zero because JPEG ringing leaves the background a few levels above pure
    black, and a `> 0` test would mark the whole frame as subject."""
    with Image.open(source) as opened:
        if opened.mode == "RGBA":
            opened.save(destination)
            return
        rgb = opened.convert("RGB")
        array = np.asarray(rgb, dtype=np.int32)
        alpha = (array.sum(axis=2) > background_level).astype(np.uint8) * 255
        Image.fromarray(np.dstack([np.asarray(rgb, dtype=np.uint8), alpha]),
                        mode="RGBA").save(destination)


def split(transforms: Path, out_dir: Path, exclude: set, test_cameras: set,
          frame_range: str | None = None, rgba: bool = False) -> dict:
    """Write the split. Returns a summary of what it wrote."""
    data = json.loads(transforms.read_text())
    entries = data["frames"]
    if not entries:
        raise ValueError(f"{transforms} has no frames")

    present = {label_of(e) for e in entries}
    unknown = (exclude | test_cameras) - present
    if unknown:
        raise ValueError(f"cameras not in {transforms.name}: {sorted(unknown)} "
                         f"(present: {sorted(present)})")
    if not test_cameras:
        raise ValueError("give --test_camera: a split with no test view cannot be scored")
    if not test_cameras <= exclude:
        raise ValueError(f"--test_camera {sorted(test_cameras - exclude)} is not excluded from "
                         "training. A probe the model trained on is not a probe; add it to "
                         "--exclude, along with the rest of its stereo pair")

    lo = hi = None
    if frame_range is not None:
        try:
            lo_text, hi_text = frame_range.split("-")
            lo, hi = int(lo_text), int(hi_text)
        except ValueError:
            raise ValueError(f"--frames must look like LO-HI, got {frame_range!r}") from None

    def in_window(entry: dict) -> bool:
        if lo is None:
            return True
        number = trailing_number(str(entry.get("file_path", "")))
        return number is not None and lo <= number <= hi

    train, test = [], []
    for entry in entries:
        if not in_window(entry):
            continue
        label = label_of(entry)
        if label in test_cameras:
            test.append(entry)
        if label not in exclude:
            train.append(entry)

    if not train:
        raise ValueError("the split left no training views")
    if not test:
        raise ValueError("the split left no test views")

    out_dir.mkdir(parents=True, exist_ok=True)
    header = {k: v for k, v in data.items() if k != "frames"}
    (out_dir / "transforms_train.json").write_text(json.dumps({**header, "frames": train}, indent=1))
    (out_dir / "transforms_test.json").write_text(json.dumps({**header, "frames": test}, indent=1))

    # Symlink each image under a .png name rather than copying the files. Two
    # reasons for the naming. The reader builds its path as file_path + ".png",
    # so a .jpg capture would not be found at all; and PIL sniffs content rather
    # than trusting the extension, so a JPEG behind a .png name loads correctly.
    # The same trick is why the refit datasets link real .jpg frames as .png.
    needed = {(label_of(e), str(e["file_path"])) for e in train + test}
    linked, unresolved = set(), []
    for label, file_path in sorted(needed):
        source = resolve_source(transforms.parent, file_path)
        if source is None:
            unresolved.append(file_path)
            continue
        destination = out_dir / f"{file_path}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        if rgba:
            write_rgba(source, destination)
        else:
            os.symlink(source, destination)
        linked.add(label)
    if unresolved:
        raise FileNotFoundError(
            f"{len(unresolved)} image(s) named in {transforms.name} are not on disk, "
            f"for example {unresolved[0]!r}")

    return {"train_views": len(train), "test_views": len(test),
            "train_cameras": sorted({label_of(e) for e in train}),
            "test_cameras": sorted({label_of(e) for e in test}),
            "excluded": sorted(exclude), "linked": sorted(linked), "out_dir": str(out_dir)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transforms", required=True, type=Path,
                    help="source 4D transforms.json (an entry per camera and frame)")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--exclude", action="append", default=[], metavar="LABEL",
                    help="camera to keep out of training; repeatable. Exclude the probe's whole "
                         "stereo pair, since its twin sits under a degree away")
    ap.add_argument("--test_camera", action="append", default=[], metavar="LABEL",
                    help="camera to score against; must also be excluded from training")
    ap.add_argument("--rgba", action="store_true",
                    help="write RGBA frames with the subject silhouette in alpha, which is "
                         "where bake_sogst's lifetime mask-consistency filter reads its masks")
    ap.add_argument("--frames", default=None, metavar="LO-HI",
                    help="limit both splits to a capture-frame range")
    args = ap.parse_args()

    try:
        summary = split(args.transforms, args.out_dir, set(args.exclude), set(args.test_camera),
                        args.frames, rgba=args.rgba)
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"{summary['out_dir']}")
    print(f"  train: {summary['train_views']} views from {len(summary['train_cameras'])} cameras")
    print(f"  test:  {summary['test_views']} views from {summary['test_cameras']}")
    print(f"  HELD OUT of training: {summary['excluded']}")
    if summary["linked"]:
        print(f"  linked {len(summary['linked'])} image directories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
