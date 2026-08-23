#!/usr/bin/env python3
"""
build_frame_dataset.py

Extract ONE timestamp of a 4D transforms.json into a single-frame nerfstudio
dataset, optionally leaving cameras out.

Why this exists: the bakeoff in docs/render_and_repair.md is only meaningful if
the splat being repaired has never seen the camera it is scored against. Every
per-frame splat in the existing runs was trained on all twelve cameras, so a
probe at any real camera measures how well a repair preserves a view the
reconstruction already has, rather than whether it can supply one the rig never
covered. `--exclude` builds the training set that makes the second question
answerable.

Exclude the probe camera's whole stereo pair, not just the probe. This rig is six
pairs 0.3-0.8 degrees apart (see render_orbit_views.py), so holding out one half
of a pair leaves its twin supplying almost exactly the same view and the holdout
is not held out in any meaningful sense.

Per-camera image sizes differ in these captures (measured on 260529-171110 frame
68: 1024, 1200, 1552 and 912 px square across four cameras), and the principal
point moves frame to frame because each frame is cropped around the moving
subject. So `w`, `h`, `cx` and `cy` are written per frame from that frame's own
entry, with the size read off the image itself rather than trusted from the
transforms' top-level `w`/`h`, which was measured to disagree with every camera.

conda env: none (numpy + PIL).

Usage:
    python3 build_frame_dataset.py \\
        --transforms /path/to/omg4_full4d/transforms_train.json \\
        --frame 68 \\
        --out_dir /path/to/frame68_holdout \\
        [--exclude cam03 --exclude cam04] [--link]

Output:
    out_dir/transforms.json   nerfstudio format, one frame per included camera
    out_dir/images/<label>.<ext>   that camera's photo at this instant
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

from render_orbit_views import camera_label_from_path, trailing_number


def frame_entries(transforms: Path, frame: int) -> dict:
    """{camera label: transforms entry} for one timestamp."""
    data = json.loads(transforms.read_text())
    found = {}
    for entry in data["frames"]:
        file_path = str(entry.get("file_path", ""))
        if trailing_number(file_path) != frame:
            continue
        label = str(entry.get("camera_label", "")) or camera_label_from_path(file_path)
        if label:
            found[label] = entry
    return found


def resolve_image(transforms: Path, file_path: str) -> Path | None:
    """The photo behind a file_path, re-sniffing the extension because these
    transforms record extensionless paths (`cam04/frame_00068`)."""
    candidate = transforms.parent / file_path
    if candidate.is_file():
        return candidate
    for extension in (".jpg", ".jpeg", ".png", ".webp"):
        alternative = candidate.with_suffix(extension)
        if alternative.is_file():
            return alternative
    return None


def build(transforms: Path, frame: int, out_dir: Path, exclude: set, link: bool = False) -> dict:
    """Write the single-frame dataset. Returns a summary of what it wrote."""
    entries = frame_entries(transforms, frame)
    if not entries:
        raise ValueError(f"{transforms} has no entry whose file_path ends in frame {frame}")

    unknown = exclude - set(entries)
    if unknown:
        raise ValueError(f"--exclude names cameras absent at frame {frame}: {sorted(unknown)} "
                         f"(present: {sorted(entries)})")

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    frames, skipped = [], []
    for label in sorted(entries):
        if label in exclude:
            continue
        entry = entries[label]
        source = resolve_image(transforms, str(entry["file_path"]))
        if source is None:
            skipped.append(label)
            continue
        destination = images_dir / f"{label}{source.suffix}"
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if link:
            os.symlink(source.resolve(), destination)
        else:
            shutil.copyfile(source, destination)

        with Image.open(source) as image:   # closed explicitly; a leaked handle per
            width, height = image.size      # camera adds up over a full sequence rebuild
        frames.append({
            "file_path": f"images/{destination.name}",
            "transform_matrix": entry["transform_matrix"],
            "fl_x": entry["fl_x"], "fl_y": entry["fl_y"],
            "cx": entry["cx"], "cy": entry["cy"],
            "w": width, "h": height,
            "camera_label": label,
        })

    if not frames:
        raise ValueError(f"no images resolved for frame {frame}; check that {transforms.parent} "
                         "holds the per-camera image directories")

    (out_dir / "transforms.json").write_text(json.dumps({
        "camera_model": "OPENCV", "frames": frames,
    }, indent=1))
    return {"frame": frame, "cameras": [f["camera_label"] for f in frames],
            "excluded": sorted(exclude), "missing": skipped, "out_dir": str(out_dir)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transforms", required=True, type=Path,
                    help="4D transforms.json holding an entry per camera and timestamp")
    ap.add_argument("--frame", required=True, type=int,
                    help="capture frame number, matched against each file_path's trailing digits")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--exclude", action="append", default=[], metavar="LABEL",
                    help="camera to leave out; repeatable. Exclude a probe camera's whole stereo "
                         "pair, since its twin sits under a degree away")
    ap.add_argument("--link", action="store_true",
                    help="symlink images instead of copying them")
    args = ap.parse_args()

    try:
        summary = build(args.transforms, args.frame, args.out_dir, set(args.exclude), args.link)
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"frame {summary['frame']}: {len(summary['cameras'])} cameras -> {summary['out_dir']}")
    print(f"  included: {' '.join(summary['cameras'])}")
    if summary["excluded"]:
        print(f"  HELD OUT: {' '.join(summary['excluded'])}")
    if summary["missing"]:
        print(f"  WARNING: no image on disk for {' '.join(summary['missing'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
