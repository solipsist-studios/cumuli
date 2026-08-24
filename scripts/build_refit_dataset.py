#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""build_refit_dataset.py - real views plus repaired novel views, for a 4D refit.

This is the step the whole novel-view chain exists to feed. Scoring a repair at
a held-out CAMERA measures how faithful it is where a real camera already
stands; it says nothing about the failure this addresses, which is floaters and
smear at angles far from any camera. The only way to test that is to train on
the repaired views and look at what the reconstruction does in the directions
the rig never covered.

Takes the real transforms (minus whatever is held out) and one or more repaired
sweeps, and writes a dataset the OMG4 trainer reads directly: every repaired
frame becomes a training view at the pose and instant it was rendered from.

Poses round-trip through the same convention as everything else. A sweep records
w2c; the trainer wants nerfstudio/OpenGL c2w, so the inverse is taken and the Y
and Z axis columns negated back.

KNOWN LIMIT: render_pair_sweep writes a per-frame `loss_weight` (Hwang et al.'s
angular falloff, highest at the real-pinned endpoints), and the OMG4 trainer has
no per-view weighting to apply it with. It is carried into the output for a
trainer that can use it, and `--weight_by_repeat` approximates it by repeating
high-weight views, which is crude but is the only lever this trainer offers.

conda env: cumuli.

Usage:
    python scripts/build_refit_dataset.py \\
        --real_transforms /path/to/heldout1/transforms_train.json \\
        --test_transforms /path/to/heldout1/transforms_test.json \\
        --repaired /path/to/rep/cam02_to_cam06:/path/to/sweeps/cam02_to_cam06 \\
        --out_dir /path/to/refit_dataset

    --repaired is REPAIRED_DIR:SWEEP_DIR and is repeatable. The sweep directory
    supplies cameras.json; the repaired directory supplies the frames.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def w2c_to_opengl_c2w(w2c: np.ndarray) -> np.ndarray:
    """Inverse of rig_geometry.opengl_c2w_to_w2c."""
    m = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
    m[:3, 1:3] *= -1
    return m


def repaired_entries(repaired_dir: Path, sweep_dir: Path, out_dir: Path,
                     tag: str, link: bool) -> list:
    """Transforms entries for one repaired sweep, copying its frames in.

    Frames that the repair did not produce are skipped rather than substituted
    with the render: a refit should train on what the model actually output, and
    silently mixing the two would make the experiment unreadable."""
    meta = json.loads((sweep_dir / "cameras.json").read_text())
    images_dir = out_dir / tag
    images_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for record in meta["frames"]:
        index = record["idx"]
        source = repaired_dir / f"sweep_{index:04d}.png"
        if not source.exists():
            continue
        name = f"{index:04d}"
        destination = images_dir / f"{name}.png"
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if link:
            os.symlink(source.resolve(), destination)
        else:
            shutil.copyfile(source, destination)

        entries.append({
            "file_path": f"{tag}/{name}",
            "transform_matrix": w2c_to_opengl_c2w(np.array(record["w2c"])).tolist(),
            "fl_x": meta["fl_x"], "fl_y": meta["fl_y"],
            "cx": meta["cx"], "cy": meta["cy"],
            "w": meta["res"], "h": meta["res"],
            "time": record["time"],
            "camera_label": tag,
            # carried for a trainer that can weight views; OMG4 cannot
            "loss_weight": record.get("loss_weight", 1.0),
        })
    return entries


def build(real_transforms: Path, test_transforms: Path | None, repaired: list,
          out_dir: Path, link: bool = True, weight_by_repeat: bool = False) -> dict:
    """Write the refit dataset. Returns a summary."""
    real = json.loads(real_transforms.read_text())
    header = {k: v for k, v in real.items() if k != "frames"}
    real_frames = real["frames"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # the real views already live beside the source transforms; link them across
    # under the .png names the reader builds
    linked = 0
    for entry in real_frames:
        file_path = str(entry["file_path"])
        source = (real_transforms.parent / f"{file_path}.png")
        if not source.exists():
            continue
        destination = out_dir / f"{file_path}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        os.symlink(source.resolve(), destination)
        linked += 1

    synthetic = []
    for spec in repaired:
        repaired_dir, _, sweep_dir = spec.partition(":")
        if not sweep_dir:
            raise ValueError(f"--repaired wants REPAIRED_DIR:SWEEP_DIR, got {spec!r}")
        tag = Path(sweep_dir).name
        entries = repaired_entries(Path(repaired_dir), Path(sweep_dir), out_dir, tag, link)
        if not entries:
            raise ValueError(f"{repaired_dir} produced no frames")
        synthetic.extend(entries)

    if weight_by_repeat:
        expanded = []
        for entry in synthetic:
            expanded.extend([entry] * max(1, int(round(entry.get("loss_weight", 1.0)))))
        synthetic = expanded

    (out_dir / "transforms_train.json").write_text(
        json.dumps({**header, "frames": real_frames + synthetic}, indent=1))
    if test_transforms is not None:
        shutil.copyfile(test_transforms, out_dir / "transforms_test.json")
        for entry in json.loads(test_transforms.read_text())["frames"]:
            file_path = str(entry["file_path"])
            source = test_transforms.parent / f"{file_path}.png"
            destination = out_dir / f"{file_path}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.exists() and not (destination.exists() or destination.is_symlink()):
                os.symlink(source.resolve(), destination)

    return {"real_views": len(real_frames), "synthetic_views": len(synthetic),
            "linked_real": linked, "sweeps": len(repaired), "out_dir": str(out_dir)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_transforms", required=True, type=Path,
                    help="transforms_train.json for the real views (already excluding any holdout)")
    ap.add_argument("--test_transforms", type=Path, default=None,
                    help="transforms_test.json to carry across, so the refit is scored on the "
                         "same held-out camera")
    ap.add_argument("--repaired", action="append", default=[], metavar="REPAIRED:SWEEP",
                    help="repaired frame directory and the sweep it came from; repeatable")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--copy", action="store_true", help="copy repaired frames instead of linking")
    ap.add_argument("--weight_by_repeat", action="store_true",
                    help="repeat each synthetic view round(loss_weight) times, approximating the "
                         "angular weighting OMG4 cannot express directly")
    args = ap.parse_args()

    if not args.repaired:
        print("ERROR: give at least one --repaired REPAIRED_DIR:SWEEP_DIR", file=sys.stderr)
        return 1
    try:
        summary = build(args.real_transforms, args.test_transforms, args.repaired,
                        args.out_dir, link=not args.copy,
                        weight_by_repeat=args.weight_by_repeat)
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"{summary['out_dir']}")
    print(f"  real:      {summary['real_views']} views ({summary['linked_real']} images linked)")
    print(f"  synthetic: {summary['synthetic_views']} views from {summary['sweeps']} sweep(s)")
    total = summary["real_views"] + summary["synthetic_views"]
    print(f"  synthetic share: {100 * summary['synthetic_views'] / total:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
