#!/usr/bin/env python3
"""
validate_stage_output.py

Sanity-checks one stage's on-disk output before run_unified_pipeline.py's
main() proceeds to the next stage. A stage script can exit 0 while still
having produced garbage -- HLOC/pose refinement silently collapsing every
camera to the same (bootstrap) pose, BiRefNet/clean_masks.py dropping the
subject out of a mask entirely, a camera going missing partway through the
candidate window -- and today that garbage only gets noticed by a human
looking at the final splat. This catches it right after the stage that
produced it, before the next (often more expensive) stage runs on top of it.

Runs in the same conda env as the other generic (numpy/Pillow) stages --
run_unified_pipeline.py itself stays free of third-party imports and shells
out to this exactly like any other stage via run_script(), so a validation
failure raises the same StageError -> fail() -> sys.exit(1) path as every
other stage failure.

Usage:
    python3 validate_stage_output.py --stage poses --out_dir /path/to/run \\
        --real_cameras 0001,0002,0003
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import run_unified_pipeline as unified


class ValidationError(RuntimeError):
    pass


def validate_sync(L, real_cameras):
    if not L["sync_grid"].is_file():
        raise ValidationError(f"{L['sync_grid']} was not produced")
    if not L["sync_offsets"].is_file():
        raise ValidationError(f"{L['sync_offsets']} was not produced")

    data = json.loads(L["sync_offsets"].read_text())
    offset_cams = sorted(Path(name).stem for name in data["offsets"])
    if offset_cams != sorted(real_cameras):
        raise ValidationError(
            f"sync_offsets.json covers {offset_cams}, expected {sorted(real_cameras)}")


def validate_production(L, real_cameras):
    found = sorted(p.stem for p in L["production_undist"].glob("*.jpg"))
    if found != sorted(real_cameras):
        raise ValidationError(
            f"expected one undistorted frame per camera {sorted(real_cameras)}, got {found}")

    for cam in real_cameras:
        path = L["production_undist"] / f"{cam}.jpg"
        with Image.open(path) as im:
            w, h = im.size
        if w < 1000 or h < 1000:
            raise ValidationError(f"{cam}: suspiciously small undistorted frame ({w}x{h})")


def validate_poses(L, real_cameras):
    if not L["transforms_refined"].is_file():
        raise ValidationError(f"{L['transforms_refined']} was not produced")

    data = json.loads(L["transforms_refined"].read_text())
    frames = data["frames"]
    if len(frames) != len(real_cameras):
        raise ValidationError(
            f"transforms_refined.json has {len(frames)} frames, expected {len(real_cameras)}")

    labels = [fr["camera_label"] for fr in frames]
    if len(labels) != len(set(labels)):
        raise ValidationError(f"duplicate camera_label in transforms_refined.json: {labels}")

    positions = []
    for fr in frames:
        m = np.asarray(fr["transform_matrix"], dtype=np.float64)
        if m.shape != (4, 4) or not np.all(np.isfinite(m)):
            raise ValidationError(f"{fr['camera_label']} has an invalid transform_matrix")
        positions.append(m[:3, 3])

    # Cameras physically surround the subject -- poses collapsing to (near)
    # the same position means pose solving/refinement silently fell back to
    # the bootstrap identity pose instead of actually solving anything.
    positions = np.array(positions)
    spread = positions.max(axis=0) - positions.min(axis=0)
    if not np.any(spread > 0.01):
        raise ValidationError(
            f"camera positions are suspiciously identical (spread={spread}) -- "
            "pose solve/refinement likely failed silently")


def validate_masks(L, real_cameras):
    if not L["flat_transforms"].is_file():
        raise ValidationError(f"{L['flat_transforms']} was not produced")

    labels = [fr["camera_label"] for fr in json.loads(L["flat_transforms"].read_text())["frames"]]
    if len(labels) != len(real_cameras):
        raise ValidationError(
            f"transforms.json (flattened) has {len(labels)} cameras, expected {len(real_cameras)}")

    for label in labels:
        mask_path = L["flat_fmasks_clean"] / f"{label}.png"
        if not mask_path.is_file():
            raise ValidationError(f"missing cleaned mask for {label}")
        with Image.open(mask_path) as im:
            mask = np.asarray(im.convert("L"))
        frac = float((mask > 127).mean())
        # A real subject silhouette should cover a visible but minority
        # fraction of the frame -- ~0 means the subject got dropped
        # entirely (BiRefNet/cleanup failure), >90% means it kept the
        # background instead of the subject.
        if not (0.005 < frac < 0.9):
            raise ValidationError(f"{label}: implausible mask coverage {frac:.3f}")


def validate_branch(L, real_cameras):
    sparse_dir = L["train_set"] / "sparse" / "0"
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (sparse_dir / name).is_file():
            raise ValidationError(f"{sparse_dir / name} was not produced")

    cam_lines = [l for l in (sparse_dir / "cameras.txt").read_text().splitlines() if not l.startswith("#")]
    img_lines = [l for l in (sparse_dir / "images.txt").read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
    pts_lines = [l for l in (sparse_dir / "points3D.txt").read_text().splitlines() if not l.startswith("#")]

    if len(cam_lines) != len(real_cameras):
        raise ValidationError(f"cameras.txt has {len(cam_lines)} entries, expected {len(real_cameras)}")
    if len(img_lines) != len(real_cameras):
        raise ValidationError(f"images.txt has {len(img_lines)} entries, expected {len(real_cameras)}")
    if not pts_lines:
        raise ValidationError("points3D.txt has zero triangulated points")

    plys = sorted(L["brush_output"].glob("*.ply"))
    if not plys:
        raise ValidationError(f"no .ply exported under {L['brush_output']}")


VALIDATORS = {
    "sync": validate_sync,
    "production": validate_production,
    "poses": validate_poses,
    "masks": validate_masks,
    "branch": validate_branch,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=list(VALIDATORS))
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--real_cameras", required=True,
                         help="Comma-separated real camera ids (discover_cameras()'s video stems)")
    args = parser.parse_args()

    L = unified.build_layout(args.out_dir)
    real_cameras = args.real_cameras.split(",")

    try:
        VALIDATORS[args.stage](L, real_cameras)
    except ValidationError as e:
        print(f"Error: stage {args.stage!r} output failed validation: {e}")
        sys.exit(1)

    print(f"OK: stage {args.stage!r} output looks valid ({len(real_cameras)} cameras)")


if __name__ == "__main__":
    main()
