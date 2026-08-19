#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

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
import zipfile
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

    # compute_sync_offsets.py keeps a camera whose audio failed in the json
    # as an {"error": ...} stub, so name coverage alone isn't enough -- every
    # entry must actually be a usable offset.
    for name, entry in data["offsets"].items():
        if "error" in entry:
            raise ValidationError(
                f"{name} has no usable sync data (compute_sync_offsets.py recorded: {entry['error']})")
        if "frame_offset" not in entry or "fps" not in entry:
            raise ValidationError(f"{name}'s sync entry is missing frame_offset/fps: {entry}")
        if not (np.isfinite(entry["frame_offset"]) and np.isfinite(entry["fps"]) and entry["fps"] > 0):
            raise ValidationError(f"{name} has a non-finite/degenerate sync entry: {entry}")


def validate_production(L, real_cameras):
    # .jpg normally, .png when the run went through --pp3_dir color correction
    # (run_unified_pipeline.py's image_ext) -- accept either.
    frames = {p.stem: p for ext in ("*.jpg", "*.png") for p in L["production_undist"].glob(ext)}
    found = sorted(frames)
    if found != sorted(real_cameras):
        raise ValidationError(
            f"expected one undistorted frame per camera {sorted(real_cameras)}, got {found}")

    for cam in real_cameras:
        with Image.open(frames[cam]) as im:
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
        if not np.allclose(m[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValidationError(
                f"{fr['camera_label']}'s transform_matrix bottom row is {m[3].tolist()}, "
                "not [0, 0, 0, 1] -- not a valid homogeneous c2w transform")
        # The 3x3 block must be a proper rotation (det +1, orthonormal) --
        # HLOC/refinement output rigid c2w transforms, so anything else
        # (sheared, scaled, reflected, zeroed) means a corrupted solve.
        R = m[:3, :3]
        det = np.linalg.det(R)
        if det < 0.5 or not np.allclose(R @ R.T, np.eye(3), atol=0.05):
            raise ValidationError(
                f"{fr['camera_label']}'s transform_matrix rotation block is not a "
                f"proper rotation (det={det:.3f}) -- corrupted pose solve")
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
    if len(labels) != len(set(labels)):
        raise ValidationError(f"duplicate camera_label in flattened transforms.json: {labels}")

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


def validate_dataset4d(L, real_cameras):
    root = L["dataset4d"]
    frames = None
    for name in ("transforms_train.json", "transforms_test.json"):
        path = root / name
        if not path.is_file():
            raise ValidationError(f"{path} was not produced")
        try:
            data = json.loads(path.read_text())
        except ValueError as e:
            raise ValidationError(f"{path} is not valid JSON: {e}")
        if name == "transforms_train.json":
            frames = data.get("frames", [])

    if not frames:
        raise ValidationError(f"{root / 'transforms_train.json'} has zero frames")

    # Per-view intrinsics are the contract. build_flipbook_4dgs_dataset.py
    # writes fl/c/w/h/time on every frame entry, and there is deliberately
    # no global intrinsics block (the cameras differ).
    for key in ("file_path", "time", "fl_x", "fl_y", "cx", "cy"):
        missing = [i for i, fr in enumerate(frames) if key not in fr]
        if missing:
            raise ValidationError(
                f"transforms_train.json: {len(missing)} frames lack {key!r} "
                f"(first at index {missing[0]})")

    # file_path is extensionless. The image must exist and must be RGBA:
    # the mask is baked into alpha, and an RGB image means the bake step
    # silently failed while every file still exists.
    probe = root / (frames[0]["file_path"] + ".png")
    if not probe.is_file():
        raise ValidationError(f"{probe} referenced by transforms_train.json is missing")
    with Image.open(probe) as im:
        if im.mode != "RGBA":
            raise ValidationError(f"{probe} is mode {im.mode}, expected RGBA (mask in alpha)")

    ply = root / "points3d.ply"
    if not ply.is_file():
        raise ValidationError(f"{ply} was not produced")
    if ply_vertex_count(ply) < 1:
        raise ValidationError(f"{ply} declares zero points (failed visual-hull init?)")
    if not ply_header_has_property(ply, "time"):
        raise ValidationError(
            f"{ply} header lacks a per-point 'time' property -- the 4D trainer "
            f"buckets its init points by time and cannot use a static cloud")


def validate_train4d(L, real_cameras):
    model_dir = L["train4d_model"]
    ckpts = sorted(model_dir.glob("chkpnt*.pth")) if model_dir.is_dir() else []
    if not ckpts:
        raise ValidationError(f"no chkpnt*.pth under {model_dir}")

    # A crashed trainer can leave a small partial file behind. A real rotor
    # checkpoint is tens of MB at minimum, so 1 MB is a safe floor.
    newest = max(ckpts, key=lambda p: p.stat().st_mtime)
    if newest.stat().st_size < 1_000_000:
        raise ValidationError(
            f"{newest} is {newest.stat().st_size} bytes -- truncated checkpoint?")

    out = L["sogst_out"]
    if not out.is_file():
        raise ValidationError(f"{out} was not produced")
    # stdlib-only probe so this module stays importable in the torch-free
    # generic env: a .sogst is a ZIP whose first entry is meta.json.
    try:
        with zipfile.ZipFile(out) as zf:
            meta = json.loads(zf.read("meta.json"))
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        raise ValidationError(f"{out} is not a readable .sogst archive: {e}")
    if meta.get("format") != "sogst":
        raise ValidationError(
            f"{out} meta.format is {meta.get('format')!r}, expected 'sogst'")
    if int(meta.get("count", 0)) < 1:
        raise ValidationError(f"{out} declares zero splats")


def ply_vertex_count(path):
    """Vertex count from a .ply header (ascii or binary body), or -1 if the
    header is missing/unparseable."""
    with path.open("rb") as f:
        header = f.read(4096).decode("ascii", errors="replace")
    if not header.startswith("ply"):
        return -1
    for line in header.splitlines():
        parts = line.split()
        if parts[:2] == ["element", "vertex"] and len(parts) == 3 and parts[2].isdigit():
            return int(parts[2])
    return -1




def ply_header_has_property(path, name):
    """True when the .ply header declares a property with this name. Header
    is read the same way as ply_vertex_count (first 4096 bytes)."""
    with path.open("rb") as f:
        header = f.read(4096).decode("ascii", errors="replace")
    if not header.startswith("ply"):
        return False
    for line in header.splitlines():
        parts = line.split()
        if parts[:1] == ["property"] and parts[-1:] == [name]:
            return True
    return False


VALIDATORS = {
    "sync": validate_sync,
    "production": validate_production,
    "poses": validate_poses,
    "masks": validate_masks,
    "dataset4d": validate_dataset4d,
    "train4d": validate_train4d,
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
