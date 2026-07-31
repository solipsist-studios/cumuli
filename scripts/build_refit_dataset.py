#!/usr/bin/env python3
"""
build_refit_dataset.py

Merge real camera views with the repaired synthetic orbit views into one 4D
training dataset -- the last step of render-and-repair before the refit train
(see docs/render_and_repair.md). Consumes an existing 4D dataset
(build_4dgs_dataset.py output) plus klein_repair_views.py output.

Why this exists: the point of rendering and repairing novel views is to train
on them. This writes the merged transforms_train.json the 4D trainer reads,
placing each repaired view at its render pose and its frame's timestamp.

Windowing: pass --time_min/--time_max to build a dataset covering only part of
the clip. Narrow windows are how the seam-stitched chain is produced -- one
model per window, each with an undiluted splat budget over its own time range,
merged afterwards by merge_omg4_segments.py. A single wide window spreads
capacity across the whole clip and visibly under-resolves fast motion.

HEAD VIEWS ARE OPT-IN, and that default is load-bearing. Close-up head views
-- both real crops and synthetic zooms -- were measured to POISON shared-window
4D refits on this capture: a five-variant bisection put real head crops at
+0.12% bright-opaque artifact splats and synthetic head zooms at +0.06%, while
body-only supervision came in at 0.008%, the cleanest measured. The mechanism
is that binary masks cut through motion-blurred, backlit head boundaries,
leaving blown highlights inside the mask; a static per-frame fit can paint that
blur, but a shared-window 4D fit can only reconcile it as bright semi-opaque
fog. Enable --include_head_views / --include_synthetic_head_views only on
captures without that lighting problem, and check the result.

Real images are symlinked under a .png extension even when they are .jpg. This
is not cosmetic: the 4D trainer's Blender-format loader applies ONE global
extension to every frame in transforms_train.json, and PIL sniffs actual
content rather than trusting the name, so a uniform .png naming lets .jpg reals
and .png synthetics coexist in one dataset.

conda env: none (numpy + scipy).

Usage:
    python3 build_refit_dataset.py \\
        --real_dataset /path/to/dataset_4dgs \\
        --repaired_root /path/to/repaired \\
        --out_dir /path/to/dataset_refit \\
        [--time_min 0.0 --time_max 0.467] \\
        [--perframe_root /path/to/perframe --include_head_views] \\
        [--include_synthetic_head_views]

Output:
    out_dir/transforms_train.json (real + repaired), transforms_test.json
    (real held-out views only -- synthetics never become test views), a
    realcams/ symlink tree, and a points3d.ply symlink to the real dataset's
    initialization cloud.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TIME_EPSILON = 1e-6  # tolerance when testing a frame's time against the window bounds


def colmap_quaternion_to_c2w(qw: float, qx: float, qy: float, qz: float, t: np.ndarray) -> np.ndarray:
    """COLMAP images.txt stores world-to-camera (R, t). Return the OpenGL c2w
    the Blender-format loader expects -- invert, then undo the Y/Z axis flip,
    the inverse of build_colmap_sparse.opengl_c2w_to_colmap_w2c."""
    w2c = np.eye(4)
    w2c[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1
    return c2w


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Render-pose w2c (render_orbit_views.py cameras.json) -> loader c2w."""
    c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
    c2w[:3, 1:3] *= -1
    return c2w


def read_colmap_sparse(sparse_dir: Path) -> tuple:
    """Minimal COLMAP text reader: ({camera_id: (w, h, fx, fy, cx, cy)},
    {image_name: (camera_id, c2w)}). The repo writes COLMAP text in
    build_colmap_sparse.py but has no reader; this covers the PINHOLE case
    that build_densification_crops.py appends crop views to."""
    cameras = {}
    for line in (sparse_dir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cameras[int(parts[0])] = (int(parts[2]), int(parts[3]),
                                  float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))

    images = {}
    for line in (sparse_dir / "images.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 10:  # the POINTS2D continuation line
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        translation = np.array(list(map(float, parts[5:8])))
        images[parts[9]] = (int(parts[8]), colmap_quaternion_to_c2w(qw, qx, qy, qz, translation))
    return cameras, images


def frame_index(path: Path) -> int:
    """Trailing integer of a frame directory name (frame_0007 -> 7, frame07 -> 7)."""
    digits = ""
    for char in reversed(path.name):
        if not char.isdigit():
            break
        digits = char + digits
    if not digits:
        raise ValueError(f"{path.name} does not end in a frame number")
    return int(digits)


def link_real_views(frames: list, real_dataset: Path, out_dir: Path,
                    time_min: float, time_max: float) -> list:
    """Symlink in-window real images under realcams/ with a .png name and
    rewrite their file_path. Frames whose image is missing are skipped."""
    linked = []
    for frame in frames:
        if not (time_min - TIME_EPSILON <= frame["time"] <= time_max + TIME_EPSILON):
            continue
        relative = Path(frame["file_path"])
        source = next((c for c in real_dataset.glob(f"{relative}.*") if c.is_file()), None)
        if source is None:
            continue
        destination = out_dir / "realcams" / relative.parent / f"{relative.name}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(source.resolve())
        updated = dict(frame)
        updated["file_path"] = f"realcams/{relative.parent}/{relative.name}"
        linked.append(updated)
    return linked


def collect_head_crops(perframe_root: Path, out_dir: Path, times: list,
                       time_min: float, time_max: float) -> list:
    """Real head-crop views from the per-frame COLMAP datasets, as extra train
    views at their frame's timestamp. These are already-real, already-sharp
    photos, so they need no repair pass."""
    collected, missing = [], 0
    for frame_dir in sorted(p for p in perframe_root.iterdir() if p.is_dir()):
        try:
            index = frame_index(frame_dir)
        except ValueError:
            continue
        if index >= len(times):
            continue
        timestamp = times[index]
        if not (time_min - TIME_EPSILON <= timestamp <= time_max + TIME_EPSILON):
            continue
        sparse_dir = frame_dir / "sparse" / "0"
        if not (sparse_dir / "cameras.txt").exists():
            missing += 1
            continue

        cameras, images = read_colmap_sparse(sparse_dir)
        for name, (camera_id, c2w) in images.items():
            stem = Path(name).stem
            if not stem.startswith("head_"):
                continue
            source = next((c for c in frame_dir.rglob(name) if c.is_file()), None)
            if source is None:
                continue
            _, _, fx, fy, cx, cy = cameras[camera_id]
            destination = out_dir / "realheads" / frame_dir.name / f"{stem}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.symlink_to(source.resolve())
            collected.append({"file_path": f"realheads/{frame_dir.name}/{stem}",
                              "transform_matrix": c2w.tolist(), "time": timestamp,
                              "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy})
    if missing:
        print(f"  warning: {missing} frame(s) had no COLMAP sparse data, their head crops were skipped")
    return collected


def collect_repaired_views(repaired_root: Path, out_dir: Path, times: list,
                           time_min: float, time_max: float, include_head: bool) -> list:
    """Repaired synthetic views, placed at their render pose and frame time.
    file_path is computed relative to out_dir rather than assumed, so pointing
    --repaired_root elsewhere cannot silently train on a stale directory."""
    relative_root = Path(os.path.relpath(repaired_root.resolve(), out_dir.resolve()))
    collected, missing = [], 0
    for frame_dir in sorted(p for p in repaired_root.iterdir() if p.is_dir()):
        try:
            index = frame_index(frame_dir)
        except ValueError:
            continue
        if index >= len(times):
            continue
        timestamp = times[index]
        if not (time_min - TIME_EPSILON <= timestamp <= time_max + TIME_EPSILON):
            continue
        cameras_json = frame_dir / "cameras.json"
        if not cameras_json.exists():
            print(f"  warning: {frame_dir.name} has no cameras.json, skipping its synthetic views")
            continue

        meta = json.loads(cameras_json.read_text())
        passes = [("body", meta, meta.get("frames", []))]
        if include_head and meta.get("head"):
            passes.append(("head", meta["head"], meta["head"].get("frames", [])))

        for prefix, intrinsics, records in passes:
            for record in records:
                name = f"{prefix}_{record['idx']:03d}"
                if not (frame_dir / f"{name}.png").exists():
                    missing += 1
                    continue
                collected.append({
                    "file_path": f"{relative_root}/{frame_dir.name}/{name}",
                    "transform_matrix": w2c_to_c2w(record["w2c"]).tolist(), "time": timestamp,
                    "fl_x": intrinsics["fl_x"], "fl_y": intrinsics["fl_y"],
                    "cx": intrinsics["cx"], "cy": intrinsics["cy"]})
    if missing:
        print(f"  warning: {missing} repaired view(s) listed in cameras.json were missing on disk")
    return collected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dataset", required=True, type=Path,
                    help="existing 4D dataset (build_4dgs_dataset.py output): transforms_train.json, "
                         "transforms_test.json, per-camera images, points3d.ply")
    ap.add_argument("--repaired_root", required=True, type=Path,
                    help="klein_repair_views.py output root, one frame_NNNN/ subdirectory per frame")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--time_min", type=float, default=None, help="window start (default: clip start)")
    ap.add_argument("--time_max", type=float, default=None, help="window end (default: clip end)")
    ap.add_argument("--perframe_root", type=Path, default=None,
                    help="per-frame COLMAP datasets, source of the real head crops "
                         "(required by --include_head_views)")
    ap.add_argument("--include_head_views", action="store_true",
                    help="add real head-crop views -- measured to poison backlit shared-window refits, "
                         "see the module docstring before enabling")
    ap.add_argument("--include_synthetic_head_views", action="store_true",
                    help="add the repaired head-zoom views (same caveat)")
    args = ap.parse_args()

    if args.include_head_views and args.perframe_root is None:
        ap.error("--include_head_views requires --perframe_root")

    try:
        train = json.loads((args.real_dataset / "transforms_train.json").read_text())
        test = json.loads((args.real_dataset / "transforms_test.json").read_text())
    except OSError as e:
        print(f"Error: could not read the real dataset: {e}", file=sys.stderr)
        return 1

    times = sorted({frame["time"] for frame in train["frames"]})
    if not times:
        print("Error: the real dataset has no frames", file=sys.stderr)
        return 1
    time_min = args.time_min if args.time_min is not None else times[0]
    time_max = args.time_max if args.time_max is not None else times[-1]
    print(f"window [{time_min:.5f}, {time_max:.5f}] of {len(times)} frames "
          f"spanning [{times[0]:.5f}, {times[-1]:.5f}]")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    real_train = link_real_views(train["frames"], args.real_dataset, args.out_dir, time_min, time_max)
    real_test = link_real_views(test["frames"], args.real_dataset, args.out_dir, time_min, time_max)
    print(f"real body views: {len(real_train)} train, {len(real_test)} test")

    head_views = []
    if args.include_head_views:
        head_views = collect_head_crops(args.perframe_root, args.out_dir, times, time_min, time_max)
        print(f"real head-crop views: {len(head_views)} (train only)")

    repaired = collect_repaired_views(args.repaired_root, args.out_dir, times, time_min, time_max,
                                      args.include_synthetic_head_views)
    print(f"repaired synthetic views: {len(repaired)} (train only)")
    if not repaired:
        print("Error: no repaired views found -- check --repaired_root and the window bounds",
              file=sys.stderr)
        return 1

    (args.out_dir / "transforms_train.json").write_text(
        json.dumps({"frames": real_train + head_views + repaired}))
    (args.out_dir / "transforms_test.json").write_text(json.dumps({"frames": real_test}))

    points = args.out_dir / "points3d.ply"
    source_points = args.real_dataset / "points3d.ply"
    if not points.exists() and source_points.exists():
        points.symlink_to(source_points.resolve())

    total = len(real_train) + len(head_views) + len(repaired)
    print(f"train: {total} views ({len(real_train)} real body + {len(head_views)} real head "
          f"+ {len(repaired)} repaired)  test: {len(real_test)}")
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
