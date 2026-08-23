#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""window_keypoints_3d.py - triangulated 3D keypoints for a window of a 4D dataset.

Produces the `poses_3d/<frame>.json` that project_skeleton_conditioning.py
projects into a sweep's novel views. Runs the pipeline's own stages
(predict_keypoints_2d.py, split_keypoints_per_camera.py,
triangulate_and_project_keypoints.py) once per capture frame and collects the
results into one directory.

Why per frame rather than once for the whole window: datasets that crop each
frame around the moving subject move the principal point frame to frame, so a
single camera file cannot describe more than one instant. Triangulating the
whole window against frame 1's intrinsics would bend every 3D point by that
drift. Each frame therefore gets its own camera file, built the same way
build_frame_dataset.py builds one.

Why re-predict keypoints rather than reuse an existing prediction file: 2D
keypoints are only meaningful alongside the intrinsics of the image they were
detected in. Older prediction files on a capture may be in a different image
space (full-resolution undistorted rather than per-frame crops), and silently
triangulating those against crop intrinsics produces plausible-looking 3D points
that are wrong. Predicting on exactly the images the transforms describes
removes that whole class of error.

Masks: the keypoint stage requires them. When the source images already have the
background removed, --masks_from_alpha derives each mask from the image itself
(alpha channel when present, otherwise non-black pixels) rather than requiring a
separate masks directory.

conda env: cumuli, except the keypoint stage, which runs under --sapiens_env
(default sapiens2) because Sapiens needs its own dependency set.

Usage:
    python scripts/window_keypoints_3d.py \\
        --transforms /path/to/transforms.json \\
        --frames 58-74 \\
        --out_dir /path/to/window_kp \\
        --sapiens_checkpoint_root ~/Dev/sapiens_ckpt \\
        [--masks_from_alpha] [--sapiens_env sapiens2]

Output:
    out_dir/poses_3d/<frame>.json     triangulated 3D keypoints per frame
    out_dir/frames/<frame>/           the per-frame camera file and staged images
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import build_frame_dataset
from rig_geometry import load_rig

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
BACKGROUND_LEVEL = 8  # 8-bit sum below this counts as removed background


def frame_numbers(transforms: Path, frame_range: str | None) -> list:
    """Capture frames to process, from what the rig actually contains."""
    cameras, _, _, _ = load_rig(transforms)
    available = sorted({n for camera in cameras for n in camera["per_frame"] if n is not None})
    if not available:
        raise ValueError(f"{transforms} records no numbered frames")
    if frame_range is None:
        return available
    try:
        lo_text, hi_text = frame_range.split("-")
        lo, hi = int(lo_text), int(hi_text)
    except ValueError:
        raise ValueError(f"--frames must look like LO-HI, got {frame_range!r}") from None
    selected = [n for n in available if lo <= n <= hi]
    if not selected:
        raise ValueError(f"no frames in {frame_range!r}; the rig has "
                         f"{available[0]}-{available[-1]}")
    return selected


def write_masks(images_dir: Path, masks_dir: Path) -> int:
    """A binary mask per staged image, from its alpha channel when it has one and
    from its non-black pixels otherwise.

    The second case is what background-removed captures need: the subject is
    already cut out against black, so the silhouette is recoverable without a
    separate masks directory or a second segmentation pass."""
    masks_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        with Image.open(image_path) as opened:
            if opened.mode == "RGBA":
                mask = np.asarray(opened.split()[-1], dtype=np.uint8) > 127
            else:
                rgb = np.asarray(opened.convert("RGB"), dtype=np.int32)
                mask = rgb.sum(axis=2) > BACKGROUND_LEVEL
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
            masks_dir / f"{image_path.stem}.png")
        written += 1
    return written


def run(command: list, label: str) -> None:
    """Run a pipeline stage, failing loudly with the stage's own output."""
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def process_frame(transforms: Path, frame: int, work_dir: Path, poses_3d: Path, *,
                  sapiens_env: str, checkpoint_root: Path | None, model_size: str,
                  masks_from_alpha: bool, masks_dir: Path | None, skip_existing: bool) -> bool:
    """Stage one capture frame and triangulate it. Returns True when it produced
    a 3D keypoint file."""
    label = f"{frame:06d}"
    out_json = poses_3d / f"{label}.json"
    if skip_existing and out_json.exists():
        print(f"  frame {frame}: already triangulated, skipping")
        return True

    frame_dir = work_dir / label
    summary = build_frame_dataset.build(transforms, frame, frame_dir, set(), link=False,
                                        numeric_labels=True)
    images_dir = frame_dir / "images"

    if masks_from_alpha:
        count = write_masks(images_dir, frame_dir / "masks")
        frame_masks = frame_dir / "masks"
        if count == 0:
            raise RuntimeError(f"frame {frame}: staged no images to derive masks from")
    elif masks_dir is not None:
        frame_masks = masks_dir
    else:
        raise ValueError("give --masks_from_alpha or --masks_dir")

    kp2d_flat = frame_dir / "poses_2d_flat"
    command = [sys.executable, str(SCRIPTS / "predict_keypoints_2d.py"),
               "--images_dir", str(images_dir),
               "--out_kp2d_dir", str(kp2d_flat),
               "--fmasks_dir", str(frame_masks),
               "--sapiens_model_size", model_size]
    if checkpoint_root is not None:
        command += ["--sapiens_checkpoint_root", str(checkpoint_root)]
    if sapiens_env:
        command = ["conda", "run", "--no-capture-output", "-n", sapiens_env] + command
    run(command, f"predict_keypoints_2d (frame {frame})")

    # the splitter globs <kp2d_flat_dir>/*/*.json, and predict_keypoints_2d
    # writes <out_kp2d_dir>/<images dir name>/<camera>.json, so it wants the
    # parent rather than the images subdirectory
    kp2d = frame_dir / "poses_2d"
    run([sys.executable, str(SCRIPTS / "split_keypoints_per_camera.py"),
         "--kp2d_flat_dir", str(kp2d_flat),
         "--out_dir", str(kp2d), "--tem_label", label],
        f"split_keypoints_per_camera (frame {frame})")

    # --out_pcd_dir is nominally optional, but triangulate_skeleton.py joins it
    # unconditionally and dies with "expected str, bytes or os.PathLike object,
    # not NoneType" when it is absent. Always give it somewhere to write.
    frame_kp3d = frame_dir / "poses_3d"
    run([sys.executable, str(SCRIPTS / "triangulate_and_project_keypoints.py"),
         "--camera_path", str(frame_dir / "transforms.json"),
         "--kp2d_dir", str(kp2d),
         "--out_kp3d_dir", str(frame_kp3d),
         "--out_pcd_dir", str(frame_dir / "poses_pcd")],
        f"triangulate_and_project_keypoints (frame {frame})")

    produced = frame_kp3d / f"{label}.json"
    if not produced.exists():
        candidates = sorted(frame_kp3d.glob("*.json"))
        if not candidates:
            print(f"  frame {frame}: triangulation produced nothing", file=sys.stderr)
            return False
        produced = candidates[0]
    poses_3d.mkdir(parents=True, exist_ok=True)
    out_json.write_text(produced.read_text())
    print(f"  frame {frame}: {len(summary['cameras'])} cameras -> {out_json.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transforms", required=True, type=Path,
                    help="4D transforms.json: an entry per camera and capture frame")
    ap.add_argument("--frames", default=None, metavar="LO-HI",
                    help="capture-frame range to process (default: every frame in the rig)")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--masks_from_alpha", action="store_true",
                    help="derive each mask from the image itself, for captures whose background "
                         "is already removed")
    ap.add_argument("--masks_dir", type=Path, default=None,
                    help="existing masks directory, when the images are not pre-masked")
    ap.add_argument("--sapiens_env", default="sapiens2",
                    help="conda env for the keypoint stage; empty to run in the current env")
    ap.add_argument("--sapiens_checkpoint_root", type=Path, default=None)
    ap.add_argument("--sapiens_model_size", default="1b")
    ap.add_argument("--skip_existing", action="store_true",
                    help="leave frames that already have a 3D keypoint file alone")
    args = ap.parse_args()

    try:
        frames = frame_numbers(args.transforms, args.frames)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    work_dir = args.out_dir / "frames"
    poses_3d = args.out_dir / "poses_3d"
    print(f"{len(frames)} frames ({frames[0]}-{frames[-1]}) -> {poses_3d}")

    done, failed = 0, []
    for frame in frames:
        try:
            if process_frame(args.transforms, frame, work_dir, poses_3d,
                             sapiens_env=args.sapiens_env,
                             checkpoint_root=args.sapiens_checkpoint_root,
                             model_size=args.sapiens_model_size,
                             masks_from_alpha=args.masks_from_alpha,
                             masks_dir=args.masks_dir,
                             skip_existing=args.skip_existing):
                done += 1
            else:
                failed.append(frame)
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"ERROR on frame {frame}: {exc}", file=sys.stderr)
            failed.append(frame)

    print(f"triangulated {done}/{len(frames)} frames -> {poses_3d}")
    if failed:
        print(f"  FAILED: {failed}", file=sys.stderr)
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
