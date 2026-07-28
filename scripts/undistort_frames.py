#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
undistort_frames.py

Undistorts one extracted frame per camera using per-camera calibration
PKLs, writing every camera's output into a single flat directory (rather
than looping camera-by-camera in the shell). Also saves the corresponding
undistorted-calibration PKL for each camera (zero distortion, matching
intrinsics) -- these are required later by run_hloc.py.

Two modes:

1. Default: wraps deps/camera-calibration/offline_undistort.py, called
   once per camera (same resolution in and out, new K chosen by OpenCV).

2. --target_pkl_dir: single-warp mode. Builds ONE remap from the native
   fisheye frame directly to the pinhole geometry defined by an existing
   per-camera target calibration pkl (e.g. a 4k undistorted calibration
   at a different resolution). One resampling step instead of
   downscale-then-undistort, and the output intrinsics exactly match the
   target pkls -- use this when a known-good undistorted calibration
   already exists for the rig.

Calibration validation (both modes): every camera's calibration is
checked against the actual frame resolution before use. A calibration
whose image_size disagrees with the media by a uniform scale factor
(e.g. K calibrated at the GoPro 5.3K full-sensor width 5568 while the
videos are 5312 wide -- a real bug that silently warped every camera by
~113px of focal) is auto-rescaled with a loud warning, or rejected with
--no_rescale. Always read these warnings: a scaled K means the wrong
calibration source is being used.

Usage:
    python3 undistort_frames.py \\
        --frames_dir /path/to/extracted/frames \\
        --calib_dir /path/to/calibration_pkls \\
        --out_dir /path/to/undistorted \\
        --out_pkl_dir /path/to/undistorted_pkls \\
        [--model OPENCV_FISHEYE] [--target_pkl_dir /path/to/4k_undistorted]

Expects:
    frames_dir/<camera_id>.jpg          e.g. 0001.jpg, 0002.jpg, ...
    calib_dir/Camera_<camera_id>.pkl    e.g. Camera_0001.pkl, ...
    (target_pkl_dir/Camera_<camera_id>.pkl in single-warp mode)

Output:
    out_dir/<camera_id>.jpg              undistorted frames
    out_pkl_dir/Camera_<camera_id>.pkl   (zero-distortion calibration, for HLOC)
"""

import argparse
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from image_formats import SUPPORTED_IMAGE_EXTS

REPO_ROOT = Path(__file__).resolve().parent.parent
UNDISTORT_SCRIPT = REPO_ROOT / "deps" / "camera-calibration" / "offline_undistort.py"

SCALE_TOLERANCE = 0.02  # 2%: beyond this, image_size vs media disagreement is treated as a scaled-K bug


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_calib_path(calib_dir: Path, cam_id: str) -> Path:
    """Match a calibration pkl to a camera id across the naming conventions
    seen across capture rigs: Camera_<id>.pkl (this script's documented
    convention) or cam<N>_calibration_data.pkl (no zero-padding, "cam"
    prefix -- e.g. the 260521 rig's calibration_pkls_heidi). Returns the
    first candidate that exists, or the documented-convention path (for a
    clean "no calibration file at ..." error message) if neither does.
    """
    candidates = [calib_dir / f"Camera_{cam_id}.pkl"]
    try:
        candidates.append(calib_dir / f"cam{int(cam_id)}_calibration_data.pkl")
    except ValueError:
        pass
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def validate_and_rescale(calib: dict, media_w: int, media_h: int, cam_id: str, allow_rescale: bool):
    """Check calibration against the actual frame resolution.

    Returns (calib, was_rescaled). Raises ValueError when the calibration
    cannot be reconciled with the media resolution.
    """
    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    img_size = calib.get("image_size")

    if img_size is not None and None not in tuple(img_size):
        cal_w, cal_h = int(img_size[0]), int(img_size[1])
        if (cal_w, cal_h) == (media_w, media_h):
            return calib, False
        rw, rh = media_w / cal_w, media_h / cal_h
        if abs(rw - rh) > 0.01:
            raise ValueError(
                f"calibration image_size {cal_w}x{cal_h} vs media {media_w}x{media_h}: "
                f"non-uniform scale ({rw:.4f} vs {rh:.4f}) -- wrong calibration file?")
        if not allow_rescale:
            raise ValueError(
                f"calibration image_size {cal_w}x{cal_h} != media {media_w}x{media_h} "
                f"(scale {rw:.4f}) and --no_rescale is set")
        print(f"  WARNING {cam_id}: calibration was made at {cal_w}x{cal_h} but media is "
              f"{media_w}x{media_h} -- rescaling K by {rw:.4f}. If this factor is unexpected "
              f"(e.g. 5568/5312=1.0482), you are probably using a K scaled to the wrong sensor "
              f"width; prefer the calibration made at the media's native resolution.")
        calib = dict(calib)
        K = K.copy()
        K[0, :] *= rw   # fx, skew, cx
        K[1, :] *= rh   # fy, cy
        calib["camera_matrix"] = K
        calib["image_size"] = (media_w, media_h)
        return calib, True

    # No image_size stored: heuristic principal-point check only.
    cx, cy = K[0, 2], K[1, 2]
    ratio_x, ratio_y = cx / (media_w / 2), cy / (media_h / 2)
    if abs(ratio_x - 1) > 0.10 or abs(ratio_y - 1) > 0.10:
        print(f"  WARNING {cam_id}: principal point ({cx:.0f},{cy:.0f}) is far from the media "
              f"center ({media_w / 2:.0f},{media_h / 2:.0f}) (ratios {ratio_x:.3f}/{ratio_y:.3f}) "
              f"and the pkl stores no image_size -- verify this calibration matches the media "
              f"resolution (a uniformly scaled K causes systematic warp on every camera).")
    return calib, False


def single_warp_undistort(frame_path: Path, calib: dict, target: dict, model: str, out_path: Path):
    """One remap from the native (distorted) frame directly to the target pinhole geometry."""
    import cv2

    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    D = np.asarray(calib["distortion_coefficients"], dtype=np.float64)
    K_t = np.asarray(target["camera_matrix"], dtype=np.float64)
    t_w, t_h = (int(v) for v in target["image_size"])

    img = cv2.imread(str(frame_path), flags=cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"could not read {frame_path}")

    if model == "OPENCV_FISHEYE":
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_t, (t_w, t_h), cv2.CV_16SC2)
    else:
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K_t, (t_w, t_h), cv2.CV_16SC2)
    out = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    if not cv2.imwrite(str(out_path), out):
        raise ValueError(f"could not write {out_path}")
    return (t_w, t_h), K_t


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames_dir", required=True, type=Path, help="Directory of one image per camera (<camera_id>.jpg)")
    parser.add_argument("--calib_dir", required=True, type=Path, help="Directory of Camera_<camera_id>.pkl calibration files")
    parser.add_argument("--out_dir", required=True, type=Path, help="Single output directory for all undistorted frames")
    parser.add_argument("--out_pkl_dir", required=True, type=Path, help="Output directory for undistorted (zero-distortion) calibration pkls")
    parser.add_argument("--model", default="OPENCV_FISHEYE", choices=["OPENCV", "OPENCV_FISHEYE"])
    parser.add_argument("--image_ext", default=".jpg", choices=SUPPORTED_IMAGE_EXTS,
                         help="Extension of frames in frames_dir (default: .jpg)")
    parser.add_argument("--target_pkl_dir", type=Path, default=None,
                        help="Per-camera TARGET pinhole pkls (Camera_<camera_id>.pkl): enables single-warp "
                             "mode, mapping native fisheye frames directly to that geometry in one resample.")
    parser.add_argument("--no_rescale", action="store_true",
                        help="Reject (instead of auto-rescaling) calibrations whose image_size disagrees with the media.")
    args = parser.parse_args()

    if args.target_pkl_dir is None and not UNDISTORT_SCRIPT.is_file():
        print(f"Error: {UNDISTORT_SCRIPT} not found -- is the camera-calibration submodule checked out?")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_pkl_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(p for p in args.frames_dir.glob(f"*{args.image_ext}") if p.is_file())
    if not frame_paths:
        print(f"Error: no {args.image_ext} files found in {args.frames_dir}")
        sys.exit(1)

    mode = "single-warp (native -> target pinhole)" if args.target_pkl_dir else "offline_undistort.py wrapper"
    print(f"Found {len(frame_paths)} frames in {args.frames_dir} -- mode: {mode}")

    ok, failed = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        for frame_path in frame_paths:
            cam_id = frame_path.stem
            calib_path = resolve_calib_path(args.calib_dir, cam_id)
            if not calib_path.is_file():
                print(f"  SKIP {cam_id}: no calibration file at {calib_path}")
                failed.append(cam_id)
                continue

            with Image.open(frame_path) as im:
                media_w, media_h = im.size

            try:
                calib = load_pkl(calib_path)
                calib, rescaled = validate_and_rescale(calib, media_w, media_h, cam_id,
                                                       allow_rescale=not args.no_rescale)
            except ValueError as e:
                print(f"  ERROR {cam_id}: {e}")
                failed.append(cam_id)
                continue

            out_pkl_path = args.out_pkl_dir / f"Camera_{cam_id}.pkl"

            if args.target_pkl_dir is not None:
                target_path = resolve_calib_path(args.target_pkl_dir, cam_id)
                if not target_path.is_file():
                    print(f"  SKIP {cam_id}: no target pkl at {target_path}")
                    failed.append(cam_id)
                    continue
                try:
                    target = load_pkl(target_path)
                    out_path = args.out_dir / f"{cam_id}{args.image_ext}"
                    print(f"  Undistorting {cam_id} (single warp) ...")
                    (t_w, t_h), K_t = single_warp_undistort(frame_path, calib, target, args.model, out_path)
                except Exception as e:
                    print(f"  ERROR {cam_id}: {e}")
                    failed.append(cam_id)
                    continue
                undistorted_calib = {
                    "camera_matrix": K_t,
                    "distortion_coefficients": np.zeros((1, 4), dtype=np.float64),
                    "rotation_vectors": None,
                    "translation_vectors": None,
                    "reprojection_error": None,
                    "image_size": (t_w, t_h),
                    "model": "OPENCV",
                }
                with open(out_pkl_path, "wb") as f:
                    pickle.dump(undistorted_calib, f)
                ok += 1
                continue

            # wrapper mode: if the calibration was rescaled, offline_undistort.py must
            # see the corrected values, so point it at a corrected temp pkl.
            if rescaled:
                calib_path = Path(tmp) / f"Camera_{cam_id}.pkl"
                with open(calib_path, "wb") as f:
                    pickle.dump(calib, f)

            # Pass -o as a full file path (with suffix), not a bare directory:
            # offline_undistort.py's single-image mode only writes the exact
            # <camera_id>.jpg name we want when the output target has a
            # suffix. A bare directory triggers its batch-mode naming
            # (undistorted_<name>), which would silently break every
            # downstream stage expecting <camera_id>.jpg.
            out_frame_path = args.out_dir / f"{cam_id}{args.image_ext}"
            cmd = [
                sys.executable, str(UNDISTORT_SCRIPT),
                "-c", str(calib_path),
                str(frame_path),
                "-o", str(out_frame_path),
                "-u", str(out_pkl_path),
                "--model", args.model,
            ]
            print(f"  Undistorting {cam_id} ...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR undistorting {cam_id}:\n{result.stderr}")
                failed.append(cam_id)
                continue
            ok += 1

    print(f"\nDone. {ok}/{len(frame_paths)} cameras undistorted into {args.out_dir}")
    if failed:
        print(f"Failed/skipped cameras: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
