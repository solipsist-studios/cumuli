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
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from image_formats import SUPPORTED_IMAGE_EXTS

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = REPO_ROOT / "deps" / "camera-calibration"
UNDISTORT_SCRIPT = CALIBRATION_DIR / "offline_undistort.py"

# PNG compression level 1 (cv2's default is 3): several times faster to encode for a file only
# slightly larger. The frames are intermediate files that later stages read straight back.
PNG_COMPRESSION = 1

SCALE_TOLERANCE = 0.02  # 2%: beyond this, image_size vs media disagreement is treated as a scaled-K bug


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_calib_path(calib_dir: Path, cam_id: str) -> Path:
    """Match a calibration pkl to a camera id across the naming conventions
    seen across capture rigs: Camera_<id>.pkl (this script's documented
    convention) or cam<N>_calibration_data.pkl (no zero-padding, "cam"
    prefix, seen on one production rig). Returns the
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


def check_camera_matrix_sane(K: np.ndarray, label: str = "camera_matrix"):
    """Reject a malformed or degenerate camera matrix (wrong shape, NaN/Inf,
    or a non-positive focal length) loudly. Without this, a bad matrix
    silently passes through into cv2, which can produce a garbage (e.g.
    all-black, or structurally destroyed) output frame while the run still
    reports success -- applies equally to a source calibration and a
    single-warp TARGET calibration, since cv2 doesn't validate either.
    """
    if K.shape != (3, 3):
        raise ValueError(f"{label} has shape {K.shape}, expected (3, 3)")
    if not np.all(np.isfinite(K)):
        raise ValueError(f"{label} contains NaN/Inf values")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"{label} has non-positive focal length (fx={K[0, 0]}, fy={K[1, 1]})")


def validate_and_rescale(calib: dict, media_w: int, media_h: int, cam_id: str, allow_rescale: bool):
    """Check calibration against the actual frame resolution.

    Returns (calib, was_rescaled). Raises ValueError when the calibration
    cannot be reconciled with the media resolution.
    """
    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    check_camera_matrix_sane(K)

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


def write_image(path: Path, img) -> bool:
    import cv2
    params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION] if Path(path).suffix.lower() == ".png" else []
    return bool(cv2.imwrite(str(path), img, params))


def single_warp_undistort(frame_path: Path, calib: dict, target: dict, model: str, out_path: Path):
    """One remap from the native (distorted) frame directly to the target pinhole geometry."""
    import cv2

    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    D = np.asarray(calib["distortion_coefficients"], dtype=np.float64)
    K_t = np.asarray(target["camera_matrix"], dtype=np.float64)
    t_w, t_h = (int(v) for v in target["image_size"])

    # K was already checked by validate_and_rescale() in the normal
    # main() call path, but single_warp_undistort() can be called
    # directly -- re-check defensively. K_t (the TARGET matrix) never
    # goes through validate_and_rescale() at all, so this is the only
    # place it gets validated.
    check_camera_matrix_sane(K, "camera_matrix")
    check_camera_matrix_sane(K_t, "target camera_matrix")

    img = cv2.imread(str(frame_path), flags=cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"could not read {frame_path}")

    if model == "OPENCV_FISHEYE":
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_t, (t_w, t_h), cv2.CV_16SC2)
    else:
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K_t, (t_w, t_h), cv2.CV_16SC2)
    out = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    if not write_image(out_path, out):
        raise ValueError(f"could not write {out_path}")
    return (t_w, t_h), K_t


def default_jobs() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def process_camera(job: dict):
    """Undistort one camera. Returns (camera id, ok, log lines); runs in a worker process, so it prints nothing itself."""
    cam_id, frame_path = job["cam_id"], Path(job["frame_path"])
    log = []
    calib_path = resolve_calib_path(job["calib_dir"], cam_id)
    if not calib_path.is_file():
        return cam_id, False, [f"  SKIP {cam_id}: no calibration file at {calib_path}"]
    try:
        with Image.open(frame_path) as im:
            media_w, media_h = im.size
    except OSError as e:
        return cam_id, False, [f"  ERROR {cam_id}: could not read {frame_path}: {e}"]

    try:
        calib = load_pkl(calib_path)
        calib, _ = validate_and_rescale(calib, media_w, media_h, cam_id, allow_rescale=not job["no_rescale"])
    except Exception as e:
        # Broad on purpose: a calibration file can fail in ways beyond validate_and_rescale's own
        # ValueError (unreadable or truncated pickle, or one that is not the expected dict). Any of
        # these must skip just this camera, not crash the whole batch.
        return cam_id, False, [f"  ERROR {cam_id}: {e}"]

    out_pkl_path = job["out_pkl_dir"] / f"Camera_{cam_id}.pkl"
    out_path = job["out_dir"] / f"{cam_id}{job['image_ext']}"

    if job["target_pkl_dir"] is not None:
        target_path = resolve_calib_path(job["target_pkl_dir"], cam_id)
        if not target_path.is_file():
            return cam_id, False, [f"  SKIP {cam_id}: no target pkl at {target_path}"]
        try:
            target = load_pkl(target_path)
            log.append(f"  Undistorting {cam_id} (single warp) ...")
            (t_w, t_h), K_t = single_warp_undistort(frame_path, calib, target, job["model"], out_path)
        except Exception as e:
            return cam_id, False, log + [f"  ERROR {cam_id}: {e}"]
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
        return cam_id, True, log

    # Default mode: the same work offline_undistort.py does for one image, called in-process so a
    # camera costs no interpreter start-up. The model comes from the calibration file, as it does there.
    import cv2
    sys.path.insert(0, str(CALIBRATION_DIR))
    import offline_undistort as ou

    log.append(f"  Undistorting {cam_id} ...")
    try:
        mtx = np.asarray(calib["camera_matrix"], dtype=np.float64)
        dist = np.asarray(calib["distortion_coefficients"], dtype=np.float64)
        fisheye = ou.infer_model(calib.get("model"), dist) == "OPENCV_FISHEYE"
        img = cv2.imread(str(frame_path), flags=cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise ValueError(f"could not read {frame_path}")
        undistorted = ou.undistort_image(img, mtx, dist, fisheye=fisheye)
        if not write_image(out_path, undistorted):
            raise ValueError(f"could not write {out_path}")
        data = ou.build_undistorted_calibration_data(mtx, dist, (media_w, media_h), fisheye)
        with open(out_pkl_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        return cam_id, False, log + [f"  ERROR undistorting {cam_id}:\n{e}"]
    return cam_id, True, log


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
    parser.add_argument("--jobs", type=int, default=None,
                        help="Cameras to undistort at once (default: one per CPU core). Each holds a full-size frame in memory.")
    args = parser.parse_args()

    if args.target_pkl_dir is None and not UNDISTORT_SCRIPT.is_file():
        print(f"Error: {UNDISTORT_SCRIPT} not found -- is the camera-calibration submodule checked out?")
        sys.exit(1)

    for label, path in (("--out_dir", args.out_dir), ("--out_pkl_dir", args.out_pkl_dir)):
        if path.exists() and not path.is_dir():
            print(f"Error: {label} {path} exists and is not a directory")
            sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_pkl_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(p for p in args.frames_dir.glob(f"*{args.image_ext}") if p.is_file())
    if not frame_paths:
        print(f"Error: no {args.image_ext} files found in {args.frames_dir}")
        sys.exit(1)

    mode = "single-warp (native -> target pinhole)" if args.target_pkl_dir else "offline_undistort.py wrapper"
    print(f"Found {len(frame_paths)} frames in {args.frames_dir} -- mode: {mode}")

    jobs = [dict(cam_id=p.stem, frame_path=p, calib_dir=args.calib_dir, out_dir=args.out_dir, out_pkl_dir=args.out_pkl_dir,
                 model=args.model, image_ext=args.image_ext, target_pkl_dir=args.target_pkl_dir, no_rescale=args.no_rescale)
            for p in frame_paths]
    workers = max(1, min(args.jobs or default_jobs(), len(jobs)))
    print(f"Undistorting {len(jobs)} cameras with {workers} worker(s)")
    ok, failed = 0, []
    if workers == 1:
        results = map(process_camera, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(process_camera, jobs)
    for cam_id, good, lines in results:
        for line in lines:
            print(line, flush=True)
        if good:
            ok += 1
        else:
            failed.append(cam_id)

    print(f"\nDone. {ok}/{len(frame_paths)} cameras undistorted into {args.out_dir}")
    if failed:
        print(f"Failed/skipped cameras: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
