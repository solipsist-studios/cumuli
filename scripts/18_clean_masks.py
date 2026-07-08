#!/usr/bin/env python3
"""
18_clean_masks.py

Skeleton-guided cleanup of BiRefNet foreground masks (stage 07 output).
Runs after stage 08/09, since it needs the 2D keypoints.

Raw BiRefNet masks routinely contain non-subject blobs (tripod covers,
bystanders in the background) and occasionally DROP part of the subject
(e.g. the head) when the full-frame detection misses them. Both failure
modes silently poison masked training and make eval metrics look broken
(scoring a good model against a mask containing bystanders reads ~9-10dB).

Cleanup per camera:
  1. Keep only connected components of the mask that are hit by at least
     --min_hits confident 2D keypoints -- removes bystanders/props.
  2. If the fraction of confident keypoints INSIDE the mask is below
     --warn_coverage, part of the subject is missing. With --retry (and
     --images_dir), BiRefNet is re-run on a keypoint-guided crop around
     the subject and the result is merged in before filtering -- this
     recovers e.g. a dropped head. Without --retry it just warns.
  3. NO dilation, ever: dilated alpha supervision teaches a white
     background fringe at the silhouette.

conda env: none for the filtering; diffuman4d when --retry is used
(it invokes deps/Diffuman4D/scripts/preprocess/remove_background.py).

Usage:
    python3 18_clean_masks.py \\
        --fmasks_dir /path/to/fmasks_flat \\
        --kp2d_dir /path/to/poses_2d \\
        --out_dir /path/to/fmasks_clean \\
        [--images_dir /path/to/images_flat --retry]

Output:
    out_dir/<camera>.png cleaned masks + a cleanup_report.json with
    per-camera coverage / component stats. Point every later stage
    (and ALL eval scoring) at the cleaned masks, not the raw ones.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOVE_BG_SCRIPT = REPO_ROOT / "deps" / "Diffuman4D" / "scripts" / "preprocess" / "remove_background.py"


def find_keypoints_json(kp2d_dir: Path, cam: str):
    """Accept either stage-08 layout (<kp2d_dir>/<subdir>/<cam>.json) or
    stage-09 layout (<kp2d_dir>/<cam>/<tem>.json)."""
    cam_dir = kp2d_dir / cam
    if cam_dir.is_dir():
        jsons = sorted(cam_dir.glob("*.json"))
        if jsons:
            return jsons[0]
    matches = sorted(kp2d_dir.rglob(f"{cam}.json"))
    return matches[0] if matches else None


def load_keypoints(json_path: Path, score_thr: float):
    with open(json_path) as f:
        data = json.load(f)
    inst = data["instance_info"][0]
    kpts = np.asarray(inst["keypoints"], dtype=np.float64)
    scores = np.asarray(inst["keypoint_scores"], dtype=np.float64)
    return kpts[scores >= score_thr]


def coverage(mask_bool: np.ndarray, kpts: np.ndarray):
    """Fraction of keypoints falling on foreground pixels."""
    if len(kpts) == 0:
        return 0.0
    h, w = mask_bool.shape
    xs = np.clip(np.round(kpts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(kpts[:, 1]).astype(int), 0, h - 1)
    return float(mask_bool[ys, xs].mean())


def filter_components(mask_bool: np.ndarray, kpts: np.ndarray, min_hits: int):
    """Keep only connected components hit by >= min_hits keypoints."""
    labels, n = ndimage.label(mask_bool)
    if n == 0:
        return mask_bool, 0, 0
    h, w = mask_bool.shape
    xs = np.clip(np.round(kpts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(kpts[:, 1]).astype(int), 0, h - 1)
    hit_labels = labels[ys, xs]
    counts = np.bincount(hit_labels[hit_labels > 0], minlength=n + 1)
    keep = np.flatnonzero(counts >= min_hits)
    keep = keep[keep > 0]
    return np.isin(labels, keep), int(len(keep)), int(n)


def keypoint_crop_box(kpts: np.ndarray, w: int, h: int, margin: float):
    x0, y0 = kpts.min(axis=0)
    x1, y1 = kpts.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (int(max(x0 - mx, 0)), int(max(y0 - my, 0)),
            int(min(x1 + mx, w)), int(min(y1 + my, h)))


def run_birefnet_on_crops(crops: dict, image_ext: str):
    """crops: {cam: (image_path, box)}. Returns {cam: (crop_mask_array, box)}."""
    if not REMOVE_BG_SCRIPT.is_file():
        print(f"Error: {REMOVE_BG_SCRIPT} not found -- is the Diffuman4D submodule checked out?")
        sys.exit(1)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        masks_dir = Path(tmp) / "masks"
        crops_dir.mkdir()
        boxes = {}
        for cam, (image_path, box) in crops.items():
            with Image.open(image_path) as im:
                im.crop(box).save(crops_dir / f"{cam}{image_ext}")
            boxes[cam] = box
        cmd = [sys.executable, str(REMOVE_BG_SCRIPT), str(crops_dir), str(masks_dir),
               "--image_ext", image_ext]
        print("  Running BiRefNet retry on keypoint-guided crops:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(REPO_ROOT / "deps" / "Diffuman4D"))
        if result.returncode != 0:
            print("  ERROR: BiRefNet retry failed; continuing with original masks.")
            return {}
        for cam, box in boxes.items():
            candidates = sorted(masks_dir.rglob(f"{cam}.*"))
            if not candidates:
                print(f"  WARNING: retry produced no mask for {cam}")
                continue
            with Image.open(candidates[0]) as im:
                out[cam] = (np.asarray(im.convert("L")), box)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fmasks_dir", required=True, type=Path, help="Raw BiRefNet masks (<camera>.png)")
    parser.add_argument("--kp2d_dir", required=True, type=Path,
                        help="2D keypoints: stage-08 or stage-09 output layout")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--images_dir", type=Path, default=None, help="Original images (needed for --retry)")
    parser.add_argument("--retry", action="store_true",
                        help="Re-run BiRefNet on a keypoint-guided crop for low-coverage cameras (diffuman4d env)")
    parser.add_argument("--score_thr", type=float, default=0.5)
    parser.add_argument("--min_hits", type=int, default=1,
                        help="Keypoint hits required to keep a mask component (default 1)")
    parser.add_argument("--warn_coverage", type=float, default=0.9,
                        help="Warn (and trigger --retry) when keypoints-inside-mask fraction is below this")
    parser.add_argument("--crop_margin", type=float, default=0.15,
                        help="Margin around the keypoint bbox for the retry crop, as a fraction of its size")
    parser.add_argument("--image_ext", default=".png", help="Extension for retry crops (default .png)")
    args = parser.parse_args()

    mask_paths = sorted(p for p in args.fmasks_dir.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not mask_paths:
        print(f"Error: no masks found in {args.fmasks_dir}")
        sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: load everything, measure coverage, collect retry candidates.
    cams = {}
    retries = {}
    for mask_path in mask_paths:
        cam = mask_path.stem
        kp_json = find_keypoints_json(args.kp2d_dir, cam)
        if kp_json is None:
            print(f"  WARNING {cam}: no keypoints JSON found -- copying raw mask unfiltered.")
            with Image.open(mask_path) as im:
                im.convert("L").save(args.out_dir / f"{cam}.png")
            continue
        kpts = load_keypoints(kp_json, args.score_thr)
        with Image.open(mask_path) as im:
            mask = np.asarray(im.convert("L"))
        mask_bool = mask > 127
        cov = coverage(mask_bool, kpts)
        cams[cam] = {"mask": mask_bool, "kpts": kpts, "coverage_before": cov}
        if cov < args.warn_coverage and args.retry:
            if len(kpts) == 0:
                print(f"  WARNING {cam}: zero confident keypoints (score_thr={args.score_thr}); "
                      f"cannot compute a retry crop region, skipping retry.")
                continue
            if args.images_dir is None:
                print(f"  WARNING {cam}: coverage {cov:.2f} but --images_dir not given; cannot retry.")
                continue
            image_candidates = sorted(args.images_dir.glob(f"{cam}.*"))
            if not image_candidates:
                print(f"  WARNING {cam}: coverage {cov:.2f} but no image found in {args.images_dir}.")
                continue
            h, w = mask_bool.shape
            retries[cam] = (image_candidates[0], keypoint_crop_box(kpts, w, h, args.crop_margin))

    if retries:
        print(f"\nLow-coverage cameras needing BiRefNet crop retry: {sorted(retries)}")
        for cam, (crop_mask, box) in run_birefnet_on_crops(retries, args.image_ext).items():
            x0, y0, x1, y1 = box
            region = cams[cam]["mask"][y0:y1, x0:x1]
            cams[cam]["mask"][y0:y1, x0:x1] = region | (crop_mask[:y1 - y0, :x1 - x0] > 127)
            print(f"  {cam}: merged retry mask into crop region {box}")

    # Pass 2: component filtering + save. No dilation.
    report = {}
    for cam in sorted(cams):
        entry = cams[cam]
        cleaned, kept, total = filter_components(entry["mask"], entry["kpts"], args.min_hits)
        cov_after = coverage(cleaned, entry["kpts"])
        Image.fromarray((cleaned * 255).astype(np.uint8)).save(args.out_dir / f"{cam}.png")
        flag = ""
        if kept == 0 and total > 0:
            flag = ("  <-- MASK EMPTIED: no component had enough confident keypoint hits, "
                    "this camera's cleaned mask is now fully blank -- inspect by hand, "
                    "lower --min_hits/--score_thr, or exclude this camera")
        elif cov_after < args.warn_coverage:
            flag = ("  <-- LOW COVERAGE: part of the subject is likely missing from the mask; "
                    "try --retry, or inspect this camera's mask by hand")
        print(f"  {cam}: components {kept}/{total} kept | keypoint coverage "
              f"{entry['coverage_before']:.2f} -> {cov_after:.2f}{flag}")
        report[cam] = {
            "components_total": total,
            "components_kept": kept,
            "coverage_before": round(entry["coverage_before"], 4),
            "coverage_after": round(cov_after, 4),
            "retried": cam in retries,
        }

    report_path = args.out_dir / "cleanup_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDone. Cleaned masks in {args.out_dir}, report at {report_path}")
    print("Use the CLEANED masks for all downstream stages and eval scoring --")
    print("raw BiRefNet masks can contain bystanders and make good models look broken.")


if __name__ == "__main__":
    main()
