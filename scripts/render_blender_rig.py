#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
render_blender_rig.py - drive blender_render_rig.py and lay out its output.

Runs the render inside Blender, then does the CPU work in the `cumuli` env
where PIL and OpenCV live. Splitting it this way keeps the mask policy and
the image maths in a module that unit tests can reach without Blender.

Output layout under --out_dir:

    render/                    raw Blender output plus ground truth
    flipbook_src/frame_NNNN/   images_flat/<label>.png, fmasks_clean/<label>.png,
                               transforms.json  (rig cameras)
    eval_src/frame_NNNN/       the same for the eval cameras
    composite/frame_NNNN/      subject over backdrop, for the localization path
    calib/Camera_<label>.pkl   native calibration, the shape undistort_frames.py
                               and run_hloc.py read
    calib_undistorted/         zero-distortion target calibration (fisheye rigs)
    rig_gt_transforms.json     ground-truth poses and intrinsics for scoring

The flipbook layout is exactly what build_flipbook_4dgs_dataset.py consumes,
so the synthetic path joins the existing pipeline at the same point a real
capture does, with no format adapter in between.

Masks come from the render's own alpha channel. Blender writes straight
alpha, so the colour channels carry the subject's unpremultiplied colour and
the composite below is the ordinary over operator.

Fisheye rigs are undistorted here rather than by undistort_frames.py,
because that script reads images with OpenCV's colour flag and would drop
the alpha matte. The undistortion uses the same maps and the same target
matrix that undistort_frames.py --target_pkl_dir would use, and the target
calibration is written to calib_undistorted/ so the localization path can
warp the composites through that identical single-warp route.
"""

import argparse
import concurrent.futures
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_launch import relaunch_in_blender  # noqa: E402

RENDER_SCRIPT = SCRIPT_DIR / "blender_render_rig.py"


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blend", required=True, help="Normalised .blend to render")
    p.add_argument("--rig_spec", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--manifest", default=None,
                   help="scene_manifest.json (defaults to one beside --blend)")
    p.add_argument("--action", default=None)
    p.add_argument("--frame_start", type=int, default=1)
    p.add_argument("--frame_count", type=int, default=1)
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--index_offset", type=int, default=0,
                   help="Passed through: number frames from here, for a clip "
                        "split across concurrent Blender instances")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--engine", default="CYCLES")
    p.add_argument("--device", default="OPTIX")
    p.add_argument("--no_denoise", action="store_true")
    p.add_argument("--view_transform", default=None)
    p.add_argument("--background_plates", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--skip_render", action="store_true",
                   help="Reuse the images already under render/ and redo only "
                        "the CPU post step")
    p.add_argument("--undistort_balance", type=float, default=0.0,
                   help="cv2.fisheye balance for the undistorted target "
                        "matrix: 0 keeps only fully valid pixels, 1 keeps the "
                        "whole field with black borders (default 0)")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--blender", default=None)
    return p


def resolve_spec_calibration(rig_spec_path, render_dir):
    """Read a spec's calibration_pkl here and hand Blender JSON.

    Blender bundles its own numpy, so a pickle written by the pipeline env
    is not guaranteed to load there. Reading it in this process and passing
    plain JSON removes that coupling entirely."""
    spec = json.loads(Path(rig_spec_path).expanduser().read_text())
    pkl = spec.get("calibration_pkl")
    if not pkl:
        return None
    candidates = [Path(pkl).expanduser()]
    if not candidates[0].is_absolute():
        candidates = [Path(rig_spec_path).expanduser().resolve().parent / pkl,
                      SCRIPT_DIR.parent / pkl]
    for cand in candidates:
        if cand.is_file():
            with open(cand, "rb") as f:
                calib = pickle.load(f)
            payload = {
                "camera_matrix": np.asarray(calib["camera_matrix"],
                                            dtype=np.float64).tolist(),
                "distortion_coefficients": np.asarray(
                    calib.get("distortion_coefficients", []),
                    dtype=np.float64).reshape(-1).tolist(),
                "image_size": [int(v) for v in calib["image_size"]],
                "model": calib.get("model", "OPENCV_FISHEYE"),
            }
            if "sensor_width_mm" in spec.get("intrinsics", {}):
                payload["sensor_width_mm"] = spec["intrinsics"]["sensor_width_mm"]
            render_dir.mkdir(parents=True, exist_ok=True)
            out = render_dir / "spec_calibration.json"
            out.write_text(json.dumps(payload, indent=2))
            print(f"  calibration {cand.name} -> {out}")
            return out
    raise SystemExit(
        f"calibration_pkl {pkl!r} from {rig_spec_path} not found. Tried: "
        f"{[str(c) for c in candidates]}")


def forward_args(args, render_dir, calibration_json=None):
    out = ["--rig_spec", str(args.rig_spec), "--out_dir", str(render_dir),
           "--frame_start", str(args.frame_start),
           "--frame_count", str(args.frame_count),
           "--frame_step", str(args.frame_step),
           "--index_offset", str(args.index_offset),
           "--samples", str(args.samples),
           "--engine", str(args.engine), "--device", str(args.device)]
    if args.manifest:
        out += ["--manifest", str(args.manifest)]
    if args.action:
        out += ["--action", str(args.action)]
    if args.no_denoise:
        out.append("--no_denoise")
    if args.view_transform:
        out += ["--view_transform", str(args.view_transform)]
    if args.background_plates:
        out.append("--background_plates")
    if args.skip_existing:
        out.append("--skip_existing")
    if calibration_json:
        out += ["--calibration_json", str(calibration_json)]
    return out


# ---------------------------------------------------------------- calibration
def write_calibration_pkls(cameras, calib_dir):
    """Native per-camera calibration in the pkl schema the pipeline reads.

    Written here rather than inside Blender on purpose: a pickle holding
    numpy arrays is only readable by a compatible numpy, and Blender's
    bundled numpy is a different build from the one in the `cumuli` env.
    Blender emits JSON; this turns it into pickles with the reader's own
    numpy."""
    calib_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for cam in cameras:
        intr = cam["intrinsics"]
        data = {
            "camera_matrix": np.asarray(intr["camera_matrix"], dtype=np.float64),
            "distortion_coefficients": np.asarray(
                intr["distortion_coefficients"], dtype=np.float64),
            "image_size": tuple(int(v) for v in intr["render_size"]),
            "model": intr["model"],
            "rotation_vectors": None,
            "translation_vectors": None,
            "reprojection_error": None,
        }
        path = calib_dir / f"Camera_{cam['label']}.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
        written[cam["label"]] = data
    return written


def undistorted_target(calib, balance):
    """Zero-distortion pinhole calibration a fisheye camera maps onto.

    Mirrors what undistort_frames.py --target_pkl_dir consumes, so the RGBA
    warp done here and the RGB warp done by that script produce the same
    geometry."""
    import cv2

    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    D = np.asarray(calib["distortion_coefficients"],
                   dtype=np.float64).reshape(-1)[:4].reshape(4, 1)
    size = tuple(int(v) for v in calib["image_size"])
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, np.eye(3), balance=float(balance))
    return {
        "camera_matrix": np.asarray(K_new, dtype=np.float64),
        "distortion_coefficients": np.zeros(4, dtype=np.float64),
        "image_size": size,
        "model": "OPENCV",
        "rotation_vectors": None,
        "translation_vectors": None,
        "reprojection_error": None,
    }


def fisheye_maps(calib, target):
    import cv2

    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    D = np.asarray(calib["distortion_coefficients"],
                   dtype=np.float64).reshape(-1)[:4].reshape(4, 1)
    K_t = np.asarray(target["camera_matrix"], dtype=np.float64)
    size = tuple(int(v) for v in target["image_size"])
    return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_t, size,
                                               cv2.CV_16SC2)


def remap_rgba(array, maps):
    """Undistort an RGBA image, alpha included.

    undistort_frames.py reads through cv2.IMREAD_COLOR and would silently
    drop the matte, which on this path is the ground-truth mask."""
    import cv2

    map1, map2 = maps
    return cv2.remap(array, map1, map2, interpolation=cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


# --------------------------------------------------------------- post step
def split_rgba(src, image_dst, mask_dst, maps=None):
    """One rendered RGBA frame into an RGB image and its matte."""
    with Image.open(src) as im:
        rgba = np.asarray(im.convert("RGBA"))
    if maps is not None:
        rgba = remap_rgba(rgba, maps)
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    mask_dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba[..., :3]).save(image_dst, optimize=False)
    Image.fromarray(rgba[..., 3]).save(mask_dst, optimize=False)
    return image_dst


def composite_over(subject_path, plate_path, dst, maps=None):
    """Subject over backdrop, straight alpha."""
    with Image.open(subject_path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.float32)
    if maps is not None:
        rgba = remap_rgba(rgba.astype(np.uint8), maps).astype(np.float32)
    with Image.open(plate_path) as im:
        plate = np.asarray(im.convert("RGB"), dtype=np.float32)
    if plate.shape[:2] != rgba.shape[:2]:
        raise SystemExit(
            f"plate {plate_path} is {plate.shape[:2]} but the subject frame "
            f"{subject_path} is {rgba.shape[:2]}")
    alpha = rgba[..., 3:4] / 255.0
    out = rgba[..., :3] * alpha + plate * (1.0 - alpha)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(dst, optimize=False)
    return dst


def frame_transforms(cameras, intrinsics_by_label, image_ext=".png"):
    """The per-frame transforms.json build_flipbook_4dgs_dataset.py reads.

    Per-camera intrinsics with explicitly zero distortion, and an OpenGL
    camera-to-world matrix, which is the interchange convention every stage
    after the pose solve uses."""
    frames = []
    for cam in cameras:
        K, (w, h) = intrinsics_by_label[cam["label"]]
        frames.append({
            "camera_label": cam["label"],
            "file_path": f"images_flat/{cam['label']}{image_ext}",
            "transform_matrix": cam["c2w_opengl"],
            "fl_x": float(K[0, 0]), "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "w": int(w), "h": int(h),
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        })
    return {"camera_model": "OPENCV", "frames": frames}


def check_frame_coverage(render_dir, resolved):
    """Refuse to post-process a subset by accident.

    Concurrent Blender instances (--index_offset) each write
    rig_resolved.json, so the last one to start leaves a file describing
    only its own shard. Post-processing then silently covers a fraction of
    the clip and reports success, which is worse than failing. Compare the
    listed frames against what is actually on disk."""
    listed = {f["index"] for f in resolved.get("frames", [])}
    subject = render_dir / "subject"
    on_disk = {int(d.name.split("_")[1]) for d in subject.glob("frame_*")
               if d.is_dir()} if subject.is_dir() else set()
    missing = on_disk - listed
    if missing:
        raise SystemExit(
            f"{render_dir / 'rig_resolved.json'} lists {len(listed)} frame(s) "
            f"but {len(on_disk)} are rendered on disk, so {len(missing)} would "
            f"be silently skipped (first index {min(missing)}). This is what a "
            "clip rendered in concurrent shards looks like: each instance "
            "overwrote the others' metadata. Regenerate it over the whole "
            "range, then post-process:\n"
            "    blender -b <scene.blend> --python scripts/blender_render_rig.py -- \\\n"
            "        --rig_spec <spec> --out_dir <run>/render --rig_only \\\n"
            "        --frame_start <first> --frame_count <total>\n"
            "    python3 scripts/render_blender_rig.py ... --skip_render")


def post_process(args, out_dir, render_dir, resolved):
    check_frame_coverage(render_dir, resolved)
    cameras = resolved["cameras"]
    train = [c for c in cameras if c["role"] == "train"]
    evals = [c for c in cameras if c["role"] == "eval"]
    frames = resolved["frames"]

    native = write_calibration_pkls(cameras, out_dir / "calib")
    model = next(iter(native.values()))["model"] if native else "PINHOLE"

    maps_by_label = {}
    intrinsics_by_label = {}
    if model == "OPENCV_FISHEYE":
        target_dir = out_dir / "calib_undistorted"
        target_dir.mkdir(parents=True, exist_ok=True)
        for label, calib in native.items():
            target = undistorted_target(calib, args.undistort_balance)
            with open(target_dir / f"Camera_{label}.pkl", "wb") as f:
                pickle.dump(target, f)
            maps_by_label[label] = fisheye_maps(calib, target)
            intrinsics_by_label[label] = (target["camera_matrix"],
                                          target["image_size"])
        print(f"  fisheye rig: wrote undistorted target calibration for "
              f"{len(native)} camera(s) to {target_dir}")
    else:
        for label, calib in native.items():
            intrinsics_by_label[label] = (
                np.asarray(calib["camera_matrix"], dtype=np.float64),
                tuple(int(v) for v in calib["image_size"]))

    jobs = []
    for role, cams, src_sub, dst_sub in (
            ("train", train, "subject", "flipbook_src"),
            ("eval", evals, "subject_eval", "eval_src")):
        if not cams:
            continue
        for frame in frames:
            idx = frame["index"]
            frame_dir = out_dir / dst_sub / f"frame_{idx:04d}"
            for cam in cams:
                label = cam["label"]
                jobs.append((
                    render_dir / src_sub / f"frame_{idx:04d}" / f"{label}.png",
                    frame_dir / "images_flat" / f"{label}.png",
                    frame_dir / "fmasks_clean" / f"{label}.png",
                    maps_by_label.get(label)))

    missing = [j[0] for j in jobs if not j[0].exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} rendered frame(s) are missing, first "
            f"{missing[0]}. Re-run without --skip_render, or drop "
            "--skip_existing if an earlier run was interrupted.")

    print(f"  splitting {len(jobs)} rendered frame(s) into image and matte")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(split_rgba, *job) for job in jobs]
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            fut.result()
            if n % 200 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}", flush=True)

    for role, cams, dst_sub in (("train", train, "flipbook_src"),
                                ("eval", evals, "eval_src")):
        if not cams:
            continue
        payload = frame_transforms(cams, intrinsics_by_label)
        for frame in frames:
            path = out_dir / dst_sub / f"frame_{frame['index']:04d}" / "transforms.json"
            path.write_text(json.dumps(payload, indent=1))

    n_composited = composite_frames(args, out_dir, render_dir, train, frames,
                                    maps_by_label)

    gt = {
        "rig_name": resolved.get("rig_name"),
        "camera_model": model,
        "fps": resolved.get("fps"),
        "frames": frames,
        "target": resolved.get("target"),
        "cameras": [
            {
                "label": c["label"], "name": c["name"], "role": c["role"],
                "transform_matrix": c["c2w_opengl"],
                "c2w_blender": c["c2w_blender"],
                "position": c["position"],
                "native_intrinsics": c["intrinsics"],
                "fl_x": float(intrinsics_by_label[c["label"]][0][0, 0]),
                "fl_y": float(intrinsics_by_label[c["label"]][0][1, 1]),
                "cx": float(intrinsics_by_label[c["label"]][0][0, 2]),
                "cy": float(intrinsics_by_label[c["label"]][0][1, 2]),
                "w": int(intrinsics_by_label[c["label"]][1][0]),
                "h": int(intrinsics_by_label[c["label"]][1][1]),
            }
            for c in cameras
        ],
    }
    (out_dir / "rig_gt_transforms.json").write_text(json.dumps(gt, indent=2))
    shutil.copyfile(render_dir / "camera_label_map.json",
                    out_dir / "camera_label_map.json")
    return len(jobs), n_composited


def composite_frames(args, out_dir, render_dir, train, frames, maps_by_label):
    plates = render_dir / "plates"
    if not plates.is_dir():
        return 0
    jobs = []
    for frame in frames:
        idx = frame["index"]
        for cam in train:
            label = cam["label"]
            plate = plates / f"{label}.png"
            if not plate.exists():
                continue
            jobs.append((
                render_dir / "subject" / f"frame_{idx:04d}" / f"{label}.png",
                plate,
                out_dir / "composite" / f"frame_{idx:04d}" / f"{label}.png",
                maps_by_label.get(label)))
    if not jobs:
        return 0
    print(f"  compositing {len(jobs)} subject-over-backdrop frame(s)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(composite_over, *job) for job in jobs]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()
    return len(jobs)


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    render_dir = out_dir / "render"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest is None:
        guess = Path(args.blend).expanduser().resolve().parent / "scene_manifest.json"
        if guess.is_file():
            args.manifest = str(guess)
            print(f"Using scene manifest {guess}")

    started = time.time()
    if not args.skip_render:
        calibration_json = resolve_spec_calibration(args.rig_spec, render_dir)
        relaunch_in_blender(RENDER_SCRIPT,
                            forward_args(args, render_dir, calibration_json),
                            blend=Path(args.blend).expanduser().resolve(),
                            blender=args.blender)
    resolved_path = render_dir / "rig_resolved.json"
    if not resolved_path.is_file():
        raise SystemExit(
            f"{resolved_path} was not produced. The Blender run failed before "
            "it wrote its ground truth.")
    resolved = json.loads(resolved_path.read_text())

    print("Post-processing renders")
    n_split, n_comp = post_process(args, out_dir, render_dir, resolved)
    elapsed = time.time() - started

    print(f"Done in {elapsed:.1f}s: {n_split} frame(s) split, "
          f"{n_comp} composited")
    print(f"  flipbook:   {out_dir / 'flipbook_src'}")
    print(f"  eval:       {out_dir / 'eval_src'}")
    print(f"  ground truth: {out_dir / 'rig_gt_transforms.json'}")


if __name__ == "__main__":
    main()
