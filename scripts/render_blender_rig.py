#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
render_blender_rig.py - render a camera rig over an animation and lay out
the flipbook.

Dual mode, like the other Blender-facing scripts (see blender_launch.py).
Run with python3 from the `cumuli` env, it re-launches itself inside
Blender to render, then does the CPU post step here, where PIL and OpenCV
live. Blender's bundled Python has neither, so the post step cannot run
inside it. Keeping that step in this process also lets unit tests reach the
mask policy and the image maths without Blender.

    python3 scripts/render_blender_rig.py --blend scene.blend \\
        --rig_spec configs/rigs/ring16.json --out_dir /tmp/run \\
        --manifest scene_manifest.json --frame_start 100 --frame_count 48

Inside Blender it takes a normalised scene (prepare_blender_scene.py) and a
rig spec (camera_rig_spec.py), builds the cameras, and renders under
<out_dir>/render/:

  subject/frame_NNNN/<label>.png       rig cameras, subject only, RGBA
  subject_eval/frame_NNNN/<label>.png  eval cameras, subject only, RGBA
  plates/<label>.png                   backdrop only, opaque (optional)

The subject pass renders on a transparent film with the background
collection hidden, so the alpha channel IS the subject matte. That matte is
exact, which is the point of the ground-truth path: no BiRefNet, no
skeleton-guided cleanup, no silhouette error to confound a comparison
between camera rigs. Blender writes straight (unassociated) alpha, measured
rather than assumed, so the colour channel is the subject's own colour even
where alpha is fractional, and hair edges are not darkened twice.

Plates are one render per camera rather than one per frame, because both
the backdrop and the cameras are static.

Ground truth written alongside, also under render/:

  rig_resolved.json      every camera's pose, intrinsics, distortion,
                         Blender settings, and the render configuration
  camera_label_map.json  label -> camera name, same shape build_flat_dataset
                         writes, so downstream tooling reads it unchanged
  joints/frame_NNNN.json armature bone positions in world space, the known
                         3D points score_poses_vs_gt.py reprojects

The post step then writes, under --out_dir:

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

Masks come from the render's own alpha channel. Because that alpha is
straight, the composite below is the ordinary over operator.

Fisheye rigs are undistorted here rather than by undistort_frames.py,
because that script reads images with OpenCV's colour flag and would drop
the alpha matte. The undistortion uses the same maps and the same target
matrix that undistort_frames.py --target_pkl_dir would use, and the target
calibration is written to calib_undistorted/ so the localization path can
warp the composites through that identical single-warp route.

--rig_only builds the cameras and writes the ground truth without rendering
or post-processing: a seconds-long check that a spec resolves where it was
meant to. --skip_render reuses the images already under render/ and redoes
only the post step.
"""

import argparse
import concurrent.futures
import gc
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Only numpy and the pure modules at module level: Blender's bundled Python
# imports this file too, and it has no PIL or OpenCV. The post-step
# functions import those themselves.
import numpy as np  # noqa: E402

import blender_camera_intrinsics as bci  # noqa: E402
import camera_rig_spec as rig_spec  # noqa: E402
from blender_launch import IN_BLENDER, relaunch_in_blender, script_argv  # noqa: E402

SUBJECT_COLLECTION = "cumuli_subject"
BACKGROUND_COLLECTION = "cumuli_background"


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blend", default=None,
                   help="Normalised .blend to render (required unless "
                        "--skip_render)")
    p.add_argument("--rig_spec", required=True)
    p.add_argument("--out_dir", required=True,
                   help="Run directory. Blender's own output goes under "
                        "<out_dir>/render")
    p.add_argument("--manifest", default=None,
                   help="scene_manifest.json, required by subject-relative "
                        "specs (defaults to one beside --blend)")
    p.add_argument("--action", default=None,
                   help="Armature action to play. Defaults to whatever the "
                        "scene already has assigned.")
    p.add_argument("--frame_start", type=int, default=1)
    p.add_argument("--frame_count", type=int, default=1)
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--index_offset", type=int, default=0,
                   help="Number the first rendered frame from here instead of "
                        "zero. Rendering is CPU-bound rather than GPU-bound "
                        "(measured: the GPU idles about 90%% of the time while "
                        "the depsgraph and BVH rebuild each frame), so splitting "
                        "a clip across concurrent Blender instances is close to "
                        "a linear speedup. Each instance needs its own offset, "
                        "or they overwrite each other's frame_NNNN directories.")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--engine", default="CYCLES")
    p.add_argument("--device", default="OPTIX",
                   choices=["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI", "CPU"])
    p.add_argument("--no_denoise", action="store_true")
    p.add_argument("--view_transform", default=None,
                   help="Colour management view transform. Defaults to the "
                        "scene's own, which is recorded in rig_resolved.json "
                        "either way so a comparison is reproducible.")
    p.add_argument("--background_plates", action="store_true",
                   help="Also render the backdrop alone, one image per rig "
                        "camera")
    p.add_argument("--skip_existing", action="store_true",
                   help="Leave images that are already on disk, for resuming")
    p.add_argument("--rig_only", action="store_true",
                   help="Build cameras and write ground truth, then stop: no "
                        "render and no post step. A seconds-long check that a "
                        "spec resolves where it was meant to, before "
                        "committing an hour to rendering it: "
                        "rig_resolved.json carries every camera's pose and "
                        "intrinsics.")
    p.add_argument("--skip_render", action="store_true",
                   help="Reuse the images already under render/ and redo only "
                        "the CPU post step")
    p.add_argument("--undistort_balance", type=float, default=0.0,
                   help="cv2.fisheye balance for the undistorted target "
                        "matrix: 0 keeps only fully valid pixels, 1 keeps the "
                        "whole field with black borders (default 0)")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--blender", default=None)
    # Set by the outer process when it re-launches this script in Blender.
    # It reads the spec's calibration_pkl with the pipeline env's own numpy,
    # which need not match Blender's bundled one, and hands over JSON.
    p.add_argument("--calibration_json", default=None, help=argparse.SUPPRESS)
    return p


# ============================================================ outside Blender
def resolve_spec_calibration(rig_spec_path, render_dir):
    """Read a spec's calibration_pkl here and hand Blender JSON.

    Blender bundles its own numpy, so a pickle written by the pipeline env
    is not guaranteed to load there. Reading it in this process and passing
    plain JSON removes that coupling entirely."""
    spec = json.loads(Path(rig_spec_path).expanduser().read_text())
    pkl = spec.get("calibration_pkl")
    if not pkl:
        return None
    try:
        calib = rig_spec.pickle_calib_loader(rig_spec_path)(pkl)
    except rig_spec.RigSpecError as e:
        raise SystemExit(f"{rig_spec_path}: {e}")
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
    print(f"  calibration {pkl} -> {out}")
    return out


def forward_args(args, out_dir, calibration_json=None):
    """The flags the Blender half of this script reads, for the re-launch."""
    out = ["--rig_spec", str(args.rig_spec), "--out_dir", str(out_dir),
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
    if args.rig_only:
        out.append("--rig_only")
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
    from PIL import Image

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
    from PIL import Image

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
            "    python3 scripts/render_blender_rig.py ... --rig_only \\\n"
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


# ============================================================== inside Blender
def make_calib_loader(spec_path, calibration_json=None):
    """Resolve a spec's `calibration_pkl` into a calibration dict.

    A JSON handoff is preferred because a pickle carrying numpy arrays is
    only guaranteed readable by a compatible numpy, and Blender bundles its
    own. Reading the pickle directly still works when the versions agree,
    which keeps a direct `blender -b ... --python` run of this script
    working."""
    if calibration_json:
        data = json.loads(Path(calibration_json).expanduser().read_text())
        data["camera_matrix"] = np.asarray(data["camera_matrix"], dtype=np.float64)
        data["distortion_coefficients"] = np.asarray(
            data["distortion_coefficients"], dtype=np.float64)
        return lambda _path: dict(data)

    return rig_spec.pickle_calib_loader(spec_path)


# ------------------------------------------------------------------- scene
def configure_engine(scene, args):
    # Assign rather than check against the enum first. Engines registered by
    # an add-on, Cycles included, do not appear in
    # bl_rna.properties["engine"].enum_items, so validating against that list
    # rejects CYCLES on a Blender that renders with it perfectly well.
    try:
        scene.render.engine = args.engine
    except TypeError:
        listed = sorted(e.identifier for e in
                        scene.render.bl_rna.properties["engine"].enum_items)
        raise SystemExit(
            f"--engine {args.engine!r} is not available in this Blender. "
            f"Built-in engines: {listed}. Cycles comes from an add-on: enable "
            "it in Preferences, or run with --factory-startup where it is on "
            "by default.")
    scene.render.use_motion_blur = False
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.use_file_extension = True
    if args.view_transform:
        scene.view_settings.view_transform = args.view_transform

    device_used = "CPU"
    if args.engine == "CYCLES":
        scene.cycles.samples = int(args.samples)
        scene.cycles.use_denoising = not args.no_denoise
        device_used = configure_cycles_device(args.device)
        scene.cycles.device = "GPU" if device_used != "CPU" else "CPU"
    return device_used


def configure_cycles_device(preferred):
    """Enable GPU rendering, falling back rather than failing.

    A rig comparison that silently runs on CPU takes hours instead of
    minutes, so the chosen backend is returned and recorded."""
    import bpy

    if preferred == "CPU":
        return "CPU"
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        print("  WARNING: the Cycles add-on is not enabled, rendering on CPU")
        return "CPU"
    prefs = addon.preferences
    for backend in (preferred, "OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        usable = [d for d in prefs.devices if d.type == backend]
        if not usable:
            continue
        for dev in prefs.devices:
            dev.use = dev.type == backend
        print(f"  Cycles device: {backend} "
              f"({', '.join(d.name for d in usable)})")
        return backend
    print("  WARNING: no GPU device found, rendering on CPU")
    return "CPU"


def set_action(action_name):
    import bpy

    if not action_name:
        return None
    action = bpy.data.actions.get(action_name)
    if action is None:
        raise SystemExit(
            f"--action {action_name!r} not found. Actions in this scene: "
            f"{sorted(a.name for a in bpy.data.actions)}")
    assigned = []
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        anim = obj.animation_data or obj.animation_data_create()
        anim.action = action
        # Blender 4.4 introduced action slots: assigning the action alone
        # leaves it unbound and the armature does not move.
        slots = getattr(action, "slots", None)
        if slots and hasattr(anim, "action_slot"):
            match = next((s for s in slots
                          if getattr(s, "target_id_type", "OBJECT") == "OBJECT"),
                         slots[0])
            anim.action_slot = match
        assigned.append(obj.name)
    print(f"  action {action_name!r} assigned to {assigned}")
    return action_name


def collection_visibility(subject_visible, background_visible):
    import bpy

    for name, visible in ((SUBJECT_COLLECTION, subject_visible),
                          (BACKGROUND_COLLECTION, background_visible)):
        coll = bpy.data.collections.get(name)
        if coll is None:
            if not visible:
                print(f"  note: no {name} collection, nothing to hide")
            continue
        coll.hide_render = not visible


# ----------------------------------------------------------------- cameras
def clear_cameras_and_lights(remove_lights):
    import bpy

    removed = 0
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA" or (remove_lights and obj.type == "LIGHT"):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    return removed


def make_camera(cam, sensor_width_mm):
    """One bpy camera from a resolved RigCamera."""
    import bpy
    from mathutils import Matrix

    settings = bci.calib_to_blender(cam.calib, render_size=cam.resolution,
                                    sensor_width_mm=sensor_width_mm)
    data = bpy.data.cameras.new(cam.name)
    for key, value in settings.items():
        if key.startswith("_"):
            continue
        setattr(data, key, value)
    obj = bpy.data.objects.new(cam.name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = Matrix([list(row) for row in cam.c2w_blender])
    return obj, settings


def make_lights(lights):
    import bpy
    from mathutils import Matrix

    created = []
    for spec in lights:
        data = bpy.data.lights.new(spec["name"], type="AREA")
        data.shape = spec.get("shape", "RECTANGLE")
        data.size = spec["size"]
        data.size_y = spec["size_y"]
        data.energy = spec["energy"]
        obj = bpy.data.objects.new(spec["name"], data)
        bpy.context.scene.collection.objects.link(obj)
        obj.matrix_world = Matrix([list(r) for r in spec["c2w_blender"]])
        created.append(obj)
    return created


# ------------------------------------------------------------------ render
def purge_render_results():
    """Drop the RENDER_RESULT datablock after every frame.

    Without this Blender's memory grows until a long multi-camera run is
    killed by the OOM reaper. Carried over from serial_render.py, where it
    was the fix for exactly that crash."""
    import bpy

    for img in list(bpy.data.images):
        if img.type == "RENDER_RESULT":
            bpy.data.images.remove(img)
    gc.collect()


def render_one(scene, cam_obj, resolution, out_path, skip_existing):
    import bpy

    if skip_existing and out_path.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.camera = cam_obj
    scene.render.resolution_x, scene.render.resolution_y = resolution
    # Blender appends the format's extension itself.
    scene.render.filepath = str(out_path.with_suffix(""))
    bpy.ops.render.render(write_still=True)
    purge_render_results()
    return True


def export_joints(scene, frame, out_path):
    """Armature bone positions in world space: the known 3D points that make
    a pose estimate scoreable without any correspondence search."""
    import bpy

    joints = {}
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        mat = obj.matrix_world
        for bone in obj.pose.bones:
            joints[f"{obj.name}/{bone.name}"] = {
                "head": list(mat @ bone.head),
                "tail": list(mat @ bone.tail),
            }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"frame": int(frame), "joints": joints}))
    return len(joints)


def write_ground_truth(out_dir, rig, cameras_meta, args, scene, device_used,
                       frames, manifest):
    import bpy

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rig_name": args_rig_name(args),
        "blender_version": bpy.app.version_string,
        "engine": scene.render.engine,
        "device": device_used,
        "samples": int(args.samples),
        "denoise": not args.no_denoise,
        "view_transform": scene.view_settings.view_transform,
        "look": getattr(scene.view_settings, "look", None),
        "film_transparent_subject_pass": True,
        "alpha_mode": "straight",
        "fps": scene.render.fps / scene.render.fps_base,
        "frames": [{"index": args.index_offset + i, "scene_frame": int(f)}
                   for i, f in enumerate(frames)],
        "target": rig["target"],
        "centre": rig["centre"],
        "scene_manifest": manifest,
        "cameras": cameras_meta,
    }
    (out_dir / "rig_resolved.json").write_text(json.dumps(payload, indent=2))

    label_map = {c["label"]: c["name"] for c in cameras_meta
                 if c["role"] == "train"}
    (out_dir / "camera_label_map.json").write_text(json.dumps(label_map, indent=2))
    return payload


def args_rig_name(args):
    return Path(args.rig_spec).stem


def main_in_blender(args):
    import bpy

    # Everything Blender writes goes under render/; the post step owns the
    # rest of --out_dir.
    out_dir = Path(args.out_dir).expanduser().resolve() / "render"
    spec = rig_spec.load_spec(args.rig_spec)
    manifest = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).expanduser().read_text())

    scene = bpy.context.scene
    print(f"Blender {bpy.app.version_string}")
    device_used = configure_engine(scene, args)
    set_action(args.action)

    loader = make_calib_loader(args.rig_spec, args.calibration_json)
    rig = rig_spec.resolve_rig(spec, manifest, calib_loader=loader)
    sensor_width = float(rig["calibration"].get("sensor_width_mm",
                                                bci.DEFAULT_SENSOR_WIDTH_MM))

    removed = clear_cameras_and_lights(remove_lights=bool(rig["lights"]))
    print(f"  removed {removed} existing camera/light object(s)")
    if rig["lights"]:
        make_lights(rig["lights"])
        print(f"  created {len(rig['lights'])} rig light(s)")

    cam_objects = {}
    cameras_meta = []
    for cam in rig["train"] + rig["eval"]:
        obj, settings = make_camera(cam, sensor_width)
        cam_objects[cam.name] = (obj, cam)
        meta = cam.as_dict()
        meta["blender_settings"] = {k: v for k, v in settings.items()
                                    if not k.startswith("_")}
        meta["intrinsics"] = settings["_meta"]
        cameras_meta.append(meta)
    print(f"  created {len(rig['train'])} rig camera(s) and "
          f"{len(rig['eval'])} eval camera(s)")

    frames = [args.frame_start + i * args.frame_step
              for i in range(max(1, args.frame_count))]
    write_ground_truth(out_dir, rig, cameras_meta, args, scene, device_used,
                       frames, manifest)
    print(f"  wrote {out_dir / 'rig_resolved.json'}")

    if args.rig_only:
        print("  --rig_only: skipping every render")
        return

    stats = {"rendered": 0, "skipped": 0, "seconds": 0.0}
    started = time.time()

    # ---- subject pass, transparent film, background hidden ----------------
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    collection_visibility(subject_visible=True, background_visible=False)

    for local_index, frame in enumerate(frames):
        index = args.index_offset + local_index
        scene.frame_set(int(frame))
        export_joints(scene, frame, out_dir / "joints" / f"frame_{index:04d}.json")
        for role, sub in (("train", "subject"), ("eval", "subject_eval")):
            for cam in rig[role]:
                obj, _ = cam_objects[cam.name]
                path = out_dir / sub / f"frame_{index:04d}" / f"{cam.label}.png"
                if render_one(scene, obj, cam.resolution, path, args.skip_existing):
                    stats["rendered"] += 1
                else:
                    stats["skipped"] += 1
        done = local_index + 1
        elapsed = time.time() - started
        print(f"  frame {done}/{len(frames)} (scene frame {frame}, index "
              f"{index}) {elapsed:.1f}s elapsed", flush=True)

    # ---- background plates, one per camera --------------------------------
    if args.background_plates:
        bg = bpy.data.collections.get(BACKGROUND_COLLECTION)
        if bg is None or not bg.all_objects:
            print("  WARNING: --background_plates requested but the scene has "
                  "no background objects. Skipping plates; the localization "
                  "path needs a backdrop for feature matching.")
        else:
            scene.render.film_transparent = False
            scene.render.image_settings.color_mode = "RGB"
            collection_visibility(subject_visible=False, background_visible=True)
            scene.frame_set(int(frames[0]))
            # Rig cameras only: plates exist to composite the subject over
            # for the localization path, and eval cameras are scored against
            # the subject alone.
            for cam in rig["train"]:
                obj, _ = cam_objects[cam.name]
                path = out_dir / "plates" / f"{cam.label}.png"
                if render_one(scene, obj, cam.resolution, path,
                              args.skip_existing):
                    stats["rendered"] += 1
                else:
                    stats["skipped"] += 1
            print(f"  wrote background plates for {len(rig['train'])} cameras")

    collection_visibility(subject_visible=True, background_visible=True)
    stats["seconds"] = time.time() - started
    n_images = max(stats["rendered"], 1)
    stats["seconds_per_image"] = stats["seconds"] / n_images
    stats["frames"] = len(frames)
    stats["train_cameras"] = len(rig["train"])
    stats["eval_cameras"] = len(rig["eval"])
    (out_dir / "render_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Rendered {stats['rendered']} image(s) in {stats['seconds']:.1f}s "
          f"({stats['seconds_per_image']:.2f}s each), skipped {stats['skipped']}")



def main():
    parser = build_parser()
    if IN_BLENDER:
        main_in_blender(parser.parse_args(script_argv()))
        return

    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    render_dir = out_dir / "render"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest is None and args.blend:
        guess = Path(args.blend).expanduser().resolve().parent / "scene_manifest.json"
        if guess.is_file():
            args.manifest = str(guess)
            print(f"Using scene manifest {guess}")

    started = time.time()
    if not args.skip_render:
        if not args.blend:
            parser.error("--blend is required unless --skip_render")
        calibration_json = resolve_spec_calibration(args.rig_spec, render_dir)
        relaunch_in_blender(Path(__file__).resolve(),
                            forward_args(args, out_dir, calibration_json),
                            blend=Path(args.blend).expanduser().resolve(),
                            blender=args.blender)
    resolved_path = render_dir / "rig_resolved.json"
    if not resolved_path.is_file():
        raise SystemExit(
            f"{resolved_path} was not produced. The Blender run failed before "
            "it wrote its ground truth.")
    if args.rig_only:
        print(f"--rig_only: wrote {resolved_path}, skipping the post step")
        return
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
