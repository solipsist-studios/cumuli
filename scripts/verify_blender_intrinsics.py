#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
verify_blender_intrinsics.py - prove a Blender camera matches its calibration.

The question this answers is not "did the conversion run" but "does Blender
actually put a 3D point where OpenCV says it goes". Getting that wrong warps
every rendered frame in a way that looks entirely plausible: the subject is
still a person, the backdrop is still a room, and the poses are still
consistent with each other. Only the reconstruction is quietly wrong.

Method, which is a measurement rather than an argument:

  1. Choose a grid of TARGET PIXELS spread across the image.
  2. Unproject each one through the OpenCV calibration to a ray, and put a
     small emissive sphere on that ray at a fixed distance.
  3. Render with the Blender camera built by blender_camera_intrinsics.py.
  4. Find each marker's centroid and compare with the pixel it was placed
     for. Agreement means the two models describe the same lens.

The same run reports what a NAIVE PINHOLE reading of the camera would have
predicted, using `lens / sensor_width * width` as a focal length. On a
fisheye rig that number is meaningless, because a panoramic camera ignores
`lens` entirely, and the comparison shows by how much. That mistake is not
hypothetical: it is what the March Ariana export wrote into its
ground-truth transforms, leaving only the poses in that file usable.

Run it in the pipeline env; it launches Blender itself.

    python3 scripts/verify_blender_intrinsics.py \\
        --rig_spec configs/rigs/ring16_gopro_fisheye.json \\
        --out_dir /tmp/intrinsics_check --max_error_px 1.0
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402

import blender_camera_intrinsics as bci  # noqa: E402
import camera_rig_spec as rig_spec  # noqa: E402
from blender_launch import IN_BLENDER, relaunch_in_blender, script_argv  # noqa: E402

MARKER_DISTANCE = 6.0          # metres along each ray
MARKER_PIXEL_RADIUS = 4.0      # apparent radius, so a centroid is well defined


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rig_spec", default=None,
                   help="Rig spec whose camera model is checked")
    p.add_argument("--calibration_pkl", default=None,
                   help="Check this calibration directly instead of a spec")
    p.add_argument("--resolution", default=None,
                   help="WxH render size (defaults to the spec's)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--grid", default="6x5",
                   help="Marker grid, columns x rows (default 6x5)")
    p.add_argument("--inset", type=float, default=0.08,
                   help="Keep markers this fraction of the frame away from "
                        "the edge, where a fisheye's mapping is steepest and "
                        "a partly clipped marker biases its centroid")
    p.add_argument("--max_error_px", type=float, default=1.0,
                   help="Median error above this fails the check (default 1)")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--blender", default=None)
    # Passed to the in-Blender half.
    p.add_argument("--plan_json", default=None, help=argparse.SUPPRESS)
    return p


# --------------------------------------------------------------- plan (env)
def load_calibration(args):
    if args.calibration_pkl:
        import pickle
        with open(Path(args.calibration_pkl).expanduser(), "rb") as f:
            return pickle.load(f), None
    if not args.rig_spec:
        raise SystemExit("give --rig_spec or --calibration_pkl")
    spec = rig_spec.load_spec(args.rig_spec)
    calib = rig_spec.resolve_calibration(
        spec, spec["resolution"],
        calib_loader=rig_spec.pickle_calib_loader(args.rig_spec))
    return calib, spec


def target_pixels(width, height, grid, inset):
    cols, rows = (int(v) for v in grid.lower().split("x"))
    xs = np.linspace(width * inset, width * (1 - inset), cols)
    ys = np.linspace(height * inset, height * (1 - inset), rows)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float64)


def unproject(pixels, calib):
    """Target pixels to unit rays in OpenCV camera axes (x right, y down,
    z forward)."""
    import cv2

    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    dist = np.asarray(calib.get("distortion_coefficients", []),
                      dtype=np.float64).reshape(-1)
    model = calib.get("model") or bci.infer_model(dist)
    pts = pixels.reshape(-1, 1, 2)
    if model == bci.MODEL_FISHEYE:
        normalised = cv2.fisheye.undistortPoints(
            pts, K, dist[:4].reshape(4, 1)).reshape(-1, 2)
    elif model == bci.MODEL_OPENCV and np.any(np.abs(dist) > 1e-12):
        normalised = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    else:
        inv = np.linalg.inv(K)
        homog = np.concatenate([pixels, np.ones((len(pixels), 1))], axis=1)
        normalised = (inv @ homog.T).T[:, :2]
    rays = np.concatenate([normalised, np.ones((len(normalised), 1))], axis=1)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def build_plan(args):
    """Everything the Blender half needs, as plain JSON."""
    calib, spec = load_calibration(args)
    if args.resolution:
        w, h = (int(v) for v in args.resolution.lower().split("x"))
    elif spec:
        w, h = (int(v) for v in spec["resolution"])
    else:
        w, h = (int(v) for v in calib["image_size"])

    K, dist, (w, h) = bci.rescale_calibration(calib, (w, h))
    sensor_width = float(calib.get("sensor_width_mm", bci.DEFAULT_SENSOR_WIDTH_MM))
    rescaled = {"camera_matrix": K, "distortion_coefficients": dist,
                "image_size": (w, h),
                "model": calib.get("model") or bci.infer_model(dist)}
    settings = bci.calib_to_blender(rescaled, render_size=(w, h),
                                    sensor_width_mm=sensor_width)

    pixels = target_pixels(w, h, args.grid, args.inset)
    rays = unproject(pixels, rescaled)
    # Blender camera axes are x right, y UP, z BACK.
    blender_dirs = np.stack([rays[:, 0], -rays[:, 1], -rays[:, 2]], axis=1)
    radius = MARKER_DISTANCE * MARKER_PIXEL_RADIUS / float(K[0, 0])

    return {
        "resolution": [w, h],
        "samples": int(args.samples),
        "marker_distance": MARKER_DISTANCE,
        "marker_radius": float(radius),
        "camera_settings": {k: v for k, v in settings.items()
                            if not k.startswith("_")},
        "camera_model": rescaled["model"],
        "camera_matrix": np.asarray(K).tolist(),
        "distortion_coefficients": np.asarray(dist).reshape(-1).tolist(),
        "sensor_width_mm": sensor_width,
        "target_pixels": pixels.tolist(),
        "marker_directions": blender_dirs.tolist(),
        "fisheye_fit": settings["_meta"].get("fisheye_fit"),
        "image": str(Path(args.out_dir).expanduser().resolve() / "markers.png"),
    }


# ------------------------------------------------------------ render (Blender)
def render_markers(plan):
    import bpy
    from mathutils import Matrix, Vector

    w, h = plan["resolution"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = plan["samples"]
    scene.cycles.use_denoising = False
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("verify")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.0

    data = bpy.data.cameras.new("verify_cam")
    for key, value in plan["camera_settings"].items():
        setattr(data, key, value)
    cam = bpy.data.objects.new("verify_cam", data)
    scene.collection.objects.link(cam)
    cam.matrix_world = Matrix.Identity(4)     # at the origin, looking down -Z
    scene.camera = cam

    mat = bpy.data.materials.new("marker")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs[1].default_value = 500.0
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs[0], output.inputs[0])

    # Flat discs facing the camera, not spheres. A sphere's image is a conic
    # whose centroid sits OUTWARD of its projected centre, by more the
    # further off axis it lies, which showed up as a systematic 0.7 px bias
    # even on a pinhole camera whose conversion is exact by construction. A
    # disc normal to the viewing ray is symmetric about its centre, so its
    # centroid is the projection of that centre.
    for direction in plan["marker_directions"]:
        ray = Vector(direction)
        location = ray * plan["marker_distance"]
        bpy.ops.mesh.primitive_circle_add(
            radius=plan["marker_radius"], vertices=32, fill_type="NGON",
            location=location)
        disc = bpy.context.active_object
        disc.rotation_euler = ray.to_track_quat("Z", "Y").to_euler()
        disc.data.materials.append(mat)

    out = Path(plan["image"])
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out.with_suffix(""))
    bpy.ops.render.render(write_still=True)
    print(f"VERIFY_RENDER_OK {out}")


# ----------------------------------------------------------- analyse (env)
def detect_centroids(image_path, threshold=40, min_pixels=3):
    from PIL import Image
    from scipy import ndimage

    grey = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
    labelled, count = ndimage.label(grey > threshold)
    if count == 0:
        return np.zeros((0, 2)), grey.shape
    sizes = ndimage.sum(grey > threshold, labelled, range(1, count + 1))
    centres = ndimage.center_of_mass(grey, labelled, range(1, count + 1))
    keep = [(c[1], c[0]) for c, s in zip(centres, sizes) if s >= min_pixels]
    return np.array(keep, dtype=np.float64), grey.shape


def match_to_targets(detected, targets, tolerance):
    """Nearest detected centroid per target, within tolerance."""
    rows = []
    for i, target in enumerate(targets):
        if len(detected) == 0:
            rows.append({"index": i, "target": target.tolist(), "error_px": None})
            continue
        d = np.linalg.norm(detected - target, axis=1)
        j = int(np.argmin(d))
        rows.append({
            "index": i,
            "target": target.tolist(),
            "measured": detected[j].tolist() if d[j] <= tolerance else None,
            "error_px": float(d[j]) if d[j] <= tolerance else None,
        })
    return rows


def naive_pinhole_prediction(plan):
    """Where the markers would land under `lens / sensor_width * width`.

    For a PERSP camera this is the correct reading and the difference is
    zero. For a PANO camera `lens` plays no part in the projection at all,
    and the gap is the size of the mistake."""
    settings = plan["camera_settings"]
    lens = settings.get("lens")
    if lens is None:
        # A panoramic camera has no `lens` in its settings; the historical
        # mistake read the Blender default, which is 50 mm.
        lens = 50.0
    w, h = plan["resolution"]
    focal = lens / settings["sensor_width"] * w
    dirs = np.asarray(plan["marker_directions"], dtype=np.float64)
    # Back to OpenCV camera axes, then a plain pinhole projection.
    cam = np.stack([dirs[:, 0], -dirs[:, 1], -dirs[:, 2]], axis=1)
    front = cam[:, 2] > 1e-9
    z = np.where(front, cam[:, 2], 1.0)
    u = focal * cam[:, 0] / z + w / 2.0
    v = focal * cam[:, 1] / z + h / 2.0
    return np.stack([u, v], axis=1), float(focal)


def analyse(plan, args):
    targets = np.asarray(plan["target_pixels"], dtype=np.float64)
    detected, _shape = detect_centroids(plan["image"])
    tolerance = max(plan["resolution"]) * 0.05
    rows = match_to_targets(detected, targets, tolerance)
    errors = np.array([r["error_px"] for r in rows if r["error_px"] is not None])

    naive, naive_focal = naive_pinhole_prediction(plan)
    naive_gap = np.linalg.norm(naive - targets, axis=1)

    report = {
        "camera_model": plan["camera_model"],
        "resolution": plan["resolution"],
        "markers_placed": len(targets),
        "markers_found": int(len(detected)),
        "markers_matched": int(len(errors)),
        "max_error_px_allowed": args.max_error_px,
        "naive_pinhole": {
            "focal_px": naive_focal,
            "median_disagreement_px": float(np.median(naive_gap)),
            "max_disagreement_px": float(np.max(naive_gap)),
        },
        "fisheye_fit": plan.get("fisheye_fit"),
        "per_marker": rows,
    }
    if len(errors):
        report["median_error_px"] = float(np.median(errors))
        report["max_error_px"] = float(np.max(errors))
    else:
        report["median_error_px"] = None
        report["max_error_px"] = None

    unmatched = len(targets) - len(errors)
    report["passed"] = bool(
        unmatched == 0 and report["median_error_px"] is not None
        and report["median_error_px"] <= args.max_error_px)
    return report


def print_report(report):
    print(f"camera model      {report['camera_model']} at "
          f"{report['resolution'][0]}x{report['resolution'][1]}")
    print(f"markers           {report['markers_matched']} matched of "
          f"{report['markers_placed']} placed "
          f"({report['markers_found']} found in the render)")
    if report["median_error_px"] is None:
        print("projection error  no marker matched its target")
    else:
        print(f"projection error  median {report['median_error_px']:.3f} px, "
              f"max {report['max_error_px']:.3f} px "
              f"(limit {report['max_error_px_allowed']:.2f} px median)")
    naive = report["naive_pinhole"]
    print(f"naive pinhole     reading lens/sensor as a focal length would "
          f"place the same points {naive['median_disagreement_px']:.1f} px "
          f"away (median), {naive['max_disagreement_px']:.1f} px at worst")
    fit = report.get("fisheye_fit")
    if fit:
        print(f"polynomial fit    {fit['max_residual_px']:.3f} px worst-case "
              f"residual over a {fit['fit_r_max_px']:.0f} px radius, which is "
              "the floor on accuracy for a degree-4 fisheye fit")
    print("RESULT            " + ("PASS" if report["passed"] else "FAIL"))


def main():
    parser = build_parser()
    if IN_BLENDER:
        args = parser.parse_args(script_argv())
        render_markers(json.loads(Path(args.plan_json).read_text()))
        return

    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args)
    plan_path = out_dir / "verification_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))

    relaunch_in_blender(Path(__file__).resolve(),
                        ["--out_dir", str(out_dir), "--plan_json", str(plan_path)],
                        blender=args.blender)

    report = analyse(plan, args)
    report_path = out_dir / "intrinsics_verification.json"
    report_path.write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"wrote {report_path}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
