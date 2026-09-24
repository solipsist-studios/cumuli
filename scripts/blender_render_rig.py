#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
blender_render_rig.py - render a camera rig over an animation, inside Blender.

Takes a normalised scene (prepare_blender_scene.py) and a rig spec
(camera_rig_spec.py), builds the cameras, and renders three things:

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
the backdrop and the cameras are static. The driver composites subject over
plate to produce the frames the localization path needs.

Ground truth written alongside:

  rig_resolved.json      every camera's pose, intrinsics, distortion,
                         Blender settings, and the render configuration
  camera_label_map.json  label -> camera name, same shape build_flat_dataset
                         writes, so downstream tooling reads it unchanged
  joints/frame_NNNN.json armature bone positions in world space, the known
                         3D points score_poses_vs_gt.py reprojects

Run through render_blender_rig.py, which adds the CPU post step. To run it
directly:

    blender -b scene.blend --python scripts/blender_render_rig.py -- \\
        --rig_spec configs/rigs/ring16.json --out_dir /tmp/run/render \\
        --manifest scene_manifest.json --frame_start 100 --frame_count 48
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402

import camera_rig_spec as rig_spec  # noqa: E402
import blender_camera_intrinsics as bci  # noqa: E402
from blender_launch import script_argv  # noqa: E402

SUBJECT_COLLECTION = "cumuli_subject"
BACKGROUND_COLLECTION = "cumuli_background"


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rig_spec", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--manifest", default=None,
                   help="scene_manifest.json, required by subject-relative specs")
    p.add_argument("--action", default=None,
                   help="Armature action to play. Defaults to whatever the "
                        "scene already has assigned.")
    p.add_argument("--frame_start", type=int, default=1)
    p.add_argument("--frame_count", type=int, default=1)
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--index_offset", type=int, default=0,
                   help="Number the first rendered frame from here instead of "
                        "zero. Rendering is CPU-bound rather than GPU-bound "
                        "(measured: the GPU idles about 90% of the time while "
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
                   help="Also render the backdrop alone, one image per camera")
    p.add_argument("--skip_existing", action="store_true",
                   help="Leave images that are already on disk, for resuming")
    p.add_argument("--rig_only", action="store_true",
                   help="Build cameras and write ground truth, render nothing. "
                        "A seconds-long check that a spec resolves where it "
                        "was meant to, before committing an hour to rendering "
                        "it: rig_resolved.json carries every camera's pose and "
                        "intrinsics.")
    p.add_argument("--calibration_json", default=None,
                   help="Calibration to use instead of reading the spec's "
                        "calibration_pkl here. render_blender_rig.py passes "
                        "this: it reads the pickle with the pipeline env's own "
                        "numpy, which need not match Blender's bundled one.")
    return p


def make_calib_loader(spec_path, calibration_json=None):
    """Resolve a spec's `calibration_pkl` into a calibration dict.

    A JSON handoff is preferred because a pickle carrying numpy arrays is
    only guaranteed readable by a compatible numpy, and Blender bundles its
    own. Reading the pickle directly still works when the versions agree,
    which keeps this script usable on its own."""
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

    out_dir = Path(args.out_dir).expanduser().resolve()
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
            for role, sub in (("train", "plates"), ("eval", "plates_eval")):
                for cam in rig[role]:
                    obj, _ = cam_objects[cam.name]
                    path = out_dir / sub / f"{cam.label}.png"
                    if render_one(scene, obj, cam.resolution, path,
                                  args.skip_existing):
                        stats["rendered"] += 1
                    else:
                        stats["skipped"] += 1
            print(f"  wrote background plates for "
                  f"{len(rig['train']) + len(rig['eval'])} cameras")

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
    main_in_blender(build_parser().parse_args(script_argv()))


if __name__ == "__main__":
    main()
