#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
prepare_blender_scene.py - normalise a character scene for synthetic capture.

This is the front door of the synthetic content pipeline. Give it a
character and an animation, in whatever shape the authoring tool exported,
and it produces one normalised .blend plus a scene_manifest.json that every
later stage reads. Adding a new character is running this script, not
editing code.

What it does:

  1. Opens a .blend, or imports an .fbx (the Daz to Character Creator to
     Blender export path).
  2. Repairs texture paths against --textures and packs every image into
     the .blend, then refuses to continue if any image is still missing.
     Authoring exports routinely carry absolute paths from another machine:
     the March Ariana blend has 146 of its 189 images pointing at a Windows
     temp directory, and an unpacked scene renders an untextured subject
     without any error at all.
  3. Sorts renderable objects into a subject collection and a background
     collection, so the render step can isolate the subject (for alpha
     mattes) or the backdrop (for plates) by toggling one collection.
  4. Sets the frame rate, turns motion blur off (a 4D splat is fitted to
     instants, and blur bakes camera-relative smear into the training
     images), and checks the unit scale.
  5. Measures the subject's world bounding box across the animation and
     writes scene_manifest.json.

The manifest is what makes rig specs portable: a spec asking for a radius
of 1.8 subject heights resolves against these numbers, so the same rig
frames a tall character and a short one correctly.

Usage (finds Blender itself, or set CUMULI_BLENDER):

    python3 scripts/prepare_blender_scene.py \\
        --input ~/Dev/datasets/ariana_src/ariana_16.blend \\
        --textures "~/Dev/datasets/ariana_src/imports/HD Ariana" \\
        --out ~/Dev/datasets/ariana_src/ariana_packed.blend \\
        --fps 24

Output:
    <out>                       normalised, fully packed .blend
    <out dir>/scene_manifest.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_launch import IN_BLENDER, relaunch_in_blender, script_argv  # noqa: E402

SUBJECT_COLLECTION = "cumuli_subject"
BACKGROUND_COLLECTION = "cumuli_background"

# Object types that put pixels on screen. Blender renamed grease pencil
# between 4.x and 5.x, so both spellings are accepted.
RENDERABLE_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME",
                    "POINTCLOUD", "GPENCIL", "GREASEPENCIL"}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".exr", ".tif", ".tiff",
              ".bmp", ".webp", ".hdr", ".psd"}

# Blender appends .001, .002 and so on to duplicate datablock names, and the
# suffix can end up inside a referenced filename. It lands at the end of the
# stem ("Foo.001") or in the middle of a compound name, which is how the
# Ariana export produced "Std_Skin_Head.001_Flow Pack.exr" for a file that
# is on disk as "Std_Skin_Head_Flow Pack.exr". Only strip it where a
# separator or the end of the stem follows, so a real "v1.001a" survives.
DUPLICATE_SUFFIX = re.compile(r"\.\d{3}(?=_|$)")


def normalise_stem(filename):
    """Comparison key for a texture filename.

    Drops the directory, the extension, and any Blender duplicate suffix,
    then lowercases. Character exports routinely disagree with their own
    .blend about all three: the March Ariana export stores
    `Eyelash1_Transparency_Opacity.jpg` while the blend asks for the same
    name ending .png, and stores `Std_Skin_Head_Flow Pack.exr` while the
    blend asks for `Std_Skin_Head.001_Flow Pack.exr`."""
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    stem = DUPLICATE_SUFFIX.sub("", stem)
    return stem.lower()


def index_texture_dir(textures_dir):
    """Map normalised stem -> list of files on disk."""
    index = {}
    root = Path(textures_dir).expanduser()
    if not root.is_dir():
        return index
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            index.setdefault(normalise_stem(path.name), []).append(path)
    return index


def pick_candidate(candidates, wanted_path):
    """Prefer the same extension, then the shortest path."""
    wanted_ext = Path(str(wanted_path).replace("\\", "/")).suffix.lower()
    return sorted(candidates,
                  key=lambda p: (p.suffix.lower() != wanted_ext, len(str(p))))[0]


def fuzzy_relink(textures_dir):
    """Repair image paths that differ only by extension or duplicate suffix.

    Blender's own find_missing_files matches filenames exactly, so it cannot
    fix either mismatch. Every substitution is printed, because guessing at
    a texture is exactly the kind of thing that should be visible in a log
    rather than inferred later from a render that looks slightly wrong."""
    import bpy

    index = index_texture_dir(textures_dir)
    if not index:
        return []
    relinked = []
    for img in bpy.data.images:
        if img.packed_file or not img.filepath:
            continue
        if img.source in {"GENERATED", "VIEWER"}:
            continue
        if Path(bpy.path.abspath(img.filepath)).exists():
            continue
        candidates = index.get(normalise_stem(img.filepath))
        if not candidates:
            continue
        chosen = pick_candidate(candidates, img.filepath)
        relinked.append((img.filepath, str(chosen)))
        img.filepath = str(chosen)
        try:
            img.reload()
        except RuntimeError as e:
            print(f"  WARNING: reload failed for {chosen}: {e}")
    return relinked


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Source .blend or .fbx (the character and animation)")
    p.add_argument("--out", required=True,
                   help="Path of the normalised .blend to write")
    p.add_argument("--textures", default=None,
                   help="Directory searched recursively for missing textures "
                        "(for example the .fbm folder beside an FBX export)")
    p.add_argument("--subject_collection", default=None,
                   help="Existing collection holding the subject. When "
                        "omitted, the subject is detected from the armature: "
                        "every object deformed by or parented under it.")
    p.add_argument("--background_collection", default=None,
                   help="Existing collection holding the backdrop. When "
                        "omitted, every renderable object that is not part of "
                        "the subject becomes background.")
    p.add_argument("--fps", type=float, default=None,
                   help="Scene frame rate. Defaults to the scene's own.")
    p.add_argument("--bbox_samples", type=int, default=8,
                   help="Animation frames sampled for the subject bounding "
                        "box (default 8). Each sample evaluates the whole "
                        "character, so raising this costs real time on a "
                        "dense figure.")
    p.add_argument("--bbox_range", default=None,
                   help="START:END frame range for the bounding box sample. "
                        "Defaults to the union of the armature action ranges.")
    p.add_argument("--no_fuzzy_textures", action="store_true",
                   help="Only accept exact filename matches when repairing "
                        "texture paths. By default a missing image also "
                        "matches a file whose name differs solely by "
                        "extension or by a Blender .001 duplicate suffix, "
                        "which is how character exports usually break. Every "
                        "such match is logged and recorded in the manifest.")
    p.add_argument("--allow_missing_textures", action="store_true",
                   help="Continue even when images are still unresolved. Off "
                        "by default: an unpacked texture renders as a silent "
                        "grey surface rather than an error.")
    p.add_argument("--blender", default=None,
                   help="Blender binary to use (default: CUMULI_BLENDER, then PATH)")
    return p


# ------------------------------------------------------------ Blender side
def relink_and_pack(textures_dir, fuzzy=True):
    """Repair image paths, then pack.

    Returns (total, packed, missing, relinked). Three passes, weakest match
    last: absolute paths, then Blender's exact-filename search, then the
    stem-based search that tolerates a swapped extension or a duplicate
    suffix."""
    import bpy

    bpy.ops.file.make_paths_absolute()
    relinked = []
    if textures_dir:
        directory = str(Path(textures_dir).expanduser().resolve())
        print(f"  searching {directory} for missing files")
        try:
            bpy.ops.file.find_missing_files(directory=directory)
        except RuntimeError as e:
            print(f"  WARNING: find_missing_files failed: {e}")
        if fuzzy:
            relinked = fuzzy_relink(directory)
            if relinked:
                print(f"  matched {len(relinked)} image(s) by name, ignoring "
                      "extension and duplicate suffix:")
                for old, new in relinked:
                    print(f"    {Path(old.replace(chr(92), '/')).name}  ->  {new}")

    try:
        bpy.ops.file.pack_all()
    except RuntimeError as e:
        # pack_all raises when at least one file is still unresolved. The
        # images that could be packed are packed regardless, so carry on and
        # let the caller decide from the counts below.
        print(f"  WARNING: pack_all reported: {e}")

    total = len(bpy.data.images)
    packed = 0
    missing = []
    for img in bpy.data.images:
        if img.packed_file:
            packed += 1
            continue
        if img.source in {"GENERATED", "VIEWER"} or not img.filepath:
            packed += 1          # nothing on disk to lose
            continue
        resolved = Path(bpy.path.abspath(img.filepath))
        if not resolved.exists():
            missing.append(img.filepath)
    return total, packed, missing, relinked


def parent_chain(obj):
    while obj is not None:
        yield obj
        obj = obj.parent


def detect_subject_objects(armatures):
    """Objects belonging to the character: the armatures themselves, meshes
    they deform, and anything parented under either."""
    import bpy

    arm_set = set(armatures)
    subject = set(arm_set)
    for obj in bpy.data.objects:
        if obj in subject:
            continue
        deformed = any(getattr(m, "object", None) in arm_set
                       for m in getattr(obj, "modifiers", []))
        parented = any(p in arm_set for p in parent_chain(obj.parent))
        if deformed or parented:
            subject.add(obj)

    # Second pass: accessories parented to a subject object rather than to
    # the armature directly (hair, jewellery, clothing pinned to a mesh).
    changed = True
    while changed:
        changed = False
        for obj in bpy.data.objects:
            if obj in subject or obj.parent is None:
                continue
            if obj.parent in subject:
                subject.add(obj)
                changed = True
    return subject


def ensure_collection(name, scene):
    import bpy
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if name not in {c.name for c in scene.collection.children}:
        scene.collection.children.link(coll)
    return coll


def move_to_collection(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def organise_collections(scene, args):
    """Put every renderable object into exactly one of the two collections.

    Exactly one matters: an object linked into both would render whenever
    either collection is visible, which quietly breaks the subject-only and
    background-only passes."""
    import bpy

    subject_coll = ensure_collection(SUBJECT_COLLECTION, scene)
    background_coll = ensure_collection(BACKGROUND_COLLECTION, scene)

    if args.subject_collection:
        named = bpy.data.collections.get(args.subject_collection)
        if named is None:
            raise SystemExit(
                f"--subject_collection {args.subject_collection!r} not found. "
                f"Collections in this scene: "
                f"{sorted(c.name for c in bpy.data.collections)}")
        subject_objs = set(named.all_objects)
    else:
        armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
        if armatures:
            subject_objs = detect_subject_objects(armatures)
            print(f"  subject detected from {len(armatures)} armature(s): "
                  f"{sorted(a.name for a in armatures)}")
        else:
            subject_objs = {o for o in bpy.data.objects
                            if o.type in RENDERABLE_TYPES}
            print("  WARNING: no armature found. Treating every renderable "
                  "object as subject and leaving the background empty. Pass "
                  "--subject_collection to split them explicitly.")

    named_bg = None
    if args.background_collection:
        named_bg = bpy.data.collections.get(args.background_collection)
        if named_bg is None:
            raise SystemExit(
                f"--background_collection {args.background_collection!r} not found")

    n_subject = n_background = 0
    for obj in list(bpy.data.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            continue                      # the render step owns these
        if obj in subject_objs:
            move_to_collection(obj, subject_coll)
            n_subject += 1
        elif obj.type in RENDERABLE_TYPES:
            if named_bg is not None and obj not in set(named_bg.all_objects):
                continue
            move_to_collection(obj, background_coll)
            n_background += 1
    return subject_coll, background_coll, n_subject, n_background


def action_ranges():
    import bpy
    out = []
    for act in bpy.data.actions:
        lo, hi = act.frame_range
        out.append({"name": act.name, "frame_range": [float(lo), float(hi)]})
    return out


def armature_info():
    import bpy
    out = []
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        action = None
        if obj.animation_data and obj.animation_data.action:
            act = obj.animation_data.action
            lo, hi = act.frame_range
            action = {"name": act.name, "frame_range": [float(lo), float(hi)]}
        out.append({"name": obj.name, "action": action})
    return out


def subject_bbox(scene, subject_coll, frames):
    """World-space bounding box of the subject over the sampled frames.

    Uses each evaluated object's bound_box rather than its vertices: the
    corners already account for modifiers, and a dense figure has hundreds
    of thousands of vertices per frame."""
    import bpy
    from mathutils import Vector

    objs = [o for o in subject_coll.all_objects if o.type in RENDERABLE_TYPES]
    if not objs:
        return None

    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    original = scene.frame_current
    sampled = []
    for frame in frames:
        scene.frame_set(int(frame))
        deps = bpy.context.evaluated_depsgraph_get()
        for obj in objs:
            ev = obj.evaluated_get(deps)
            mat = ev.matrix_world
            for corner in ev.bound_box:
                world = mat @ Vector(corner)
                for i in range(3):
                    lo[i] = min(lo[i], world[i])
                    hi[i] = max(hi[i], world[i])
        sampled.append(int(frame))
    scene.frame_set(original)

    if lo.x == float("inf"):
        return None
    return {"min": [lo.x, lo.y, lo.z], "max": [hi.x, hi.y, hi.z],
            "sampled_frames": sampled}


def bbox_frames(scene, args, armatures):
    if args.bbox_range:
        start, end = (int(v) for v in args.bbox_range.split(":"))
    else:
        ranges = [a["action"]["frame_range"] for a in armatures if a["action"]]
        if ranges:
            start = int(min(r[0] for r in ranges))
            end = int(max(r[1] for r in ranges))
        else:
            start, end = scene.frame_start, scene.frame_end
    n = max(1, int(args.bbox_samples))
    if end <= start or n == 1:
        return [start], (start, end)
    step = (end - start) / (n - 1)
    return [int(round(start + i * step)) for i in range(n)], (start, end)


def main_in_blender(args):
    import bpy

    src = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Blender {bpy.app.version_string}")
    if src.suffix.lower() == ".blend":
        print(f"Opening {src}")
        bpy.ops.wm.open_mainfile(filepath=str(src))
    elif src.suffix.lower() == ".fbx":
        print(f"Importing {src}")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if not hasattr(bpy.ops.import_scene, "fbx"):
            raise SystemExit(
                "this Blender has no FBX importer enabled. Enable the "
                "Import-Export: FBX add-on, or convert the file first.")
        bpy.ops.import_scene.fbx(filepath=str(src))
    else:
        raise SystemExit(f"--input must be .blend or .fbx, got {src.suffix!r}")

    scene = bpy.context.scene

    print("Repairing and packing textures")
    total, packed, missing, relinked = relink_and_pack(
        args.textures, fuzzy=not args.no_fuzzy_textures)
    print(f"  images: {total} total, {packed} packed, {len(missing)} missing")
    for path in missing[:10]:
        print(f"    MISSING {path}")
    if missing and not args.allow_missing_textures:
        raise SystemExit(
            f"{len(missing)} image(s) could not be resolved. Point --textures "
            "at the export's texture folder, or pass "
            "--allow_missing_textures to accept an untextured render.")

    print("Organising collections")
    subject_coll, _background_coll, n_subject, n_background = \
        organise_collections(scene, args)
    print(f"  {SUBJECT_COLLECTION}: {n_subject} objects | "
          f"{BACKGROUND_COLLECTION}: {n_background} objects")

    if args.fps is not None:
        scene.render.fps = int(round(args.fps))
        scene.render.fps_base = scene.render.fps / args.fps
    fps = scene.render.fps / scene.render.fps_base
    scene.render.use_motion_blur = False
    unit_scale = scene.unit_settings.scale_length
    if abs(unit_scale - 1.0) > 1e-6:
        print(f"  WARNING: scene unit scale is {unit_scale}, not 1.0. Rig "
              "specs and pose scoring assume one Blender unit is one metre.")

    arms = armature_info()
    frames, (range_start, range_end) = bbox_frames(scene, args, arms)
    print(f"Measuring subject bounds over {len(frames)} frame(s) in "
          f"[{range_start}, {range_end}]")
    bbox = subject_bbox(scene, subject_coll, frames)
    if bbox is None:
        raise SystemExit(
            "the subject collection holds no renderable object, so no "
            "bounding box could be measured")
    height = bbox["max"][2] - bbox["min"][2]
    print(f"  subject height {height:.4f} m, floor z {bbox['min'][2]:.4f}")

    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=False)
    print(f"Wrote {out}")

    manifest = {
        "blend": str(out),
        "source": str(src),
        "blender_version": bpy.app.version_string,
        "fps": float(fps),
        "unit_scale": float(unit_scale),
        "subject_collection": SUBJECT_COLLECTION,
        "background_collection": BACKGROUND_COLLECTION,
        "subject_objects": n_subject,
        "background_objects": n_background,
        "has_background": n_background > 0,
        "armatures": arms,
        "actions": action_ranges(),
        "frame_range": [int(range_start), int(range_end)],
        "scene_frame_range": [int(scene.frame_start), int(scene.frame_end)],
        "subject_bbox": bbox,
        "subject_height": float(height),
        "floor_z": float(bbox["min"][2]),
        "images": {"total": total, "packed": packed, "missing": len(missing),
                   "relinked_by_stem": [{"was": w, "now": n} for w, n in relinked]},
    }
    manifest_path = out.parent / "scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")


def forward_args(args):
    """Arguments to hand the in-Blender run.

    Rebuilt from the parsed namespace rather than filtered out of sys.argv,
    so a value that happens to equal a flag name cannot corrupt the command
    line."""
    out = ["--input", str(args.input), "--out", str(args.out),
           "--bbox_samples", str(args.bbox_samples)]
    if args.textures:
        out += ["--textures", str(args.textures)]
    if args.subject_collection:
        out += ["--subject_collection", str(args.subject_collection)]
    if args.background_collection:
        out += ["--background_collection", str(args.background_collection)]
    if args.fps is not None:
        out += ["--fps", str(args.fps)]
    if args.bbox_range:
        out += ["--bbox_range", str(args.bbox_range)]
    if args.allow_missing_textures:
        out.append("--allow_missing_textures")
    if args.no_fuzzy_textures:
        out.append("--no_fuzzy_textures")
    return out


def main():
    parser = build_parser()
    if IN_BLENDER:
        main_in_blender(parser.parse_args(script_argv()))
        return
    args = parser.parse_args()
    relaunch_in_blender(Path(__file__).resolve(), forward_args(args),
                        blender=args.blender)


if __name__ == "__main__":
    main()
