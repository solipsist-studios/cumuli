#!/usr/bin/env python3
"""
render_and_repair_sequence.py

Run the render-and-repair stages across a whole per-frame sequence: for every
frame, render the synthetic orbit and repair it. The loop driver for
render_orbit_views.py and klein_repair_views.py, in the same relationship
render_frame_sequence.py has to run_unified_pipeline.py -- it imports those
modules and calls their stage functions directly rather than reimplementing
them, so a change to a stage applies here too.

Running both stages in one process matters: the orbit renderer would otherwise
pay CUDA and gsplat startup on every frame, and staying resident also keeps
ComfyUI's models warm between repair calls.

Ordering: each frame is rendered and repaired before moving to the next, so an
interrupted run leaves a prefix of complete frames rather than every frame half
done. Both stages skip work whose output already exists, so re-running resumes.

conda env: diffuman4d (needs render_orbit_views.py's gsplat; the repair stage
itself only makes HTTP calls).

Usage:
    python3 render_and_repair_sequence.py \\
        --sequence_root /path/to/pipeline_run \\
        --transforms /path/to/transforms.json \\
        --orbit_root /path/to/orbit \\
        --repaired_root /path/to/repaired \\
        --comfy_input_dir /path/to/ComfyUI/input \\
        --comfy_output_dir /path/to/ComfyUI/output \\
        [--splat_pattern 'exports/*_maskfilt.ply'] \\
        [--frames 0-89] [--prompt "..."] [--skip_repair]

Output:
    orbit_root/<frame>/ and repaired_root/<frame>/, one directory per frame,
    laid out exactly as the single-frame scripts produce them -- which is what
    build_refit_dataset.py --repaired_root expects.
"""

import argparse
import sys
from pathlib import Path

import klein_repair_views
import render_orbit_views


def resolve_frames(sequence_root: Path, frame_range: str | None) -> list:
    """Frame directories to process, in order, optionally limited to a
    LO-HI (inclusive, by trailing frame number) range."""
    directories = sorted(p for p in sequence_root.iterdir() if p.is_dir())
    if frame_range is None:
        return directories
    try:
        lo_text, hi_text = frame_range.split("-")
        lo, hi = int(lo_text), int(hi_text)
    except ValueError:
        raise ValueError(f"--frames must look like LO-HI, got {frame_range!r}") from None

    selected = []
    for directory in directories:
        digits = ""
        for char in reversed(directory.name):
            if not char.isdigit():
                break
            digits = char + digits
        if digits and lo <= int(digits) <= hi:
            selected.append(directory)
    return selected


def find_splat(frame_dir: Path, pattern: str) -> Path | None:
    matches = sorted(frame_dir.glob(pattern))
    return matches[-1] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence_root", required=True, type=Path,
                    help="per-frame run root (render_frame_sequence.py output), one dir per frame")
    ap.add_argument("--transforms", required=True, type=Path, help="the static rig's transforms.json")
    ap.add_argument("--orbit_root", required=True, type=Path)
    ap.add_argument("--repaired_root", required=True, type=Path)
    ap.add_argument("--splat_pattern", default="exports/*_maskfilt.ply",
                    help="glob inside each frame directory for that frame's mask-filtered splat "
                         "(default 'exports/*_maskfilt.ply'; the last match in sorted order wins)")
    ap.add_argument("--frames", default=None, help="limit to a LO-HI frame-number range")
    ap.add_argument("--kp2d_subdir", default="poses_2d",
                    help="keypoint directory inside each frame dir, for head-pass aiming "
                         "and identity-reference ranking (default poses_2d)")
    ap.add_argument("--tem_label", default="000000")
    ap.add_argument("--crops_subdir", default="crops",
                    help="real head crops inside each frame dir, used as identity anchors "
                         "(build_densification_crops.py output; default 'crops')")
    ap.add_argument("--subject_anchor_pattern", default="poses_pcd_fullres/*.ply",
                    help="glob inside each frame dir for the triangulated subject cloud that "
                         "bounds the splat before rendering (default 'poses_pcd_fullres/*.ply')")
    ap.add_argument("--subject_radius", type=float, default=3.0,
                    help="radius around the subject anchor (default 3.0; set 0 to disable)")
    ap.add_argument("--skip_render", action="store_true", help="only repair existing orbit renders")
    ap.add_argument("--skip_repair", action="store_true", help="only render orbits, do not repair")

    render = ap.add_argument_group("orbit render (see render_orbit_views.py)")
    render.add_argument("--res", type=int, default=1536)
    render.add_argument("--zoom", type=float, default=1.6)
    render.add_argument("--elev_rows", type=int, default=3)
    render.add_argument("--row_azimuths", type=int, default=12)
    render.add_argument("--head_views", type=int, default=8)
    render.add_argument("--head_zoom", type=float, default=11.0)
    render.add_argument("--no_head_views", action="store_true")

    repair = ap.add_argument_group("repair (see klein_repair_views.py)")
    repair.add_argument("--comfy_input_dir", type=Path, default=None)
    repair.add_argument("--comfy_output_dir", type=Path, default=None)
    repair.add_argument("--comfy_url", default="http://127.0.0.1:8188")
    repair.add_argument("--stage_tag", default="render_repair")
    repair.add_argument("--prompt", default=klein_repair_views.DEFAULT_PROMPT)
    repair.add_argument("--negative_prompt", default=klein_repair_views.DEFAULT_NEGATIVE_PROMPT)
    repair.add_argument("--reference_images", nargs="*", default=None)
    repair.add_argument("--denoise", type=float, default=0.15)
    repair.add_argument("--denoise_head", type=float, default=0.22)
    repair.add_argument("--seed", type=int, default=12345)
    repair.add_argument("--force", action="store_true")
    repair.add_argument("--free_vram_when_done", action="store_true",
                        help="unload ComfyUI's models at the end, before handing the GPU to a trainer")
    args = ap.parse_args()

    if not args.skip_repair and (args.comfy_input_dir is None or args.comfy_output_dir is None):
        ap.error("--comfy_input_dir and --comfy_output_dir are required unless --skip_repair")

    try:
        frames = resolve_frames(args.sequence_root, args.frames)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not frames:
        print(f"Error: no frame directories under {args.sequence_root}", file=sys.stderr)
        return 1
    print(f"{len(frames)} frame(s): {frames[0].name} .. {frames[-1].name}")

    failures = []
    for index, frame_dir in enumerate(frames, start=1):
        tag = frame_dir.name
        orbit_dir = args.orbit_root / tag
        print(f"\n=== {tag} ({index}/{len(frames)})")

        try:
            if not args.skip_render:
                splat = find_splat(frame_dir, args.splat_pattern)
                if splat is None:
                    raise FileNotFoundError(f"no splat matching {args.splat_pattern!r} in {frame_dir}")
                anchors = sorted(frame_dir.glob(args.subject_anchor_pattern))
                kp2d = frame_dir / args.kp2d_subdir
                render_orbit_views.render_orbit(
                    splat, args.transforms, orbit_dir,
                    kp2d_dir=kp2d if kp2d.is_dir() else None, tem_label=args.tem_label,
                    subject_anchor_ply=anchors[0] if anchors and args.subject_radius > 0 else None,
                    subject_radius=args.subject_radius if anchors and args.subject_radius > 0 else None,
                    res=args.res, zoom=args.zoom, elev_rows=args.elev_rows,
                    row_azimuths=args.row_azimuths,
                    head_views=0 if args.no_head_views else args.head_views,
                    head_zoom=args.head_zoom)

            if not args.skip_repair:
                crops = frame_dir / args.crops_subdir
                kp2d = frame_dir / args.kp2d_subdir
                klein_repair_views.repair_views(
                    orbit_dir, args.repaired_root / tag, args.comfy_input_dir, args.comfy_output_dir,
                    comfy_url=args.comfy_url, stage_tag=args.stage_tag,
                    prompt=args.prompt, negative_prompt=args.negative_prompt,
                    reference_images=args.reference_images,
                    reference_crops_dir=crops if crops.is_dir() else None,
                    kp2d_dir=kp2d if kp2d.is_dir() else None, tem_label=args.tem_label,
                    denoise=args.denoise, denoise_head=args.denoise_head, seed=args.seed,
                    force=args.force)
        except (OSError, ValueError, KeyError, klein_repair_views.ComfyError) as e:
            print(f"  FAILED {tag}: {e}", file=sys.stderr)
            failures.append(tag)

    if args.free_vram_when_done and not args.skip_repair:
        klein_repair_views.free_comfy_memory(args.comfy_url)

    if failures:
        print(f"\n{len(failures)} frame(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nall {len(frames)} frame(s) complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
