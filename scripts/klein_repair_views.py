#!/usr/bin/env python3
"""
klein_repair_views.py

Repair one frame's synthetic orbit renders with a generative image pass -- the
"repair" half of render-and-repair (see docs/render_and_repair.md). Consumes
render_orbit_views.py output, emits RGBA views ready for
build_refit_dataset.py.

Why this exists: a splat rendered from a direction no real camera covered is
plausible but soft, and in the rig's coverage gaps it is outright incoherent.
Feeding those renders back into training as-is teaches the model its own
artifacts. A low-denoise img2img pass over each render, anchored on real
photos of the subject, restores facial identity and smooths gap jank while
holding the pose, framing and silhouette the render already got right.

Method (Flux.2 Klein 9B through a running ComfyUI, --comfy_url):

  * WHOLE-IMAGE img2img at --denoise 0.15, not a face-crop detailer. The
    face-crop approach (DetailerForEach at denoise 0.3) was tried first and
    abandoned for two measured reasons: it drifted subject identity even with
    reference anchoring, and it left everything outside the detected face box
    untouched -- so coverage-gap jank always survived. A low-denoise pass over
    the whole frame holds closer to the input everywhere, face included. A
    two-pass variant (whole image, then face crop) was also tried and was
    worse on identity than the whole-image pass alone.

  * IDENTITY ANCHORS: two real photos of the subject are VAE-encoded and
    chained as ReferenceLatent onto both positive and negative conditioning.
    Prefer per-frame anchors (--reference_crops_dir with --kp2d_dir picks the
    two most head-on real head crops for THIS frame, ranked by eye separation
    x keypoint confidence) over one fixed pair for the whole clip: a fixed
    pair mismatches whatever expression and head angle the subject actually
    had, which measurably contributes to identity drift.

  * FIXED --seed across every view of every frame. The orbit is a continuous
    sweep; a varying seed makes expression and detail flicker between
    neighbouring views, which the 4D fit then has to average away.

  * Head views get --denoise_head (default 0.22), slightly stronger than the
    body pass -- they are native close-ups, so there is real facial detail to
    work with rather than upsampled pixels.

Alpha is NOT round-tripped through the generative model (it returns RGB
only). Each output pairs Klein's RGB with the ORIGINAL render's alpha, after
a floater cleanup that keeps the largest connected component plus anything
substantially attached to it.

There is deliberately no upscale step. An earlier version ran 4x ESRGAN to
6144px then downscaled to 2048 for the dataset -- two downscales after an
upscale, with the information ceiling still at the model's native 1536
output. It cost thousands of GPU calls for no measurable gain.

conda env: none (numpy + PIL + scipy; ComfyUI does the GPU work over HTTP).

Usage:
    python3 klein_repair_views.py \\
        --orbit_dir /path/to/orbit/frame_0000 \\
        --out_dir /path/to/repaired/frame_0000 \\
        --comfy_input_dir /path/to/ComfyUI/input \\
        --comfy_output_dir /path/to/ComfyUI/output \\
        [--comfy_url http://127.0.0.1:8188] \\
        [--prompt "..."] [--negative_prompt "..."] \\
        [--reference_crops_dir /path/to/frame_0000/crops --kp2d_dir /path/to/poses_2d \\
         --tem_label 000000] \\
        [--reference_images ref_a.png ref_b.png] \\
        [--denoise 0.15] [--denoise_head 0.22] [--seed 12345] [--force]

Output:
    out_dir/body_NNN.png + head_NNN.png (RGBA at the render resolution) and a
    copy of the orbit's cameras.json, which is the layout build_refit_dataset.py
    expects. Existing outputs are skipped unless --force, so the stage resumes
    cleanly after an interruption.
"""

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from clean_masks import find_keypoints_json

DEFAULT_PROMPT = ("A high resolution photograph of a person, sharp focus, realistic skin "
                  "texture, soft natural light, detailed.")
DEFAULT_NEGATIVE_PROMPT = ("blurry, deformed, extra fingers, low quality, cartoon, plastic skin, "
                           "knitted texture, oversaturated")

ALPHA_FOREGROUND_THRESHOLD = 40  # 8-bit alpha above this counts as subject coverage
ATTACHED_COMPONENT_FRACTION = 0.15  # component overlapping the dilated main mask by this much is kept
MAIN_MASK_DILATION = 12  # 3x3 iterations ~= a 25x25 square dilation, the "attached" search radius
CLOSING_ITERATIONS = 4   # 3x3 iterations ~= a 9x9 closing, seals pinholes in the silhouette
EDGE_BLUR_SIGMA = 1.0    # softens the binary edge without inventing coverage (clipped to the original)
REFERENCE_SIZE = 512     # reference crops are encoded at this size, per the validated graph
NOSE_KP, LEFT_EYE_KP, RIGHT_EYE_KP = 0, 1, 2  # goliath-308 face keypoint indices
FACE_SCORE_THRESHOLD = 0.3


class ComfyError(RuntimeError):
    pass


def clean_alpha(alpha: np.ndarray) -> np.ndarray:
    """Largest connected component of the render's alpha, plus any component
    substantially attached to it (a raised hand separated by a thin gap), with
    pinholes closed and the edge softened. Never exceeds the original alpha, so
    this can only remove coverage, never invent it."""
    foreground = alpha > ALPHA_FOREGROUND_THRESHOLD
    labels, n = ndimage.label(foreground)
    if n == 0:
        return np.zeros_like(alpha)

    sizes = ndimage.sum_labels(foreground, labels, index=np.arange(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    keep = labels == main

    if n > 1:
        nearby = ndimage.binary_dilation(keep, structure=np.ones((3, 3), bool),
                                         iterations=MAIN_MASK_DILATION)
        for label in range(1, n + 1):
            if label == main:
                continue
            component = labels == label
            if (component & nearby).sum() > ATTACHED_COMPONENT_FRACTION * sizes[label - 1]:
                keep |= component

    keep = ndimage.binary_closing(keep, structure=np.ones((3, 3), bool), iterations=CLOSING_ITERATIONS)
    softened = ndimage.gaussian_filter(keep.astype(np.float32) * 255.0, sigma=EDGE_BLUR_SIGMA)
    return np.minimum(softened, alpha.astype(np.float32)).astype(np.uint8)


def rank_reference_crops(crops_dir: Path, kp2d_dir: Path | None, tem_label: str, count: int) -> list:
    """The `count` most head-on real head crops in `crops_dir`, ranked by eye
    separation x face keypoint confidence.

    Unlike the orbit camera's continuous aiming (which triangulates in 3D),
    this is a discrete "pick the best existing photo" choice, so the cheap 2D
    heuristic is the right tool. Without --kp2d_dir, falls back to the first
    `count` crops in sorted order."""
    candidates = sorted(p for p in crops_dir.glob("head_*.png"))
    if kp2d_dir is None or not candidates:
        return candidates[:count]

    scored = []
    for crop in candidates:
        label = crop.stem[len("head_"):]
        json_path = find_keypoints_json(kp2d_dir, label)
        if json_path is None:
            continue
        specific = kp2d_dir / label / f"{tem_label}.json"
        if specific.exists():
            json_path = specific
        data = json.loads(json_path.read_text())
        instances = data.get("instance_info") or data.get("instances")
        if not instances:
            continue
        inst = instances[0]
        points = np.asarray(inst["keypoints"], dtype=np.float64)
        scores = np.asarray(inst["keypoint_scores"], dtype=np.float64)
        if len(points) <= RIGHT_EYE_KP:
            continue
        confidence = scores[[NOSE_KP, LEFT_EYE_KP, RIGHT_EYE_KP]]
        if confidence.min() < FACE_SCORE_THRESHOLD:
            continue
        bbox = inst.get("bbox")
        bbox_width = max(bbox[2] - bbox[0], 1.0) if bbox else 300.0
        eye_separation = abs(points[LEFT_EYE_KP, 0] - points[RIGHT_EYE_KP, 0]) / bbox_width
        scored.append((float(confidence.sum() * eye_separation), crop))

    scored.sort(key=lambda item: -item[0])
    return [crop for _, crop in scored[:count]] or candidates[:count]


def reference_latent_chain(base_node: str, latent_nodes: list) -> tuple:
    """Chain ReferenceLatent nodes onto one conditioning branch. Returns the new
    nodes and the id of the last one, to wire into the sampler."""
    nodes, previous = {}, base_node
    for i, latent in enumerate(latent_nodes):
        node_id = f"{base_node}_ref{i}"
        nodes[node_id] = {"class_type": "ReferenceLatent",
                          "inputs": {"conditioning": [previous, 0], "latent": [latent, 0]}}
        previous = node_id
    return nodes, previous


def build_graph(image_path: str, filename_prefix: str, reference_paths: list, *,
                prompt: str, negative_prompt: str, denoise: float, seed: int,
                steps: int, cfg: float, unet_name: str, clip_name: str, vae_name: str) -> dict:
    """ComfyUI API-format graph for one whole-image img2img repair pass."""
    graph = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "pos_txt": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg_txt": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative_prompt}},
        "in_img": {"class_type": "LoadImage", "inputs": {"image": image_path}},
    }
    graph["in_lat"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["in_img", 0], "vae": ["vae", 0]}}

    latent_ids = []
    for i, reference in enumerate(reference_paths):
        graph[f"ref{i}_img"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        graph[f"ref{i}_scale"] = {"class_type": "ImageScale",
                                  "inputs": {"image": [f"ref{i}_img", 0], "upscale_method": "nearest-exact",
                                             "width": REFERENCE_SIZE, "height": REFERENCE_SIZE, "crop": "disabled"}}
        graph[f"ref{i}_lat"] = {"class_type": "VAEEncode",
                                "inputs": {"pixels": [f"ref{i}_scale", 0], "vae": ["vae", 0]}}
        latent_ids.append(f"ref{i}_lat")

    positive_nodes, positive = reference_latent_chain("pos_txt", latent_ids)
    negative_nodes, negative = reference_latent_chain("neg_txt", latent_ids)
    graph.update(positive_nodes)
    graph.update(negative_nodes)

    graph["sampler"] = {"class_type": "KSampler", "inputs": {
        "model": ["unet", 0], "positive": [positive, 0], "negative": [negative, 0],
        "latent_image": ["in_lat", 0], "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}}
    graph["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    graph["save"] = {"class_type": "SaveImage",
                     "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix}}
    return graph


def submit(comfy_url: str, graph: dict, timeout: float) -> list:
    """Queue one graph and block until it finishes. Returns the (subfolder,
    filename) pairs it saved."""
    payload = json.dumps({"prompt": graph}).encode()
    try:
        response = urllib.request.urlopen(f"{comfy_url}/prompt", data=payload)
        prompt_id = json.loads(response.read())["prompt_id"]
    except (urllib.error.URLError, KeyError, ValueError) as e:
        raise ComfyError(f"could not queue prompt at {comfy_url}: {e}") from e

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            history = json.loads(urllib.request.urlopen(f"{comfy_url}/history/{prompt_id}").read())
        except (urllib.error.URLError, ValueError):
            time.sleep(2)
            continue
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("completed"):
                return [(image.get("subfolder", ""), image["filename"])
                        for node in entry.get("outputs", {}).values()
                        for image in node.get("images", [])]
            if status.get("status_str") == "error":
                raise ComfyError(f"ComfyUI reported an error: {status}")
        time.sleep(2)
    raise ComfyError(f"timed out after {timeout}s waiting for prompt {prompt_id}")


def free_comfy_memory(comfy_url: str) -> None:
    """Ask ComfyUI to unload models. Worth calling before handing the GPU to a
    trainer; failures are not fatal."""
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode()
    request = urllib.request.Request(f"{comfy_url}/free", data=payload,
                                     headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=30)
    except (urllib.error.URLError, OSError) as e:
        print(f"  warning: could not free ComfyUI memory: {e}")


def repair_views(orbit_dir: Path, out_dir: Path, comfy_input_dir: Path, comfy_output_dir: Path, *,
                 comfy_url: str = "http://127.0.0.1:8188", stage_tag: str = "render_repair",
                 prompt: str = DEFAULT_PROMPT, negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
                 reference_images: list | None = None, reference_crops_dir: Path | None = None,
                 kp2d_dir: Path | None = None, tem_label: str = "000000", n_references: int = 2,
                 denoise: float = 0.15, denoise_head: float = 0.22, seed: int = 12345,
                 steps: int = 8, cfg: float = 2.0, timeout: float = 600.0, force: bool = False,
                 unet_name: str = "Flux.2 Klein 9B/Flux2-Klein-9B-True-v2-bf16.safetensors",
                 clip_name: str = "qwen_3_8b_fp8mixed.safetensors",
                 vae_name: str = "flux2-vae.safetensors") -> int:
    """Repair every view in one orbit directory. Returns the number written."""
    renders = sorted(p for p in orbit_dir.glob("*.png") if p.stem.split("_")[0] in ("body", "head"))
    if not renders:
        raise ValueError(f"{orbit_dir}: no body_NNN.png / head_NNN.png renders found")

    frame_tag = orbit_dir.name
    staging = comfy_input_dir / stage_tag / frame_tag
    staging.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # identity references, staged into ComfyUI's input tree so LoadImage can reach them
    references = []
    if reference_crops_dir is not None:
        for i, crop in enumerate(rank_reference_crops(reference_crops_dir, kp2d_dir, tem_label, n_references)):
            staged = staging / f"reference_{i}.png"
            if not staged.exists():
                shutil.copy(crop, staged)
            references.append(f"{stage_tag}/{frame_tag}/{staged.name}")
    references.extend(reference_images or [])
    references = references[:n_references]
    if not references:
        print(f"  warning: {frame_tag} has no identity references -- repair will drift more")

    written = 0
    for render in renders:
        destination = out_dir / render.name
        if destination.exists() and not force:
            continue
        staged = staging / render.name
        if not staged.exists():
            shutil.copy(render, staged)

        graph = build_graph(f"{stage_tag}/{frame_tag}/{render.name}",
                            f"{stage_tag}/{frame_tag}_{render.stem}", references,
                            prompt=prompt, negative_prompt=negative_prompt,
                            denoise=denoise_head if render.stem.startswith("head") else denoise,
                            seed=seed, steps=steps, cfg=cfg,
                            unet_name=unet_name, clip_name=clip_name, vae_name=vae_name)
        saved = submit(comfy_url, graph, timeout)
        if not saved:
            raise ComfyError(f"{frame_tag}/{render.name}: ComfyUI returned no image")

        subfolder, filename = saved[-1]
        repaired = Image.open(comfy_output_dir / subfolder / filename).convert("RGB")
        original = Image.open(render).convert("RGBA")
        if repaired.size != original.size:
            repaired = repaired.resize(original.size, Image.LANCZOS)

        alpha = clean_alpha(np.asarray(original)[:, :, 3])
        Image.fromarray(np.dstack([np.asarray(repaired), alpha]), mode="RGBA").save(destination)
        written += 1

    cameras = orbit_dir / "cameras.json"
    if cameras.exists():
        shutil.copy(cameras, out_dir / "cameras.json")
    print(f"{frame_tag}: repaired {written} view(s), {len(renders) - written} already present -> {out_dir}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orbit_dir", required=True, type=Path, help="one frame of render_orbit_views.py output")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--comfy_input_dir", required=True, type=Path,
                    help="ComfyUI's input/ directory -- renders are staged here for LoadImage")
    ap.add_argument("--comfy_output_dir", required=True, type=Path,
                    help="ComfyUI's output/ directory -- repaired images are read back from here")
    ap.add_argument("--comfy_url", default="http://127.0.0.1:8188")
    ap.add_argument("--stage_tag", default="render_repair",
                    help="subfolder name used inside ComfyUI's input/ and output/ trees")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="describe the subject; a specific description holds identity better than a generic one")
    ap.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    ap.add_argument("--reference_images", nargs="*", default=None,
                    help="identity anchors as ComfyUI-input-relative paths, used when "
                         "--reference_crops_dir finds nothing (e.g. two fixed portraits)")
    ap.add_argument("--reference_crops_dir", type=Path, default=None,
                    help="this frame's real head crops (build_densification_crops.py output); "
                         "the most head-on are picked as per-frame identity anchors")
    ap.add_argument("--kp2d_dir", type=Path, default=None, help="poses_2d, used to rank the crops")
    ap.add_argument("--tem_label", default="000000")
    ap.add_argument("--n_references", type=int, default=2, help="identity anchors to chain (default 2)")
    ap.add_argument("--denoise", type=float, default=0.15,
                    help="body-pass img2img strength (default 0.15; higher drifts identity)")
    ap.add_argument("--denoise_head", type=float, default=0.22, help="head-pass strength (default 0.22)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="fixed across all views -- a varying seed flickers between neighbouring poses")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds to wait per view (default 600)")
    ap.add_argument("--force", action="store_true", help="re-repair views that already exist in --out_dir")
    ap.add_argument("--free_vram_when_done", action="store_true",
                    help="ask ComfyUI to unload models afterwards, before handing the GPU to a trainer")
    ap.add_argument("--unet_name", default="Flux.2 Klein 9B/Flux2-Klein-9B-True-v2-bf16.safetensors")
    ap.add_argument("--clip_name", default="qwen_3_8b_fp8mixed.safetensors")
    ap.add_argument("--vae_name", default="flux2-vae.safetensors")
    args = ap.parse_args()

    try:
        repair_views(args.orbit_dir, args.out_dir, args.comfy_input_dir, args.comfy_output_dir,
                     comfy_url=args.comfy_url, stage_tag=args.stage_tag,
                     prompt=args.prompt, negative_prompt=args.negative_prompt,
                     reference_images=args.reference_images, reference_crops_dir=args.reference_crops_dir,
                     kp2d_dir=args.kp2d_dir, tem_label=args.tem_label, n_references=args.n_references,
                     denoise=args.denoise, denoise_head=args.denoise_head, seed=args.seed,
                     steps=args.steps, cfg=args.cfg, timeout=args.timeout, force=args.force,
                     unet_name=args.unet_name, clip_name=args.clip_name, vae_name=args.vae_name)
    except (ComfyError, OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.free_vram_when_done:
        free_comfy_memory(args.comfy_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
