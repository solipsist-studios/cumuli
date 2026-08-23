#!/usr/bin/env python3
"""
video_repair_views.py

Repair a rendered pair sweep with a video model, as one clip rather than as
independent images. The video successor to klein_repair_views.py; see
docs/render_and_repair.md.

Why a video model: the image chain holds neighbouring views together by brute
force, one fixed seed for every view of every frame, because an image model
repairs each view blind to the others. A video model produces a coherent
sequence natively. What it needs in exchange is a sequence, which is what
render_pair_sweep.py produces.

WHAT THIS SENDS

  * The renders are the BASE, denoised only slightly (--denoise, default 0.15).
    The render is already structurally correct and, more importantly, already
    carries the real cameras' colour and exposure. Generating the subject from
    scratch instead was measured at 18.0 -> 13.3 dB on a held-out camera.
  * The two endpoints are REPLACED by the real photos that render_pair_sweep
    warped into the sweep frustum, so the clip begins and ends on real pixels
    and the model interpolates between two true anchors.
  * The skeleton maps, when given (--control_dir), ride along as the control
    video: authoritative body geometry where the render has holes or floaters.

Wan 2.2 is a two-expert model, the high-noise expert handling early denoising
steps and the low-noise expert the late ones. At the low denoise this pass uses,
sampling starts past the high-noise expert's range, so only the LOW checkpoint is
loaded. Raising --denoise past roughly 0.5 without adding the high-noise pass
would be asking one expert to do both jobs.

CLIP LENGTH: Wan encodes time in blocks of four, so a clip needs 4k+1 frames
(17, 21, 33, ...). Render the sweep with a matching --frames range; this script
refuses a length it knows the model will reject rather than letting the server
fail deep in a graph.

conda env: none (numpy + PIL; ComfyUI does the GPU work over HTTP), matching
klein_repair_views.py.

Usage:
    python3 video_repair_views.py \\
        --sweep_dir /path/to/sweeps/cam02_to_cam06 \\
        --out_dir /path/to/repaired/wan22 \\
        --comfy_input_dir /path/to/ComfyUI/input \\
        --comfy_output_dir /path/to/ComfyUI/output \\
        --prompt "A high resolution photograph of a young child in a dark navy blazer" \\
        [--control_dir /path/to/skeletons/.../kpmap/cam02_to_cam06] \\
        [--denoise 0.15] [--steps 20] [--cfg 1.0] [--seed 0]

Output:
    out_dir/sweep_NNNN.png   one repaired frame per swept frame, same names as
                             the renders, so score_novel_views.py can be pointed
                             straight at the probe index
"""

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_PROMPT = ("A high resolution photograph of a person, sharp focus, realistic skin texture, "
                  "natural lighting")
DEFAULT_NEGATIVE = ("blurry, deformed, extra limbs, low quality, cartoon, plastic skin, oversaturated, "
                    "watermark, text")
DEFAULT_MODEL = "Wan2_2-Fun-Control-A14B-LOW_fp8_e4m3fn_scaled_KJ_fixed.safetensors"
DEFAULT_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
DEFAULT_VAE = "wan_2.1_vae.safetensors"
POLL_SECONDS = 3.0
HIGH_NOISE_DENOISE = 0.5  # above this, one expert is being asked to do both jobs

# Per-backend defaults, so --backend alone names a runnable configuration and the
# bakeoff's rows differ by model rather than by how carefully each was configured.
BACKENDS = {
    "wan22": {"model": DEFAULT_MODEL, "clip": DEFAULT_CLIP, "vae": DEFAULT_VAE, "shift": 8.0},
    # LTX 2.5 ships its transformer, VAE and text encoder separately, so it loads
    # like Wan rather than as one checkpoint. No 2.5 VAE is published to this
    # install; the 2.3 distilled VAE is the stand-in, on the reasoning that both
    # releases are the same 22B distilled family and VAEs rarely change within
    # one. If the latent geometry disagrees the sampler says so immediately, and
    # the fix is the vae/ directory of huggingface.co/Lightricks/LTX-2.5.
    "ltx": {"model": "LTX 2.5/diffusion_models/"
                     "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
            "clip": "gemma_3_12B_it.safetensors",
            "vae": "ltx-2.3-22b-distilled_video_vae.safetensors", "shift": 1.0},
    "h3": {"model": "Minimax H3/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors",
           "clip": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
           "vae": "vae/minimax_h3_video_vae_fp16.safetensors", "shift": 5.0},
}


def valid_clip_length(count: int) -> bool:
    """Wan compresses time in blocks of four, so a clip needs 4k+1 frames."""
    return count >= 5 and (count - 1) % 4 == 0


def stage_frames(sweep_dir: Path, staging: Path, meta: dict) -> list:
    """Copy the swept renders into a clean directory, substituting the real
    warped photos at whichever indices carry them.

    A clean directory matters because ImageBatchPath takes everything in the
    folder it is pointed at, and the sweep directory also holds cameras.json and
    the warped real photos; batching those in would corrupt the clip order."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    pinned = {entry["idx"]: name for name, entry in (meta.get("real_endpoints") or {}).items()
              if name in ("real_first", "real_last")}
    staged = []
    for record in meta["frames"]:
        index = record["idx"]
        source = sweep_dir / (f"{pinned[index]}.png" if index in pinned else f"sweep_{index:04d}.png")
        if not source.exists():
            raise FileNotFoundError(f"{source} is missing -- re-run render_pair_sweep.py")
        destination = staging / f"{index:04d}.png"
        shutil.copyfile(source, destination)
        staged.append((index, source.name))
    return staged


def sampler_and_output(model_ref: list, positive: list, negative: list, *,
                       denoise: float, steps: int, cfg: float, seed: int,
                       sampler_name: str, scheduler: str, filename_prefix: str) -> dict:
    """The tail every backend shares: denoise the encoded renders, decode, save.

    Keeping this identical across backends is what makes the bakeoff a
    comparison. If one model were sampled differently from another, the score
    would be measuring the harness rather than the model."""
    return {
        "sampler": {"class_type": "KSampler",
                    "inputs": {"model": model_ref, "seed": seed, "steps": steps, "cfg": cfg,
                               "sampler_name": sampler_name, "scheduler": scheduler,
                               "positive": positive, "negative": negative,
                               "latent_image": ["base_latent", 0], "denoise": denoise}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix}},
    }


def loader_and_encode(render_dir: str, vae_node: dict) -> dict:
    """The head every backend shares: the renders, loaded as one batch and
    VAE-encoded into the starting latent.

    VHS_LoadImagesPath, not ImageBatchPath: the latter declares output_is_list,
    so ComfyUI maps every downstream node over one image at a time. VAEEncode
    then produced 17 single-frame latents instead of one 17-frame clip, and the
    sampler died concatenating a temporal size of 1 against the conditioning's 5."""
    return {
        "vae": vae_node,
        "renders": {"class_type": "VHS_LoadImagesPath", "inputs": {"directory": render_dir}},
        "base_latent": {"class_type": "VAEEncode",
                        "inputs": {"pixels": ["renders", 0], "vae": ["vae", 0]}},
    }


def build_graph_ltx(render_dir: str, control_dir: str | None, width: int, height: int, length: int, *,
                    prompt: str, negative: str, denoise: float, steps: int, cfg: float, seed: int,
                    model_name: str, clip_name: str, vae_name: str, shift: float,
                    filename_prefix: str) -> dict:
    """LTX 2.5, whose transformer, text encoder and VAE ship as separate files.

    LTXVConditioning stamps the clip's frame rate onto the conditioning, which
    LTX needs and which has no analogue in the other backends; without it the
    model has no idea how fast the sweep is moving."""
    graph = loader_and_encode(render_dir,
                              {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}})
    graph.update({
        "model": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "ltxv"}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "negative": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["clip", 0]}},
        "cond": {"class_type": "LTXVConditioning",
                 "inputs": {"positive": ["positive", 0], "negative": ["negative", 0],
                            "frame_rate": 29.97}},
    })
    graph.update(sampler_and_output(["model", 0], ["cond", 0], ["cond", 1], denoise=denoise,
                                    steps=steps, cfg=cfg, seed=seed, sampler_name="euler",
                                    scheduler="normal", filename_prefix=filename_prefix))
    return graph


def build_graph_h3(render_dir: str, control_dir: str | None, width: int, height: int, length: int, *,
                   prompt: str, negative: str, denoise: float, steps: int, cfg: float, seed: int,
                   model_name: str, clip_name: str, vae_name: str, shift: float,
                   filename_prefix: str) -> dict:
    """MiniMax H3, which generates video and audio jointly.

    Two consequences for a repair pass. Its conditioner emits ONE conditioning
    rather than a pair, so the negative is that conditioning zeroed out, which is
    how ComfyUI expresses "no guidance from this branch" for a model carrying no
    separate negative. And its denoising target is a PAIRED audio-video latent:
    the transformer reads `audio_src = x[1]`, so handing it a bare video latent
    from VAEEncode fails with "list index out of range".

    So the starting latent is built in two steps: MiniMaxH3ImageToVideo makes a
    well-formed AV latent, then ReplaceVideoLatentFrames swaps our encoded
    renders into its video half and leaves the audio half intact. That keeps the
    renders as what gets denoised, which is the whole point of the pass, without
    having to synthesize an audio latent we have no source for."""
    graph = loader_and_encode(render_dir,
                              {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}})
    graph.update({
        "model": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "minimax"}},
        "shift": {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": ["model", 0], "shift_video": shift, "shift_audio": shift}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
                            "width": width, "height": height, "length": length}},
        "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["cond", 0]}},
        "first": {"class_type": "VHS_LoadImagesPath", "inputs": {"directory": f"{render_dir}_first"}},
        "last": {"class_type": "VHS_LoadImagesPath", "inputs": {"directory": f"{render_dir}_last"}},
    })
    graph["cond"]["inputs"]["first_frame"] = ["first", 0]
    graph["cond"]["inputs"]["last_frame"] = ["last", 0]
    graph.update(sampler_and_output(["shift", 0], ["cond", 0], ["zero", 0], denoise=denoise,
                                    steps=steps, cfg=cfg, seed=seed, sampler_name="euler",
                                    scheduler="normal", filename_prefix=filename_prefix))
    # H3 denoises a PAIRED audio-video latent (its transformer reads
    # `audio_src = x[1]`), and that pair arrives as a NestedTensor which no
    # installed node can splice an external video latent into --
    # ReplaceVideoLatentFrames fails with "'NestedTensor' object has no attribute
    # 'clone'". So H3 cannot run the low-denoise repair the other backends run.
    # What it CAN do is its native first/last-frame job: generate the tween
    # between the two real endpoint photos. That is a different experiment, and a
    # worthwhile one, because it is the "let the model invent the in-between
    # views" proposal that render-and-repair was built to replace. Scored at the
    # same probe, it measures that proposal directly.
    graph["sampler"]["inputs"]["latent_image"] = ["cond", 1]
    return graph


def build_graph(render_dir: str, control_dir: str | None, width: int, height: int, length: int, *,
                prompt: str, negative: str, denoise: float, steps: int, cfg: float, seed: int,
                model_name: str, clip_name: str, vae_name: str, shift: float,
                filename_prefix: str) -> dict:
    """The Wan 2.2 graph.

    Shape: the renders are VAE-encoded and handed to the sampler as the starting
    latent, so a low denoise preserves them. Wan22FunControlToVideo is used for
    its CONDITIONING outputs only; its own latent is discarded, because that
    latent is what the model would generate from scratch and generating from
    scratch is the thing this pass exists to avoid."""
    graph = {
        "model": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": clip_name, "type": "wan"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "shift": {"class_type": "ModelSamplingSD3",
                  "inputs": {"model": ["model", 0], "shift": shift}},
        "positive": {"class_type": "CLIPTextEncode",
                     "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "negative": {"class_type": "CLIPTextEncode",
                     "inputs": {"text": negative, "clip": ["clip", 0]}},
        # VHS_LoadImagesPath, not ImageBatchPath: the latter declares
        # output_is_list, so ComfyUI maps every downstream node over one image at
        # a time. VAEEncode then produced 17 single-frame latents instead of one
        # 17-frame clip, and the sampler died concatenating a temporal size of 1
        # against the conditioning's 5.
        "renders": {"class_type": "VHS_LoadImagesPath", "inputs": {"directory": render_dir}},
        "base_latent": {"class_type": "VAEEncode",
                        "inputs": {"pixels": ["renders", 0], "vae": ["vae", 0]}},
        "control": {"class_type": "Wan22FunControlToVideo",
                    "inputs": {"positive": ["positive", 0], "negative": ["negative", 0],
                               "vae": ["vae", 0], "width": width, "height": height,
                               "length": length, "batch_size": 1}},
        "sampler": {"class_type": "KSampler",
                    "inputs": {"model": ["shift", 0], "seed": seed, "steps": steps, "cfg": cfg,
                               "sampler_name": "uni_pc", "scheduler": "simple",
                               "positive": ["control", 0], "negative": ["control", 1],
                               "latent_image": ["base_latent", 0], "denoise": denoise}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix}},
    }
    # control_video is not optional in practice. Left unset, Wan22FunControlToVideo
    # emits a conditioning whose concat latent is one frame long, and the sampler
    # dies concatenating it with a real clip's latent ("Expected size 1 but got
    # size 5"). With no skeleton maps to hand, the renders themselves are the
    # control signal, which is the honest fallback: structure comes from the same
    # geometry the base image already has.
    graph["control_frames"] = {"class_type": "VHS_LoadImagesPath",
                               "inputs": {"directory": control_dir or render_dir}}
    graph["control"]["inputs"]["control_video"] = ["control_frames", 0]
    return graph


def submit(comfy_url: str, graph: dict, timeout: float) -> list:
    """Queue the graph, wait for it, and return the output image filenames in
    the order the server produced them."""
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": graph, "client_id": client_id}).encode()
    request = urllib.request.Request(f"{comfy_url}/prompt", data=payload,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            prompt_id = json.load(response)["prompt_id"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"ComfyUI rejected the graph ({exc.code}): {detail}") from None

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        with urllib.request.urlopen(f"{comfy_url}/history/{prompt_id}", timeout=60) as response:
            history = json.load(response)
        if prompt_id not in history:
            continue
        entry = history[prompt_id]
        status = entry.get("status", {})
        if status.get("status_str") == "error" or status.get("completed") is False:
            messages = status.get("messages", [])
            raise RuntimeError(f"ComfyUI reported an error: {json.dumps(messages)[:2000]}")
        images = [image
                  for output in entry.get("outputs", {}).values()
                  for image in output.get("images", [])]
        if images:
            return images
        return []
    raise TimeoutError(f"ComfyUI did not finish within {timeout:.0f}s "
                       "(a 14B model's first load can take minutes; raise --timeout)")


def fetch_output(comfy_url: str, image: dict, destination: Path,
                 comfy_output_dir: Path | None = None) -> None:
    """Save one produced frame, preferring ComfyUI's /view endpoint.

    Downloading beats reading the filesystem: ComfyUI's output directory is
    configurable and need not be where the caller thinks, and /view is
    authoritative about subfolders. The local copy stays as a fallback for a
    server whose /view is unavailable."""
    query = urllib.parse.urlencode({"filename": image["filename"],
                                    "subfolder": image.get("subfolder", ""),
                                    "type": image.get("type", "output")})
    try:
        with urllib.request.urlopen(f"{comfy_url}/view?{query}", timeout=120) as response:
            destination.write_bytes(response.read())
        return
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        if comfy_output_dir is None:
            raise RuntimeError(f"could not fetch {image['filename']} from {comfy_url}: {exc}") from None
    source = comfy_output_dir / image.get("subfolder", "") / image["filename"]
    if not source.exists():
        raise FileNotFoundError(f"{source} not found and /view failed -- check --comfy_output_dir")
    shutil.copyfile(source, destination)


def repair_sweep(sweep_dir: Path, out_dir: Path, comfy_input_dir: Path, comfy_output_dir: Path, *,
                 control_dir: Path | None = None, prompt: str = DEFAULT_PROMPT,
                 negative: str = DEFAULT_NEGATIVE, denoise: float = 0.15, steps: int = 20,
                 cfg: float = 1.0, seed: int = 0, model_name: str = DEFAULT_MODEL,
                 clip_name: str = DEFAULT_CLIP, vae_name: str = DEFAULT_VAE, shift: float = 8.0,
                 comfy_url: str = "http://127.0.0.1:8188", timeout: float = 1800.0,
                 backend: str = "wan22") -> dict:
    """Repair one sweep. Returns a summary of what it wrote."""
    builder = {"wan22": build_graph, "ltx": build_graph_ltx, "h3": build_graph_h3}[backend]
    meta = json.loads((sweep_dir / "cameras.json").read_text())
    frames = meta["frames"]
    if not valid_clip_length(len(frames)):
        nearest = max(5, ((len(frames) - 1) // 4) * 4 + 1)
        raise ValueError(
            f"{sweep_dir.name} has {len(frames)} frames; Wan needs 4k+1 (…, 17, 21, 33). "
            f"Re-render the sweep with {nearest} frames, for example --frames covering "
            f"{nearest} captures")
    if denoise > HIGH_NOISE_DENOISE:
        print(f"  WARNING: --denoise {denoise} is past {HIGH_NOISE_DENOISE}, where Wan 2.2 expects "
              "its high-noise expert to run first. This graph loads only the low-noise "
              "checkpoint, so quality past that point is not what the model was built for",
              file=sys.stderr)

    staging = comfy_input_dir / f"sweep_{sweep_dir.name}"
    staged = stage_frames(sweep_dir, staging, meta)

    # H3 generates between two endpoint frames rather than denoising the clip, so
    # it needs each endpoint on its own. One image per directory, because
    # VHS_LoadImagesPath batches a directory and there is no single-image loader
    # that reads an arbitrary path.
    if backend == "h3":
        for name, index in (("first", staged[0][0]), ("last", staged[-1][0])):
            single = Path(f"{staging}_{name}")
            if single.exists():
                shutil.rmtree(single)
            single.mkdir(parents=True)
            shutil.copyfile(staging / f"{index:04d}.png", single / "0000.png")
    control_staging = None
    if control_dir is not None:
        maps = sorted(p for p in control_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".webp"))
        if len(maps) != len(frames):
            raise ValueError(f"{control_dir} holds {len(maps)} control maps for {len(frames)} frames; "
                             "run project_skeleton_conditioning.py over this same sweep")
        control_staging = comfy_input_dir / f"control_{sweep_dir.name}"
        if control_staging.exists():
            shutil.rmtree(control_staging)
        control_staging.mkdir(parents=True)
        for index, source in enumerate(maps):
            shutil.copyfile(source, control_staging / f"{index:04d}.png")

    prefix = f"videorepair_{backend}_{sweep_dir.name}"
    graph = builder(str(staging), str(control_staging) if control_staging else None,
                    meta["res"], meta["res"], len(frames),
                    prompt=prompt, negative=negative, denoise=denoise, steps=steps, cfg=cfg,
                    seed=seed, model_name=model_name, clip_name=clip_name, vae_name=vae_name,
                    shift=shift, filename_prefix=prefix)

    pinned = sorted((meta.get("real_endpoints") or {}).get(n, {}).get("idx")
                    for n in ("real_first", "real_last")
                    if (meta.get("real_endpoints") or {}).get(n))
    print(f"{sweep_dir.name} [{backend}]: {len(frames)} frames at {meta['res']}px, "
          f"denoise {denoise}, {'skeleton control' if control_dir else 'no control signal'}, "
          f"real pins at {pinned or 'NONE'}")

    produced = submit(comfy_url, graph, timeout)
    if not produced:
        raise RuntimeError("ComfyUI returned no images; check its console for the failing node")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for (index, _), image in zip(staged, produced):
        destination = out_dir / f"sweep_{index:04d}.png"
        fetch_output(comfy_url, image, destination, comfy_output_dir)
        written.append(destination)

    if len(produced) != len(frames):
        print(f"  WARNING: asked for {len(frames)} frames and got {len(produced)}; "
              "the probe index may not line up with the render's", file=sys.stderr)
    return {"frames": len(written), "out_dir": str(out_dir),
            "probe": (meta.get("probe") or {}).get("idx")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True, type=Path,
                    help="render_pair_sweep.py output directory")
    ap.add_argument("--out_dir", required=True, type=Path, help="destination for repaired frames")
    ap.add_argument("--comfy_input_dir", required=True, type=Path,
                    help="ComfyUI's input directory; staged frames are written here")
    ap.add_argument("--comfy_output_dir", required=True, type=Path,
                    help="ComfyUI's output directory, where results are collected from")
    ap.add_argument("--control_dir", type=Path, default=None,
                    help="skeleton conditioning maps for this sweep "
                         "(project_skeleton_conditioning.py --draw output)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE)
    ap.add_argument("--denoise", type=float, default=0.15,
                    help="how far from the render to travel; low keeps the real cameras' look")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="wan22", choices=sorted(BACKENDS),
                    help="which video model to repair with. Each carries its own model/clip/vae "
                         "defaults so the bakeoff's rows differ by model, not by configuration care")
    ap.add_argument("--shift", type=float, default=None, help="sigma shift (default: per backend)")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--clip_name", default=None)
    ap.add_argument("--vae_name", default=None)
    ap.add_argument("--comfy_url", default="http://127.0.0.1:8188")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="seconds to wait; a 14B model's first load alone can take minutes")
    args = ap.parse_args()

    defaults = BACKENDS[args.backend]
    try:
        summary = repair_sweep(
            args.sweep_dir, args.out_dir, args.comfy_input_dir, args.comfy_output_dir,
            control_dir=args.control_dir, prompt=args.prompt, negative=args.negative_prompt,
            denoise=args.denoise, steps=args.steps, cfg=args.cfg, seed=args.seed,
            model_name=args.model_name or defaults["model"],
            clip_name=args.clip_name or defaults["clip"],
            vae_name=args.vae_name or defaults["vae"],
            shift=args.shift if args.shift is not None else defaults["shift"],
            comfy_url=args.comfy_url, timeout=args.timeout, backend=args.backend)
    except (FileNotFoundError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"  wrote {summary['frames']} frames to {summary['out_dir']}")
    if summary["probe"] is not None:
        print(f"  score it: python3 scripts/score_novel_views.py --sweep_dir {args.sweep_dir} "
              f"--candidate wan22 {summary['out_dir']}/sweep_{summary['probe']:04d}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
