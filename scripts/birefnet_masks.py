#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 AND Apache-2.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
# MODIFIED FROM Diffuman4D (Copyright 2025 ZJU3DV, Apache License 2.0),
# scripts/preprocess/remove_background.py, by Solipsist Studios Inc.: this
# file changes that code (CLI, parallelism, image handling, removed options).
# The license is scripts/vendor/diffuman4d/LICENSE and the full list of
# changes is in scripts/vendor/diffuman4d/NOTICE.md.

"""
birefnet_masks.py

BiRefNet foreground masks for every image in a directory. The model comes
from Hugging Face through transformers (`trust_remote_code`, which needs
einops, timm and kornia in the env). generate_masks.py and
clean_masks.py --retry both run this script in a subprocess, so the model's
GPU memory is released when it exits.

Ported from Diffuman4D's scripts/preprocess/remove_background.py; see
scripts/vendor/diffuman4d/NOTICE.md for what changed.

conda env: cumuli

Usage:
    python3 birefnet_masks.py IMAGES_DIR OUT_FMASKS_DIR \\
        [--image_ext .png] [--batch_size 8] [--max_subjects N]

Output: OUT_FMASKS_DIR/<same relative path>.png, one 8-bit mask per image
(images are found recursively).
"""

from __future__ import annotations

import argparse
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# torch, torchvision, tqdm and transformers are imported where they are
# used, so the mask post-processing here stays importable (and unit
# testable) without the model stack.

IMAGE_SIZE = (1024, 1024)


@functools.cache
def transform_image():
    """BiRefNet's own preprocessing (https://huggingface.co/ZhengPeng7/BiRefNet)."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
    torch.set_float32_matmul_precision("high")
    model.to(device)
    model.eval()
    if torch.device(device).type == "cuda":
        model.half()  # fp16 is a CUDA-only speedup; CPU does not reliably support it
    else:
        # transformers 5 honours the checkpoint's own fp16 dtype at load
        # (transformers 4 upcast to float32 silently), so on CPU the model
        # arrives half and float32 inputs crash with a dtype mismatch.
        # Cast back explicitly; skipping .half() alone is not enough.
        model.float()
    return model


def extract_masks(images: list[Image.Image], model, sema: threading.Semaphore) -> list[Image.Image]:
    import torch
    from torchvision.transforms.functional import to_pil_image

    batch = torch.stack([transform_image()(image) for image in images]).to(model.device)
    if batch.device.type == "cuda":
        batch = batch.half()
    with sema, torch.no_grad():
        preds = model(batch)[-1].sigmoid().cpu()
    return [to_pil_image(pred).resize(image.size).convert("L") for image, pred in zip(images, preds)]


def keep_top_subject_bboxes(fmask: Image.Image, max_subjects: int) -> Image.Image:
    """Zero the mask outside the bounding boxes of its max_subjects most
    confident connected components."""
    if max_subjects <= 0:
        return Image.new("L", fmask.size, 0)

    mask = np.asarray(fmask, dtype=np.uint8)
    fg = mask > 0
    if not fg.any():
        return fmask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=4)
    if num_labels <= 1:
        return fmask

    conf_sums = np.bincount(labels.ravel(), weights=mask.ravel().astype(np.float32), minlength=num_labels)
    components = []
    for label in range(1, num_labels):
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        components.append((conf_sums[label], area, top, top + height - 1, left, left + width - 1))

    components.sort(key=lambda c: (c[0], c[1]), reverse=True)
    keep = np.zeros_like(mask, dtype=bool)
    for _, _, min_y, max_y, min_x, max_x in components[:max_subjects]:
        keep[min_y:max_y + 1, min_x:max_x + 1] = True
    return Image.fromarray(np.where(keep, mask, 0).astype(np.uint8), mode="L")


def load_image(path: Path) -> Image.Image:
    """Decode an image fully and close its file."""
    with Image.open(path) as im:
        return im.copy()


def mask_batch(image_paths, fmask_paths, model, sema, max_subjects):
    images = [load_image(p) for p in image_paths]
    for fmask, fmask_path in zip(extract_masks(images, model, sema), fmask_paths):
        if max_subjects is not None:
            fmask = keep_top_subject_bboxes(fmask, max_subjects)
        fmask_path.parent.mkdir(parents=True, exist_ok=True)
        fmask.save(fmask_path)


def remove_background(images_dir: Path, out_fmasks_dir: Path, model_name: str = "ZhengPeng7/BiRefNet",
                      image_ext: str = ".png", mask_ext: str = ".png", max_subjects: int | None = None,
                      batch_size: int = 8, num_workers: int = 4) -> int:
    """Write a mask per image; returns the number of images masked."""
    import torch
    from tqdm import tqdm

    images_dir, out_fmasks_dir = Path(images_dir), Path(out_fmasks_dir)
    image_paths = sorted(images_dir.glob(f"**/*{image_ext}"))
    fmask_paths = [out_fmasks_dir / p.relative_to(images_dir).with_suffix(mask_ext) for p in image_paths]
    if not image_paths:
        print(f"WARNING: no *{image_ext} images under {images_dir}")
        return 0

    # One model per visible GPU, or one CPU model. A semaphore per model
    # serialises its inference (prevents CUDA OOM) while the worker threads
    # overlap image decode and mask encode with it.
    n_gpu = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n_gpu)] if n_gpu else ["cpu"]
    models = []
    for device in devices:
        print(f"Loading '{model_name}' to {device}")
        models.append(load_model(model_name, device))
    semas = [threading.Semaphore(1) for _ in devices]

    starts = range(0, len(image_paths), batch_size)
    jobs = [(image_paths[s:s + batch_size], fmask_paths[s:s + batch_size],
             models[k % len(models)], semas[k % len(semas)], max_subjects)
            for k, s in enumerate(starts)]
    with ThreadPoolExecutor(max_workers=num_workers * len(devices)) as pool:
        futures = [pool.submit(mask_batch, *job) for job in jobs]
        for future in tqdm(futures, desc="Predicting foreground masks"):
            future.result()  # re-raises a worker's exception here
    return len(image_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("out_fmasks_dir", type=Path)
    parser.add_argument("--model_name", default="ZhengPeng7/BiRefNet",
                        help="Hugging Face model id (ZhengPeng7/BiRefNet, ZhengPeng7/BiRefNet-portrait, ...)")
    parser.add_argument("--image_ext", default=".png")
    parser.add_argument("--mask_ext", default=".png")
    parser.add_argument("--max_subjects", type=int, default=None,
                        help="Keep only the N most confident mask components (by bounding box)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Images per inference batch (8 fits ~16GB of GPU memory in fp16)")
    parser.add_argument("--num_workers", type=int, default=4, help="Worker threads per device")
    args = parser.parse_args()

    remove_background(args.images_dir, args.out_fmasks_dir, model_name=args.model_name,
                      image_ext=args.image_ext, mask_ext=args.mask_ext, max_subjects=args.max_subjects,
                      batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
