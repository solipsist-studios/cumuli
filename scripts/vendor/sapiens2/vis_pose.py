# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the Sapiens2 License found in the
# LICENSE.md file in this directory.
#
# Modified from sapiens/pose/tools/vis/vis_pose.py (sapiens2 @7e5bae8) by
# Solipsist Studios, by way of the solipsist-studios/Diffuman4D fork. See
# README.md in this directory for the list of changes.

import json
import os
import sapiens
import warnings
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from sapiens.pose.evaluators import nms
from sapiens.pose.models import init_model
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers.utils import logging as hf_logging

from pose_render_utils import visualize_keypoints

# DETR — COCO person = label 1.
_detector_cache: dict = {}
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _resolve_keypoints308_metainfo_file() -> str:
    """Resolve keypoints308 metainfo file path from installed sapiens package."""
    candidates = [
        os.path.normpath("configs/_base_/keypoints308.py"),
        os.path.join(os.path.dirname(sapiens.__file__), "pose", "configs", "_base_", "keypoints308.py"),
        os.path.join(os.path.dirname(sapiens.__file__), "configs", "_base_", "keypoints308.py"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    raise FileNotFoundError(
        "Could not find keypoints308 metainfo file. Tried: " + ", ".join(candidates)
    )


def _get_detector(device, ckpt_dir):
    if "model" not in _detector_cache:
        # Suppress known upstream loading warnings from DETR checkpoints/preprocessor configs.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r"The `max_size` parameter is deprecated.*",
            )
            _detector_cache["proc"] = DetrImageProcessor.from_pretrained(ckpt_dir)

        # Normalize to the newer size schema to avoid repeated deprecation warnings.
        proc_size = getattr(_detector_cache["proc"], "size", None)
        if isinstance(proc_size, dict) and "max_size" in proc_size and "longest_edge" not in proc_size:
            patched = dict(proc_size)
            patched["longest_edge"] = patched.pop("max_size")
            _detector_cache["proc"].size = patched

        prev_verbosity = hf_logging.get_verbosity()
        try:
            hf_logging.set_verbosity_error()
            _detector_cache["model"] = (
                DetrForObjectDetection.from_pretrained(ckpt_dir).eval().to(device)
            )
        finally:
            hf_logging.set_verbosity(prev_verbosity)
    return _detector_cache["proc"], _detector_cache["model"]


def _detect_persons(image_bgr: np.ndarray, args) -> np.ndarray:
    proc, model = _get_detector(args.device, args.det_checkpoint)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    inputs = proc(images=pil_img, return_tensors="pt").to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image_rgb.shape[:2]], device=args.device)
    results = proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=args.bbox_thr
    )[0]
    person_mask = results["labels"] == 1
    boxes = results["boxes"][person_mask].cpu().numpy()
    scores = results["scores"][person_mask].cpu().numpy().reshape(-1, 1)
    bboxes = np.concatenate([boxes, scores], axis=1)
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]  # B x 4; x1, y1, x2, y2
    return bboxes  # may be empty; callers handle empty via mask fallback or full-image fallback


def _load_mask_file(mask_path: str, image_shape: tuple[int, int, int]) -> np.ndarray | None:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    h, w = image_shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def _load_mask_for_image(mask_dir: str | None, image_name: str, image_shape: tuple[int, int, int]) -> np.ndarray | None:
    if not mask_dir:
        return None
    stem = Path(image_name).stem
    for ext in _IMAGE_SUFFIXES:
        candidate = os.path.join(mask_dir, f"{stem}{ext}")
        if os.path.exists(candidate):
            return _load_mask_file(candidate, image_shape)
    return None


def _mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _score_bbox_vs_mask(bbox: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Return (overlap_ratio, mask_coverage) for a bbox against a binary mask.

    overlap_ratio  = mask_pixels_in_bbox / bbox_area
                     (what fraction of the bbox is foreground)
    mask_coverage  = mask_pixels_in_bbox / total_mask_pixels
                     (what fraction of the foreground the bbox captures)
    """
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    x1 = int(np.clip(np.floor(x1), 0, w - 1))
    y1 = int(np.clip(np.floor(y1), 0, h - 1))
    x2 = int(np.clip(np.ceil(x2), 0, w - 1))
    y2 = int(np.clip(np.ceil(y2), 0, h - 1))
    if x2 < x1 or y2 < y1:
        return 0.0, 0.0
    patch = mask[y1 : y2 + 1, x1 : x2 + 1]
    if patch.size == 0:
        return 0.0, 0.0
    intersect = float(patch.sum())
    bbox_area = float(max(1, (x2 - x1 + 1) * (y2 - y1 + 1)))
    mask_total = float(mask.sum())
    overlap_ratio = intersect / bbox_area
    mask_coverage = intersect / max(1.0, mask_total)
    return overlap_ratio, mask_coverage


def _apply_bone_length_filter(
    keypoints: list,
    keypoint_scores: list,
    skeleton_links: list,
    max_bone_sigma: float,
    max_bilateral_ratio: float,
    flip_pairs: list[tuple[int, int]] | None = None,
    keypoint_info: dict | None = None,
) -> list:
    """Zero out the score of the lower-confidence endpoint of any link whose
    length is a statistical outlier relative to the *other bones in the same
    instance*.

     Two complementary checks are available:

     1. **Bilateral symmetry** — when left/right pairing metadata is supplied, every link
         is compared against its mirror counterpart (e.g. left-forearm vs
         right-forearm).  If one side is more than ``max_bilateral_ratio`` times
         longer than the other, the longer side's lower-confidence endpoint is
         zeroed.  This is the default because it compares like-with-like rather
         than mixing torso, hand, and facial scales into one distribution.

     2. **MAD outlier** — optional. Computes median + MAD of all bone lengths in
         the instance and flags any bone more than ``max_bone_sigma`` sigma above
         the median.  For the 308-keypoint whole-body graph this can be too
         aggressive because the graph mixes very small distal links with large
         torso/limb links, so it is disabled by default.

    A zeroed score propagates to triangulate_skeleton's score_thr gate so the
    bad point is excluded from 3-D reconstruction.

    Returns a new list of score arrays (originals untouched).
    """
    if (not max_bone_sigma and not max_bilateral_ratio) or not skeleton_links:
        return keypoint_scores

    adjacency: dict[int, set[int]] = {}
    for i, j in skeleton_links:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)

    node_degree = {idx: len(neighbors) for idx, neighbors in adjacency.items()}

    # Protect the core body graph from suppression.  The linked whole-body
    # topology here uses low ids for the torso / face scaffold, while distal
    # foot and hand chains fan out from those protected roots.
    protected_nodes = {
        idx for idx, degree in node_degree.items()
        if idx <= 14 or degree >= 3
    }

    node_depth: dict[int, int] = {}
    frontier = [(idx, 0) for idx in protected_nodes]
    while frontier:
        idx, depth = frontier.pop(0)
        for neighbor in adjacency.get(idx, ()): 
            if neighbor in protected_nodes or neighbor in node_depth:
                continue
            node_depth[neighbor] = depth + 1
            frontier.append((neighbor, depth + 1))

    def _link_is_distal(i: int, j: int) -> bool:
        return (i in node_depth) or (j in node_depth)

    def _select_suspicious_endpoint(i: int, j: int, scores: np.ndarray) -> int | None:
        """Pick the endpoint most likely to be the hallucinated point.

        Only suppress non-core distal-chain points. Prefer the endpoint that is
        farther from the protected body/hand root over deleting any shared or
        proximal articulation. Only fall back to confidence for ties, and skip
        fully ambiguous cases.
        """
        depth_i = node_depth.get(i)
        depth_j = node_depth.get(j)
        score_i = float(scores[i])
        score_j = float(scores[j])

        if depth_i is None and depth_j is None:
            return None
        if depth_i is None:
            return j
        if depth_j is None:
            return i
        if depth_i != depth_j:
            return i if depth_i > depth_j else j

        deg_i = len(adjacency.get(i, ()))
        deg_j = len(adjacency.get(j, ()))
        if deg_i != deg_j:
            return i if deg_i < deg_j else j

        if abs(score_i - score_j) >= 0.15:
            return i if score_i < score_j else j

        # Degrees and confidences are too similar; do not guess.
        return None

    def _link_confidence(i: int, j: int, scores: np.ndarray) -> tuple[float, float]:
        """Return conservative and average confidence for one link."""
        score_i = float(scores[i])
        score_j = float(scores[j])
        return min(score_i, score_j), 0.5 * (score_i + score_j)

    def _should_suppress_longer_link(
        longer_link: tuple[int, int],
        shorter_link: tuple[int, int],
        scores: np.ndarray,
    ) -> bool:
        """Only let a mirrored link veto the other side when it is reliable.

        This avoids deleting a good visible hand because the opposite hand is
        partially occluded, short, or otherwise low-confidence.
        """
        longer_min, longer_avg = _link_confidence(*longer_link, scores)
        shorter_min, shorter_avg = _link_confidence(*shorter_link, scores)

        # A weak mirrored link is not a trustworthy reference.
        if shorter_min < 0.35:
            return False

        # If the candidate longer link is materially better supported than the
        # shorter mirrored link, prefer to keep it; the shorter side is likely
        # the occluded / degraded observation.
        if longer_min > shorter_min + 0.10:
            return False
        if longer_avg > shorter_avg + 0.10:
            return False

        return True

    # ── pre-compute bilateral link pairs from left/right metainfo ──────────
    bilateral_pairs: list[tuple[tuple, tuple]] = []
    swap_id: dict[int, int] = {}
    if flip_pairs:
        for left_idx, right_idx in flip_pairs:
            try:
                left_idx = int(left_idx)
                right_idx = int(right_idx)
            except (TypeError, ValueError):
                continue
            swap_id[left_idx] = right_idx
    elif isinstance(keypoint_info, dict):
        # Fallback for metainfo variants that expose named swap partners.
        normalized_info: dict[int, dict] = {}
        for raw_idx, info in keypoint_info.items():
            if not isinstance(info, dict):
                continue
            if "name" not in info:
                continue
            idx = info.get("id", raw_idx)
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            normalized_info[idx] = info

        name_to_id = {
            info["name"]: idx
            for idx, info in normalized_info.items()
            if isinstance(info.get("name"), str)
        }
        for idx, info in normalized_info.items():
            swap_name = info.get("swap", "")
            if swap_name and swap_name in name_to_id:
                swap_id[idx] = name_to_id[swap_name]

    if swap_id:
        link_set = {(i, j) for i, j in skeleton_links} | {(j, i) for i, j in skeleton_links}
        seen: set = set()
        for i, j in skeleton_links:
            si, sj = swap_id.get(i), swap_id.get(j)
            if si is None or sj is None:
                continue
            if (si, sj) not in link_set and (sj, si) not in link_set:
                continue
            canonical = tuple(sorted([(i, j), (si, sj)]))
            if canonical in seen:
                continue
            seen.add(canonical)
            bilateral_pairs.append(((i, j), (si, sj)))

    filtered = [s.copy() for s in keypoint_scores]

    for kpts, scores in zip(keypoints, filtered):
        n = len(kpts)

        # ── step 1: optional MAD outlier across all bones ──────────────────
        lengths: list[float] = []
        per_link_lengths: list[float | None] = []
        for i, j in skeleton_links:
            if i >= n or j >= n:
                per_link_lengths.append(None)
                continue
            d = float(np.linalg.norm(kpts[i] - kpts[j]))
            per_link_lengths.append(d)
            if not _link_is_distal(i, j):
                continue
            # Only include high-confidence bones in the reference distribution
            # so that a cluster of jank bones does not inflate the median.
            if scores[i] > 0.3 and scores[j] > 0.3:
                lengths.append(d)

        if max_bone_sigma and len(lengths) >= 5:
            arr = np.asarray(lengths, dtype=np.float64)
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median)))
            # Floor MAD so a perfectly uniform skeleton still has a finite gate.
            mad = max(mad, median * 0.1)
            threshold = median + max_bone_sigma * mad

            for (i, j), d in zip(skeleton_links, per_link_lengths):
                if d is None or d <= threshold:
                    continue
                if i >= n or j >= n:
                    continue
                if not _link_is_distal(i, j):
                    continue
                bad_idx = _select_suspicious_endpoint(i, j, scores)
                if bad_idx is not None:
                    scores[bad_idx] = 0.0

        # ── step 2: bilateral symmetry ─────────────────────────────────────
        for (i, j), (si, sj) in bilateral_pairs:
            if i >= n or j >= n or si >= n or sj >= n:
                continue
            if not max_bilateral_ratio:
                continue
            if not _link_is_distal(i, j):
                continue
            d1 = float(np.linalg.norm(kpts[i] - kpts[j]))
            d2 = float(np.linalg.norm(kpts[si] - kpts[sj]))
            if d1 < 1e-6 or d2 < 1e-6:
                continue
            ratio = d1 / d2
            if ratio > max_bilateral_ratio:
                if not _should_suppress_longer_link((i, j), (si, sj), scores):
                    continue
                bad_idx = _select_suspicious_endpoint(i, j, scores)
                if bad_idx is not None:
                    scores[bad_idx] = 0.0
            elif 1.0 / ratio > max_bilateral_ratio:
                if not _should_suppress_longer_link((si, sj), (i, j), scores):
                    continue
                bad_idx = _select_suspicious_endpoint(si, sj, scores)
                if bad_idx is not None:
                    scores[bad_idx] = 0.0

    return filtered


def _filter_bboxes_by_mask(bboxes: np.ndarray, mask: np.ndarray | None, min_overlap: float) -> np.ndarray:
    """Filter detections by mask.  When a mask is available the mask bbox is
    always injected as a candidate so a missed DETR detection can still be
    recovered.  Scoring uses mask_coverage (fraction of mask captured by the
    box) which is more discriminative than the previous bbox-density metric."""
    if mask is None:
        if len(bboxes) == 0:
            return bboxes
        return bboxes

    # Always add the tight mask bbox as a synthetic candidate.
    mask_box = _mask_bbox(mask)
    if mask_box is not None:
        bboxes = np.concatenate([bboxes, mask_box[None]], axis=0) if len(bboxes) > 0 else mask_box[None]

    if len(bboxes) == 0:
        return bboxes

    kept = []
    for bbox in bboxes:
        _, mask_coverage = _score_bbox_vs_mask(bbox, mask)
        if mask_coverage >= min_overlap:
            kept.append(bbox)

    if kept:
        return np.asarray(kept, dtype=np.float32)

    # Nothing met the threshold; return the mask box so inference still runs.
    if mask_box is not None:
        return mask_box[None]
    return bboxes


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _limit_subjects(
    bboxes: np.ndarray,
    mask: np.ndarray | None,
    max_subjects: int | None,
) -> np.ndarray:
    if max_subjects is None or max_subjects <= 0 or len(bboxes) <= max_subjects:
        return bboxes

    scored = []
    for i, bbox in enumerate(bboxes):
        if mask is not None:
            _, mask_coverage = _score_bbox_vs_mask(bbox, mask)
        else:
            mask_coverage = 0.0
        area = _bbox_area(bbox)
        scored.append((i, mask_coverage, area))

    # Prefer high mask coverage; break ties with larger boxes.
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    keep_ids = sorted([idx for idx, _, _ in scored[:max_subjects]])
    return bboxes[keep_ids]


def _instance_pose_quality(scores: np.ndarray, kpt_thr: float) -> tuple[float, float, float]:
    """Rank one instance by how likely it is to produce a usable skeleton."""
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return (0.0, 0.0, 0.0)

    # Use a softer gate than visualization to avoid over-penalizing difficult frames.
    weak_thr = max(0.15, float(kpt_thr) * 0.5)
    num_confident = float(np.sum(flat >= weak_thr))

    topk = min(20, flat.size)
    if topk > 0:
        topk_mean = float(np.mean(np.partition(flat, -topk)[-topk:]))
    else:
        topk_mean = 0.0
    mean_score = float(np.mean(flat))
    return (num_confident, topk_mean, mean_score)


def _select_instances_by_pose_quality(
    keypoints: list[np.ndarray],
    keypoint_scores: list[np.ndarray],
    bboxes: np.ndarray,
    max_subjects: int | None,
    kpt_thr: float,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    if max_subjects is None or max_subjects <= 0 or len(keypoints) <= max_subjects:
        return keypoints, keypoint_scores, bboxes

    scored = []
    for idx, scores in enumerate(keypoint_scores):
        quality = _instance_pose_quality(scores, kpt_thr)
        scored.append((idx, quality))

    # Keep deterministic order in case of ties.
    scored.sort(key=lambda x: (x[1][0], x[1][1], x[1][2], -x[0]), reverse=True)
    keep = sorted([idx for idx, _ in scored[:max_subjects]])

    kept_keypoints = [keypoints[idx] for idx in keep]
    kept_scores = [keypoint_scores[idx] for idx in keep]
    kept_bboxes = bboxes[keep]
    return kept_keypoints, kept_scores, kept_bboxes


def process_one_image(args, image, model, fg_mask: np.ndarray | None = None):
    bboxes = _detect_persons(image, args)
    bboxes = _filter_bboxes_by_mask(bboxes, fg_mask, args.mask_bbox_overlap_thr)
    # Keep a small candidate pool for inference, then choose final subjects by
    # pose quality. This avoids max_subjects=1 picking a weak detection.
    prelimit = None
    if args.max_subjects is not None and args.max_subjects > 0:
        prelimit = max(args.max_subjects * 4, args.max_subjects)
    bboxes = _limit_subjects(bboxes, fg_mask, prelimit)
    if bboxes is None or len(bboxes) == 0:
        h, w = image.shape[:2]
        bboxes = np.asarray([[0.0, 0.0, float(w), float(h)]], dtype=np.float32)
    inputs_list = []
    data_samples_list = []
    for bbox in bboxes:
        data_info = dict(img=image)
        data_info["bbox"] = bbox[None]  # shape (1, 4)
        data_info["bbox_score"] = np.ones(1, dtype=np.float32)  # shape (1,)
        data = model.pipeline(data_info)
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        data_samples_list.append(data["data_samples"])

    inputs = torch.cat(inputs_list, dim=0)  # B x 3 x H x W
    with torch.no_grad():
        pred = model(inputs)  # B x 3 x H x W
        if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
            pred_flipped = model(inputs.flip(-1))  # B x 3 x H x W
            pred_flipped = pred_flipped.flip(-1)  ## B x K x heatmap_H x heatmap_W
            flip_indices = model.pose_metainfo["flip_indices"]
            assert len(flip_indices) == pred_flipped.shape[1]  ## K
            pred_flipped = pred_flipped[:, flip_indices]
            pred = (pred + pred_flipped) / 2.0

    # ------------------------------------------
    pred = pred.cpu().numpy()  ## B x K x heatmap_H x heatmap_W
    keypoints = []
    keypoint_scores = []
    for i, data_samples in enumerate(data_samples_list):
        ## kps in crop image
        ## keypoints_i is 1 x K x 2
        # keypoint_scores_i is 1 x K
        keypoints_i, keypoint_scores_i = model.codec.decode(pred[i])
        input_size = data_samples["meta"]["input_size"]  ## 1 x 2, 768 x 1024
        bbox_center = data_samples["meta"]["bbox_center"]  ## 1 x 2
        bbox_scale = data_samples["meta"]["bbox_scale"]  ## 1 x 2

        keypoints_i = (
            keypoints_i / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        )
        keypoints.append(keypoints_i[0])  ## remove fake batch dim
        keypoint_scores.append(keypoint_scores_i[0])  ## remove fake batch dim

    # Zero scores for endpoints of anatomically implausible bones so they are
    # excluded from both the JSON output and triangulation downstream.
    meta = getattr(model, "pose_metainfo", {})
    skeleton_links = meta.get("skeleton_links", []) if isinstance(meta, dict) else []
    flip_pairs = meta.get("flip_pairs", []) if isinstance(meta, dict) else []
    keypoint_info = None
    if isinstance(meta, dict):
        if isinstance(meta.get("keypoint_info", None), dict):
            keypoint_info = meta["keypoint_info"]
        elif isinstance(meta.get("coco_wholebody_to_goliath_keypoint_info", None), dict):
            keypoint_info = meta["coco_wholebody_to_goliath_keypoint_info"]
    max_bone_sigma = getattr(args, "max_bone_sigma", 0.0)
    max_bilateral_ratio = getattr(args, "max_bilateral_ratio", 3.5)
    keypoint_scores = _apply_bone_length_filter(
        keypoints,
        keypoint_scores,
        skeleton_links,
        max_bone_sigma,
        max_bilateral_ratio,
        flip_pairs,
        keypoint_info,
    )

    keypoints, keypoint_scores, bboxes = _select_instances_by_pose_quality(
        keypoints,
        keypoint_scores,
        bboxes,
        args.max_subjects,
        args.kpt_thr,
    )

    return keypoints, keypoint_scores, bboxes


# -------------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("det_checkpoint", help="Local DETR snapshot directory")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", required=True, help="Input image dir")
    parser.add_argument("--output", required=True, help="Path to output dir")
    parser.add_argument("--fmasks-dir", default=None, help="Foreground masks directory")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--radius", type=int, default=3, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Link thickness for visualization"
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds"
    )
    parser.add_argument(
        "--bbox-thr", type=float, default=0.3, help="Bounding box score threshold"
    )
    parser.add_argument(
        "--nms-thr", type=float, default=0.3, help="IoU threshold for bounding box NMS"
    )
    parser.add_argument(
        "--mask-bbox-overlap-thr",
        type=float,
        default=0.15,
        help="Minimum fraction of the foreground mask that a bbox must cover to be kept",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of detected subjects to keep per frame after mask filtering",
    )
    parser.add_argument(
        "--max-bone-sigma",
        type=float,
        default=0.0,
        help="Optional JSON-level MAD filter: zero the lower-confidence endpoint of any bone "
             "whose length exceeds median + N*MAD across the instance's own bone distribution. "
             "Disabled by default because whole-body links span multiple natural scales.",
    )
    parser.add_argument(
        "--max-bilateral-ratio",
        type=float,
        default=3.5,
        help="JSON-level bilateral filter: if one mirrored link is longer than this ratio times "
             "its counterpart, zero the longer link's lower-confidence endpoint. Set 0 to disable.",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Disable saving per-video predictions JSON (saved by default).",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Also write a skeleton overlay image per input next to the JSON "
             "(off by default: a full-resolution PNG encode per image that "
             "nothing downstream reads; keep for debugging).",
    )
    parser.add_argument(
        "--predictions-name",
        default=None,
        help="Override predictions JSON filename (used by helper for per-chunk writes).",
    )

    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)
    os.makedirs(args.output, exist_ok=True)

    ## add pose metainfo to model
    num_keypoints = model.cfg.num_keypoints
    if num_keypoints == 308:
        metainfo_file = _resolve_keypoints308_metainfo_file()
        model.pose_metainfo = parse_pose_metainfo(
            dict(from_file=metainfo_file)
        )

    ## add codec to model
    codec_type = model.cfg.codec.pop("type")
    assert codec_type == "UDPHeatmap", "Only support UDPHeatmap"
    model.codec = UDPHeatmap(**model.cfg.codec)

    # warm up the bbox detector (loads from args.det_checkpoint)
    _get_detector(args.device, args.det_checkpoint)

    # Get image list
    manifest_masks = {}
    if os.path.isdir(args.input):
        input_dir = args.input
        image_names = [
            name
            for name in sorted(os.listdir(input_dir))
            if name.lower().endswith(_IMAGE_SUFFIXES)
        ]
    else:
        # Manifest mode: one image per line, optionally followed by a tab and
        # that image's foreground-mask path. Paths may live in different
        # directories (for example one per frame of a sequence), so records
        # carry the path as written rather than the basename, and the
        # caller routes the JSON by that path. This is how a whole frame
        # window runs through one model load.
        with open(args.input, "r") as f:
            manifest_rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
        image_names = [row[0] for row in manifest_rows]
        manifest_masks = {row[0]: row[1] for row in manifest_rows if len(row) > 1 and row[1]}
        input_dir = None

    frames_records = []
    image_size = None
    num_keypoints_seen = None

    for image_name in tqdm(image_names, total=len(image_names)):
        image_path = image_name if input_dir is None else os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[vis_pose] could not read {image_path}, skipping")
            continue
        if image_name in manifest_masks:
            fg_mask = _load_mask_file(manifest_masks[image_name], image.shape)
        else:
            fg_mask = _load_mask_for_image(args.fmasks_dir, os.path.basename(image_name), image.shape)

        try:
            keypoints, keypoint_scores, bboxes = process_one_image(
                args, image, model, fg_mask=fg_mask
            )
        except Exception as e:
            print(f"[vis_pose] inference failed on {image_name}: {e}")
            continue

        if image_size is None:
            image_size = [int(image.shape[1]), int(image.shape[0])]
        if num_keypoints_seen is None and len(keypoints) > 0:
            num_keypoints_seen = int(np.asarray(keypoints[0]).shape[0])

        if args.save_vis:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            vis_image_rgb = visualize_keypoints(
                image=image_rgb,
                keypoints=keypoints,
                keypoints_visible=np.ones_like(keypoint_scores) > 0,
                keypoint_scores=keypoint_scores,
                radius=args.radius,
                thickness=args.thickness,
                kpt_thr=args.kpt_thr,
                skeleton=model.pose_metainfo["skeleton_links"],
                kpt_color=model.pose_metainfo["keypoint_colors"],
                link_color=model.pose_metainfo["skeleton_link_colors"],
            )
            vis_image = cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR)
            # Manifest entries can be absolute paths; flatten them to a name.
            vis_name = image_name if input_dir is not None else image_name.strip(os.sep).replace(os.sep, "__")
            save_path = os.path.join(args.output, vis_name)
            cv2.imwrite(save_path, vis_image)

        if not args.no_save_json:
            try:
                instances = []
                for kpts, scores, bbox in zip(keypoints, keypoint_scores, bboxes):
                    instances.append({
                        "bbox": [float(v) for v in np.asarray(bbox).reshape(-1)[:4]],
                        "keypoints": np.asarray(kpts, dtype=float).tolist(),
                        "keypoint_scores": np.asarray(scores, dtype=float).reshape(-1).tolist(),
                    })
                frames_records.append({
                    "image_name": image_name,
                    "instances": instances,
                })
            except Exception as e:
                print(f"[vis_pose] json record failed on {image_name}: {e}")

    if not args.no_save_json:
        nn = os.path.basename(os.path.normpath(args.output))
        parent_nn = os.path.basename(os.path.dirname(os.path.normpath(args.output)))
        video_label = parent_nn if nn.endswith("_output") else nn
        json_filename = args.predictions_name or f"{video_label}_predictions.json"
        json_path = os.path.join(args.output, json_filename)
        payload = {
            "video": video_label,
            "image_size": image_size,
            "num_keypoints": num_keypoints_seen,
            "kpt_thr_used": float(args.kpt_thr),
            "frames": frames_records,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f)
        print(f"[vis_pose] wrote predictions: {json_path} ({len(frames_records)} frames)")


if __name__ == "__main__":
    main()
