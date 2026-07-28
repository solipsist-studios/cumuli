"""
tests/integration/baseline_metrics.py

The 3 golden-baseline measurements (pose reprojection error, per-camera
mask coverage, masked PSNR on the held-out eval camera) used by both
test_pipeline_end_to_end.py (regression checks against golden_baseline.json,
with margin) and scripts/update_integration_baseline.py (the post-merge
job that re-measures and writes a fresh golden_baseline.json). Pulled out
of the test file so the two call sites can't silently drift apart -- same
"import, not a copy" practice already used for
tests/unit/test_integration_baseline_checks.py (see
INTEGRATION_TESTS_COMMIT_PLAN.md, Step 13).

These functions do no assertion/comparison themselves -- they just read a
completed pipeline run's on-disk output (given its build_layout() dict `L`)
and return a number. Comparing that number against a baseline, with margin,
is the caller's job.
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image


def masked_psnr(render: np.ndarray, gt_rgb: np.ndarray, mask: np.ndarray) -> float:
    """PSNR in dB restricted to pixels where mask is True. Unmasked
    (full-frame) PSNR is misleading here -- ~98.75% of this fixture's
    frame is plain black background, trivially easy to match, so it
    dominates an unmasked score instead of subject fidelity."""
    diff2 = (render.astype(np.float64) - gt_rgb.astype(np.float64)) ** 2
    mse = diff2[mask].mean()
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / mse)


def parse_reprojection_error(log_text: str) -> float | None:
    """Parses run_pose_refinement.py's own printed 'Median residual: X ->
    Y' line (there's no machine-readable report file for this). Returns
    the final (post-refinement) median in px, or None if the line wasn't
    found in the log."""
    matches = re.findall(r"Median residual:\s*[\d.]+px\s*->\s*([\d.]+)px", log_text)
    return float(matches[-1]) if matches else None


def flat_label_for(L, real_camera: str) -> str:
    """build_flat_dataset.py writes camera_label_map.json mapping flat
    2-digit labels ("00") to HLOC-style labels ("Camera_0002") -- read it
    rather than assuming a fixed sort order, since that's what the
    pipeline itself actually guarantees."""
    import json
    label_map = json.loads(L["flat_label_map"].read_text())
    target = f"Camera_{real_camera}"
    for flat, hloc_label in label_map.items():
        if hloc_label == target:
            return flat
    raise AssertionError(f"{target} not found in {L['flat_label_map']}: {label_map}")


def mask_coverage(L, real_camera: str) -> float:
    """Cleaned-mask coverage fraction (fraction of pixels > 127) for one
    real camera."""
    flat = flat_label_for(L, real_camera)
    mask_path = L["flat_fmasks_clean"] / f"{flat}.png"
    if not mask_path.is_file():
        raise FileNotFoundError(f"missing cleaned mask for {real_camera} (flat label {flat}): {mask_path}")
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert("L"))
    return float((mask > 127).mean())


def held_out_psnr(L, total_train_iters: int, held_out_real_camera: str) -> tuple[float, Path]:
    """Masked PSNR between Brush's held-out eval render (excluded from
    training via --eval_split_every) and the real photo from that same
    camera. Returns (score, render_path) -- the caller may want
    render_path to copy the render out as a CI artifact."""
    held_out_flat = flat_label_for(L, held_out_real_camera)
    eval_dir = L["brush_output"] / f"eval_{total_train_iters}"
    render_path = eval_dir / f"{held_out_flat}.png"
    if not render_path.is_file():
        raise FileNotFoundError(
            f"expected brush_app's --eval-save-to-disk render at {render_path} -- "
            f"found instead: {sorted(eval_dir.glob('*')) if eval_dir.is_dir() else '(eval dir missing)'}"
        )

    with Image.open(render_path) as im:
        render = np.asarray(im.convert("RGB"))

    gt_path = L["train_set"] / "images_rgba" / f"{held_out_flat}.png"
    with Image.open(gt_path) as im:
        gt_im = im.resize(render.shape[1::-1], Image.LANCZOS)
        gt_arr = np.asarray(gt_im)

    gt_rgb = gt_arr[..., :3]
    mask = gt_arr[..., 3] > 127  # the baked alpha channel IS the cleaned mask

    score = masked_psnr(render, gt_rgb, mask)
    return score, render_path
