#!/usr/bin/env python3
"""
score_novel_views.py

Score a candidate novel view against the real photo of a held-out camera --
the bakeoff gate for choosing a repair model (see docs/render_and_repair.md).

Why this exists: model choice between video backbones is easy to make on how
the samples look and hard to make on whether they help the 4D fit. Those come
apart. A repair that is sharper, better lit and more attractive than the render
still hurts training if its colour, exposure and micro-detail sit off the real
cameras' manifold -- that is exactly how the first render-and-repair attempt
lost 4.7 dB (18.0 -> 13.3) on a genuinely held-out camera. So the bakeoff scores
against a real photo nothing in the pipeline ever saw, and the number decides.

Run render_pair_sweep.py --holdout_label first: it snaps one swept frame to the
held-out camera's exact centre and warps that camera's real photo into the sweep
frustum as probe_real.png. Point this script at the sweep directory and at each
candidate's repaired output.

WHAT THE NUMBERS MEAN

  * Scores are masked to where both images have coverage. probe_real.png is
    transparent wherever the real photo's field of view did not reach into the
    square frustum, and scoring those pixels would compare a render against
    nearest-edge padding.
  * `render` is always scored alongside the candidates and is the number that
    matters most: a repair that scores BELOW the raw render is actively
    destroying agreement with the real cameras, however good it looks. Several
    plausible repairs do this.
  * PSNR is reported over the masked region and, when --subject_mask is given,
    over the subject alone. Background agreement is easy and dominates the
    full-frame number; the subject is what the 4D fit is for.
  * LPIPS is reported when the `lpips` package imports, and skipped with a note
    otherwise. It disagrees with PSNR exactly where a model has traded fidelity
    for plausibility, which is the trade this bakeoff exists to catch, so a
    candidate that wins LPIPS while losing PSNR should be treated as unproven
    rather than as a winner.

conda env: none for PSNR/SSIM (numpy + PIL). LPIPS needs torch + lpips, so the
cumuli env gets that column and a bare env still gets a usable ranking.

Usage:
    python3 score_novel_views.py \\
        --sweep_dir /path/to/sweeps/03_to_05 \\
        --candidate wan22 /path/to/repaired/wan22/sweep_0040.png \\
        --candidate ltx25 /path/to/repaired/ltx25/sweep_0040.png \\
        --candidate diffuman4d /path/to/repaired/diffuman4d/sweep_0040.png \\
        [--subject_mask /path/to/fmasks_clean/05.png] \\
        [--report_json /path/to/bakeoff.json]

    --candidate is NAME PATH and repeatable. With no --candidate the script
    scores the raw render alone, which is the baseline every candidate must beat.
"""

import argparse
import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

COVERAGE_THRESHOLD = 40   # 8-bit alpha above this counts as real coverage, matching klein_repair_views.py
SSIM_C1 = (0.01 * 255) ** 2
SSIM_C2 = (0.03 * 255) ** 2
SSIM_WINDOW = 7           # uniform window; a Gaussian window moves SSIM by <0.002 here and needs scipy
MIN_SCORED_PIXELS = 1000  # below this the overlap is too small for the numbers to mean anything


def load_rgba(path: Path, size: int | None = None) -> tuple:
    """(H, W, 3) float RGB and a boolean coverage mask from an image, resized to
    `size` if it does not already match. Images without alpha are fully covered."""
    image = Image.open(path)
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)
    if image.mode == "RGBA":
        array = np.asarray(image, dtype=np.float64)
        return array[..., :3], array[..., 3] > COVERAGE_THRESHOLD
    array = np.asarray(image.convert("RGB"), dtype=np.float64)
    return array, np.ones(array.shape[:2], dtype=bool)


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """PSNR over the masked pixels only. Infinite for identical inputs, which
    means a candidate file is the reference file rather than a perfect repair."""
    error = ((a - b) ** 2)[mask]
    mse = float(error.mean())
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean SSIM on luminance over a uniform window, averaged across masked
    pixels. Windows straddling the mask edge are excluded rather than padded, so
    the score never credits agreement with invented pixels."""
    luma_a = a @ np.array([0.299, 0.587, 0.114])
    luma_b = b @ np.array([0.299, 0.587, 0.114])

    def box(image):
        cumulative = np.cumsum(np.cumsum(np.pad(image, ((1, 0), (1, 0))), axis=0), axis=1)
        w = SSIM_WINDOW
        return (cumulative[w:, w:] - cumulative[:-w, w:] - cumulative[w:, :-w] + cumulative[:-w, :-w]) / (w * w)

    mean_a, mean_b = box(luma_a), box(luma_b)
    var_a = box(luma_a ** 2) - mean_a ** 2
    var_b = box(luma_b ** 2) - mean_b ** 2
    covariance = box(luma_a * luma_b) - mean_a * mean_b

    ssim = (((2 * mean_a * mean_b + SSIM_C1) * (2 * covariance + SSIM_C2))
            / ((mean_a ** 2 + mean_b ** 2 + SSIM_C1) * (var_a + var_b + SSIM_C2)))
    whole_window = box(mask.astype(np.float64)) > 0.999
    return float(ssim[whole_window].mean()) if whole_window.any() else float("nan")


@lru_cache(maxsize=1)
def lpips_network():
    """The LPIPS network, built once per process, or None when the package is
    absent. Cached because constructing it loads AlexNet weights from disk --
    about 1.7 s, which a per-candidate rebuild would pay again for every row of
    the table.

    The warning filter is for lpips itself: it calls torchvision with the
    long-deprecated `pretrained=` argument, and this repo runs pytest with
    filterwarnings=error, so third-party deprecation noise would otherwise fail
    the suite for something no change here can fix."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            import lpips
            import torch
            network = lpips.LPIPS(net="alex", verbose=False)
    except ImportError:
        return None

    # CPU when the GPU is unavailable OR full. Scoring routinely runs while a
    # trainer holds most of the card, and a metric that dies with CUDA OOM is
    # worse than a metric that takes a few seconds longer.
    # torch.AcceleratorError subclasses RuntimeError, so one clause covers both.
    if torch.cuda.is_available():
        try:
            return network.cuda(), "cuda"
        except RuntimeError:
            # A .cuda() that fails partway leaves the module with some buffers
            # already moved, which then raises "found at least two devices".
            # Force the whole thing back before handing it out.
            network = network.cpu()
            print("  note: GPU busy or full; computing LPIPS on the CPU", file=sys.stderr)
    return network, "cpu"


def lpips_distance(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    """LPIPS with uncovered pixels zeroed in both images, or None when the
    package is absent. Zeroing rather than cropping keeps the comparison at the
    network's expected scale; both images get the identical treatment, so the
    blacked-out region contributes equally to each candidate."""
    loaded = lpips_network()
    if loaded is None:
        return None
    network, device = loaded
    import torch

    def to_tensor(image):
        masked = np.where(mask[..., None], image, 0.0)
        tensor = torch.from_numpy(masked.transpose(2, 0, 1)[None]).float() / 127.5 - 1.0
        return tensor.to(device)

    try:
        with torch.no_grad():
            return float(network(to_tensor(a), to_tensor(b)).item())
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print("  note: GPU ran out of memory mid-LPIPS; skipping that column", file=sys.stderr)
        return None


def score_pair(candidate_path: Path, reference: np.ndarray, coverage: np.ndarray,
               subject: np.ndarray | None) -> dict:
    """Every metric for one candidate against the held-out reference."""
    size = reference.shape[0]
    image, candidate_coverage = load_rgba(candidate_path, size)
    mask = coverage & candidate_coverage
    scored = int(mask.sum())
    if scored < MIN_SCORED_PIXELS:
        raise ValueError(f"{candidate_path}: only {scored} pixels overlap the held-out camera's "
                         "coverage -- the candidate is probably at a different pose or resolution")

    result = {"path": str(candidate_path), "scored_pixels": scored,
              "psnr": masked_psnr(image, reference, mask),
              "ssim": masked_ssim(image, reference, mask)}
    distance = lpips_distance(image, reference, mask)
    if distance is not None:
        result["lpips"] = distance
    if subject is not None:
        subject_mask = mask & subject
        result["psnr_subject"] = (masked_psnr(image, reference, subject_mask)
                                  if subject_mask.sum() >= MIN_SCORED_PIXELS else None)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True, type=Path,
                    help="render_pair_sweep.py output directory, holding probe_real.png and cameras.json")
    ap.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "PATH"), default=[],
                    help="a repaired image to score at the probe pose; repeatable")
    ap.add_argument("--subject_mask", type=Path, default=None,
                    help="cleaned mask for the held-out camera (fmasks_clean/<label>.png), adding a "
                         "subject-only PSNR column. NOTE this must be warped into the sweep frustum "
                         "the same way probe_real.png was; pass the warped copy, not the raw mask")
    ap.add_argument("--report_json", type=Path, default=None, help="write the full table as JSON")
    args = ap.parse_args()

    cameras_path = args.sweep_dir / "cameras.json"
    if not cameras_path.exists():
        print(f"ERROR: {cameras_path} not found -- run render_pair_sweep.py first", file=sys.stderr)
        return 1
    meta = json.loads(cameras_path.read_text())
    probe = meta.get("probe")
    if not probe or not probe.get("image"):
        print(f"ERROR: {cameras_path} records no held-out probe image. Re-run render_pair_sweep.py "
              "with --holdout_label naming a real camera between the swept pair", file=sys.stderr)
        return 1

    reference, coverage = load_rgba(args.sweep_dir / probe["image"])
    subject = None
    if args.subject_mask is not None:
        subject_image, _ = load_rgba(args.subject_mask, reference.shape[0])
        subject = subject_image[..., 0] > 127

    entries = [("render", args.sweep_dir / probe["render"])] + [(n, Path(p)) for n, p in args.candidate]
    results = {}
    for name, path in entries:
        if not path.exists():
            print(f"ERROR: {name}: {path} does not exist", file=sys.stderr)
            return 1
        results[name] = score_pair(path, reference, coverage, subject)

    has_lpips = any("lpips" in r for r in results.values())
    has_subject = subject is not None
    header = f"{'candidate':<16}{'PSNR':>9}{'SSIM':>9}"
    if has_subject:
        header += f"{'PSNR-subj':>11}"
    if has_lpips:
        header += f"{'LPIPS':>9}"
    print(f"\nheld-out camera {probe['camera']!r}, sweep frame {probe['idx']}, "
          f"{results['render']['scored_pixels']:,} px scored")
    print(header)
    print("-" * len(header))
    baseline = results["render"]["psnr"]
    for name, result in results.items():
        line = f"{name:<16}{result['psnr']:>9.2f}{result['ssim']:>9.4f}"
        if has_subject:
            subject_psnr = result.get("psnr_subject")
            line += f"{subject_psnr:>11.2f}" if subject_psnr is not None else f"{'--':>11}"
        if has_lpips:
            line += f"{result.get('lpips', float('nan')):>9.4f}"
        if name != "render":
            line += "   " + ("BEATS render" if result["psnr"] > baseline else "below render")
        print(line)
    if not has_lpips:
        print("\n(LPIPS skipped: the `lpips` package did not import in this env)")
    print("\nA candidate scoring below the raw render is moving pixels away from the real cameras; "
          "training on it is the 18.0 -> 13.3 dB failure in docs/render_and_repair.md.")

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(
            {"sweep_dir": str(args.sweep_dir), "probe": probe, "results": results}, indent=1))
        print(f"wrote {args.report_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
