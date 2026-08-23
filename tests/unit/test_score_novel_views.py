"""Unit tests for score_novel_views.py.

The bakeoff decides which video model gets built into the pipeline, so the
metrics have to be trustworthy in the one way that matters here: they must
compare only pixels where BOTH images have real coverage. A scorer that quietly
includes the transparent padding around a warped photo would rank candidates on
how well they match black."""

import json

import numpy as np
import pytest
from PIL import Image

import score_novel_views as snv


def write_rgba(path, rgb, alpha=255):
    """An RGBA image with a uniform or per-pixel alpha."""
    array = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    array[..., :3] = rgb
    array[..., 3] = alpha
    Image.fromarray(array, mode="RGBA").save(path)


def flat(value, size=64):
    return np.full((size, size, 3), value, dtype=np.uint8)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_psnr_of_identical_images_is_infinite():
    image = flat(120).astype(np.float64)
    mask = np.ones(image.shape[:2], dtype=bool)
    assert snv.masked_psnr(image, image, mask) == float("inf")


def test_psnr_matches_the_closed_form():
    reference = flat(100).astype(np.float64)
    candidate = flat(110).astype(np.float64)
    mask = np.ones(reference.shape[:2], dtype=bool)
    expected = 10.0 * np.log10(255.0 ** 2 / 100.0)  # squared error is 10^2 everywhere
    assert snv.masked_psnr(candidate, reference, mask) == pytest.approx(expected)


def test_psnr_ignores_pixels_outside_the_mask():
    # The pixels the mask excludes are wildly wrong; if they leaked into the
    # score, the result would not be the score of the masked region.
    reference = flat(100).astype(np.float64)
    candidate = flat(100).astype(np.float64)
    candidate[:32] = 0.0

    mask = np.zeros(reference.shape[:2], dtype=bool)
    mask[32:] = True
    assert snv.masked_psnr(candidate, reference, mask) == float("inf")


def test_ssim_of_identical_images_is_one():
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))[..., None].repeat(3, axis=2).astype(np.float64)
    mask = np.ones(image.shape[:2], dtype=bool)
    assert snv.masked_ssim(image, image, mask) == pytest.approx(1.0, abs=1e-6)


def test_ssim_falls_when_structure_differs():
    rng = np.random.default_rng(1)
    reference = rng.integers(0, 255, size=(64, 64, 3)).astype(np.float64)
    candidate = rng.integers(0, 255, size=(64, 64, 3)).astype(np.float64)
    mask = np.ones((64, 64), dtype=bool)
    assert snv.masked_ssim(candidate, reference, mask) < 0.2


def test_ssim_is_nan_when_no_window_fits_inside_the_mask():
    image = flat(100).astype(np.float64)
    mask = np.zeros((64, 64), dtype=bool)
    mask[0, 0] = True  # a single pixel; no 7x7 window lies wholly inside it
    assert np.isnan(snv.masked_ssim(image, image, mask))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def test_load_rgba_reads_coverage_from_alpha(tmp_path):
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[16:48] = 255
    path = tmp_path / "probe.png"
    write_rgba(path, flat(90), alpha)

    _, coverage = snv.load_rgba(path)
    assert coverage[32, 32]
    assert not coverage[0, 0]
    assert coverage.sum() == 32 * 64


def test_load_rgba_treats_an_image_without_alpha_as_fully_covered(tmp_path):
    path = tmp_path / "candidate.png"
    Image.fromarray(flat(90)).save(path)

    rgb, coverage = snv.load_rgba(path)
    assert coverage.all()
    assert rgb.shape == (64, 64, 3)


def test_load_rgba_resizes_to_the_reference(tmp_path):
    path = tmp_path / "candidate.png"
    Image.fromarray(flat(90, size=32)).save(path)
    rgb, coverage = snv.load_rgba(path, size=64)
    assert rgb.shape == (64, 64, 3)
    assert coverage.shape == (64, 64)


# --------------------------------------------------------------------------
# score_pair
# --------------------------------------------------------------------------

def test_score_pair_intersects_the_two_coverage_masks(tmp_path):
    # The candidate covers the top half, the reference the left half. Only the
    # top-left quadrant may be scored -- and the candidate is deliberately wrong
    # everywhere else, so a leak shows up immediately.
    reference = flat(100).astype(np.float64)
    coverage = np.zeros((64, 64), dtype=bool)
    coverage[:, :32] = True

    candidate = flat(100)
    candidate[32:] = 0
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[:32] = 255
    path = tmp_path / "candidate.png"
    write_rgba(path, candidate, alpha)

    result = snv.score_pair(path, reference, coverage, None)
    assert result["scored_pixels"] == 32 * 32
    assert result["psnr"] == float("inf")


def test_score_pair_rejects_a_near_empty_overlap(tmp_path):
    reference = flat(100).astype(np.float64)
    coverage = np.zeros((64, 64), dtype=bool)
    coverage[:2, :2] = True
    path = tmp_path / "candidate.png"
    write_rgba(path, flat(100))

    with pytest.raises(ValueError, match="different pose or resolution"):
        snv.score_pair(path, reference, coverage, None)


def test_score_pair_adds_a_subject_only_column(tmp_path):
    # Background matches perfectly, subject does not. The full-frame number is
    # flattered by the easy background; the subject column is the honest one.
    reference = flat(100).astype(np.float64)
    coverage = np.ones((64, 64), dtype=bool)
    subject = np.zeros((64, 64), dtype=bool)
    subject[:40] = True

    candidate = flat(100)
    candidate[:40] = 140
    path = tmp_path / "candidate.png"
    write_rgba(path, candidate)

    result = snv.score_pair(path, reference, coverage, subject)
    assert result["psnr_subject"] < result["psnr"]


def test_score_pair_reports_no_subject_psnr_when_the_subject_is_tiny(tmp_path):
    reference = flat(100).astype(np.float64)
    subject = np.zeros((64, 64), dtype=bool)
    subject[:1, :4] = True
    path = tmp_path / "candidate.png"
    write_rgba(path, flat(100))

    result = snv.score_pair(path, reference, np.ones((64, 64), dtype=bool), subject)
    assert result["psnr_subject"] is None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def make_sweep(tmp_path, probe_block=None):
    """A minimal render_pair_sweep.py output directory."""
    sweep = tmp_path / "03_to_07"
    sweep.mkdir()
    write_rgba(sweep / "probe_real.png", flat(100))
    write_rgba(sweep / "sweep_0020.png", flat(112))
    meta = {"pair": ["03", "07"], "res": 64,
            "probe": probe_block if probe_block is not None
            else {"camera": "05", "idx": 20, "image": "probe_real.png", "render": "sweep_0020.png"}}
    (sweep / "cameras.json").write_text(json.dumps(meta))
    return sweep


def test_cli_scores_the_render_and_each_candidate(tmp_path, monkeypatch, capsys):
    sweep = make_sweep(tmp_path)
    good = tmp_path / "good.png"
    write_rgba(good, flat(101))          # closer to the reference than the render
    report = tmp_path / "bakeoff.json"

    monkeypatch.setattr("sys.argv", ["score_novel_views.py", "--sweep_dir", str(sweep),
                                     "--candidate", "wan22", str(good),
                                     "--report_json", str(report)])
    assert snv.main() == 0

    written = json.loads(report.read_text())
    assert written["probe"]["camera"] == "05"
    assert written["results"]["wan22"]["psnr"] > written["results"]["render"]["psnr"]
    assert "BEATS render" in capsys.readouterr().out


def test_cli_flags_a_candidate_that_is_worse_than_the_render(tmp_path, monkeypatch, capsys):
    sweep = make_sweep(tmp_path)
    bad = tmp_path / "bad.png"
    write_rgba(bad, flat(180))           # much further from the reference than the render

    monkeypatch.setattr("sys.argv", ["score_novel_views.py", "--sweep_dir", str(sweep),
                                     "--candidate", "ltx25", str(bad)])
    assert snv.main() == 0
    assert "below render" in capsys.readouterr().out


def test_cli_refuses_a_sweep_with_no_probe(tmp_path, monkeypatch, capsys):
    sweep = make_sweep(tmp_path, probe_block={})
    monkeypatch.setattr("sys.argv", ["score_novel_views.py", "--sweep_dir", str(sweep)])
    assert snv.main() == 1
    assert "--holdout_label" in capsys.readouterr().err


def test_cli_refuses_a_missing_sweep_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["score_novel_views.py", "--sweep_dir", str(tmp_path / "absent")])
    assert snv.main() == 1
    assert "render_pair_sweep.py first" in capsys.readouterr().err


def test_cli_refuses_a_missing_candidate(tmp_path, monkeypatch, capsys):
    sweep = make_sweep(tmp_path)
    monkeypatch.setattr("sys.argv", ["score_novel_views.py", "--sweep_dir", str(sweep),
                                     "--candidate", "h3", str(tmp_path / "absent.png")])
    assert snv.main() == 1
    assert "does not exist" in capsys.readouterr().err
