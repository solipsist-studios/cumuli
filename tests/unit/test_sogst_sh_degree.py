# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Baking and packing at SH degree 0, 1, 2 and 3.

The container spec (docs/sogst-format.md section 4.7) has always allowed 1-3
bands, but every writer used to hard-code 3 and the reference decoder wrote
stride 15 whatever the band count.  A lower-degree file therefore needs the
whole chain to agree on the channel-major stride `coeffs`: bake, PLY, pack,
decode.  A wrong stride renders as plausible but wrong view-dependent colour,
so these tests check values, not just shapes.
"""

import numpy as np
import pytest
import torch

import bake_sogst
import make_sogst_fixture as fixture
from eval_render import decode_sogst_fields
from sogst_io import SOGST_SHN_WIDTHS, shn_bands_for_width
from sogst_pack import compute_sogst_order, pack_sogst
from sogst_ply import read_sogst_ply


def lower_bands(f_rest, coeffs):
    n = f_rest.shape[0]
    return np.ascontiguousarray(
        f_rest.reshape(n, 3, 15)[:, :, :coeffs].reshape(n, 3 * coeffs))


@pytest.mark.parametrize("width, bands", [(9, 1), (24, 2), (45, 3)])
def test_bands_follow_the_block_width(width, bands):
    assert shn_bands_for_width(width) == bands


@pytest.mark.parametrize("width", [0, 15, 30, 47])
def test_other_widths_are_refused(width):
    with pytest.raises(ValueError, match="9, 24 or 45"):
        shn_bands_for_width(width)


@pytest.mark.parametrize("bands, coeffs", [(1, 3), (2, 8), (3, 15)])
def test_pack_and_decode_at_each_band_count(tmp_path, bands, coeffs):
    """The packer declares the band count the block carries, sizes the
    centroid texture to match, and the decoder reads it back at stride
    `coeffs`.  VQ is lossy, so the check is error against the data's own
    spread: a stride bug compares unrelated coefficients and lands near 1."""
    fields, meta = fixture.build_fixture(count=1200, degree=1, include_sh=True, seed=5)
    fields["f_rest"] = lower_bands(fields["f_rest"], coeffs)
    path = tmp_path / "s.sogst"
    order_segments = compute_sogst_order(fields, meta["time_min"], meta["time_max"], 0.1)
    written = pack_sogst(str(path), fields, meta["time_min"], meta["time_max"],
                         meta["fps"], shn_count=4096, order_segments=order_segments)
    assert written["shN"]["bands"] == bands

    _header, decoded = decode_sogst_fields(str(path))
    assert decoded["f_rest"].shape == (1200, 3 * coeffs)

    # the archive stores splats in play order, the same order verify_sogst uses
    ref = fields["f_rest"][order_segments[0]]
    ratio = np.abs(decoded["f_rest"] - ref).mean() / ref.std()
    assert ratio < 0.1, ratio


def test_centroid_texture_width_matches_bands(tmp_path):
    import zipfile
    from sogst_pack import decode_webp
    fields, meta = fixture.build_fixture(count=600, degree=1, include_sh=True, seed=2)
    fields["f_rest"] = lower_bands(fields["f_rest"], 8)
    path = tmp_path / "s.sogst"
    pack_sogst(str(path), fields, meta["time_min"], meta["time_max"], meta["fps"],
               shn_count=256)
    with zipfile.ZipFile(path) as zf:
        cent = decode_webp(zf.read("shN_centroids.webp"))
    assert cent.shape[1] == SOGST_SHN_WIDTHS[2]


# --------------------------------------------------------------------------
# bake_sogst: truncation and the checkpoint temporal fold
# --------------------------------------------------------------------------

def test_truncate_keeps_leading_bands():
    f_rest = np.arange(4 * 15 * 3, dtype=np.float32).reshape(4, 15, 3)
    assert bake_sogst.truncate_sh(f_rest, None) is f_rest
    assert bake_sogst.truncate_sh(f_rest, 0) is None
    np.testing.assert_array_equal(bake_sogst.truncate_sh(f_rest, 1), f_rest[:, :3])
    np.testing.assert_array_equal(bake_sogst.truncate_sh(f_rest, 2), f_rest[:, :8])
    np.testing.assert_array_equal(bake_sogst.truncate_sh(f_rest, 3), f_rest)


def test_truncate_refuses_bands_the_model_lacks():
    f_rest = np.zeros((4, 8, 3), dtype=np.float32)      # a degree-2 model
    with pytest.raises(ValueError, match="exceeds"):
        bake_sogst.truncate_sh(f_rest, 3)


def test_overshoot_clamp_accepts_lower_degrees():
    rng = np.random.default_rng(0)
    f_dc = rng.normal(0, 0.5, (50, 3)).astype(np.float32)
    for coeffs in (3, 8, 15):
        f_rest = rng.normal(0, 2.0, (50, coeffs, 3)).astype(np.float32)
        out = bake_sogst.clamp_sh_overshoot(f_dc, f_rest, 1.5)
        assert out.shape == f_rest.shape


def write_checkpoint(path, n, model_degree, rng):
    """A minimal train_scratch.py checkpoint: the model_args tuple
    convert_from_checkpoint() unpacks, with features_rest laid out as the
    3S+2 temporal copies a rotor 4DGS model stores."""
    S = (model_degree + 1) ** 2 - 1
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
    ident = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    features_dc = rng.uniform(0.5, 1.0, (n, 1, 3))
    features_rest = rng.normal(0.0, 0.05, (n, 3 * S + 2, 3))
    model_args = (
        model_degree, t(rng.normal(0, 0.1, (n, 3))), t(features_dc), t(features_rest),
        t(np.full((n, 3), -4.0)), t(ident), t(np.full((n, 1), 3.0)),
        None, None, None, None, None, 1.0,
        t(rng.uniform(0.2, 0.8, (n, 1))), t(np.full((n, 1), -1.0)), t(ident),
        True, None, 2)
    torch.save((model_args, 1000), path)
    return features_dc[:, 0, :], features_rest, S


@pytest.mark.parametrize("model_degree, sh_degree", [
    (3, None), (3, 2), (3, 1), (3, 0), (2, None), (2, 1)])
def test_checkpoint_bake_at_each_degree(tmp_path, monkeypatch, model_degree, sh_degree):
    rng = np.random.default_rng(model_degree)
    n = 64
    ckpt = tmp_path / "chkpnt1000.pth"
    features_dc, features_rest, S = write_checkpoint(ckpt, n, model_degree, rng)
    ply = tmp_path / "out.ply"
    monkeypatch.setattr(bake_sogst, "PLY_EXPORT_PATH", str(ply))
    monkeypatch.setattr(bake_sogst, "SOGST_EXPORT_OPTIONS", None)

    bake_sogst.convert_from_checkpoint(str(ckpt), None, 0.0, 1.0, 30.0, 0.0,
                                       sh_degree=sh_degree, sh_clamp=0.0)
    _header, fields = read_sogst_ply(str(ply))
    assert len(fields["x"]) == n

    # every temporal DC copy folds into f_dc, whatever degree is baked
    want_dc = features_dc + features_rest[:, S] + features_rest[:, 2 * S + 1]
    got_dc = np.stack([fields[f"f_dc_{c}"] for c in range(3)], axis=1)
    np.testing.assert_allclose(got_dc, want_dc, rtol=1e-5, atol=1e-6)

    degree = model_degree if sh_degree is None else sh_degree
    if degree == 0:
        assert "f_rest" not in fields
        return
    coeffs = (degree + 1) ** 2 - 1
    folded = (features_rest[:, 0:S] + features_rest[:, S + 1:2 * S + 1]
              + features_rest[:, 2 * S + 2:3 * S + 2])[:, :coeffs]      # [n, coeffs, 3]
    want = folded.transpose(0, 2, 1).reshape(n, 3 * coeffs)            # channel-major
    np.testing.assert_allclose(fields["f_rest"], want, rtol=1e-5, atol=1e-6)
