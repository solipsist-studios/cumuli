# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""bake_sogst.py with a 4D interchange PLY as input.

cumuli-trainer writes spacetime Gaussians directly, so the bake only runs
its post-filters on them.  These tests pin that nothing else happens: no
slicing, no scale change, the PLY's own clip scalars, and the OMG4-only
flags refused rather than silently applied.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import bake_sogst
import make_sogst_fixture as fixture
from sogst_ply import read_sogst_ply, write_sogst_ply

SCRIPT = Path(bake_sogst.__file__)


def _input_ply(tmp_path, degree=2, count=512):
    fields, meta = fixture.build_fixture(count=count, degree=degree)
    fields = {k: v for k, v in fields.items() if k not in ("ax", "ay", "az")}
    path = tmp_path / "trained.ply"
    write_sogst_ply(str(path), fields, meta["time_min"], meta["time_max"], meta["fps"])
    # compare against what is on disk: the PLY stores float32
    return path, read_sogst_ply(str(path))[1], meta


def _bake(monkeypatch, tmp_path, ply, **kw):
    out = tmp_path / "baked.ply"
    monkeypatch.setattr(bake_sogst, "PLY_EXPORT_PATH", str(out))
    monkeypatch.setattr(bake_sogst, "SOGST_EXPORT_OPTIONS", None)
    header, _ = read_sogst_ply(str(ply))
    bake_sogst.convert_from_ply(str(ply), None, header["time_min"], header["time_max"],
                                header["fps"], kw.pop("prune_threshold", 0.0),
                                sh_clamp=kw.pop("sh_clamp", 0.0), **kw)
    return read_sogst_ply(str(out))


def test_kept_splats_pass_through_unchanged(monkeypatch, tmp_path):
    ply, fields, meta = _input_ply(tmp_path)
    header, out = _bake(monkeypatch, tmp_path, ply)

    assert (header["time_min"], header["time_max"], header["fps"]) == \
        (meta["time_min"], meta["time_max"], meta["fps"])
    n_out = len(out["x"])
    assert 0 < n_out <= len(fields["x"])
    # Filters may drop splats but must not alter the survivors: match rows
    # by position (unique in the fixture) and compare every column.
    index = {(x, y, z): i for i, (x, y, z) in
             enumerate(zip(fields["x"], fields["y"], fields["z"]))}
    rows = [index[(x, y, z)] for x, y, z in zip(out["x"], out["y"], out["z"])]
    for name, values in out.items():
        np.testing.assert_allclose(values, np.asarray(fields[name])[rows],
                                   rtol=1e-6, atol=1e-7, err_msg=name)


def test_sh_degree_truncates_the_ply_bands(monkeypatch, tmp_path):
    ply, fields, _ = _input_ply(tmp_path, degree=3)
    _, out = _bake(monkeypatch, tmp_path, ply, sh_degree=1)
    assert out["f_rest"].shape[1] == 9                  # 3 coefficients x 3 channels
    # channel-major: the kept coefficients are the first 3 of each channel's 15
    index = {x: i for i, x in enumerate(fields["x"])}
    rows = [index[x] for x in out["x"]]
    src = np.asarray(fields["f_rest"])[rows].reshape(len(rows), 3, 15)[:, :, :3]
    np.testing.assert_allclose(out["f_rest"], src.reshape(len(rows), 9), rtol=1e-6)


def test_prune_threshold_uses_peak_alpha_inside_the_clip(monkeypatch, tmp_path):
    ply, fields, _ = _input_ply(tmp_path)
    _, all_kept = _bake(monkeypatch, tmp_path, ply)
    _, pruned = _bake(monkeypatch, tmp_path, ply, prune_threshold=0.5)
    assert len(pruned["x"]) < len(all_kept["x"])
    assert (1.0 / (1.0 + np.exp(-pruned["opacity"])) >= 0.5).all()


def _run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


def test_cli_takes_clip_scalars_from_the_ply(tmp_path):
    ply, _, meta = _input_ply(tmp_path)
    out = tmp_path / "cli.ply"
    result = _run_cli("--input", ply, "--emit_ply", out, "--prune_threshold", 0)
    assert result.returncode == 0, result.stderr
    header, _ = read_sogst_ply(str(out))
    assert header["time_max"] == pytest.approx(meta["time_max"])
    assert header["fps"] == pytest.approx(meta["fps"])


@pytest.mark.parametrize("flags, message", [
    (["--time_max", "99"], "disagrees with the PLY header"),
    (["--scale_boost", "1.4672"], "does not apply to a .ply input"),
])
def test_cli_refuses_flags_that_contradict_a_ply(tmp_path, flags, message):
    ply, _, _ = _input_ply(tmp_path)
    result = _run_cli("--input", ply, "--emit_ply", tmp_path / "x.ply", *flags)
    assert result.returncode != 0
    assert message in result.stderr
