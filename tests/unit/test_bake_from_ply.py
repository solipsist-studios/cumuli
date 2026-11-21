# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""bake_sogst.py with a 4D interchange PLY as input.

cumuli-trainer writes spacetime Gaussians directly, so the bake only runs
its post-filters on them.  These tests pin that nothing else happens: no
slicing, no scale change, and the PLY's own clip scalars.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import bake_sogst
import make_sogst_fixture as fixture
import cumuli_core_path  # noqa: F401
from cumuli_core.ply import read_sogst_ply, write_sogst_ply

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


def test_chroma_clamp_keeps_brightness_and_bounds_colour():
    rng = np.random.default_rng(0)
    f_rest = rng.normal(0, 0.5, (64, 8, 3)).astype(np.float32)
    out = bake_sogst.clamp_sh_chroma(f_rest, 0.02)
    # the luminance (mean over channels) of every coefficient is untouched
    np.testing.assert_allclose(out.mean(axis=2), f_rest.mean(axis=2), atol=1e-6)
    chroma = out - out.mean(axis=2, keepdims=True)
    bound = (np.abs(chroma) * bake_sogst.SH_Y_MAX[None, :8, None]).sum(axis=1).max(axis=1)
    assert bound.max() <= 0.02 + 1e-6
    assert bake_sogst.clamp_sh_chroma(f_rest, 0.0) is f_rest
    assert bake_sogst.clamp_sh_chroma(None, 0.02) is None


def test_chroma_clamp_through_the_ply_bake(monkeypatch, tmp_path):
    ply, fields, _ = _input_ply(tmp_path)
    _, out = _bake(monkeypatch, tmp_path, ply, sh_chroma_clamp=0.02)
    n = len(out["x"])
    c = out["f_rest"].shape[1] // 3
    rest = out["f_rest"].reshape(n, 3, c).transpose(0, 2, 1)
    chroma = rest - rest.mean(axis=2, keepdims=True)
    bound = (np.abs(chroma) * bake_sogst.SH_Y_MAX[None, :c, None]).sum(axis=1).max(axis=1)
    assert bound.max() <= 0.02 + 1e-5


def _write_one_camera_dataset(root, fx=1000.0, w=1000, h=1000):
    """transforms_train.json with one camera at the origin looking down -z
    (OpenGL), so a point at depth d is sampled at fx / d pixels per unit."""
    import json
    root.mkdir(parents=True, exist_ok=True)
    frame = {"file_path": "realcams/cam00/frame_00001", "camera_label": "00", "time": 0.0,
             "fl_x": fx, "fl_y": fx, "cx": w / 2, "cy": h / 2, "w": w, "h": h,
             "transform_matrix": np.eye(4).tolist()}
    (root / "transforms_train.json").write_text(json.dumps({"frames": [frame]}))


def test_smooth_3d_widens_by_the_sampling_rate_and_conserves_density(tmp_path):
    _write_one_camera_dataset(tmp_path / "ds")
    xyz = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, -4.0]])      # depth 2 and 4 in front
    log_scales = np.log(np.array([[0.01, 0.01, 0.0001], [0.05, 0.05, 0.05]], np.float32))
    logit = np.array([2.0, 2.0], np.float32)
    new_ls, new_logit = bake_sogst.smooth_3d(str(tmp_path / "ds"), xyz, log_scales, logit, 1.0)
    sigma = np.array([2.0 / 1000.0, 4.0 / 1000.0])            # 1 px at fx/d
    np.testing.assert_allclose(np.exp(new_ls), np.sqrt(np.exp(2 * log_scales) + sigma[:, None] ** 2), rtol=1e-5)
    # density: alpha * s1 s2 s3 is unchanged
    a0 = 1 / (1 + np.exp(-logit)); a1 = 1 / (1 + np.exp(-new_logit))
    np.testing.assert_allclose(a1 * np.exp(new_ls).prod(1), a0 * np.exp(log_scales).prod(1), rtol=1e-4)
    # the needle (third axis 0.1 mm) fades far more than the round splat
    assert a1[0] / a0[0] < a1[1] / a0[1]
    same = bake_sogst.smooth_3d(str(tmp_path / "ds"), xyz, log_scales, logit, 0.0)
    assert same[0] is log_scales and same[1] is logit


def test_smooth_px_needs_the_training_dataset(tmp_path):
    ply, _, _ = _input_ply(tmp_path)
    result = _run_cli("--input", ply, "--emit_ply", tmp_path / "x.ply", "--smooth_px", "1")
    assert result.returncode != 0
    assert "--smooth_px needs --mask_filter_root" in result.stderr


def test_smooth_time_widens_lifetimes_and_conserves_the_time_integral():
    ts = np.array([0.05, 1.0], np.float32)
    logit = np.array([3.0, 3.0], np.float32)
    new_ts, new_logit = bake_sogst.smooth_time(ts, logit, 3.0, 30.0)
    s = 3.0 / 30.0
    np.testing.assert_allclose(new_ts, np.sqrt(ts.astype(np.float64) ** 2 + s ** 2), rtol=1e-5)
    a0 = 1 / (1 + np.exp(-logit)); a1 = 1 / (1 + np.exp(-new_logit))
    np.testing.assert_allclose(a1 * new_ts, a0 * ts, rtol=1e-4)     # peak x width kept
    assert a1[0] / a0[0] < a1[1] / a0[1]                               # short-lived fade most
    same = bake_sogst.smooth_time(ts, logit, 0.0, 30.0)
    assert same[0] is ts and same[1] is logit


def test_smooth_frames_through_the_ply_bake(monkeypatch, tmp_path):
    ply, fields, meta = _input_ply(tmp_path)
    _, out = _bake(monkeypatch, tmp_path, ply, smooth_frames=2.0)
    assert np.median(out["t_sigma"]) > np.median(np.asarray(fields["t_sigma"]))


def test_cli_refuses_anything_but_an_interchange_ply(tmp_path):
    result = _run_cli("--input", tmp_path / "chkpnt30000.pth", "--emit_ply", tmp_path / "x.ply")
    assert result.returncode != 0
    assert "must be a 4D interchange .ply" in result.stderr


# ---------------------------------------------------------------------------
# Degree-2 motion (docs/sogst-format.md sections 1, 4.8 and 7)
# ---------------------------------------------------------------------------

ACCEL = ("ax", "ay", "az")
MOTION = ("vx", "vy", "vz", "t_center", "ax", "ay", "az")


def _degree2_ply(tmp_path, count=512):
    """The parabola fixture with accel that differs per splat and per axis,
    so a filter that drops a row from one array but not another shows up."""
    fields, meta = fixture.build_fixture(count=count, degree=2)
    rng = np.random.default_rng(17)
    for name in ACCEL:
        fields[name] = rng.normal(0.0, 3.0, count).astype(np.float32)
    path = tmp_path / "trained2.ply"
    write_sogst_ply(str(path), fields, meta["time_min"], meta["time_max"], meta["fps"])
    return path, read_sogst_ply(str(path))[1], meta


def _rows_of(source, out):
    index = {(x, y, z): i for i, (x, y, z) in
             enumerate(zip(source["x"], source["y"], source["z"]))}
    return [index[(x, y, z)] for x, y, z in zip(out["x"], out["y"], out["z"])]


def _assert_motion_stays_on_its_row(source, out):
    assert all(name in out for name in ACCEL)
    rows = _rows_of(source, out)
    for name in MOTION:
        np.testing.assert_array_equal(out[name], np.asarray(source[name])[rows], err_msg=name)


@pytest.mark.parametrize("options", [
    {},
    {"prune_threshold": 0.5},
    {"top_k_fraction": 0.5},
    {"keep_main_cluster": True},
    {"filter_black_floaters": True},
    {"sh_degree": 0},
    {"sh_degree": 1},
    {"sh_chroma_clamp": 0.02},
    {"sh_clamp": 1.5},
    {"smooth_frames": 2.0},
], ids=lambda o: next(iter(o), "defaults"))
def test_every_option_keeps_accel_aligned_with_its_splat(monkeypatch, tmp_path, options):
    """No option evaluates or moves the trajectory at t_center, so survivors
    keep position, velocity, accel and t_center exactly."""
    ply, fields, _ = _degree2_ply(tmp_path)
    header, out = _bake(monkeypatch, tmp_path, ply, **options)
    assert header["motion_degree"] == 2
    assert 0 < len(out["x"]) <= len(fields["x"])
    _assert_motion_stays_on_its_row(fields, out)


def test_the_options_that_filter_do_filter_on_this_fixture(monkeypatch, tmp_path):
    """Otherwise the alignment test above could pass without dropping a row."""
    ply, fields, _ = _degree2_ply(tmp_path)
    n = len(fields["x"])
    for options in ({"prune_threshold": 0.5}, {"top_k_fraction": 0.5}):
        _, out = _bake(monkeypatch, tmp_path, ply, **options)
        assert len(out["x"]) < n, options


def _write_half_mask_dataset(root, times=(0.0, 0.25, 0.5, 0.75, 1.0), size=64):
    """One camera at the origin looking down -z, with a silhouette covering
    the image's left half: from depth 2, x < 0 is inside and x > 0 is out."""
    import json

    from PIL import Image
    frames = []
    (root / "realcams").mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(times):
        rgba = np.zeros((size, size, 4), np.uint8)
        rgba[:, : size // 2, 3] = 255
        Image.fromarray(rgba, "RGBA").save(root / "realcams" / f"f{i}.png")
        frames.append({"file_path": f"realcams/f{i}", "camera_label": "00", "time": t,
                       "fl_x": size / 2, "fl_y": size / 2, "cx": size / 2, "cy": size / 2,
                       "w": size, "h": size, "transform_matrix": np.eye(4).tolist()})
    (root / "transforms_train.json").write_text(json.dumps({"frames": frames}))
    return root


def _splats_for_the_mask_test(path, accel_x, degree=2):
    """Three splats at depth 2 with their t_center mid-clip. All three are
    inside the silhouette at t_center, so only their motion can tell them
    apart. The mask test samples t=0 and t=1, where dt is +-0.5 for all."""
    n = len(accel_x)
    zeros = np.zeros(n, np.float32)
    fields = {
        "x": np.full(n, -1.0, np.float32), "y": np.arange(n, dtype=np.float32) * 0.01,
        "z": np.full(n, -2.0, np.float32),
        "rot_0": np.ones(n, np.float32), "rot_1": zeros, "rot_2": zeros, "rot_3": zeros,
        "scale_0": np.full(n, -4.0, np.float32), "scale_1": np.full(n, -4.0, np.float32),
        "scale_2": np.full(n, -4.0, np.float32),
        "opacity": np.full(n, 4.0, np.float32),
        "f_dc_0": np.ones(n, np.float32), "f_dc_1": np.ones(n, np.float32),
        "f_dc_2": np.ones(n, np.float32),
        "vx": zeros, "vy": zeros, "vz": zeros,
        "t_center": np.full(n, 0.5, np.float32), "t_sigma": np.full(n, 10.0, np.float32),
    }
    if degree == 2:
        fields["ax"] = np.asarray(accel_x, np.float32)
        fields["ay"] = zeros
        fields["az"] = zeros
    write_sogst_ply(str(path), fields, 0.0, 1.0, 4.0)
    return path


def test_mask_filter_follows_the_degree_2_trajectory(monkeypatch, tmp_path):
    """x(t) = -1 + a * (t - 0.5)^2. At a = 8 the splat is at x = +1, outside
    the silhouette, at both tested instants, yet a linear evaluation never
    moves it from x = -1. At a = 1 it stays inside at x = -0.75."""
    root = _write_half_mask_dataset(tmp_path / "ds")
    ply = _splats_for_the_mask_test(tmp_path / "m.ply", [0.0, 1.0, 8.0])
    _, out = _bake(monkeypatch, tmp_path, ply, mask_filter_root=str(root))
    np.testing.assert_array_equal(out["ax"], np.array([0.0, 1.0], np.float32))


def test_mask_filter_drops_a_degree_2_splat_only_for_its_motion(monkeypatch, tmp_path):
    """The same three positions at degree 1 have no motion to leave on."""
    root = _write_half_mask_dataset(tmp_path / "ds")
    ply = _splats_for_the_mask_test(tmp_path / "m.ply", [0.0, 1.0, 8.0], degree=1)
    _, out = _bake(monkeypatch, tmp_path, ply, mask_filter_root=str(root))
    assert len(out["x"]) == 3
    assert "ax" not in out


def test_mask_filter_and_smooth_px_keep_rows_aligned(monkeypatch, tmp_path):
    ply, fields, _ = _degree2_ply(tmp_path)
    root = _write_half_mask_dataset(tmp_path / "ds", size=256)
    _, out = _bake(monkeypatch, tmp_path, ply, mask_filter_root=str(root), smooth_px=0.5)
    assert 0 < len(out["x"]) < len(fields["x"])     # the half mask did drop some
    _assert_motion_stays_on_its_row(fields, out)


@pytest.mark.parametrize("segment_flags, groups", [
    ([], True), (["--segment_duration", "0.2"], True), (["--segment_duration", "0"], False),
])
def test_cli_writes_degree_2_to_the_archive_ply_and_sidecar(tmp_path, segment_flags, groups):
    import json
    import zipfile

    from eval_render import decode_sogst_fields

    ply, fields, meta = _degree2_ply(tmp_path)
    archive, emitted = tmp_path / "c.sogst", tmp_path / "emit.ply"
    result = _run_cli("--input", ply, "--output", archive, "--emit_ply", emitted,
                      "--emit_ply_sidecar", "--no_sh", *segment_flags)
    assert result.returncode == 0, result.stderr

    header, written = read_sogst_ply(str(emitted))
    assert header["motion_degree"] == 2
    _assert_motion_stays_on_its_row(fields, written)
    sidecar = json.loads((tmp_path / "emit.sogst.json").read_text())
    assert sidecar["motion_degree"] == 2

    with zipfile.ZipFile(archive) as zf:
        written_meta = json.loads(zf.read("meta.json"))
        names = [n for n in zf.namelist() if n.endswith("accel_l.webp")]
    assert written_meta["motion"]["degree"] == 2
    assert written_meta["accel"]["files"] == ["accel_l.webp", "accel_u.webp"]
    assert (len(names) > 1) == groups               # one per group when segmented

    # the archive and the PLY describe the same accel, row for row
    _, decoded = decode_sogst_fields(str(archive))
    assert len(decoded["ax"]) == len(written["ax"])
    for name in ACCEL:
        want = np.sort(np.asarray(written[name], np.float64))
        np.testing.assert_allclose(np.sort(np.asarray(decoded[name])), want, atol=2e-2)
    assert (meta["time_min"], meta["time_max"]) == (header["time_min"], header["time_max"])


def test_degree_1_bake_stays_degree_1(monkeypatch, tmp_path):
    """Spec section 4.8: accel is never synthesised."""
    ply, _, _ = _input_ply(tmp_path)
    header, out = _bake(monkeypatch, tmp_path, ply, smooth_frames=1.0, top_k_fraction=0.9)
    assert header["motion_degree"] == 1
    assert not any(name in out for name in ACCEL)
# --- the dark-occluder filter: on by default, optional, configurable ----------

PLANTED = 16


def _ply_with_planted_occluders(tmp_path, count=512):
    """A trained PLY whose first PLANTED splats match the dark-occluder test
    exactly: black, opaque, and larger than every other splat."""
    fields, meta = fixture.build_fixture(count=count, degree=2)
    fields = {k: np.array(v) for k, v in fields.items() if k not in ("ax", "ay", "az")}
    big = max(float(fields[f"scale_{i}"].max()) for i in range(3)) + 1.0
    for i in range(PLANTED):
        for channel in range(3):
            fields[f"f_dc_{channel}"][i] = -3.0       # colour clips to black
        fields["opacity"][i] = 6.0                    # sigmoid is about 0.9975
        fields["scale_0"][i] = big                    # the largest axis of any splat
    path = tmp_path / "planted.ply"
    write_sogst_ply(str(path), fields, meta["time_min"], meta["time_max"], meta["fps"])
    return path, count, big


def _planted_survivors(out, big):
    return int(np.isclose(out["scale_0"], big).sum())


def test_dark_occluder_filter_is_on_by_default(monkeypatch, tmp_path):
    ply, count, big = _ply_with_planted_occluders(tmp_path)
    _, out = _bake(monkeypatch, tmp_path, ply)
    assert _planted_survivors(out, big) == 0
    assert count - len(out["x"]) >= PLANTED


def test_dark_occluder_filter_can_be_turned_off(monkeypatch, tmp_path):
    ply, count, big = _ply_with_planted_occluders(tmp_path)
    _, out = _bake(monkeypatch, tmp_path, ply, dark_occluder=False)
    assert len(out["x"]) == count
    assert _planted_survivors(out, big) == PLANTED


@pytest.mark.parametrize("params", [{"size_percentile": 100.0},     # nothing exceeds the largest splat
                                    {"brightness_thresh": 0.0},     # nothing is darker than black
                                    {"opacity_thresh": 1.0}])       # nothing is more opaque than 1
def test_dark_occluder_thresholds_reach_the_filter(monkeypatch, tmp_path, params):
    ply, count, big = _ply_with_planted_occluders(tmp_path)
    _, out = _bake(monkeypatch, tmp_path, ply, dark_occluder_params=params)
    assert len(out["x"]) == count
    assert _planted_survivors(out, big) == PLANTED


def test_dark_occluder_thresholds_can_widen_the_filter(monkeypatch, tmp_path):
    ply, count, _ = _ply_with_planted_occluders(tmp_path)
    _, default = _bake(monkeypatch, tmp_path, ply)
    _, wider = _bake(monkeypatch, tmp_path, ply, dark_occluder_params={
        "brightness_thresh": 1.1, "opacity_thresh": 0.0, "size_percentile": 0.0})
    assert len(wider["x"]) < len(default["x"])


def test_cli_no_dark_occluder_filter_keeps_every_splat(tmp_path):
    ply, count, big = _ply_with_planted_occluders(tmp_path)
    out = tmp_path / "cli.ply"
    result = _run_cli("--input", ply, "--emit_ply", out, "--prune_threshold", 0,
                      "--no_dark_occluder_filter")
    assert result.returncode == 0, result.stderr
    assert "Dark-occluder filter: off" in result.stdout
    _, baked = read_sogst_ply(str(out))
    assert len(baked["x"]) == count
    assert _planted_survivors(baked, big) == PLANTED


def test_cli_dark_occluder_filter_is_on_without_the_flag(tmp_path):
    ply, count, big = _ply_with_planted_occluders(tmp_path)
    out = tmp_path / "cli.ply"
    result = _run_cli("--input", ply, "--emit_ply", out, "--prune_threshold", 0)
    assert result.returncode == 0, result.stderr
    assert "Dark-occluder filter: dropping" in result.stdout
    _, baked = read_sogst_ply(str(out))
    assert _planted_survivors(baked, big) == 0


def test_cli_dark_occluder_thresholds_reach_the_filter(tmp_path):
    ply, count, big = _ply_with_planted_occluders(tmp_path)
    out = tmp_path / "cli.ply"
    result = _run_cli("--input", ply, "--emit_ply", out, "--prune_threshold", 0,
                      "--dark_occluder_size_percentile", 100)
    assert result.returncode == 0, result.stderr
    _, baked = read_sogst_ply(str(out))
    assert len(baked["x"]) == count


def test_cli_refuses_dark_occluder_thresholds_with_the_filter_off(tmp_path):
    ply, _, _ = _ply_with_planted_occluders(tmp_path)
    result = _run_cli("--input", ply, "--emit_ply", tmp_path / "x.ply",
                      "--no_dark_occluder_filter", "--dark_occluder_opacity", "0.5")
    assert result.returncode != 0
    assert "has no effect with --no_dark_occluder_filter" in result.stderr


@pytest.mark.parametrize("flag,value", [("--dark_occluder_brightness", "1.5"),
                                        ("--dark_occluder_opacity", "-0.1"),
                                        ("--dark_occluder_size_percentile", "101")])
def test_cli_refuses_out_of_range_dark_occluder_thresholds(tmp_path, flag, value):
    ply, _, _ = _ply_with_planted_occluders(tmp_path)
    result = _run_cli("--input", ply, "--emit_ply", tmp_path / "x.ply", flag, value)
    assert result.returncode != 0
    assert f"{flag} must be in" in result.stderr
