# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Tests for seeding a window's init cloud from the previous window.

Two things here are load-bearing and silent when wrong. The advance
transform must put a splat exactly where the model renders it at the
handover, or the seed is worse than the hull it replaces. And the written
cloud must hold MORE points than the trainer's --num_pts, because
readNerfSyntheticInfo only subsamples a cloud longer than its budget and
takes a different branch for a shorter one.
"""


import numpy as np
import pytest

import seed_window_init as seeder
from build_4dgs_dataset import write_ply_with_time

plyfile = pytest.importorskip("plyfile", reason="the seeder reads a hull ply")


def make_fields(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    fields = {
        "x": rng.standard_normal(n).astype(np.float32),
        "y": rng.standard_normal(n).astype(np.float32),
        "z": rng.standard_normal(n).astype(np.float32),
        "vx": rng.standard_normal(n).astype(np.float32),
        "vy": np.zeros(n, np.float32), "vz": np.zeros(n, np.float32),
        "opacity": np.full(n, 4.0, np.float32),
        "t_center": rng.uniform(0.0, 1.25, n).astype(np.float32),
        "t_sigma": np.full(n, 0.5, np.float32),
        "f_dc_0": np.zeros(n, np.float32),
        "f_dc_1": np.zeros(n, np.float32),
        "f_dc_2": np.zeros(n, np.float32),
    }
    return fields


def write_hull(path, n=300_000, seed=1):
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, 3)).astype(np.float32)
    colours = rng.integers(0, 255, (n, 3)).astype(np.uint8)
    times = rng.uniform(0.0, 1.2, n).astype(np.float32)
    write_ply_with_time(path, points, colours, times)
    return path


def test_advance_places_a_splat_where_the_model_renders_it():
    fields = {"x": np.array([1.0], np.float32), "y": np.array([0.0], np.float32),
              "z": np.array([0.0], np.float32), "vx": np.array([2.0], np.float32),
              "vy": np.array([0.0], np.float32), "vz": np.array([0.0], np.float32),
              "t_center": np.array([0.5], np.float32)}
    moved = seeder.advance(fields, 1.25)
    assert moved[0, 0] == pytest.approx(1.0 + 2.0 * 0.75)
    assert moved[0, 1] == pytest.approx(0.0)


def test_advance_at_the_centre_is_a_no_op():
    fields = {"x": np.array([3.0], np.float32), "y": np.array([4.0], np.float32),
              "z": np.array([5.0], np.float32), "vx": np.array([9.0], np.float32),
              "vy": np.array([9.0], np.float32), "vz": np.array([9.0], np.float32),
              "t_center": np.array([2.0], np.float32)}
    assert seeder.advance(fields, 2.0)[0] == pytest.approx([3.0, 4.0, 5.0])


def test_temporal_alpha_peaks_at_the_centre():
    fields = {"opacity": np.array([0.0], np.float32),
              "t_center": np.array([1.0], np.float32),
              "t_sigma": np.array([0.25], np.float32)}
    assert seeder.temporal_alpha(fields, 1.0)[0] == pytest.approx(0.5)
    one_sigma = seeder.temporal_alpha(fields, 1.25)[0]
    assert one_sigma == pytest.approx(0.5 * np.exp(-0.5), rel=1e-6)


def test_temporal_alpha_treats_sigma_as_a_magnitude():
    """t_sigma is used as |sigma| everywhere else, so a negative one must not
    flip a splat's opacity or make it NaN."""
    positive = {"opacity": np.array([1.0], np.float32),
                "t_center": np.array([0.0], np.float32),
                "t_sigma": np.array([0.3], np.float32)}
    negative = dict(positive, t_sigma=np.array([-0.3], np.float32))
    assert seeder.temporal_alpha(negative, 0.2)[0] == pytest.approx(
        seeder.temporal_alpha(positive, 0.2)[0])


def test_colour_conversion_inverts_the_sh_encoding():
    grey = np.float32((0.75 - 0.5) / seeder.SH_C0)
    fields = {f"f_dc_{i}": np.array([grey], np.float32) for i in range(3)}
    assert seeder.splat_colours(fields)[0].tolist() == [191, 191, 191]


def test_seeding_ignores_splats_that_do_not_render(monkeypatch):
    fields = make_fields(500)
    fields["t_center"][:400] = 90.0          # far outside any window
    monkeypatch.setattr(seeder, "decode_sogst_fields",
                        lambda path: (None, fields))
    _, _, live = seeder.seed_points("ignored.sogst", 0.5, 100)
    assert live == 100


def test_seeding_refuses_an_instant_the_window_never_saw(monkeypatch):
    fields = make_fields(50)
    fields["t_center"][:] = 0.5
    monkeypatch.setattr(seeder, "decode_sogst_fields",
                        lambda path: (None, fields))
    with pytest.raises(SystemExit, match="renders nothing there"):
        seeder.seed_points("ignored.sogst", 500.0, 10)


def test_mix_hits_the_requested_total(tmp_path, monkeypatch):
    monkeypatch.setattr(seeder, "decode_sogst_fields",
                        lambda path: (None, make_fields(50_000)))
    hull = write_hull(tmp_path / "hull.ply", 60_000)
    points, colours, times, report = seeder.build_seeded_cloud(
        "ignored.sogst", hull, 0.6, 1.2, total=40_000, seed_fraction=0.5)
    assert len(points) == len(colours) == len(times) == 40_000
    assert report["seeded"] == 20_000
    assert report["hull"] == 20_000


def test_a_short_seed_is_topped_up_from_the_hull(tmp_path, monkeypatch):
    """Only splats that render at the handover are eligible, so the seeded
    half can come up short. The cloud must still reach its total."""
    fields = make_fields(5_000)
    fields["t_center"][:4_500] = 90.0
    monkeypatch.setattr(seeder, "decode_sogst_fields", lambda path: (None, fields))
    hull = write_hull(tmp_path / "hull.ply", 60_000)
    points, _, _, report = seeder.build_seeded_cloud(
        "ignored.sogst", hull, 0.6, 1.2, total=40_000, seed_fraction=0.5)
    assert report["seeded"] == 500
    assert report["requested_seed"] == 20_000
    assert len(points) == 40_000


def test_pure_hull_seed_fraction_leaves_the_cloud_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(seeder, "decode_sogst_fields",
                        lambda path: (None, make_fields(50_000)))
    hull = write_hull(tmp_path / "hull.ply", 60_000)
    _, _, _, report = seeder.build_seeded_cloud(
        "ignored.sogst", hull, 0.6, 1.2, total=40_000, seed_fraction=0.0)
    assert report["seeded"] == 0
    assert report["hull"] == 40_000


def test_seeded_times_span_the_new_window(tmp_path, monkeypatch):
    """A cloud with one timestamp starts every seeded splat centred on the
    handover and leaves the rest of the window to densification."""
    monkeypatch.setattr(seeder, "decode_sogst_fields",
                        lambda path: (None, make_fields(50_000)))
    hull = write_hull(tmp_path / "hull.ply", 10_000)
    _, _, times, _ = seeder.build_seeded_cloud(
        "ignored.sogst", hull, 0.6, 1.25, total=20_000, seed_fraction=1.0)
    assert times.min() < 0.1
    assert times.max() > 1.1


def test_written_cloud_stays_longer_than_the_trainer_budget(tmp_path, monkeypatch):
    """readNerfSyntheticInfo only subsamples a cloud LONGER than --num_pts."""
    monkeypatch.setattr(seeder, "decode_sogst_fields",
                        lambda path: (None, make_fields(50_000)))
    hull = write_hull(tmp_path / "hull.ply", 300_000)
    points, colours, times, _ = seeder.build_seeded_cloud(
        "ignored.sogst", hull, 0.6, 1.2, total=300_000, seed_fraction=0.5)
    out = tmp_path / "points3d.ply"
    write_ply_with_time(out, points, colours, times.astype(np.float32))
    vertex = plyfile.PlyData.read(str(out))["vertex"]
    assert len(vertex) == 300_000 > 200_000
    assert set(["x", "y", "z", "red", "green", "blue", "time"]).issubset(
        vertex.data.dtype.names)
