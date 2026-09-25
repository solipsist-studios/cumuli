# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Tests for choosing where to cut a clip into 4DGS windows.

The planner exists because the measured coefficients say boundary placement
cannot improve quality or reduce cost, leaving only the window count and the
seam positions. That conclusion is data, not design, so the tests that matter
most check the dynamic program is a real optimiser rather than a dressed-up
equal split: it is compared against brute force, and given a fitted motion
coefficient large enough to matter it must move the boundaries.
"""

import itertools
import json

import numpy as np
import pytest

import plan_temporal_windows as planner


def flat_signal(steps=24):
    return np.ones(steps)


def spiky_signal(steps=24, spike_at=(6, 7, 8), height=8.0):
    signal = np.ones(steps)
    for i in spike_at:
        signal[i] = height
    return signal


def brute_force(signal, n_windows, coef, min_frames, seam_smoothing=1):
    """Every legal split, scored the way plan_windows scores one."""
    total_frames = len(signal) + 1
    density = planner.normalise(signal)
    seam_density = planner.smooth(density, seam_smoothing)
    cumulative = np.concatenate([[0.0], np.cumsum(density)])
    total_motion = cumulative[-1]
    best, best_cut = None, None
    for cuts in itertools.combinations(range(1, total_frames), n_windows - 1):
        boundaries = [0] + list(cuts) + [total_frames]
        if any(b - a < min_frames for a, b in zip(boundaries, boundaries[1:])):
            continue
        cost = 0.0
        for a, b in zip(boundaries, boundaries[1:]):
            share = (cumulative[b - 1] - cumulative[a]) / total_motion
            cost += (b - a) / total_frames * coef.window_quality(b - a, share)
        for b in boundaries[1:-1]:
            cost += coef.seam_cost(seam_density[b - 1])
        if best is None or cost < best - 1e-15:
            best, best_cut = cost, boundaries
    return best, best_cut


def test_normalise_gives_mean_one():
    assert planner.normalise(np.array([1.0, 3.0, 8.0])).mean() == pytest.approx(1.0)


def test_normalise_survives_a_dead_signal():
    """A clip where nothing moves must plan, not divide by zero."""
    assert np.all(planner.normalise(np.zeros(5)) == 1.0)


def test_smooth_is_a_boxcar():
    smoothed = planner.smooth(np.array([0.0, 0.0, 9.0, 0.0, 0.0]), 3)
    assert smoothed[2] == pytest.approx(3.0)
    assert smoothed[1] == pytest.approx(3.0)


def test_pick_cameras_spreads_around_the_rig():
    labels = [f"{i:02d}" for i in range(12)]
    assert planner.pick_cameras(labels, 4) == ["00", "03", "06", "09"]
    assert planner.pick_cameras(labels, 99) == labels


def test_dp_matches_brute_force():
    coef = planner.Coefficients.defaults()
    signal = spiky_signal(24)
    for n in (2, 3, 4):
        expected_cost, expected_cut = brute_force(signal, n, coef, 3, 1)
        plan = planner.plan_windows(signal, n, coef, min_frames=3,
                                    seam_smoothing=1)
        assert plan.cost == pytest.approx(expected_cost, rel=1e-12)
        assert plan.boundaries == expected_cut


def test_equal_lengths_when_only_length_costs():
    """With no seam term and no motion term the objective is sum of squared
    lengths, which is minimised by cutting evenly."""
    coef = planner.Coefficients.defaults()
    coef.quality_per_motion = 0.0
    coef.seam_intercept = coef.seam_per_motion = 0.0
    plan = planner.plan_windows(spiky_signal(24), 3, coef, min_frames=2)
    lengths = [b - a for a, b in zip(plan.boundaries, plan.boundaries[1:])]
    assert max(lengths) - min(lengths) <= 1


def test_seam_cost_pulls_cuts_onto_quiet_frames():
    coef = planner.Coefficients.defaults()
    coef.seam_per_motion = 0.05          # loud enough to outweigh length
    signal = np.ones(24)
    signal[11] = 40.0                    # the cut an even split would take
    plan = planner.plan_windows(signal, 2, coef, min_frames=3, seam_smoothing=1)
    assert plan.boundaries[1] != 12


def test_a_real_motion_coefficient_moves_the_boundaries():
    """The reduction to near-equal lengths is a property of the measured
    coefficients, not of the program. Refit on material where motion drives
    quality and the plan must respond, or the DP is decoration."""
    coef = planner.Coefficients.defaults()
    coef.seam_intercept = coef.seam_per_motion = 0.0
    coef.quality_per_motion = 0.02       # far above the fitted -9.9e-05
    signal = np.ones(24)
    signal[:6] = 20.0                    # all the motion at the start
    plan = planner.plan_windows(signal, 3, coef, min_frames=2)
    lengths = [b - a for a, b in zip(plan.boundaries, plan.boundaries[1:])]
    assert lengths[0] < lengths[-1]
    assert max(lengths) - min(lengths) > 1


def test_minimum_window_length_is_respected():
    plan = planner.plan_windows(flat_signal(30), 3, min_frames=9)
    lengths = [b - a for a, b in zip(plan.boundaries, plan.boundaries[1:])]
    assert min(lengths) >= 9


def test_maximum_window_length_is_respected():
    plan = planner.plan_windows(flat_signal(40), 4, min_frames=2, max_frames=12)
    lengths = [b - a for a, b in zip(plan.boundaries, plan.boundaries[1:])]
    assert max(lengths) <= 12


def test_impossible_constraints_are_refused():
    with pytest.raises(ValueError, match="do not fit"):
        planner.plan_windows(flat_signal(10), 4, min_frames=9)


def test_choose_window_count_takes_the_cheapest_reaching_a_target():
    signal = flat_signal(120)
    chosen, plans = planner.choose_window_count(signal, min_frames=8,
                                                target_lpips=0.0080)
    assert plans[chosen].stitched_lpips <= 0.0080
    for fewer in range(1, chosen):
        if fewer in plans:
            assert plans[fewer].stitched_lpips > 0.0080


def test_choose_window_count_without_a_target_minimises_stitched_error():
    signal = flat_signal(120)
    chosen, plans = planner.choose_window_count(signal, min_frames=8)
    assert plans[chosen].stitched_lpips == min(
        p.stitched_lpips for p in plans.values())


def test_more_windows_never_costs_fewer_splats():
    coef = planner.Coefficients.defaults()
    assert coef.splats(4) > coef.splats(3) > coef.splats(2)


def test_plan_json_puts_seams_between_frames():
    """The cut sits between two frames, so its time is (b - 0.5) / fps. Any
    other convention makes merge_sogst_segments cut somewhere else than the
    plan predicted."""
    plan = planner.plan_windows(flat_signal(120), 4, min_frames=8)
    document = planner.build_plan_json(plan, "/runs/take", 100, 24.0, "images",
                                       planner.Coefficients.defaults(),
                                       "/runs/win_{index}")
    for boundary, seam in zip(plan.boundaries[1:-1], document["seams_seconds"]):
        assert seam == pytest.approx((boundary - 0.5) / 24.0)
    assert document["windows"][0]["frame_start"] == 100
    assert document["windows"][1]["out_dir"] == "/runs/win_1"


def test_plan_json_window_frames_cover_the_clip_once():
    plan = planner.plan_windows(flat_signal(120), 4, min_frames=8)
    document = planner.build_plan_json(plan, "/runs/take", 100, 24.0, "images",
                                       planner.Coefficients.defaults(), None)
    covered = sum(w["frame_count"] for w in document["windows"])
    assert covered == 121
    starts = [w["local_frame_start"] for w in document["windows"]]
    assert starts == sorted(starts)


def test_error_signal_reads_a_report(tmp_path):
    views = [{"time": f / 24.0, "lpips": 0.01 if f < 5 else 0.02}
             for f in range(10) for _ in range(2)]
    report = tmp_path / "eval_4d.json"
    report.write_text(json.dumps({"views": views}))
    signal = planner.error_motion_signal(report, fps=24.0)
    assert len(signal) == 9
    assert signal[0] == pytest.approx(0.01)
    assert signal[-1] == pytest.approx(0.02)


def test_fit_recovers_planted_coefficients(tmp_path):
    """Synthesise runs from a known law and check the fit reads it back."""
    import zipfile

    signal = np.linspace(0.5, 1.5, 60)
    signal = planner.normalise(signal)
    truth = dict(q0=0.005, q1=3e-05, q2=0.002, s0=100_000.0, s1=900_000.0)
    runs, offsets = [], []
    for i, (offset, count) in enumerate([(0, 61), (0, 31), (30, 31), (0, 16),
                                         (45, 16)]):
        run = tmp_path / f"run_{i}"
        run.mkdir()
        share = signal[offset:offset + count - 1].sum() / signal.sum()
        lpips = truth["q0"] + truth["q1"] * count + truth["q2"] * share
        (run / "experiment.json").write_text(json.dumps(
            {"frames": {"start": 100, "count": count, "fps": 24.0}}))
        (run / "eval_4d.json").write_text(json.dumps(
            {"views": [{"time": 0.0, "lpips": lpips}]}))
        with zipfile.ZipFile(run / "splat_4d.sogst", "w") as archive:
            archive.writestr("meta.json", json.dumps(
                {"count": truth["s0"] + truth["s1"] * share}))
        runs.append(str(run))
        offsets.append(offset)

    coef, quality = planner.fit_coefficients(runs, signal, offsets)
    assert coef.quality_intercept == pytest.approx(truth["q0"], abs=1e-9)
    assert coef.quality_per_frame == pytest.approx(truth["q1"], rel=1e-6)
    assert coef.quality_per_motion == pytest.approx(truth["q2"], rel=1e-6)
    assert coef.splats_fixed == pytest.approx(truth["s0"], rel=1e-6)
    assert coef.splats_per_motion == pytest.approx(truth["s1"], rel=1e-6)
    assert quality["r2_quality"] == pytest.approx(1.0)


def test_seam_fit_splits_the_stitched_gap(tmp_path):
    """The whole stitched-minus-pooled gap must be attributed, and the seam
    sitting in the busier place must carry more of it."""
    frames = 60
    per_frame = np.full(frames, 0.0070)
    per_frame[19:24] = 0.0100        # a loud seam at frame 21
    per_frame[39:44] = 0.0080        # a quieter one at frame 41
    views = [{"time": f / 24.0, "lpips": float(per_frame[f])}
             for f in range(frames)]
    report = tmp_path / "stitched.json"
    report.write_text(json.dumps({"views": views}))

    signal = np.ones(frames - 1)
    signal[20] = 5.0
    pooled = float(per_frame.mean()) - 0.0006

    s0, s1, detail = planner.fit_seam_coefficients(
        report, pooled, [21, 41], signal, fps=24.0, smoothing=1)
    assert sum(detail["costs"]) == pytest.approx(0.0006, rel=1e-9)
    assert detail["costs"][0] > detail["costs"][1]
    assert s1 > 0


def test_seam_fit_refuses_a_clip_with_no_bumps(tmp_path):
    views = [{"time": f / 24.0, "lpips": 0.007} for f in range(30)]
    report = tmp_path / "flat.json"
    report.write_text(json.dumps({"views": views}))
    with pytest.raises(SystemExit, match="nothing to attribute"):
        planner.fit_seam_coefficients(report, 0.006, [10, 20],
                                      np.ones(29), fps=24.0)


# ---------------------------------------------------------------- uniform default

def test_uniform_split_matches_the_measured_arm():
    # Arm A of the window matrix: 121 frames, four windows, 31/30/30/30.
    boundaries = planner.uniform_boundaries(121, 4)
    assert boundaries == [0, 31, 61, 91, 121]


def test_uniform_split_covers_every_frame_once():
    for frames in (8, 25, 121):
        for n in range(1, 8):
            b = planner.uniform_boundaries(frames, n)
            lengths = [y - x for x, y in itertools.pairwise(b)]
            assert b[0] == 0 and b[-1] == frames
            assert max(lengths) - min(lengths) <= 1


def test_uniform_window_count_from_seconds():
    assert planner.uniform_window_count(121, 24.0, 1.25) == 4
    assert planner.uniform_window_count(10, 24.0, 1.25) == 1


def _fake_run(tmp_path, frames=121):
    run = tmp_path / "run"
    for i in range(frames):
        (run / "flipbook_src" / f"frame_{i:04d}").mkdir(parents=True)
    (run / "experiment.json").write_text(json.dumps(
        {"frames": {"start": 100, "count": frames, "fps": 24}}))
    return run


def _run_main(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["plan_temporal_windows.py", *argv])
    planner.main()


def test_default_is_a_uniform_cut_that_reads_no_images(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)

    def no_signal(*a, **k):
        raise AssertionError("uniform mode computed a motion signal")
    monkeypatch.setattr(planner, "image_motion_signal", no_signal)
    out = tmp_path / "plan.json"
    _run_main(monkeypatch, ["--run", str(run), "--out", str(out)])

    plan = json.loads(out.read_text())
    assert plan["cut"] == "uniform"
    assert plan["signal"] == "none"
    assert [w["frame_count"] for w in plan["windows"]] == [31, 30, 30, 30]
    assert plan["windows"][0]["frame_start"] == 100


def test_windows_overrides_the_uniform_length(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    out = tmp_path / "plan.json"
    _run_main(monkeypatch, ["--run", str(run), "--windows", "3",
                            "--out", str(out)])
    plan = json.loads(out.read_text())
    assert [w["frame_count"] for w in plan["windows"]] == [41, 40, 40]


@pytest.mark.parametrize("flag", [
    ["--signal", "joints"], ["--target_lpips", "0.008"],
    ["--fit_from", "somewhere"], ["--min_frames", "4"],
    ["--seam_smoothing", "3"], ["--splat_budget", "2e6"],
])
def test_planner_flags_need_adaptive(tmp_path, monkeypatch, flag):
    run = _fake_run(tmp_path)
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["--run", str(run), *flag])


def test_window_seconds_is_refused_with_adaptive(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["--run", str(run), "--cut", "adaptive",
                                "--window_seconds", "1.0"])


def test_adaptive_runs_the_planner(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, frames=25)
    monkeypatch.setattr(planner, "image_motion_signal",
                        lambda *a, **k: spiky_signal())
    out = tmp_path / "plan.json"
    _run_main(monkeypatch, ["--run", str(run), "--cut", "adaptive",
                            "--windows", "2", "--out", str(out)])
    plan = json.loads(out.read_text())
    assert plan["cut"] == "adaptive"
    assert plan["signal"] == "images"
    assert sum(w["frame_count"] for w in plan["windows"]) == 25
