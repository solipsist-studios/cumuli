# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Wiring for run_window_plan.py.

The subprocess layer is monkeypatched, so what is under test is which
scripts run, in what order, in which interpreter, and with which flags.
"""

import json

import pytest

import run_window_plan as rwp
from run_unified_pipeline import CONDA_ENV


def write_plan(tmp_path, n_windows=2, frames=5):
    windows = []
    for i in range(n_windows):
        windows.append({
            "index": i,
            "frame_start": 100 + i * frames,
            "local_frame_start": i * frames,
            "frame_count": frames,
            "offset_seconds": i * frames / 24.0,
            "out_dir": str(tmp_path / f"win_{i}"),
        })
    plan = tmp_path / "window_plan.json"
    plan.write_text(json.dumps({"fps": 24.0, "windows": windows,
                                "seams_seconds": []}))
    return plan


def write_master(tmp_path, n_frames=10):
    master = tmp_path / "master"
    for sub in ("flipbook_src", "eval_src"):
        for i in range(n_frames):
            (master / sub / f"frame_{i:04d}").mkdir(parents=True)
    return master


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    def fake_run_script(script, args, conda_env=None, label=None, **kwargs):
        recorded.append({"script": script, "args": [str(a) for a in args],
                         "conda_env": conda_env})

    monkeypatch.setattr(rwp, "run_script", fake_run_script)
    return recorded


def run_main(monkeypatch, tmp_path, *extra):
    plan = write_plan(tmp_path)
    master = write_master(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "run_window_plan.py", "--plan", str(plan), "--master", str(master),
        "--blend", "b.blend", "--rig_spec", "r.json", *extra])
    rwp.main()


# ------------------------------------------------------ pass-through flags
@pytest.mark.parametrize("argv,expected", [
    (["--extra=--samples 64"], ["--samples", "64"]),
    (["--extra=--skip_eval"], ["--skip_eval"]),
    (["--extra", "--samples 64 --engine CYCLES"],
     ["--samples", "64", "--engine", "CYCLES"]),
    ([], []),
])
def test_extra_carries_flags_through_to_the_pipeline(argv, expected):
    """nargs="*" rejected every value starting with "-", so no flag could
    ever be passed through. One shell-split string can carry any."""
    args = rwp.build_parser().parse_args(
        ["--plan", "p", "--master", "m", "--blend", "b", "--rig_spec", "r"] + argv)
    command = rwp.pipeline_args(args, "out", 100, 5, "dataset4d", None)
    assert command[len(command) - len(expected):] == expected


def test_merge_args_reach_the_merge(monkeypatch, tmp_path, calls):
    run_main(monkeypatch, tmp_path, "--merge_out", str(tmp_path / "s.sogst"),
             "--merge_args=--mode hard --fade 0.2", "--skip_eval")
    merge = next(c for c in calls if c["script"] == "merge_sogst_segments.py")
    assert merge["args"][-4:] == ["--mode", "hard", "--fade", "0.2"]


# ------------------------------------------------------------ interpreters
def test_helpers_run_in_the_cumuli_env_and_the_pipeline_does_not(
        monkeypatch, tmp_path, calls):
    """eval_render.py needs torch; running it with whatever Python launched
    this script worked only when that happened to be the cumuli env."""
    run_main(monkeypatch, tmp_path, "--seed",
             "--merge_out", str(tmp_path / "s.sogst"))
    envs = {c["script"]: c["conda_env"] for c in calls}
    assert envs["run_synthetic_pipeline.py"] is None
    for helper in ("seed_window_init.py", "merge_sogst_segments.py",
                   "eval_render.py"):
        assert envs[helper] == CONDA_ENV


def test_seeded_windows_build_then_seed_then_train(monkeypatch, tmp_path, calls):
    run_main(monkeypatch, tmp_path, "--seed")
    order = [(c["script"], c["args"][c["args"].index("--start_from_stage") + 1]
              if "--start_from_stage" in c["args"] else None) for c in calls]
    assert order == [
        ("run_synthetic_pipeline.py", "dataset4d"),       # window 0, one pass
        ("run_synthetic_pipeline.py", "dataset4d"),       # window 1, build only
        ("seed_window_init.py", None),
        ("run_synthetic_pipeline.py", "train4d"),
    ]
    window1_build = calls[1]["args"]
    assert window1_build[window1_build.index("--stop_after_stage") + 1] == "dataset4d"


def test_a_failing_step_stops_with_its_label(monkeypatch, tmp_path):
    def failing(*args, **kwargs):
        raise rwp.StageError("window 0 failed (exit 1)")

    monkeypatch.setattr(rwp, "run_script", failing)
    with pytest.raises(SystemExit, match="window 0 failed"):
        run_main(monkeypatch, tmp_path)
