# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Reading a set of runs as a table.

The ordering is the point: LPIPS first, because on a masked subject PSNR is
dominated by background pixels no camera configuration affects. Runs that
scored different views are flagged rather than ranked against each other.
"""

import json

import pytest

import compare_experiments as compare


def record(run, lpips=None, psnr=None, ssim=None, views=4, frames=48,
           layout="rings", counts=(8, 8), poses="gt", cameras=16):
    spec = {"layout": layout}
    if layout == "rings":
        spec["rings"] = [{"count": c} for c in counts]
    elif layout == "cage":
        spec["cage"] = {"theta": 12, "phi": 4}
    payload = {
        "run": run,
        "rig_spec": spec,
        "poses": poses,
        "frames": {"count": frames},
        "render": {"train_cameras": cameras, "seconds": 100.0},
    }
    if lpips is not None:
        payload["eval"] = {"lpips": lpips, "psnr_db": psnr, "ssim": ssim,
                           "views": views}
    return payload


def write_runs(tmp_path, records):
    for rec in records:
        run_dir = tmp_path / rec["run"]
        run_dir.mkdir(parents=True)
        (run_dir / "experiment.json").write_text(json.dumps(rec))
    return tmp_path


# ------------------------------------------------------------- discovery
def test_a_run_directory_is_read(tmp_path):
    write_runs(tmp_path, [record("a", 0.1, 30.0, 0.9)])
    found = compare.find_records([tmp_path / "a"])
    assert len(found) == 1 and found[0][1]["run"] == "a"


def test_a_parent_directory_finds_every_run_under_it(tmp_path):
    write_runs(tmp_path, [record("a", 0.1, 30.0, 0.9),
                          record("b", 0.2, 29.0, 0.88)])
    assert len(compare.find_records([tmp_path])) == 2


def test_an_experiment_file_is_accepted_directly(tmp_path):
    write_runs(tmp_path, [record("a", 0.1, 30.0, 0.9)])
    found = compare.find_records([tmp_path / "a" / "experiment.json"])
    assert len(found) == 1


def test_a_path_with_no_experiment_is_skipped_not_fatal(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert compare.find_records([tmp_path / "empty"]) == []
    assert "no experiment.json" in capsys.readouterr().err


def test_malformed_json_is_skipped_with_a_reason(tmp_path, capsys):
    run = tmp_path / "bad"
    run.mkdir()
    (run / "experiment.json").write_text("{not json")
    assert compare.find_records([run]) == []
    assert "skipping" in capsys.readouterr().err


# --------------------------------------------------------------- summary
@pytest.mark.parametrize("layout,kwargs,expected", [
    ("rings", {"counts": (8, 8)}, "rings 8+8"),
    ("rings", {"counts": (12,)}, "rings 12"),
    ("cage", {}, "cage 12x4"),
])
def test_rig_summary_describes_the_layout(layout, kwargs, expected):
    assert compare.rig_summary(record("a", layout=layout, **kwargs)) == expected


def test_row_pulls_the_pose_error_for_the_pose_source_used():
    """A run trained on refined poses should be described by the refined
    error, not the raw solve's."""
    rec = record("a", 0.1, 30.0, 0.9, poses="refined")
    rec["pose_scores"] = {"sources": [
        {"source": "hloc", "position_error_m": {"median": 0.020}},
        {"source": "refined", "position_error_m": {"median": 0.004}},
    ]}
    row = compare.row_for(__import__("pathlib").Path("x/experiment.json"), rec)
    assert row["pose_err_mm"] == pytest.approx(4.0)


def test_row_uses_the_raw_solve_when_that_is_what_trained():
    rec = record("a", 0.1, 30.0, 0.9, poses="hloc")
    rec["pose_scores"] = {"sources": [
        {"source": "hloc", "position_error_m": {"median": 0.020}},
        {"source": "refined", "position_error_m": {"median": 0.004}},
    ]}
    row = compare.row_for(__import__("pathlib").Path("x/experiment.json"), rec)
    assert row["pose_err_mm"] == pytest.approx(20.0)


# --------------------------------------------------------------- ordering
def test_rows_sort_by_lpips_best_first(tmp_path, monkeypatch, capsys):
    write_runs(tmp_path, [record("worse", 0.20, 32.0, 0.90),
                          record("better", 0.05, 28.0, 0.93)])
    import sys

    monkeypatch.setattr(sys, "argv", ["compare_experiments.py", str(tmp_path)])
    compare.main()
    out = capsys.readouterr().out
    assert out.index("better") < out.index("worse")


def test_disagreement_between_psnr_and_lpips_is_called_out(tmp_path, monkeypatch, capsys):
    """The better-looking run here has the worse PSNR, which is exactly the
    case that a PSNR-only comparison gets backwards."""
    write_runs(tmp_path, [record("sharp", 0.05, 28.0, 0.93),
                          record("flat", 0.20, 32.0, 0.90)])
    import sys

    monkeypatch.setattr(sys, "argv", ["compare_experiments.py", str(tmp_path)])
    compare.main()
    out = capsys.readouterr().out
    assert "rank these runs differently" in out


def test_runs_that_scored_different_views_are_flagged(tmp_path, monkeypatch, capsys):
    write_runs(tmp_path, [record("a", 0.05, 30.0, 0.9, views=4),
                          record("b", 0.10, 29.0, 0.9, views=8)])
    import sys

    monkeypatch.setattr(sys, "argv", ["compare_experiments.py", str(tmp_path)])
    compare.main()
    out = capsys.readouterr().out
    assert "different eval setup" in out
    assert "not directly" in out


def test_unscored_runs_are_listed_rather_than_dropped(tmp_path, monkeypatch, capsys):
    write_runs(tmp_path, [record("done", 0.05, 30.0, 0.9), record("pending")])
    import sys

    monkeypatch.setattr(sys, "argv", ["compare_experiments.py", str(tmp_path)])
    compare.main()
    out = capsys.readouterr().out
    assert "carry no eval scores yet" in out
    assert "pending" in out


def test_json_output_carries_every_column(tmp_path, monkeypatch, capsys):
    write_runs(tmp_path, [record("a", 0.05, 30.0, 0.9)])
    import sys

    monkeypatch.setattr(sys, "argv", ["compare_experiments.py", str(tmp_path), "--json"])
    compare.main()
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["run"] == "a"
    assert rows[0]["lpips"] == 0.05
    assert rows[0]["eval_views"] == 4


def test_no_runs_at_all_is_an_error(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["compare_experiments.py", str(tmp_path / "nothing")])
    with pytest.raises(SystemExit):
        compare.main()


def test_sorting_by_psnr_is_available_when_asked_for(tmp_path, monkeypatch, capsys):
    write_runs(tmp_path, [record("lowpsnr", 0.05, 28.0, 0.93),
                          record("highpsnr", 0.20, 32.0, 0.90)])
    import sys

    monkeypatch.setattr(sys, "argv", ["compare_experiments.py", str(tmp_path), "--sort", "psnr"])
    compare.main()
    out = capsys.readouterr().out
    assert out.index("highpsnr") < out.index("lowpsnr")
