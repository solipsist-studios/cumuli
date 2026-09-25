# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Stage wiring for the synthetic pipeline.

Mirrors tests/unit/test_run_unified_pipeline.py: the subprocess layer is
monkeypatched, so what is under test is the order stages run in, which
stages run at all, and the small conversions the orchestrator does itself.
"""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

import camera_rig_spec as crs
import run_synthetic_pipeline as synth


MANIFEST = {
    "fps": 24.0,
    "subject_bbox": {"min": [-0.5, -0.4, 0.0], "max": [0.5, 0.4, 1.8]},
    "subject_height": 1.8,
}

SPEC = {
    "name": "t", "layout": "rings", "target": "subject_center",
    "rings": [{"count": 4, "radius": 3.0, "height": 1.0}],
    "resolution": [64, 48], "intrinsics": {"lens_mm": 35},
    "eval": {"count": 2},
}


@pytest.fixture
def rig(tmp_path):
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"")
    (tmp_path / "scene_manifest.json").write_text(json.dumps(MANIFEST))
    spec = tmp_path / "rig.json"
    spec.write_text(json.dumps(SPEC))
    return {"blend": blend, "spec": spec, "out": tmp_path / "run"}


def base_argv(rig, **extra):
    argv = ["prog", "--blend", str(rig["blend"]), "--rig_spec", str(rig["spec"]),
            "--out_dir", str(rig["out"]), "--no_validate"]
    for key, value in extra.items():
        if value is True:
            argv.append(f"--{key}")
        else:
            argv += [f"--{key}", str(value)]
    return argv


def patch_stages(monkeypatch, calls):
    def make(name, result=None):
        def stage(*args, **kwargs):
            calls.append(name)
            return result
        return stage

    monkeypatch.setattr(synth, "stage_render", make("render"))
    monkeypatch.setattr(synth, "stage_dataset4d", make("dataset4d"))
    monkeypatch.setattr(synth, "stage_train4d", make("train4d"))
    monkeypatch.setattr(synth, "stage_localize",
                        lambda args, L, n: (calls.append("localize"),
                                            L["flipbook_est"])[1])
    monkeypatch.setattr(synth, "run_script",
                        lambda *a, **k: calls.append("run_script"))


# ------------------------------------------------------------ stage order
def test_ground_truth_path_skips_localize(monkeypatch, rig):
    calls = []
    patch_stages(monkeypatch, calls)
    monkeypatch.setattr("sys.argv", base_argv(rig, frame_count=4))
    synth.main()
    assert calls == ["render", "dataset4d", "train4d"]


@pytest.mark.parametrize("poses", ["hloc", "refined"])
def test_estimated_pose_paths_run_localize(monkeypatch, rig, poses):
    calls = []
    patch_stages(monkeypatch, calls)
    monkeypatch.setattr("sys.argv", base_argv(rig, frame_count=4, poses=poses))
    synth.main()
    assert calls == ["render", "localize", "dataset4d", "train4d"]


def test_start_from_stage_skips_earlier_stages(monkeypatch, rig):
    calls = []
    patch_stages(monkeypatch, calls)
    monkeypatch.setattr("sys.argv",
                        base_argv(rig, frame_count=4, start_from_stage="dataset4d"))
    synth.main()
    assert calls == ["dataset4d", "train4d"]


def test_stop_after_stage_ends_the_run(monkeypatch, rig):
    calls = []
    patch_stages(monkeypatch, calls)
    monkeypatch.setattr("sys.argv",
                        base_argv(rig, frame_count=4, stop_after_stage="render"))
    synth.main()
    assert calls == ["render"]


def test_stop_before_start_is_rejected(monkeypatch, rig):
    calls = []
    patch_stages(monkeypatch, calls)
    monkeypatch.setattr("sys.argv", base_argv(
        rig, frame_count=4, start_from_stage="dataset4d",
        stop_after_stage="render"))
    with pytest.raises(SystemExit):
        synth.main()
    assert calls == []


def test_a_failing_stage_stops_the_pipeline(monkeypatch, rig):
    calls = []
    patch_stages(monkeypatch, calls)

    def boom(*args, **kwargs):
        calls.append("dataset4d")
        raise synth.StageError("dataset build failed")

    monkeypatch.setattr(synth, "stage_dataset4d", boom)
    monkeypatch.setattr("sys.argv", base_argv(rig, frame_count=4))
    with pytest.raises(SystemExit):
        synth.main()
    assert "train4d" not in calls


def test_a_failing_stage_still_records_the_experiment(monkeypatch, rig):
    """A run that died halfway is still evidence, and losing its settings
    means re-deriving what was being tried."""
    calls = []
    patch_stages(monkeypatch, calls)
    monkeypatch.setattr(synth, "stage_render", lambda *a, **k: (_ for _ in ()).throw(
        synth.StageError("render failed")))
    monkeypatch.setattr("sys.argv", base_argv(rig, frame_count=4))
    with pytest.raises(SystemExit):
        synth.main()
    assert (rig["out"] / "experiment.json").is_file()


# ---------------------------------------------------------------- inputs
def test_a_missing_blend_is_reported_before_anything_runs(monkeypatch, rig, tmp_path):
    calls = []
    patch_stages(monkeypatch, calls)
    argv = base_argv(rig)
    argv[argv.index("--blend") + 1] = str(tmp_path / "nope.blend")
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        synth.main()
    assert calls == []


def test_an_invalid_rig_spec_is_reported_before_anything_runs(monkeypatch, rig):
    calls = []
    patch_stages(monkeypatch, calls)
    rig["spec"].write_text(json.dumps({"layout": "spiral", "resolution": [8, 8]}))
    monkeypatch.setattr("sys.argv", base_argv(rig))
    with pytest.raises(SystemExit):
        synth.main()
    assert calls == []


def test_frame_rate_defaults_to_the_scene_manifest(monkeypatch, rig):
    """A clip trained at the wrong rate is stretched in time, and nothing
    downstream can tell."""
    captured = {}
    patch_stages(monkeypatch, [])
    monkeypatch.setattr(synth, "write_experiment",
                        lambda args, *a, **k: (captured.update(fps=args.train_fps), {})[1])
    monkeypatch.setattr("sys.argv", base_argv(rig, frame_count=4))
    synth.main()
    assert captured["fps"] == 24.0


def test_an_explicit_frame_rate_beats_the_manifest(monkeypatch, rig):
    captured = {}
    patch_stages(monkeypatch, [])
    monkeypatch.setattr(synth, "write_experiment",
                        lambda args, *a, **k: (captured.update(fps=args.train_fps), {})[1])
    monkeypatch.setattr("sys.argv", base_argv(rig, frame_count=4, train_fps=60))
    synth.main()
    assert captured["fps"] == 60.0


def test_the_trainer_window_follows_the_rendered_frame_count(monkeypatch, rig):
    """stage_train4d derives the clip duration from train_window, so the two
    must agree or the model is fitted over the wrong time span."""
    captured = {}
    patch_stages(monkeypatch, [])
    monkeypatch.setattr(synth, "write_experiment",
                        lambda args, *a, **k: (captured.update(w=args.train_window), {})[1])
    monkeypatch.setattr("sys.argv", base_argv(rig, frame_count=17))
    synth.main()
    assert captured["w"] == 17


# ----------------------------------------------------------- conversions
def test_bbox_converts_from_blender_z_up_to_the_dataset_world():
    """(x, y, z) becomes (x, z, -y), so the y and z extents swap ends. A
    bbox left in Blender's frame carves a hull nowhere near the subject."""
    lo, hi = synth.blender_bbox_to_dataset(
        {"min": [-1.0, -2.0, 0.0], "max": [1.0, 3.0, 4.0]})
    assert np.allclose(lo, [-1.0, 0.0, -3.0])
    assert np.allclose(hi, [1.0, 4.0, 2.0])


def test_bbox_conversion_round_trips_a_corner():
    lo, hi = synth.blender_bbox_to_dataset(
        {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]})
    assert np.allclose(lo, [0.0, 0.0, -1.0])
    assert np.allclose(hi, [1.0, 1.0, 0.0])


@pytest.mark.parametrize("count,n,expected", [
    (48, 1, [24]),
    (10, 2, [0, 9]),
    (5, 5, [0, 1, 2, 3, 4]),
    (4, 10, [0, 1, 2, 3]),     # never more instants than frames
])
def test_refinement_instants_spread_across_the_clip(count, n, expected):
    assert synth.spread_indices(count, n) == expected


def test_refinement_instants_include_both_ends():
    got = synth.spread_indices(48, 10)
    assert got[0] == 0 and got[-1] == 47 and len(got) == 10


# ----------------------------------------------------------- experiment
def test_experiment_record_leads_with_lpips(tmp_path):
    out = tmp_path / "run"
    L = synth.build_layout(out)
    out.mkdir(parents=True)
    L["eval4d_report"].write_text(json.dumps({
        "mean": {"lpips": 0.081, "psnr_db": 31.2, "ssim": 0.95},
        "views": [{}, {}, {}, {}]}))
    args = NS(out_dir=out, rig_spec="r.json", blend="b.blend", action=None,
              frame_start=1, frame_count=8, frame_step=1, train_fps=24.0,
              samples=64, engine="CYCLES", no_denoise=False, poses="gt",
              masks="gt", total_train_iters=3000, num_pts=1000,
              dataset_downscale=2)
    record = synth.write_experiment(args, L, 0.0, SPEC, MANIFEST)

    assert record["eval"]["lpips"] == 0.081
    assert record["eval"]["views"] == 4
    assert list(record["eval"])[0] == "lpips"
    assert json.loads(L["experiment"].read_text())["eval"]["psnr_db"] == 31.2


def test_experiment_record_survives_a_run_with_no_scores(tmp_path):
    out = tmp_path / "run"
    out.mkdir(parents=True)
    L = synth.build_layout(out)
    args = NS(out_dir=out, rig_spec="r.json", blend="b.blend", action=None,
              frame_start=1, frame_count=8, frame_step=1, train_fps=24.0,
              samples=64, engine="CYCLES", no_denoise=False, poses="gt",
              masks="gt", total_train_iters=3000, num_pts=1000,
              dataset_downscale=2)
    record = synth.write_experiment(args, L, 0.0, SPEC, MANIFEST)
    assert "eval" not in record
    assert record["frames"]["count"] == 8


def test_layout_keys_match_the_names_the_shared_train_stage_reads():
    """stage_train4d is reused verbatim from the capture pipeline and looks
    these up by name."""
    L = synth.build_layout(__import__("pathlib").Path("/tmp/run"))
    for key in ("dataset4d", "train4d_config", "train4d_model", "sogst_out",
                "eval4d_report"):
        assert key in L


def test_spec_resolves_to_the_camera_counts_the_banner_reports(tmp_path):
    resolved = crs.resolve_rig(SPEC, MANIFEST)
    assert len(resolved["train"]) == 4
    assert len(resolved["eval"]) == 2


def test_the_hull_bbox_is_passed_as_one_joined_argument(monkeypatch, rig):
    """A subject extending left of the origin gives a bbox starting with a
    minus sign, and argparse reads a bare "-0.5,..." as another flag. Caught
    by an end-to-end run, so the joined form is pinned here."""
    captured = {}
    monkeypatch.setattr(synth, "run_script",
                        lambda name, args, **k: captured.update(args=args))
    L = synth.build_layout(rig["out"])
    flipbook = rig["out"] / "flipbook_src"
    (flipbook / "frame_0000").mkdir(parents=True)
    (flipbook / "frame_0000" / "transforms.json").write_text(
        json.dumps({"frames": [{"camera_label": "00"}, {"camera_label": "01"}]}))

    args = NS(train_fps=24.0, dataset_downscale=2, dataset_jobs=4,
              hull_min_views=9, eval_camera=None)
    synth.stage_dataset4d(args, L, flipbook, MANIFEST)

    joined = [str(a) for a in captured["args"]]
    bbox_args = [a for a in joined if a.startswith("--init_bbox")]
    assert len(bbox_args) == 1
    assert bbox_args[0].startswith("--init_bbox=")
    assert "--init_bbox" not in joined          # never the split form
    # Six numbers, and the first is negative for this subject.
    values = bbox_args[0].split("=", 1)[1].split(",")
    assert len(values) == 6
    assert values[0].startswith("-")


def test_the_hull_min_views_is_clamped_to_the_camera_count(monkeypatch, rig):
    """The builder's default of 9 exceeds a small rig's camera count, and
    the hull would then be empty by construction."""
    captured = {}
    monkeypatch.setattr(synth, "run_script",
                        lambda name, args, **k: captured.update(args=args))
    L = synth.build_layout(rig["out"])
    flipbook = rig["out"] / "flipbook_src"
    (flipbook / "frame_0000").mkdir(parents=True)
    (flipbook / "frame_0000" / "transforms.json").write_text(
        json.dumps({"frames": [{"camera_label": "00"}, {"camera_label": "01"}]}))

    args = NS(train_fps=24.0, dataset_downscale=2, dataset_jobs=4,
              hull_min_views=9, eval_camera=None)
    synth.stage_dataset4d(args, L, flipbook, MANIFEST)
    joined = [str(a) for a in captured["args"]]
    assert joined[joined.index("--hull_min_views") + 1] == "2"
