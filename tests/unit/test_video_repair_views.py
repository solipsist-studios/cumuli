"""Unit tests for video_repair_views.py (graph assembly and staging only -- the
sampling happens in ComfyUI over HTTP).

The properties worth guarding are the ones that produced silent-looking failures
when they were wrong: a clip length the model rejects, a staging directory that
picks up files it should not, and a graph that quietly generates from scratch
instead of denoising the renders."""

import json

import numpy as np
import pytest
from PIL import Image

import video_repair_views as vrv


def make_sweep(tmp_path, frames=17, pinned=True):
    sweep = tmp_path / "cam02_to_cam06"
    sweep.mkdir(exist_ok=True)
    for index in range(frames):
        Image.fromarray(np.full((8, 8, 3), index, dtype=np.uint8)).save(sweep / f"sweep_{index:04d}.png")
    Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8)).save(sweep / "real_first.png")
    Image.fromarray(np.full((8, 8, 3), 201, dtype=np.uint8)).save(sweep / "real_last.png")
    (sweep / "cameras.json").write_text("{}")  # overwritten by callers that need meta
    meta = {"res": 1024, "frames": [{"idx": i} for i in range(frames)]}
    if pinned:
        meta["real_endpoints"] = {"real_first": {"idx": 0}, "real_last": {"idx": frames - 1}}
    (sweep / "cameras.json").write_text(json.dumps(meta))
    return sweep, meta


# --------------------------------------------------------------------------
# Clip length
# --------------------------------------------------------------------------

@pytest.mark.parametrize("count, ok", [(17, True), (21, True), (33, True), (5, True),
                                       (20, False), (16, False), (4, False), (0, False)])
def test_valid_clip_length(count, ok):
    # Wan compresses time in blocks of four; a 20-frame clip is rejected by the
    # server deep inside the graph, which is a confusing way to find out.
    assert vrv.valid_clip_length(count) is ok


def test_repair_refuses_a_bad_clip_length(tmp_path):
    sweep, _ = make_sweep(tmp_path, frames=20)
    with pytest.raises(ValueError, match="4k\\+1"):
        vrv.repair_sweep(sweep, tmp_path / "out", tmp_path / "in", tmp_path / "comfy_out")


def test_repair_names_a_workable_length_in_the_error(tmp_path):
    sweep, _ = make_sweep(tmp_path, frames=20)
    with pytest.raises(ValueError, match="17 frames"):
        vrv.repair_sweep(sweep, tmp_path / "out", tmp_path / "in", tmp_path / "comfy_out")


# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------

def test_staging_substitutes_the_real_photos_at_the_endpoints(tmp_path):
    # The endpoints are the whole point of the pinning: the clip has to begin and
    # end on real pixels, not on renders.
    sweep, meta = make_sweep(tmp_path, frames=17)
    staging = tmp_path / "staged"
    vrv.stage_frames(sweep, staging, meta)

    first = np.asarray(Image.open(staging / "0000.png").convert("RGB"))
    last = np.asarray(Image.open(staging / "0016.png").convert("RGB"))
    middle = np.asarray(Image.open(staging / "0008.png").convert("RGB"))
    assert first[0, 0, 0] == 200      # real_first
    assert last[0, 0, 0] == 201       # real_last
    assert middle[0, 0, 0] == 8       # the render


def test_staging_uses_renders_when_nothing_is_pinned(tmp_path):
    sweep, meta = make_sweep(tmp_path, frames=17, pinned=False)
    staging = tmp_path / "staged"
    vrv.stage_frames(sweep, staging, meta)
    assert np.asarray(Image.open(staging / "0000.png").convert("RGB"))[0, 0, 0] == 0


def test_staging_directory_holds_only_the_clip(tmp_path):
    # The loader batches everything in the folder, so cameras.json's neighbours
    # (probe_real.png in particular) must not come along and shift the order.
    sweep, meta = make_sweep(tmp_path, frames=17)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(sweep / "probe_real.png")
    staging = tmp_path / "staged"
    vrv.stage_frames(sweep, staging, meta)

    staged = sorted(p.name for p in staging.iterdir())
    assert staged == [f"{i:04d}.png" for i in range(17)]


def test_staging_is_rebuilt_from_scratch(tmp_path):
    sweep, meta = make_sweep(tmp_path, frames=17)
    staging = tmp_path / "staged"
    staging.mkdir()
    (staging / "leftover.png").write_bytes(b"")
    vrv.stage_frames(sweep, staging, meta)
    assert not (staging / "leftover.png").exists()


def test_staging_reports_a_missing_render(tmp_path):
    sweep, meta = make_sweep(tmp_path, frames=17)
    (sweep / "sweep_0008.png").unlink()
    with pytest.raises(FileNotFoundError, match="render_pair_sweep.py"):
        vrv.stage_frames(sweep, tmp_path / "staged", meta)


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

def build(**overrides):
    kwargs = dict(prompt="p", negative="n", denoise=0.15, steps=20, cfg=1.0, seed=0,
                  model_name="m.safetensors", clip_name="c.safetensors", vae_name="v.safetensors",
                  shift=8.0, filename_prefix="pfx")
    kwargs.update(overrides)
    return vrv.build_graph("/renders", kwargs.pop("control_dir", None), 1024, 1024, 17, **kwargs)


def test_graph_denoises_the_renders_rather_than_generating():
    # The sampler must start from the VAE-encoded renders. Starting from the
    # control node's own latent would generate the subject from scratch, which is
    # the measured 18.0 -> 13.3 dB failure.
    graph = build()
    assert graph["sampler"]["inputs"]["latent_image"] == ["base_latent", 0]
    assert graph["base_latent"]["inputs"]["pixels"] == ["renders", 0]


def test_graph_takes_conditioning_from_the_control_node():
    graph = build()
    assert graph["sampler"]["inputs"]["positive"] == ["control", 0]
    assert graph["sampler"]["inputs"]["negative"] == ["control", 1]


def test_graph_always_supplies_a_control_video():
    # Left unset, Wan22FunControlToVideo emits a one-frame concat latent and the
    # sampler dies concatenating it against a real clip.
    assert build()["control"]["inputs"]["control_video"] == ["control_frames", 0]


def test_graph_falls_back_to_the_renders_as_control():
    assert build()["control_frames"]["inputs"]["directory"] == "/renders"


def test_graph_uses_the_skeleton_maps_when_given():
    graph = build(control_dir="/skeletons")
    assert graph["control_frames"]["inputs"]["directory"] == "/skeletons"


def test_graph_loads_images_as_a_batch_not_a_list():
    # ImageBatchPath declares output_is_list, so ComfyUI maps downstream nodes
    # over one image at a time and VAEEncode yields 17 single-frame latents.
    graph = build()
    assert graph["renders"]["class_type"] == "VHS_LoadImagesPath"
    assert graph["control_frames"]["class_type"] == "VHS_LoadImagesPath"


def test_graph_passes_the_denoise_through():
    assert build(denoise=0.3)["sampler"]["inputs"]["denoise"] == pytest.approx(0.3)
