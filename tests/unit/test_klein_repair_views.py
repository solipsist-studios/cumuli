"""Unit tests for klein_repair_views.py (graph construction and alpha cleanup --
no ComfyUI required)."""

import json

import numpy as np
import pytest

import klein_repair_views as krv


# --------------------------------------------------------------------------
# Alpha cleanup
# --------------------------------------------------------------------------

def test_clean_alpha_keeps_the_subject_and_drops_a_detached_speck():
    alpha = np.zeros((200, 200), dtype=np.uint8)
    alpha[50:150, 50:150] = 255   # subject
    alpha[10:14, 180:184] = 255   # floater far away

    cleaned = krv.clean_alpha(alpha)
    assert cleaned[100, 100] > 200
    assert cleaned[11, 181] == 0


def test_clean_alpha_keeps_a_component_attached_to_the_subject():
    # A raised hand can sit a few pixels clear of the body in a render; dropping
    # it would train a one-armed subject.
    alpha = np.zeros((200, 200), dtype=np.uint8)
    alpha[50:150, 50:150] = 255
    alpha[95:105, 152:170] = 255  # 2px gap from the body, well inside the dilation radius

    cleaned = krv.clean_alpha(alpha)
    assert cleaned[100, 160] > 100


def test_clean_alpha_never_exceeds_the_input():
    rng = np.random.default_rng(0)
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[20:80, 20:80] = rng.integers(60, 255, size=(60, 60), dtype=np.uint8)

    cleaned = krv.clean_alpha(alpha)
    assert np.all(cleaned <= alpha)


def test_clean_alpha_on_an_empty_render_returns_empty():
    assert not krv.clean_alpha(np.zeros((50, 50), dtype=np.uint8)).any()


def test_clean_alpha_closes_pinholes_inside_the_silhouette():
    alpha = np.zeros((120, 120), dtype=np.uint8)
    alpha[30:90, 30:90] = 255
    alpha[59:61, 59:61] = 0  # small hole

    cleaned = krv.clean_alpha(alpha)
    # the hole is closed in the binary mask but clipped back to the original alpha,
    # which is 0 there -- so the result must not invent coverage
    assert cleaned[60, 60] == 0
    assert cleaned[40, 40] > 200


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def graph_for(references=("a.png", "b.png"), **kwargs):
    params = dict(prompt="p", negative_prompt="n", denoise=0.15, seed=12345, steps=8, cfg=2.0,
                  unet_name="u.safetensors", clip_name="c.safetensors", vae_name="v.safetensors")
    params.update(kwargs)
    return krv.build_graph("in/frame.png", "out/frame", list(references), **params)


def test_graph_wires_the_sampler_to_the_encoded_input_image():
    graph = graph_for()
    assert graph["sampler"]["inputs"]["latent_image"] == ["in_lat", 0]
    assert graph["in_lat"]["inputs"]["pixels"] == ["in_img", 0]
    assert graph["in_img"]["inputs"]["image"] == "in/frame.png"


def test_reference_latents_chain_onto_both_conditioning_branches():
    # Anchoring only the positive branch was measurably worse for identity.
    graph = graph_for()
    positive = graph["sampler"]["inputs"]["positive"][0]
    negative = graph["sampler"]["inputs"]["negative"][0]

    assert positive == "pos_txt_ref1"
    assert negative == "neg_txt_ref1"
    assert graph["pos_txt_ref0"]["inputs"]["conditioning"] == ["pos_txt", 0]
    assert graph["pos_txt_ref1"]["inputs"]["conditioning"] == ["pos_txt_ref0", 0]
    assert graph["pos_txt_ref0"]["inputs"]["latent"] == ["ref0_lat", 0]
    assert graph["pos_txt_ref1"]["inputs"]["latent"] == ["ref1_lat", 0]


def test_graph_without_references_still_samples():
    graph = graph_for(references=())
    assert graph["sampler"]["inputs"]["positive"] == ["pos_txt", 0]
    assert not any(key.startswith("ref") for key in graph)


def test_references_are_scaled_before_encoding():
    graph = graph_for()
    assert graph["ref0_scale"]["inputs"]["width"] == krv.REFERENCE_SIZE
    assert graph["ref0_lat"]["inputs"]["pixels"] == ["ref0_scale", 0]


@pytest.mark.parametrize("denoise", [0.15, 0.22])
def test_sampler_settings_are_passed_through(denoise):
    graph = graph_for(denoise=denoise, seed=999, steps=6, cfg=3.5)
    sampler = graph["sampler"]["inputs"]
    assert sampler["denoise"] == denoise
    assert sampler["seed"] == 999
    assert sampler["steps"] == 6
    assert sampler["cfg"] == 3.5
    assert sampler["sampler_name"] == "euler"


def test_prompts_reach_the_text_encoders():
    graph = graph_for(prompt="a boy", negative_prompt="adult")
    assert graph["pos_txt"]["inputs"]["text"] == "a boy"
    assert graph["neg_txt"]["inputs"]["text"] == "adult"


def test_graph_is_json_serialisable():
    # It goes to ComfyUI as JSON; a stray numpy scalar would fail at request time.
    json.dumps({"prompt": graph_for()})


# --------------------------------------------------------------------------
# Reference selection
# --------------------------------------------------------------------------

def write_crop(directory, label):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"head_{label}.png"
    path.write_bytes(b"png")
    return path


def write_keypoints(kp2d_dir, label, eye_separation, score=0.9):
    camera_dir = kp2d_dir / label
    camera_dir.mkdir(parents=True, exist_ok=True)
    keypoints = [[0.0, 0.0], [-eye_separation / 2, 0.0], [eye_separation / 2, 0.0]]
    (camera_dir / "000000.json").write_text(json.dumps(
        {"instance_info": [{"keypoints": keypoints, "keypoint_scores": [score] * 3,
                            "bbox": [0, 0, 100, 100]}]}))


def test_ranking_prefers_the_most_head_on_crop(tmp_path):
    crops = tmp_path / "crops"
    kp2d = tmp_path / "poses_2d"
    write_crop(crops, "0001")
    write_crop(crops, "0002")
    write_keypoints(kp2d, "0001", eye_separation=5.0)    # near profile
    write_keypoints(kp2d, "0002", eye_separation=40.0)   # head-on

    picked = krv.rank_reference_crops(crops, kp2d, "000000", 1)
    assert picked == [crops / "head_0002.png"]


def test_ranking_falls_back_to_sorted_order_without_keypoints(tmp_path):
    crops = tmp_path / "crops"
    write_crop(crops, "0002")
    write_crop(crops, "0001")

    picked = krv.rank_reference_crops(crops, None, "000000", 2)
    assert [p.name for p in picked] == ["head_0001.png", "head_0002.png"]


def test_ranking_returns_nothing_when_no_crops_exist(tmp_path):
    crops = tmp_path / "crops"
    crops.mkdir()
    assert krv.rank_reference_crops(crops, None, "000000", 2) == []


def test_low_confidence_detections_do_not_win_over_available_crops(tmp_path):
    # If every detection is below threshold the ranking is empty, and the caller
    # should still get crops rather than silently losing identity anchoring.
    crops = tmp_path / "crops"
    kp2d = tmp_path / "poses_2d"
    write_crop(crops, "0001")
    write_keypoints(kp2d, "0001", eye_separation=30.0, score=0.05)

    picked = krv.rank_reference_crops(crops, kp2d, "000000", 2)
    assert picked == [crops / "head_0001.png"]
