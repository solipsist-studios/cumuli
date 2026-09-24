# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Texture path repair for character exports.

The Blender half of this script needs Blender; the matching logic does not,
and it is the part that decides whether a subject renders with its skin and
hair or as untextured grey. An authoring export routinely disagrees with
its own .blend about a texture's extension or carries a duplicate suffix,
and Blender's own search matches filenames exactly, so it cannot repair
either case.
"""

import pytest

import prepare_blender_scene as prep


# ------------------------------------------------------------ normalising
@pytest.mark.parametrize("name,expected", [
    ("Skin_Diffuse.png", "skin_diffuse"),
    ("Skin_Diffuse.jpg", "skin_diffuse"),
    ("Std_Skin_Head.001_Flow Pack.exr", "std_skin_head_flow pack"),
    ("Hair.001.png", "hair"),
    ("Bar.002_Normal.jpg", "bar_normal"),
    ("C:/Users/x/AppData/Local/Temp/tmp/Brows_Opacity.png", "brows_opacity"),
    ("C:\\Users\\x\\Brows_Opacity.png", "brows_opacity"),
])
def test_stem_ignores_directory_extension_and_duplicate_suffix(name, expected):
    assert prep.normalise_stem(name) == expected


def test_a_version_number_that_is_not_a_duplicate_suffix_survives():
    """Only a three-digit group followed by a separator or the end is
    Blender's duplicate marker."""
    assert prep.normalise_stem("tex_v1.001a.png") == "tex_v1.001a"


def test_the_two_real_mismatches_from_the_reference_export_match():
    """Extension swapped one way, then the other, plus an embedded .001.
    All three appear in the March Ariana export."""
    assert (prep.normalise_stem("Eyelash1_Transparency_Opacity.png")
            == prep.normalise_stem("Eyelash1_Transparency_Opacity.jpg"))
    assert (prep.normalise_stem("Hair_Transparency_Opacity.jpg")
            == prep.normalise_stem("Hair_Transparency_Opacity.png"))
    assert (prep.normalise_stem("Std_Skin_Head.001_SSTM Pack.exr")
            == prep.normalise_stem("Std_Skin_Head_SSTM Pack.exr"))


# ---------------------------------------------------------------- index
def test_index_finds_images_recursively_and_ignores_other_files(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "Skin_Diffuse.jpg").write_bytes(b"")
    (tmp_path / "a" / "b" / "Hair_Opacity.png").write_bytes(b"")
    (tmp_path / "a" / "notes.txt").write_bytes(b"")
    index = prep.index_texture_dir(tmp_path)
    assert set(index) == {"skin_diffuse", "hair_opacity"}


def test_index_of_a_missing_directory_is_empty(tmp_path):
    assert prep.index_texture_dir(tmp_path / "nope") == {}


def test_several_files_can_share_a_stem(tmp_path):
    (tmp_path / "Skin.png").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "Skin.jpg").write_bytes(b"")
    assert len(prep.index_texture_dir(tmp_path)["skin"]) == 2


# ------------------------------------------------------------- selection
def test_the_same_extension_wins_when_several_files_match(tmp_path):
    from pathlib import Path

    candidates = [Path("/x/deep/nested/Skin.jpg"), Path("/x/Skin.png")]
    chosen = prep.pick_candidate(candidates, "C:/orig/Skin.png")
    assert chosen.suffix == ".png"


def test_the_shortest_path_wins_among_equal_extensions():
    from pathlib import Path

    candidates = [Path("/x/deep/nested/very/long/Skin.png"), Path("/x/Skin.png")]
    assert prep.pick_candidate(candidates, "Skin.png") == Path("/x/Skin.png")


def test_a_swapped_extension_is_chosen_when_nothing_matches_exactly():
    from pathlib import Path

    candidates = [Path("/x/Eyelash_Opacity.jpg")]
    chosen = prep.pick_candidate(candidates, "C:/t/Eyelash_Opacity.png")
    assert chosen.suffix == ".jpg"


# ------------------------------------------------------------ forwarding
def test_forwarded_args_are_rebuilt_from_the_namespace():
    from types import SimpleNamespace

    args = SimpleNamespace(
        input="a.fbx", out="b.blend", bbox_samples=8, textures="--out",
        subject_collection=None, background_collection=None, fps=24.0,
        bbox_range="1:100", allow_missing_textures=True,
        no_fuzzy_textures=False)
    out = prep.forward_args(args)
    assert out[out.index("--textures") + 1] == "--out"
    assert "--allow_missing_textures" in out
    assert "--no_fuzzy_textures" not in out
    assert out[out.index("--fps") + 1] == "24.0"


def test_optional_arguments_are_omitted_when_unset():
    from types import SimpleNamespace

    args = SimpleNamespace(
        input="a.blend", out="b.blend", bbox_samples=4, textures=None,
        subject_collection=None, background_collection=None, fps=None,
        bbox_range=None, allow_missing_textures=False, no_fuzzy_textures=True)
    out = prep.forward_args(args)
    assert "--textures" not in out and "--fps" not in out
    assert "--no_fuzzy_textures" in out
