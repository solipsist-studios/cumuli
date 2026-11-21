# Vendored Sapiens2 Pose Demo

`predict_keypoints_2d.py` runs `vis_pose.py` in a subprocess to predict the
308 goliath keypoints with a Sapiens2 pose checkpoint.

## License

These two files are Sapiens Materials under the **Sapiens2 License**, a copy
of which is in [LICENSE.md](LICENSE.md). They are not covered by this
repository's PolyForm Noncommercial license. The Sapiens2 License permits
redistribution of modified copies only under its own terms and only with a
copy of the agreement, which is why `LICENSE.md` sits next to them. Its
acceptable-use terms (see the repository README) apply to anyone running
this code.

## Provenance

The originals are `sapiens/pose/tools/vis/vis_pose.py` and
`sapiens/pose/tools/vis/pose_render_utils.py` from
[facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2) at
`7e5bae8`, the commit `environment.yml` installs. Solipsist Studios modified
them in the `solipsist-studios/Diffuman4D` fork
(`scripts/preprocess/sapiens/2/demo/`, last at `3656cd4`), and they moved here
unchanged apart from the license headers when cumuli dropped that submodule.

## Changes From the Originals

`vis_pose.py`:

- Detects people with DETR (`facebook/detr-resnet-101-dc5`) from a local
  snapshot directory, and filters and ranks the boxes against an optional
  foreground mask (`--fmasks-dir`), always adding the mask's own bounding
  box as a candidate.
- Keeps at most `--max-subjects` instances, chosen by pose quality.
- Zeroes the score of implausible distal keypoints with a bilateral
  bone-length check (`--max-bilateral-ratio`) and an optional MAD check
  (`--max-bone-sigma`).
- Writes one combined `<name>_predictions.json` per run. Skeleton overlay
  images are opt-in (`--save-vis`).
- Accepts a manifest file as `--input` (one `image<TAB>mask` per line), so a
  whole frame window runs through one model load.

`pose_render_utils.py`: skips drawing links longer than a fraction of the
image diagonal (`max_bone_frac`).
