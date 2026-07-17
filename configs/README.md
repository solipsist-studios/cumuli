# Rig configs

`run_unified_pipeline.py --config <file>.json` loads per-rig defaults for the
flags that are about the machine/setup rather than a specific capture --
conda env names, `--brush_app`, `--display`, `SAPIENS_CHECKPOINT_ROOT`, and
the HLOC feature-extraction settings. Explicit CLI flags always win over the
config; the config only fills in what you didn't pass.

Per-run flags (`--video_dir`, `--calib_dir`, `--out_dir`, `--target_time`,
etc.) aren't configurable here -- those change every run and belong on the
command line.

Copy [example_rig.json](example_rig.json), fill in your own paths, and pass
it on every run:

```bash
python3 scripts/run_unified_pipeline.py \
    --config configs/my_rig.json \
    --video_dir <movies_dir> \
    --calib_dir <calibration_pkls_dir> \
    --out_dir <out_dir> \
    --target_time <e.g. 2500ms>
```

The full list of configurable keys is `CONFIGURABLE_DEFAULTS` in
`scripts/run_unified_pipeline.py` -- passing an unrecognized key is an error,
not a silent no-op.

`multiframe_sfm_script` is configurable but deliberately left out of
[example_rig.json](example_rig.json): its built-in default
(`scripts/multiframe_sfm.py`, resolved relative to the script's own
location) already works regardless of your current directory. Only set it
in your own config if you're testing local changes to a different copy --
and use an absolute path if you do, since a relative one is resolved
against wherever you happen to run the orchestrator from, not the repo root.
