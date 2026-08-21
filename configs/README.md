<!--
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
-->

# Rig configs

`run_unified_pipeline.py --config <file>.json` loads per-rig defaults for the
flags that are about the machine/setup rather than a specific capture --
`--trainer_repo`, `SAPIENS_CHECKPOINT_ROOT`,
and the HLOC feature-extraction settings. Explicit CLI flags always win over the
config. The config only fills in what you did not pass.

Per-run flags (`--video_dir`, `--calib_dir`, `--out_dir`, `--target_time`,
etc.) are not configurable here. Those change every run and belong on
the command line.

Copy [example_rig.json](example_rig.json), fill in your own paths, and pass
it on every run:

```bash
python3 scripts/run_unified_pipeline.py \
    --config configs/my_rig.json \
    --video_dir <movies_dir> \
    --calib_dir <calibration_pkls_dir> \
    --out_dir <out_dir> \
    --target_time <for example 2500ms>
```

The full list of configurable keys is `CONFIGURABLE_DEFAULTS` in
`scripts/run_unified_pipeline.py`. Passing an unrecognized key is an
error, not a silent no-op.

`multiframe_sfm_script` is configurable but deliberately left out of
[example_rig.json](example_rig.json): its built-in default
(`scripts/multiframe_sfm.py`, resolved relative to the script's own
location) already works regardless of your current directory. Only set it
in your own config if you are testing local changes to a different copy,
and use an absolute path if you do, since a relative one is resolved
against wherever you happen to run the orchestrator from, not the repo root.
