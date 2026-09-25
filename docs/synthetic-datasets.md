<!--
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
-->

# Synthetic Datasets: Blender to `.sogst`

A second content pipeline that starts from a Blender scene instead of GoPro
footage. It does two jobs:

1. **Produce splats of any character and animation.** Author in Daz or
   Character Creator, normalise the scene once, then run the pipeline. No
   step is specific to one subject.
2. **Compare camera configurations cheaply.** Change one JSON file, run
   again, read the scores side by side. The subject, lighting, and animation
   are identical between runs, so a change in the numbers is a change in the
   cameras.

The synthetic path joins the existing pipeline at the same point a real
capture does. `render_blender_rig.py` writes the flipbook layout that
`build_flipbook_4dgs_dataset.py` already consumes, and the training stage is
the same function the capture pipeline calls.

## A New Character, Start to Finish

```bash
# 1. Normalise the scene once per character. Packs textures, sorts subject
#    and background collections, measures the subject, writes a manifest.
python3 scripts/prepare_blender_scene.py \
    --input ~/assets/character.fbx \
    --textures ~/assets/character.fbm \
    --out ~/assets/character_packed.blend --fps 24

# 2. Render, build the dataset, train, score.
python3 scripts/run_synthetic_pipeline.py \
    --blend ~/assets/character_packed.blend \
    --rig_spec configs/rigs/ring16.json \
    --out_dir ~/runs/character_ring16 \
    --frame_start 100 --frame_count 48 --samples 128
```

The output is `splat_4d.sogst` in the run directory, plus `eval_4d.json`
with the scores and `experiment.json` recording what produced them.

Nothing above names a particular subject. A rig spec written against subject
proportions frames a short character and a tall one the same way, which is
what makes one spec reusable across a library of characters.

## Scene Preparation

`prepare_blender_scene.py` exists because authoring exports arrive broken in
predictable ways.

**Texture paths.** An export carries absolute paths from the machine that
made it. The reference Ariana blend had 146 of its 189 images pointing at a
Windows temp directory. The script repairs paths in three passes, weakest
last: make paths absolute, Blender's exact-filename search, then a
stem-based search that tolerates a swapped extension or a `.001` duplicate
suffix. That last pass matters more than it sounds: the same export stores
`Eyelash1_Transparency_Opacity.jpg` while the blend asks for the `.png`, and
stores `Std_Skin_Head_Flow Pack.exr` while the blend asks for
`Std_Skin_Head.001_Flow Pack.exr`. Every fuzzy match is logged and recorded
in the manifest. An unresolved image stops the run, because an unpacked
texture renders as plain grey with no error at all.

**Collections.** Renderable objects are sorted into `cumuli_subject` and
`cumuli_background`, each object in exactly one. The subject is detected
from the armature: every mesh it deforms, plus anything parented beneath.
That split is what lets the renderer isolate the subject for its matte and
the backdrop for its plates.

**The manifest.** `scene_manifest.json` records the frame rate, the
armature actions with their ranges, and the subject's world bounding box
sampled across the animation. Rig specs and the hull carve both read it.

## Rig Specs

A spec is a small JSON file under `configs/rigs/`.

```json
{
  "name": "ring16",
  "layout": "rings",
  "target": "subject_center",
  "rings": [
    {"count": 8, "radius": {"subject_heights": 1.7}, "height": {"subject_fraction": 0.55}},
    {"count": 8, "radius": {"subject_heights": 1.7}, "height": {"subject_fraction": 1.15},
     "azimuth_offset_deg": 22.5}
  ],
  "resolution": [1920, 1680],
  "camera_model": "PINHOLE",
  "intrinsics": {"lens_mm": 32, "sensor_width_mm": 36},
  "eval": {"count": 4},
  "lights": {"count": 12, "power_w": 150}
}
```

**Layouts.** `rings` takes a list of ring entries. `cage` is the truncated
spherical cage ported from `camera_cage.py`: azimuth columns by elevation
rows, with the per-row stagger. `explicit` places cameras by hand.

**Ring entries** take `count` and `radius`, plus:

| Key | Effect |
|---|---|
| `height` | one altitude for the whole ring |
| `heights` | a cycle of altitudes, camera `i` taking `heights[i % len]`, staggering elevation while azimuths stay even |
| `azimuth_offset_deg` | azimuth of the first camera |
| `azimuth_span_deg` | makes the ring an arc of this width, cameras spread inclusive of both ends |
| `azimuth_centre_deg` | centres an arc on this azimuth; only valid with a span |

A ring of `count: 1` is a single camera at its offset, which is how a lone
rear camera is written.

## The Rigs Provided

| Spec | Cameras | Shape |
|---|---|---|
| `ring12` | 12 | One ring at chest height, thirty degrees apart |
| `ring12_staggered` | 12 | The same azimuths, cycling three elevations |
| `arc11_plus_rear` | 12 | An eleven-camera arc across the front, plus one behind |
| `ring16` | 16 | Two staggered eight-camera rings, the March reference |
| `cage12x4` | 48 | Truncated spherical cage, twelve columns by four rows |

`ring12` and `ring12_staggered` are a matched pair: same count, same
azimuths, differing only in elevation, so a gap between them measures what
elevation diversity alone buys. `arc11_plus_rear` has the same twelve
cameras concentrated in front, and asks what a single rear view is worth.

Its `azimuth_centre_deg` is the world azimuth the subject FACES, and must
be set per scene. The value of -82 is measured for the Ariana scene from
the armature: take the vector from each right bone to its left partner,
cross it with world up, and average. That agreed to within 5.5 degrees
across 42 bone pairs and moved by 2 degrees over the whole clip. A render
confirms it: the arc sweeps left profile to right profile through a full
frontal view, and the lone rear camera sees the back of the head.

Every spec here carries an **identical eval ring**: four cameras at 18.5
degrees offset, 0.85 of subject height. Identical eval views are what make
two runs comparable at all. The particular offset was chosen by sweeping
it against every rig in this directory and keeping the value whose nearest
training camera is 10 to 15 degrees away in all of them. Without that, an
eval view sitting three degrees from one rig's camera and nineteen from
another's flatters the first for reasons that have nothing to do with its
geometry.

**Subject-relative values.** `{"subject_heights": k}` is a length, `k` times
the subject's height. `{"subject_fraction": k}` is a world height, `k` of
the way up the subject from its floor. `"subject_center"` and
`"subject_floor_center"` name points. Literal numbers still work.

**Camera model.** `PINHOLE` builds a perspective camera from a focal length.
`OPENCV_FISHEYE` needs a `calibration_pkl` and builds a panoramic camera
carrying that lens's real distortion, which is what makes the localization
path faithful to the physical rig. `OPENCV` renders undistorted, since
Cycles applies no radial distortion to a perspective camera. Each
camera's written calibration is therefore pinhole with zero distortion,
matching the pixels; the source coefficients are kept for provenance in
`rig_resolved.json` as `source_distortion_coefficients`.

**Eval cameras.** `eval` describes a ring that never trains and is scored
instead, four cameras by default, offset half a step from the first training
ring. Every rig is scored against the same views, which is what makes two
rigs comparable. Holding out a rig camera instead moves the test viewpoint
whenever the rig changes.

## Camera Intrinsics, and Why They Are Verified

Converting an OpenCV calibration to a Blender camera is the part of this
work most likely to be silently wrong. A wrong conversion still renders a
person in a room from consistent viewpoints; only the reconstruction
suffers. `blender_camera_intrinsics.py` therefore encodes conventions that
were **measured** against Blender 5.1.2, not read from documentation:

- The fisheye polynomial maps **radius to angle**, with radius in
  millimetres on the sensor. Doubling `sensor_width` halved every marker's
  pixel radius; doubling the resolution doubled it; `fisheye_lens` and
  `fisheye_fov` changed nothing.
- Blender's raster sits **half a pixel** from OpenCV's. With zero shift the
  optical axis lands at array index `(w/2 - 0.5, h/2 - 0.5)`.
- `shift_y` is normalised by **image height for panoramic cameras** and by
  the fitted sensor axis for perspective ones. `shift_x` uses width in both.

Check any rig before trusting a dataset built from it:

```bash
python3 scripts/verify_blender_intrinsics.py \
    --rig_spec configs/rigs/ring16_gopro_fisheye.json \
    --out_dir /tmp/intrinsics_check --max_error_px 0.5
```

It places markers by unprojecting target pixels through the calibration,
renders, and measures where they landed. Current results on this rig set:

| Camera model | Median error | Max error |
|---|---|---|
| Pinhole | 0.10 px | 0.28 px |
| Pinhole, off-centre principal point | 0.10 px | 0.28 px |
| GoPro fisheye | 0.14 px | 0.29 px |

The report also states what a naive pinhole reading of the same camera would
have predicted. On the fisheye rig that is 1841 px median, because a
panoramic camera ignores `lens` entirely. That mistake is not hypothetical:
it is what the March Ariana export wrote into `transforms_gt.json`, leaving
only the poses in that file usable.

## Metrics

`eval_4d.json` and `experiment.json` carry LPIPS, PSNR, and SSIM over the
eval ring. **LPIPS leads.** On a masked subject most of the frame is empty
background that every model renders perfectly, so PSNR is dominated by
pixels no camera configuration affects and compresses the differences worth
seeing. PSNR is still reported, because it is directly comparable with the
trainer's own eval numbers.

## Performance

Measured on an RTX 5090, 1920x1680, Cycles with OptiX denoising, on the
Ariana scene:

| Stage | Cost |
|---|---|
| Render, 32 samples | 3.0 to 3.6 s per image |
| 12 frames x 20 cameras | 240 images, 14 minutes |
| Dataset build, downscale 2 | under a minute |
| Train, 3000 iterations, 60k points | about 3 minutes |

## Rendering in Parallel

Rendering is CPU-bound, not GPU-bound. Measured on the Ariana scene at
1920x1680, the GPU sits at 9% mean utilisation and idles about 90% of the
time, while Blender holds roughly 11 cores rebuilding the depsgraph and BVH
for a 225k-vertex character every frame. Splitting a clip across concurrent
Blender instances therefore helps.

RAM is the binding constraint, not VRAM. Each instance peaks at 12.9 GB of
32.6 GB VRAM, so two fit easily there, but each also holds about 15 GB of
system RAM because the packed blend carries 188 textures. Two instances on a
60 GB machine spill about 8 GB into swap and deliver roughly **1.4 to 1.5x**,
not 2x. A third would thrash. Keeping textures external rather than packed
would cut per-instance RAM and raise that ceiling.

Split by frame with `--index_offset`, which is the base number each instance
writes its `frame_NNNN` directories from. Without it, a second instance
starting at a later scene frame writes `frame_0000` over the first
instance's:

```bash
python3 scripts/render_blender_rig.py ... --frame_start 100 --frame_count 61 \
    --index_offset 0  &
python3 scripts/render_blender_rig.py ... --frame_start 161 --frame_count 60 \
    --index_offset 61 &
wait
```

Each instance writes its own `rig_resolved.json`, so the last one to start
leaves a file describing only its shard. Regenerate it over the whole range
and then post-process once:

```bash
blender -b <scene.blend> --python scripts/render_rig_in_blender.py -- \
    --rig_spec <spec> --out_dir <run>/render --rig_only \
    --frame_start 100 --frame_count 121
python3 scripts/render_blender_rig.py ... --skip_render
```

The post step refuses to run when the metadata lists fewer frames than are
rendered on disk, because silently producing a 61-frame flipbook from a
121-frame render is worse than failing.

A 48-frame run of 16 rig cameras plus 4 eval cameras is 960 images, so
roughly an hour at 32 samples. Raise samples for a production asset and
lower them for a rig comparison, where the relative ordering is what
matters. `--skip_existing` resumes an interrupted render, and
`--start_from_stage` resumes the pipeline.

A 12-frame ground-truth smoke run scores LPIPS 0.0052 and 38.8 dB on its
four eval cameras after only 3000 iterations, against the trainer's own
38.9 dB on the same views. Those two numbers agreeing is the check worth
repeating: they come from different rasterizers reading different files,
so agreement means the bake and the eval both faithfully represent what
was trained.

## Long Clips: Train in Windows, Ship One File

A 5.00 s take at 24 fps from the flat 12-camera ring reconstructs better as
four ~1.25 s models than as one wide fit. The ladder that established this
ran 12, 31, 61 and 121 frames: LPIPS tracks duration almost perfectly
(r = +0.995) while PSNR does not see it at all (r = -0.343), which is the
clearest case yet for reading both.

| Configuration | LPIPS | PSNR dB | MB | Train time |
|---|---|---|---|---|
| One model over 5.00 s | 0.00899 | 39.21 | 27.4 | 1.0x |
| Four windows, scored separately | 0.00710 | 39.50 | 41.1 | 1.9x |
| Denser: gradient thresholds halved | 0.00876 | 39.16 | 33.9 | 1.2x |
| Tighter initial sigma: `t_init_div` 500 | 0.00885 | 39.19 | 27.6 | 1.0x |

Windowing is the only lever that moved the number much. Raising the splat
budget or tightening the initial temporal sigma each bought under 3%.

Four models are not an asset, so `scripts/merge_sogst_segments.py` stitches
them back:

```bash
python3 scripts/merge_sogst_segments.py \
    --segment win_0/splat_4d.sogst 0.0 \
    --segment win_1/splat_4d.sogst 1.2916667 \
    --segment win_2/splat_4d.sogst 2.5416667 \
    --segment win_3/splat_4d.sogst 3.7916667 \
    --out clip.sogst --report_json stitch.json
```

Each offset is that window's start in global seconds: the source frame its
local frame 0 came from, divided by fps. The stitched file scores LPIPS
0.00778 and PSNR 39.35 dB, so it keeps roughly three fifths of windowing's
gain over the single fit while staying one file.

### Two Ways the Stitch Goes Wrong Silently

**Dropping the spherical harmonics.** `decode_sogst_fields` returns the
higher-order coefficients as one `f_rest` block of `[N, 45]`, while the
packer's own PLY path uses `f_rest_0` through `f_rest_44`. Looking for the
split names finds nothing, writes a flat-shaded model, and nothing warns:
the first stitch scored 0.0186. The merger now refuses to write a model
whose inputs carried harmonics it lost.

**Letting the temporal tails through.** A splat renders wherever its
temporal Gaussian is non-zero, and its position is `xyz + v * (t -
t_center)`, so a splat evaluated seconds outside the window it trained on
extrapolates across the room. Alpha out there is small, but a faint splat in
the wrong place is a streak, and a perceptual metric punishes structure.
Partitioning on `t_center` alone does not stop it, because the criterion is
where a splat is centred rather than where it still contributes. Measured on
this clip, the ungated stitch scored 0.0164, four times worse than the
windows over the first window's span alone.

The fix caps each splat's `t_sigma` at `max(distance to the nearest seam,
--tail_floor) / --tail_k`. A splat deep inside its slot keeps a long life; a
splat near a seam is capped hard and dims into the seam while its neighbour
dims out of it, which is the crossfade a seam wants anyway. Defaults `k=2`
and `floor=0.4` were measured; the parameter is flat between `k=1.5` and
`k=3`, all landing within 0.0003 LPIPS, so the setting matters far less than
having a gate at all.

Two smaller points. The archive stores `t_center` as an index into a
256-entry codebook fit over the whole file, so merging four windows spreads
those codes across four times the range and costs 0.005 s of time error
against 0.001 s for a window alone, displacing a splat by a median 3.9% of
its own size. The merger snaps every `t_center` onto that codebook and
slides `xyz` by `v * delta`, which leaves the position at every instant
exactly unchanged. Separately, what error survives is concentrated within
0.15 s of a seam, where adjacent windows disagree about geometry; views a
second away from any seam score 0.0076.

Overlapping the windows across a seam and ramping opacity through the
overlap beats cutting to it, so `fade` is the default. The two metrics
disagree about how wide the overlap should be, which is worth seeing:

| Seam treatment | LPIPS | PSNR dB |
|---|---|---|
| Hard cut | 0.00789 | 39.22 |
| Fade 0.15 s | 0.00782 | 39.26 |
| Fade 0.35 s | 0.00778 | 39.35 |
| Fade 0.60 s | 0.00784 | 39.43 |
| Fade 1.00 s | 0.00802 | 39.48 |

LPIPS bottoms out at 0.35 s while PSNR climbs all the way to 1.00 s. A wide
overlap blurs the two windows together across the seam, and blur is what
PSNR rewards and LPIPS punishes. The default takes the LPIPS side.

### Choosing Where to Cut

Uniform frame intervals are the fixed-GOP of video coding, so the obvious
question is whether content should choose the cuts instead. Three
measurements on the ring12 runs answer most of it, and the answer is more
specific than yes.

**A motion signal is already on disk.** Mean absolute luminance change
between consecutive training frames, masked, averaged over four cameras at
240x210, correlates +0.969 with armature joint speed from
`render/joints/frame_*.json` and +0.716 with the single wide model's
per-frame LPIPS. It needs no armature, so a real capture plans the same way
a synthetic run does. On the 5 s take its dynamic range is 10.4x and the
first half carries 81% of the motion.

**Quality follows window length, not window content.** Across nine runs the
fit is `LPIPS = 0.00640 + 2.307e-5 x frames` at R2 0.981. Adding a motion
term takes R2 to 0.9808 with a coefficient of -9.9e-05, under 2% of the
score across the whole range. The four uniform windows settle it: at a fixed
~30 frames their motion content varies 6.3x while their scores stay inside
0.00691 to 0.00721, and the busiest window scores best.

**Total cost is set by the window count alone.** Splats per window fit
`181k + 1150k x that window's share of clip motion` at R2 0.927, and motion
shares sum to one however the clip is cut, so the total is
`N x 181k + 1150k` whatever the boundaries. The per-window term is paid for
existing, not for what the window covers.

**Seam cost is the one thing placement changes.** On the stitched file the
LPIPS bump within +-3 frames of each seam was +13.4%, +7.3% and +3.6% over
the clip median, against motion at those seams at the 95th, 54th and 24th
percentile: three points, perfectly ordered, spanning 3.7x. The uniform
split put its first seam at the busiest instant in the clip.

So the interesting question is not where prediction breaks down, it is where
the cut is invisible. Two things are worth choosing, the window count and
the seam positions, and both fall out of a closed-form model:

```
LPIPS_pooled(N)   = 0.00640 + 2.307e-5 * (F / N)
LPIPS_stitched(N) = LPIPS_pooled(N) + seam cost per seam * (N - 1)
splats(N)         = N * 181k + 1150k
```

`scripts/plan_temporal_windows.py` computes the signal, fits or loads the
coefficients, and solves a dynamic program over the cuts:

```bash
python3 scripts/plan_temporal_windows.py --run ~/runs/ring12_5s --report
python3 scripts/plan_temporal_windows.py --run ~/runs/ring12_5s \
    --windows 4 --out_dir_template '~/runs/planned/win_{index}' \
    --out window_plan.json
```

`--report` prints the rate-distortion table over window counts, `--windows`
fixes a count, `--target_lpips` picks the cheapest count that reaches a
score, and `--boundaries` forces a split by hand, which is how a control arm
is run through the same code path. The plan feeds
`merge_sogst_segments.py --plan` directly, so the seams and time origins are
never retyped.

The coefficients are fitted, not hard-coded, and `--fit_from` recomputes them
from finished runs:

```bash
python3 scripts/plan_temporal_windows.py --run $R/ring12_5s \
    --fit_from $R/dur_12:0 $R/dur_24:0 $R/dur_48:0 $R/dur_96:0 \
    $R/ring12_5s:0 $R/win_0:0 $R/win_1:31 $R/win_2:61 $R/win_3:91
```

The `:offset` suffix is required for windows and cannot be inferred: a window
trained from symlinked frames records the parent run's `--frame_start`, so
every window of a clip claims the same start in its `experiment.json`. Seam
coefficients refit separately from a stitched model against its own windows,
through `--fit_seams_from`, `--fit_seams_pooled` and `--fit_seams_at`.

With the measured coefficients the program returns near-equal lengths with
the seams pulled onto quiet instants, which is a result rather than a rule.
Refit on material where motion does drive per-window quality and the same
program moves the boundaries; a unit test asserts exactly that, so the
reduction cannot quietly become a hard-coded equal split.

### What the Matrix Measured

Eight window plans over the same 5.00 s take, every window at 30,000
iterations from 200,000 initial points, every arm scored on the same 484 eval
views. Pooled is the windows scored separately; stitched is the single file.

| Arm | Lengths | Pooled | Stitched | PSNR dB | Splats | MB |
|---|---|---|---|---|---|---|
| G planner | 36/34/26/25 | 0.00710 | **0.00774** | 39.39 | 1.90M | 42.2 |
| A uniform | 31/30/30/30 | 0.00710 | 0.00778 | 39.35 | 1.87M | 41.3 |
| B seam-snapped | 43/30/26/22 | 0.00721 | 0.00780 | 39.40 | 1.93M | 41.4 |
| F seeded, half | 31/30/30/30 | 0.00709 | 0.00780 | 39.35 | 1.85M | 41.0 |
| H seeded, full | 31/30/30/30 | 0.00711 | 0.00780 | 39.36 | 1.85M | 41.0 |
| C anti-adaptive | 31/18/30/42 | 0.00716 | 0.00783 | 39.33 | 1.87M | 41.2 |
| D equal-motion | 22/14/18/67 | 0.00731 | 0.00794 | 39.34 | 1.91M | 42.0 |
| E planner, 3 | 57/36/28 | 0.00752 | 0.00802 | 39.39 | 1.71M | 38.6 |

**Content-adaptive cutting is worth about half a percent.** The planner's own
plan beats uniform by 0.5% LPIPS and 0.04 dB for 1.5% more splats. That is
real and reproducible, and it is small enough that uniform cutting remains a
perfectly good default. Cutting on content the intuitive way is actively
worse: arm D equalises difficulty per window, which is what the video-coding
analogy suggests, and it is the worst four-window arm at +2.0%.

**The length law survived a much harder test.** Refitting on the sixteen
windows of arms A through D alone, spanning 14 to 67 frames and trained under
four different splits, gives `LPIPS = 0.00643 + 2.173e-5 x frames` at
r = +0.839, against `0.00640 + 2.307e-5` from the original nine runs. Two
independent datasets, the same law.

**Cost invariance held exactly.** The four arms with four windows landed
within 3% of each other on splat count (1.87M, 1.93M, 1.87M, 1.91M) despite
lengths ranging from 14 to 67 frames. Where the boundaries fall does not
change what the clip costs; how many there are does.

**The handover penalty is a clean function of motion.** Measured in absolute
LPIPS as the stitched score over a seam's +-3 frames minus the same frames
scored by the separate window models, so it isolates the handover from how
good those windows were, seventeen seams over six arms fit

```
penalty = 0.00035 + 0.00057 * motion at the seam        R2 0.914
```

with a correlation of +0.956. The length of the incoming window adds nothing
(R2 0.001 on its own). Scaling that to the clip mean recovers the seam
coefficients the planner already ships, so no refit was needed. Beware
measuring bumps against a file's own median instead: that mixes the handover
with the arm's overall quality and produced a spurious dependence on window
length before this control was applied.

**Fewer windows is a poor trade here.** Three windows cost 3.0% LPIPS for
8.6% fewer splats and 6.6% less storage. The closed-form optimum with the
refit slope is N = 3.3, so three and four straddle it and four wins.

**Seeding works and saturates.** Starting each window from its predecessor
saves 1.3% of the splat budget at `--seed_fraction 0.5` and 1.4% at 1.0, with
stitched quality unchanged at 0.00780 either way. The mechanism is real, the
dose response is flat, and at this scale it does not pay for the extra stage.
A single-subject masked capture has little static structure to amortise,
which the near-static splat measurement above already suggested.

### Seeding a Window From the One Before It

`scripts/seed_window_init.py` starts a window from its predecessor's trained
model rather than from the visual hull alone. It is the idea behind ATGS
(SIGGRAPH 2026), which organises Gaussians around time-conditioned anchors
so no primitive has to track long-range motion, in the form this trainer
already accepts.

The obvious route is a trap worth writing down. `train_scratch.py` restores
`first_iter` from the checkpoint (line 124) while densification is gated on
`iteration < densify_until_iter` (line 354), so resuming a 30k checkpoint
lands past the densification window: the model inherits a full splat set,
never re-densifies, and saturates the cap immediately.

Seeding the point cloud avoids all of it. `readNerfSyntheticInfo` reads
`points3d.ply` and, when it holds more points than `--num_pts`, randomly
subsamples to that budget (`scene/dataset_readers.py:311-330`). A cloud drawn
from the previous window's splats, advanced to the handover instant by
`xyz + v * (t - t_center)` and sampled by their opacity there, starts the
next window sparse but structurally correct, with the schedule running from
iteration 0. The cloud is half seeded and half hull by default, because the
hull is coarse but is carved from this window's own masks and carries a time
per point spanning the window, while the seeded half is accurate only at the
instant it was evaluated. Keep the total above `--num_pts`: the trainer only
subsamples clouds longer than its budget.

`scripts/run_window_plan.py` runs a whole plan, symlinking each window's
frames out of the master run so nothing is re-rendered, seeding when asked,
then merging and scoring. Extra flags for every pipeline call, or for the
merge, go in as one string each, written with `=` so argparse does not read
the value as one of its own options:

```bash
python3 scripts/run_window_plan.py --plan window_plan.json \
    --master ~/runs/ring12_5s --blend ~/assets/ariana_packed.blend \
    --rig_spec configs/rigs/ring12.json --merge_out ~/runs/planned.sogst \
    --extra="--samples 64" --merge_args="--mode hard"
```

### What ATGS Does and Does Not Give Us

ATGS anchors carry a position, a 64-d feature, a velocity and a keyframe
index, query a static 128^3 x 64 spatial grid and per-segment temporal hash
grids, and an MLP decodes q temporal Gaussians at the queried time; only
anchors within +-3 keyframes of the query are active.

It does not answer the keyframe question, it sidesteps it. Keyframes are
sampled at fixed intervals with no content criterion, and the ablation on VRU
Long shows quality rising monotonically with keyframe count: K=12 gives
23.76 dB, K=25 24.17, K=125 24.30, K=250 (one per frame) 24.42. That
independently reproduces the length law above. Its window-size ablation says
the same about gating as our own tail-gate sweep: W in {1,3,5,7} all score
~24.4 and removing windowing drops to 24.02, so having a gate matters and its
width does not.

Adopting it wholesale is out of scope. It decodes through a hash grid and an
MLP per frame, so nothing bakes to `.sogst` and a viewer would need a neural
decoder, and its storage is not clearly better: 110 MB for 300 frames is
0.37 MB per frame against 0.34 for the stitched file here, on scenes too
different to compare. Sharing also cannot be retrofitted to windows already
trained: only 4.2% of the busiest window is near-static (|v| < 0.02 m/s) and
those sets overlap the next window by 5 to 30%, because a masked moving human
has nothing to amortise.

Two practical notes for anyone trying the code. It is under the Inria
Gaussian-Splatting licence, research and evaluation only. And it pins
cudatoolkit 11.8 with torch 2.2.0, which cannot target an RTX 5090 at all:
compute capability 12.0 needs CUDA 12.8 or newer, so the shipped `env.yml`
produces an environment that does not run on this machine. Porting it means
the same treatment `setup_cumuli_env.sh` gives OMG4, plus sourcing
`diff-gaussian-rasterization` and `simple-knn` from the 3DGS repo (the clone
has no `submodules/` directory) and swapping mmcv for mmengine, which is used
only for `Config.fromfile`.


### ATGS Measured on Our Data

The port runs. Trained on the 5 s take at 30,000 iterations, the same budget
every window in the matrix got, and scored on the same 484 eval views with
`eval_render.py`'s exact metrics:

| Model | LPIPS | PSNR dB | Size | Iterations |
|---|---|---|---|---|
| Arm G, four planned windows | 0.00774 | 39.39 | 42 MB | 4 x 30k |
| Arm A, four uniform windows | 0.00778 | 39.35 | 41 MB | 4 x 30k |
| Single 5 s fit | 0.00899 | 39.21 | 27 MB | 30k |
| ATGS | 0.00970 | 39.61 | 2410 MB | 30k |

The metrics disagree, which is the whole reason both are reported. ATGS wins
PSNR by 0.22 dB over our best and loses LPIPS by 25%. Against the comparison
it actually deserves, the single wide fit at the same 30k iterations, it wins
PSNR by 0.40 dB and still loses LPIPS.

Three caveats before reading too much into either number. ATGS writes JPEG
renders while the eval ground truth is PNG, and a JPEG round trip of the
ground truth alone scores 0.00149 LPIPS at 51.3 dB, so a meaningful part of
its LPIPS gap is compression rather than reconstruction; PSNR is barely
affected at that level. The 2410 MB excludes optimizer state and is dominated
by the hash encoder at 1777 MB and the voxel grid at 537 MB, against 92 MB
for the anchors themselves. And that size is far off the 110 MB per 300
frames the paper reports, because the init cloud here gives every point its
own keyframe, K = 121, which is the setting their own ablation says is best
for quality and which nobody tuned for storage.

None of which changes the adoption question: there is no path from a hash
grid and an MLP to a `.sogst`, so this is a quality reference, not a
candidate.

## A Trap in the Eval Numbers

`eval_render.py --downscale` defaults to 2, because it was written for
n3v-style datasets whose transforms carry FULL-resolution intrinsics beside
half-resolution ground truth. Anything from `build_flipbook_4dgs_dataset.py`
is different: its transforms already carry output-resolution intrinsics, so
it needs `--downscale 1`.

Getting this wrong renders every view at half scale against correctly sized
ground truth and costs about 22 dB, which reads as a badly trained model
rather than a measurement error. Both orchestrators now pass `--downscale 1`
explicitly, and `eval_render.py` refuses the mismatch instead of scoring it,
so the failure is loud. If you invoke `eval_render.py` by hand on a
flipbook-built dataset, pass it yourself.

## Files

| Path | Role |
|---|---|
| `scripts/prepare_blender_scene.py` | Normalise a character scene, write the manifest |
| `scripts/camera_rig_spec.py` | Resolve a rig spec into cameras (pure, no bpy) |
| `scripts/blender_camera_intrinsics.py` | Calibration to Blender camera and back (pure) |
| `scripts/render_rig_in_blender.py` | The render loop, inside Blender |
| `scripts/render_blender_rig.py` | Drives the render, lays out the flipbook |
| `scripts/verify_blender_intrinsics.py` | Projection gate for a rig's camera model |
| `scripts/run_synthetic_pipeline.py` | The orchestrator |
| `scripts/merge_sogst_segments.py` | Stitch windowed models into one clip |
| `scripts/plan_temporal_windows.py` | Choose where to cut a clip into windows |
| `scripts/seed_window_init.py` | Seed a window from the previous window's model |
| `scripts/run_window_plan.py` | Train a plan, stitch it, score it |
| `configs/rigs/` | Rig specs and the reference GoPro calibration |
