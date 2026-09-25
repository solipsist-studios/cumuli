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

```bash
python3 scripts/compare_experiments.py ~/runs/ariana_*
```

Rows sort by LPIPS, best first. Runs whose PSNR ranking contradicts their
LPIPS ranking are called out rather than averaged away, and runs that scored
a different number of views or frames are flagged rather than ranked against
each other.

## Measuring Localization Error

`--poses gt` trains on Blender's own camera poses, which isolates what the
rig geometry alone costs. `--poses hloc` or `--poses refined` instead runs
the real pose chain on composited frames, scores it against the truth, and
trains on the estimate:

```bash
python3 scripts/run_synthetic_pipeline.py ... --poses refined
```

The `localize` stage runs `run_hloc.py`, Sapiens keypoints over several
instants, and `run_pose_refinement.py`, then `score_poses_vs_gt.py` reports
position error in metres, rotation error in degrees, the recovered metric
scale, and the reprojection error of known armature joints. Structure from
motion recovers geometry only up to a similarity, so the scorer fits the
best rotation, translation, and uniform scale first and reports what
remains.

Training then uses the same pixels as the ground-truth run and differs only
in the poses, so the gap between the two runs' scores is what the pose error
costs.

This path needs a backdrop: a subject alone on transparency gives feature
matching nothing static to work with. The render step writes background
plates and composites automatically whenever poses are estimated. One known
simplification is that composites carry no contact shadow from the subject
onto the backdrop.

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
python3 scripts/render_blender_rig.py ... --rig_only \
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

## A Trap in the Eval Numbers

`eval_render.py --downscale` used to default to 2, because it was written
for n3v-style datasets whose transforms carry FULL-resolution intrinsics
beside half-resolution ground truth. Anything from
`build_flipbook_4dgs_dataset.py` is different: its transforms already carry
output-resolution intrinsics, so it needs `--downscale 1`, which is now the
default. Pass `--downscale 2` only for an n3v-style dataset.

Getting this wrong renders every view at half scale against correctly sized
ground truth and costs about 22 dB, which reads as a badly trained model
rather than a measurement error. Both orchestrators also pass `--downscale
1` explicitly, and `eval_render.py` refuses the mismatch instead of scoring
it, so the failure is loud.

## Files

| Path | Role |
|---|---|
| `scripts/prepare_blender_scene.py` | Normalise a character scene, write the manifest |
| `scripts/camera_rig_spec.py` | Resolve a rig spec into cameras (pure, no bpy) |
| `scripts/blender_camera_intrinsics.py` | Calibration to Blender camera and back (pure) |
| `scripts/render_blender_rig.py` | Renders the rig inside Blender, then lays out the flipbook |
| `scripts/verify_blender_intrinsics.py` | Projection gate for a rig's camera model |
| `scripts/run_synthetic_pipeline.py` | The orchestrator |
| `scripts/score_poses_vs_gt.py` | Pose error against the rendering truth |
| `scripts/compare_experiments.py` | Tabulate runs, LPIPS first |
| `configs/rigs/` | Rig specs and the reference GoPro calibration |
