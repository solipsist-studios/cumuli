# Render-and-repair 4D training

End-to-end recipe for turning a per-frame splat sequence into a single 4D
asset whose novel views hold up all the way around the subject. This is the
branch that produced the 90-frame `tatum_jump` deliverable from a 12-camera
GoPro rig; substitute your own paths.

Read `pipeline.md` first -- everything here starts from a completed per-frame
sequence (`render_frame_sequence.py`), and reuses that walkthrough's masks,
poses and keypoints unchanged.

## The problem this solves

A 12-camera rig covers barely half the azimuth circle. Train a 4D model on
those views alone and every direction the rig never saw fills with floaters
and smear -- the model is free to put anything there, because nothing
penalizes it.

Adding *generated* views is the obvious fix and the obvious trap. Feeding a
diffusion model's own idea of the subject into training was measured to make
true novel-view reconstruction **worse** (18.0 dB -> 13.3 dB on a genuinely
held-out camera), because generated pixels are a different look -- colour,
exposure, micro-detail -- from the real ones, and averaging the two pulls the
splat's appearance off. Masking away the obviously-hallucinated regions did
not rescue it; the inconsistency is in the kept body pixels too.

What does work is not asking a model to invent the subject at all:

1. Train an ordinary per-frame splat from real views only. It is already
   correct -- just soft, and janky where the rig has gaps.
2. **Render** it from the directions you want supervision for.
3. **Repair** each render at low denoise: a pass strong enough to restore
   facial detail and smooth gap artifacts, weak enough that pose, framing,
   silhouette and colour all stay where the render put them.
4. Train the 4D model on real + repaired views.

Step 3 is doing appearance restoration on an image that is already
structurally right, not synthesis. That is what keeps it consistent with the
real cameras.

## Prerequisites

- A completed per-frame run: one trained splat per frame, plus that
  sequence's cleaned masks and 2D keypoints (`pipeline.md`).
- Each frame's splat mask-filtered (`filter_splat_by_masks.py`) --
  **not optional here**. The orbit renderer derives its look-at target from
  the splat itself, and un-filtered background junk drags that target
  off-subject; on one real run most synthetic views came out with the subject
  out of frame or occluded entirely.
- A running ComfyUI with Flux.2 Klein 9B, its Qwen text encoder and VAE.
- A 4D trainer (the OMG4 / 4D-gaussian-splatting family) for the refit step;
  it is not vendored here.

## 1. Render the orbits

```bash
conda activate diffuman4d
python3 scripts/render_orbit_views.py \
    --splat_ply ~/run/frame_0000/exports/frame_0000_30000_maskfilt.ply \
    --transforms ~/run/transforms.json \
    --out_dir ~/render_repair/orbit/frame_0000 \
    --kp2d_dir ~/run/frame_0000/poses_2d \
    --subject_anchor_ply ~/run/frame_0000/poses_pcd_fullres/000000.ply \
    --subject_radius 3.0
# -> orbit/frame_0000/{body_000..035,head_000..007}.png + cameras.json
```

What it does: renders 3 elevation rows x 12 azimuths around the subject plus
8 zoomed head poses, writing RGBA (alpha straight from rasterized coverage)
and a `cameras.json` recording every pose so the dataset builder can place
them later.

Production choices that matter:

- **Full 360 degree azimuth, densified toward the real cameras.** An earlier
  version swept uniformly across only the rig's ~217 degree arc, centred on
  that arc's midpoint -- and since the rig's coverage is skewed to one side of
  the subject's path, most poses landed on its back and side. Sampling the
  gaps *between* real cameras instead puts more views where the rig is dense
  (in practice, where the subject faces) and still closes the circle.
- **`--merge_deg 3.0`.** The 12 cameras are physically 6 stereo pairs, each
  pair 0.3-0.8 degrees apart. Without merging, every pair's near-zero internal
  gap subdivides into visually identical duplicate renders.
- **`--max_reach_deg 20.0`.** A sample at the exact middle of a wide gap sits
  at the point farthest from any real camera in either direction -- the
  worst-supported position possible -- and renders visibly incoherent. Capping
  the reach means wide gaps simply get less coverage instead of one
  maximally-bad view.
- **`--elev_rows 3`.** Collapsing to a single row to spend the whole budget on
  azimuth was tried and reverted the same day: it lost the low/mid/high
  viewing-angle diversity the 4D fit needs.
- **Head poses are rendered natively at `--res`, not cropped** from a body
  render. Cropping an already-1536px image adds no information; a narrower
  frustum at the same output resolution puts real rasterized detail on the
  face.

## 2. Repair the renders

```bash
python3 scripts/klein_repair_views.py \
    --orbit_dir ~/render_repair/orbit/frame_0000 \
    --out_dir ~/render_repair/repaired/frame_0000 \
    --comfy_input_dir /path/to/ComfyUI/input \
    --comfy_output_dir /path/to/ComfyUI/output \
    --prompt "A high resolution photograph of a young child mid-jump, dark navy blazer, sharp focus, realistic skin texture." \
    --reference_crops_dir ~/run/frame_0000/crops \
    --kp2d_dir ~/run/frame_0000/poses_2d
# -> repaired/frame_0000/{body,head}_NNN.png + cameras.json
```

What it does: submits each render to ComfyUI as a whole-image img2img pass
anchored on two real photos of the subject, then pairs the returned RGB with
the original render's alpha after a floater cleanup.

Production choices that matter:

- **Whole-image at `--denoise 0.15`, not a face-crop detailer.** The face-crop
  approach (DetailerForEach at denoise 0.3) was the first version and lost on
  both counts: it drifted identity even with reference anchoring, and
  everything outside the detected face box went untouched, so coverage-gap
  jank always survived. A two-pass variant (whole image then face crop) was
  also worse on identity than the whole-image pass alone.
- **Per-frame identity anchors.** Two fixed portraits for the whole clip
  mismatch whatever expression and head angle the subject actually had at each
  instant. `--reference_crops_dir` picks that frame's two most head-on real
  head crops instead.
- **One fixed `--seed` for every view of every frame.** The orbit is a
  continuous sweep; a varying seed flickers expression and detail between
  neighbouring poses, which the 4D fit then averages into mush.
- **No upscale step.** An earlier version ran 4x ESRGAN to 6144px then
  downscaled to 2048 -- two downscales after an upscale, with the information
  ceiling still at the model's native 1536 output. Thousands of GPU calls for
  no measurable gain.

Both stages skip existing outputs, so an interrupted run resumes.

## 1+2 together, across a whole sequence

The single-frame scripts above are for inspecting one frame. For a real
sequence use the driver, which imports both stages and runs them in one
process -- so gsplat and CUDA start once rather than per frame, and ComfyUI's
models stay warm between calls:

```bash
python3 scripts/render_and_repair_sequence.py \
    --sequence_root ~/run \
    --transforms ~/run/transforms.json \
    --orbit_root ~/render_repair/orbit \
    --repaired_root ~/render_repair/repaired \
    --comfy_input_dir /path/to/ComfyUI/input \
    --comfy_output_dir /path/to/ComfyUI/output \
    --prompt "..." \
    --free_vram_when_done
```

Each frame is rendered *and* repaired before the next starts, so an interrupted
run leaves a prefix of complete frames rather than every frame half done.
`--free_vram_when_done` unloads ComfyUI's models before you hand the GPU to a
trainer.

## 3. Build the refit dataset, one window at a time

```bash
python3 scripts/build_refit_dataset.py \
    --real_dataset ~/dataset_4dgs \
    --repaired_root ~/render_repair/repaired \
    --out_dir ~/dataset_refit_w1 \
    --time_min 0.0 --time_max 0.467134
```

What it does: merges the real views in the window with every repaired view at
its render pose and frame timestamp, writing the `transforms_train.json` the
4D trainer reads.

Production choices that matter:

- **Windows, not one wide fit.** A single model spread over the whole clip
  splits its splat budget across every instant and visibly under-resolves fast
  motion. One model per window, each with an undiluted budget, then stitched
  (step 5). On the 3-second reference clip, six windows beat three.
- **Head views are off by default, and that default is load-bearing.** See
  "Head views poison shared-window refits" below.
- **Real images are linked under a `.png` name even when they are `.jpg`.**
  The Blender-format loader applies one global extension to every frame in
  `transforms_train.json`, and PIL sniffs content rather than trusting the
  name -- uniform `.png` naming is what lets real `.jpg` frames and `.png`
  synthetics coexist in one dataset.

Then train one model per window with your 4D trainer, pointed at each
`--out_dir`.

## 4. Filter and export each window

```bash
conda activate omg4
python3 scripts/filter_4d_checkpoint_by_masks.py \
    --checkpoint output/refit_w1/chkpnt30000.pth \
    --transforms ~/dataset_refit_w1/transforms_train.json \
    --sequence_root ~/run \
    --out_npy ~/keep_w1.npy \
    --fps 29.97

python3 scripts/xz_to_omg4.py \
    --input output/refit_w1/chkpnt30000.pth \
    --output ~/segments/window_1.omg4 \
    --time_min 0.0 --time_max 0.467134 --fps 29.97 --sh_clamp 1.5 \
    --extra_keep_mask ~/keep_w1.npy
```

`filter_4d_checkpoint_by_masks.py` is the temporal counterpart of
`filter_splat_by_masks.py`: each Gaussian is tested against the masks of the
frame its own `t_center` falls on, because projecting a moving subject's
Gaussians into every frame's mask would fail everything that moves.

## 5. Stitch the windows

Write a manifest naming each window and the `t_center` range it owns.
`configs/example_segments.json` is a template:

```bash
python3 scripts/merge_omg4_segments.py \
    --manifest configs/example_segments.json \
    --segment_dir ~/segments \
    --mode hard \
    --out ~/clip_stitched.omg4
```

What it does: partitions Gaussians by temporal centre at the seams, clamps
each one's temporal sigma so it cannot stay alive outside its window, and
concatenates.

Production choices that matter:

- **The sigma clamp is not optional.** Each window's wide-sigma Gaussians stay
  faintly alive far outside their slot on a merged timeline and drift along
  their velocity vectors, rendering as glinting particles flying off the
  subject. The segment files never showed this alone because each header
  clamped playback to its own window.
- **`--seam_tail_ext 0.25`.** Clamping tails to die exactly *at* the seam made
  the whole image dip through black there, because at the seam instant nothing
  from either side was still visible. Membership stays a hard cut; only the
  clamp interval widens.
- **`--mode hard` is the default for a reason.** `fade` cross-dissolves the
  seams but keeps both sides' Gaussians through each fade zone, so the file
  grows. Reach for it only if a seam is visible.
- **`--segment_dir`** re-points one manifest at a staged set of variant
  exports (pruned, compressed), so comparing variants needs no second
  manifest.

## Head views poison shared-window refits

Close-up head views help per-frame fits and hurt 4D refits. A five-variant
bisection on one window, identical in everything but the view set:

| refit dataset | bright-opaque artifact splats |
| --- | --- |
| real body + repaired body views | **0.008%** |
| real body + older synthetic set | 0.017% |
| + synthetic head zooms | 0.065% |
| + real head crops | 0.134% |

Mechanism: the head crops are binary-masked, and on a backlit capture that
mask cuts through a motion-blurred boundary with blown window light optically
mixed into the pixels *inside* it, heavily supervised at high face resolution.
A static per-frame fit can just paint that blur. A shared-window 4D fit cannot
reconcile a blur that changes shape every frame, and compromises with bright
semi-opaque fog around the face.

So `build_refit_dataset.py` excludes both kinds of head view unless asked. The
crops are still worth rendering and still worth using for per-frame training
(`build_densification_crops.py`) -- it is specifically the shared-window refit
that cannot digest them. On a capture without blown backlighting, try
`--include_head_views` and measure.

## Known limits

- **The repair model is a subject prior.** Klein holds a child's face well at
  low denoise; a different subject may need a different model, and the prompt
  is a real knob (`--prompt`), not decoration.
- **Alpha is never round-tripped through the repair model** -- it returns RGB
  only, so silhouettes come from the render. A render with a bad silhouette
  stays bad; fix it upstream in the mask filter.
- **Coverage gaps are smoothed, not filled.** `--max_reach_deg` deliberately
  leaves the deep middle of a wide gap unrendered. Genuinely closing it needs
  another camera, not another sampling trick.
