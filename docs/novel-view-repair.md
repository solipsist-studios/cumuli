# Novel-view repair

Manufacturing the angular coverage a sparse rig never captured, by rendering the
trained 4D model from the directions it lacks and repairing those renders with a
video model. Read `pipeline.md` first: everything here starts from a trained
`.sogst` (or 4D interchange PLY) and that walkthrough's poses and masks.

## The problem, and the trap

A twelve-camera rig covers barely half the azimuth circle. Every direction it
never saw is unconstrained, so the fit puts floaters and smear there because
nothing penalises it.

Adding generated views is the obvious fix and the obvious trap. Two independent
measurements on this project say so:

- Feeding a diffusion model's own idea of the subject into training made
  held-out novel-view reconstruction **worse**, 18.0 dB to 13.3 dB. Masking away
  the obviously-hallucinated regions did not rescue it; the inconsistency is in
  the kept body pixels too.
- Asking a video model to generate the views *between* two real cameras, with
  both endpoints pinned to real photos, scored **8.5 dB below simply rendering
  the model** (21.23 vs 29.78 against a held-out camera). The output was not
  garbage: it was a clean, plausible subject whose silhouette, bulk and arm
  position were wrong, because the model interpolated along a path it imagined
  rather than the true camera arc.

What works is refusing to let the model invent the subject at all. Render from
the directions you want supervision for, repair those renders at low denoise,
and train on real plus repaired views. The render is already structurally right
and already carries the real cameras' colour and exposure; the repair only has
to restore detail.

## 1. Render the sweeps

```bash
python scripts/render_pair_sweep.py \
    --model out/splat_4d.sogst \
    --transforms out/transforms.json \
    --out_dir out/sweeps \
    --adjacent_pairs --frames 58-74 --res 1024
```

A sweep is a short arc from one real camera to the next that advances in time as
it turns. Production choices that matter:

- **Time advances with the camera.** A time-frozen sweep asks a video model to
  hold a subject perfectly still while the camera flies, the one thing video
  models are worst at. One capture frame per sweep frame means the motion in the
  clip is the subject's real motion, and every generated frame gets a distinct
  (pose, time). Sweep a pair in both directions for two synthetic poses per
  instant.
- **Both ends are pinned to real pixels.** Each swept frame shares one square
  frustum, but its centre is interpolated between the two real camera centres and
  equals a real centre exactly at the endpoints. Cameras sharing a centre are
  related by an exact homography, so the real photo warps into the sweep frustum
  losslessly and becomes the clip's first and last frame.
- **`--merge_deg 3.0`** skips pairs closer than three degrees. A rig built from
  stereo pairs puts its two halves under a degree apart, and sweeping between
  them generates a clip of near-identical frames.
- **Clip length is the frame count.** Match the video model's native length; Wan
  needs 4k+1 (17, 21, 33...).
- **Temporal evaluation matches `eval_render.py` exactly**, so a sweep frame and
  an eval render of the same model at the same instant agree.

`cameras.json` carries a per-frame `loss_weight`, peaking at the real-pinned
endpoints and falling to 1.0 at the arc's midpoint. That is Equation S2 of Hwang
et al., "4D Human-Scene Reconstruction from Low-Overlap Captures" (SIGGRAPH
2026). Weighting every generated frame equally spends as much of the fit's
attention on the least supported view as on the best.

## 2. Skeleton conditioning

```bash
python scripts/project_skeleton_conditioning.py \
    --sweep_dir out/sweeps/cam02_to_cam06 --out_dir out/skeletons/cam02_to_cam06 --draw
```

A render is right about appearance and wrong about geometry exactly where the
rig had no coverage. A skeleton is the opposite. Feeding the render as the video
model's base image and the skeleton as its control channel lets each supply what
the other lacks, without asking a model to invent the subject.

This re-solves no geometry: it projects the `poses_3d` points
`triangulate_and_project_keypoints.py` already produced through the sweep's
cameras, and wraps Diffuman4D's own `draw_skeleton.py` so the link topology and
palette match what that model was conditioned on.

Two details make a projected skeleton usable at unseen views. Every joint gets
its true camera-space depth, so limbs sort back-to-front instead of an arm behind
the torso being painted on top of it. And face keypoints fade as the sweep passes
behind the head, scaled by `(1 + cos)/2` against the camera, rather than being
drawn through the skull.

Confidence comes from `keypoint_reproj`, which is an **error** in pixels where
low is good, despite `triangulate_one_point`'s stale docstring naming that return
`kp3d_score`. Reading it as a score inverts every confidence in the sweep.

**Body geometry belongs in a control channel, not in the base image.** A SMPL-X
body carries no clothing, hair or identity, so a denoise strong enough to dress it
is a denoise strong enough to invent the subject. Skeletons are also the cheaper
start: Diffuman4D conditions on drawn skeletons rather than meshes, and this
pipeline already computes their input. Reach for SMPL-X only when the bakeoff
shows silhouette or occlusion errors a stick figure cannot fix.

## 3. Repair

```bash
python scripts/video_repair_views.py --backend wan22 \
    --sweep_dir out/sweeps/cam02_to_cam06 --out_dir out/repaired/wan22 \
    --comfy_input_dir /path/to/ComfyUI/input --comfy_output_dir /path/to/ComfyUI/output \
    --control_dir out/skeletons/cam02_to_cam06/kpmap/cam02_to_cam06 \
    --denoise 0.15
```

The renders are VAE-encoded and handed to the sampler as the starting latent, so
a low denoise preserves them; the endpoints are swapped for the warped real
photos; skeleton maps ride along as the control video. Every backend shares the
same head and tail, so a bakeoff's rows differ by model rather than by how each
was configured.

Three ComfyUI behaviours worth knowing, all found by running it:

- `ImageBatchPath` declares `output_is_list`, so ComfyUI maps every downstream
  node over one image at a time; `VAEEncode` then yields N single-frame latents
  instead of one N-frame clip. `VHS_LoadImagesPath` returns a real batch.
- `Wan22FunControlToVideo` needs a `control_video`. Without one its concat latent
  is a single frame and the sampler dies concatenating it against a real clip.
- Outputs are fetched through ComfyUI's `/view` endpoint rather than read off
  disk, because its output directory is configurable.

MiniMax H3 cannot run this pass: it denoises a paired audio-video latent
delivered as a `NestedTensor`, and no installed node can splice an external video
latent into it. The `h3` backend therefore runs H3's native first/last-frame
generation, which is the "invent the tween" experiment above.

## 4. Score before believing anything

```bash
python scripts/render_pair_sweep.py ... --pair cam02 cam06 --holdout_label cam04
python scripts/score_novel_views.py --sweep_dir out/sweeps/cam02_to_cam06 \
    --candidate wan22 out/repaired/wan22/sweep_0010.png
```

`--holdout_label` names a real camera between the pair. Its nearest swept frame
is snapped to that camera's exact centre, so its photo warps in as ground truth.
`score_novel_views.py` scores every candidate **and the raw render**: a candidate
below the raw render is destroying agreement with the real cameras, however good
it looks.

### Measured: skeleton control did not help, and hurt at high denoise

Run on `tatum_jump.sogst`, 17 frames (58-74) swept `cam02` -> `cam06`, probe held
out at `cam04`, scored on the subject region against that camera's real photo.
Repair by Wan2.2-Fun-Control-A14B (low-noise expert).

| candidate | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| raw render | 27.36 | 0.8957 | **0.0756** |
| **VAE round trip only** | **35.06** | **0.9764** | 0.0775 |
| no control, denoise 0.15 | 33.54 | 0.9679 | 0.0907 |
| skeleton control, denoise 0.15 | 33.06 | 0.9677 | 0.0907 |
| no control, denoise 0.40 | 31.61 | 0.9623 | 0.0986 |
| skeleton control, denoise 0.40 | 27.74 | 0.9434 | 0.1218 |

Three things this settles, and one it does not.

**The VAE round trip still beats everything**, reproducing on this pipeline what
the per-frame branch measured: encode-and-decode with no sampling at all gains
7.7 dB over the raw render, and every sampled setting falls below it. The gain
is a low-pass on render speckle, not repair.

**Skeleton control did not help at low denoise** (33.06 vs 33.54, LPIPS
identical) and **cost 3.9 dB at denoise 0.40** (27.74 vs 31.61).

**The denoise curve did not invert.** The hypothesis worth testing was that
pinning geometry with a control channel would make higher denoise pay off. It did
not: more denoise is monotonically worse, with control and without, and control
makes the decline steeper.

What this does NOT settle is whether skeleton control can work at all here. The
drawn skeleton covers about 0.7% of the frame (7,797 non-black pixels at
1024x1024), so as a control *video* it is almost entirely black, and
Wan-Fun-Control was trained on dense control signals. A near-empty control plausibly
pushes the output toward empty, which matches the extra smoothing visible at
denoise 0.40. Before concluding that body geometry cannot help, try a denser
control signal: thicker strokes (`draw_one_skeleton`'s `radius`/`thickness`), or
a filled body rather than a stick figure.

Two rules the first bakeoff earned:

- **Run the VAE round trip as a control.** On the first run every Wan setting
  beat the raw render on PSNR, which looked like a win until a pure
  encode-decode with no sampling beat all of them (36.15 dB vs 29.78). The gain
  was a mild low-pass removing render speckle, not repair.
- **Hold the probe camera out of training.** Otherwise the render already scores
  ~30 dB there and the measurement answers a question nobody asked. Build the
  training set with `build_frame_dataset.py --exclude`, and exclude the probe's
  whole stereo pair, since its twin sits under a degree away.
