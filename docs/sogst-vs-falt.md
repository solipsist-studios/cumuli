# `.sogst` vs `.falt`: A Measured Comparison

> **LOCAL ONLY. Do not push this file.** Every number below comes from degree-2
> archives, which fall under the degree-2 hold (no remote, no cloud tools). Section 6
> describes an unreleased trainer change that is also local only. Before any version of
> this report goes public, re-measure on a degree-1 take and delete section 6.

This report compares our 4D Gaussian splat format, `.sogst`, with Faltwerk's `.falt`
(`github.com/edrethardo/Faltwerk`, MIT, commit `edf6c41`), on the same capture and the
same scoring views. It is written for whoever decides what `.sogst` adopts next. Section 1
describes the two formats, sections 2 and 3 give the head-to-head result, sections 4 and 5
record which Faltwerk ideas transferred and which did not, section 6 covers the one
still open, and section 7 lists the caveats.

**Short answer.** At a matched file size, `.sogst` scores about 1.3 to 2.3 dB higher PSNR
and slightly lower LPIPS than `.falt`, from a pipeline that trains about 4.5 times faster.
Faltwerk's long-lived splats move more cleanly through fast motion, and its first frame
arrives sooner; its viewer plays back more slowly. Three Faltwerk ideas, adapted to
`.sogst`, cut our files by up to 78% with no measured quality loss.

## 1. The Two Formats

| | `.sogst` | `.falt` |
|---|---|---|
| Splat lifetime | Short (median `t_sigma` 57 to 77 ms); a Gaussian opacity window | Long; explicit birth and death frames, identity carried across sections |
| Motion | Per-splat velocity and acceleration (degree 2) | Per-attribute curves with 1 to 9 knots per section, covering position, rotation, scale, opacity and colour |
| Container | ZIP of lossless WebP textures plus `meta.json`; SOG v2 compatible | One binary section per 16 frames, entropy-coded with rANS |
| Streaming unit | 0.1 s segment | 16-frame section (0.53 s at 30 fps) |
| Decode | WebP, then GPU texture upload | Full rANS decode of a section in a worker |
| Viewer | PlayCanvas (WebGL or WebGPU) | Faltwerk's own WebGPU player |

## 2. Test Setup

- **Capture.** Take 260529-171310 ("annika_and_dave"): two subjects, 12 cameras,
  144 frames, 4.77 s at 29.97 fps, 2048×1792.
- **Our pipeline.** The locked degree-2 recipe: four temporal windows, a gated stitch,
  a fine-tune, an SH-clamps-only bake. Training took 73 minutes (window training loops
  plus the fine-tune).
- **Faltwerk.** Its standard rig recipe: track a per-frame "teacher" model, fit curves,
  fine-tune against the images, encode. Faltwerk does not make the teacher. We ran it
  three ways:
  - `run_pf`: Faltwerk's normal input, 144 independent static 3DGS fits, one per frame.
  - `run_q`: a teacher sliced per frame from our trained model, quality-first.
  - `run_r11`: the same teacher with Faltwerk's own rate target at 11.4 Mbit/s.
- **Scoring.** PSNR, SSIM and LPIPS on all 144 views of `cam00`, with ground truth
  composited onto black. Both pipelines trained on all 12 cameras, `cam00` included,
  so the scores compare pipelines but flatter absolute quality. Faltwerk's output was
  decoded and rendered with its own code; only the metric code is shared.

## 3. Results

### Quality and Size

| Stream | Size | PSNR (dB) | LPIPS |
|---|---|---|---|
| `.falt` end to end (`run_pf`) | 8.36 MB | 38.138 | 0.01506 |
| `.falt` sliced teacher, quality-first (`run_q`) | 12.83 MB | 37.088 | 0.01466 |
| `.falt` sliced teacher, matched rate (`run_r11`) | 6.37 MB | 37.138 | 0.01454 |
| `.sogst` shipped bake | 30.39 MB | 39.389 | 0.01288 |
| `.sogst` recommended packing (section 4) | 22.33 MB | 39.362 | 0.01291 |
| `.sogst` 3.0 MB/s rate target + fine-tune | 13.55 MB | 39.448 | 0.01303 |
| `.sogst` 1.5 MB/s rate target + fine-tune | 6.81 MB | 39.395 | 0.01395 |

The 1.5 MB/s `.sogst` file is 19% smaller than Faltwerk's end-to-end stream and better on
both metrics: +1.26 dB PSNR, −0.0011 LPIPS. Against the matched-rate `.falt` stream of
nearly the same size, it leads by 2.26 dB and 0.0006 LPIPS.

Faltwerk did better from its own input than from ours: independent per-frame teachers
gave 1.05 dB more PSNR than the teacher sliced from our model, at about the same LPIPS.
Distilling our model was therefore no head start.

### Training Time

| Stage | `.falt` end to end | `.sogst` |
|---|---|---|
| Per-frame teachers | 95 min | – |
| Tracking (stage 1) | 201 min | – |
| Curve fit | under 1 min | – |
| Fine-tune | 28 min | – |
| Encode | about 1 min | – |
| Window training + stitch fine-tune | – | 73 min |
| Rate target prune + fine-tune (optional) | – | under 2 min of training per target |
| **Total** | **about 5 h 26 min** | **about 73 min** |

Both exclude dataset preparation, which the two pipelines share.

### Visual Quality and Playback

These observations come from viewing `run_r11` and the 1.5 MB/s `.sogst` file side by side.

- **Fast motion.** `.falt` looks softer but smears less between 0.3 and 0.8 s. Our short
  splats cross-fade during fast motion: an arm is drawn by splats fading out at the old
  pose and new ones fading in at the new pose. Faltwerk moves one splat along a curve.
- **Solidity.** `.falt` reads more like a surface and shows fewer floaters. It draws
  about 20,000 to 25,000 splats per frame, against a median of 40,500 for the 1.5 MB/s
  `.sogst` file (121,962 for the shipped bake). Its encoder also removes "giant" and
  "runaway" splats explicitly. Ours has no equivalent cut.
- **Colour.** `.falt` looked more washed out. This has not been measured.
- **Time to first frame.** `.falt` is faster. It needs only its first section: 0.89 MB
  for `run_r11`. The 1.5 MB/s `.sogst` file needs 4.98 MB of its 6.81 MB (73%) before
  it can draw (`streams.reveal_bytes`). Almost all of that is the persistent group of
  always-drawn splats, which holds 251,356 of the file's 302,053 splats (83%); the first
  segment adds 0.05 MB.
- **Playback.** `.falt` plays back choppily. Its viewer reports 50 to 61 ms of GPU time
  per frame for about 22,000 splats, which caps playback near 16 to 20 fps. That is a
  viewer cost (per-frame curve evaluation plus a full bitonic depth sort), not a format
  limit. A 16-frame section must also be fully decoded before it plays, and the player
  decodes only two sections ahead.

## 4. What Transferred to `.sogst`

The comparison started from Faltwerk's encoder ideas. Three carried over. All three
change only the encoder: a version-1 player decodes the output unchanged.

1. **Precision from the camera footprint.** The low byte of each 16-bit position,
   velocity and acceleration value is close to random noise. WebP cannot compress it,
   and those three planes made up about 30% of every archive (24.0 to 24.3 bits per
   splat). The encoder now drops low bits on a coarser grid and records them as
   `drop_bits` (spec revision 15, §2.2.1). The rule: the largest drop that keeps the p99
   on-screen error at or below 0.5 px at the nearest training camera. That gives 5/6/7
   bits (position/velocity/acceleration) on this take, −23% at −0.002 dB. The rule has
   to be computed per take; another take allowed only 2/6/6.
2. **A smaller SH palette.** 16,384 entries in place of 65,536 saves a further 1 MB here
   (the palette is fixed-size). Combined with item 1: **22.33 MB, −26.5%, at −0.027 dB
   and +0.00003 LPIPS.** Faltwerk's rule of choosing each splat's SH degree from its
   energy found nothing to cut: under 3% of splats fall below 8/255 in either band.
3. **A delivery-rate target.** `scripts/fit_rate_target.py` ranks splats by their largest
   blend weight over the training views, searches the kept count by packing until the
   archive fits, then fine-tunes the kept set for 3,000 steps with densification off. The
   1.5 MB/s file keeps 303,678 of 1,034,209 splats, is 78% smaller than the shipped bake,
   and matches its PSNR (39.395 against 39.389 dB) at +0.0011 LPIPS.

Two Faltwerk practices also went in: a CRC-32 check on archive entries (spec §2), and
golden-digest tests of the encoder's exact output bytes.

## 5. What Did Not Transfer

- **Lossless video.** Faltwerk codes no video, but the comparison raised the question of
  a video-first container like MPEG's V-PCC Gaussian splat profile. Encoding the
  per-segment textures as lossless HEVC made them larger: +28% on the shipped archive
  and +62% on the recommended packing, decoded back losslessly in both cases. Inter-frame
  prediction saved only 3 to 6 points over intra-only, because consecutive `.sogst`
  segments hold different splats and give the codec nothing to predict from. A video
  container only pays off with a lossy codec and splats that persist across frames.
- **The rANS bitstream and its predictors.** These would break SOG v2 compatibility and
  direct texture upload, and every player would need a bit-exact integer decoder.

## 6. Open: Longer-Lived Splats (LOCAL ONLY)

Faltwerk's cleaner fast motion comes from long-lived splats. The closest `.sogst` change
is an experimental trainer option (branch `local/temporal-rot-scale`, never to be pushed)
that gives each splat an angular velocity and a log-scale rate, with the scale change
bounded to 2× either way.

On one window of take 260529-174623, with `cam00` held out:

| Arm | Seeds | Splats | Packed size | PSNR (dB) | LPIPS |
|---|---|---|---|---|---|
| Degree-2 (current) | 5 | 219,468 | 4.90 MB | 24.63 | 0.0663 |
| + rotation/scale, unbounded | 5 | 160,977 | 4.59 MB | 23.89 | 0.0651 |
| + rotation/scale, bounded at 2× | 3 | 113,948 | 3.45 MB | 23.83 | 0.0602 |

The bounded variant's splats live 2.2 to 3.3 times longer (median `t_sigma` 134 to
190 ms against 57 to 61 ms) without growing in size. LPIPS is lower and much steadier
across seeds; PSNR drops by about 0.8 dB. That take turned out to be a broken splat, so a
re-test on annika_and_dave (window 0, `cam00` held out, bounds of 2× and 4×, 3 seeds each)
is running, scored separately over the 0.3 to 0.8 s motion stretch. Its results belong
here when it finishes.

## 7. Caveats

- One take. Every head-to-head number comes from annika_and_dave.
- `cam00` was in training for every row of section 3. Only section 6 holds it out.
- Faltwerk's per-frame teachers used our trainer with a short schedule (3,000 iterations,
  SH degree 0). Faltwerk ships no teacher trainer, so another one could change its result.
- The visual notes in section 3 come from one viewing session, and the colour
  observation is unmeasured.
- Faltwerk's own evaluation and our harness agree on its PSNR to the third decimal
  (37.088 dB on `run_q`), which supports the cross-format scoring.

## 8. Reproduction

Scripts and results live in `/media/Exos/4dgs/sogst_falt_phases/` on the capture machine.

| Step | Script |
|---|---|
| Scoring (RGBA ground truth composited onto black) | `tools/bench_eval.py`, `tools/falt_bench.py` |
| Faltwerk inputs from a cumuli run | `tools/prep_faltwerk.py`, `tools/perframe_teachers.py` |
| Faltwerk recipe runs | `faltwerk/run_q`, `run_r11`, `run_pf` (`recipe.sh` in each) |
| `.sogst` variants for this take | `tools/run_171310.sh` |
| Lossless HEVC comparison | `tools/hevc_textures.py` |
| Viewer comparison page | `tools/make_viewer_compare.py` |

The encoder changes are on local branches: `sogst/position-precision` (precision and
`drop_bits`), `sogst/rate-target` (`fit_rate_target.py`) and `sogst/spec-rev15` (spec
revision 15).
