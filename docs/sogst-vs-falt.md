# `.sogst` and `.falt`: Choosing a 4D Gaussian Splat Format

This report is for developers deciding which dynamic (4D) Gaussian splat format to
support in a player, an engine or a pipeline. It compares `.sogst`
([spec](sogst-format.md)) with Faltwerk's `.falt`
([github.com/edrethardo/Faltwerk](https://github.com/edrethardo/Faltwerk)) on one capture, scored on the same views.

Section 1 is the decision guide. Section 2 compares what each format asks of an
implementer. Section 3 measures quality, size and encoding time, and section 4 playback
and appearance. Section 5 explains why neither format uses a video codec, section 6
lists the caveats, and section 7 covers reproducing the results.

## 1. Which to Support

| If your priority is | Choose | Why |
|---|---|---|
| Quality per byte | `.sogst` | 1.3 to 2.3 dB higher PSNR and lower LPIPS at a matched file size (section 3) |
| Smooth playback on modest GPUs | `.sogst` | Held 60 fps on a MacBook Air and an RTX 4090. Faltwerk's player needed 14 to 24 ms of GPU time per frame on the Air and dropped to 37 to 57 fps (section 4) |
| Reusing an existing splat decoder | `.sogst` | Static attributes are byte-compatible with PlayCanvas SOG v2 |
| Encoding speed | `.sogst` | 73 minutes against 5 h 26 min to a finished stream (section 3) |
| Fastest first frame | `.falt` | Draws after 14% of the file, against 73% for `.sogst` (section 4) |
| Clean fast motion | `.falt` | Long-lived splats move along curves and do not cross-fade (section 4) |
| Permissive licence for the whole stack | `.falt` | Code and spec under MIT. `.sogst`'s spec is CC-BY-4.0, so independent implementations are free; its reference encoder is PolyForm Noncommercial |

## 2. What Each Format Asks of an Implementer

| | `.sogst` | `.falt` |
|---|---|---|
| Container | ZIP (stored, no compression) of lossless WebP textures plus `meta.json` | One binary section per 16 frames, rANS entropy-coded |
| Decoder | A WebP decoder and a JSON parser. Splat `i` is texel `i` | An integer rANS decoder; optional k-NN prediction, "follow" and B-spline modes; a TypeScript reference port is provided |
| Training | Configurable training windows, with a final stitch and fine-tune step | Per-frame 3DGS "teacher" splats required to train fixed-size 16-frame windows, with a fine-tune step |
| Per-frame work | Evaluate `xyz + v·dt + a·dt²` and a Gaussian opacity window | Evaluate one curve per attribute per splat |
| Splat lifetime | Short; a Gaussian opacity window per splat | Long; explicit birth and death frames, identity kept across sections |
| View-dependent colour | SH degree 1 to 3, palette-coded (degree 2 in this comparison) | SH degree 1 to 3, codebook-coded; Faltwerk's rig recipe produces degree 0 only |
| Streaming unit | 0.1 s segment, byte-rangeable | 16-frame section (0.53 s at 30 fps), decoded whole |
| Reference player | A PlayCanvas-based viewer | Faltwerk's WebGPU player (needs HTTPS or localhost) |
| Spec | English, normative, with a conformance checklist and a decoded-field comparison tool | German, describing the bitstream versions 1 to 6 |
| Licence | Spec CC-BY-4.0; reference implementation PolyForm Noncommercial | MIT |

## 3. Quality, Size and Encoding Time

### Setup 
Both formats encoded the same capture: 
- two subjects
- 12 cameras
- 144 frames
- 4.77 s at 29.97 fps
- 2048×1792. 

Every stream was scored on all 144 views of one camera (`cam00`), with ground truth composited 
onto black. That camera was in training for both, so the scores compare the formats but flatter 
absolute quality. Faltwerk decoded and rendered its own streams; the two share only the metric
code. Faltwerk's own evaluation and ours agree on its PSNR to the third decimal
(37.088 dB).

| Stream | Size | PSNR (dB) | LPIPS |
|---|---|---|---|
| `.falt` rate target (11.4 Mbit/s) | 6.37 MB | 37.138 | 0.01454 |
| `.falt` end to end | 8.36 MB | 38.138 | 0.01506 |
| `.falt` quality-first | 12.83 MB | 37.088 | 0.01466 |
| `.sogst` 1.5 MB/s rate target | 6.81 MB | 39.395 | 0.01395 |
| `.sogst` 3.0 MB/s rate target | 13.55 MB | 39.448 | 0.01303 |
| `.sogst` recommended packing | 22.33 MB | 39.362 | 0.01291 |
| `.sogst` full bake | 30.39 MB | 39.389 | 0.01288 |

### Results
![Archive size against PSNR for .sogst and .falt streams](img/sogst-vs-falt/size-vs-psnr.svg)

![Archive size against LPIPS for .sogst and .falt streams](img/sogst-vs-falt/size-vs-lpips.svg)

At nearly the same size as the `.falt` rate-target stream, the 1.5 MB/s `.sogst` file
scores 2.26 dB higher PSNR and 0.0006 lower LPIPS. It is also 19% smaller than the
end-to-end `.falt` stream and better on both metrics.

![Training time to a finished stream: .sogst 1 h 13 min, .falt 5 h 26 min](img/sogst-vs-falt/training-time.svg)

| Stage | `.sogst` | `.falt` end to end |
|---|---|---|
| Per-frame teacher fits | – | 95 min |
| Tracking | – | 201 min |
| Curve fit | – | under 1 min |
| Training: four windows plus stitch fine-tune | 73 min | – |
| Fine-tune | – | 28 min |
| Encode | not timed (bake and pack, minutes) | about 1 min |
| **Total** | **about 1 h 13 min plus the bake** | **about 5 h 26 min** |

Both ran on one RTX 5090. Neither total includes dataset preparation, which the two
share. A `.sogst` rate target adds under 2 minutes of fine-tuning per target.

## 4. Playback and Appearance

![Bytes needed before the first frame: .sogst 4.98 of 6.81 MB, .falt 0.89 of 6.37 MB](img/sogst-vs-falt/first-frame.svg)

- **First frame.** `.falt` draws after its first section: 0.89 MB of 6.37 MB. A
  `.sogst` file draws after its persistent group and first segment, recorded in the
  manifest as `streams.reveal_bytes`: 4.98 MB of 6.81 MB. The persistent group holds the
  always-drawn splats, 83% of them in this file. That is an encoder setting, not a
  format limit.
- **Frame rate.** Measured in a real browser (Brave) with vsync at 60 Hz, on two
  machines. This compares the two reference players more than the two formats.

  ![Playback frame rate in each format's player on a MacBook Air and an RTX 4090](img/sogst-vs-falt/playback-fps.svg)

  On an RTX 4090 both players held 60 fps, with no frame over 50 ms; Faltwerk's player
  reported about 5 ms of GPU time per frame. On a MacBook Air the `.sogst` viewer still
  held 60 fps, including the full bake of about 120,000 splats per frame. Faltwerk's
  player reported 14 to 24 ms of GPU time per frame there, against a 16.7 ms budget, and
  dropped to 57 fps (rate target), 43 fps (end to end) and 37 fps (quality-first),
  while drawing only 20,000 to 50,000 splats per frame. A frame that misses the budget
  waits for the next display refresh, so the quality-first stream mostly ran at half
  rate. Faltwerk's player evaluates every curve and runs a full bitonic depth sort on
  the GPU each frame.

  | Stream | MacBook Air: fps | Air: frames over 33 ms in 20 s | Air: player GPU time | RTX 4090: fps |
  |---|---|---|---|---|
  | `.sogst` 1.5 MB/s (WebGL) | 60.0 | 0 | – | 60.0 |
  | `.sogst` full bake (WebGL) | 59.9 | 1 | – | 59.9 |
  | `.falt` rate target | 57.1 | 41 | 13.7 ms | 59.9 |
  | `.falt` end to end | 43.2 | 203 | 21.3 ms | 60.0 |
  | `.falt` quality-first | 36.7 | 264 | 24.0 ms | 60.0 |

  On WebGPU the `.sogst` viewer also held 60 fps once loaded, with one long frame during
  loading (0.4 to 0.9 s on the Air).
- **Fast motion.** `.falt` looks softer but smears less in the fastest passage
  (0.3 to 0.8 s). `.sogst` splats are short-lived (median opacity window `t_sigma` of
  68 to 80 ms),
  so fast motion is drawn by old splats fading out while new ones fade in. `.falt` moves
  one splat along a curve.
- **Solidity.** `.falt` reads more like a surface and shows fewer floaters. It draws
  about 20,000 to 25,000 splats per frame against a median of 40,500 for the 1.5 MB/s
  `.sogst` file, and its encoder removes oversized and far-travelling splats.
- **Colour.** `.falt` looks flatter, with less view-dependent shading. Every `.falt`
  stream in this comparison stores colour at spherical-harmonic (SH) degree 0, so a
  splat's colour is the same from every direction; the encoder logs `sn_deg 0` for all
  three. The `.falt` format can carry higher SH bands, but Faltwerk's rig recipe tracks
  and fine-tunes at degree 0, so its encoder finds no higher-band energy to keep. Its
  fine-tune also fits the teacher's degree-0 render, not the camera images: the images
  supply only the silhouette. The `.sogst` streams carry SH degree 2 and fine-tune
  against the camera images.

## 5. Why Not a Video Codec

MPEG's V-PCC profile for Gaussian splats stores splat attributes in video. We tested the
idea on `.sogst` by encoding each archive's textures as lossless HEVC, one frame per
segment. The result was larger than WebP: +28% on the full bake and +62% on the
recommended packing, with every frame decoded back losslessly. Inter-frame prediction
saved only 3 to 6 points over intra-only coding, because consecutive `.sogst` segments
hold different splats and give a video codec nothing to predict from. A video container
pays off only with a lossy codec and splats that persist across frames, which neither
format uses today.

## 6. Caveats

- One capture. Every number comes from the same 4.77 s two-person take.
- The scoring camera was in training for both formats. Absolute scores are optimistic;
  the comparison between formats is not affected, since both saw it.
- Faltwerk ships no teacher trainer. Our per-frame fits used 3,000 iterations at SH
  degree 0; a different teacher could move its result. A teacher with higher SH bands
  would not change `.falt`'s colour, because Faltwerk's tracking and fine-tune render
  the teacher at degree 0.
- The appearance notes in section 4 come from one viewing session.
- Frame rates come from one run per stream per machine, at the window size given in the
  chart. Faltwerk's GPU time grows with the number of pixels drawn, so a larger window
  on the same MacBook Air drops its frame rate further. The MacBook Air's exact chip is
  not recorded: the browser masks the GPU name.

## 7. Reproducing

- **`.falt`:** Faltwerk's standard recipe, `tools/faltwerk_recipe.py --source images`,
  with a rig scene file (`kind: rig`) and one 3DGS PLY per frame as the teacher. Its
  `docs/recipe.md` describes the inputs.
- **`.sogst`:** this repository's pipeline (`docs/pipeline.md`), then
  `scripts/bake_sogst.py` with `--means_drop_bits`, `--motion_drop_bits`,
  `--accel_drop_bits` and `--shn_count`, and optionally `scripts/fit_rate_target.py`
  for a delivery rate.
- **Scoring:** render each stream with its own player code at the test camera's views,
  composite the ground truth onto black, and compute PSNR, SSIM and LPIPS over all views.
