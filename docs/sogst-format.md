<!--
SPDX-License-Identifier: CC-BY-4.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

This specification is deliberately licensed differently from the rest of this
repository (PolyForm-Noncommercial-1.0.0). A format nobody may implement
commercially is not a format. The reference *implementation* in scripts/
remains PolyForm; this document, and any independent implementation written
from it, do not.
-->

# The `.sogst` format — SOG + spacetime

**Version 3 container. Specification revision 1, 2026-08-13.**

`.sogst` stores a dynamic (4D) Gaussian splat scene as a ZIP archive of WebP
attribute textures plus a JSON manifest. Static attributes follow the PlayCanvas
**SOG v2** conventions byte for byte, so an existing SOG decoder reconstructs
them unmodified; the spacetime extension adds per-splat linear motion and a
temporal radial-basis window, an optional second-order motion term, and an
optional temporal segmentation that lets a player cull and stream by time.

The name is literal: **SOG** for the container, **st** for spacetime.

## 0. Status, scope, and conformance language

This document specifies **the version-3 container only**. Versions 1 and 2 were
unpublished internal predecessors; they are described here only where a reader
must still accept them (§8).

The key words MUST, MUST NOT, REQUIRED, SHOULD, SHOULD NOT, MAY and OPTIONAL are
to be interpreted as in RFC 2119.

Two roles are defined:

- an **encoder** produces `.sogst` archives;
- a **player** (decoder) consumes them.

A **minimal player** MAY ignore the `shN` and `accel` groups and MUST still
render a conforming file correctly, at reduced fidelity. Everything else is
required.

The reference implementation is `scripts/sog_pack.py` (encoder),
`scripts/splat4d_io.py` (container writer) and `scripts/eval_render.py`
`decode_v3_fields()` (the only complete inverse of the encoder — use it as the
oracle when validating an independent implementation).

## 1. The representation

A scene is `N` **spacetime Gaussians**. Each carries the usual 3DGS attributes —
position, rotation, scale, opacity, spherical-harmonic colour — plus a linear
velocity and a temporal window. Attributes are stored *at the splat's own
temporal centre*, not at t = 0.

At clip time `t` (seconds, absolute — not normalised), a splat is evaluated as:

```
dt        = t - t_center

mean(t)   = xyz + v*dt                        (motion.degree == 1)
mean(t)   = xyz + v*dt + a*dt*dt              (motion.degree == 2)

alpha(t)  = sigmoid(opacity) * exp(-0.5 * (dt / t_sigma)^2)
```

Rotation, scale and colour are constant in `t`.

Three things about this are easy to get wrong, and all three fail silently:

1. **The temporal factor is unnormalised.** There is no `1/sqrt(2*pi*sigma^2)`
   term. Both the reference encoder and the reference renderer have that
   normalisation deliberately absent; adding it darkens every splat by a
   sigma-dependent factor and the error looks like a global exposure bug.
2. **`t_sigma` is a standard deviation in seconds, not a variance.** It MUST be
   greater than zero.
3. **`a` is the raw `dt^2` coefficient, not half-acceleration.** There is no
   factor of 1/2. This matches the SpacetimeGaussians coefficient convention.

Position, velocity and acceleration are in **scene units** and **scene units per
second** (and per second squared). `t_center` and `t_sigma` are in seconds on the
same clock as `time.min` / `time.max`.

## 2. Container

A `.sogst` file is a ZIP archive.

- Entries MUST be stored with **no compression** (`ZIP_STORED`). WebP payloads
  are already compressed, and stored entries let a player byte-range into the
  archive.
- Entries MUST NOT carry ZIP extra fields. A conforming writer's local entry
  header is therefore exactly `30 + len(name)` bytes, which is what makes the
  streaming offsets of §6 computable analytically.
- `meta.json` MUST be the **first** entry.
- A player identifies a v3 file by the leading ZIP magic `PK\x03\x04` (v1 and v2
  files begin with the ASCII magic `OMG4`, `0x34474D4F` little-endian).

Every other entry is a lossless WebP texture.

### 2.1 Texture conventions

- Textures are **near-square**: `width = ceil(sqrt(M))`,
  `height = ceil(M / width)`, where `M` is the number of splats the texture
  covers (the whole file, or one group — see §6).
- Splat `i` of the covered range lives at row-major texel `i`.
- Padding texels past `M` are unspecified; encoders SHOULD write zero. Players
  MUST NOT read them.
- Every texture MUST be encoded **lossless** WebP with the `exact` flag set.
  Both matter: every texel is a codebook index or a byte of a 16-bit integer, so
  one lossy pixel decodes to a wrong value, and without `exact` libwebp is free
  to rewrite the RGB of fully-transparent texels — which destroys indices stored
  alongside a zero alpha.

### 2.2 The 16-bit split-plane convention

Positions, velocities and accelerations are stored as 16-bit values split across
two textures — `*_l.webp` carrying the low byte and `*_u.webp` the high byte of
each axis in R, G, B. A **log transform** is applied first, so precision follows
magnitude:

```
encode:  T   = sign(x) * ln(1 + |x|)
         q   = round((T - mins[c]) / (maxs[c] - mins[c]) * 65535)   clamped to [0, 65535]
         lo  = q & 0xFF          hi = q >> 8

decode:  q   = (hi * 256 + lo) / 65535
         T   = mins[c] + q * (maxs[c] - mins[c])
         x   = sign(T) * (exp(|T|) - 1)
```

`mins` / `maxs` are per-axis and live in `meta.json`. When `maxs[c] == mins[c]`
an encoder MUST use a span of 1.0 to avoid dividing by zero; the decode is then
constant at `mins[c]` regardless, so no player-side special case is needed.

Alpha in `*_l.webp` / `*_u.webp` is unused and MUST be written as 255.

Velocity uses this scheme rather than a codebook deliberately: 256 levels
visibly quantizes motion.

## 3. `meta.json`

```jsonc
{
  "version": 3,
  "format": "sogst",
  "asset":  { "generator": "…" },
  "count":  123456,
  "time":   { "min": 0.0, "max": 10.0, "fps": 30.0 },

  "means":  { "mins": [3], "maxs": [3], "files": ["means_l.webp", "means_u.webp"] },
  "scales": { "codebook": [256], "files": ["scales.webp"] },
  "quats":  { "files": ["quats.webp"] },
  "sh0":    { "codebook": [256], "files": ["sh0.webp"] },
  "motion": { "degree": 1, "mins": [3], "maxs": [3],
              "files": ["motion_l.webp", "motion_u.webp"] },
  "trbf":   { "center": { "codebook": [256] },
              "sigma":  { "codebook": [256] },
              "files": ["trbf.webp"] },

  "shN":    { "count": 65536, "bands": 3, "codebook": [256],
              "files": ["shN_centroids.webp", "shN_labels.webp"] },   // OPTIONAL
  "accel":  { "mins": [3], "maxs": [3],
              "files": ["accel_l.webp", "accel_u.webp"] },            // OPTIONAL

  "cov2d_scale": [1.0, 1.0],                                          // OPTIONAL
  "segments": { … },                                                  // OPTIONAL, §5
  "streams":  { … }                                                   // OPTIONAL, §6
}
```

| key | required | meaning |
|---|---|---|
| `version` | yes | Always `3`. **Not** bumped by the `.omg4` → `.sogst` rename — renumbering would break deployed readers for nothing. |
| `format` | yes for new writers | `"sogst"`. The container self-identifies here rather than relying on the file extension. Players MUST accept its absence (files written before this revision) and MUST NOT reject an unknown value outright — check `version`. |
| `asset.generator` | no | Free-form producer string. |
| `count` | yes | `N`, total splats in the file. |
| `time.min`, `time.max` | yes | Clip bounds in seconds. |
| `time.fps` | yes | Advisory, for UI frame counters and scrub granularity. It does **not** affect evaluation — the model is continuous in `t`. |
| `cov2d_scale` | no | `[kx, ky]` screen-space 2D-covariance multiplier a player applies when rasterising. Compensates trainers whose screen-space footprints were inflated. Absent means `[1, 1]`. |

`files` arrays name the archive entries backing each group. In the streamed
layout (§6) they name the *basenames*; actual entries are prefixed. A player
SHOULD resolve textures through `files` rather than hardcoding names.

## 4. Attribute groups

### 4.1 `means` — `means_l.webp`, `means_u.webp`

Position at `t_center`, split-16 per §2.2 over `means.mins` / `means.maxs`.

### 4.2 `quats` — `quats.webp`

Rotation as a unit quaternion in **smallest-three** form. Let the quaternion be
`(w, x, y, z)`, normalised, with the sign chosen so its largest-magnitude
component is positive.

- **A** = `252 + i`, where `i` is the index **in `(w, x, y, z)` order** of the
  largest-magnitude component — the one that is dropped. So A is 252 when `w`
  was dropped, 253 for `x`, 254 for `y`, 255 for `z`.
- **RGB** hold the remaining three components **in their original `(w, x, y, z)`
  relative order**, i.e. dropping `x` stores `(w, y, z)`, dropping `w` stores
  `(x, y, z)`.
- Each stored component `s` lies in `[-1/sqrt(2), +1/sqrt(2)]` and is byte-coded
  as `round((s / sqrt(2) + 0.5) * 255)`; the decode is `(b/255 - 0.5) * sqrt(2)`.
- The dropped component is recovered as `sqrt(max(0, 1 - a^2 - b^2 - c^2))` and
  is non-negative by construction.

### 4.3 `scales` — `scales.webp`

RGB are indices into the shared 256-entry `scales.codebook`. Values are in
**natural-log space**; a player applies `exp()`. Alpha unused, written 255.

### 4.4 `sh0` — `sh0.webp`

- **RGB**: indices into the shared 256-entry `sh0.codebook`, giving the raw SH
  DC coefficients `f_dc_0..2`. These are **not** RGB colour;
  `colour = 0.5 + C0 * f_dc` with `C0 = 0.28209479177387814`.
- **A**: **linear** opacity, `round(255 * sigmoid(opacity_logit))`. Note the
  asymmetry — everything else in this format is stored in the trainer's
  parameter space, but opacity is stored already-activated and quantized to 8
  bits. This is the peak opacity, reached at `t_center`.

### 4.5 `motion` — `motion_l.webp`, `motion_u.webp`

Linear velocity in scene units per second, split-16 per §2.2 over
`motion.mins` / `motion.maxs`. `motion.degree` is `1` or `2`; see §4.8.

### 4.6 `trbf` — `trbf.webp`

The temporal radial-basis window.

- **R**: index into `trbf.center.codebook` → `t_center` in seconds.
- **G**: index into `trbf.sigma.codebook` → `t_sigma` in seconds, `> 0`.
- **B**: unused, 0. **A**: unused, 255.

Both codebooks have 256 entries. The reference encoder clusters `t_sigma` in the
log domain so precision is allocated by relative rather than absolute error; the
codebook it emits is in linear space either way, so players are unaffected.

### 4.7 `shN` — `shN_centroids.webp`, `shN_labels.webp` (OPTIONAL)

Higher-order spherical harmonics, vector-quantized. Present only when
`meta.shN` is.

- `bands` ∈ {1, 2, 3}, giving `coeffs` ∈ {3, 8, 15} coefficients per colour
  channel and a centroid vector of `3 * coeffs` values.
- `shN_centroids.webp` width MUST be `64 * coeffs` — **192, 512 or 960** for 1,
  2 or 3 bands. A decoder infers the band count from this width, so it is
  normative, not incidental. Height is `ceil(count / 64)`.
- Palette entry `n` occupies `coeffs` consecutive texels starting at
  `u = (n % 64) * coeffs` on row `v = n / 64`. Texel `(u + k, v)` channel `j`
  holds an index into the shared 256-entry `shN.codebook` for the coefficient
  at **channel-major** position `j * coeffs + k`.
- `shN_labels.webp` **R** = low byte, **G** = high byte of the 16-bit palette
  entry id. B unused, A = 255. `shN.count` is the number of palette entries and
  MUST be ≤ 65536.

> **Known limitation of the reference decoder.** `decode_v3_fields()` writes
> reconstructed coefficients into a fixed 45-wide `f_rest` array at stride 15
> regardless of `bands`, so its output layout is only correct for `bands == 3`.
> The reference encoder only ever emits `bands == 3`. New encoders SHOULD emit
> `bands == 3`; players implementing `bands < 3` MUST use the channel-major
> stride `coeffs` given above, not 15.

### 4.8 `accel` — `accel_l.webp`, `accel_u.webp` (OPTIONAL)

Degree-2 motion. Split-16 per §2.2 over `accel.mins` / `accel.maxs`, in scene
units per second squared, as the raw `dt^2` coefficient (§1).

Gating rules, all normative:

- `accel` is present **iff** `motion.degree == 2`.
- An encoder MUST NOT emit `accel` unless the source model provides genuine
  second-order coefficients. Synthesising them (from finite differences, or as
  zeros) is non-conforming: it inflates the file for no fidelity and misleads
  players about the model.
- A player MAY ignore `accel` and evaluate the degree-1 form, and MUST still
  render a `degree == 2` file without error.
- It is all-or-nothing: an encoder emitting `accel` MUST provide all three axes
  for all `N` splats.

## 5. Ordering and temporal segments (OPTIONAL)

Splats in a `.sogst` file are not in arbitrary order. They are laid out as

```
[ persistent | segment 0 | segment 1 | … | segment n-1 ]
```

with each group **Morton-ordered** internally by position (30-bit code, 10 bits
per axis over the scene bounding box). Morton ordering is what makes the WebP
textures compress — spatially adjacent splats land in adjacent texels and their
byte planes become locally smooth. It is an encoder-side quality choice, not
something a player must verify.

A splat is **persistent** when its active interval
`[t_center - k_sigma*t_sigma, t_center + k_sigma*t_sigma]` is longer than
`persistent_span_mult * duration`; that is, when it is visible across enough of
the clip that per-segment culling would not pay for itself. Persistent splats are
always drawn. The rest are bucketed by `t_center` into fixed-length segments.

```jsonc
"segments": {
  "duration": 0.1,                  // segment length, seconds
  "k_sigma": 3.8,                   // active-interval half-width, in sigmas
  "persistent_span_mult": 3.0,      // persistence threshold, in segment durations
  "persistent": [0, 41234],         // half-open index range of always-drawn splats
  "list": [
    { "t0": 0.00, "t1": 0.42, "range": [41234, 43110] },
    …
  ]
}
```

- **All index ranges are half-open, `[first, last)`.** `persistent` is
  `[0, P)`. Segment `range` is `[first, last)`. An empty segment has
  `first == last` and is legal.
- `t0` / `t1` are the *actual* time coverage of the segment's members — the min
  and max of their active intervals — **not** the bucket boundaries. They
  therefore overlap adjacent segments, and `t0` may precede `time.min`. For an
  empty segment they fall back to the nominal bucket bounds
  `[time.min + s*duration, time.min + (s+1)*duration]`.
- `persistent_span_mult` is recorded so that a file's ordering can be
  re-derived — without it, no validator or re-packer can reproduce the
  persistent/dynamic split from the file alone. Players ignore it.

**The drawing rule.** At time `t`, a player draws `[0, P)` plus the single
contiguous index span covering every segment whose `[t0, t1]` contains `t`:

```
active = { s in segments.list : s.t0 <= t <= s.t1 }
if active is empty:  draw [0, P)
else:                draw [0, P)  and  [ min(s.range[0]), max(s.range[1]) )
```

Because segments are ordered by time and their coverage overlaps, the active set
is contiguous, so this is two draw ranges, never a scatter. Splats outside the
span have temporal opacity of roughly `exp(-k_sigma^2/2)` or less and are
culled without visible error.

Segmentation is optional. When `meta.segments` is absent the file is a single
Morton-ordered block and a player draws all `N` splats at every `t`.

## 6. The streamed layout (OPTIONAL)

An archive MAY store one texture set per group instead of whole-clip textures, so
that a sequential download becomes progressively renderable. Quantization stays
**global** — codebooks, mins/maxs and shN centroids live in `meta.json` and are
shared by every group — so a group is decodable the moment its own bytes arrive.

Entries are written in **play order**:

```
meta.json
persistent/means_l.webp, persistent/means_u.webp, …      (geometry + DC colour)
seg_000/means_l.webp, …
seg_001/…
…
shN_centroids.webp                                        (when SH is deferred)
persistent/shN_labels.webp, seg_000/shN_labels.webp, …
```

- Group prefixes are `persistent` and `seg_NNN`, zero-padded to three digits,
  indexed against `segments.list`. Empty segments get no prefix and no entries.
- Texture basenames within a group are exactly the §4 names. Only their position
  in the archive changes.
- When SH is deferred, all `shN_labels` entries move to the tail, behind
  `shN_centroids.webp`. A player starts DC-only playback and layers the
  view-dependent term in as the trailing entries arrive.

```jsonc
"streams": {
  "persistent": "persistent",       // or null when there are no persistent splats
  "segments": ["seg_000", null, "seg_002", …],   // parallel to segments.list
  "sh_deferred": true,
  "reveal_bytes": 4823914,
  "geometry_bytes": 41220118
}
```

- **`reveal_bytes`** — the byte offset one past the end of the last entry a
  player needs before it can put anything on screen: the persistent group plus
  the first non-empty segment's geometry. A progress bar SHOULD fill against
  this, not against the whole file.
- **`geometry_bytes`** — the offset one past the last geometry entry, i.e. where
  the deferred SH tail begins. A player uses it with measured bandwidth to
  decide whether to hold the playhead until gap-free playback is possible.

Both are absolute offsets from the start of the archive to the **local header of
the next entry**. They are computable analytically precisely because entries are
stored with no extra fields (§2); a writer MUST verify its computed offsets
against the written file. Note that the values feed back into the size of the
`meta.json` that contains them — a writer must iterate to a fixed point.

## 7. The interchange PLY

The producer of per-splat data and the encoder of the container are separate
programs (Python and TypeScript respectively). Their contract is **a single
binary PLY** carrying per-splat temporal columns. This section is normative for
both.

### 7.1 Header

- `format binary_little_endian 1.0`.
- One element, `vertex`, with count `N`.
- **Every property is `float`** (float32). A reader MAY reject any non-float
  property rather than guess.

### 7.2 Columns

19 base columns, in this order, all REQUIRED:

| column(s) | convention |
|---|---|
| `x, y, z` | Position **at `t_center`**, not at t = 0. Any re-anchoring in time MUST recompute position, velocity and acceleration together. |
| `rot_0..rot_3` | Quaternion, **w first**: `rot_0 = w`. Need not be normalised — the encoder re-normalises and canonicalises sign. |
| `scale_0..scale_2` | **Natural-log** space. |
| `opacity` | **Logit** space; the *peak* opacity, at `t_center`. |
| `f_dc_0..f_dc_2` | Raw SH DC coefficients, **not** RGB. |
| `vx, vy, vz` | Linear velocity, scene units per second — the coefficient of `(t - t_center)`. |
| `t_center` | **Seconds**, absolute clip time. Not normalised. |
| `t_sigma` | **Standard deviation in seconds, not a variance.** MUST be `> 0`. |

Then, OPTIONAL:

| column(s) | convention |
|---|---|
| `f_rest_0..f_rest_44` | 45 values, **channel-major**: index `j*15 + k` is channel `j`, coefficient `k`. All 45 or none. |
| `ax, ay, az` | **Raw `dt^2` coefficient, not half-acceleration.** All three or none; see §4.8's gating rules, which apply equally to the PLY. |

All columns MUST have exactly `N` entries. Nothing downstream cross-checks
lengths, so a mismatch corrupts silently.

### 7.3 Clip-level scalars

`time_min`, `time_max`, `fps` and `cov2d_scale` are properties of the clip, not
of a splat, so they have no column. **They are carried in PLY comments**, which
keeps the interchange unit a single file:

```
comment sogst.version 1
comment sogst.time_min 0.0
comment sogst.time_max 10.0
comment sogst.fps 30.0
comment sogst.motion_degree 1
comment sogst.cov2d_scale 1.0 1.0
```

- `sogst.time_min`, `sogst.time_max` and `sogst.fps` are REQUIRED.
- An encoder that does not find them **MUST fail with an error**. It MUST NOT
  substitute a default. This is the single most likely place for two
  implementations to diverge, and the failure is silent: an assumed `fps = 30`
  on 24 fps content produces a file that plays at the wrong speed and renders
  perfectly while doing it.
- `sogst.motion_degree` is OPTIONAL and advisory; the presence of `ax, ay, az`
  is what actually determines the degree. When both are present they MUST agree.
- `sogst.cov2d_scale` is OPTIONAL; absent means `1.0 1.0`.
- Unknown `sogst.*` comments MUST be ignored, not rejected.

A sidecar `<name>.sogst.json` with the same keys MAY accompany the PLY, for
toolchains that strip comments. When both are present the sidecar wins. A
conforming producer MUST write the comments regardless of whether it also writes
a sidecar.

## 8. Compatibility and the `.omg4` rename

The format was previously called `.omg4`, after the paper whose training code
first fed the pipeline. Nothing in the container came from that work — the
representation is spacetime-shaped, the container is PlayCanvas SOG, the
streaming layer is ours — so the name was retired in favour of `.sogst`.

The migration is designed so that nothing has to be re-baked:

- Writers emit `.sogst` and set `meta.format = "sogst"`.
- Players MUST keep accepting the `.omg4` extension indefinitely.
- Players MUST keep accepting the v1/v2 ASCII magic `OMG4` (`0x34474D4F`
  little-endian) for already-deployed binary assets.
- `meta.version` stays `3`.

A v3 archive written before this revision has no `format` key; it is otherwise
identical and MUST be accepted.

## 9. Conformance checklist

An encoder conforms when:

- [ ] `meta.json` is the first entry; all entries are `ZIP_STORED` with no extra
      fields.
- [ ] Every texture is lossless WebP with `exact` set.
- [ ] `count` equals the covered splat count, and every group's textures cover
      exactly that many texels.
- [ ] `t_sigma > 0` for every splat.
- [ ] `accel` is present iff `motion.degree == 2`, and only from genuine
      second-order source coefficients.
- [ ] `shN_centroids` width is `64 * coeffs` for the declared band count.
- [ ] When streaming: `reveal_bytes` and `geometry_bytes` were verified against
      the written archive, not just computed.

A player conforms when:

- [ ] It evaluates `mean(t)` and `alpha(t)` exactly as §1 states, with the
      temporal factor **unnormalised**.
- [ ] It renders a `degree == 2` file correctly, whether or not it implements
      `accel`.
- [ ] It renders a file with no `shN` group correctly.
- [ ] It treats segment ranges as half-open and applies the §5 drawing rule.
- [ ] It accepts `.omg4`-named files, `OMG4`-magic v2 files, and v3 archives
      with no `format` key.

## 10. Cross-implementation validation

Two implementations of one contract diverge silently. The check that catches it:
pack the *same* reference PLY with both, decode both with `decode_v3_fields()`,
and assert per-field maximum absolute error within quantization tolerance.

**Compare decoded fields, not rendered PSNR.** Codebook initialisation is
implementation-defined, so byte equality is the wrong bar; PSNR is too coarse to
tell you *which* field is wrong. A per-field error table localises the bug
immediately — a wrong `f_rest` stride and a wrong quaternion mode mapping look
identical in a PSNR number and nothing alike in a field table.
