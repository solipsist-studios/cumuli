<!--
SPDX-License-Identifier: CC-BY-4.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

This specification is deliberately licensed differently from the rest of this
repository (PolyForm-Noncommercial-1.0.0). The reference *implementation* in
scripts/ remains PolyForm. This document, and any independent implementation
written from it, do not.
-->

# The `.sogst` format — SOG + spacetime

**Container version 1. Specification revision 14, 2026-08-18.**

`.sogst` stores a dynamic (4D) Gaussian splat scene as a ZIP archive of WebP
attribute textures plus a JSON manifest. Static attributes follow the PlayCanvas
**SOG v2** conventions byte for byte, so an existing SOG decoder reconstructs
them unmodified. The spacetime extension adds per-splat linear motion, a
temporal radial-basis window, an optional second-order motion term, and an
optional temporal segmentation. The segmentation lets a player cull and stream
by time.

The name is literal: **SOG** for the container, **st** for spacetime.

## 0. Status, scope, and conformance language

This document specifies the container completely. It has no predecessors that a
reader must accommodate: no earlier version of this format was ever released,
and there is no legacy form to accept.

The key words MUST, MUST NOT, REQUIRED, SHOULD, SHOULD NOT, MAY and OPTIONAL are
to be interpreted as in RFC 2119.

Two roles are defined:

- an **encoder** produces `.sogst` archives.
- a **player** (decoder) consumes them.

A **minimal player** MAY ignore the `shN` and `accel` groups and MUST still
render a conforming file correctly, at reduced fidelity. Everything else is
required.

The reference implementation is `scripts/sogst_pack.py` (encoder),
`scripts/sogst_io.py` (container writer) and `decode_sogst_fields()` in
`scripts/eval_render.py`. That decoder is the only complete inverse of the
encoder. Use it as the oracle when you validate an independent implementation.
`scripts/sogst_ply.py` reads and writes the interchange PLY of §7.

## 1. The representation

A scene is `N` **spacetime Gaussians**. Each carries the usual 3DGS attributes
(position, rotation, scale, opacity, spherical-harmonic colour) plus a linear
velocity and a temporal window. The file stores attributes *at the splat's own
temporal centre*, not at t = 0.

At clip time `t` (seconds, absolute, not normalised), a player evaluates a
splat as:

```
dt        = t - t_center

mean(t)   = xyz + v*dt                        (motion.degree == 1)
mean(t)   = xyz + v*dt + a*dt*dt              (motion.degree == 2)

alpha(t)  = sigmoid(opacity) * exp(-0.5 * (dt / t_sigma)^2)
```

Rotation, scale and colour are constant in `t`.

Three things about this are easy to get wrong, and all three fail silently:

1. **The temporal factor is unnormalised.** There is no `1/sqrt(2*pi*sigma^2)`
   term. The reference encoder and the reference renderer both omit that
   normalisation deliberately. If you add it, every splat darkens by a
   sigma-dependent factor, and the error looks like a global exposure bug.
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
- Entries MUST NOT carry ZIP extra fields.
- Entries MUST NOT use a **data descriptor**: general-purpose bit 3 MUST be
  clear, and the writer MUST put the compressed and uncompressed sizes in the
  local header. This is not pedantry about the container. A streaming ZIP
  writer that does not know an entry's size until it has finished writing it
  will set bit 3. It will write zeros for both sizes and append a 16-byte
  descriptor after the payload. That descriptor has no extra fields, so it
  satisfies the rule above as literally worded, while it destroys the
  property that rule exists to protect. A player that walks the archive
  forward reads a compressed size of zero and cannot find the next entry. Every entry also
  costs 16 bytes more than §6's offsets assume.

- `meta.json` MUST be the **first** entry.
- A player identifies a `.sogst` file by the leading ZIP magic `PK\x03\x04`,
  then by `meta.version` and `meta.format`.

Together, the first three rules make a conforming writer's local entry header
exactly `30 + len(name)` bytes. That fixed size is what makes the streaming
offsets of §6 computable analytically.

Every other entry is a lossless WebP texture.

### 2.1 Texture conventions

Normative, in order of what actually matters:

- **Splat `i` of the covered range lives at row-major texel `i`**, where `M` is
  the number of splats the texture covers: the whole file, or one group (§6).
  Every texture in a group MUST use the same dimensions.
- `width * height` MUST be at least `M`. **A player MUST take the dimensions
  from the WebP header and MUST NOT assume any particular width.**
- Textures SHOULD be **near-square**: `width = ceil(sqrt(M))`,
  `height = ceil(M / width)`. An encoder MAY round the dimensions up, for
  example to a multiple of 4, which is what PlayCanvas's own SOG writer does
  for texture-upload alignment. Because padding is at the tail of raster
  order, splat `i` is at flat texel `i` under either convention. The two
  conventions interoperate, and a decoder needs no special case. Two encoders
  that follow different conventions will produce archives of different sizes.
  That difference is expected, and it is why §10 compares decoded fields
  rather than bytes.
- Padding texels past `M` are unspecified. Encoders SHOULD write zero. Players
  MUST NOT read them.
- Every texture MUST be encoded **lossless** WebP. Every texel is a codebook
  index or a byte of a 16-bit integer, so one lossy pixel decodes to a wrong
  value.
- Encoders MUST set libwebp's `exact` flag. Without it, libwebp may rewrite the
  RGB of blocks that are entirely transparent, destroying data stored alongside
  a zero alpha.
- **A player MUST NOT depend on the RGB of any texel whose alpha is zero.**
  This is the complement of the rule above, and it bounds the damage when an
  encoder cannot comply. In the group set defined by §4, the only variable
  alpha is `sh0.webp`'s opacity, so the only reachable loss is the colour of
  splats that are fully transparent anyway. An encoder that cannot set `exact`
  is non-conforming on that clause, but it produces files that no conforming
  player can distinguish. (A prebuilt libwebp binding that exposes only the
  simple lossless API has no way to set it.) Any future group that stores
  meaningful data behind a zero alpha would turn that latent deviation into a
  real defect. That is why the requirement stays a MUST.

### 2.2 The 16-bit split-plane convention

The format stores positions, velocities and accelerations as 16-bit values
split across two textures. `*_l.webp` carries the low byte of each axis in R,
G, B. `*_u.webp` carries the high byte. The encoder applies a **log
transform** first, so precision follows magnitude:

```
encode:  T   = sign(x) * ln(1 + |x|)
         q   = round((T - mins[c]) / (maxs[c] - mins[c]) * 65535)   clamped to [0, 65535]
         lo  = q & 0xFF          hi = q >> 8

decode:  q   = (hi * 256 + lo) / 65535
         T   = mins[c] + q * (maxs[c] - mins[c])
         x   = sign(T) * (exp(|T|) - 1)
```

`mins` / `maxs` are per-axis and live in `meta.json`. When `maxs[c] == mins[c]`,
an encoder MUST use a span of 1.0 to avoid division by zero. The decode is then
constant at `mins[c]` regardless, so a player needs no special case.

Alpha in `*_l.webp` / `*_u.webp` is unused, and an encoder MUST write it as
255.

Velocity uses this scheme rather than a codebook deliberately: 256 levels
visibly quantizes motion.

## 3. `meta.json`

```jsonc
{
  "version": 1,
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
| `version` | yes | Always `1`. A player MUST reject any other value rather than guess. |
| `format` | yes | `"sogst"`. The container self-identifies here rather than relying on the file extension. A player MUST reject a file that lacks it or carries another value. |
| `asset.generator` | no | Free-form producer string. |
| `count` | yes | `N`, total splats in the file. |
| `time.min`, `time.max` | yes | Clip bounds in seconds. |
| `time.fps` | yes | Advisory, for UI frame counters and scrub granularity. It does **not** affect evaluation, because the model is continuous in `t`. |
| `cov2d_scale` | no | `[kx, ky]` screen-space 2D-covariance multiplier a player applies when rasterising. Compensates trainers whose screen-space footprints were inflated. Absent means `[1, 1]`. |

`files` arrays name the archive entries that back each group. In the streamed
layout (§6), they name the *basenames*, and the actual entries are prefixed. A
player SHOULD resolve textures through `files` rather than hardcode names.

### 3.1 Unknown keys and forward compatibility

A player MUST ignore manifest keys that it does not recognise. This applies to
top-level keys and to keys inside groups. A player MUST NOT reject a file
because the file carries an unknown key. A player MUST also ignore archive
entries that belong to a group it does not recognise. Only the `version` and
`format` checks above permit rejection.

The version number is the compatibility signal. The key set is not:

- A future revision MAY add new OPTIONAL groups and keys **under version 1**.
  The condition is the contract that §0 already imposes for `shN` and `accel`:
  a player that ignores the new keys must still render the file correctly at
  reduced fidelity. Temporally varying rotation is an example. It would arrive
  as a new coefficient group beside `motion`. Players that know the group
  evaluate it. To players that do not know it, the group is invisible.
- A change that alters the meaning of anything this revision specifies MUST
  bump `version`. This covers the evaluation semantics, every existing key,
  and every existing texture layout. Such a change MUST NOT be introduced
  under version 1.

An encoder SHOULD NOT emit private keys at the top level. If an encoder must
attach producer-specific data, the `asset` key is the place for it.

**An unknown entry may appear at any position in a group's byte order. It may
appear before that group's required entries.** This ordering is where
implementations of "ignore" go wrong. A streamed player that gates group
completion on a *count* of buffered entries fills the count early when an
unknown entry arrives before the group's last required file. It then decodes
the group one file short. The same entry after the known files causes no
error. Filter unrecognised names before you buffer or count. Judge completion
against the set of required names, never against a count of arrived entries.
The first player implementation of this section had exactly this defect. The
defect was order-dependent: a test that appended the unknown entry passed on
the broken player. Test with the unknown entry placed *before* the known
files (§9).

## 4. Attribute groups

### 4.1 `means` — `means_l.webp`, `means_u.webp`

Position at `t_center`, split-16 per §2.2 over `means.mins` / `means.maxs`.

### 4.2 `quats` — `quats.webp`

Rotation as a unit quaternion in **smallest-three** form. Let the quaternion be
`(w, x, y, z)`, normalised, with the sign chosen so its largest-magnitude
component is positive.

- **A** = `252 + i`, where `i` is the index **in `(w, x, y, z)` order** of the
  largest-magnitude component, which is the one that is dropped. So A is 252
  when `w` was dropped, 253 for `x`, 254 for `y`, 255 for `z`.
- **RGB** hold the remaining three components **in their original `(w, x, y, z)`
  relative order**: dropping `x` stores `(w, y, z)`, and dropping `w` stores
  `(x, y, z)`.
- Each stored component `s` lies in `[-1/sqrt(2), +1/sqrt(2)]` and is byte-coded
  as `round((s / sqrt(2) + 0.5) * 255)`. The decode is `(b/255 - 0.5) * sqrt(2)`.
- The dropped component is recovered as `sqrt(max(0, 1 - a^2 - b^2 - c^2))` and
  is non-negative by construction.

### 4.3 `scales` — `scales.webp`

RGB are indices into the shared 256-entry `scales.codebook`. Values are in
**natural-log space**, and a player applies `exp()`. Alpha is unused, and an
encoder writes 255.

### 4.4 `sh0` — `sh0.webp`

- **RGB**: indices into the shared 256-entry `sh0.codebook`, giving the raw SH
  DC coefficients `f_dc_0..2`. These are **not** RGB colour:
  `colour = 0.5 + C0 * f_dc` with `C0 = 0.28209479177387814`.
- **A**: **linear** opacity, `round(255 * sigmoid(opacity_logit))`. Note the
  asymmetry: the format stores everything else in the trainer's parameter
  space, but it stores opacity already activated and quantized to 8 bits.
  This is the peak opacity, reached at `t_center`.

### 4.5 `motion` — `motion_l.webp`, `motion_u.webp`

Linear velocity in scene units per second, split-16 per §2.2 over
`motion.mins` / `motion.maxs`. `motion.degree` is `1` or `2`. See §4.8.

### 4.6 `trbf` — `trbf.webp`

The temporal radial-basis window.

- **R**: index into `trbf.center.codebook` → `t_center` in seconds.
- **G**: index into `trbf.sigma.codebook` → `t_sigma` in seconds, `> 0`.
- **B**: unused, 0. **A**: unused, 255.

Both codebooks have 256 entries. The reference encoder clusters `t_sigma` in
the log domain, so precision follows relative rather than absolute error. The
codebook it emits is in linear space either way, so players are unaffected.

### 4.7 `shN` — `shN_centroids.webp`, `shN_labels.webp` (OPTIONAL)

Higher-order spherical harmonics, vector-quantized. Present only when
`meta.shN` is.

- `bands` ∈ {1, 2, 3}, giving `coeffs` ∈ {3, 8, 15} coefficients per colour
  channel and a centroid vector of `3 * coeffs` values.
- `shN_centroids.webp` width MUST be `64 * coeffs`: **192, 512 or 960** for 1,
  2 or 3 bands. A decoder infers the band count from this width, so it is
  normative, not incidental. Height is `ceil(count / 64)`.
- Palette entry `n` occupies `coeffs` consecutive texels starting at
  `u = (n % 64) * coeffs` on row `v = n / 64`. Texel `(u + k, v)` channel `j`
  holds an index into the shared 256-entry `shN.codebook` for the coefficient
  at **channel-major** position `j * coeffs + k`.
- `shN_labels.webp` **R** = low byte, **G** = high byte of the 16-bit palette
  entry id. B unused, A = 255. `shN.count` is the number of palette entries and
  MUST be ≤ 65536.

> **Known limitation of the reference decoder.** `decode_sogst_fields()` writes
> reconstructed coefficients into a fixed 45-wide `f_rest` array at stride 15
> regardless of `bands`, so its output layout is only correct for `bands == 3`.
> The reference encoder only ever emits `bands == 3`. New encoders SHOULD emit
> `bands == 3`. Players that implement `bands < 3` MUST use the channel-major
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
textures compress: spatially adjacent splats land in adjacent texels, and
their byte planes become locally smooth.

**The group table is normative. The order within a group is not.** A player
observes `[0, P)` plus a contiguous span of whole segments. So two
implementations must agree on which group each splat is in, and on where each
group's range begins and ends. The permutation *inside* a group is a
compression heuristic and nothing a player can distinguish. Two conforming
encoders will differ there as a matter of course. The quantizer scale, the
clamp, and the tie-break among splats that share a Morton code are all
unconstrained, and any one of them reshuffles every splat in the file.

So this specification does **not** define the Morton algorithm normatively,
and a validator MUST NOT judge an implementation non-conforming for producing
a different intra-group permutation. An encoder SHOULD use some
spatially-coherent ordering, because the file is materially larger without
one. §10 says how to compare two files given this.

A splat is **persistent** when its active interval
`[t_center - k_sigma*t_sigma, t_center + k_sigma*t_sigma]` is longer than
`persistent_span_mult * duration`. That is, the splat is visible across enough
of the clip that per-segment culling would not pay for itself. A player always
draws persistent splats. The encoder buckets the rest by `t_center` into
fixed-length segments.

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
- `t0` / `t1` are the *actual* time coverage of the segment's members (the min
  and max of their active intervals), **not** the bucket boundaries. They
  therefore overlap adjacent segments, and `t0` may precede `time.min`. For an
  empty segment they fall back to the nominal bucket bounds
  `[time.min + s*duration, time.min + (s+1)*duration]`.
- **A populated segment's `[t0, t1]` is the union of its members' active
  intervals**: `t0 = min(t_center - k_sigma*t_sigma)` and
  `t1 = max(t_center + k_sigma*t_sigma)` over the segment's splats. This is a
  definition, not an encoder preference, and two things follow from it that
  nothing else in this section needs to argue.

  **It is an equality on the encoder's source values, and only
  quantization-bounded when recomputed from a decoded archive.** Do not
  implement it as an exact conformance assertion, because it will fail on
  every valid file. `t_center` and `t_sigma` are codebook-quantized (§4), and
  the recomputed bound inherits `k_sigma` times that error. The discrepancy
  therefore scales with the segment's largest `t_sigma`, and it is per-file
  rather than a fixed tolerance. On `blocks_gap`, decoded, the worst departure
  across all populated segments is 4.5e-3 s against a 0.1 s `duration`. A
  validator that wants to check this should derive its tolerance from the
  file's own `t_sigma` codebook spacing, the way §10 derives every other
  tolerance.
- **`list` is ordered by segment index, and is NOT sorted by `t0`. A player
  MUST NOT binary-search it.** Because `t0` reaches back by `k_sigma*t_sigma`
  from the earliest member, a populated segment's `t0` lands *before* its own
  bucket start. So where an empty run precedes it, that populated `t0`
  precedes the synthetic bound of the empty segment before it. In
  `blocks_gap`, segment 12 is empty with `t0 = 1.200` while segment 13 begins
  at `t0 = 1.187`. Given a long enough empty run, this is inevitable, not
  merely possible. A binary search over `t0` silently returns the wrong
  segment on a conforming archive. Scan the list. It is bounded by 65536
  entries, and in practice by tens. Spans overlap for the same reason:
  several segments routinely contain the same `t`.
- **Do not "fix" this by clamping `t0`/`t1` to the bucket bounds.** The
  identity above settles it without measuring anything. Bucket bounds are
  strictly narrower than the support the encoder recorded, so clamping always
  cuts splats that are still on screen. Measurement only calibrates how bad
  it looks. On `blocks_gap` at `t = 1.249`, clamping drops **776 splats of
  segment 13 whose temporal factor exceeds 0.01, 342 of them above 0.1**, and
  the brightest is at 0.22. They pop. The overlap is not sloppiness in the
  segment table. It is what makes the table correct.
- The file records `persistent_span_mult` so that its ordering can be
  re-derived. Without it, no validator or re-packer can reproduce the
  persistent/dynamic split from the file alone. Players ignore it.
- **`list` has one entry per segment across the whole clip, whether or not that
  segment holds any splats**, so its length is `ceil((time.max - time.min) /
  duration)` and is set by `duration` alone. A very small `duration` on a long
  clip therefore produces a valid file whose `meta.json` is mostly an empty
  segment table. Encoders MUST NOT emit more than **65536** segments, and SHOULD
  reject a `duration` that would.

**The drawing rule.** At time `t`, a player draws `[0, P)` plus the single
contiguous index span covering every segment whose `[t0, t1]` contains `t`:

```
active = { s in segments.list : s.t0 <= t <= s.t1 }
if active is empty:  draw [0, P)
else:                draw [0, P)  and  [ min(s.range[0]), max(s.range[1]) )
```

Because segments are ordered by time and their coverage overlaps, the active
set is contiguous, so this is two draw ranges, never a scatter. Splats outside
the span have temporal opacity of roughly `exp(-k_sigma^2/2)` or less, so a
player culls them without visible error.

Segmentation is optional. When `meta.segments` is absent the file is a single
Morton-ordered block and a player draws all `N` splats at every `t`.

## 6. The streamed layout (OPTIONAL)

An archive MAY store one texture set per group instead of whole-clip textures,
so that a sequential download becomes progressively renderable. Quantization
stays **global**: codebooks, mins/maxs and shN centroids live in `meta.json`,
and every group shares them. So a group is decodable the moment its own bytes
arrive.

A writer emits entries in **play order**:

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
  `shN_centroids.webp`. A player starts DC-only playback and adds the
  view-dependent term as the trailing entries arrive.

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
  player needs before it can put anything on screen. That is the persistent
  group plus the first non-empty segment's geometry. A progress bar SHOULD
  fill against this, not against the whole file.
- **`geometry_bytes`** — the offset one past the last geometry entry, that is,
  where the deferred SH tail begins. A player uses it with measured bandwidth
  to decide whether to hold the playhead until gap-free playback is possible.

Both are absolute offsets from the start of the archive to the **local header
of the next entry, or, when no entry follows, the start of the central
directory**. That second case is not exotic: an archive with no `shN` group
has nothing after its geometry, so `geometry_bytes` lands exactly on the
central directory. On `blocks_gap`, that is offset 172404 with the file 180305
bytes long. A validator that checks these offsets against the set of entry
header offsets MUST include the central-directory start. It MUST take that
start from the **EOCD record**, not from a scan for the `PK\x01\x02`
signature. A scan
finds *a* directory header, not reliably the first one, and the difference is
invisible until an archive ends on this boundary.

The offsets are computable analytically precisely because entries carry no
extra fields (§2). A writer MUST verify its computed offsets against the
written file. Note that the values feed back into the size of the `meta.json`
that contains them, so a writer must iterate to a fixed point.

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
| `rot_0..rot_3` | Quaternion, **w first**: `rot_0 = w`. It need not be normalised. The encoder re-normalises and canonicalises sign. |
| `scale_0..scale_2` | **Natural-log** space. |
| `opacity` | **Logit** space. The *peak* opacity, at `t_center`. |
| `f_dc_0..f_dc_2` | Raw SH DC coefficients, **not** RGB. |
| `vx, vy, vz` | Linear velocity, scene units per second: the coefficient of `(t - t_center)`. |
| `t_center` | **Seconds**, absolute clip time. Not normalised. |
| `t_sigma` | **Standard deviation in seconds, not a variance.** MUST be `> 0`. |

Then, OPTIONAL:

| column(s) | convention |
|---|---|
| `f_rest_0..f_rest_44` | 45 values, **channel-major**: index `j*15 + k` is channel `j`, coefficient `k`. All 45 or none. |
| `ax, ay, az` | **Raw `dt^2` coefficient, not half-acceleration.** All three or none. See §4.8's gating rules, which apply equally to the PLY. |

All columns MUST have exactly `N` entries. Nothing downstream cross-checks
lengths, so a mismatch corrupts silently.

### 7.3 Clip-level scalars

`time_min`, `time_max`, `fps` and `cov2d_scale` are properties of the clip, not
of a splat, so they have no column. **The PLY carries them in comments**, which
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
- An encoder that finds them in neither the comments nor the sidecar (below)
  **MUST fail with an error**. It MUST NOT substitute a default. This is the
  single most likely place for two implementations to diverge, and the
  failure is silent. An assumed `fps = 30` on 24 fps content produces a file
  that plays at the wrong speed and renders perfectly while doing it.
- `sogst.motion_degree` is OPTIONAL and advisory. The presence of `ax, ay, az`
  is what actually determines the degree. When both are present they MUST
  agree.
- `sogst.cov2d_scale` is OPTIONAL. Absent means `1.0 1.0`.
- Unknown `sogst.*` comments MUST be ignored, not rejected.

A sidecar JSON file with the same keys MAY accompany the PLY, for toolchains
that cannot write PLY comments or that strip them. Its name is the PLY's path
with a trailing `.ply` **removed if present** and `.sogst.json` appended. So
`scene.ply` pairs with `scene.sogst.json`, not `scene.ply.sogst.json`. When
both are present the sidecar wins, because a toolchain that rewrites a PLY can
carry stale comments through while the sidecar is regenerated.

A conforming producer MUST supply the REQUIRED scalars in at least one of the
two carriers. It SHOULD write the comments when its PLY library supports them,
because the comments keep the interchange unit a single file. A producer that
cannot write comments MUST write the sidecar.

## 8. Relationship to TSOG (non-normative)

**TSOG** ("Temporally and Spatially Ordered Gaussians", Gmira, Alexiou,
Potetsianakis and Thomas, Xiaomi Technology Netherlands, arXiv:2607.28049, July
2026) is an independently developed extension of PlayCanvas SOG to 4D. It was
published while this specification was being finalised. Neither design derives
from the other. This section records how they relate, for anyone who
encounters both.

The two designs agree on everything SOG already decided, and on the obvious
next steps:

- Static attributes stay byte-compatible with SOG v2.
- Splats are Morton-ordered spatially.
- High-precision temporal quantities are 16-bit split-plane WebP pairs with
  min/max ranges in the manifest.
- Motion is a per-splat polynomial with a first-order (velocity) base case.
- Temporal opacity is a FreeTimeGS-style Gaussian window. TSOG's "center and
  scale" timeline type is this format's `trbf.center`/`trbf.sigma`.

The two designs differ in scope. TSOG defines a generic parameterisation
scheme: any attribute may gain `temporal_<attribute>_<order>` coefficient
images, including polynomial rotation, and a discrete frame-id timeline mode
exists beside the continuous one. But TSOG ships as loose WebP files plus
JSON. It has no container, no streaming layout, and no temporal segmentation
or culling. It quantizes motion coefficients linearly, without a log
transform. It states no functional form and no normalisation for the temporal
opacity window. It has no conformance rules and no validation procedure.
`.sogst` fixes one normative model and specifies it completely: the ZIP
container and streamed layout (§§2, 6), segment-based temporal culling (§5),
the log transform (§2.2), exact evaluation semantics (§1), and the conformance
and cross-implementation machinery (§§9–10).

The manifests are mutually incompatible. TSOG uses the `timeline` and
`temporal.*` keys. This format uses `motion`, `accel`, `trbf`, and
`segments`. A TSOG asset is not a `.sogst` archive (it is not an archive at
all). Capabilities that TSOG has
and this format lacks, temporally varying rotation foremost, are additive by
design. They would land through §3.1 as new OPTIONAL groups, not through
adoption of TSOG's naming.

## 9. Conformance checklist

An encoder conforms when:

- [ ] `meta.json` is the first entry. All entries are `ZIP_STORED` with no
      extra fields.
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
- [ ] It rejects any file that is not a ZIP with `meta.version == 1` and
      `meta.format == "sogst"` (there is no legacy form to accept). It
      names the offending value in the error.
- [ ] It decodes a file whose manifest carries an unrecognised key (top-level
      or inside a group) identically to the same file without that key (§3.1).
      When you test the streamed path, place the unknown archive entry
      *before* each group's required entries. A count-gated group-completion
      bug is order-dependent: a fixture that appends the unknown entry passes
      on a player that is still broken. "Identically" means compare the decode
      or the rendered output against the unmodified file. "Loads without
      error" does not show that the entries were ignored.

      Running this check against a real player exposed two traps:

      - **A durable full-file cache can serve the fixture from a path the
        test does not target.** A warm cache hit decodes the whole-clip path,
        and that result proves nothing about the streamed path a fix touched.
        Guarantee a cold load: use a fresh filename for each fixture. A fresh
        filename is cheaper and more reliable than clearing the store,
        because deletion of the store can block on the page's own open
        connection. Confirm the load path from the server's request log, not
        from in-page instruments, which reset across navigation.
      - **You cannot make a streamed fixture by inserting an entry into an
        existing archive.** `streams.reveal_bytes` and `geometry_bytes` are
        absolute offsets, and `meta.json` is the first entry. An inserted
        entry, or a resized manifest alone, shifts every entry after it, and
        the load fails for reasons unrelated to entry tolerance. Rebuild the
        fixture through a writer that recomputes and verifies the offsets
        (§6).
- [ ] It reaches that rejection from the manifest rather than from a
      downstream parse failure. An application that dispatches on file
      extension routes an unrecognised file to whatever its default loader
      is. The observed failure mode is a full download of a
      several-hundred-megabyte asset, followed by a confusing error from an
      unrelated parser, or by no visible error at all.
      Deciding from `meta.json`, which is the first entry and so arrives in
      the first range request, fails in the first few kilobytes with an
      accurate message.

## 10. Cross-implementation validation

Two implementations of one contract diverge silently. The check that catches it:
pack the *same* reference PLY with both, decode both with `decode_sogst_fields()`,
and assert per-field maximum absolute error within quantization tolerance.

**Compare decoded fields, not rendered PSNR.** Codebook initialisation is
implementation-defined, so byte equality is the wrong bar. PSNR is too coarse
to tell you *which* field is wrong. A per-field error table localises the bug
immediately: a wrong `f_rest` stride and a wrong quaternion mode mapping look
identical in a PSNR number, and nothing alike in a field table.

`scripts/compare_sogst.py` implements this:

```
python compare_sogst.py --a python.sogst --b typescript.sogst   # the real check
python compare_sogst.py --a scene.ply    --b scene.sogst        # one pack's cost
```

Four things about how it judges, because the naive version of each is wrong:

- **Split-plane fields (positions, velocity, accel) allow no outliers.** Their
  encoding is fully determined by `mins`/`maxs`, so any disagreement is a bug.
- **The tool judges codebook fields (scales, `f_dc`, `t_center`, `t_sigma`)
  on the *fraction* of splats past tolerance, not on the worst one.** A value
  sitting on a
  bin boundary can legitimately land in different bins in two conforming
  encoders. The bugs worth catching move essentially every splat, not 0.5% of
  them.
- **The tool judges `t_sigma` by the temporal weight it produces, not by its
  value in seconds.** A sigma much longer than the clip is saturated: two
  such values can differ by tens of seconds and be pixel-identical. Judging
  raw seconds reports a large error for splats a renderer cannot tell apart.
- **A coordinate exactly on a quantization boundary is an alignment problem,
  not an encoding one.** The tool pairs splats by their 16-bit split-plane
  integers, and a value on a boundary rounds one way in a float32 encoder and
  the other way in a float64 source. If that one step reorders the splat, the
  tool pairs it with a *neighbour* (a different splat), and then every field
  reports a large error from a conforming archive. Two splats in 8,192
  produced six field failures and a spurious membership divergence. The tool
  re-pairs such splats by position **within their own group**, tolerates a
  one-step key difference, and stops doing either above 0.2% of the file. The
  bound is what keeps it a test. A wrong persistence predicate or a mis-ported
  bucketing misplaces splats by the thousand (three deliberately displaced
  splats still fail loudly), while boundary rounding touches a handful.

`f_rest` gets a scale-relative bar rather than an absolute one. VQ placement
is implementation-defined, but a wrong coefficient layout compares unrelated
coefficients, and so it produces a mean error near the data's own standard
deviation. On the parabola fixture the separation is 0.000 (correct) against
0.743 (channel/coefficient transpose).

**Ordering needs care, and the naive comparison is badly wrong here.** Splat
order splits into a normative part and a free part (§5), and a field comparison
has to respect the split:

- The tool compares the **group table** directly, and a mismatch is a hard
  failure that stops the run. Every field comparison downstream of a wrong
  range table is meaningless.
- The tool then sorts each group into a **canonical, encoder-independent
  order** before it compares fields. Sort by the 16-bit split-plane *integers*
  for position, with velocity to break ties. Those encodings are fully
  determined by `meta`, so two conforming encoders produce identical keys. Use
  the integers rather than the decoded floats. When one side is an unquantized
  PLY, two splats closer together than a quantization step could otherwise
  sort one way before quantization and the other way after.
- The tool then compares **group membership** (which splats ended up in each
  range) exactly, on position keys alone. This is the check that catches a
  wrong persistent predicate or wrong `t_center` bucketing. It survives the
  realignment because a splat sorted into the wrong group has no counterpart
  where it landed.
- The tool **reports** the intra-group permutation distance and **never
  fails** on it.

A comparison of raw index order instead reports a conforming encoder as broken
in every respect: 100% of splats displaced, and every field over tolerance.
The misalignment scrambles quaternions and SH along with everything else, so
the diagnostics for a wrong quaternion mode mapping and a wrong `f_rest`
stride both fire at correct code. That was this document's own tooling on its
first encounter with a second implementation, and that is why the guidance is
here.

## Appendix A. Revision history

**Revision 14** — review feedback on the first public draft.

- §7.3: the clip-scalar carrier requirement is restated. A producer MUST
  supply the REQUIRED scalars in at least one carrier (PLY comments or the
  sidecar), SHOULD prefer the comments, and MUST write the sidecar when it
  cannot write comments. Revision 13 and earlier required the comments
  unconditionally, which contradicted the sidecar's stated purpose. The
  sidecar-wins precedence is unchanged, and an encoder still MUST fail when
  it finds the scalars in neither carrier.
- The old §8 (development history) is deleted. It documented formats that
  were never released, for readers who were present during development. Its
  one normative point, that a player rejects anything that is not a ZIP with
  `meta.version == 1` and `meta.format == "sogst"`, already lives in §3 and
  §9. "Relationship to TSOG" is renumbered from §8.1 to §8. Older Appendix
  entries keep their original section numbers.
- The license header no longer editorializes, and the §7.3 filename example
  is neutral.

**Revision 13** — editorial only. **No implementer action.** This revision
rewrites the whole document into Simplified Technical English structure:
shorter sentences, active voice, no semicolons in prose. No rule changed, and
no normative statement changed meaning. A diff against revision 12 shows
wording changes only.

**Revision 12** — §9 testing notes, from re-verifying revision 11's exemplar.
**No implementer action** on the format. It records two traps for anyone who
runs the §3.1 check against a real player.

- Revision 11 cites a pixel-identical verification as the model. That
  verification had a hazard in its own provenance and had to be
  re-established. The player carries a durable full-file cache, and a cache
  hit decodes the whole-clip path, which was already conformant. A warm
  post-fix PASS would therefore have proven nothing about the streamed code
  the fix touched. The pre-fix FAIL was provably streamed: the stack trace
  said so. The post-fix PASS proved nothing until it ran again cold. The cold
  run used fresh filenames to force fresh cache keys, and it used the static
  server's request log as the arbiter of which path executed. The result
  held. §9 now records the cold-load guarantee and the arbiter choice.
- §9 now also warns that a streamed unknown-entry fixture must come from a
  writer, never from inserting an entry into an existing archive. The §6
  offsets are absolute and `meta.json` is the first entry, so an inserted
  entry, or a resized manifest alone, makes the load fail for reasons
  unrelated to what the fixture tests. The reference suite's fixtures already
  rebuild through `write_sogst_streamed`. The new text states why that is the
  only valid construction.

**Revision 11** — the first player implementation of §3.1 hardened it, the
same day.

- The viewer's §3.1 check found a real, order-dependent violation in the
  viewer's own streamed path. The path gated group completion on a *count* of
  buffered entries. An unknown entry that arrived before a group's last
  required file filled the count early, and the decode ran one file short.
  The same entry appended after the known files was skipped without error.
  §3.1 now states the failure mode and the rule: judge completion against the
  set of required names, never against a count. §9's bullet now requires the
  unknown-entry fixture to place the entry *before* the known files, because
  the appending fixture passed on the broken player. The reference suite's
  own streamed fixture had the appending shape. That shape is harmless
  against the reference decoder, which reads entries by name, but it is wrong
  as a template. The fixture moved to first-in-group.
- §9's bullet also now says what "identically" requires: compare the decode
  or the rendered output against the unmodified file. "Loads without error"
  tolerates the entries. It does not establish that the entries were ignored.
  The viewer's verification compared rendered pixels. The reference suite
  compares decoded fields.

**Revision 10** — forward compatibility, prompted by the publication of TSOG.

- §3.1 (new): a player MUST ignore unrecognised manifest keys and the archive
  entries that back unrecognised groups. Rejection stays reserved for
  `version`/`format`. Additive OPTIONAL groups may arrive under version 1.
  Anything that changes the meaning of existing content must bump `version`.
  Without this rule, a strictly written version-1 player could legitimately
  reject any future additive extension. That rejection would force a version
  bump, and a hard reject from every deployed player, for changes that are
  safe to ignore. §9 gains the matching checklist bullet.
- §8.1 (new, non-normative): the relationship to TSOG (arXiv:2607.28049), an
  independently developed SOG 4D extension published 2026-07-30. The two
  designs converge on the texture conventions. They diverge on packaging,
  streaming, segmentation, quantization transform, and conformance rigor. The
  manifests are incompatible. TSOG's temporally varying rotation is the
  worked example §3.1 exists for.

No change to the payload or to what a conforming encoder emits.

**Revision 9** — §6 wording fix, found by a validator it misled.

- §6 said `reveal_bytes` and `geometry_bytes` are offsets "to the local header
  of the next entry". That is false whenever no entry follows: an archive with
  no `shN` group has nothing after its geometry, so `geometry_bytes` lands on
  the **start of the central directory**. §6 now says so, and warns that a
  validator must read that offset from the EOCD record rather than scan for
  the `PK\x01\x02` signature. A scan finds *a* directory header, not reliably
  the first, and nothing reveals the difference until an archive ends on this
  boundary. Both `blocks_gap` archives failed an independent structural
  checker identically because of this. That identical failure is what
  identified it as the checker's bug rather than the encoders'.

**Revision 8** — tooling and fixtures. **No implementer action**, but the
validation hole is worth knowing about if you have been trusting a PASS.

- **`compare_sogst.py` could not detect a segmentation disagreement between an
  archive and its own source PLY.** A PLY carries no group table, so the tool
  derived the PLY's ordering from the archive's parameters and then *adopted
  the archive's group table for both sides*. That made the one thing it most
  needed to check structurally invisible. It now derives the PLY's own table
  and compares it. This is not hypothetical: it passed a fixture whose `.ply`
  and `.sogst` put 1,218 splats in different segments.
- **That fixture is fixed, and the cause generalises.** A gap bound is
  normally a multiple of `duration`, so folding splats onto it parks a plateau
  exactly on a bucket boundary. `t/duration` buckets one way at float64 and
  the other way after a float32 PLY round-trip, which moves the plateau
  between adjacent segments. Any producer that computes a bucket index on one
  side of a PLY write and compares against the other side can hit this.
  `make_sogst_fixture.py` now nudges folded values clear of the boundary *and*
  packs from the written PLY rather than from memory. The pair is therefore
  consistent by construction.

**Revision 7** — §5 only, and it is worth reading if you wrote a player.

- §5 now states the **support-bound identity**: a populated segment's
  `[t0, t1]` is the union of its members' active intervals. It was always what
  the encoder wrote. Saying it makes the rest of this entry follow instead of
  needing argument. Revision 8 added the caveat that makes it safe to
  implement: the identity is exact on source values and only
  quantization-bounded when recomputed from a decoded archive, because
  `t_center`/`t_sigma` are codebook-quantized and the recomputed bound
  inherits `k_sigma` times that error. Stated as an exact equality it would
  fail on every conforming file.
- **`segments.list` is not sorted by `t0`, and a player MUST NOT binary-search
  it.** Revision 6 and earlier said spans overlap and that empty segments fall
  back to nominal bucket bounds, but never drew the conclusion. Because `t0`
  reaches back by `k_sigma*t_sigma`, a populated segment that follows a long
  enough empty run *necessarily* reports an earlier `t0` than the synthetic
  bound before it. Binary search then returns the wrong segment on a
  conforming file. The `blocks_gap` fixture found this: segment 12 is empty at
  `t0 = 1.200`, and segment 13 begins at `t0 = 1.187`. No encoder change was
  needed. This was always the behaviour.
- §5 rejects the obvious repair. Clamping to bucket bounds is wrong by the
  identity alone, because bucket bounds are narrower than the recorded
  support. The measurement only sizes it: 776 splats of segment 13 above a
  0.01 temporal factor at `t = 1.249`, and 342 above 0.1. The overlap is what
  makes the table correct.

  The first published version of this entry said 44 splats. That number came
  from a count scoped to a segment membership that existed only in the
  fixture's PLY and not in its archive. The two disagreed, for the reason in
  the revision-8 entry below. 44 is small enough to argue that the pop is
  tolerable. 776 is not.

**Revision 6** — tooling and fixtures only. **No implementer action:** nothing
here changes what a conforming encoder writes or a conforming player reads.

- §10 gains the boundary-rounding rule. Pairing splats by their split-plane
  integers breaks when a coordinate sits exactly on a boundary, because a
  float32 encoder and a float64 source round it opposite ways. The tool now
  re-pairs those within their group, and it bounds itself at 0.2% so a real
  membership bug still fails.
- A new `blocks_gap` fixture. `parabola_gap` shows a player does not break on
  an empty segment run, but it cannot show the player *culled* one: an
  expanding cloud looks much the same whether or not culling happens.
  `blocks_gap` puts each segment in its own spatially separated cluster, so
  the drawn population is countable. This came from the player implementer,
  who declined to claim coverage they did not have. That report is the kind
  that makes a fixture worth building.

**Revision 5** — from the first player implementation of revision 4. No change
to the bytes. §9 only.

- §9 requires a player to reach its rejection *from the manifest* and to name
  the offending value. Revision 4 deleted the development-era formats but said
  nothing about what a consumer does when handed one. The answer turned out to
  be worse than rejecting: an extension-dispatching application routes the
  file to its default loader, downloads all several hundred megabytes of it,
  and then fails inside a parser that has no idea what it was given. In the
  observed case there was no user-visible error at all. Deleting a format
  leaves that failure mode behind. The fix belongs in the consumer's dispatch,
  not in per-format knowledge the deletion was meant to remove.
- §§4.8/10 corrected `decode_v3_fields()` to `decode_sogst_fields()`, missed in
  the revision-4 rename.

**Revision 4** — the development-era formats are gone, and the container is
renumbered from 3 to **1**.

- §3 `version` is `1` and `format` is `"sogst"`. Both are REQUIRED, and a
  player MUST reject anything else. Revision 3 kept `version: 3` and tolerated
  a missing `format` to protect deployed assets. There are none, so that
  tolerance only widened the surface a second implementation had to get right.
- §2 no longer mentions the `OMG4` magic: the binary containers it identified
  have been deleted, not deprecated.
- §8 is now a history note rather than a compatibility section, and records how
  already-baked archives were migrated by manifest rewrite rather than re-bake.
- §9's player checklist replaces "accepts the legacy forms" with "rejects
  anything that is not version 1".

No change to the payload. A revision-3 archive and a revision-4 archive of the
same scene differ only in `meta.json`.

**Revision 3** — from cross-validating the two implementations on a real
75,848-splat asset.

- §5 now states outright that **the group table is normative and the order
  within a group is not**, and declines to specify the Morton algorithm. The two
  implementations agree exactly on group membership and ranges while differing
  in every intra-group position, because their Morton quantizer scale and
  tie-breaking differ. Revision 2 left §5 ("an encoder-side quality choice") and
  §10 ("splat order is part of the format") in direct contradiction.
- §10 replaces index-by-index comparison with: compare the group table, align
  canonically within groups, compare membership exactly, report the intra-group
  permutation as information.

**Revision 2** — everything here came from the first independent implementation
of revision 1 (the TypeScript encoder), which is what a spec revision is for.
No change alters the bytes a revision-1-conforming encoder produces.

- §2 forbids ZIP **data descriptors** explicitly. Revision 1 forbade extra
  fields, which a streaming ZIP writer satisfies while still emitting
  descriptors and zeroed local-header sizes. That defeats the byte-rangeability
  the rule existed to protect, and it shifts every §6 offset by 16 bytes per
  entry.
- §2.1 no longer pins texture dimensions. Splat `i` at row-major texel `i` is
  the normative part. Near-square is a SHOULD, dimension roundup (PlayCanvas's
  SOG writer aligns to a multiple of 4) is explicitly permitted, and players
  must read dimensions from the WebP header.
- §2.1 adds the complement to the `exact` requirement: a player MUST NOT depend
  on the RGB of a zero-alpha texel. This bounds the exposure of an encoder that
  cannot set `exact` to the colour of fully-transparent splats.
- §5 bounds `segments.list` at 65536 entries. Every segment gets an entry
  whether or not it holds splats, so a small `duration` on a long clip was a
  valid file with a pathological `meta.json`.
- §7.3 pins the sidecar filename: `scene.ply` → `scene.sogst.json`.
