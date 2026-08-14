#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""compare_sogst.py - cross-implementation equivalence check for .sogst.

Two encoders implementing one specification diverge silently.  This script
is the check that catches it: decode two files with the reference decoder
and report per-field absolute error against a tolerance derived from each
file's own quantization parameters.

    # the important one: same PLY packed by Python and by TypeScript
    python compare_sogst.py --a python.sogst --b typescript.sogst

    # quantization cost of a single pack, against the unquantized source
    python compare_sogst.py --a scene.ply --b scene.sogst

Either side may be a .sogst/.omg4 archive or a 4D interchange PLY.

**Compare decoded fields, not rendered PSNR** (docs/sogst-format.md
section 10).  Codebook initialisation is implementation-defined, so byte
equality is the wrong bar; PSNR is too coarse to localise which field is
wrong.  A wrong f_rest stride and a wrong quaternion mode mapping look
identical in a PSNR number and nothing alike in a per-field table.

**Only part of splat ordering is observable, and the tool checks exactly
that part.**  A player draws [0, P) plus a contiguous span of whole
segments, so what is normative is the group table and which group each
splat lands in.  The permutation *within* a group is Morton ordering -- a
compression heuristic, explicitly not normative (section 5) -- and two
conforming encoders differ there routinely: a different quantizer scale or
a different tie-break among splats sharing a Morton code reshuffles every
splat while changing nothing a player can see.

So the group table is compared directly and a mismatch is a hard failure,
while each group is sorted into a canonical, encoder-independent order
before fields are compared.  Group *membership* is still checked, by what
survives that realignment: a splat sorted into the wrong group has no
counterpart where it landed, so its position error stays large.  The
intra-group permutation distance is reported as information, never as a
failure.
"""

import argparse
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sogst_io import SOGST_FIELDS  # noqa: E402

SCALAR_FIELDS = list(SOGST_FIELDS)
ACCEL_FIELDS = ['ax', 'ay', 'az']
QUAT_FIELDS = ['rot_0', 'rot_1', 'rot_2', 'rot_3']

# Fields that survive a pack unquantized are compared exactly; everything
# else gets a tolerance derived from the encoding, below.
SPLIT16_GROUPS = {
    'means': ['x', 'y', 'z'],
    'motion': ['vx', 'vy', 'vz'],
    'accel': ACCEL_FIELDS,
}


def load(path):
    """Decode a .sogst/.omg4 archive or an interchange PLY into (meta, fields).

    `meta` is the archive's meta.json, or None for a PLY (which carries no
    quantization parameters -- it is the unquantized side).
    """
    if zipfile.is_zipfile(path):
        from eval_render import decode_sogst_fields
        _header, fields = decode_sogst_fields(path)
        with zipfile.ZipFile(path) as zf:
            meta = json.loads(zf.read('meta.json'))
        return meta, fields
    if path.endswith('.ply'):
        from sogst_ply import read_sogst_ply
        _header, fields = read_sogst_ply(path, require_scalars=False)
        return None, fields
    raise ValueError(f'{path}: expected a .sogst archive or a 4D interchange .ply')


def split16_tolerance(meta, group, axis, values):
    """Worst-case decode step for a split-16 log-transformed axis.

    One 16-bit step in the log domain becomes a value-dependent step in
    linear space (d/dT of sign(T)*(exp(|T|)-1) is exp(|T|)), so the
    tolerance is per-element, not a constant.
    """
    if meta is None or group not in meta:
        return None
    mins = np.asarray(meta[group]['mins'], dtype=np.float64)
    maxs = np.asarray(meta[group]['maxs'], dtype=np.float64)
    step = (maxs[axis] - mins[axis]) / 65535.0
    # The float32 PLY -> float64 meta -> log/exp round trip costs a few ulp,
    # and a constant-valued axis has mins == maxs and therefore a step of
    # exactly zero. Without this floor such an axis can never pass.
    epsilon = 1e-6 * np.abs(values) + 1e-9
    return step * (1.0 + np.abs(values)) + epsilon    # exp(|T|) == 1 + |value|


def codebook_tolerance(meta, path):
    """Half the largest gap between adjacent codebook entries -- the worst
    quantization error a 256-entry codebook can produce."""
    if meta is None:
        return None
    node = meta
    for key in path:
        if key not in node:
            return None
        node = node[key]
    cb = np.sort(np.asarray(node, dtype=np.float64))
    return float(np.max(np.diff(cb)) / 2.0) if cb.size > 1 else 0.0


def field_tolerance(meta_a, meta_b, name, values):
    """Combined tolerance for one field: each side may quantize once, so the
    two contributions add.  Returns None when no bound is derivable."""
    parts = []
    for meta in (meta_a, meta_b):
        if meta is None:
            continue
        tol = None
        for group, names in SPLIT16_GROUPS.items():
            if name in names:
                tol = split16_tolerance(meta, group, names.index(name), values)
        if name.startswith('scale_'):
            tol = codebook_tolerance(meta, ('scales', 'codebook'))
        elif name.startswith('f_dc_'):
            tol = codebook_tolerance(meta, ('sh0', 'codebook'))
        elif name == 't_center':
            tol = codebook_tolerance(meta, ('trbf', 'center', 'codebook'))
        elif name == 't_sigma':
            # Reported, but not the pass/fail bar -- see the dedicated
            # temporal-window check in compare(). A sigma far longer than
            # the clip is saturated: the raw error can be tens of seconds
            # while the rendered difference is nil.
            tol = None
        elif name == 'opacity':
            # stored as an 8-bit LINEAR alpha, so the logit-space error
            # blows up near the tails; bound it in linear space instead
            tol = None
        if tol is not None:
            parts.append(tol)
    if not parts:
        return None
    total = parts[0]
    for extra in parts[1:]:
        total = total + extra
    return total


# Fraction of splats allowed past tolerance on a codebook-quantized field.
# Codebook construction is implementation-defined (spec section 10), so a
# value sitting near a bin boundary can legitimately land in a different bin
# in two conforming encoders.  That affects a handful of splats; the bugs
# this tool exists to catch -- a wrong f_rest stride, a wrong quaternion
# mode mapping, a wrong log transform -- affect essentially all of them.
CODEBOOK_OUTLIER_FRACTION = 0.01

# Ceiling on the share of splats repair_boundary_pairings will re-pair. Set
# well below any real bug's blast radius: boundary rounding hits a handful of
# splats (2 in 8,192 on the block fixture), while a mis-ported bucketing or
# persistence predicate misplaces them by the thousand. Above this the repair
# stands down and the membership check fails, which is the point.
ALIGNMENT_REPAIR_LIMIT = 0.002

CODEBOOK_FIELDS = ('scale_', 'f_dc_', 't_center', 't_sigma')

# f_rest mean error as a fraction of the data's own standard deviation.
# Two conforming encoders differ only by VQ centroid placement and land
# a couple of percent in; a wrong coefficient layout compares unrelated
# coefficients and lands near 1. Measured separation on the parabola
# fixture: 0.000 correct vs 0.743 for a channel/coefficient transpose.
SH_LAYOUT_RATIO = 0.25


def is_codebook_field(name):
    return any(name.startswith(prefix) for prefix in CODEBOOK_FIELDS)


def judge(name, err, tol, codebook=False):
    """Score one field. Returns a row tuple for the report table.

    Deterministic encodings (the 16-bit split planes) are judged on the
    worst splat.  Codebook encodings are judged on the fraction of splats
    past tolerance, because bin assignment at a boundary is
    implementation-defined and a max-based bar produces false alarms.
    """
    err = np.asarray(err, dtype=np.float64)
    if tol is None:
        return (name, err, None, 'advisory', True)
    over = err > np.asarray(tol) * 1.001 + 1e-9
    fraction = float(over.mean())
    if codebook:
        ok = fraction <= CODEBOOK_OUTLIER_FRACTION
        stat = f'{fraction:.3%} over'
    else:
        ok = not bool(over.any())
        stat = 'max' if ok else f'{fraction:.3%} over'
    return (name, err, tol, stat, ok)


def reorder_source_like(fields, meta):
    """Apply an archive's splat ordering to unordered source fields.

    A PLY is in producer order; an archive is in [persistent | segments]
    order.  Comparing the two index-by-index without this reports every
    field as wrong.  The ordering is re-derivable because the archive
    records the three parameters that determine it -- duration, k_sigma and
    persistent_span_mult (which is recorded for exactly this reason; see
    docs/sogst-format.md section 5).

    Returns (reordered_fields, derived_meta).  The derived segment table is
    the PLY's *own* grouping under the archive's parameters, and the caller
    MUST compare it against the archive's rather than assuming they agree.
    They can disagree for real: a t_center sitting exactly on a bucket
    boundary buckets one way at float64 and the other after a float32 PLY
    round-trip, which silently moves a whole plateau of splats between two
    adjacent segments.
    """
    from sogst_pack import compute_sogst_order

    segments = (meta or {}).get('segments')
    time = (meta or {}).get('time', {})
    if not segments:
        # No segmentation: the archive is a single Morton-ordered block.
        order, derived = compute_sogst_order(
            fields, time.get('min', 0.0), time.get('max', 0.0), segment_duration=0)
    else:
        order, derived = compute_sogst_order(
            fields, time.get('min', 0.0), time.get('max', 0.0),
            segment_duration=segments['duration'],
            k_sigma=segments.get('k_sigma', 3.8),
            persistent_span_mult=segments.get('persistent_span_mult', 3.0))
    reordered = {k: (v[order] if v.ndim == 1 else v[order, :]) for k, v in fields.items()}
    return reordered, {'segments': derived, 'time': time}


def group_ranges(meta, n):
    """The half-open index ranges a player can actually distinguish.

    A player draws [0, P) plus a contiguous span of whole segments, so what
    is observable about splat order is which group a splat is in, and where
    each group's range starts and ends.  The permutation *within* a group is
    a compression heuristic (Morton), explicitly not normative -- see
    docs/sogst-format.md section 5.
    """
    segments = (meta or {}).get('segments')
    if not segments:
        return [('all', 0, n)]
    groups = []
    p0, p1 = segments['persistent']
    if p1 > p0:
        groups.append(('persistent', p0, p1))
    for i, seg in enumerate(segments['list']):
        lo, hi = seg['range']
        if hi > lo:
            groups.append((f'seg_{i:03d}', lo, hi))
    return groups


# Fields quantized by a scheme that is fully determined by meta (the 16-bit
# split planes), so two conforming encoders produce identical values and
# they can be used as a sort key.  Codebook fields cannot: their values
# depend on a clustering that is implementation-defined.
CANONICAL_KEY_FIELDS = ['x', 'y', 'z', 'vx', 'vy', 'vz']


def canonical_keys(fields, meta):
    """Sort keys that are identical on both sides regardless of encoder.

    Built from the 16-bit split-plane *integers* rather than the decoded
    floats.  That matters when one side is an unquantized PLY: two splats
    whose positions differ by less than a quantization step can sort one way
    before quantization and the other way after, which would misalign them.
    Projecting both sides onto the same integer grid removes that entirely
    (the encode is idempotent on already-decoded values, so an archive side
    reproduces exactly the integers it was stored with).

    Falls back to raw floats when no meta is available (PLY against PLY).
    """
    keys = []
    for group, names in (('means', ['x', 'y', 'z']), ('motion', ['vx', 'vy', 'vz'])):
        params = (meta or {}).get(group)
        for axis, name in enumerate(names):
            if name not in fields:
                continue
            values = np.asarray(fields[name], np.float64)
            if not params:
                keys.append(values)
                continue
            mins = float(np.asarray(params['mins'])[axis])
            maxs = float(np.asarray(params['maxs'])[axis])
            span = (maxs - mins) if maxs > mins else 1.0
            t = np.sign(values) * np.log1p(np.abs(values))
            keys.append(np.clip(np.rint((t - mins) / span * 65535.0), 0, 65535))
    return keys


def canonical_permutation(fields, groups, meta):
    """Permutation putting each group into a canonical, encoder-independent
    order, so two files can be compared splat-for-splat.

    Returns (permutation, ambiguous) where `ambiguous` counts splats sharing
    a complete key with a neighbour, for which the pairing is a coin flip.
    Both sides MUST be keyed with the same meta.
    """
    keys = canonical_keys(fields, meta)
    order = np.empty(len(fields['x']), dtype=np.int64)
    ambiguous = 0
    for _label, lo, hi in groups:
        local = np.lexsort(tuple(reversed([k[lo:hi] for k in keys])))
        order[lo:hi] = local + lo
        if hi - lo > 1:
            stacked = np.stack([k[lo:hi][local] for k in keys], axis=1)
            same = np.all(stacked[1:] == stacked[:-1], axis=1)
            ambiguous += int(same.sum())
    return order, ambiguous


def apply_permutation(fields, order):
    return {k: (v[order] if v.ndim == 1 else v[order, :]) for k, v in fields.items()}


def repair_boundary_pairings(a, b, keys_a, keys_b, groups, limit_frac):
    """Re-pair the few splats whose alignment key rounds differently on the
    two sides, and report how many.

    Exact-key alignment has one failure mode it cannot fix on its own: a
    coordinate sitting exactly on a quantization boundary rounds one way in a
    float32 encoder and the other way in a float64 PLY, so the two sides
    disagree by a single step on one axis.  That single step moves the splat
    in the lexsort, which pairs it against a *neighbour* -- and the neighbour
    is a different splat, so every field then reports a large error.  Two
    splats in 8,192 produced six field failures and a spurious membership
    divergence before this existed.

    So: where the sorted keys disagree, re-pair those rows among themselves
    by nearest decoded position, within a group (a splat may not change
    group -- that is the thing membership is testing, and repairing across
    groups would erase the very divergence we are looking for).

    Deliberately bounded.  Above `limit_frac` this does nothing and lets the
    membership check fail, because a real bug -- a wrong persistence
    predicate, a mis-ported bucketing -- misplaces splats in bulk, and
    silently repairing that would turn the tool into one that always passes.
    """
    mismatched = ~np.all(keys_a == keys_b, axis=1)
    count = int(mismatched.sum())
    n = len(mismatched)
    none_repaired = np.zeros(n, dtype=bool)
    if not count or count > max(1, int(limit_frac * n)):
        return b, 0, none_repaired

    pa = np.stack([np.asarray(a[k], np.float64) for k in ('x', 'y', 'z')], axis=1)
    pb = np.stack([np.asarray(b[k], np.float64) for k in ('x', 'y', 'z')], axis=1)
    order = np.arange(n)
    fixed = np.zeros(n, dtype=bool)
    repaired = 0
    for _label, lo, hi in groups:
        idx = np.nonzero(mismatched[lo:hi])[0] + lo
        if len(idx) < 2:
            continue                    # nothing in-group to swap with
        free = list(idx)
        for i in idx:
            j = min(free, key=lambda c: float(np.sum((pa[i] - pb[c]) ** 2)))
            order[i] = j
            free.remove(j)
            fixed[i] = True
            repaired += 1
    if not repaired:
        return b, 0, none_repaired
    return apply_permutation(b, order), repaired, fixed


def compare(path_a, path_b, verbose=False):
    meta_a, a = load(path_a)
    meta_b, b = load(path_b)

    # A PLY is unordered source; an archive carries the packing order. Put
    # the PLY through the archive's ordering so the comparison is
    # index-aligned. With two archives, both are already ordered and any
    # divergence is a real finding.
    derived_a = derived_b = None
    if meta_a is None and meta_b is not None:
        a, derived_a = reorder_source_like(a, meta_b)
        print('(A is source PLY: reordered using B\'s recorded segment parameters)')
    elif meta_b is None and meta_a is not None:
        b, derived_b = reorder_source_like(b, meta_a)
        print('(B is source PLY: reordered using A\'s recorded segment parameters)')

    n_a, n_b = len(a['x']), len(b['x'])
    print(f'A: {path_a}  ({n_a:,} splats)')
    print(f'B: {path_b}  ({n_b:,} splats)')
    if n_a != n_b:
        print(f'\nFAIL: splat counts differ ({n_a:,} vs {n_b:,}). Nothing further '
              'is comparable -- check pruning thresholds and whether both sides '
              'packed the same source.')
        return 1
    print()

    failures = []
    rows = []

    # -- grouping, which IS normative, before anything else ----------------
    # A player draws [0, P) plus a contiguous span of whole segments. So the
    # observable part of splat order is the group table and which group each
    # splat lands in -- not the permutation within a group, which is a
    # compression heuristic (section 5). Check the former; align away the
    # latter.
    # A PLY carries no group table of its own, so reorder_source_like derives
    # one by re-running the ordering under the archive's parameters. Compare
    # that, do NOT adopt the archive's: assuming they agree makes a
    # segmentation disagreement between an archive and its own source
    # structurally invisible, which is how a fixture whose .ply and .sogst
    # put 1,218 splats in different segments passed this tool.
    eff_a = meta_a or derived_a
    eff_b = meta_b or derived_b
    groups_a = group_ranges(eff_a, n_a)
    groups_b = group_ranges(eff_b, n_b)
    if eff_a and eff_b:
        seg_a, seg_b = eff_a.get('segments'), eff_b.get('segments')
        if bool(seg_a) != bool(seg_b):
            print(f'FAIL: segmentation present in only one file '
                  f'({"A" if seg_a else "B"})')
            return 1
        if [g[1:] for g in groups_a] != [g[1:] for g in groups_b]:
            print('FAIL: group index ranges differ -- this is the finding, and it '
                  'makes every field comparison downstream meaningless.')
            if seg_a and seg_a['persistent'] != seg_b['persistent']:
                print(f"  persistent: {seg_a['persistent']} vs {seg_b['persistent']} "
                      '(check the persistent predicate and persistent_span_mult)')
            if seg_a and len(seg_a['list']) != len(seg_b['list']):
                print(f"  segment count: {len(seg_a['list'])} vs {len(seg_b['list'])} "
                      '(check duration and the clip bounds)')
            else:
                bad = [i for i, (x, y) in enumerate(zip(groups_a, groups_b))
                       if x[1:] != y[1:]]
                if bad:
                    print(f'  first differing group {groups_a[bad[0]][0]}: '
                          f'{groups_a[bad[0]][1:]} vs {groups_b[bad[0]][1:]} '
                          '(ranges are HALF-OPEN, [first, last) -- section 5)')
            return 1
        print(f'Groups: {len(groups_a)} '
              f'({"persistent + " if groups_a and groups_a[0][0] == "persistent" else ""}'
              f'{len(groups_a) - (1 if groups_a and groups_a[0][0] == "persistent" else 0)} '
              'segments) -- index ranges identical')

    # -- align within groups -----------------------------------------------
    # Morton ordering is an encoder-side compression choice and two
    # conforming implementations legitimately produce different permutations
    # (different quantizer scale, different tie-breaking). Comparing raw
    # index order reports that as every field being wrong. Sort each group
    # canonically instead, by split-plane fields, which are bit-identical
    # between conforming encoders precisely because their encoding is fully
    # determined by meta.
    key_meta = meta_a or meta_b          # both sides MUST use the same grid
    perm_a, amb_a = canonical_permutation(a, groups_a, key_meta)
    perm_b, amb_b = canonical_permutation(b, groups_b, key_meta)
    displaced = int(np.count_nonzero(perm_a != perm_b))
    a = apply_permutation(a, perm_a)
    b = apply_permutation(b, perm_b)
    ambiguous = max(amb_a, amb_b)
    if ambiguous:
        print(f'NOTE: {ambiguous:,} splat(s) share a complete position+velocity key '
              'with a neighbour, so their pairing is arbitrary. Field errors for '
              'those are not meaningful.')
    print(f'Intra-group order: {displaced:,} of {n_a:,} splats in a different '
          f'position ({displaced / max(n_a, 1):.1%}) -- informational. The Morton '
          'permutation within a group is not observable by a player (section 5); '
          'only group membership and the range table are, and both are checked '
          'above.\n')

    # Group membership -- which IS observable, since a player culls whole
    # groups. Compared exactly rather than by a threshold: both sides are
    # now sorted by the same integer grid, so a group holding the same
    # splats has an identical key matrix. Anything else means the two
    # encoders put different splats in the same range.
    # Position only. Velocity earns its place in the sort key (it breaks
    # ties) but not in this test: a splat whose velocity was encoded wrongly
    # is the same splat in the same group, and reporting it as a membership
    # difference would point at the wrong bug.
    keys_a = np.stack(canonical_keys(a, key_meta)[:3], axis=1)
    keys_b = np.stack(canonical_keys(b, key_meta)[:3], axis=1)
    b, repaired, repaired_mask = repair_boundary_pairings(
        a, b, keys_a, keys_b, groups_a, ALIGNMENT_REPAIR_LIMIT)
    if repaired:
        keys_b = np.stack(canonical_keys(b, key_meta)[:3], axis=1)

    # A boundary-rounded splat shows up in one of two ways. If the one-step
    # difference reordered it against a neighbour, the repair above already
    # re-paired the two. If it did not reorder anything, the pairing was
    # correct all along and only the key differs -- that is this mask.
    #
    # Tolerating one step is safe for the question being asked. A splat that
    # genuinely landed in the wrong group sits somewhere else in the scene,
    # thousands of steps away, not one. Two splats within a single step of
    # each other are co-located to the limit of what the format can express,
    # so confusing them is not an error a player could render.
    delta = np.abs(keys_a - keys_b).max(axis=1)
    rounded = (delta > 0) & (delta <= 1)
    if repaired or rounded.any():
        near = int(rounded.sum())
        print(f'NOTE: {repaired + near:,} of {n_a:,} splats '
              f'({(repaired + near) / max(n_a, 1):.3%}) have an alignment key '
              'that rounds differently on the two sides -- a coordinate sitting '
              'exactly on a quantization boundary, which a float32 encoder and a '
              f'float64 source resolve opposite ways. {repaired:,} had been '
              f'reordered against a neighbour and were re-paired by position '
              f'within their own group; {near:,} were already paired correctly. '
              'Both kinds are excluded from the membership check below -- the '
              're-paired ones were only ever matched inside their own group, and '
              'the rest sit one step from their partner in the same group -- so '
              'their membership is settled either way. An artifact of the '
              'comparison, not an encoder disagreement.\n')
    mismatched = ~np.all(keys_a == keys_b, axis=1) & ~repaired_mask & ~rounded
    if mismatched.any():
        offenders = [(label, int(mismatched[lo:hi].sum()), hi - lo)
                     for label, lo, hi in groups_a if mismatched[lo:hi].any()]
        total = int(mismatched.sum())
        print(f'GROUP MEMBERSHIP DIVERGES: {total:,} of {n_a:,} splats '
              f'({total / n_a:.2%}) are in a different group, across '
              f'{len(offenders)} group(s).')
        for label, bad, size in offenders[:5]:
            print(f'    {label}: {bad:,} of {size:,}')
        if len(offenders) > 5:
            print(f'    ... and {len(offenders) - 5} more')
        print('  The range table matches but the ranges hold different splats, so a '
              'player culling by segment would draw the wrong ones. Check the '
              'persistent predicate and the t_center bucketing in the '
              'compute_sogst_order port.')
        if ambiguous:
            print(f'  ({ambiguous:,} splat(s) had ambiguous keys, so a few of these '
                  'may be pairing noise rather than real membership differences.)')
        print()
        failures.append('group membership')

    for name in SCALAR_FIELDS + ACCEL_FIELDS:
        if name not in a or name not in b:
            if (name in a) != (name in b):
                print(f'FAIL: {name} present in only one file '
                      f'({"A" if name in a else "B"}) -- accel is all-or-nothing')
                failures.append(name)
            continue
        if name in QUAT_FIELDS:
            continue
        va = np.asarray(a[name], np.float64)
        vb = np.asarray(b[name], np.float64)
        err = np.abs(va - vb)
        tol = field_tolerance(meta_a, meta_b, name, np.maximum(np.abs(va), np.abs(vb)))
        if name == 'opacity':
            # compare where it is actually stored: linear alpha, 8-bit
            la = 1.0 / (1.0 + np.exp(-va))
            lb = 1.0 / (1.0 + np.exp(-vb))
            err = np.abs(la - lb)
            tol = 2.0 / 255.0
            name = 'opacity(linear)'
        if name == 't_sigma':
            name = 't_sigma (advisory)'
        rows.append(judge(name, err, tol, codebook=is_codebook_field(name)))
        if not rows[-1][-1]:
            failures.append(name)

    # t_sigma judged by what it does rather than what it is: the temporal
    # weight it produces at each splat's own worst-case dt within the clip.
    # Two sigmas well beyond the clip length are indistinguishable to a
    # renderer, and a raw-seconds comparison flags them as a large error.
    if 't_sigma' in a and 't_sigma' in b:
        time = (meta_a or meta_b or {}).get('time', {})
        t0, t1 = float(time.get('min', 0.0)), float(time.get('max', 0.0))
        if t1 > t0:
            tc = np.asarray(a['t_center'], np.float64)
            dt = np.maximum(np.abs(tc - t0), np.abs(t1 - tc))
            sa = np.maximum(np.asarray(a['t_sigma'], np.float64), 1e-9)
            sb = np.maximum(np.asarray(b['t_sigma'], np.float64), 1e-9)
            err = np.abs(np.exp(-0.5 * (dt / sa) ** 2) - np.exp(-0.5 * (dt / sb) ** 2))
            # one 8-bit alpha step; codebook-judged, since t_sigma is
            # codebook-quantized and bin assignment is implementation-defined
            rows.append(judge('t_sigma (as alpha)', err, 2.0 / 255.0, codebook=True))
            if not rows[-1][-1]:
                failures.append('t_sigma')

    # Quaternions: compare the rotation, not the components. Sign and the
    # choice of dropped component are both free.
    if all(c in a and c in b for c in QUAT_FIELDS):
        qa = np.stack([np.asarray(a[c], np.float64) for c in QUAT_FIELDS], axis=1)
        qb = np.stack([np.asarray(b[c], np.float64) for c in QUAT_FIELDS], axis=1)
        qa /= np.maximum(np.linalg.norm(qa, axis=1, keepdims=True), 1e-12)
        qb /= np.maximum(np.linalg.norm(qb, axis=1, keepdims=True), 1e-12)
        # angle between rotations, in degrees
        dot = np.clip(np.abs((qa * qb).sum(axis=1)), 0.0, 1.0)
        angle = np.degrees(2.0 * np.arccos(dot))
        rows.append(judge('rotation (deg)', angle, 2.0))
        if not rows[-1][-1]:
            failures.append('rotation')
            print('NOTE: a large rotation error with everything else passing usually '
                  'means the smallest-three mode mapping is wrong -- spec section 4.2. '
                  'Mode byte 252 is a dropped W, and the kept components stay in '
                  '(w,x,y,z) relative order.\n')

    # Higher-order SH. VQ centroid initialisation is implementation-defined,
    # so an absolute tolerance would produce false alarms -- but the failure
    # this actually has to catch is a wrong coefficient layout (section 4.7:
    # channel-major with stride `coeffs`, not stride 15), and that is
    # separable from VQ noise by scale. Two conforming encoders land within
    # a few percent of the data's own spread; a transposed layout lands at
    # a large fraction of it, because it is comparing unrelated
    # coefficients.
    if 'f_rest' in a and 'f_rest' in b:
        fa = np.asarray(a['f_rest'], np.float64)
        fb = np.asarray(b['f_rest'], np.float64)
        spread = float(np.std(fa))
        err = np.abs(fa - fb)
        if spread > 0:
            ratio = float(err.mean()) / spread
            ok = ratio <= SH_LAYOUT_RATIO
            # Show measured against threshold explicitly. Printing the ratio
            # alone under a "past tol" heading reads as a derived tolerance,
            # which it is not -- the threshold is the fixed SH_LAYOUT_RATIO.
            rows.append(('f_rest (of spread)', err, None,
                         f'{ratio:.3f}/{SH_LAYOUT_RATIO:.2f}', ok))
            if not ok:
                failures.append('f_rest')
                print(f'NOTE: f_rest mean error is {ratio:.0%} of the data\'s own spread -- '
                      'far beyond VQ noise. That signature is a coefficient LAYOUT bug, '
                      'not a quantization difference: spec section 4.7 requires '
                      'channel-major order, index j*coeffs + k.\n')
        else:
            rows.append(judge('f_rest (VQ, advisory)', err, None))
    elif ('f_rest' in a) != ('f_rest' in b):
        print(f'NOTE: f_rest present only in {"A" if "f_rest" in a else "B"} '
              '(one side stripped SH?)')

    width = max(len(r[0]) for r in rows)
    print(f"{'field':{width}s} {'mean':>12s} {'p99':>12s} {'max':>12s} "
          f"{'tolerance':>12s} {'measure':>12s}  result")
    for name, err, tol, stat, ok in rows:
        tol_max = '' if tol is None else f'{np.max(tol):12.6f}'
        verdict = 'advisory' if stat == 'advisory' else ('ok' if ok else 'FAIL')
        print(f'{name:{width}s} {err.mean():12.6f} {np.percentile(err, 99):12.6f} '
              f'{err.max():12.6f} {tol_max:>12s} {stat:>12s}  {verdict}')
    print(f'\n(codebook-quantized fields allow up to '
          f'{CODEBOOK_OUTLIER_FRACTION:.0%} of splats past tolerance: bin assignment '
          'at a boundary is implementation-defined. Split-plane fields allow none.)')

    if meta_a and meta_b:
        for key in ('format', 'version', 'count'):
            if meta_a.get(key) != meta_b.get(key):
                print(f'NOTE: meta.{key} differs: {meta_a.get(key)!r} vs {meta_b.get(key)!r}')

    print()
    if failures:
        print(f'FAIL: {len(failures)} check(s) outside tolerance: {failures}')
        return 1
    print('PASS: group table identical, every field within the tolerance implied '
          'by its encoding.')
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', required=True, help='Reference .sogst/.omg4 or .ply')
    parser.add_argument('--b', required=True, help='Candidate .sogst/.omg4 or .ply')
    args = parser.parse_args()
    sys.exit(compare(args.a, args.b))


if __name__ == '__main__':
    main()
