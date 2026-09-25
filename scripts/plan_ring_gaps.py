#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""plan_ring_gaps.py - where a real rig needs 4DAnyone's help to reach a
target camera configuration.

WHY
---
A real rig covers only part of the azimuth circle (one measured rig: -71 to
+149 degrees, 220 of 360). Earlier work (build_hybrid_dataset.py) generated
one full 4DAnyone orbit per run and culled the views that turned out to
duplicate a real camera. This script does the same geometry the other
direction: given a target rig-spec (camera_rig_spec.py's format) and the
real rig's own solved poses, compute the azimuth gaps BEFORE generating
anything, so 4DAnyone is only asked for the views actually missing.

4DAnyone's own geometry primitive is one continuous partial-span ring per
run (views_per_layer, layer_pitches, start_yaw, yaw_span -- confirmed
against the 4DAnyone repo itself, fdanyone/geometry/cameras.py). There is
no per-camera multi-arc mode, so several disjoint gaps need several runs.
This script's output is the plan for those runs, not the runs themselves.

AZIMUTH-ZERO ALIGNMENT: A REAL TRAP
------------------------------------
rig_geometry.orbit_basis() defines azimuth 0 as "toward the first real
camera in transforms.json" -- arbitrary. A rig-spec's ring azimuths are
defined relative to its OWN azimuth_offset_deg=0 reference (the scene's
+X axis), which has no necessary relationship to the real rig's azimuth
zero. --front_azimuth_deg names which real-rig azimuth the target spec's
azimuth zero corresponds to physically. Get this wrong and every computed
gap is confidently wrong in the same wrong direction. Defaults to 0.0 with
a loud warning, since a silent default here is worse than an explicit one.

WHAT A RIG-SPEC'S RADIUS/HEIGHT DO NOT AFFECT
-----------------------------------------------
A ring's azimuth (azimuth_offset_deg / azimuth_span_deg / azimuth_centre_deg)
never depends on radius, height, or subject size -- only on the ring's own
angular parameters (see camera_rig_spec.py's _ring_angles). This script
reads ONLY azimuth, so a rig-spec that expresses radius/height as
subject-relative values (e.g. ring16.json's {"subject_heights": 1.7})
resolves fine against a synthetic placeholder manifest: the placeholder's
actual size is irrelevant to the angles this script extracts. No manifest
is required unless the spec's TARGET point itself is subject-relative
(e.g. "subject_center") and the caller wants that centre to be a specific
real-world point rather than an arbitrary placeholder -- again irrelevant
here, since gap-finding only cares about azimuth relative to whatever
centre the spec resolves to.

WHAT REMAINS AN OPEN ASSUMPTION, NOT A VERIFIED FACT
-------------------------------------------------------
4DAnyone's `start_yaw` is documented as "0 faces the person," which reads
as relative to the SOURCE camera's own facing direction for that run, not
a global azimuth. This script's RunPlan.start_yaw is computed as the gap's
azimuth extent relative to the chosen anchor camera's own azimuth, on that
reading. This has NOT been verified against 4DAnyone's actual framing-
analysis code (fdanyone/skeleton/pipeline.py's analyze_input_framing) and
should be checked against a real run before trusting the generated
geometry lands where this script intends.

conda env: cumuli (pure numpy + camera_rig_spec.py + rig_geometry.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_rig_spec as crs  # noqa: E402
import rig_geometry as rg  # noqa: E402

# Matches build_hybrid_dataset.py --min_separation_deg's measured value:
# redundant-view culling and gap-finding are the same boundary computed
# from opposite sides, so they should never disagree about where it is.
DEFAULT_MIN_SEPARATION_DEG = 20.0


# fdanyone's documented layer_pitches range.
PITCH_MIN_DEG, PITCH_MAX_DEG = -15.0, 45.0

# A placeholder subject bbox, used only so a rig-spec's subject-relative
# radius/height resolve without a real scene manifest. Its size does not
# affect anything this script reads (see the module docstring).
_PLACEHOLDER_MANIFEST = {"subject_bbox": {"min": [-0.3, -0.3, 0.0], "max": [0.3, 0.3, 1.7]}}


@dataclass(frozen=True)
class GapArc:
    """A contiguous run of target azimuths (real-rig frame, degrees) with no
    real camera within min_separation_deg."""
    start_deg: float
    end_deg: float
    center_deg: float
    span_deg: float
    target_azimuths_deg: tuple[float, ...]
    pitch_deg: float


@dataclass(frozen=True)
class RunPlan:
    """One 4DAnyone invocation: fills exactly one GapArc from one anchor."""
    anchor_camera: str
    views_per_layer: int
    layer_pitches: tuple[float, ...]
    start_yaw: float
    yaw_span: float
    target_azimuths_deg: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "anchor_camera": self.anchor_camera,
            "views_per_layer": self.views_per_layer,
            "layer_pitches": list(self.layer_pitches),
            "start_yaw": self.start_yaw,
            "yaw_span": self.yaw_span,
            "target_azimuths_deg": list(self.target_azimuths_deg),
        }


def real_rig_azimuths(real_transforms_path: Path):
    """Real camera azimuths (deg), using the SAME up/target/e1/e2 definition
    build_hybrid_dataset.py's culling pass uses (SVD plane-normal of camera
    centres, not rig_geometry.load_rig's own per-camera-up average), so
    gap-finding and culling agree about where the boundary is.

    Returns (az_by_label: dict[str, float], target: np.ndarray, up: np.ndarray).
    """
    cameras, _load_rig_up, _focal, _width = rg.load_rig(real_transforms_path)
    centers = np.stack([c["center"] for c in cameras])
    target = centers.mean(0)
    _, _, vt = np.linalg.svd(centers - target)
    up = vt[-1]
    if up[1] > 0:
        up = -up
    e1, e2 = rg.orbit_basis(cameras, target, up)
    az = {c["label"]: rg.spherical(c["center"], target, up, e1, e2)[0] for c in cameras}
    return az, target, up


def target_rig_by_ring(rig_spec_path: Path, manifest: dict | None = None):
    """Target-spec cameras grouped by ring/layer index, each camera's azimuth
    in the SPEC's OWN frame (azimuth_offset_deg=0 -> 0 here) and its pitch in
    degrees (elevation above the horizontal through the spec's own centre).

    Grouping by `.ring` (camera_rig_spec.RigCamera's ring/row index) is what
    keeps a staggered multi-ring spec like ring16.json from being treated as
    one flat, meaningless azimuth sequence: each ring becomes its own gap
    search and its own generation layer. A `cage` layout's row index and an
    `explicit` layout's single implicit group both fall out of the same
    field, so this needs no per-layout special-casing.

    Returns {ring_index_or_None: [(label, azimuth_deg, pitch_deg), ...]}.
    """
    spec = crs.load_spec(rig_spec_path)
    resolved = crs.resolve_rig(spec, manifest=manifest or _PLACEHOLDER_MANIFEST)
    centre = np.asarray(resolved["centre"], dtype=np.float64)

    groups: dict = {}
    for cam in resolved["train"]:
        offset = cam.position - centre
        radius = float(np.linalg.norm(offset[:2]))  # horizontal (Blender XY) distance
        az = math.degrees(math.atan2(offset[1], offset[0]))
        pitch = math.degrees(math.atan2(offset[2], radius)) if radius > 1e-9 else 90.0
        groups.setdefault(cam.ring, []).append((cam.label, az, pitch))
    return groups


def find_gap_arcs(target_entries, real_az_deg: dict, front_azimuth_deg: float,
                   min_separation_deg: float) -> list[GapArc]:
    """Contiguous runs of target azimuths (one ring/layer's worth, already in
    the SPEC's own frame) whose distance to the nearest real azimuth exceeds
    min_separation_deg, expressed in the REAL RIG's frame.

    `target_entries`: [(label, azimuth_deg_spec_frame, pitch_deg), ...] for
    one ring, as returned by target_rig_by_ring (one dict value). Sorted by
    azimuth internally, then walked as a ring (wrapping at +/-180) to find
    contiguous uncovered runs -- a rings-layout spec already produces evenly
    stepped azimuths, so sorting recovers ring order regardless of how the
    spec listed them.
    """
    if not target_entries:
        return []
    real_azs = list(real_az_deg.values())
    ordered = sorted(target_entries, key=lambda e: e[1])
    n = len(ordered)
    covered = []
    for _, az_spec, _ in ordered:
        az_real = az_spec + front_azimuth_deg
        nearest = min(abs(rg.shortest_arc(az_real, ra)) for ra in real_azs)
        covered.append(nearest < min_separation_deg)

    if all(covered):
        return []
    if not any(covered):
        # The whole ring is a gap: one arc spanning the full circle. Break it
        # at index 0 rather than reporting a zero-width start==end arc.
        start_idx, indices = 0, list(range(n))
    else:
        # Rotate so the walk starts right after a covered point, which makes
        # a gap that straddles the array's wrap point (e.g. spans indices
        # n-1 -> 0) appear as one contiguous run instead of two fragments.
        start_idx = next(i for i in range(n) if covered[i] and not covered[(i + 1) % n])
        indices = [(start_idx + 1 + k) % n for k in range(n)]

    arcs = []
    run: list[int] = []
    for i in indices:
        if not covered[i]:
            run.append(i)
        elif run:
            arcs.append(run)
            run = []
    if run:
        arcs.append(run)

    out = []
    for run_idx in arcs:
        labels = [ordered[i][0] for i in run_idx]
        azs_real = [ordered[i][1] + front_azimuth_deg for i in run_idx]
        pitches = [ordered[i][2] for i in run_idx]
        # Unwrap so start/end/center are continuous across the +/-180 seam.
        unwrapped = [azs_real[0]]
        for a in azs_real[1:]:
            unwrapped.append(unwrapped[-1] + rg.shortest_arc(unwrapped[-1], a))
        out.append(GapArc(
            start_deg=unwrapped[0], end_deg=unwrapped[-1],
            center_deg=(unwrapped[0] + unwrapped[-1]) / 2.0,
            span_deg=abs(unwrapped[-1] - unwrapped[0]),
            target_azimuths_deg=tuple(azs_real),
            pitch_deg=float(np.mean(pitches)),
        ))
    return out


def choose_anchor_camera(gap: GapArc, real_az_deg: dict) -> str:
    """The real camera whose azimuth is closest to the gap's centre -- the
    real camera bordering this gap, not literally one run per real camera.
    A real camera bordering no gap gets no companion run."""
    return min(real_az_deg, key=lambda label: abs(rg.shortest_arc(gap.center_deg, real_az_deg[label])))


def round_views_per_layer(n_targets: int) -> int:
    """Smallest value >= n_targets that 4DAnyone accepts (divides 4 or 6)."""
    n_targets = max(1, n_targets)
    m4 = math.ceil(n_targets / 4) * 4
    m6 = math.ceil(n_targets / 6) * 6
    return min(m4, m6)


def plan_generation_runs(real_transforms: Path, rig_spec: Path, *,
                          front_azimuth_deg: float = 0.0,
                          min_separation_deg: float = DEFAULT_MIN_SEPARATION_DEG,
                          manifest: dict | None = None) -> list[RunPlan]:
    """The end-to-end plan: real rig + target rig-spec -> one RunPlan per
    (gap, ring/layer). See the module docstring for the azimuth-zero and
    start_yaw caveats -- this function does not silently work around either."""
    real_az, _target, _up = real_rig_azimuths(real_transforms)
    by_ring = target_rig_by_ring(rig_spec, manifest)

    plans = []
    out_of_range = []
    for ring_index, entries in sorted(by_ring.items(), key=lambda kv: (kv[0] is None, kv[0])):
        gaps = find_gap_arcs(entries, real_az, front_azimuth_deg, min_separation_deg)
        for gap in gaps:
            if not (PITCH_MIN_DEG <= gap.pitch_deg <= PITCH_MAX_DEG):
                out_of_range.append((ring_index, gap))
                continue
            anchor = choose_anchor_camera(gap, real_az)
            anchor_az = real_az[anchor]
            # WORKING ASSUMPTION (see module docstring): start_yaw=0 faces
            # the anchor camera's own direction, so the gap's azimuth extent
            # is expressed relative to the anchor's azimuth, not the global
            # frame. Verify against a real run before trusting this.
            start_yaw = ((gap.start_deg - anchor_az) + 180.0) % 360.0 - 180.0
            plans.append(RunPlan(
                anchor_camera=anchor,
                views_per_layer=round_views_per_layer(len(gap.target_azimuths_deg)),
                layer_pitches=(round(gap.pitch_deg, 1),),
                start_yaw=round(start_yaw, 2),
                yaw_span=round(min(gap.span_deg, 360.0), 2),
                target_azimuths_deg=gap.target_azimuths_deg,
            ))
    if out_of_range:
        print(f"WARNING: {len(out_of_range)} gap(s) fall outside 4DAnyone's "
              f"layer_pitches range [{PITCH_MIN_DEG}, {PITCH_MAX_DEG}] deg and "
              "were skipped (not planned): "
              + ", ".join(f"ring {r} pitch {g.pitch_deg:.1f}" for r, g in out_of_range),
              file=sys.stderr)
    return plans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_transforms", required=True, type=Path,
                    help="the real rig's solved transforms.json (e.g. transforms_train.json)")
    ap.add_argument("--rig_spec", required=True, type=Path,
                    help="target camera_rig_spec.py JSON, e.g. configs/rigs/ring16.json")
    ap.add_argument("--front_azimuth_deg", type=float, default=0.0,
                    help="the real-rig azimuth (rig_geometry's frame, toward the first real "
                         "camera) that the rig-spec's own azimuth zero points at physically. "
                         "Defaults to 0.0 -- almost certainly wrong for a real capture; see "
                         "this script's module docstring before trusting the default.")
    ap.add_argument("--min_separation_deg", type=float, default=DEFAULT_MIN_SEPARATION_DEG,
                    help="target azimuths within this many degrees of a real camera are "
                         "considered already covered (matches build_hybrid_dataset.py's flag "
                         "of the same name)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="optional scene manifest JSON (subject_bbox), for a spec whose "
                         "TARGET point is subject-relative. Not needed for azimuth alone; "
                         "see the module docstring.")
    ap.add_argument("--out_json", type=Path, default=None,
                    help="write the plan here; defaults to stdout")
    args = ap.parse_args()

    if args.front_azimuth_deg == 0.0:
        print("WARNING: --front_azimuth_deg defaults to 0.0, which assumes the rig-spec's "
              "azimuth zero already points the same physical direction as the real rig's "
              "first camera. This is almost never true by accident -- see the "
              "AZIMUTH-ZERO ALIGNMENT section of this script's module docstring.",
              file=sys.stderr)

    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    plans = plan_generation_runs(
        args.real_transforms, args.rig_spec,
        front_azimuth_deg=args.front_azimuth_deg,
        min_separation_deg=args.min_separation_deg,
        manifest=manifest)

    payload = [p.to_dict() for p in plans]
    text = json.dumps(payload, indent=1)
    if args.out_json:
        args.out_json.write_text(text)
        print(f"wrote {len(plans)} run(s) to {args.out_json}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
