#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""fit_rate_target.py - prune an interchange PLY until its .sogst fits a delivery rate.

A rate target is a delivery value (what a web or headset player can stream),
not a scene value, so it sits after training rather than inside it.  The
splats are ranked by contribution: each one's largest total blend weight, in
pixels, over the training views at their own timestamps (the measure the
trainer's export_min_contribution uses).  The kept count is then searched by
packing for real and measuring, until the archive is at most
--fill (default 0.97) of the target.  Splat count drives size almost
linearly, so a secant step converges in two or three packs.

Writes <out>.ply (the pruned interchange PLY, the input to a fine-tune of the
kept splats) and <out>.sogst.  Pruning alone strips half-transparent surface
splats and lets darker inner splats show through, so a short fine-tune of the
kept set with densification off is the intended next step (cumuli-trainer
with init_ply = <out>.ply, strategy cap_max = the kept count and
refine_start past the last iteration).

Usage:
    python fit_rate_target.py --input final.ply --transforms dataset/transforms_train.json \
        --target_mbps 1.0 --out runs/x/rate_1mbps [--every 4] [--means-drop-bits 1 ...]

--target_mb gives an absolute size instead of a rate.  The rate is over the
clip duration in the PLY header: bytes / (time_max - time_min).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cumuli_core_path  # noqa: E402,F401
from cumuli_core.cameras import load_transforms  # noqa: E402
from cumuli_core.ply import read_sogst_ply, write_sogst_ply  # noqa: E402

from sogst_pack import compute_sogst_order, pack_sogst  # noqa: E402


def contribution(fields, header, cams, device="cuda"):
    """[N] max over views of each splat's total blend weight in pixels."""
    import torch
    from gsplat import rasterization
    from cumuli_core.render import camera_tensors
    from cumuli_core.spacetime import from_fields

    st = from_fields(fields, device)
    best = torch.zeros(st.xyz.shape[0], device=device)
    for cam in cams:
        means, alpha = st.slice(header["time_min"] + cam.time)
        vm, K = camera_tensors([cam], device)
        w = torch.ones(means.shape[0], 1, device=device, requires_grad=True)
        img, _, _ = rasterization(means.detach(), st.quats_wxyz, st.scales, alpha.detach(), w,
                                  vm, K, cam.width, cam.height, sh_degree=None)
        img.sum().backward()
        best = torch.maximum(best, w.grad[:, 0].detach())
    return best.cpu().numpy()


def subset(fields, keep):
    return {k: (v[keep] if v.ndim == 1 else v[keep, :]) for k, v in fields.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="interchange PLY")
    ap.add_argument("--transforms", required=True, help="transforms_train.json (OpenGL c2w + time)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--target_mbps", type=float, help="delivery rate in MB/s of clip time")
    g.add_argument("--target_mb", type=float, help="absolute archive size in MB")
    ap.add_argument("--out", required=True, help="output stem: writes <out>.ply and <out>.sogst")
    ap.add_argument("--fill", type=float, default=0.97, help="aim at this fraction of the target")
    ap.add_argument("--every", type=int, default=1, help="score every Nth training view")
    ap.add_argument("--max_packs", type=int, default=6)
    ap.add_argument("--shn_count", type=int, default=65536)
    for group in ("means", "motion", "accel"):
        ap.add_argument(f"--{group}-drop-bits", type=int, default=0, choices=range(9))
    a = ap.parse_args()

    header, fields = read_sogst_ply(a.input)
    n = header["count"]
    duration = header["time_max"] - header["time_min"]
    target = (a.target_mb if a.target_mb else a.target_mbps * duration) * 1e6
    aim = a.fill * target
    drop = {"means": a.means_drop_bits, "motion": a.motion_drop_bits, "accel": a.accel_drop_bits}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sogst_path = a.out + ".sogst"

    cams = load_transforms(a.transforms, 1.0, a.every)
    score = contribution(fields, header, cams)
    rank = np.argsort(-score, kind="stable")
    print(f"{n:,} splats over {duration:.3f} s; target {target / 1e6:.3f} MB, aiming at {aim / 1e6:.3f} MB; "
          f"scored on {len(cams)} views; contribution p50 {np.median(score):.2f} px")

    def pack(k):
        keep = np.sort(rank[:k])
        sub = subset(fields, keep)
        order = compute_sogst_order(sub, header["time_min"], header["time_max"], 0.1)
        pack_sogst(sogst_path, sub, header["time_min"], header["time_max"], header["fps"],
                   shn_count=a.shn_count, cov2d_scale=header.get("cov2d_scale"),
                   order_segments=order, drop_bits=drop)
        size = os.path.getsize(sogst_path)
        print(f"  keep {k:,} ({k / n:.1%}): {size / 1e6:.3f} MB")
        return keep, size

    history = []
    k = n
    keep, size = pack(k)
    history.append((k, size))
    if size <= target:
        print("already fits; nothing pruned")
    else:
        k = int(n * aim / size)
        for _ in range(a.max_packs - 1):
            keep, size = pack(k)
            history.append((k, size))
            if aim * 0.985 <= size <= target:
                break
            # secant on bytes-per-splat through the two most recent packs
            (k0, s0), (k1, s1) = history[-2], history[-1]
            slope = (s1 - s0) / (k1 - k0) if k1 != k0 else s1 / k1
            k = int(np.clip(k1 + (aim - s1) / slope, 1, n))
        if size > target:
            raise SystemExit(f"did not fit {target / 1e6:.3f} MB in {a.max_packs} packs; last {size / 1e6:.3f} MB")

    sub = subset(fields, keep)
    write_sogst_ply(a.out + ".ply", sub, header["time_min"], header["time_max"], header["fps"],
                    cov2d_scale=header.get("cov2d_scale"), generator="cumuli fit_rate_target")
    report = {"input": a.input, "count_in": n, "count_out": int(len(keep)), "bytes": size,
              "target_bytes": target, "duration_s": duration, "mbps": size / 1e6 / duration,
              "drop_bits": drop, "packs": [{"keep": int(k), "bytes": int(s)} for k, s in history],
              "min_kept_contribution_px": float(score[keep].min())}
    with open(a.out + ".json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {a.out}.sogst ({size / 1e6:.3f} MB, {size / 1e6 / duration:.3f} MB/s, "
          f"{len(keep):,} splats) and {a.out}.ply")


if __name__ == "__main__":
    main()
