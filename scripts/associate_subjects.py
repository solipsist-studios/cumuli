#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
associate_subjects.py

Multi-subject support for the keypoint stages. predict_keypoints_2d.py keeps every
detection per image, but split_keypoints_per_camera.py and the pose refiner use only
the first one. With several people in view that picks a different
person in different cameras, so the bundle adjustment sees inconsistent "wand"
points and drops most of them as outliers.

This script clusters detections across cameras by epipolar consistency under a
given set of poses: two detections belong to the same person when their confident
keypoints satisfy the epipolar constraint between the two cameras. It keeps the
--num_subjects best-supported clusters (most cameras, then largest), labels them
across frames by 3D-free image-space continuity of cluster centroids in a
reference camera, and writes one poses_2d-style directory per subject:

    <out_dir>/<frame_dir_name>_s<k>/<camera_label>/000000.json

Each such directory is one "instant" for run_pose_refinement.py (--kp2d_dirs), so
N frames x K subjects give N*K independent wand instants.

Usage:
    python3 associate_subjects.py --transforms transforms_refined.json \
        --pred_dirs kp2d/f0,kp2d/f1,... --out_dir subjects2d --num_subjects 2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refine_poses_with_keypoints import load_transforms, match_camera, triangulate_linear  # noqa: E402

# Gates. Each has a command-line flag; main() overwrites these from the arguments.
SCORE_THR = 0.5      # --score_thr: a keypoint counts when it scores above this
MIN_COMMON = 25      # --min_common: confident keypoints two views must share to be compared
MAX_ASPECT = 0.8     # --max_aspect: a detection wider than this times its height is not a person
EDGE_PX = 14.0       # --epipolar_px: how far, in pixels, a keypoint may sit from its epipolar line
DARK_RATIO = 0.5     # --dark_ratio: drop a detection darker than this times the anchor subject (merge only)
SIZE_RATIO = 1.3     # --size_ratio: drop a detection taller than this times the anchor subject (merge only)
EDGE_MARGIN = 2      # --edge_margin_px: drop a detection whose box touches the frame edge (merge only)


def fundamental(cam_a, cam_b):
    Ra, ta = cam_a["w2c"][:3, :3], cam_a["w2c"][:3, 3]
    Rb, tb = cam_b["w2c"][:3, :3], cam_b["w2c"][:3, 3]
    R = Rb @ Ra.T
    t = tb - R @ ta
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return np.linalg.inv(cam_b["K"]).T @ tx @ R @ np.linalg.inv(cam_a["K"])


def sampson(F, xa, xb):
    ha = np.c_[xa, np.ones(len(xa))]
    hb = np.c_[xb, np.ones(len(xb))]
    Fx = ha @ F.T
    Ftx = hb @ F
    num = np.sum(hb * Fx, axis=1) ** 2
    den = Fx[:, 0] ** 2 + Fx[:, 1] ** 2 + Ftx[:, 0] ** 2 + Ftx[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def load_instances(pred_json, cams):
    data = json.load(open(pred_json))
    out = {}
    for fr in data["frames"]:
        label = match_camera(fr["image_name"], list(cams))
        if label is None:
            continue
        # Group detections (one box around several people) are wider than any one person; they poison the clustering.
        def person_like(i):
            b = i["bbox"]
            return (b[2] - b[0]) <= MAX_ASPECT * (b[3] - b[1])
        out[label] = [{"kp": np.array(i["keypoints"], float), "sc": np.array(i["keypoint_scores"], float),
                       "raw": i} for i in fr.get("instances") or [] if person_like(i)]
    return out


def cluster(instances, cams):
    labels = sorted(instances)
    nodes = [(c, i) for c in labels for i in range(len(instances[c]))]
    parent = {n: n for n in nodes}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    cost = {}
    for ai, a in enumerate(labels):
        for b in labels[ai + 1:]:
            F = fundamental(cams[a], cams[b])
            for i, ia in enumerate(instances[a]):
                for j, ib in enumerate(instances[b]):
                    m = (ia["sc"] > SCORE_THR) & (ib["sc"] > SCORE_THR)
                    if m.sum() < MIN_COMMON:
                        continue
                    d = np.median(sampson(F, ia["kp"][m], ib["kp"][m]))
                    cost[((a, i), (b, j))] = d
    # Greedy, lowest cost first, never putting two detections of one camera in one cluster.
    members = {n: {n} for n in nodes}
    for (na, nb), d in sorted(cost.items(), key=lambda kv: kv[1]):
        if d > EDGE_PX:
            break
        ra, rb = find(na), find(nb)
        if ra == rb:
            continue
        cams_a = {c for c, _ in members[ra]}
        cams_b = {c for c, _ in members[rb]}
        if cams_a & cams_b:
            continue
        parent[rb] = ra
        members[ra] |= members[rb]
        del members[rb]
    return list(members.values())


def cluster_center_3d(cl, instances, cams):
    """Median over confident keypoints of the DLT-triangulated 3D point."""
    pts = []
    for k in range(308):
        use = [(c, i) for c, i in cl if instances[c][i]["sc"][k] > SCORE_THR]
        if len(use) < 3:
            continue
        Ks = [cams[c]["K"] for c, _ in use]
        w2cs = [cams[c]["w2c"] for c, _ in use]
        uvs = [instances[c][i]["kp"][k] for c, i in use]
        ws = [instances[c][i]["sc"][k] for c, i in use]
        try:
            pts.append(np.asarray(triangulate_linear(Ks, w2cs, uvs, ws), float).ravel()[:3])
        except Exception:
            continue
    return np.median(np.array(pts), axis=0) if pts else None


def pick_subjects(clusters, instances, cams, k):
    """Of the well-supported clusters, the k whose 3D centres are closest together
    (subjects stand together; bystanders stand apart)."""
    import itertools
    top = len({c for c, _ in clusters[0]})
    cand = [cl for cl in clusters if len({c for c, _ in cl}) >= max(5, int(np.ceil(0.8 * top)))][:6]
    cent = [cluster_center_3d(cl, instances, cams) for cl in cand]
    ok = [(cl, c) for cl, c in zip(cand, cent) if c is not None]
    if len(ok) <= k:
        return [cl for cl, _ in ok] or clusters[:k]
    best = min(itertools.combinations(range(len(ok)), k),
               key=lambda idx: sum(np.linalg.norm(ok[a][1] - ok[b][1]) for a, b in itertools.combinations(idx, 2)))
    return [ok[i][0] for i in best]


def cluster_size_key(cl, instances):
    h = [instances[c][i]["raw"]["bbox"][3] - instances[c][i]["raw"]["bbox"][1] for c, i in cl]
    return (len({c for c, _ in cl}), float(np.median(h)))


def select_for_masks(clusters, instances, adj_thr=0.9, keep_frac=0.7):
    """Subjects for mask cleaning: the best-supported cluster (anchor) plus everything that stays beside it in
    most views. Detections of the other subjects fragment into several small clusters under pose error, while a bystander is beside the anchor in only a few views and is excluded. Returns a set of (camera, idx)."""
    def box(n):
        b = np.array(instances[n[0]][n[1]]["raw"]["bbox"][:4], float)
        return (b[:2] + b[2:]) / 2, b[3] - b[1]
    anchor = clusters[0]
    anc = {c: box((c, i)) for c, i in anchor}
    def adjacent(n, ref):
        ctr, h = box(n)
        a_ctr, a_h = ref
        return np.linalg.norm(ctr - a_ctr) < adj_thr * max(h, a_h)
    keep = set(anchor); excluded = set()
    for cl in clusters[1:]:
        shared = [n for n in cl if n[0] in anc]
        if len(cl) < 3 or not shared:
            continue
        frac = np.mean([adjacent(n, anc[n[0]]) for n in shared])
        (keep if frac >= keep_frac else excluded).update(cl)
    ref = {}
    for c, i in keep:
        ref.setdefault(c, []).append(box((c, i)))
    for c, lst in instances.items():
        for i in range(len(lst)):
            n = (c, i)
            if n in keep or n in excluded or c not in ref:
                continue
            if any(adjacent(n, r) for r in ref[c]):
                keep.add(n)
    return keep


def main():
    global SCORE_THR, MIN_COMMON, MAX_ASPECT, EDGE_PX, DARK_RATIO, SIZE_RATIO, EDGE_MARGIN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--pred_dirs", required=True, help="comma list of dirs each holding one *_predictions.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_subjects", type=int, default=2)
    ap.add_argument("--ref_camera", default="Camera_0007")
    ap.add_argument("--images_dir", help="with --merge_out: per-camera images; drops detections that are much darker (--dark_ratio), much taller (--size_ratio) or cut off by the frame edge (--edge_margin_px) compared with the anchor subject in that camera")
    ap.add_argument("--merge_out", action="store_true",
                    help="write <out_dir>/<camera>/000000.json holding ALL chosen subjects (for clean_masks.py); one pred dir")
    ap.add_argument("--score_thr", type=float, default=SCORE_THR, help="Keypoint score a point must exceed (default %(default)s)")
    ap.add_argument("--min_common", type=int, default=MIN_COMMON, help="Shared confident keypoints needed to compare two views (default %(default)s)")
    ap.add_argument("--max_aspect", type=float, default=MAX_ASPECT, help="Widest a person box may be, as a fraction of its height (default %(default)s)")
    ap.add_argument("--epipolar_px", type=float, default=EDGE_PX, help="Epipolar distance, in pixels, within which two views agree (default %(default)s)")
    ap.add_argument("--dark_ratio", type=float, default=DARK_RATIO, help="With --merge_out: drop detections darker than this times the anchor (default %(default)s)")
    ap.add_argument("--size_ratio", type=float, default=SIZE_RATIO, help="With --merge_out: drop detections taller than this times the anchor (default %(default)s)")
    ap.add_argument("--edge_margin_px", type=int, default=EDGE_MARGIN, help="With --merge_out: drop detections whose box is within this many pixels of the frame edge (default %(default)s)")
    args = ap.parse_args()
    SCORE_THR, MIN_COMMON, MAX_ASPECT, EDGE_PX = args.score_thr, args.min_common, args.max_aspect, args.epipolar_px
    DARK_RATIO, SIZE_RATIO, EDGE_MARGIN = args.dark_ratio, args.size_ratio, args.edge_margin_px

    _, cams = load_transforms(args.transforms)
    out_dir = Path(args.out_dir)
    prev_centroids = None
    report = []
    for d in args.pred_dirs.split(","):
        d = Path(d)
        pj = next(d.glob("*_predictions.json"))
        inst = load_instances(pj, cams)
        clusters = cluster(inst, cams)
        clusters.sort(key=lambda cl: cluster_size_key(cl, inst), reverse=True)
        chosen = pick_subjects(clusters, inst, cams, args.num_subjects)
        info = [(len({c for c, _ in cl}), round(cluster_size_key(cl, inst)[1])) for cl in clusters[:5]]
        # Order subjects by continuity of the mean keypoint position in the reference camera.
        ref = args.ref_camera if args.ref_camera in cams else sorted(cams)[0]

        def centroid(cl):
            pts = [inst[c][i]["kp"][inst[c][i]["sc"] > SCORE_THR].mean(axis=0) for c, i in cl if c == ref]
            return pts[0] if pts else None

        cents = [centroid(cl) for cl in chosen]
        if (prev_centroids is not None and len(prev_centroids) == len(cents)
                and all(c is not None for c in cents) and all(c is not None for c in prev_centroids)):
            direct = sum(np.linalg.norm(cents[k] - prev_centroids[k]) for k in range(len(chosen)))
            swapped = sum(np.linalg.norm(cents[k] - prev_centroids[len(chosen) - 1 - k]) for k in range(len(chosen)))
            if swapped < direct:
                chosen = chosen[::-1]
                cents = cents[::-1]
        if all(c is not None for c in cents):
            prev_centroids = cents
        elif prev_centroids is None:
            prev_centroids = [np.zeros(2)] * len(chosen)
        if args.merge_out:
            keep = select_for_masks(clusters, inst)
            if args.images_dir:
                from PIL import Image
                anchor_cams = {c: i for c, i in clusters[0]}
                def luma(c, i):
                    im = Image.open(Path(args.images_dir) / f"{c}.png").convert("L")
                    b = inst[c][i]["raw"]["bbox"][:4]; w, h = b[2] - b[0], b[3] - b[1]
                    r = np.asarray(im.crop((int(b[0] + .25 * w), int(b[1] + .2 * h), int(b[0] + .75 * w), int(b[1] + .7 * h))).reduce(2), float)
                    return float(np.median(r))
                drop = set()
                for c, ai in anchor_cams.items():
                    la = luma(c, ai); ha = inst[c][ai]["raw"]["bbox"][3] - inst[c][ai]["raw"]["bbox"][1]
                    for cc, i in keep:
                        if cc != c or i == ai:
                            continue
                        b = inst[c][i]["raw"]["bbox"]
                        iw, ih = Image.open(Path(args.images_dir) / f"{c}.png").size
                        m = EDGE_MARGIN
                        edge = b[0] <= m or b[1] <= m or b[2] >= iw - m or b[3] >= ih - m   # cut off by the frame
                        if luma(c, i) < DARK_RATIO * la or (b[3] - b[1]) > SIZE_RATIO * ha or edge:
                            drop.add((c, i))
                keep -= drop
                print(f"  dropped {len(drop)} dark/oversized detections", flush=True)
            per_cam = {}
            for c, i in sorted(keep):
                raw = inst[c][i]["raw"]
                per_cam.setdefault(c, []).append({"keypoints": raw["keypoints"], "keypoint_scores": raw["keypoint_scores"]})
            for c, lst in per_cam.items():
                (out_dir / c).mkdir(parents=True, exist_ok=True)
                json.dump({"instance_info": lst}, open(out_dir / c / "000000.json", "w"))
            json.dump({c: [list(map(float, inst[c][i]["raw"]["bbox"][:4])) for cc, i in sorted(keep) if cc == c] for c in per_cam},
                      open(out_dir / "kept_boxes.json", "w"))
            print(f"{d.name}: kept {len(keep)} detections over {len(per_cam)} cameras", flush=True)
            continue
        for k, cl in enumerate(chosen):
            for c, i in cl:
                cam_dir = out_dir / f"{d.name}_s{k}" / c
                cam_dir.mkdir(parents=True, exist_ok=True)
                raw = inst[c][i]["raw"]
                json.dump({"instance_info": [{"keypoints": raw["keypoints"], "keypoint_scores": raw["keypoint_scores"]}]},
                          open(cam_dir / "000000.json", "w"))
        report.append({"frame_dir": d.name, "top_clusters_(n_cams,bbox_h)": info,
                       "chosen_cams": [sorted(c for c, _ in cl) for cl in chosen]})
        print(f"{d.name}: top clusters (n_cams, bbox_h) {info}; chosen cams {[len(cl) for cl in chosen]}", flush=True)
    json.dump(report, open(out_dir / "association_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
