# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Merge the real 12-camera rig with 4DAnyone's generated ring into one dataset.

WHY
---
Training on the generated ring alone leaves identity drift at the face-on view,
because no generated frame is anchored to a photograph of the subject's face
from that angle. The rig has those photographs. Mixing both means the real
cameras supply identity while the generated ring supplies the coverage the rig
never had.

THE THREE THINGS THAT HAVE TO LINE UP
-------------------------------------
Space: the generated cameras live in 4DAnyone's canonical human world and the
rig lives in its calibration world, so generated poses are mapped through the
similarity from `align_fdanyone_rig.py`. Its rotation block is scale*R, so the
scale is divided out before rotating a camera's basis; only positions take the
full similarity.

Time: real frames carry true capture seconds. Generated frames are timestamped
on 4DAnyone's canonical clock, which for a retimed clip disagrees with the
capture time of the frame the image was actually sampled from by up to half a
frame. `source_frame_indices` gives the true source frame, so generated views
are re-timestamped onto the capture clock here.

Alpha: the opacity-mask loss needs real alpha, and the rig's per-frame crops are
RGB with the background burned to black, which carries no alpha at all. The
masks still exist at full resolution, so this re-crops them and writes them into
the alpha channel. The crop is recoverable exactly: intrinsics are crop-local at
the 4096-wide undistorted grid, so origin = full-frame principal point (scaled
5312 -> 4096) minus the frame's cx/cy, and side is the image's own width. All
1750 origins come out integer, and re-cropping reproduces the shipped JPEG to
0.84/255 inside the mask.

WEIGHTING
---------
`loss_weight` scales the photometric term per view (OMG4 reads it per frame,
default 1.0). Each timestamp has 12 real views against 24 generated ones, so at
equal weight the generated views would carry twice the gradient and identity
would drift toward them again -- the failure this is meant to fix. The default
0.5 equalises aggregate influence (12 x 1.0 against 24 x 0.5); --gen_weight
overrides it. A flat weight is deliberate: the Eq. S2 ramp models confidence
decaying with angular distance from a source view, which is not what is being
expressed here.

COLOUR MATCHING (--color_match): measured HARMFUL, kept for the record
---------------------------------------------------------------------
Generated views are 27% darker and 42% less saturated than the rig, so matching
them looked obviously right. It lost 0.27 dB on held-out real cam04 (23.57
against 24.00 without). The correction multiplies pixels that have already been
through a log-to-display conversion and a contrast S-curve, which matches the
mean while distorting the mid-tones that carry the subject -- a gain belongs in
linear light, not on top of a tone curve. Do this in RawTherapee from the LOG
source frames instead; the pipeline already has the --pp3_dir hook for it, and
all 12 profiles are currently byte-identical so the rig has never been
per-camera matched at all.

WHETHER TO USE THIS SCRIPT AT ALL
---------------------------------
On the 2026-08-26 measurements the generated ring only ever hurt: held-out
cam04 scored 30.04 dB with real cameras alone, 28.05 at gen_weight 0.15 and
24.00 at 0.5, a linear 12 dB lost per unit of weight with the optimum at
zero. Alignment was not the fix either -- halving the misalignment moved it
only 0.16 dB. The generated views had to be conditioned on the RIG's pose
(SMPL-X fitted to triangulate_rig_keypoints.py output reaches 1.45 cm
against GVHMR's 6.4 cm) before a hybrid was worth training at all.

CORRECTED 2026-08-27, with the rig-conditioned pose fit and this script's
`--min_separation_deg` culling both in place: held-out cam04 reached 31.19
dB against a real-only control's 30.87 dB, a real (if modest, +0.32 dB)
gain -- but only once views within ~20 degrees of a real camera were
culled, and only where the culling has genuine ground to stand on (the
subject's covered azimuths, not the ones no camera reaches at all). Do not
run this without `--min_separation_deg` set close to that value.

MULTIPLE GENERATION RUNS (`--gen_run`)
---------------------------------------
A single 4DAnyone run can only produce one continuous partial-span ring
(views_per_layer/layer_pitches/start_yaw/yaw_span), so filling several
disjoint gaps against a target rig-spec takes several runs. `--gen_run NAME
DIR` is repeatable, one per generation run; every run shares the same
`--transform` and `--motion_dir`, since `fit_rig_motion.py` fits the rig's
pose ONCE per capture and every generation run is conditioned on that same
motion (verified: `fit_rig_motion.py`'s `--source_camera` only feeds a
discarded sanity check, not the exported motion itself). Frames from each
run are namespaced by run name (`gen/<name>_<camera>/...`) so camera-id
collisions between independently-numbered 4DAnyone runs cannot alias. Only
the FIRST run's `points3d.ply` is used for the init cloud -- every run
samples the same underlying SMPL-X body and motion, so later copies would
just duplicate points, not add coverage.

HOLDOUT
-------
--holdout names a REAL camera excluded from training and written to
transforms_test.json. Scoring against a real held-out camera is a far more
honest measure than the generated-view self-consistency PSNR the ring-only runs
reported, which only ever asked whether the splat reproduced 4DAnyone's output.

conda env: cumuli.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

UND_W = 4096          # full4d_und grid the rig crops were taken from
SFM_W = 5312          # undistorted SfM frame the intrinsics were solved on
# Those two are the Tatum rig's numbers, where the crops were taken on a
# rescaled undistorted grid. A capture cropped straight off the SfM frame has
# the two equal, so --und_grid_width/--sfm_grid_width override them.


def load_similarity(path: Path):
    """(rotation, translations, scale) of T_rig<-canonical.

    `translations` is either a single vector, or one per generated frame when
    the file came from align_fdanyone_perframe.py. A global transform leaves
    the two sources disagreeing about the subject's position by up to 37.9 cm,
    which is why the per-frame form exists."""
    data = json.loads(path.read_text())
    if "translations" in data:
        return (np.asarray(data["R_global"], dtype=np.float64),
                np.asarray(data["translations"], dtype=np.float64),
                float(data["scale"]))
    if "T_rig_from_canonical" not in data and "T_world_from_rig" in data:
        # fit_rig_motion.py exports the other direction; accept either rather
        # than fail late in a two-hour chain over a key name
        data["T_rig_from_canonical"] = np.linalg.inv(
            np.asarray(data["T_world_from_rig"], dtype=np.float64)).tolist()
    T = np.asarray(data["T_rig_from_canonical"], dtype=np.float64)
    scale = float(np.linalg.det(T[:3, :3]) ** (1.0 / 3.0))
    return T[:3, :3] / scale, T[:3, 3], scale


def map_pose(c2w: np.ndarray, R: np.ndarray, t: np.ndarray, scale: float) -> np.ndarray:
    """Camera-to-world through a similarity: the basis rotates, the position
    takes the full transform. Scaling the basis would leave it non-orthonormal
    and the renderer would read it as a skewed camera."""
    out = np.eye(4)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = scale * (R @ c2w[:3, 3]) + t
    return out


def real_crop_origin(frame: dict, full: dict,
                     und_w: int = UND_W, sfm_w: int = SFM_W) -> tuple[int, int]:
    s = und_w / sfm_w
    x0 = full["cx"] * s - frame["cx"]
    y0 = full["cy"] * s - frame["cy"]
    for value, axis in ((x0, "x"), (y0, "y")):
        if abs(value - round(value)) > 1e-3:
            raise ValueError(f"non-integer crop origin on {axis} for {frame['file_path']}: {value}")
    return int(round(x0)), int(round(y0))


def subject_stats(paths, max_views=60):
    """Per-channel mean and spread of subject pixels across a sample of RGBA views."""
    step = max(1, len(paths) // max_views)
    sums = np.zeros(3)
    sqs = np.zeros(3)
    count = 0
    for path in paths[::step]:
        a = np.asarray(Image.open(path))
        if a.shape[2] < 4:
            continue
        inside = a[:, :, 3] > 200
        if inside.sum() < 500:
            continue
        px = a[:, :, :3][inside].astype(np.float64)
        sums += px.sum(0)
        sqs += (px ** 2).sum(0)
        count += len(px)
    if count == 0:
        return None, None
    mean = sums / count
    return mean, np.sqrt(np.maximum(sqs / count - mean ** 2, 1e-6))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_transforms", required=True, type=Path,
                    help="omg4_full4d/transforms_train.json (12 cameras, capture-time seconds)")
    ap.add_argument("--real_root", required=True, type=Path,
                    help="directory holding cam01/..cam12/ crops named in real_transforms")
    ap.add_argument("--masks_root", required=True, type=Path,
                    help="full4d_masks/<camera_id>/f<NNNN>.png at full undistorted resolution")
    ap.add_argument("--undistorted_transforms", required=True, type=Path,
                    help="transforms_multiframe_undistorted.json, for the full-frame principal point")
    ap.add_argument("--gen_run", required=True, action="append", nargs=2,
                    metavar=("NAME", "DIR"),
                    help="a converted 4DAnyone RGBA dataset (transforms_train.json + gen*/) "
                         "and a name to namespace its cameras by. Repeatable: one per "
                         "generation run. All runs share --transform and --motion_dir.")
    ap.add_argument("--motion_dir", required=True, type=Path,
                    help="GVHMR result dir, for source_frame_indices. Shared across every "
                         "--gen_run: fit_rig_motion.py fits the rig's pose once per capture.")
    ap.add_argument("--transform", required=True, type=Path,
                    help="T_rig<-canonical from align_fdanyone_rig.py, or the per-frame "
                         "form from align_fdanyone_perframe.py (detected by content)")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--gen_weight", type=float, default=0.5)
    ap.add_argument("--holdout", default="cam04", help="real camera held out for testing")
    ap.add_argument("--capture_fps_num", type=int, default=30000)
    ap.add_argument("--capture_fps_den", type=int, default=1001)
    ap.add_argument("--limit", type=int, default=0,
                    help="rehearsal: process at most N views from each source")
    ap.add_argument("--und_grid_width", type=int, default=UND_W,
                    help="width of the grid the real crops were taken on")
    ap.add_argument("--sfm_grid_width", type=int, default=SFM_W,
                    help="width of the frame the real intrinsics were solved on; "
                         "equal to --und_grid_width when the crops came straight "
                         "off the SfM frame")
    ap.add_argument("--min_separation_deg", type=float, default=0.0,
                    help="drop generated views whose camera sits within this many degrees "
                         "of azimuth of a real one; 0 keeps all of them")
    ap.add_argument("--color_match", action="store_true",
                    help="rescale generated RGB so its subject pixels match the real "
                         "cameras' per-channel mean and spread. MEASURED HARMFUL: see "
                         "the COLOUR MATCHING note in this module's docstring")
    args = ap.parse_args()

    frame_seconds = args.capture_fps_den / args.capture_fps_num
    args.out_dir.mkdir(parents=True, exist_ok=True)

    real = json.loads(args.real_transforms.read_text())
    full_by_label = {f["camera_label"]: f
                     for f in json.loads(args.undistorted_transforms.read_text())["frames"]}

    train, test = [], []

    # ---- real cameras: re-crop the mask into alpha -------------------------
    real_dir = args.out_dir / "realcams"
    real_frames = real["frames"][:args.limit] if args.limit else real["frames"]
    for frame in real_frames:
        cam, name = frame["file_path"].split("/")
        camera_id = f"{int(cam[3:]):04d}"
        capture_frame = int(name.split("_")[-1])
        source = args.real_root / f"{frame['file_path']}.jpg"
        if not source.exists():
            continue
        rgb = Image.open(source).convert("RGB")
        x0, y0 = real_crop_origin(frame, full_by_label[camera_id],
                                  args.und_grid_width, args.sfm_grid_width)
        side = rgb.size[0]
        mask_path = args.masks_root / camera_id / f"f{capture_frame:04d}.png"
        mask = Image.open(mask_path).convert("L").crop((x0, y0, x0 + side, y0 + side))
        rgba = rgb.copy()
        rgba.putalpha(mask)
        out_path = real_dir / cam / f"{name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(out_path)

        entry = dict(frame)
        entry["file_path"] = f"realcams/{cam}/{name}"
        entry["w"], entry["h"] = side, side
        entry["loss_weight"] = 1.0
        (test if cam == args.holdout else train).append(entry)

    # ---- colour match: the sources disagree on exposure, not just position --
    # Generated views measured 27% darker and 42% less saturated than the rig.
    # A uniform luminance bias that size caps PSNR at ~25.9 dB before any
    # geometric error, which is most of the hybrid's 6 dB deficit; per-frame
    # alignment moved it only 0.16 dB, so appearance is what dominates.
    gain = offset = None
    if args.color_match:
        real_paths = sorted((args.out_dir / "realcams").rglob("*.png"))
        # Exclude the holdout camera: fitting on it would leak the test set.
        real_paths = [p for p in real_paths if p.parent.name != args.holdout]
        gen_paths = sorted(
            p for _, gen_dir_arg in args.gen_run
            for p in Path(gen_dir_arg).glob("gen*/*.png"))
        r_mean, r_std = subject_stats(real_paths)
        g_mean, g_std = subject_stats(gen_paths)
        if r_mean is None or g_mean is None:
            raise SystemExit("colour match: no subject pixels found in one of the sources")
        gain = r_std / g_std
        offset = r_mean - gain * g_mean
        print(f"colour match: real mean {np.round(r_mean, 1)} sd {np.round(r_std, 1)}, "
              f"generated mean {np.round(g_mean, 1)} sd {np.round(g_std, 1)}")
        print(f"  per-channel gain {np.round(gain, 3)}, offset {np.round(offset, 1)}")

    # ---- generated ring(s): reposed into rig world, retimed to capture -----
    # Every run shares one similarity and one motion: fit_rig_motion.py fits
    # the rig's pose ONCE per capture, and each generation run is conditioned
    # on that same motion (see the MULTIPLE GENERATION RUNS docstring note).
    R, t, scale = load_similarity(args.transform)
    gen_runs = [(name, Path(gen_dir_arg),
                 json.loads((Path(gen_dir_arg) / "transforms_train.json").read_text()))
                for name, gen_dir_arg in args.gen_run]
    source_frame = json.loads((args.motion_dir / "motion.json").read_text())["source_frame_indices"]

    # ---- cull generated views that duplicate a real camera ------------------
    # Jeff's point: the ring is evenly spaced every 15 degrees while the real
    # cameras cover a 220 degree arc, so several generated views land within a
    # couple of degrees of a real one. Those are guesses standing next to a
    # photograph of the same viewpoint, contradicting it; the trainer has to
    # reconcile them and the real camera loses ground. The views worth keeping
    # are the ones covering the back arc no real camera reaches. Every
    # generated view from every run is one flat pool here -- a view close to a
    # real camera is redundant regardless of which run produced it.
    #
    # Separation is measured on the rig's own camera sphere (rig_geometry), the
    # same frame render_pair_sweep and the conditioning stages use.
    cull = set()
    if args.min_separation_deg > 0:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import rig_geometry as rg
        real_centres, gen_centres = {}, {}
        for entry in train + test:
            label = entry["file_path"].split("/")[1]
            centre = np.asarray(entry["transform_matrix"], dtype=np.float64)[:3, 3]
            (real_centres if entry["file_path"].startswith("realcams/") else
             gen_centres).setdefault(label, centre)
        for run_name, _, gen in gen_runs:
            for frame in gen["frames"]:
                label = f"{run_name}_{frame['file_path'].split('/')[0]}"
                if label not in gen_centres:
                    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
                    gen_centres[label] = map_pose(c2w, R, t[0] if t.ndim == 2 else t, scale)[:3, 3]
        stack = np.stack(list(real_centres.values()))
        target = stack.mean(0)
        _, _, Vt = np.linalg.svd(stack - target)
        up = Vt[-1]
        if up[1] > 0:
            up = -up
        e1, e2 = rg.orbit_basis([{"center": v} for v in real_centres.values()], target, up)
        real_az = [rg.spherical(v, target, up, e1, e2)[0] for v in real_centres.values()]
        for label, centre in gen_centres.items():
            az = rg.spherical(centre, target, up, e1, e2)[0]
            if min(abs(rg.shortest_arc(az, ra)) for ra in real_az) < args.min_separation_deg:
                cull.add(label)
        print(f"culling {len(cull)} of {len(gen_centres)} generated views within "
              f"{args.min_separation_deg:g} deg of a real camera: {sorted(cull)}")

    gen_dir_out = args.out_dir / "gen"
    for run_name, run_dir, gen in gen_runs:
        gen_frames = gen["frames"][:args.limit] if args.limit else gen["frames"]
        for frame in gen_frames:
            cam, name = frame["file_path"].split("/")
            label = f"{run_name}_{cam}"
            if label in cull:
                continue
            index = int(name.split("_")[-1]) - 1        # converter writes 1-based names
            src = run_dir / f"{frame['file_path']}.png"
            if not src.exists():
                continue
            dst = gen_dir_out / label / f"{name}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                if gain is None:
                    # Unmodified, so link it rather than duplicating ~2 GB on a
                    # volume sitting at 99%.
                    try:
                        dst.hardlink_to(src)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    a = np.asarray(Image.open(src)).astype(np.float64)
                    a[:, :, :3] = np.clip(a[:, :, :3] * gain + offset, 0, 255)
                    Image.fromarray(a.astype(np.uint8), mode="RGBA").save(dst)

            entry = dict(frame)
            entry["file_path"] = f"gen/{label}/{name}"
            t_frame = t[index] if t.ndim == 2 else t
            entry["transform_matrix"] = map_pose(
                np.asarray(frame["transform_matrix"], dtype=np.float64), R, t_frame, scale).tolist()
            entry["time"] = source_frame[index] * frame_seconds
            entry["loss_weight"] = args.gen_weight
            train.append(entry)

    # ---- init cloud: the time-stamped SMPL-X points, moved to rig world ----
    # Only the FIRST run's cloud is used: every run samples the same
    # underlying SMPL-X body and motion, so later runs' clouds would only
    # duplicate points, not add coverage.
    init = gen_runs[0][1] / "points3d.ply"
    if init.exists():
        from plyfile import PlyData, PlyElement
        ply = PlyData.read(str(init))
        v = ply["vertex"]
        xyz = np.stack([np.asarray(v[k], dtype=np.float64) for k in ("x", "y", "z")], axis=1)
        if t.ndim == 2:
            # The init cloud carries a per-point time, so each point takes the
            # translation of the frame it came from rather than a clip average.
            times = np.asarray(v["time"], dtype=np.float64)
            frame_of = np.clip((times / max(times.max(), 1e-9) * (len(t) - 1)).round()
                               .astype(int), 0, len(t) - 1)
            xyz = scale * (R @ xyz.T).T + t[frame_of]
        else:
            xyz = scale * (R @ xyz.T).T + t
        data = np.asarray(v.data).copy()
        for i, k in enumerate(("x", "y", "z")):
            data[k] = xyz[:, i].astype(data[k].dtype)
        PlyData([PlyElement.describe(data, "vertex")], text=False).write(
            str(args.out_dir / "points3d.ply"))

    span = max(f["time"] for f in train + test)
    for name, frames in (("transforms_train.json", train), ("transforms_test.json", test)):
        (args.out_dir / name).write_text(json.dumps(
            {"w": real.get("w"), "h": real.get("h"), "frames": frames}, indent=1))

    n_real = sum(1 for f in train if f["file_path"].startswith("realcams/"))
    print(f"train: {len(train)} views ({n_real} real at 1.0, "
          f"{len(train) - n_real} generated at {args.gen_weight})")
    print(f"test:  {len(test)} views (real camera {args.holdout}, held out)")
    print(f"time span 0..{span:.3f}s -- set time_duration to match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
