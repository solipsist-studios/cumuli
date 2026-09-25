# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Build a 4DAnyone motion directory from the RIG's own triangulated pose.

WHY
---
4DAnyone estimates the subject's motion from one monocular view with GVHMR.
Against the rig's triangulated keypoints that estimate is 6.4 cm out, which is
25 px of disagreement at the real cameras -- far outside anything gaussian
reconstruction can fuse, and the reason every hybrid so far lost 6 dB to a
real-only control. Fitting SMPL-X to the rig's own keypoints instead reaches
1.45 cm, and `MotionResult.load` will read a supplied motion directory in place
of running GVHMR, so the generated views can be conditioned on the true pose.

The payoff is not just a better fit. Generated cameras come back in a world
related to the rig by a transform we CHOSE, so alignment stops being something
estimated after the fact and becomes exact by construction.

WORLD
-----
4DAnyone requires `gvhmr_gravity_aligned_y_up`, and its own runs put the origin
at the subject's initial root projected onto the ground, +Y up. The rig world is
none of those things: it is scale-free (1.343 units per metre), its up axis is
the floor normal rather than an axis, and its heading is arbitrary. This builds
the transform explicitly -- up from a plane fitted to the foot contact points,
scale from the SMPL-X body fit, heading chosen so the subject faces -Z at t=0
(which also makes generated assets share a heading instead of each landing at a
random yaw), and the origin placed under the subject's first pelvis.

WHAT IS FITTED
--------------
Shape is shared across the clip since the subject does not change size; global
orientation, translation and body pose are per frame. Regularisation keeps the
solution anatomically plausible rather than letting it contort onto the joint
targets, which matters because goliath landmarks sit on the surface while
SMPL-X joints sit inside the body -- a systematic offset the fit should not try
to close.

conda env: cumuli (smplx + safetensors added alongside the torch pin this
env already carries; see environment.yml).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INVALID_ABS = 100.0
# goliath308 -> SMPL-X.
#
# The last two pairs are the feet, and leaving them out was a real defect: with
# only ankles constrained, ankle ROTATION is unconstrained by data and gets set
# by the pose regulariser, which has no idea which way a foot should face. The
# fitted feet came out 107 deg off on the left and 97 on the right, reaching
# 175 -- pointing backwards. 4DAnyone rendered that skeleton faithfully and the
# generator drew the anatomically impossible foot it was asked for.
#
# goliath's big toes (15, 18) match SMPL-X's foot joints (10, 11), which sit at
# the ball of the foot, so the pair fixes the ankle in all three axes.
PAIRS = [(5, 16), (6, 17), (7, 18), (8, 19), (62, 20), (41, 21),
         (9, 1), (10, 2), (11, 4), (12, 5), (13, 7), (14, 8),
         (15, 10), (18, 11)]
# goliath308 -> COCO-17 order, for the 2D keypoints 4DAnyone's framing wants
COCO_FROM_GOLIATH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
FOOT_JOINTS = (17, 20, 13, 14, 15, 18)


def load_keypoints(kp3d_dir: Path):
    frames, points = [], []
    for path in sorted(kp3d_dir.glob("*.json")):
        kp = np.asarray(json.loads(path.read_text())["instance_info"][0]["keypoints"])
        frames.append(int(path.stem))
        points.append(kp)
    return np.asarray(frames), np.asarray(points)


def floor_normal(points: np.ndarray) -> np.ndarray:
    """Up, from a plane through the foot keypoints in ground contact."""
    feet = points[:, FOOT_JOINTS, :].reshape(-1, 3)
    feet = feet[np.abs(feet).max(1) < INVALID_ABS]
    # rig y grows downward, so contact is the largest y
    contact = feet[feet[:, 1] >= np.percentile(feet[:, 1], 85)]
    centre = contact.mean(0)
    _, _, Vt = np.linalg.svd(contact - centre)
    normal = Vt[-1]
    if normal[1] > 0:
        normal = -normal
    residual = np.abs((contact - centre) @ normal)
    return normal, centre, residual.mean(), len(contact)


def basis_from_up_and_forward(up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Rows are the world axes expressed in rig coordinates."""
    y = up / np.linalg.norm(up)
    f = forward - y * (forward @ y)
    f /= np.linalg.norm(f)
    z = -f                      # the subject faces -Z
    x = np.cross(y, z)
    return np.stack([x, y, z])


def rodrigues_to_matrix(v):
    import torch
    theta = torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
    k = v / theta
    K = torch.zeros(v.shape[0], 3, 3, dtype=v.dtype, device=v.device)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(v.shape[0], 3, 3)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


def matrix_to_rodrigues(R):
    import torch
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.arccos(cos)
    axis = torch.stack([R[:, 2, 1] - R[:, 1, 2],
                        R[:, 0, 2] - R[:, 2, 0],
                        R[:, 1, 0] - R[:, 0, 1]], dim=-1)
    axis = axis / (2 * torch.sin(theta)[..., None]).clamp(min=1e-8)
    return axis * theta[..., None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kp3d_dir", required=True, type=Path)
    ap.add_argument("--reference_motion", required=True, type=Path,
                    help="an existing GVHMR result whose timeline and source identity to copy")
    ap.add_argument("--rig_transforms", required=True, type=Path,
                    help="omg4_full4d/transforms_train.json, for the source camera's pose")
    ap.add_argument("--source_camera", default="cam01")
    ap.add_argument("--smplx_model", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--units_per_metre", type=float, default=1.343,
                    help="rig scale; 1.343 from the SMPL-X body fit, 1.351 from subject height")
    ap.add_argument("--frame_offset", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3000)
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    import smplx

    frames, points = load_keypoints(args.kp3d_dir)
    up, floor_centre, flat, n_contact = floor_normal(points)
    print(f"floor from {n_contact} contact points, planar to {flat * 1000 / args.units_per_metre:.1f} mm")
    print(f"  world up in rig coordinates: {np.round(up, 4)}")

    reference = json.loads((args.reference_motion / "motion.json").read_text())
    source_frame = reference["source_frame_indices"]
    n = len(source_frame)

    # heading from the subject's first frame: hips give the body's left-right
    index_of = {f: i for i, f in enumerate(frames)}
    first = index_of[source_frame[0] + 1 + args.frame_offset]
    hips = points[first, 9] - points[first, 10]
    forward = np.cross(hips, up)
    basis = basis_from_up_and_forward(up, forward)
    scale = 1.0 / args.units_per_metre
    pelvis0 = (points[first, 9] + points[first, 10]) / 2
    # origin under the first pelvis, on the floor plane
    origin = pelvis0 - up * ((pelvis0 - floor_centre) @ up)

    def to_world(x):
        return (basis @ ((x - origin) * scale).T).T

    print(f"  scale {scale:.4f} m per rig unit, origin under the first pelvis")

    targets = np.full((n, len(PAIRS), 3), np.nan)
    for i, src in enumerate(source_frame):
        key = src + 1 + args.frame_offset
        if key not in index_of:
            continue
        kp = points[index_of[key]]
        for j, (g, _) in enumerate(PAIRS):
            if np.abs(kp[g]).max() < INVALID_ABS:
                targets[i, j] = to_world(kp[g][None])[0]
    mask_np = ~np.isnan(targets).any(-1)
    targets = np.nan_to_num(targets)
    print(f"{int(mask_np.sum())} joint targets across {n} frames "
          f"({int((~mask_np.any(1)).sum())} frames with none)")

    target = torch.tensor(targets, dtype=torch.float32)
    mask = torch.tensor(mask_np, dtype=torch.float32)
    model = smplx.SMPLX(model_path=str(args.smplx_model.parent), gender="neutral",
                        use_pca=False, num_betas=10, ext="npz", batch_size=n)
    smplx_idx = [s for _, s in PAIRS]

    betas = torch.zeros(1, 10, requires_grad=True)
    body_pose = torch.zeros(n, 63, requires_grad=True)
    global_orient = torch.zeros(n, 3, requires_grad=True)
    # start each frame at the centroid of its own targets; frames with no
    # targets start at the origin and are carried by the smoothness term
    seed = np.zeros((n, 3))
    seeded = mask_np.any(1)
    seed[seeded] = targets[seeded].sum(1) / mask_np[seeded].sum(1)[:, None]
    transl = torch.tensor(seed, dtype=torch.float32).clone().requires_grad_(True)

    opt = torch.optim.Adam([
        {"params": [body_pose, global_orient, transl], "lr": 0.02},
        {"params": [betas], "lr": 0.01},
    ])
    for step in range(args.iters):
        opt.zero_grad()
        out = model(betas=betas.expand(n, -1), body_pose=body_pose,
                    global_orient=global_orient, transl=transl)
        err = torch.linalg.norm(out.joints[:, smplx_idx, :] - target, dim=-1)
        data = (err * mask).sum() / mask.sum()
        smooth = 1e-2 * ((body_pose[1:] - body_pose[:-1]) ** 2).mean()
        reg = 1e-3 * (body_pose ** 2).mean() + 1e-2 * (betas ** 2).mean()
        (data + reg + smooth).backward()
        opt.step()
        if step % 500 == 0 or step == args.iters - 1:
            print(f"  step {step:5d}  joint error {data.item() * 100:6.2f} cm")

    # Frames with no targets have nothing holding their translation: the
    # smoothness term acts on body pose only, so they drift. On the first run
    # two such frames left the body 1.25 m from its neighbours, which poisoned
    # the init cloud and the views generated for them. Interpolate them from
    # the frames that WERE constrained rather than trusting the optimiser.
    with torch.no_grad():
        constrained = mask_np.any(1)
        if not constrained.all():
            idx = np.arange(n)
            filled = transl.detach().numpy().copy()
            for c in range(3):
                filled[~constrained, c] = np.interp(
                    idx[~constrained], idx[constrained], filled[constrained, c])
            transl.copy_(torch.tensor(filled, dtype=torch.float32))
            orient = global_orient.detach().numpy().copy()
            nearest = idx[constrained][np.argmin(
                np.abs(idx[~constrained, None] - idx[constrained][None, :]), axis=1)]
            orient[~constrained] = orient[nearest]
            global_orient.copy_(torch.tensor(orient, dtype=torch.float32))
            print(f"interpolated {int((~constrained).sum())} unconstrained frames "
                  f"from their neighbours")

    with torch.no_grad():
        out = model(betas=betas.expand(n, -1), body_pose=body_pose,
                    global_orient=global_orient, transl=transl)
        joints = out.joints
        step = torch.linalg.norm(transl[1:] - transl[:-1], dim=-1)
        print(f"root trajectory: extent {np.round(transl.numpy().ptp(0), 3)} m, "
              f"step mean {step.mean() * 100:.1f} cm, max {step.max() * 100:.1f} cm")
        err = (torch.linalg.norm(joints[:, smplx_idx, :] - target, dim=-1) * mask)
        per_frame = err.sum(1) / mask.sum(1).clamp(min=1)
    print(f"final joint error: mean {per_frame.mean() * 100:.2f} cm, "
          f"max {per_frame.max() * 100:.2f} cm")

    # ---- in-camera parameters: the same body, seen from the source camera ---
    rig = json.loads(args.rig_transforms.read_text())["frames"]
    cam_entry = next(f for f in rig if f["file_path"].startswith(f"{args.source_camera}/"))
    c2w_rig = np.asarray(cam_entry["transform_matrix"], dtype=np.float64)
    # camera centre and axes into the fitted world
    cam_centre = to_world(c2w_rig[:3, 3][None])[0]
    # The rig stores OpenGL camera bases (looking down -Z, +Y up). SMPL-X
    # in-camera parameters and pinhole projection both want OpenCV (+Z forward,
    # +Y down), so flip those two axes -- without this, points in front of the
    # camera carry negative depth and the projection diverges.
    cam_R = basis @ c2w_rig[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    w2c = np.eye(4)
    w2c[:3, :3] = cam_R.T
    w2c[:3, 3] = -cam_R.T @ cam_centre

    with torch.no_grad():
        rest = model(betas=betas.expand(n, -1),
                     body_pose=torch.zeros_like(body_pose),
                     global_orient=torch.zeros_like(global_orient),
                     transl=torch.zeros_like(transl))
        pelvis_rest = rest.joints[:, 0, :]
        R_delta = torch.tensor(w2c[:3, :3], dtype=torch.float32)
        t_delta = torch.tensor(w2c[:3, 3], dtype=torch.float32)
        orient_incam = matrix_to_rodrigues(R_delta @ rodrigues_to_matrix(global_orient))
        transl_incam = (R_delta @ (transl + pelvis_rest).T).T + t_delta - pelvis_rest

    # ---- observed 2D keypoints: reuse the real detections ------------------
    # These are what a 2D detector saw in the source video. They describe the
    # image, not GVHMR's 3D guess, so they stay valid under a different 3D fit
    # and copying them is more honest than synthesising them.
    ref_tensors = load_file(str(args.reference_motion / reference["tensor_file"]))
    K = ref_tensors["K_fullimg"]
    kp2d = ref_tensors["observed_keypoints_2d"]

    # Validate the in-camera transform against them: project the fitted body's
    # COCO joints through the source intrinsics and compare. A wrong in-camera
    # transform shows up here as a large error, and 4DAnyone's own framing
    # analysis measures the same agreement.
    SMPLX_FOR_COCO = [55, 56, 57, 58, 59, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
    with torch.no_grad():
        incam = model(betas=betas.expand(n, -1), body_pose=body_pose,
                      global_orient=orient_incam, transl=transl_incam)
        pts = incam.joints[:, SMPLX_FOR_COCO, :].numpy()
    Kn = K.numpy()
    observed = kp2d.numpy()
    errs = []
    for i in range(n):
        z = np.maximum(pts[i][:, 2], 1e-6)
        uv = (Kn[i] @ (pts[i] / z[:, None]).T).T[:, :2]
        body = observed[i][:, 2] > 0.3
        body[:5] = False        # face joints differ most between conventions
        if body.sum():
            errs.append(np.linalg.norm(uv[body] - observed[i][body, :2], axis=1).mean())
    if errs:
        print(f"in-camera check: fitted body reprojects {np.mean(errs):.1f} px "
              f"from the detected 2D keypoints (median {np.median(errs):.1f}) "
              f"on a {reference['image_width']}x{reference['image_height']} frame")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "smpl_params_global.betas": betas.detach().expand(n, -1).contiguous(),
        "smpl_params_global.body_pose": body_pose.detach(),
        "smpl_params_global.global_orient": global_orient.detach(),
        "smpl_params_global.transl": transl.detach(),
        # In-camera parameters are COPIED from the reference run, not derived.
        # They feed only analyze_input_framing (pipeline.py:345); everything
        # that drives skeleton conditioning comes from smpl_params_global,
        # which is the fitted pose. Copying keeps the framing profile identical
        # to the known-good run, so the generated ring geometry is unchanged and
        # pose conditioning is the only variable -- and it avoids shipping a
        # transform I could not validate: projecting my derived in-camera body
        # landed 101 px off in x and ~20% small against the detected 2D
        # keypoints, a discrepancy I have not explained.
        "smpl_params_incam.betas": ref_tensors["smpl_params_incam.betas"],
        "smpl_params_incam.body_pose": ref_tensors["smpl_params_incam.body_pose"],
        "smpl_params_incam.global_orient": ref_tensors["smpl_params_incam.global_orient"],
        "smpl_params_incam.transl": ref_tensors["smpl_params_incam.transl"],
        "K_fullimg": K,
        "observed_keypoints_2d": torch.tensor(kp2d),
    }
    save_file(tensors, str(args.out_dir / "motion.safetensors"))
    metadata = dict(reference)
    metadata["tensor_file"] = "motion.safetensors"
    (args.out_dir / "motion.json").write_text(json.dumps(metadata, indent=1))

    # 4DAnyone re-orients the world it is handed: cameras.json documents
    # "the subject initially faces +z", and skeleton/pipeline.py gets there with
    # compute_T_ayfz2ay(first_joints, inverse=True). Our fitted world faces -Z,
    # so the generated cameras come back half a turn from the body, and a hybrid
    # reconstructs two subjects facing opposite ways.
    #
    # The correction is a 180 degree yaw about the world Y axis, applied to a
    # generated camera BEFORE the rig similarity. VERIFIED BY RENDERING, not by
    # metrics: rendering the real splat from a generated pose with this applied
    # reproduces the generated image's facing, pose and details, while without
    # it the two show opposite sides. Three quantitative checks all missed this
    # -- silhouette overlap cannot separate a front from a back, PSNR against
    # generated frames is swamped by their studio background, and camera
    # positions are near-symmetric under 180 degrees in a 24-view ring.
    yaw180 = np.diag([-1.0, 1.0, -1.0])

    # TWO transforms, kept separate on purpose. Conflating them cost an hour of
    # wrong measurements: after folding the yaw into the exported matrix I kept
    # using it to map rig points into the FIT's world, where it does not belong,
    # and read 0.5-1.3 m residuals off a fit that was actually within 2 cm.
    #
    #   fit_world  - the world the SMPL-X fit was solved in. Use this to compare
    #                fitted joints against rig keypoints.
    #   camera     - the frame 4DAnyone emits cameras in, which is fit_world
    #                yawed 180 degrees. Use this to place generated views.
    fit_world_from_rig = np.eye(4)
    fit_world_from_rig[:3, :3] = basis * scale
    fit_world_from_rig[:3, 3] = -(basis * scale) @ origin

    rig_from_camera = np.linalg.inv(fit_world_from_rig)
    rig_from_camera[:3, :3] = rig_from_camera[:3, :3] @ yaw180

    # Ground compensation. 4DAnyone re-grounds whatever body it is given by
    # setting the lowest mesh vertex to y=0 (skeleton/pipeline.py: offset[1] =
    # vertices_smpl[..., 1].min()). A mesh sole sits below the toe keypoints it
    # was fitted to, so the generated body ends up ABOVE the rig body by that
    # gap -- 10.2 cm with the feet unconstrained, 3.8 cm once they are fitted.
    #
    # Shifting the body cannot fix this: the grounding is relative, so the body
    # and its own lowest vertex move together and the lift is unchanged. The
    # displacement is a constant, so correct it where it belongs, in the
    # transform that places generated views: move the cameras down by it and
    # the reconstruction follows.
    with torch.no_grad():
        lowest = float(out.vertices[..., 1].min())
    pelvis0 = joints[0, 0].numpy()
    ground_offset = np.array([pelvis0[0], lowest, pelvis0[2]])
    rig_targets, gen_ys = [], []
    for i in range(n):
        for g, s in PAIRS:
            if not mask_np[i][[p[0] for p in PAIRS].index(g)]:
                continue
            rig_targets.append(targets[i][[p[0] for p in PAIRS].index(g)][1])
            gen_ys.append(float(joints[i, s, 1]) - ground_offset[1])
    lift = float(np.mean(np.asarray(gen_ys) - np.asarray(rig_targets)))
    up_in_fit = np.array([0.0, 1.0, 0.0])
    shift = (np.linalg.inv(fit_world_from_rig)[:3, :3] @ (-lift * up_in_fit))
    rig_from_camera[:3, 3] += shift
    print(f"ground compensation: generated body sits {lift * 100:+.1f} cm above the rig; "
          f"shifting placement by {np.linalg.norm(shift):.4f} rig units to cancel it")

    transform = np.linalg.inv(rig_from_camera)
    print("exported both transforms: fit_world (for joints) and camera (for views)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "smpl_params_global.betas": betas.detach().expand(n, -1).contiguous(),
        "smpl_params_global.body_pose": body_pose.detach(),
        "smpl_params_global.global_orient": global_orient.detach(),
        "smpl_params_global.transl": transl.detach(),
        # In-camera parameters are COPIED from the reference run, not derived.
        # They feed only analyze_input_framing (pipeline.py:345); everything
        # that drives skeleton conditioning comes from smpl_params_global,
        # which is the fitted pose. Copying keeps the framing profile identical
        # to the known-good run, so the generated ring geometry is unchanged and
        # pose conditioning is the only variable -- and it avoids shipping a
        # transform I could not validate: projecting my derived in-camera body
        # landed 101 px off in x and ~20% small against the detected 2D
        # keypoints, a discrepancy I have not explained.
        "smpl_params_incam.betas": ref_tensors["smpl_params_incam.betas"],
        "smpl_params_incam.body_pose": ref_tensors["smpl_params_incam.body_pose"],
        "smpl_params_incam.global_orient": ref_tensors["smpl_params_incam.global_orient"],
        "smpl_params_incam.transl": ref_tensors["smpl_params_incam.transl"],
        "K_fullimg": K,
        "observed_keypoints_2d": torch.tensor(kp2d),
    }
    save_file(tensors, str(args.out_dir / "motion.safetensors"))
    metadata = dict(reference)
    metadata["tensor_file"] = "motion.safetensors"
    (args.out_dir / "motion.json").write_text(json.dumps(metadata, indent=1))

    # Export BOTH frames. Keeping them separate is the point: the fit lives in
    # fit_world, 4DAnyone emits cameras in that world yawed 180 degrees, and
    # conflating the two produced an hour of nonsense measurements.
    (args.out_dir / "T_world_from_rig.json").write_text(json.dumps({
        "T_world_from_rig": transform.tolist(),
        "T_fit_world_from_rig": fit_world_from_rig.tolist(),
        "units_per_metre": args.units_per_metre,
        "up_in_rig": up.tolist(),
        "note": "T_world_from_rig maps rig -> 4DAnyone's CAMERA frame (fit world "
                "yawed 180 deg); invert it to place generated views in rig world. "
                "T_fit_world_from_rig maps rig -> the world the SMPL-X fit was "
                "solved in; use that one to compare fitted joints with rig "
                "keypoints. They are NOT interchangeable.",
    }, indent=1))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
