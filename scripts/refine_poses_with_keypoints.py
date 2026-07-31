# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Refine static multi-camera poses using 2D human keypoints ("human as calibration wand").

Background SfM features live on distant walls/windows, so camera poses can fit the
background well while still disagreeing by dozens of pixels at the capture volume
center - exactly where the subject is. This script bundle-adjusts the camera poses
(intrinsics fixed) against Sapiens/COCO-style 2D keypoints detected in every camera,
across one or (much better) many time instants. The moving subject sweeps through
the capture volume and anchors the cameras where it matters.

Inputs:
  --transforms      transforms.json with per-frame intrinsics + OpenGL c2w poses
  --kp2d            either a combined predictions JSON
                    ({"frames": [{"image_name": "...", "instances": [{"keypoints": [...], 
                    "keypoint_scores": [...]}]}]}) or a directory <camera_label>/<tem>.json 
                    of per-image predictions
  --out_transforms  refined transforms.json (same frames, updated transform_matrix)

The optimization uses a Huber-robust reprojection objective over camera poses and
per-(time,keypoint) 3D points, with analytic sparsity for scipy least_squares.
Gauge freedom is removed afterwards by similarity-aligning the refined camera
centers back onto the input ones, so scene scale/origin are preserved.

Assumes goliath308 (308-keypoint) input throughout (see FACE_KEYPOINT_IDS in
main()) -- no backward compatibility with older/smaller keypoint layouts
(e.g. Sapiens 1 or coco_wholebody133); this pipeline standardizes on Sapiens 2 /
goliath308 (see predict_keypoints_2d.py).

Example:
    python refine_poses_with_keypoints.py \
        --transforms ./transforms.json \
        --kp2d ./poses_sapiens/poses_sapiens_predictions.json \
        --out_transforms ./transforms_refined.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def load_transforms(path):
    try:
        with open(path) as f:
            data = json.load(f)
        frames = data['frames']
    except FileNotFoundError:
        sys.exit(f'Error: {path} not found')
    except json.JSONDecodeError as e:
        sys.exit(f'Error: {path} is not valid JSON: {e}')
    except KeyError as e:
        sys.exit(f'Error: {path} is missing expected key {e}')

    cams = {}
    for fr in frames:
        try:
            label = str(fr.get('camera_label') or Path(fr['file_path']).stem)
            K = np.array([[fr['fl_x'], 0, fr['cx']], [0, fr['fl_y'], fr['cy']], [0, 0, 1]])
            c2w = np.array(fr['transform_matrix'], dtype=np.float64)
            c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV
            w2c = np.linalg.inv(c2w)
        except (KeyError, IndexError, ValueError, np.linalg.LinAlgError) as e:
            label = fr.get('camera_label', '<unknown>')
            print(f'  WARNING: skipping camera {label}: {e}')
            continue
        cams[label] = {'K': K, 'w2c': w2c, 'frame': fr}
    if not cams:
        sys.exit(f'Error: no usable cameras found in {path}')
    return data, cams


def _trailing_id(s: str):
    m = re.search(r'(\d+)$', s)
    return int(m.group(1)) if m else None


def match_camera(image_name, cam_labels):
    stem = Path(image_name).stem
    if stem in cam_labels:
        return stem
    # Different pipeline stages name the same physical camera differently
    # ("undistorted_0001" vs "Camera_0001" vs "0001") -- match on the
    # trailing numeric camera id when exactly one label shares it.
    stem_id = _trailing_id(stem)
    if stem_id is not None:
        id_matches = [label for label in cam_labels if _trailing_id(label) == stem_id]
        if len(id_matches) == 1:
            return id_matches[0]
    best = None
    for label in cam_labels:
        if label in stem or stem in label:
            if best is None or len(label) > len(best):
                best = label
    return best


def load_keypoints(kp2d_path, cam_labels):
    """Return observations[(tem, kp_idx)][camera_label] = (u, v, score)."""
    kp2d_path = Path(kp2d_path)
    obs = defaultdict(dict)
    if kp2d_path.is_file():
        try:
            with open(kp2d_path) as f:
                data = json.load(f)
            frames = data['frames']
        except json.JSONDecodeError as e:
            sys.exit(f'Error: {kp2d_path} is not valid JSON: {e}')
        except KeyError as e:
            sys.exit(f'Error: {kp2d_path} is missing expected key {e}')
        for fr in frames:
            try:
                image_name = fr['image_name']
            except KeyError as e:
                print(f'  WARNING: skipping malformed keypoint frame entry (missing key {e})')
                continue
            label = match_camera(image_name, cam_labels)
            if label is None or not fr.get('instances'):
                continue
            tem = str(fr.get('frame_id', '000000'))
            try:
                inst = fr['instances'][0]
                kps = inst['keypoints']
                scores = inst['keypoint_scores']
            except KeyError as e:
                print(f'  WARNING: skipping malformed keypoint instance for {image_name}: missing key {e}')
                continue
            for k, (pt, s) in enumerate(zip(kps, scores)):
                obs[(tem, k)][label] = (float(pt[0]), float(pt[1]), float(s))
    elif kp2d_path.is_dir():
        for cam_dir in sorted(p for p in kp2d_path.iterdir() if p.is_dir()):
            label = match_camera(cam_dir.name, cam_labels)
            if label is None:
                continue
            for jf in sorted(cam_dir.glob('*.json')):
                try:
                    with open(jf) as f:
                        data = json.load(f)
                    insts = data.get('instance_info') or data.get('instances') or []
                    if not insts:
                        continue
                    inst = insts[0]
                    kps = inst['keypoints']
                except (json.JSONDecodeError, KeyError) as e:
                    print(f'  WARNING: skipping {jf}: malformed ({e})')
                    continue
                scores = inst.get('keypoint_scores') or inst.get('keypoint_score') or []
                for k, pt in enumerate(kps):
                    s = float(scores[k]) if k < len(scores) else 1.0
                    obs[(jf.stem, k)][label] = (float(pt[0]), float(pt[1]), float(s))
    else:
        sys.exit(f'Error: {kp2d_path} not found (expected a file or a directory)')
    return obs


def triangulate_linear(Ks, w2cs, uvs, ws):
    A = []
    for (u, v), K, T, s in zip(uvs, Ks, w2cs, ws):
        P = K @ T[:3]
        A.append(np.sqrt(s) * (u * P[2] - P[0]))
        A.append(np.sqrt(s) * (v * P[2] - P[1]))
    _, _, Vt = np.linalg.svd(np.stack(A))
    X = Vt[-1]
    return X[:3] / (X[3] + 1e-12)


def similarity_align(src_pts, dst_pts):
    """Similarity transform (s, R, t) minimizing ||s R src + t - dst||."""
    mu_s, mu_d = src_pts.mean(0), dst_pts.mean(0)
    src0, dst0 = src_pts - mu_s, dst_pts - mu_d
    U, S, Vt = np.linalg.svd(src0.T @ dst0)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = (U @ D @ Vt).T
    s = (S * [1.0, 1.0, d]).sum() / (src0 ** 2).sum()
    t = mu_d - s * R @ mu_s
    return s, R, t


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--transforms', required=True)
    parser.add_argument('--kp2d', required=True)
    parser.add_argument('--out_transforms', required=True)
    parser.add_argument('--score_thr', type=float, default=0.5)
    parser.add_argument('--min_views', type=int, default=3)
    parser.add_argument('--huber_px', type=float, default=8.0,
                        help='Huber loss scale in pixels (residuals beyond this are downweighted).')
    parser.add_argument('--outlier_px', type=float, default=200.0,
                        help='Observations with initial reprojection error above this are dropped (sync glitches).')
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--report_only', action='store_true', help='Only report residuals, do not optimize.')
    parser.add_argument('--face_weight', type=float, default=1.0,
                        help='Extra weight multiplier applied to face/head keypoint observations '
                             '(goliath308 ids 0-4 and 69-307: nose/eyes/ears coarse, plus neck, '
                             'eyebrows, detailed eyes, nose, mouth, teeth, ears). Body joints and '
                             'hands (ids 5-68) are unaffected. Default 1.0 (no-op, unvalidated for '
                             'actually narrowing the left/right eye triangulation gap it was built '
                             'for -- see project memory. Try --face_weight 3.0 as a starting point '
                             'when picking this back up).')
    return parser.parse_args()


def triangulate_candidate_points(cams, cam_labels, cam_index, observations, score_thr, min_views):
    """Initial triangulation + observation table. Per point: {'xyz', 'obs': [(cam_idx, u, v, weight), ...], 'key'}."""
    points = []
    for key, cam_obs in observations.items():
        entries = [(cam_index[label], u, v, s) for label, (u, v, s) in cam_obs.items()
                   if s >= score_thr]
        if len(entries) < min_views:
            continue
        Ks = np.array([cams[cam_labels[e[0]]]['K'] for e in entries])
        Ts = np.array([cams[cam_labels[e[0]]]['w2c'] for e in entries])
        uvs = np.array([[e[1], e[2]] for e in entries])
        ws = np.array([e[3] for e in entries])
        xyz = triangulate_linear(Ks, Ts, uvs, ws)
        points.append({'xyz': xyz, 'obs': entries, 'key': key})
    return points


def report_residuals(rvecs, tvecs, pts, obs_points, Ks_all, cam_labels, tag):
    """pts supplies the 3D positions (list of {'xyz': ...} or raw arrays); obs_points supplies the
    per-point observation table (same length/order as pts -- may be the same list as pts)."""
    errs_by_cam = defaultdict(list)
    for p_idx, p in enumerate(pts):
        X = p['xyz'] if isinstance(p, dict) else p
        entries = obs_points[p_idx]['obs']
        for cam_idx, u, v, s in entries:
            R = Rotation.from_rotvec(rvecs[cam_idx]).as_matrix()
            x = R @ X + tvecs[cam_idx]
            if x[2] <= 1e-6:
                continue
            uv = Ks_all[cam_idx] @ (x / x[2])
            errs_by_cam[cam_idx].append(np.hypot(uv[0] - u, uv[1] - v))
    if not errs_by_cam:
        sys.exit(f'{tag}: no valid reprojections (all points behind cameras?)')
    all_errs = np.concatenate([np.asarray(v) for v in errs_by_cam.values()])
    print(f'{tag}: median {np.median(all_errs):.2f}px | mean {all_errs.mean():.2f}px | '
          f'p90 {np.percentile(all_errs, 90):.2f}px | n={len(all_errs)}')
    for cam_idx in sorted(errs_by_cam):
        e = np.array(errs_by_cam[cam_idx])
        print(f'    {cam_labels[cam_idx]}: median {np.median(e):7.2f}px  n={len(e)}')
    return all_errs


def drop_outliers(points, rvecs0, tvecs0, Ks_all, outlier_px, min_views):
    """Drop gross outlier observations (sync glitches, detector failures) before optimizing."""
    kept_points = []
    n_dropped = 0
    for p in points:
        entries = []
        for cam_idx, u, v, s in p['obs']:
            R = Rotation.from_rotvec(rvecs0[cam_idx]).as_matrix()
            x = R @ p['xyz'] + tvecs0[cam_idx]
            if x[2] <= 1e-6:
                n_dropped += 1
                continue
            uv = Ks_all[cam_idx] @ (x / x[2])
            if np.hypot(uv[0] - u, uv[1] - v) > outlier_px:
                n_dropped += 1
                continue
            entries.append((cam_idx, u, v, s))
        if len(entries) >= min_views:
            p['obs'] = entries
            kept_points.append(p)
    return kept_points, n_dropped


def build_observations(points, face_keypoint_ids, face_weight):
    n_face_obs = 0
    obs_flat = []
    for p_idx, p in enumerate(points):
        kp_idx = p['key'][1]
        w_mult = face_weight if kp_idx in face_keypoint_ids else 1.0
        for cam_idx, u, v, s in p['obs']:
            obs_flat.append((p_idx, cam_idx, u, v, np.sqrt(w_mult * s)))
            if w_mult != 1.0:
                n_face_obs += 1
    print(f'Applying {face_weight}x weight to {n_face_obs} face/head keypoint observations.')
    obs_p = np.array([o[0] for o in obs_flat])
    obs_c = np.array([o[1] for o in obs_flat])
    obs_uv = np.array([[o[2], o[3]] for o in obs_flat])
    obs_w = np.array([o[4] for o in obs_flat])
    return obs_flat, obs_p, obs_c, obs_uv, obs_w


def build_sparsity_pattern(obs_flat, n_cams, n_pts):
    n_obs = len(obs_flat)
    sparsity = lil_matrix((2 * n_obs, 6 * n_cams + 3 * n_pts), dtype=int)
    for i, (p_idx, cam_idx, *_rest) in enumerate(obs_flat):
        rows = (2 * i, 2 * i + 1)
        for r in rows:
            sparsity[r, 3 * cam_idx:3 * cam_idx + 3] = 1
            sparsity[r, 3 * n_cams + 3 * cam_idx:3 * n_cams + 3 * cam_idx + 3] = 1
            sparsity[r, 6 * n_cams + 3 * p_idx:6 * n_cams + 3 * p_idx + 3] = 1
    return sparsity


def huber_transform(r, delta):
    """Elementwise: returns r' such that 0.5*r'^2 == huber_rho(r, delta), preserving sign.
    Lets us apply Huber robustification on the TRUE pixel residual before any
    per-observation weight is folded in, so weighting (confidence, face_weight)
    can't shift a point across the outlier-robustification threshold."""
    abs_r = np.abs(r)
    out = r.copy()
    mask = abs_r > delta
    out[mask] = np.sign(r[mask]) * np.sqrt(delta * (2 * abs_r[mask] - delta))
    return out


def run_bundle_adjustment(x0, n_cams, n_pts, obs_p, obs_c, obs_uv, obs_w, Ks_all, huber_px, max_iters, sparsity):
    def unpack(x):
        rv = x[:3 * n_cams].reshape(n_cams, 3)
        tv = x[3 * n_cams:6 * n_cams].reshape(n_cams, 3)
        pts = x[6 * n_cams:].reshape(n_pts, 3)
        return rv, tv, pts

    fx = np.array([Ks_all[c][0, 0] for c in range(n_cams)])
    fy = np.array([Ks_all[c][1, 1] for c in range(n_cams)])
    cx = np.array([Ks_all[c][0, 2] for c in range(n_cams)])
    cy = np.array([Ks_all[c][1, 2] for c in range(n_cams)])

    def fun(x):
        rv, tv, pts = unpack(x)
        Rmats = Rotation.from_rotvec(rv).as_matrix()
        X = pts[obs_p]
        Rm = Rmats[obs_c]
        xc = np.einsum('nij,nj->ni', Rm, X) + tv[obs_c]
        z = np.clip(xc[:, 2], 1e-6, None)
        u = fx[obs_c] * xc[:, 0] / z + cx[obs_c]
        v = fy[obs_c] * xc[:, 1] / z + cy[obs_c]
        raw_res = np.stack([u - obs_uv[:, 0], v - obs_uv[:, 1]], axis=1)
        hres = huber_transform(raw_res, huber_px)
        res = hres * obs_w[:, None]
        return res.ravel()

    print('Optimizing (Huber-robust bundle adjustment, intrinsics fixed)...')
    # loss left at scipy's default ('linear') -- Huber robustification now happens inside
    # fun() itself, on the true pixel residual before per-observation weighting is applied.
    result = least_squares(fun, x0, jac_sparsity=sparsity, method='trf',
                           max_nfev=max_iters, verbose=2, x_scale='jac')
    return unpack(result.x)


def remove_gauge_drift(rv1, tv1, pts1, rvecs0, tvecs0, n_cams):
    """Similarity-align refined camera centers back onto the input ones (scene scale/origin
    are otherwise free to drift under bundle adjustment with no fixed/anchor camera)."""
    centers0 = np.array([-Rotation.from_rotvec(rvecs0[i]).as_matrix().T @ tvecs0[i] for i in range(n_cams)])
    centers1 = np.array([-Rotation.from_rotvec(rv1[i]).as_matrix().T @ tv1[i] for i in range(n_cams)])
    s, R_al, t_al = similarity_align(centers1, centers0)
    for i in range(n_cams):
        R_c = Rotation.from_rotvec(rv1[i]).as_matrix()
        # world' = s R_al world + t_al  =>  cam_from_world' composition
        R_new = R_c @ R_al.T
        t_new = s * tv1[i] - R_new @ t_al
        rv1[i] = Rotation.from_matrix(R_new).as_rotvec()
        tv1[i] = t_new
    pts_aligned = (s * (R_al @ pts1.T)).T + t_al
    return rv1, tv1, pts_aligned


def write_refined_transforms(data, cams, cam_labels, cam_index, rv1, tv1, out_path):
    for label in cam_labels:
        i = cam_index[label]
        w2c = np.eye(4)
        w2c[:3, :3] = Rotation.from_rotvec(rv1[i]).as_matrix()
        w2c[:3, 3] = tv1[i]
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL
        cams[label]['frame']['transform_matrix'] = c2w.tolist()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Wrote {out_path}')


def main():
    args = parse_args()

    # goliath308 (308kp) only -- no backward compatibility with older/smaller keypoint
    # layouts (e.g. Sapiens 1 or coco_wholebody133); see module docstring.
    FACE_KEYPOINT_IDS = set(range(0, 5)) | set(range(69, 308))

    data, cams = load_transforms(args.transforms)
    cam_labels = sorted(cams.keys())
    cam_index = {label: i for i, label in enumerate(cam_labels)}
    print(f'{len(cam_labels)} cameras.')

    observations = load_keypoints(args.kp2d, cam_labels)
    print(f'{len(observations)} candidate 3D points (time x keypoint).')

    points = triangulate_candidate_points(cams, cam_labels, cam_index, observations,
                                          args.score_thr, args.min_views)
    print(f'{len(points)} points with >= {args.min_views} confident views.')
    if not points:
        sys.exit('No usable points - check score_thr / keypoint inputs.')

    Ks_all = np.array([cams[label]['K'] for label in cam_labels])

    rvecs0 = np.array([Rotation.from_matrix(cams[label]['w2c'][:3, :3]).as_rotvec() for label in cam_labels])
    tvecs0 = np.array([cams[label]['w2c'][:3, 3] for label in cam_labels])

    print('\nInitial reprojection residuals (subject keypoints):')
    init_errs = report_residuals(rvecs0, tvecs0, points, points, Ks_all, cam_labels, 'BEFORE')

    if args.report_only:
        return

    points, n_dropped = drop_outliers(points, rvecs0, tvecs0, Ks_all, args.outlier_px, args.min_views)
    print(f'\nDropped {n_dropped} outlier observations; optimizing over {len(points)} points.')

    n_cams = len(cam_labels)
    n_pts = len(points)
    obs_flat, obs_p, obs_c, obs_uv, obs_w = build_observations(points, FACE_KEYPOINT_IDS, args.face_weight)
    print(f'{len(obs_flat)} observations, {6 * n_cams + 3 * n_pts} parameters.')

    x0 = np.concatenate([rvecs0.ravel(), tvecs0.ravel(),
                         np.array([p['xyz'] for p in points]).ravel()])
    sparsity = build_sparsity_pattern(obs_flat, n_cams, n_pts)

    rv1, tv1, pts1 = run_bundle_adjustment(x0, n_cams, n_pts, obs_p, obs_c, obs_uv, obs_w, Ks_all,
                                           args.huber_px, args.max_iters, sparsity)

    rv1, tv1, pts_aligned = remove_gauge_drift(rv1, tv1, pts1, rvecs0, tvecs0, n_cams)

    print('\nRefined reprojection residuals:')
    pts_report = [{'xyz': pts_aligned[i], 'obs': points[i]['obs']} for i in range(n_pts)]
    final_errs = report_residuals(rv1, tv1, pts_report, points, Ks_all, cam_labels, 'AFTER ')
    print(f'\nMedian residual: {np.median(init_errs):.2f}px -> {np.median(final_errs):.2f}px')

    write_refined_transforms(data, cams, cam_labels, cam_index, rv1, tv1, args.out_transforms)


if __name__ == '__main__':
    main()
