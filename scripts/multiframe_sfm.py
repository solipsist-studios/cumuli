# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Multi-timestamp SfM for static multi-camera rigs.

Instead of estimating camera poses from a single time instant (12 images, weak
wide-baseline matching), this samples N timestamps from each camera's video and
reconstructs all N*num_cameras images jointly, with all images from the same
physical camera sharing one COLMAP camera (intrinsics). Because the rig is
static, every timestamp is an independent measurement of the same camera pose:
tracks span both space and time, constraining the bundle adjustment far more
strongly, and moving people are rejected as outliers by cross-time geometric
verification. Final per-camera poses are robust averages across timestamps, and
the per-camera pose spread is reported as a health metric.

Requires the `hloc` environment (hloc + pycolmap + ffmpeg).

Example:
    conda run --live-stream -n hloc python multiframe_sfm.py \
        --videos_dir ./movies \
        --init_transforms ./transforms_hloc.json \
        --sync_json ./sync_offsets.json \
        --outputs_dir ./multiframe_sfm \
        --num_timestamps 12 --refine_intrinsics
"""

import argparse
import json
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

import pycolmap
from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction

from image_formats import SUPPORTED_IMAGE_EXTS, SUPPORTED_VIDEO_EXTS

CAMERA_MODEL_PARAM_BUILDERS = {
    # transforms.json intrinsics -> COLMAP params vector
    'OPENCV': lambda fr: [fr['fl_x'], fr['fl_y'], fr['cx'], fr['cy'],
                          fr.get('k1', 0.0), fr.get('k2', 0.0), fr.get('p1', 0.0), fr.get('p2', 0.0)],
    'OPENCV_FISHEYE': lambda fr: [fr['fl_x'], fr['fl_y'], fr['cx'], fr['cy'],
                                  fr.get('k1', 0.0), fr.get('k2', 0.0), fr.get('k3', 0.0), fr.get('k4', 0.0)],
    'PINHOLE': lambda fr: [fr['fl_x'], fr['fl_y'], fr['cx'], fr['cy']],
}


def probe_video(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-count_frames', '-show_entries', 'stream=nb_read_frames,r_frame_rate,width,height',
         '-of', 'json', str(path)],
        capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)['streams'][0]
    num, den = stream['r_frame_rate'].split('/')
    return {
        'nb_frames': int(stream['nb_read_frames']),
        'fps': float(num) / float(den),
        'width': int(stream['width']),
        'height': int(stream['height']),
    }


def extract_frame(video_path, frame_index, out_path, quality=2):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-v', 'error', '-y',
           '-i', str(video_path),
           '-vf', f'select=eq(n\\,{frame_index})',
           '-vsync', '0', '-frames:v', '1', '-q:v', str(quality), str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    if not out_path.exists():
        raise RuntimeError(f'Failed to extract frame {frame_index} from {video_path}')


def load_init_intrinsics(init_transforms_path, camera_labels):
    """Load per-camera intrinsics from a transforms-style JSON.

    Frames are matched to camera labels by camera_label field or file_path
    containing the label. Returns {label: (model, width, height, params)}.
    """
    with open(init_transforms_path) as f:
        data = json.load(f)

    out = {}
    frames = data.get('frames', [])
    for label in camera_labels:
        match = None
        for fr in frames:
            cam_label = str(fr.get('camera_label', ''))
            if cam_label == label or cam_label.startswith(label) or label in Path(str(fr.get('file_path', ''))).stem:
                match = fr
                break
        if match is None:
            raise KeyError(f'No frame in {init_transforms_path} matches camera label {label!r}')
        model = str(match.get('camera_model', data.get('camera_model', 'OPENCV'))).upper()
        if model not in CAMERA_MODEL_PARAM_BUILDERS:
            raise ValueError(f'Unsupported camera model {model}')
        params = CAMERA_MODEL_PARAM_BUILDERS[model](match)
        out[label] = (model, int(match.get('w', data.get('w'))), int(match.get('h', data.get('h'))), params)
    return out


def build_database(database_path, image_names_by_camera, intrinsics_by_camera):
    """Create a COLMAP database with one shared camera per physical camera."""
    if database_path.exists():
        database_path.unlink()
    reconstruction.create_empty_db(database_path)

    camera_ids = {}
    with pycolmap.Database.open(database_path) as db:
        image_id = 1
        for cam_idx, (label, image_names) in enumerate(sorted(image_names_by_camera.items()), start=1):
            model, width, height, params = intrinsics_by_camera[label]
            camera = pycolmap.Camera.create_from_model_name(cam_idx, model, params[0], width, height)
            camera.params = [float(p) for p in params]
            camera.has_prior_focal_length = True
            db.write_camera(camera, use_camera_id=True)
            camera_ids[label] = cam_idx
            for name in sorted(image_names):
                image = pycolmap.Image(name=name, camera_id=cam_idx, image_id=image_id)
                db.write_image(image, use_image_id=True)
                image_id += 1
    return camera_ids


def robust_pose_average(cam_from_world_list, max_rot_spread_deg=10.0):
    """Average a list of cam_from_world 4x4 matrices with outlier rejection.

    Returns (avg_cam_from_world, stats_dict).
    """
    Rs = np.array([T[:3, :3] for T in cam_from_world_list])
    centers = np.array([-T[:3, :3].T @ T[:3, 3] for T in cam_from_world_list])

    med_center = np.median(centers, axis=0)
    dists = np.linalg.norm(centers - med_center, axis=1)
    mad = np.median(dists) + 1e-9
    keep = dists <= max(5 * mad, 1e-6)

    def mean_rotation(R_stack):
        M = R_stack.sum(axis=0)
        U, _, Vt = np.linalg.svd(M)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        return U @ D @ Vt

    def rotation_angle_deg(Ra, Rb):
        return np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))

    # Vote-based rotation-inlier detection: for each position-inlier
    # candidate, count how many OTHER position-inliers are within
    # max_rot_spread_deg of it; whichever candidate has the most support
    # defines the inlier set. A plain "average everyone, then see who's
    # close to the average" approach lets a single extreme outlier drag the
    # average far enough that even the good poses fail the deviation check
    # (verified: this broke down for a single outlier beyond ~50-60 degrees,
    # ironically making MORE extreme outliers more likely to survive than
    # moderate ones). Voting never computes a mean contaminated by the
    # outlier in the first place, so it correctly isolates it regardless of
    # magnitude.
    candidates = np.flatnonzero(keep)
    best_support = keep & False
    for i in candidates:
        support = np.zeros(len(Rs), dtype=bool)
        for j in candidates:
            if rotation_angle_deg(Rs[i], Rs[j]) <= max_rot_spread_deg:
                support[j] = True
        if support.sum() > best_support.sum():
            best_support = support
    if best_support.sum() > 0:
        keep = best_support

    R_mean = mean_rotation(Rs[keep])
    rot_errs = np.array([rotation_angle_deg(R, R_mean) for R in Rs])
    center_mean = centers[keep].mean(axis=0)

    T = np.eye(4)
    T[:3, :3] = R_mean
    T[:3, 3] = -R_mean @ center_mean

    stats = {
        'num_views': int(len(Rs)),
        'num_inliers': int(keep.sum()),
        'center_spread': float(np.linalg.norm(centers[keep] - center_mean, axis=1).mean()),
        'center_spread_max': float(np.linalg.norm(centers[keep] - center_mean, axis=1).max()),
        'rot_spread_deg': float(rot_errs[keep].mean()),
        'rot_spread_deg_max': float(rot_errs[keep].max()),
    }
    return T, stats


def write_transforms(out_path, avg_poses, intrinsics_by_camera, undistorted_intrinsics=None,
                     file_path_format='images/{label}.png', ply_file_path=None):
    """Write a nerfstudio-style transforms.json (OpenGL c2w convention)."""
    frames = []
    for label in sorted(avg_poses.keys()):
        cam_from_world = avg_poses[label]
        c2w = np.linalg.inv(cam_from_world)
        c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL
        if undistorted_intrinsics and label in undistorted_intrinsics:
            model, w, h, params = undistorted_intrinsics[label]
        else:
            model, w, h, params = intrinsics_by_camera[label]
        frame = {
            'file_path': file_path_format.format(label=label),
            'camera_label': label,
            'transform_matrix': c2w.tolist(),
            'camera_model': model,
            'w': w, 'h': h,
            'fl_x': params[0], 'fl_y': params[1], 'cx': params[2], 'cy': params[3],
        }
        extra = params[4:]
        if model == 'OPENCV_FISHEYE':
            for key, value in zip(['k1', 'k2', 'k3', 'k4'], extra):
                frame[key] = value
        elif model == 'OPENCV':
            for key, value in zip(['k1', 'k2', 'p1', 'p2'], extra):
                frame[key] = value
        frames.append(frame)
    data = {'frames': frames}
    if ply_file_path:
        data['ply_file_path'] = ply_file_path
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_undistorted_intrinsics(calib_dir, camera_labels):
    """Load per-camera undistorted pinhole pkls named like undistorted_<label>*.pkl or <label>*.pkl."""
    calib_dir = Path(calib_dir)
    out = {}
    for label in camera_labels:
        candidates = sorted(list(calib_dir.glob(f'undistorted_{label}*.pkl')) + list(calib_dir.glob(f'{label}*.pkl')))
        if not candidates:
            raise FileNotFoundError(f'No undistorted calibration pkl for camera {label!r} in {calib_dir}')
        with open(candidates[0], 'rb') as f:
            data = pickle.load(f)
        m = np.asarray(data['camera_matrix'])
        w, h = data.get('image_size', (None, None))
        out[label] = ('OPENCV', int(w), int(h),
                      [float(m[0, 0]), float(m[1, 1]), float(m[0, 2]), float(m[1, 2]), 0.0, 0.0, 0.0, 0.0])
    return out


def export_points_ply(rec, out_path, max_error=2.0, min_track_length=3):
    points = []
    colors = []
    for _, p in rec.points3D.items():
        if p.error > max_error or p.track.length() < min_track_length:
            continue
        points.append(p.xyz)
        colors.append(p.color)
    points = np.array(points)
    colors = np.array(colors, dtype=np.uint8)
    with open(out_path, 'wb') as f:
        header = (
            'ply\nformat binary_little_endian 1.0\n'
            f'element vertex {len(points)}\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'end_header\n'
        )
        f.write(header.encode('ascii'))
        for xyz, rgb in zip(points, colors):
            f.write(np.asarray(xyz, dtype='<f4').tobytes())
            f.write(np.asarray(rgb, dtype=np.uint8).tobytes())
    return len(points)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--videos_dir', help='Directory of per-camera videos (one file per camera).')
    parser.add_argument('--frames_root', help='Alternative to --videos_dir: root of pre-extracted frames/<camera>/<time>.jpg.')
    parser.add_argument('--init_transforms', required=True,
                        help='transforms-style JSON providing per-camera initial intrinsics (camera_model + params).')
    parser.add_argument('--sync_json', help='Output of measure_sync.py; per-camera integer frame_shift is applied when sampling.')
    parser.add_argument('--outputs_dir', required=True)
    parser.add_argument('--num_timestamps', type=int, default=12)
    parser.add_argument('--timestamps', help='Comma-separated explicit frame indices (overrides --num_timestamps).')
    parser.add_argument('--feature_type', default='superpoint', choices=['superpoint', 'aliked'])
    parser.add_argument('--resize_max', type=int, default=2048)
    parser.add_argument('--max_keypoints', type=int, default=8192)
    parser.add_argument('--refine_intrinsics', action='store_true',
                        help='Refine focal length + distortion during BA (shared per camera; well-posed with many timestamps).')
    parser.add_argument('--refine_principal_point', action='store_true')
    parser.add_argument('--filter_max_reproj_error', type=float, default=4.0)
    parser.add_argument('--undistorted_calibration_dir',
                        help='Optional dir of per-camera undistorted pinhole pkls; if given, the output transforms uses those intrinsics.')
    parser.add_argument('--file_path_format', default='images/undistorted_{label}.png')
    parser.add_argument('--max_rot_spread_deg', type=float, default=10.0)
    return parser.parse_args()


def load_sync_shifts(sync_json):
    if not sync_json:
        return {}
    with open(sync_json) as f:
        sync = json.load(f)
    return {Path(name).stem: int(entry.get('frame_shift') or 0) for name, entry in sync['cameras'].items()}


def sample_frames_from_videos(videos_dir, frames_dir, sync_shifts, num_timestamps, timestamps_arg):
    videos = sorted(p for p in Path(videos_dir).iterdir() if p.suffix.lower() in SUPPORTED_VIDEO_EXTS)
    if not videos:
        raise SystemExit(f'No videos found in {videos_dir}')
    print(f'Found {len(videos)} videos.')
    infos = {v.stem: probe_video(v) for v in videos}
    min_frames = min(i['nb_frames'] for i in infos.values())
    max_shift = max([abs(s) for s in sync_shifts.values()] + [0])

    if timestamps_arg:
        timestamps = [int(t) for t in timestamps_arg.split(',')]
    else:
        lo, hi = max_shift, min_frames - 1 - max_shift
        timestamps = np.unique(np.linspace(lo, hi, num_timestamps).round().astype(int)).tolist()
    print(f'Sampling timestamps: {timestamps}')

    image_names_by_camera = defaultdict(list)
    for video in videos:
        label = video.stem
        shift = sync_shifts.get(label, 0)
        for t in timestamps:
            idx = int(np.clip(t + shift, 0, infos[label]['nb_frames'] - 1))
            rel = f'{label}/{t:06d}.jpg'
            out_path = frames_dir / rel
            if not out_path.exists():
                extract_frame(video, idx, out_path)
            image_names_by_camera[label].append(rel)
        print(f'  extracted {label} (shift {shift:+d})')
    return image_names_by_camera


def collect_frames_from_root(frames_root):
    image_names_by_camera = defaultdict(list)
    for cam_dir in sorted(p for p in Path(frames_root).iterdir() if p.is_dir()):
        for img in sorted(cam_dir.iterdir()):
            if img.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                image_names_by_camera[cam_dir.name].append(f'{cam_dir.name}/{img.name}')
    if not image_names_by_camera:
        raise SystemExit(f'No frames found under {frames_root}')
    return image_names_by_camera


def gather_frames(args, frames_dir):
    """Returns (image_names_by_camera, frames_dir) -- frames_dir may be overridden to --frames_root."""
    sync_shifts = load_sync_shifts(args.sync_json)
    if args.videos_dir:
        return sample_frames_from_videos(args.videos_dir, frames_dir, sync_shifts,
                                         args.num_timestamps, args.timestamps), frames_dir
    if args.frames_root:
        return collect_frames_from_root(args.frames_root), Path(args.frames_root)
    raise SystemExit('Provide --videos_dir or --frames_root')


def extract_and_match_features(args, frames_dir, outputs_dir, all_images):
    if args.feature_type == 'superpoint':
        feature_conf = {
            'output': f'feats-superpoint-n{args.max_keypoints}-r{args.resize_max}',
            'model': {'name': 'superpoint', 'nms_radius': 3, 'max_keypoints': args.max_keypoints},
            'preprocessing': {'grayscale': True, 'resize_max': args.resize_max},
        }
        matcher_conf = match_features.confs['superpoint+lightglue']
    else:
        feature_conf = {
            'output': f'feats-aliked-n{args.max_keypoints}-r{args.resize_max}',
            'model': {'name': 'aliked', 'model_name': 'aliked-n16', 'max_num_keypoints': args.max_keypoints},
            'preprocessing': {'grayscale': False, 'resize_max': args.resize_max},
        }
        matcher_conf = match_features.confs['aliked+lightglue']

    print('Extracting features...')
    feature_path = extract_features.main(feature_conf, frames_dir, outputs_dir, image_list=all_images)
    sfm_pairs = outputs_dir / 'pairs-exhaustive.txt'
    pairs_from_exhaustive.main(sfm_pairs, image_list=all_images)
    with open(sfm_pairs) as f:
        n_pairs = sum(1 for _ in f)
    print(f'Matching {n_pairs} pairs...')
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs_dir)
    return feature_path, sfm_pairs, match_path


def run_sfm_reconstruction(args, frames_dir, outputs_dir, image_names_by_camera, intrinsics_by_camera,
                           feature_path, sfm_pairs, match_path):
    sfm_dir = outputs_dir / 'sfm'
    sfm_dir.mkdir(exist_ok=True)
    database_path = sfm_dir / 'database.db'
    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, sfm_dir / 'colmap.LOG.')

    build_database(database_path, image_names_by_camera, intrinsics_by_camera)
    image_ids = reconstruction.get_image_ids(database_path)
    with pycolmap.Database.open(database_path) as db:
        reconstruction.import_features(image_ids, db, feature_path)
        reconstruction.import_matches(image_ids, db, sfm_pairs, match_path,
                                      min_match_score=None, skip_geometric_verification=False)
    reconstruction.estimation_and_geometric_verification(database_path, sfm_pairs, verbose=False)

    # Intrinsics stay LOCKED during incremental mapping: letting the mapper refine
    # fisheye parameters from the first few registered images destroys them and the
    # model stops growing. Refinement (if requested) happens in a final global BA
    # once all images and tracks are in.
    mapper_options = {
        'ba_refine_focal_length': False,
        'ba_refine_extra_params': False,
        'ba_refine_principal_point': False,
        'multiple_models': False,
        'max_num_models': 1,
        'mapper': {'filter_max_reproj_error': args.filter_max_reproj_error},
        'triangulation': {
            'merge_max_reproj_error': args.filter_max_reproj_error,
            'complete_max_reproj_error': args.filter_max_reproj_error,
        },
    }
    print('Running incremental mapping (intrinsics locked)...')
    rec = reconstruction.run_reconstruction(sfm_dir, database_path, frames_dir,
                                            verbose=False, options=mapper_options)
    if rec is None:
        raise SystemExit('Reconstruction failed.')
    print(rec.summary())

    if args.refine_intrinsics:
        print('Global bundle adjustment with shared intrinsics refinement...')
        ba_options = pycolmap.BundleAdjustmentOptions()
        ba_options.refine_focal_length = True
        ba_options.refine_extra_params = True
        ba_options.refine_principal_point = bool(args.refine_principal_point)
        ba_options.print_summary = False
        pycolmap.bundle_adjustment(rec, ba_options)
        print(f'After refinement: mean reprojection error '
              f'{rec.compute_mean_reprojection_error():.3f}px, '
              f'{rec.num_points3D()} points')
        rec.write(sfm_dir)

    return rec, sfm_dir


def average_camera_poses(rec, camera_labels, max_rot_spread_deg):
    poses_by_camera = defaultdict(dict)
    for _, img in rec.images.items():
        label = img.name.split('/')[0]
        pose = img.cam_from_world()
        T = np.eye(4)
        T[:3, :] = pose.matrix() if hasattr(pose, 'matrix') else np.asarray(pose)
        poses_by_camera[label][img.name] = T

    missing = [label for label in camera_labels if label not in poses_by_camera]
    if missing:
        print(f'WARNING: cameras with no registered images: {missing}')

    avg_poses = {}
    per_camera_stats = {}
    print('\nPer-camera pose repeatability across timestamps:')
    for label in sorted(poses_by_camera.keys()):
        T_list = list(poses_by_camera[label].values())
        T_avg, stats = robust_pose_average(T_list, max_rot_spread_deg)
        avg_poses[label] = T_avg
        per_camera_stats[label] = stats
        print(f'  {label}: {stats["num_inliers"]}/{stats["num_views"]} views | '
              f'center spread {stats["center_spread"]:.4f} (max {stats["center_spread_max"]:.4f}) | '
              f'rot spread {stats["rot_spread_deg"]:.3f} deg (max {stats["rot_spread_deg_max"]:.3f})')
    return avg_poses, per_camera_stats


def extract_refined_intrinsics(rec):
    refined_intrinsics = {}
    label_by_camera_id = {}
    for _, img in rec.images.items():
        label_by_camera_id[img.camera_id] = img.name.split('/')[0]
    for cam_id, cam in rec.cameras.items():
        label = label_by_camera_id.get(cam_id)
        if label is None:
            continue
        model_name = cam.model.name if hasattr(cam.model, 'name') else str(cam.model)
        model_name = model_name.replace('CameraModelId.', '')
        refined_intrinsics[label] = (model_name, cam.width, cam.height, list(map(float, cam.params)))
    return refined_intrinsics


def export_results(args, outputs_dir, rec, camera_labels, all_images, avg_poses, per_camera_stats,
                   refined_intrinsics):
    undistorted_intrinsics = None
    if args.undistorted_calibration_dir:
        undistorted_intrinsics = load_undistorted_intrinsics(args.undistorted_calibration_dir, camera_labels)

    ply_path = outputs_dir / 'background_points.ply'
    n_points = export_points_ply(rec, ply_path)
    print(f'Exported {n_points} filtered 3D points to {ply_path}')

    write_transforms(outputs_dir / 'transforms_multiframe.json', avg_poses,
                     refined_intrinsics, undistorted_intrinsics,
                     file_path_format=args.file_path_format,
                     ply_file_path='background_points.ply')

    report = {
        'num_cameras': len(camera_labels),
        'num_images': len(all_images),
        'reconstruction': {
            'num_points3D': rec.num_points3D(),
            'mean_track_length': rec.compute_mean_track_length(),
            'mean_reprojection_error': rec.compute_mean_reprojection_error(),
            'num_reg_images': rec.num_reg_images() if hasattr(rec, 'num_reg_images') else len(rec.images),
        },
        'per_camera_pose_stats': per_camera_stats,
        'refined_intrinsics': {k: {'model': v[0], 'w': v[1], 'h': v[2], 'params': v[3]}
                               for k, v in refined_intrinsics.items()},
    }
    with open(outputs_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {outputs_dir / 'transforms_multiframe.json'} and {outputs_dir / 'report.json'}")


def main():
    args = parse_args()

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = outputs_dir / 'frames'

    image_names_by_camera, frames_dir = gather_frames(args, frames_dir)
    camera_labels = sorted(image_names_by_camera.keys())
    all_images = [n for label in camera_labels for n in image_names_by_camera[label]]
    print(f'{len(camera_labels)} cameras, {len(all_images)} images total.')

    intrinsics_by_camera = load_init_intrinsics(args.init_transforms, camera_labels)

    feature_path, sfm_pairs, match_path = extract_and_match_features(args, frames_dir, outputs_dir, all_images)

    rec, sfm_dir = run_sfm_reconstruction(args, frames_dir, outputs_dir, image_names_by_camera,
                                          intrinsics_by_camera, feature_path, sfm_pairs, match_path)

    avg_poses, per_camera_stats = average_camera_poses(rec, camera_labels, args.max_rot_spread_deg)
    refined_intrinsics = extract_refined_intrinsics(rec)

    export_results(args, outputs_dir, rec, camera_labels, all_images, avg_poses, per_camera_stats,
                   refined_intrinsics)


if __name__ == '__main__':
    main()
