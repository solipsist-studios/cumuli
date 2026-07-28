#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
build_colmap_sparse.py

Builds a COLMAP sparse/0 directory (cameras.txt, images.txt, points3D.txt)
from a nerfstudio-style transforms.json (OpenGL/Blender convention: +X
right, +Y up, +Z back) and a triangulated 3D keypoint point cloud (.ply,
x/y/z only -- e.g. poses_pcd_fullres/<tem_label>.ply from
triangulate_and_project_keypoints.py run against the ORIGINAL,
un-cropped cameras/keypoints).

Coordinate conversion matches Brush's own COLMAP loader exactly (see
deps/brush/crates/brush-dataset/src/formats/mod.rs, opengl_c2w_to_pose):
c2w's Y and Z axis columns are negated to go from nerfstudio/OpenGL to
COLMAP's convention (+X right, +Y down, +Z forward), then inverted to get
the world-to-camera rotation/translation COLMAP's images.txt expects.

Per Brush's colmap-reader (crates/colmap-reader/src/lib.rs), points3D.txt
lines only require id/xyz/rgb/error -- the track (image_id, point2d_idx
pairs) can be empty, so we don't need to reconstruct exact 2D-3D
correspondences. Our point cloud has no per-point color, so a flat gray
placeholder is used.

Splat-seeded init (--seed_splat_ply, alternative to --points_ply): for
per-frame sequences (render_frame_sequence.py), source points3D from a
neighboring/reference frame's TRAINED splat instead of the triangulated
keypoint cloud: high-opacity Gaussians are subsampled (deterministically)
into the init points, with color decoded from the splat's SH DC term. A
static point cloud sits far from a fast-moving subject at most
timesteps, forcing densification to migrate/rebuild mass from scratch
every frame; seeding from an already-solved neighboring frame starts
optimization roughly where the subject actually is. Measured on real
runs: same training wall-clock as a sparse cold start, visibly cleaner
result (96%+ mask-consistency keep rate vs 54% from a static-cloud
start). Prefer a mask-filtered splat (filter_splat_by_masks.py output)
as the seed so junk isn't propagated forward.

Images are referenced by bare filename (Brush's find_image_by_name does a
suffix search across the whole dataset dir, skipping anything under a
"masks" folder) -- so --out_dir should be the dataset root that already
contains (or will contain) the actual image files, e.g. pass the same
directory as your images_flat parent.

Mask baking (--masks_dir): Brush has been observed to SILENTLY ignore a
separate masks/ folder when images and masks share the same stem and
extension (e.g. both .png) -- it just trains on the full unmasked scene
with no error. The reliable route is baking each camera's mask into the
image's alpha channel; Brush auto-detects the alpha channel and applies
its `--match-alpha-weight` loss automatically, no special training flag
needed. Passing --masks_dir here produces that RGBA image set and points
images.txt at it automatically. Do NOT dilate the masks before baking --
dilated alpha supervision teaches a white background fringe at the
silhouette (use clean_masks.py's output, which never dilates).

--images_dir: where the SOURCE images actually live, for the baking
lookup. Defaults to out_dir/image_subdir (the historical assumption that
images live inside the dataset root), but --out_dir need not contain the
images at all -- e.g. in a pipeline where images_flat/ is a sibling of
the COLMAP dataset root rather than nested inside it, pass --images_dir
pointing at the real location. --image_subdir still controls the NAME
recorded in images.txt either way.

Usage:
    python3 build_colmap_sparse.py \\
        --transforms /path/to/transforms.json \\
        --points_ply /path/to/poses_pcd_fullres/000000.ply \\
        --out_dir /path/to/dataset_root \\
        [--image_subdir images_flat] [--images_dir /path/to/images_flat] \\
        [--masks_dir /path/to/fmasks_clean --rgba_subdir images_rgba]

Output:
    out_dir/sparse/0/cameras.txt
    out_dir/sparse/0/images.txt
    out_dir/sparse/0/points3D.txt
    out_dir/<rgba_subdir>/<camera_label>.png   (only with --masks_dir)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation

SH_C0 = 0.28209479177387814  # SH DC basis constant: rgb = f_dc * SH_C0 + 0.5


def opengl_c2w_to_colmap_w2c(c2w: np.ndarray):
    """Matches Brush's opengl_c2w_to_pose, but returns COLMAP's w2c instead
    of Brush's internal c2w pose (see module docstring)."""
    c2w_cv = c2w.copy()
    c2w_cv[:3, 1] *= -1
    c2w_cv[:3, 2] *= -1
    w2c = np.linalg.inv(c2w_cv)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return R, t


def bake_rgba(image_path: Path, mask_path: Path, out_path: Path):
    from PIL import Image
    import numpy as np
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.NEAREST)
    img_arr = np.array(img)
    mask_arr = np.array(mask)
    # Zero RGB wherever fully transparent. The source photo's background isn't
    # masked out of the RGB channels by default, so a bright/white background
    # pixel can sit right behind the silhouette edge; any later bilinear/mipmap
    # filtering (e.g. Brush's own image loading at --max-resolution) then blends
    # that "invisible" color into visible edge pixels, producing a white/bright
    # halo at the boundary. Zeroing it here removes the contamination at the source.
    img_arr[mask_arr == 0] = 0
    rgba_arr = np.dstack([img_arr, mask_arr])
    rgba = Image.fromarray(rgba_arr, mode="RGBA")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out_path)


def bake_masks(frames, masks_dir: Path, images_dir: Path, out_dir: Path, rgba_subdir: str):
    print(f"Baking masks from {masks_dir} (images from {images_dir}) "
          f"into {out_dir / rgba_subdir} ...")
    n_baked, n_missing = 0, []
    for fr in frames:
        label = fr["camera_label"]
        image_path = images_dir / f"{label}{Path(fr['file_path']).suffix}"
        mask_path = masks_dir / f"{label}.png"
        out_path = out_dir / rgba_subdir / f"{label}.png"
        if not image_path.is_file() or not mask_path.is_file():
            n_missing.append(label)
            continue
        bake_rgba(image_path, mask_path, out_path)
        n_baked += 1
    print(f"  Baked {n_baked}/{len(frames)} cameras.")
    if n_missing:
        print(f"  WARNING: missing image or mask for cameras: {n_missing} (left unbaked, will 404 in images.txt)")


def write_cameras_txt(sparse_dir: Path, frames):
    """One PINHOLE camera per image (crops differ per camera)."""
    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, fr in enumerate(frames, start=1):
            f.write(f"{i} PINHOLE {fr['w']} {fr['h']} {fr['fl_x']} {fr['fl_y']} {fr['cx']} {fr['cy']}\n")


def write_images_txt(sparse_dir: Path, frames, image_subdir: str, masks_dir, rgba_subdir: str):
    """Image line + empty points2D line, one camera_id per image (1:1)."""
    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, fr in enumerate(frames, start=1):
            c2w = np.array(fr["transform_matrix"])
            R, t = opengl_c2w_to_colmap_w2c(c2w)
            qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
            if masks_dir is not None:
                name = f"{rgba_subdir}/{fr['camera_label']}.png"
            else:
                name = f"{image_subdir}/{fr['camera_label']}{Path(fr['file_path']).suffix}"
            f.write(f"{i} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i} {name}\n")
            f.write("\n")


def write_points3d_from_splat(sparse_dir: Path, seed_ply: Path, num_points: int,
                              opacity_thresh: float):
    """Subsample a trained splat into points3D.txt (see module docstring)."""
    ply = PlyData.read(str(seed_ply))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
    fdc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1)
    keep = np.isfinite(xyz).all(axis=1) & np.isfinite(fdc).all(axis=1) & (opacity > opacity_thresh)
    xyz, fdc = xyz[keep], fdc[keep]
    if len(xyz) > num_points:
        idx = np.random.default_rng(0).choice(len(xyz), num_points, replace=False)
        xyz, fdc = xyz[idx], fdc[idx]
    rgb = np.clip(np.nan_to_num((fdc * SH_C0 + 0.5) * 255), 0, 255).astype(np.uint8)
    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (p, c) in enumerate(zip(xyz, rgb), start=1):
            f.write(f"{i} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 1.0\n")
    return xyz


def write_points3d_txt(sparse_dir: Path, points_ply: Path):
    """id x y z r g b error (empty track). Returns the vertex array for the summary print."""
    ply = PlyData.read(str(points_ply))
    verts = ply["vertex"].data
    has_color = all(c in verts.dtype.names for c in ("red", "green", "blue"))

    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, v in enumerate(verts, start=1):
            if has_color:
                r, g, b = int(v["red"]), int(v["green"]), int(v["blue"])
            else:
                r, g, b = 128, 128, 128
            f.write(f"{i} {v['x']} {v['y']} {v['z']} {r} {g} {b} 1.0\n")
    return verts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transforms", required=True, type=Path)
    parser.add_argument("--points_ply", type=Path, default=None,
                         help="Triangulated keypoint point cloud (xyz .ply). "
                              "Exactly one of --points_ply / --seed_splat_ply is required.")
    parser.add_argument("--seed_splat_ply", type=Path, default=None,
                         help="Trained splat .ply to subsample into the init points instead of "
                              "--points_ply (per-frame sequences; prefer a mask-filtered splat).")
    parser.add_argument("--seed_num_points", type=int, default=5000,
                         help="Max points subsampled from --seed_splat_ply (default 5000; higher "
                              "values measurably slow training by over-densifying)")
    parser.add_argument("--seed_opacity_thresh", type=float, default=0.3,
                         help="Only seed from Gaussians with sigmoid(opacity) above this")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--image_subdir", default="images_flat",
                         help="Name recorded in images.txt as <image_subdir>/<camera_label>.ext")
    parser.add_argument("--images_dir", type=Path, default=None,
                         help="Where the source images actually live, for --masks_dir baking. "
                              "Defaults to out_dir/image_subdir.")
    parser.add_argument("--masks_dir", type=Path, default=None,
                         help="Cleaned per-camera masks (<camera_label>.png, e.g. clean_masks.py output). "
                              "When given, bakes RGBA images and points images.txt at --rgba_subdir instead.")
    parser.add_argument("--rgba_subdir", default="images_rgba",
                         help="Subdir under out_dir to write baked RGBA images (only used with --masks_dir)")
    args = parser.parse_args()

    if (args.points_ply is None) == (args.seed_splat_ply is None):
        parser.error("exactly one of --points_ply / --seed_splat_ply is required")

    with open(args.transforms) as f:
        tf = json.load(f)

    sparse_dir = args.out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(tf["frames"], key=lambda fr: fr["camera_label"])

    if args.masks_dir is not None:
        images_dir = args.images_dir or (args.out_dir / args.image_subdir)
        bake_masks(frames, args.masks_dir, images_dir, args.out_dir, args.rgba_subdir)

    write_cameras_txt(sparse_dir, frames)
    write_images_txt(sparse_dir, frames, args.image_subdir, args.masks_dir, args.rgba_subdir)
    if args.seed_splat_ply is not None:
        verts = write_points3d_from_splat(sparse_dir, args.seed_splat_ply,
                                          args.seed_num_points, args.seed_opacity_thresh)
    else:
        verts = write_points3d_txt(sparse_dir, args.points_ply)

    print(f"Wrote COLMAP sparse/0 to {sparse_dir}")
    print(f"  {len(frames)} cameras/images, {len(verts)} 3D points")
    if args.masks_dir is not None:
        print(f"  Masks baked into alpha -- Brush auto-detects this, no special training flag needed")


if __name__ == "__main__":
    main()
