"""
splat4d_io.py  –  Shared I/O utilities for 4D Gaussian Splat web-format
conversion scripts (xz_to_omg4.py and queen_pkl_to_queen.py).

This module provides:
  - Magic numbers and constants for the OMG4 and QUEEN binary formats
  - write_4dgs_header()  – write the 28-byte binary file header
  - pack_frame_aos()     – pack per-frame Gaussian attributes into AoS layout
  - read_ply_gaussians() – load Gaussian attributes from a plain PLY file

Version 1 (baked per-frame; .omg4 and .queen share the layout, differing
only in their magic number and intended source model):

    Header (28 bytes, all values little-endian):
        uint32  magic            – format identifier (OMG4_MAGIC or QUEEN_MAGIC)
        uint32  version = 1
        uint32  numSplats
        uint32  numFrames
        float32 fps
        float32 timeDurationMin
        float32 timeDurationMax

    Per-frame record (repeated numFrames times):
        float32          timestamp
        float32[N × 14]  per-splat data, AoS layout:
            x  y  z  rot_0(w)  rot_1(x)  rot_2(y)  rot_3(z)
            scale_0  scale_1  scale_2   (log-space)
            opacity                      (logit-space)
            f_dc_0  f_dc_1  f_dc_2      (raw SH DC coefficients)

Version 2 (.omg4 only – compact temporal splats, no per-frame data; the
viewer evaluates linear motion and Gaussian temporal opacity on the GPU):

    Header (32 bytes, all values little-endian):
        uint32  magic = OMG4_MAGIC
        uint32  version = 2
        uint32  numSplats (N)
        uint32  flags = 0            – reserved for future use
        float32 timeMin              – clip start time (seconds)
        float32 timeMax              – clip end time (seconds)
        float32 fps                  – advisory only (UI frame count)
        uint32  reserved = 0

    Data: 19 SoA float32[N] arrays, in this order:
        x  y  z                      – position at t = t_center
        rot_0(w) rot_1(x) rot_2(y) rot_3(z)
                                     – quaternion of the sliced 3D covariance
        scale_0  scale_1  scale_2    – log-space sqrt eigenvalues of sliced cov
        opacity                      – logit-space peak opacity
        f_dc_0  f_dc_1  f_dc_2       – raw SH DC coefficients
        vx  vy  vz                   – linear velocity (scene units / second)
        t_center                     – temporal centre (seconds)
        t_sigma                      – temporal std-dev (seconds)

    If flags bit 0 is set, 45 further float32[N] SoA arrays follow:
        f_rest_0 .. f_rest_44        – 3-band spherical harmonics, PLY
                                       channel-major order, baked at each
                                       splat's temporal centre

    Reconstruction at time t:
        position(t) = (x,y,z) + (vx,vy,vz) * (t - t_center)
        alpha(t)    = sigmoid(opacity) * exp(-0.5 * ((t - t_center)/t_sigma)^2)

    Streamable variant (flags bit 2 set — "tiled"):
        The header grows to 40 bytes:
            ... 32-byte header as above ...
            uint32  tileSize             – splats per tile (K)
            uint32  reserved2 = 0
        The body is chunked-SoA: splats are grouped into tiles of K
        (last tile short), and each tile stores ALL of its fields
        contiguously (field order as above, each field float32[K]).
        Because every tile is self-contained, a sequential download
        yields renderable splats continuously; files are written with
        splats sorted by visual importance so early tiles carry the
        most significant content (progressive densification).
"""

import os
import struct
import sys
from typing import BinaryIO

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Format identifiers
# ---------------------------------------------------------------------------

OMG4_MAGIC  = 0x34474D4F   # "OMG4" in little-endian ASCII bytes
QUEEN_MAGIC = 0x4E455551   # "QUEN" in little-endian ASCII bytes
FORMAT_VERSION = 1
OMG4_V2_VERSION = 2

# Version 2 header flag bits
OMG4_V2_FLAG_SH    = 1     # 45 SH arrays follow the base arrays
OMG4_V2_FLAG_COV2D = 2     # cov2d_scale packed into the reserved word
OMG4_V2_FLAG_TILED = 4     # streamable chunked-SoA body, 40-byte header

# Version 2 SoA field order (each is a float32[N] array)
OMG4_V2_FIELDS = [
    'x', 'y', 'z',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
    'scale_0', 'scale_1', 'scale_2',
    'opacity',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'vx', 'vy', 'vz',
    't_center',
    't_sigma',
]

FLOATS_PER_SPLAT = 14      # x y z w(rot) x(rot) y(rot) z(rot) s0 s1 s2 op dc0 dc1 dc2

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def write_4dgs_header(
    fp: BinaryIO,
    magic: int,
    num_splats: int,
    num_frames: int,
    fps: float,
    time_min: float,
    time_max: float,
) -> None:
    """Write the 28-byte header for an OMG4 or QUEEN binary file.

    Parameters
    ----------
    fp         : Writable binary file object.
    magic      : Format magic number (OMG4_MAGIC or QUEEN_MAGIC).
    num_splats : Total number of Gaussians per frame (N).
    num_frames : Total number of animation frames.
    fps        : Frames per second.
    time_min   : Animation start time (seconds).
    time_max   : Animation end time (seconds).
    """
    fp.write(struct.pack(
        '<IIIIfff',
        magic, FORMAT_VERSION,
        num_splats, num_frames,
        fps,
        time_min, time_max,
    ))


def write_omg4_v2_header(
    fp: BinaryIO,
    magic: int,
    num_splats: int,
    time_min: float,
    time_max: float,
    fps: float,
    flags: int = 0,
    cov2d_scale=None,
    tile_size=None,
) -> None:
    """Write the 32-byte header for a version-2 .omg4 file.

    Parameters
    ----------
    fp          : Writable binary file object.
    magic       : Format magic number (OMG4_MAGIC).
    num_splats  : Total number of Gaussians (N).
    time_min    : Clip start time (seconds).
    time_max    : Clip end time (seconds).
    fps         : Advisory frames per second (UI only).
    flags       : Bitfield (bit 0: SH arrays present; bit 1: cov2d_scale set;
                  bit 2: tiled streamable body).
    cov2d_scale : Optional (kx, ky) screen-space 2D-covariance scale the
                  viewer must apply (packed as 2 x float16 in the reserved
                  word). Compensates renderers whose training inflated splat
                  footprints in screen space (OMG4's FoV-sentinel bug).
    tile_size   : Optional splats-per-tile; when set the streamable flag is
                  raised and the header extends to 40 bytes.
    """
    reserved = 0
    if cov2d_scale is not None:
        kx, ky = cov2d_scale
        h = np.array([kx, ky], dtype=np.float16).view(np.uint16)
        reserved = int(h[0]) | (int(h[1]) << 16)
        flags |= OMG4_V2_FLAG_COV2D
    if tile_size is not None:
        flags |= OMG4_V2_FLAG_TILED
    fp.write(struct.pack(
        '<IIIIfffI',
        magic, OMG4_V2_VERSION,
        num_splats, flags,
        time_min, time_max, fps,
        reserved,
    ))
    if tile_size is not None:
        fp.write(struct.pack('<II', int(tile_size), 0))


def write_omg4_v2_body_tiled(fp: BinaryIO, arrays, tile_size: int) -> None:
    """Write v2 SoA arrays as a chunked-SoA (tiled) streamable body.

    Splats are grouped into tiles of `tile_size`; each tile stores every
    field contiguously (float32[K] per field, standard field order), so a
    sequential read yields fully-renderable splats tile by tile.
    """
    n = len(arrays[0])
    for start in range(0, n, tile_size):
        end = min(start + tile_size, n)
        for a in arrays:
            fp.write(np.ascontiguousarray(a[start:end], dtype=np.float32).tobytes())


# ---------------------------------------------------------------------------
# Per-frame AoS packing
# ---------------------------------------------------------------------------

def pack_frame_aos(
    pos: torch.Tensor,
    rot: torch.Tensor,
    sc: torch.Tensor,
    op: torch.Tensor,
    fdc: torch.Tensor,
) -> np.ndarray:
    """Pack per-frame Gaussian attributes into the AoS float32 layout.

    Parameters
    ----------
    pos : [N, 3] float32  – world-space positions (x, y, z)
    rot : [N, 4] float32  – unit quaternions (w, x, y, z); normalised internally
    sc  : [N, 3] float32  – log-space scales (scale_0, scale_1, scale_2)
    op  : [N]   float32   – logit-space opacity (1-D or squeeze'd)
    fdc : [N, 3] float32  – raw SH DC colour coefficients (f_dc_0..2)

    Returns
    -------
    numpy float32 array of shape [N, 14] ready to be written directly with
    .tobytes().
    """
    N = pos.shape[0]

    # Ensure unit quaternion
    rot_norm = torch.nn.functional.normalize(rot, dim=-1)   # [N, 4]

    # Flatten opacity to 1-D if needed
    op_1d = op.squeeze(-1) if op.ndim == 2 else op           # [N]

    frame = np.empty((N, FLOATS_PER_SPLAT), dtype=np.float32)
    frame[:, 0 ] = pos[:, 0].detach().numpy()         # x
    frame[:, 1 ] = pos[:, 1].detach().numpy()         # y
    frame[:, 2 ] = pos[:, 2].detach().numpy()         # z
    frame[:, 3 ] = rot_norm[:, 0].detach().numpy()    # rot_0 (w)
    frame[:, 4 ] = rot_norm[:, 1].detach().numpy()    # rot_1 (x)
    frame[:, 5 ] = rot_norm[:, 2].detach().numpy()    # rot_2 (y)
    frame[:, 6 ] = rot_norm[:, 3].detach().numpy()    # rot_3 (z)
    frame[:, 7 ] = sc[:, 0].detach().numpy()          # scale_0 (log)
    frame[:, 8 ] = sc[:, 1].detach().numpy()          # scale_1 (log)
    frame[:, 9 ] = sc[:, 2].detach().numpy()          # scale_2 (log)
    frame[:, 10] = op_1d.detach().numpy()             # opacity (logit)
    frame[:, 11] = fdc[:, 0].detach().numpy()          # f_dc_0
    frame[:, 12] = fdc[:, 1].detach().numpy()          # f_dc_1
    frame[:, 13] = fdc[:, 2].detach().numpy()          # f_dc_2

    return frame


# ---------------------------------------------------------------------------
# PLY Gaussian reader
# ---------------------------------------------------------------------------

def read_ply_gaussians(ply_path: str) -> dict:
    """Load Gaussian attributes from a standard 3DGS-style PLY file.

    Expected vertex properties (all float32 unless noted):
      x, y, z
      f_dc_0, f_dc_1, f_dc_2
      opacity
      scale_0, scale_1, scale_2
      rot_0, rot_1, rot_2, rot_3   (w, x, y, z order)

    Returns
    -------
    dict with keys 'xyz', 'f_dc', 'sc', 'rot', 'op' as float32 tensors.
    """
    try:
        from plyfile import PlyData
    except ImportError:
        sys.exit(
            "ERROR: 'plyfile' package not found.\n"
            "Install it with:  pip install plyfile"
        )

    plydata = PlyData.read(ply_path)
    el = plydata.elements[0]

    xyz = np.stack([el['x'], el['y'], el['z']], axis=1).astype(np.float32)

    f_dc = np.stack([el['f_dc_0'], el['f_dc_1'], el['f_dc_2']], axis=1).astype(np.float32)

    scale_names = sorted(
        [p.name for p in el.properties if p.name.startswith('scale_')],
        key=lambda n: int(n.split('_')[-1])
    )
    sc = np.stack([np.asarray(el[n]) for n in scale_names], axis=1).astype(np.float32)

    rot_names = sorted(
        [p.name for p in el.properties if p.name.startswith('rot_')],
        key=lambda n: int(n.split('_')[-1])
    )
    rot = np.stack([np.asarray(el[n]) for n in rot_names], axis=1).astype(np.float32)

    op = np.asarray(el['opacity']).astype(np.float32)  # logit-space

    return {
        'xyz':  torch.tensor(xyz),
        'f_dc': torch.tensor(f_dc),
        'sc':   torch.tensor(sc),
        'rot':  torch.tensor(rot),
        'op':   torch.tensor(op).unsqueeze(-1),
    }


# ---------------------------------------------------------------------------
# Convenience: report output file size
# ---------------------------------------------------------------------------

def report_output(out_path: str) -> None:
    """Print the output file path and size to stdout."""
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")
