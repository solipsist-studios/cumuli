#!/usr/bin/env python3
"""
queen_pkl_to_queen.py – Convert a QUEEN compressed 4DGS model (folder of
per-frame .pkl files) to the web-friendly .queen binary format consumed by
the supersplat-viewer.

QUEEN (QUantized Efficient ENcoding, NeurIPS 2024) stores per-frame Gaussians
in a folder structure such as:

    <input_dir>/
      Frame0001/
        compressed/
          0001.pkl          ← entropy-coded latents + decoder state
      Frame0002/
        compressed/
          0002.pkl          ← residuals relative to previous frame
      ...

Frame 1 may alternatively be stored as a plain PLY file:
    <input_dir>/Frame0001/point_cloud.ply

The .queen format stores pre-baked per-frame Gaussian attributes so that
the viewer can play back the animation directly without any GPU-side
inference.  See splat4d_io.py for the on-disk layout.

Requirements:
    torch  numpy  torchac  plyfile
    (torchac and plyfile are soft dependencies; clear errors are printed if
    either is missing and only needed for entropy-coded PKL or PLY frames.)

Usage:
    python queen_pkl_to_queen.py \\
        --input  path/to/queen_frames_directory \\
        --output path/to/scene.queen \\
        --fps    30.0
"""

import argparse
import os
import pickle
import re
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Shared I/O utilities (binary header, AoS packing, PLY reader)
sys.path.insert(0, str(Path(__file__).parent))
from splat4d_io import QUEEN_MAGIC, pack_frame_aos, write_4dgs_header, report_output, read_ply_gaussians


# ---------------------------------------------------------------------------
# CompressedLatents – self-contained reimplementation of QUEEN's entropy codec
# ---------------------------------------------------------------------------

class CompressedLatents:
    """
    Reimplementation of utils.compress_utils.CompressedLatents from the QUEEN
    repository.  Only the decompression path is needed here.

    When pickled by QUEEN, instances carry:
      num_latents (int), latent_dim (int), byte_stream (bytes),
      cdf (numpy float32 array), mapping (dict), tail_locs (dict).
    """

    def uncompress(self, scale: float = 1.0) -> torch.Tensor:
        """Decode entropy-coded bytes back to a float tensor [num_latents, latent_dim]."""
        try:
            import torchac
        except ImportError:
            sys.exit(
                "ERROR: 'torchac' package not found.\n"
                "Install it with:  pip install torchac\n"
                "(see https://github.com/fab-jul/torchac)"
            )

        import numpy as np
        cdf = torch.tensor(self.cdf).unsqueeze(0).repeat(
            self.num_latents * self.latent_dim, 1
        )
        weight = torchac.decode_float_cdf(cdf, self.byte_stream)
        weight = weight.to(torch.float32)

        inverse_mapping = {v: k for k, v in self.mapping.items()}
        weight.apply_(inverse_mapping.get)

        for val, locs in self.tail_locs.items():
            weight[locs] = val

        weight = weight / scale
        return weight.view(self.num_latents, self.latent_dim)


# ---------------------------------------------------------------------------
# Custom unpickler – maps QUEEN's internal class paths to local implementations
# ---------------------------------------------------------------------------

class _QueenUnpickler(pickle.Unpickler):
    """Unpickler that substitutes QUEEN's internal classes with local ones."""

    _CLASS_MAP = {
        ('utils.compress_utils', 'CompressedLatents'): CompressedLatents,
    }

    def find_class(self, module: str, name: str):
        key = (module, name)
        if key in self._CLASS_MAP:
            return self._CLASS_MAP[key]
        return super().find_class(module, name)


def _load_pkl(path: str) -> dict:
    """Load a QUEEN .pkl file, mapping internal classes to local ones."""
    with open(path, 'rb') as f:
        return _QueenUnpickler(f).load()


# ---------------------------------------------------------------------------
# Decoder forward passes (CPU reimplementation of QUEEN's LatentDecoder)
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    'relu':    torch.relu,
    'tanh':    torch.tanh,
    'sigmoid': torch.sigmoid,
    'none':    lambda x: x,
}


def _apply_decoder_layer(
    x: torch.Tensor,
    state: dict,
    prefix: str,
    ldecode_matrix: str,
) -> torch.Tensor:
    """Single DecoderLayer forward: x @ (dft *) scale + shift."""
    scale = state[f'{prefix}.scale']         # [latent_dim, feature_dim]
    shift = state.get(f'{prefix}.shift')     # [1, feature_dim] or None

    if 'dft' in ldecode_matrix:
        dft = state[f'{prefix}.dft']         # [latent_dim, dft_dim]
        out = (x @ dft) * scale
    else:
        out = x @ scale

    if shift is not None:
        out = out + shift
    return out


def _forward_decoder(
    latent: torch.Tensor,
    state: dict,
    args: dict,
) -> torch.Tensor:
    """
    Forward pass for LatentDecoder (and the linear part of LatentDecoderRes).

    Layout of nn.Sequential layers:
      - 0 hidden layers  (num_layers_dec=0):  layers.0
      - n hidden layers  (num_layers_dec=n):  layers.0, act, layers.2, act, …, layers.2n
    """
    num_hidden = int(args.get('num_layers_dec', 0))
    act_name   = str(args.get('activation', 'none'))
    ldm        = str(args.get('ldecode_matrix', 'learnable'))
    act_fn     = _ACTIVATIONS.get(act_name, _ACTIVATIONS['none'])

    div = state['div']   # [latent_dim]
    x   = latent / div

    for i in range(num_hidden):
        x = _apply_decoder_layer(x, state, f'layers.{2 * i}', ldm)
        x = act_fn(x)

    x = _apply_decoder_layer(x, state, f'layers.{2 * num_hidden}', ldm)
    return x


def _decode_attribute(
    latent_val,
    decoder_type: str,
    state: dict,
    args: dict,
    prev_decoded: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Decode a single Gaussian attribute from a PKL latent value.

    Parameters
    ----------
    latent_val    : CompressedLatents | torch.Tensor | dict (gated xyz)
    decoder_type  : 'DecoderIdentity' | 'LatentDecoder' | 'LatentDecoderRes'
    state         : decoder state dict (may include 'decoded_att' for Res)
    args          : decoder constructor args dict
    prev_decoded  : previous frame's decoded output (for gated residual xyz)
    """
    if decoder_type == 'DecoderIdentity':
        if isinstance(latent_val, dict):
            # Gated residual xyz: reconstruct from sparse delta on prev frame
            mapping   = latent_val['mapping']
            ung_idx   = latent_val['ungated_indices']
            residuals = latent_val['ungated_residuals_compressed'].uncompress(scale=10000.0)

            if prev_decoded is None:
                raise ValueError(
                    "Gated xyz residual found but no previous frame is available."
                )
            recon = prev_decoded[mapping].clone()
            recon[ung_idx] += residuals
            return recon
        else:
            return latent_val.float()

    elif decoder_type == 'LatentDecoder':
        latent = latent_val.uncompress()
        return _forward_decoder(latent, state, args)

    elif decoder_type == 'LatentDecoderRes':
        latent = latent_val.uncompress()
        decoded_att = state.get('decoded_att', None)
        layer_state = {k: v for k, v in state.items() if k != 'decoded_att'}
        out = _forward_decoder(latent, layer_state, args)
        if decoded_att is not None:
            out = out + decoded_att
        return out

    else:
        raise ValueError(f"Unknown decoder type: '{decoder_type}'")


# ---------------------------------------------------------------------------
# Decode all wanted attributes from a PKL data dict
# ---------------------------------------------------------------------------

_WANTED_ATTRS = ('xyz', 'f_dc', 'sc', 'rot', 'op')


def _decode_all_attributes(
    data: dict,
    prev_decoded: dict | None,
) -> dict:
    """
    Decode all wanted Gaussian attributes from a QUEEN PKL data dict.

    Parameters
    ----------
    data          : unpickled PKL dict
    prev_decoded  : decoded attributes from the previous frame (for gated xyz)

    Returns
    -------
    dict mapping attribute name → decoded float32 tensor
    """
    latents   = data['latents']
    states    = data.get('decoder_state_dict', {})
    args      = data.get('decoder_args', {})
    dec_types = data['latent_decoders_dict']

    result = {}
    for attr in _WANTED_ATTRS:
        if attr not in latents:
            continue
        dtype = dec_types.get(attr, 'DecoderIdentity')
        state = states.get(attr, {})
        arg   = args.get(attr, {})
        prev  = prev_decoded.get(attr) if prev_decoded else None
        result[attr] = _decode_attribute(latents[attr], dtype, state, arg, prev)

    return result


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def _find_frame_entries(input_dir: str) -> list:
    """
    Scan *input_dir* for QUEEN frame sub-directories.

    Accepted naming patterns (case-insensitive):
      Frame0001, frame0002, …  (optional 'frame' prefix + zero-padded integer)
      0001, 0002, …            (plain zero-padded integers)

    Returns a sorted list of (frame_number, absolute_dir_path) pairs.
    """
    pattern = re.compile(r'^(?:frame)?(\d+)$', re.IGNORECASE)
    entries = []

    for name in os.listdir(input_dir):
        full = os.path.join(input_dir, name)
        if not os.path.isdir(full):
            continue
        m = pattern.match(name)
        if m:
            entries.append((int(m.group(1)), full))

    if not entries:
        raise FileNotFoundError(
            f"No frame directories found in '{input_dir}'.\n"
            "Expected folders named 'Frame0001', 'Frame0002', … "
            "(or plain '0001', '0002', …)."
        )

    entries.sort(key=lambda t: t[0])
    return entries


def _find_pkl_or_ply(frame_dir: str) -> tuple:
    """
    Return (path, kind) where kind is 'pkl' or 'ply'.

    Search order:
      1. <frame_dir>/compressed/*.pkl   (any .pkl inside compressed/)
      2. <frame_dir>/point_cloud.ply    (PLY fallback, typically frame 1)
    """
    compressed_dir = os.path.join(frame_dir, 'compressed')
    if os.path.isdir(compressed_dir):
        pkls = sorted(
            [f for f in os.listdir(compressed_dir) if f.endswith('.pkl')]
        )
        if pkls:
            return os.path.join(compressed_dir, pkls[0]), 'pkl'

    ply_path = os.path.join(frame_dir, 'point_cloud.ply')
    if os.path.isfile(ply_path):
        return ply_path, 'ply'

    raise FileNotFoundError(
        f"No PKL (in 'compressed/') or PLY ('point_cloud.ply') found in "
        f"'{frame_dir}'."
    )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_dir: str, out_path: str, fps: float) -> None:

    # ── Discover frames ───────────────────────────────────────────────────────
    frame_entries = _find_frame_entries(input_dir)
    num_frames = len(frame_entries)
    print(f"Found {num_frames} frame(s) in '{input_dir}'")

    # ── First pass: load frame 1 to determine N ───────────────────────────────
    _, first_dir = frame_entries[0]
    first_path, first_kind = _find_pkl_or_ply(first_dir)
    print(f"  Frame 1: {first_path}  [{first_kind.upper()}]")

    if first_kind == 'ply':
        first_attrs = read_ply_gaussians(first_path)
    else:
        first_attrs = _decode_all_attributes(_load_pkl(first_path), prev_decoded=None)

    N = first_attrs['xyz'].shape[0]

    time_min = 0.0
    time_max = (num_frames - 1) / fps if num_frames > 1 else 0.0

    print(f"  {N:,} Gaussians | {num_frames} frames @ {fps} fps")

    # ── Write .queen file ─────────────────────────────────────────────────────
    with open(out_path, 'wb') as fp:
        write_4dgs_header(fp, QUEEN_MAGIC, N, num_frames, fps, time_min, time_max)

        prev_decoded = None

        for fi, (frame_num, frame_dir) in enumerate(frame_entries):
            timestamp = time_min + (time_max - time_min) * fi / max(num_frames - 1, 1)
            print(f"  Frame {fi + 1}/{num_frames}  (#{frame_num})  t={timestamp:.4f}", end='\r')

            if fi == 0:
                attrs = first_attrs
            else:
                path, kind = _find_pkl_or_ply(frame_dir)
                if kind == 'ply':
                    attrs = read_ply_gaussians(path)
                else:
                    attrs = _decode_all_attributes(_load_pkl(path), prev_decoded=prev_decoded)

            frame_data = pack_frame_aos(
                pos=attrs['xyz'],
                rot=attrs['rot'],
                sc=attrs['sc'],
                op=attrs['op'],
                fdc=attrs['f_dc'],
            )

            fp.write(struct.pack('<f', timestamp))
            fp.write(frame_data.tobytes())

            prev_decoded = attrs

    print()
    report_output(out_path)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Convert a QUEEN per-frame PKL directory to the .queen web format'
        )
    )
    parser.add_argument(
        '--input', required=True,
        help='Path to directory containing Frame0001/, Frame0002/, … sub-folders',
    )
    parser.add_argument(
        '--output', required=True,
        help='Destination .queen file',
    )
    parser.add_argument(
        '--fps', type=float, default=30.0,
        help='Frames per second (default: 30)',
    )
    args = parser.parse_args()

    convert(
        input_dir=args.input,
        out_path=args.output,
        fps=args.fps,
    )
