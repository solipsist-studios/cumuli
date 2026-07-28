# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Per-camera RawTherapee color correction, used by extract_synced_frames.py's
--pp3_dir option. Matches a .pp3 profile to a camera by filename stem (e.g.
thumbs/0001.mp4.thumb.jpg.pp3 for 0001.mp4) and applies it via rawtherapee-cli.
Per-GoPro exposure/saturation differences are a major source of cross-view
color inconsistency in the trained splat -- correcting here brings the
automated chain to parity with manually processed stills.

Output is always PNG: rawtherapee-cli's -n flag forces PNG bytes regardless
of the requested output extension.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def resolve_rawtherapee_cmd(override=None):
    if override:
        return override.split()
    if shutil.which("rawtherapee-cli"):
        return ["rawtherapee-cli"]
    if shutil.which("flatpak"):
        return ["flatpak", "run", "--command=rawtherapee-cli", "com.rawtherapee.RawTherapee"]
    print("Error: rawtherapee-cli not found (nor flatpak). Install RawTherapee or pass --rawtherapee_cmd.")
    sys.exit(1)


def find_pp3(pp3_dir: Path, camera_stem: str):
    """Match a .pp3 profile whose filename contains the camera stem."""
    candidates = sorted(p for p in pp3_dir.rglob("*.pp3") if camera_stem in p.name)
    return candidates[0] if candidates else None


def apply_pp3(rt_cmd, image_path: Path, pp3_path: Path, out_path: Path):
    # -n png output, -Y overwrite, -c input (must be last)
    cmd = rt_cmd + ["-o", str(out_path), "-p", str(pp3_path), "-n", "-Y", "-c", str(image_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not out_path.exists():
        raise RuntimeError(f"RawTherapee failed on {image_path}:\n{result.stdout}\n{result.stderr}")
