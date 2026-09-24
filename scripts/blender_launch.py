#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""
blender_launch.py - locate Blender and re-launch a script inside it.

The Blender-facing scripts in this directory are dual mode. Run normally,
they find Blender and re-execute themselves inside it; run by Blender, they
do the work. That keeps one file per job instead of a script plus a wrapper,
and it means the caller never has to remember the `-b ... --python ... --`
incantation.

    python3 scripts/prepare_blender_scene.py --input x.fbx --out y.blend

Blender's bundled Python is a different interpreter from the `cumuli` env:
it has numpy but not PIL, OpenCV, or torch. Modules imported from inside
Blender must therefore stay on numpy alone, which is why camera_rig_spec.py
and blender_camera_intrinsics.py carry no other dependency.

Set CUMULI_BLENDER to choose a specific binary.
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:  # pragma: no cover - the branch taken depends on the interpreter
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

CANDIDATE_PATHS = (
    "/usr/local/bin/blender",
    "/usr/bin/blender",
    "/snap/bin/blender",
)
CANDIDATE_GLOBS = (
    "/opt/blender*/blender",
    str(Path.home() / "blender*/blender"),
)

MIN_VERSION = (4, 2)


class BlenderNotFound(RuntimeError):
    pass


def find_blender(explicit=None):
    """Path to a Blender binary, or raise with the places that were tried."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        raise BlenderNotFound(f"--blender {explicit} is not an executable file")

    env = os.environ.get("CUMULI_BLENDER")
    if env:
        p = Path(env).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        raise BlenderNotFound(
            f"CUMULI_BLENDER={env} is not an executable file")

    found = shutil.which("blender")
    if found:
        return found
    for cand in CANDIDATE_PATHS:
        if Path(cand).is_file():
            return cand
    for pattern in CANDIDATE_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    raise BlenderNotFound(
        "no Blender binary found. Set CUMULI_BLENDER, put `blender` on PATH, "
        f"or install it under one of {CANDIDATE_PATHS + CANDIDATE_GLOBS}")


def script_argv(argv=None):
    """The arguments after `--`, which is how Blender passes them through."""
    argv = list(sys.argv if argv is None else argv)
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return []


def build_command(blender, script, args, blend=None, factory_startup=True):
    cmd = [str(blender), "-b"]
    if blend:
        cmd.append(str(blend))
    if factory_startup and not blend:
        # A user's startup file can carry add-ons and render settings that
        # change output. Opening a .blend already replaces the scene, so the
        # flag only matters for the empty-scene case.
        cmd.append("--factory-startup")
    cmd += ["--python", str(script), "--python-exit-code", "1", "--"]
    cmd += [str(a) for a in args]
    return cmd


def relaunch_in_blender(script, args, blend=None, blender=None, timeout=None,
                        env=None, echo=True):
    """Run `script` inside Blender and return the completed process.

    Blender exits 0 even when a Python script raises, which silently turns a
    crash into an empty output directory. `--python-exit-code 1` makes the
    failure visible, and this checks it."""
    exe = find_blender(blender)
    cmd = build_command(exe, script, args, blend=blend)
    if echo:
        print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise SystemExit(
            f"Blender exited {proc.returncode} running {Path(script).name}. "
            "The traceback is above, in Blender's own output.")
    return proc


def blender_version():
    """(major, minor, patch) of the running Blender, or None outside it."""
    if not IN_BLENDER:
        return None
    import bpy
    return tuple(bpy.app.version)
