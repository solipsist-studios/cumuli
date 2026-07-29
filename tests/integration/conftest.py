"""
tests/integration/conftest.py

Real, unmocked pipeline tests need a real GPU + conda envs + brush_app +
Sapiens checkpoints -- none of which exist on a plain GitHub-hosted CI
runner. Rather than skip the whole suite via CI-side conditionals, every
test here goes through `pipeline_prereqs` (autouse), which probes for what
it actually needs and skips with a specific reason when something's
missing. Same test file, same behavior, on any machine: it does the real
thing where the environment supports it, and skips cleanly everywhere else.

VCP_ALLOW_CPU_RENDERING=1 opts into training without a real GPU at all,
via Mesa's llvmpipe software rasterizer -- see pipeline_run's env
construction in test_pipeline_end_to_end.py for the full recipe and the
hard-won findings behind it (2026-07-28): WGPU_BACKEND=gl alone does NOT
avoid the GPU (an earlier claim here that it had been verified to was
wrong -- that run had silently used the NVIDIA OpenGL driver), and the
v0.3.0 release brush binary cannot run on llvmpipe at all (burn-fusion
spinlock deadlock, a nondeterministic race, plus two hard startup
panics). CPU mode therefore requires a brush binary built from source
newer than 2026-07 (VCP_BRUSH_APP) -- verified end-to-end on genuine
llvmpipe with that build. Off by default -- opt-in only, existing
GPU-required behavior is unchanged unless this is explicitly set.
"""

import ctypes.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import run_unified_pipeline as unified

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "heidi_11cam"

REQUIRED_CONDA_ENVS = ("hloc", "diffuman4d", "sapiens2", "queen")


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        return "ffmpeg not found on PATH"
    return None


def _check_gpu(allow_cpu_rendering):
    if allow_cpu_rendering:
        # Real GPU not required in this mode -- brush_app trains via Mesa's
        # software OpenGL rasterizer instead (see module docstring). This is
        # only a lightweight sanity check that OpenGL is present at all
        # (same "check it exists, not that it works" philosophy as
        # _check_brush_app below) -- the real verification is the pipeline
        # run itself.
        if ctypes.util.find_library("GL") is None:
            return ("VCP_ALLOW_CPU_RENDERING=1 but no OpenGL library found on this machine "
                    "(needed for Mesa's software rasterizer) -- install mesa-utils/libgl1-mesa-dri")
        return None
    if shutil.which("nvidia-smi") is None:
        return ("nvidia-smi not found on PATH -- no NVIDIA GPU tooling present "
                "(or set VCP_ALLOW_CPU_RENDERING=1 to train via software rendering instead -- "
                "see docs/integration-tests.md)")
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"nvidia-smi failed to run: {e}"
    if result.returncode != 0 or "GPU" not in result.stdout:
        return f"nvidia-smi found no GPU (exit {result.returncode}): {result.stderr.strip()}"
    return None


def _check_conda_envs():
    try:
        conda_bin = unified.resolve_conda()
    except unified.StageError as e:
        return str(e)
    try:
        result = subprocess.run([conda_bin, "env", "list", "--json"],
                                 capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"'{conda_bin} env list' failed to run: {e}"
    if result.returncode != 0:
        return f"'{conda_bin} env list' exited {result.returncode}: {result.stderr.strip()}"
    try:
        env_paths = json.loads(result.stdout)["envs"]
    except (json.JSONDecodeError, KeyError) as e:
        return f"could not parse '{conda_bin} env list --json' output: {e}"
    env_names = {Path(p).name for p in env_paths}
    missing = [name for name in REQUIRED_CONDA_ENVS if name not in env_names]
    if missing:
        return f"missing conda env(s): {missing} (found: {sorted(env_names)})"
    return None


def _check_brush_app(brush_app):
    if not Path(brush_app).is_file():
        return f"brush_app not found at {brush_app} (set VCP_BRUSH_APP to point at a different binary)"
    return None


def _check_sapiens_checkpoints(sapiens_checkpoint_root):
    if sapiens_checkpoint_root is None:
        return "SAPIENS_CHECKPOINT_ROOT is not set and no default checkpoint dir was found"
    root = Path(sapiens_checkpoint_root)
    if not (root / "detector").is_dir() or not (root / "pose").is_dir():
        return f"{root} doesn't look like a Sapiens checkpoint root (expected detector/ and pose/ subdirs)"
    return None


def _check_fixture():
    movies = FIXTURE_DIR / "movies"
    calib = FIXTURE_DIR / "calibration_pkls"
    if not movies.is_dir() or not any(movies.glob("*.mp4")):
        return f"fixture videos not found under {movies}"
    if not calib.is_dir() or not any(calib.glob("*.pkl")):
        return f"fixture calibration not found under {calib}"
    return None


@pytest.fixture(scope="session")
def allow_cpu_rendering():
    """VCP_ALLOW_CPU_RENDERING=1 opts into training without a real GPU --
    see module docstring."""
    return os.environ.get("VCP_ALLOW_CPU_RENDERING") == "1"


@pytest.fixture(scope="session")
def brush_app():
    """brush_app binary path: VCP_BRUSH_APP if set (as docs/integration-tests.md
    documents), else the orchestrator's own default -- resolved once here so
    the prereq check and the real pipeline command can't disagree about
    which binary they mean."""
    override = os.environ.get("VCP_BRUSH_APP")
    if override:
        return override
    return str(unified.build_parser().get_default("brush_app"))


@pytest.fixture(scope="session")
def sapiens_checkpoint_root():
    """Default checkpoint root, same fallback path used throughout this
    project's real runs (~/sapiens/2) -- override with
    VCP_SAPIENS_CHECKPOINT_ROOT for a different location."""
    override = os.environ.get("VCP_SAPIENS_CHECKPOINT_ROOT")
    if override:
        return override
    env_root = os.environ.get("SAPIENS_CHECKPOINT_ROOT")
    if env_root:
        return env_root
    default = Path.home() / "sapiens" / "2"
    return str(default) if default.is_dir() else None


@pytest.fixture(scope="session", autouse=True)
def pipeline_prereqs(sapiens_checkpoint_root, brush_app, allow_cpu_rendering):
    """Skips the whole tests/integration session (not a failure) if this
    machine can't actually run the real pipeline. See module docstring."""
    reasons = list(filter(None, [
        _check_ffmpeg(),
        _check_gpu(allow_cpu_rendering),
        _check_conda_envs(),
        _check_brush_app(brush_app),
        _check_sapiens_checkpoints(sapiens_checkpoint_root),
        _check_fixture(),
    ]))
    if reasons:
        message = "Real pipeline prerequisites not met on this machine:\n  - " + "\n  - ".join(reasons)
        # On a machine that is SUPPOSED to run this suite (the CI runner),
        # skipping would surface as a green check with zero tests run -- a
        # runner whose setup rots (broken conda env, dead GPU driver)
        # would silently disable the whole safety net. The workflow sets
        # this env var so a misconfigured runner fails loudly instead;
        # everywhere else keeps the friendly skip.
        if os.environ.get("VCP_REQUIRE_PIPELINE_PREREQS") == "1":
            pytest.fail(message)
        pytest.skip(message)


@pytest.fixture(scope="session")
def fixture_dir():
    return FIXTURE_DIR
