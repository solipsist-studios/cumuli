"""
tests/integration/conftest.py

Real, unmocked pipeline tests need real conda envs, Sapiens checkpoints,
and (for the training stage) a CUDA GPU plus the deps/OMG4 trainer
submodule -- none of which exist on a plain GitHub-hosted CI runner.
Rather than skip the whole suite via CI-side conditionals, every test
here goes through `pipeline_prereqs` (autouse), which probes for what it
actually needs and skips with a specific reason when something is
missing. Same test file, same behavior, on any machine: it does the real
thing where the environment supports it, and skips cleanly everywhere
else.

VCP_CPU_PIPELINE=1 selects the CPU-capable pipeline subset: the run stops
after the dataset4d stage (--stop_after_stage dataset4d), so no GPU, no
trainer env, and no trainer submodule are required. GPU mode (the toggle
unset) runs the full pipeline through train4d, which additionally needs
the `cumuli` conda env and an initialised deps/OMG4 submodule
(train_scratch.py is the trainer entry point).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import run_unified_pipeline as unified

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "take01_11cam"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The pipeline runs in one merged env (see environment.yml and
# scripts/setup_cumuli_env.sh); the orchestrator's CONDA_ENV constant
# names it, and this suite checks that configuration.
REQUIRED_CONDA_ENVS = ("cumuli",)
GPU_CONDA_ENVS = ()


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        return "ffmpeg not found on PATH"
    return None


def _check_gpu(cpu_pipeline):
    if cpu_pipeline:
        # The CPU subset stops before the training stage, so no GPU (and
        # no rasterizer of any kind) is needed at all.
        return None
    if shutil.which("nvidia-smi") is None:
        return ("nvidia-smi not found on PATH -- no NVIDIA GPU tooling present "
                "(or set VCP_CPU_PIPELINE=1 to run the CPU-capable subset through "
                "the dataset4d stage instead -- see docs/integration-tests.md)")
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"nvidia-smi failed to run: {e}"
    if result.returncode != 0 or "GPU" not in result.stdout:
        return f"nvidia-smi found no GPU (exit {result.returncode}): {result.stderr.strip()}"
    return None


def _check_conda_envs(cpu_pipeline):
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
    required = REQUIRED_CONDA_ENVS if cpu_pipeline else REQUIRED_CONDA_ENVS + GPU_CONDA_ENVS
    missing = [name for name in required if name not in env_names]
    if missing:
        return f"missing conda env(s): {missing} (found: {sorted(env_names)})"
    return None


def _check_trainer_repo(cpu_pipeline):
    if cpu_pipeline:
        # The CPU subset never reaches train4d, so the trainer submodule
        # is not needed.
        return None
    train_script = REPO_ROOT / "deps" / "OMG4" / "train_scratch.py"
    if not train_script.is_file():
        return (f"trainer entry point not found at {train_script} -- initialise the "
                f"submodule (git submodule update --init deps/OMG4)")
    # The trainer's CUDA extensions are built into the env by
    # scripts/setup_cumuli_env.sh, not installable from the yml -- a fresh
    # `conda env create` alone leaves them missing and training fails deep
    # inside the first iteration instead of here.
    probe = subprocess.run(
        ["conda", "run", "-n", "cumuli", "python", "-c",
         "import diff_gaussian_rasterization, simple_knn"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        return ("cumuli env lacks the trainer CUDA extensions -- provision with "
                "scripts/setup_cumuli_env.sh (not conda env create alone)")
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
def cpu_pipeline():
    """VCP_CPU_PIPELINE=1 selects the CPU-capable pipeline subset (stop
    after dataset4d) -- see module docstring."""
    return os.environ.get("VCP_CPU_PIPELINE") == "1"


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
def pipeline_prereqs(sapiens_checkpoint_root, cpu_pipeline):
    """Skips the whole tests/integration session (not a failure) if this
    machine can't actually run the real pipeline. See module docstring."""
    reasons = list(filter(None, [
        _check_ffmpeg(),
        _check_gpu(cpu_pipeline),
        _check_conda_envs(cpu_pipeline),
        _check_trainer_repo(cpu_pipeline),
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
