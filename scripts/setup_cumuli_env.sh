#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
#
# Full provisioning for the single `cumuli` pipeline env. environment.yml
# is the package manifest. This script adds the installs a yml cannot
# express (see that file's header), the editable cumuli-core and
# cumuli-trainer packages from deps/, and verifies the result.
#
# Usage:
#   bash scripts/setup_cumuli_env.sh          # GPU machine (default)
#   bash scripts/setup_cumuli_env.sh --cpu    # CPU-only (CI): cpu torch
#                                             # wheels, no gsplat JIT warm-up
#
# Overridable:
#   CONDA                 conda executable (default: ~/miniconda3/condabin/conda)
#   ENV_NAME              env name (default: cumuli)
#   TORCH_CUDA_ARCH_LIST  compute capability for gsplat's JIT build
#                         (default: probed from nvidia-smi)
#   MAX_JOBS              parallel build jobs (default: nproc)
set -euo pipefail

# CONDA may arrive as the conda EXECUTABLE or (per setup-miniconda's
# convention, which exports the install base) as a DIRECTORY. Accept
# both, then fall back to PATH and the default install location.
CONDA=${CONDA:-"$HOME/miniconda3/condabin/conda"}
if [ -d "$CONDA" ]; then CONDA="$CONDA/condabin/conda"; fi
if [ ! -x "$CONDA" ]; then CONDA="$(command -v conda || echo "$HOME/miniconda3/condabin/conda")"; fi
ENV_NAME=${ENV_NAME:-cumuli}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPU_ONLY=0
if [ "${1:-}" = "--cpu" ]; then CPU_ONLY=1; fi

HLOC_COMMIT=c13273bd0ecc2917a35910fd843712a1c6243193
EASYVOLCAP_COMMIT=4cb3c000a31b8764834c79792b355f110d947e75

if [ "$CPU_ONLY" = 1 ]; then
    echo "=== create $ENV_NAME (cpu): torch swapped BEFORE install ==="
    # The yml's cu130 torch is multi-GB and must never be downloaded on a
    # CPU runner at all. An install-then-replace would stack both wheel
    # sets on an already-tight disk budget (the failure mode the old
    # per-env CI creation was engineered around). So cpu torch is
    # installed FIRST, before the yml's requirements: no transitive
    # dependency (diffusers, lpips, kornia, ...) can then drag the
    # CUDA-bundled PyPI torch in during resolution, because pip only
    # installs torch if it is not already present.
    export PIP_NO_CACHE_DIR=1
    "$CONDA" create -y -n "$ENV_NAME" python=3.12 pip
    PY="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
    "$PY" -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision
    REQS="$(mktemp --suffix=.txt)"
    "$PY" - "$REPO_ROOT/environment.yml" > "$REQS" <<'PYREQ'
import sys
lines = []
in_pip = False
for raw in open(sys.argv[1]):
    line = raw.rstrip("\n")
    if line.strip() == "- pip:":
        in_pip = True
        continue
    if in_pip:
        if line.strip().startswith("#") or not line.strip():
            continue
        if not line.startswith("      - "):
            break
        item = line.split("- ", 1)[1].split("  #")[0].strip()
        if item.startswith("--") or item.startswith("torch==") or item == "torchvision":
            continue
        lines.append(item)
print("\n".join(lines))
PYREQ
    "$PY" -m pip install --no-cache-dir -r "$REQS"
    rm -f "$REQS"
    "$CONDA" clean -afy
else
    echo "=== create $ENV_NAME from environment.yml ==="
    "$CONDA" env create -n "$ENV_NAME" -f "$REPO_ROOT/environment.yml"
    PY="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
fi

echo "=== hloc: submodule-recursive clone @${HLOC_COMMIT:0:7}, editable --no-deps ==="
# hloc==1.5 is not on PyPI, and `pip install git+URL` never fetches the
# SuperGluePretrainedNetwork submodule SuperPoint imports.
HLOC_SRC="$HOME/.cache/cumuli-env/hloc-src"
if [ ! -e "$HLOC_SRC/.git" ]; then
    rm -rf "$HLOC_SRC"
    git clone --recurse-submodules https://github.com/cvg/Hierarchical-Localization.git "$HLOC_SRC"
fi
(cd "$HLOC_SRC" && git fetch -q origin "$HLOC_COMMIT" && git checkout -q "$HLOC_COMMIT" \
    && git submodule update --init --recursive)
"$PY" -m pip install --no-deps -e "$HLOC_SRC"

echo "=== easyvolcap: --no-deps @${EASYVOLCAP_COMMIT:0:7} ==="
# --no-deps: its declared dependency set would override the yml's pins.
"$PY" -m pip install --no-deps \
    "git+https://github.com/zju3dv/EasyVolcap.git@$EASYVOLCAP_COMMIT"

echo "=== cumuli-core: editable from deps/cumuli-core, --no-deps ==="
# The shared library (interchange PLY, cameras, spacetime slice, renderer,
# metrics). --no-deps: torch, gsplat, lpips and torchmetrics come from the
# yml pins above, and a resolver pass must not move them.
CORE="$REPO_ROOT/deps/cumuli-core"
[ -f "$CORE/pyproject.toml" ] || {
    echo "FAIL: deps/cumuli-core not initialised (git submodule update --init deps/cumuli-core)"; exit 1; }
"$PY" -m pip install --no-deps -e "$CORE"

echo "=== cumuli-trainer: editable from deps/cumuli-trainer, --no-deps ==="
TRAINER="$REPO_ROOT/deps/cumuli-trainer"
[ -f "$TRAINER/pyproject.toml" ] || {
    echo "FAIL: deps/cumuli-trainer not initialised (git submodule update --init deps/cumuli-trainer)"; exit 1; }
"$PY" -m pip install --no-deps -e "$TRAINER"

if [ "$CPU_ONLY" = 0 ]; then
    echo "=== gsplat CUDA JIT build (once, against this env's torch) ==="
    if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 || true)"
        TORCH_CUDA_ARCH_LIST="${CAP:-12.0}"
    fi
    export TORCH_CUDA_ARCH_LIST
    export MAX_JOBS=${MAX_JOBS:-$(nproc)}
    echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS"
    "$PY" - <<'WARMUP'
import torch
from gsplat import rasterization
d = "cuda"
means = torch.zeros(1, 3, device=d); means[0, 2] = 2.0
img, _, _ = rasterization(means, torch.tensor([[1.0, 0, 0, 0]], device=d),
                          torch.full((1, 3), 0.1, device=d), torch.ones(1, device=d),
                          torch.ones(1, 3, device=d), torch.eye(4, device=d)[None],
                          torch.tensor([[[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]]], device=d),
                          64, 64)
print("gsplat ok:", tuple(img.shape))
WARMUP
fi

echo "=== verify ==="
if [ "$CPU_ONLY" = 1 ]; then export CUMULI_VERIFY_CPU=1; fi
"$PY" - <<'VERIFY'
import importlib, os, sys
cpu_only = os.environ.get("CUMULI_VERIFY_CPU") == "1"
core = ["torch", "torchvision", "numpy", "scipy", "PIL", "cv2", "pycolmap",
        "hloc", "plyfile", "fire", "open3d", "transformers", "gsplat",
        "lpips", "torchmetrics", "omegaconf",
        "easyvolcap.utils.console_utils", "easyvolcap.utils.parallel_utils",
        "sapiens.pose", "diffusers", "hydra", "kornia",
        "cumuli_core.ply", "cumuli_core.spacetime", "cumuli_core.render",
        "cumuli_core.metrics", "cumuli_trainer"]
gpu = []
failed = []
for mod in core + ([] if cpu_only else gpu):
    try:
        importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001 -- report every failure at once
        failed.append(f"{mod}: {type(e).__name__}: {e}")
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
if failed:
    print("FAILED imports:\n  " + "\n  ".join(failed))
    sys.exit(1)
print("all imports ok")
VERIFY
echo "=== $ENV_NAME provisioned ==="
