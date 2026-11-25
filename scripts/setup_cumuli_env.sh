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
#   MAX_JOBS              parallel build jobs (default: nproc, at most 8: each
#                         nvcc job is memory hungry and WSL2 runs out first)
#   CUDA_HOME             CUDA toolkit for gsplat's JIT build (default: nvcc
#                         on PATH); needs nvcc >= 12, same major as torch's CUDA
#   CUMULI_NO_CONDA_CUDA  1 = never install a toolkit or compilers into the
#                         env: fail when nvcc is missing (default: install
#                         cuda-toolkit, and gcc/g++ if the machine has none,
#                         from conda into the env)
#
# Safe to re-run: an existing env is reused, not recreated (remove it first
# with `conda env remove -n cumuli` for a clean rebuild).
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

# gsplat compiles its CUDA kernels once, with the system nvcc (not the
# CUDA runtime in the torch wheel), using -std=c++20. nvcc only accepts
# that from CUDA 12.0, and torch refuses a toolkit of another major
# version than its own wheel. Check both up front and say what to do,
# instead of surfacing a ninja traceback after the env is built.
# Usage: check_nvcc [torch_cuda_version]
check_nvcc() {
    local want="${1:-}" nvcc ver major
    nvcc="${CUDA_HOME:+$CUDA_HOME/bin/nvcc}"
    if [ -z "$nvcc" ] || [ ! -x "$nvcc" ]; then nvcc="$(command -v nvcc || true)"; fi
    if [ -z "$nvcc" ]; then
        echo "FAIL: nvcc (CUDA toolkit) not found; gsplat needs it for its one-time CUDA build." >&2
        echo "  Install a CUDA ${want:-12+} toolkit, then set CUDA_HOME and PATH, e.g.:" >&2
        echo "    export CUDA_HOME=/usr/local/cuda-${want:-13.0}; export PATH=\"\$CUDA_HOME/bin:\$PATH\"" >&2
        echo "  (Use --cpu to skip the GPU build.)" >&2
        exit 1
    fi
    ver="$("$nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
    major="${ver%%.*}"
    if [ -z "$major" ] || [ "$major" -lt 12 ]; then
        echo "FAIL: $nvcc is CUDA ${ver:-unknown}; gsplat needs nvcc 12.0 or newer (it builds with -std=c++20)." >&2
        echo "  Point at a newer toolkit before re-running, e.g.:" >&2
        echo "    export CUDA_HOME=/usr/local/cuda-${want:-13.0}; export PATH=\"\$CUDA_HOME/bin:\$PATH\"" >&2
        echo "  or remove the old one (an older nvcc earlier on PATH wins; check: which -a nvcc)." >&2
        exit 1
    fi
    if [ -n "$want" ] && [ "${want%%.*}" != "$major" ]; then
        echo "FAIL: $nvcc is CUDA $ver but this env's torch was built for CUDA $want." >&2
        echo "  Torch needs a toolkit of the same major version (${want%%.*}.x). Set CUDA_HOME to one:" >&2
        echo "    export CUDA_HOME=/usr/local/cuda-${want}; export PATH=\"\$CUDA_HOME/bin:\$PATH\"" >&2
        exit 1
    fi
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$nvcc")")")}"
    echo "nvcc: $nvcc (CUDA $ver)"
}
nvcc_on_hand() { [ -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ] || command -v nvcc >/dev/null 2>&1; }
env_exists() { "$CONDA" env list | awk -v n="$ENV_NAME" '$1 == n { found = 1 } END { exit !found }'; }
if [ "$CPU_ONLY" = 0 ]; then
    if nvcc_on_hand || [ "${CUMULI_NO_CONDA_CUDA:-0}" = 1 ]; then
        check_nvcc
    else
        echo "nvcc not found: the CUDA toolkit will be installed into the $ENV_NAME env once torch is there"
        echo "  (CUMULI_NO_CONDA_CUDA=1 fails here instead; set CUDA_HOME to use a toolkit you have)"
    fi
fi

# Everything below the gsplat build is the same on re-run, so an existing env is
# reused as it is: `conda env create` on it would die with "prefix already exists".
if env_exists; then
    echo "=== $ENV_NAME exists: reusing it (conda env remove -n $ENV_NAME for a clean rebuild) ==="
    PY="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
elif [ "$CPU_ONLY" = 1 ]; then
    echo "=== create $ENV_NAME (cpu): torch swapped BEFORE install ==="
    # The yml's cu130 torch is multi-GB and must never be downloaded on a
    # CPU runner at all. An install-then-replace would stack both wheel
    # sets on an already-tight disk budget (the failure mode the old
    # per-env CI creation was engineered around). So cpu torch is
    # installed FIRST, before the yml's requirements: no transitive
    # dependency (lpips, kornia, timm, ...) can then drag the
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
    TORCH_CUDA="$("$PY" -c 'import torch; print(torch.version.cuda)')"
    ENV_PREFIX="$("$PY" -c 'import sys; print(sys.prefix)')"
    TRIPLE="$(uname -m)-conda-linux-gnu"
    if ! nvcc_on_hand && [ "${CUMULI_NO_CONDA_CUDA:-0}" != 1 ]; then
        # The toolkit must match torch's CUDA (13.0 for torch 2.13+cu130).
        echo "installing cuda-toolkit $TORCH_CUDA into $ENV_NAME (conda, nvidia channel)"
        "$CONDA" install -y -n "$ENV_NAME" -c "nvidia/label/cuda-${TORCH_CUDA}.0" "cuda-toolkit=${TORCH_CUDA}"
        export CUDA_HOME="$ENV_PREFIX" PATH="$ENV_PREFIX/bin:$PATH"
    fi
    # No system compiler (a bare WSL2 distro): nvcc and torch's extension build
    # need a host gcc/g++, which conda can provide inside the env.
    if ! { command -v cc && command -v c++; } >/dev/null 2>&1 && [ "${CUMULI_NO_CONDA_CUDA:-0}" != 1 ]; then
        case "$(uname -m)" in x86_64) PKG_ARCH=linux-64 ;; aarch64) PKG_ARCH=linux-aarch64 ;; *) PKG_ARCH="" ;; esac
        if [ -n "$PKG_ARCH" ]; then
            echo "no C/C++ compiler on this machine: installing gcc/g++ into $ENV_NAME (conda-forge)"
            "$CONDA" install -y -n "$ENV_NAME" -c conda-forge "gcc_${PKG_ARCH}" "gxx_${PKG_ARCH}"
        fi
    fi
    if [ -x "$ENV_PREFIX/bin/$TRIPLE-cc" ] && ! command -v cc >/dev/null 2>&1; then
        export CC="$ENV_PREFIX/bin/$TRIPLE-cc" CXX="$ENV_PREFIX/bin/$TRIPLE-c++"
    fi
    # Conda's CUDA 13 keeps headers and libraries under targets/<arch>-linux, not
    # include/ and lib/, so the compiler and linker are told where they are.
    if [ "${CUDA_HOME:-}" = "$ENV_PREFIX" ] && [ ! -f "$ENV_PREFIX/include/cuda_runtime.h" ]; then
        TARGETS="$(ls -d "$ENV_PREFIX"/targets/*-linux 2>/dev/null | head -1 || true)"
        if [ -n "$TARGETS" ]; then
            export CPATH="$TARGETS/include${CPATH:+:$CPATH}"
            export LIBRARY_PATH="$TARGETS/lib:$ENV_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
        fi
    fi
    check_nvcc "$TORCH_CUDA"
    if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 || true)"
        TORCH_CUDA_ARCH_LIST="${CAP:-12.0}"
    fi
    export TORCH_CUDA_ARCH_LIST
    JOBS="$(nproc)"
    if [ "$JOBS" -gt 8 ]; then JOBS=8; fi
    export MAX_JOBS=${MAX_JOBS:-$JOBS}
    echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS"
    WARMUP_LOG="$(mktemp)"
    # The build's real error is buried in the middle of torch's traceback, so keep
    # the whole output and, on failure, print the compiler's own lines first.
    if ! "$PY" - >"$WARMUP_LOG" 2>&1 <<'WARMUP'
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
    then
        echo "FAIL: the gsplat CUDA build failed. The compiler said:" >&2
        grep -E "FAILED:|error:|fatal error|No CUDA toolkit|undefined reference|cannot find" "$WARMUP_LOG" | sort -u | head -20 >&2 || true
        echo "  (full output: $WARMUP_LOG)" >&2
        echo "  After fixing it, clear the failed build before re-running: rm -rf \"\${TORCH_EXTENSIONS_DIR:-\$HOME/.cache/torch_extensions}\"" >&2
        exit 1
    fi
    cat "$WARMUP_LOG"
    rm -f "$WARMUP_LOG"
fi

echo "=== verify ==="
if [ "$CPU_ONLY" = 1 ]; then export CUMULI_VERIFY_CPU=1; fi
"$PY" - <<'VERIFY'
import importlib, os, sys
cpu_only = os.environ.get("CUMULI_VERIFY_CPU") == "1"
core = ["torch", "torchvision", "numpy", "scipy", "PIL", "cv2", "pycolmap",
        "hloc", "plyfile", "open3d", "transformers", "gsplat",
        "lpips", "torchmetrics", "omegaconf", "huggingface_hub",
        "sapiens.pose", "kornia", "einops", "timm", "h5py",
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
