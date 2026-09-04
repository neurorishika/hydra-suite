#!/usr/bin/env bash
# Build the "hydra-sam3" sidecar conda env used by DetectKit's SAM3 LoRA
# training role. Kept out of the hydra/hydra-mps/hydra-cuda envs on purpose:
# Meta's `sam3` pins `numpy<2`, which conflicts with the numpy>=2 runtimes
# those envs use for everything else. See
# docs/user-guide/detectkit-semantic-escalation.md#building-the-hydra-sam3-env
# for the recipe this script automates, including why each pin exists.
#
# Usage:
#   tools/setup_sam3_train_env.sh [cpu|mps|cuda12|cuda13]
#
# Env vars:
#   SAM3_ENV_NAME   conda env name to create (default: hydra-sam3)
#   REPO_ROOT       path to this hydra-suite checkout to install editable
#                    (default: repo root containing this script)

set -euo pipefail

PLATFORM="${1:-}"
SAM3_ENV_NAME="${SAM3_ENV_NAME:-hydra-sam3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [ -z "$PLATFORM" ]; then
    case "$(uname -s)" in
        Darwin) PLATFORM="mps" ;;
        *)
            if command -v nvidia-smi >/dev/null 2>&1; then
                PLATFORM="cuda12"
            else
                PLATFORM="cpu"
            fi
            ;;
    esac
    echo "No platform given; auto-detected '$PLATFORM' (pass cpu|mps|cuda12|cuda13 to override)."
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Activate/init conda first." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$SAM3_ENV_NAME"; then
    echo "conda env '$SAM3_ENV_NAME' already exists; reusing it (delete it first for a clean rebuild: conda env remove -n $SAM3_ENV_NAME)."
else
    echo "Creating conda env '$SAM3_ENV_NAME' (python=3.12, numpy<2)..."
    conda create -y -n "$SAM3_ENV_NAME" python=3.12 'numpy<2'
fi

case "$PLATFORM" in
    cpu)
        TORCH_INSTALL=(pip install torch torchvision)
        ;;
    mps)
        TORCH_INSTALL=(pip install torch torchvision)
        ;;
    cuda12)
        TORCH_INSTALL=(pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128)
        ;;
    cuda13)
        TORCH_INSTALL=(pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130)
        ;;
    *)
        echo "ERROR: unknown platform '$PLATFORM' (expected cpu|mps|cuda12|cuda13)" >&2
        exit 1
        ;;
esac

echo "Installing torch/torchvision for platform '$PLATFORM'..."
conda run -n "$SAM3_ENV_NAME" "${TORCH_INSTALL[@]}"

echo "Installing setuptools<81 (sam3/model_builder.py needs pkg_resources)..."
conda run -n "$SAM3_ENV_NAME" pip install 'setuptools<81'

echo "Installing sam3's training dependencies (declared + the transitive ones it misses)..."
# scipy>=1.14 requires numpy>=2.0 and breaks the numpy<2 pin sam3 needs, so
# it must be pinned below that regardless of what else asks for a newer one.
conda run -n "$SAM3_ENV_NAME" pip install einops torchmetrics 'scipy<1.14' decord iopath \
    opencv-python-headless pillow platformdirs pandas numba pycocotools psutil

echo "Installing Meta's sam3 from source (not on PyPI)..."
conda run -n "$SAM3_ENV_NAME" pip install git+https://github.com/facebookresearch/sam3.git

echo "Installing hydra-suite editable from $REPO_ROOT (--no-deps: pyproject's"
echo "unconstrained numpy>=1.24 would otherwise resolve to numpy>=2 and clobber"
echo "the numpy<2 pin sam3 needs; every runtime dep the CLI actually uses was"
echo "already installed explicitly above)..."
conda run -n "$SAM3_ENV_NAME" pip install --no-deps -e "$REPO_ROOT"

echo
echo "Done. In DetectKit's SAM3 training tab, set the env name to '$SAM3_ENV_NAME'"
echo "(or export HYDRA_SAM3_ENV=$SAM3_ENV_NAME) and click Check to verify."
