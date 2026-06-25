#!/usr/bin/env bash
# One-time environment build on WEXAC (Linux). Run from the repo root:
#   bash scripts/wexac/setup_env.sh
#
# Creates a conda env with a CUDA-enabled torch + all deps. Adjust the CUDA
# index URL to match WEXAC's driver/CUDA (cu126 / cu124 / cu128 — check with
# `nvidia-smi` on a GPU node).
set -eo pipefail

ENV_NAME="${ENV_NAME:-fuse_mm}"
PY_VER="${PY_VER:-3.12}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"
CONDA_MODULE="${CONDA_MODULE:-miniconda/26.1.1_environmentally}"

# --- conda via WEXAC module system ------------------------------------------
set +u
module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda create -y -n "$ENV_NAME" "python=$PY_VER"
conda activate "$ENV_NAME"
set -u 2>/dev/null || true

python -m pip install --upgrade pip

# torch+torchvision from the CUDA index, the rest from PyPI
python -m pip install torch torchvision --index-url "$CUDA_INDEX"
python -m pip install numpy pyyaml timm pillow tqdm open_clip_torch transformers

echo "=== sanity check ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda build:", torch.version.cuda,
      "| cuda available:", torch.cuda.is_available())
PY
echo "done. Activate later with:  conda activate $ENV_NAME   (or: source .venv-linux/bin/activate)"
