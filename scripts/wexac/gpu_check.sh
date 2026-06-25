#!/usr/bin/env bash
# Quick GPU + env sanity check. Run inside a GPU job:
#   bsub -q interactive-gpu -gpu "num=1:j_exclusive=yes" -Is bash scripts/wexac/gpu_check.sh
set -eo pipefail

source scripts/wexac/activate.sh
source scripts/wexac/env.sh

echo "=== nvidia-smi ==="
nvidia-smi

echo "=== torch CUDA ==="
python scripts/wexac/_cuda_check.py

echo "=== dataset visible from compute node? ==="
ls "$FUSE_DATASET_ROOT" | head
