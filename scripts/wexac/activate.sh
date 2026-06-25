#!/usr/bin/env bash
# Source (do NOT execute) to activate the project's conda env on WEXAC.
# WEXAC exposes conda via the module system, so `conda` is not on PATH in
# non-interactive (bsub) shells until the module is loaded.
#
#   source scripts/wexac/activate.sh
#
# Override the module/env names via env vars if they ever change.
CONDA_MODULE="${CONDA_MODULE:-miniconda/26.1.1_environmentally}"
ENV_NAME="${ENV_NAME:-fuse_mm}"

# conda/module init scripts aren't always nounset-clean; relax briefly.
set +u
module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u 2>/dev/null || true

echo "conda env active: $ENV_NAME  (python: $(command -v python))"
