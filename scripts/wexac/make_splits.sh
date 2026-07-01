#!/usr/bin/env bash
# Build the split on WEXAC with the env + Linux paths set. Light enough for the
# login node (it lists patient folders to validate the MG canonical views).
#   bash scripts/wexac/make_splits.sh            # write splits.json
#   bash scripts/wexac/make_splits.sh --dry-run  # preview only
set -eo pipefail

source scripts/wexac/activate.sh       # module load miniconda + conda activate fuse_mm
source scripts/wexac/env.sh            # FUSE_DATASET_ROOT / FUSE_LABEL_TABLE / ...

python -u scripts/01_make_splits.py "$@"
