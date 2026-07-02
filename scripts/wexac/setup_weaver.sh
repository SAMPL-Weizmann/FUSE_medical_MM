#!/usr/bin/env bash
# One-time: install metal-ama (Snorkel-MeTaL fork) into the fuse_mm env so the
# Weaver baseline can run. Run on the WEXAC login node:
#   bash scripts/wexac/setup_weaver.sh
set -eo pipefail

source scripts/wexac/activate.sh        # module load miniconda + conda activate fuse_mm

# metal-ama deps not already in the env. tensorboardX is pulled in by metal's
# package __init__ (EndModel -> logging), even though we only use LabelModel.
python -m pip install dill "networkx>=2.2" pandas scikit-learn tensorboardX

mkdir -p external
if [ -d external/metal-ama/.git ]; then
  git -C external/metal-ama pull --ff-only
else
  git clone --depth 1 https://github.com/mayeechen/metal-ama.git external/metal-ama
fi
python -m pip install -e external/metal-ama

python -c "from metal.label_model import LabelModel; print('metal-ama import OK')"
echo "weaver ready"
