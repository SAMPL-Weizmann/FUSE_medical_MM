#!/usr/bin/env bash
# Source this on WEXAC before running anything:  source scripts/wexac/env.sh
#
# It points the (Windows-default) config at the Linux mount paths via env vars,
# so configs/data.yaml is never edited per machine.
#
# WEXAC Linux mount paths (isi 'hsd' share -> /home/hsd). Confirmed present.
_STUDY="/home/hsd/yonina/Studies_under_Helsinki_approval/Beilinson_Multi_Modality_Dr_Ahuva_Grubstein/Data/final Data adi project"

# Linux path to the folder that contains MG_CLEAN_NPY and US_CLEAN_NPY:
export FUSE_DATASET_ROOT="$_STUDY/FINAL DATA/MG & US"

# Linux path to the paired label table:
export FUSE_LABEL_TABLE="$_STUDY/FINAL TABLES/Tables - Useful/clean_mm.xlsx"

# Keep artifacts on shared storage so jobs/login nodes all see them.
# Default (project-relative) is fine; override only if you want them elsewhere:
# export FUSE_SPLITS_PATH="$HOME/fuse_artifacts/splits.json"
# export FUSE_FEATURES_DIR="$HOME/fuse_artifacts/features"

# HuggingFace weight cache — put it on storage with quota/space, NOT /tmp.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "FUSE env set:"
echo "  FUSE_DATASET_ROOT = $FUSE_DATASET_ROOT"
echo "  FUSE_LABEL_TABLE  = $FUSE_LABEL_TABLE"
echo "  HF_HOME           = $HF_HOME"
