"""Step 5 — classifier heads on frozen backbone features (the verifiers).

Config-driven (configs/train.yaml): aggregation mode (per_patient | per_image),
head type (linear | mlp), and stratified patient-level k-fold CV are all chosen
by config, so each experiment combination is one edit, not a code change.
"""

from .config import load_train_config
from .data import FeatureSet, load_feature_npz
from .train import run_cv

__all__ = ["load_train_config", "FeatureSet", "load_feature_npz", "run_cv"]
