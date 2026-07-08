"""Cross-validation of the full FUSE pipeline (train -> estimate -> infer).

10-fold rotation over the cohort: each fold is S_L once, the next fold (circular)
is Test, the remaining 8 are S_U -> every patient is tested out-of-fold exactly
once. All methods are benchmarked per fold, at each lambda, on S_L/S_U/Test.
"""

from .folds import make_cv_folds, fold_assignment, load_cv_folds
from .run import run_cv

__all__ = ["make_cv_folds", "fold_assignment", "load_cv_folds", "run_cv"]
