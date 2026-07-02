"""Per-verifier and per-prediction metrics (binary; threshold soft scores at 0.5)."""

from __future__ import annotations

import numpy as np

from ..heads.metrics import roc_auc


def _sens_spec(pred01: np.ndarray, y01: np.ndarray):
    tp = int(((pred01 == 1) & (y01 == 1)).sum())
    fn = int(((pred01 == 0) & (y01 == 1)).sum())
    tn = int(((pred01 == 0) & (y01 == 0)).sum())
    fp = int(((pred01 == 1) & (y01 == 0)).sum())
    sens = tp / max(tp + fn, 1)          # P(pred=1 | y=1)
    spec = tn / max(tn + fp, 1)          # P(pred=0 | y=0)
    return sens, spec


def score_stats(score: np.ndarray, y01: np.ndarray, threshold: float = 0.5) -> dict:
    """Metrics for a soft score in [0,1] against labels y in {0,1}."""
    pred = (np.asarray(score) >= threshold).astype(int)
    sens, spec = _sens_spec(pred, y01)
    return {
        "acc": float((pred == y01).mean()),
        "balanced_acc": float((sens + spec) / 2),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "auc": roc_auc(y01, score),
    }
