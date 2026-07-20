"""Per-prediction metrics (binary; threshold soft scores at 0.5).

Reports the confusion counts (TP/TN/FP/FN) and everything derivable from them
plus the threshold-free AUC / average precision.
"""

from __future__ import annotations

import math

import numpy as np

from ..heads.metrics import average_precision, roc_auc


def _sens_spec(pred01: np.ndarray, y01: np.ndarray):
    tp = int(((pred01 == 1) & (y01 == 1)).sum())
    fn = int(((pred01 == 0) & (y01 == 1)).sum())
    tn = int(((pred01 == 0) & (y01 == 0)).sum())
    fp = int(((pred01 == 1) & (y01 == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return sens, spec


def _safe(a, b):
    return a / b if b else 0.0


def score_stats(score: np.ndarray, y01: np.ndarray, threshold=0.5) -> dict:
    """Full metric set for a soft score in [0,1] against labels y in {0,1}.

    `threshold` may be a scalar or a per-sample array (same length as `score`) —
    the latter lets pooled out-of-fold metrics apply each fold's own decision
    threshold. Only the confusion-count metrics depend on it; AUC/AP are
    threshold-free.
    """
    score = np.asarray(score, dtype=float)
    y = np.asarray(y01, dtype=int)
    pred = (score >= threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    sens = _safe(tp, tp + fn)            # recall / TPR
    spec = _safe(tn, tn + fp)            # TNR
    prec = _safe(tp, tp + fp)            # PPV
    npv = _safe(tn, tn + fn)
    f1 = _safe(2 * prec * sens, prec + sens)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe(tp * tn - fp * fn, denom) if denom else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": _safe(tp + tn, tp + tn + fp + fn),
        "balanced_acc": (sens + spec) / 2,
        "sensitivity": sens, "recall": sens, "specificity": spec,
        "precision": prec, "npv": npv, "f1": f1, "mcc": mcc,
        "youden_j": sens + spec - 1.0,
        "auc": roc_auc(y, score),
        "ap": average_precision(y, score),
    }
