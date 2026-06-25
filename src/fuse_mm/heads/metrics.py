"""Numpy implementations of the CV + metric helpers needed by Step 5.

Self-contained (no scikit-learn) so head training runs anywhere numpy does:
the network-mounted .venv on Windows and the conda env on WEXAC alike.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Cross-validation + preprocessing                                            #
# --------------------------------------------------------------------------- #
def stratified_kfold(y, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified folds: each class is shuffled then round-robin'd across folds,
    keeping class proportions ~equal in every fold. Returns (train_idx, val_idx)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_splits
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0])
            for f in range(n_splits)]


def standardize_fit(X) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_apply(X, mu, sd) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32)


# --------------------------------------------------------------------------- #
# Metrics (binary)                                                            #
# --------------------------------------------------------------------------- #
def _rankdata_avg(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # 1-based average rank
        i = j + 1
    return ranks


def roc_auc(y_true, score) -> float:
    y = np.asarray(y_true)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rankdata_avg(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y_true, score) -> float:
    y = np.asarray(y_true)
    npos = int((y == 1).sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[y == 1].sum() / npos)


def accuracy(y_true, pred) -> float:
    return float((np.asarray(y_true) == np.asarray(pred)).mean())


def balanced_accuracy(y_true, pred) -> float:
    y, p = np.asarray(y_true), np.asarray(pred)
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def f1_binary(y_true, pred) -> float:
    y, p = np.asarray(y_true), np.asarray(pred)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


def binary_metrics(y_true, prob_pos, threshold: float = 0.5) -> dict:
    pred = (np.asarray(prob_pos) >= threshold).astype(int)
    return {
        "auc": roc_auc(y_true, prob_pos),
        "ap": average_precision(y_true, prob_pos),
        "acc": accuracy(y_true, pred),
        "balanced_acc": balanced_accuracy(y_true, pred),
        "f1": f1_binary(y_true, pred),
    }
