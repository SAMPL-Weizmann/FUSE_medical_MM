"""Faithful Weaver adapter around the real Snorkel-MeTaL LabelModel (metal-ama).

Weaver (Saad-Falcon et al. 2025b) = binarize verifiers, estimate the class
balance P(y=1) from a small labeled anchor (S_L), optionally drop imbalanced
verifiers, then fit MeTaL's weak-supervision LabelModel — a method-of-moments
estimator assuming (joint) conditional independence — on the pool votes, and
predict per set. This mirrors weaver/models.py::WeakSupervised.fit/predict_proba.

Requires `metal-ama` (github.com/mayeechen/metal-ama); install via
scripts/wexac/setup_weaver.sh into the fuse_mm env.
"""

from __future__ import annotations

import numpy as np


def _binarize(V, thr):
    return (V >= thr).astype(int)          # {0,1}


def run_weaver(V, Y, est_set="unlabeled", train_set="labeled", threshold=0.5,
               drop_imbalanced=True, drop_lo=0.05, drop_hi=0.95,
               mu_epochs=5000, seed=123):
    from metal.label_model import LabelModel   # metal-ama

    # class balance P(y=1) from the labeled anchor (semi-supervised part)
    cb = float(np.asarray(Y[train_set]).mean())
    class_balance = np.array([1.0 - cb, cb])   # [P(normal), P(abnormal)]

    Lbin = {s: _binarize(Vs, threshold) for s, Vs in V.items()}

    # optionally drop verifiers whose pool marginal is too extreme (Weaver's
    # "adaptively drop verifiers"); keep all if that would leave < 3.
    keep = np.arange(Lbin[est_set].shape[1])
    if drop_imbalanced:
        marg = Lbin[est_set].mean(axis=0)
        cand = np.where((marg > drop_lo) & (marg < drop_hi))[0]
        if len(cand) >= 3:
            keep = cand

    # MeTaL wants labels in {1,...,k}; binary votes {0,1} -> {1,2}. class 2 = abnormal.
    lm = LabelModel(k=2, seed=seed)
    lm.train_model(Lbin[est_set][:, keep] + 1, class_balance=class_balance,
                   mu_epochs=mu_epochs, verbose=False)

    out = {}
    for s in V:
        proba = lm.predict_proba(Lbin[s][:, keep] + 1)   # (N, 2)
        out[s] = proba[:, 1]                              # P(class 2 = abnormal)
    return out
