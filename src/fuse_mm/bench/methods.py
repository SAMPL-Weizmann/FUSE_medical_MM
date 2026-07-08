"""Ensembling methods. Each returns {set_name: soft score in [0,1]} per patient.

Inputs:
  V : dict set -> (N_set, m) verifier scores in [0,1]
  Y : dict set -> (N_set,)  labels in {0,1}   (used only by supervised/oracle)
Unsupervised: majority_vote, naive_ensemble, fuse.
Supervised (use S_L): obv, logistic, gaussian_nb, (weaver).
Oracle: uses each set's own labels (ceiling).
"""

from __future__ import annotations

import numpy as np

from ..fuse.estimate import (
    apply_g_tau, fit_mom, optimize_ensemble as _opt_ensemble, params_from_labels,
    posterior_triplet_avg, predict_ensemble, search_g_tau,
)
from .metrics import _sens_spec

TRAIN = "labeled"
EST = "unlabeled"


# ---- unsupervised ---------------------------------------------------------- #
def majority_vote(V, Y=None):
    return {s: (Vs >= 0.5).mean(axis=1) for s, Vs in V.items()}   # fraction voting abnormal


def naive_ensemble(V, Y=None):
    return {s: Vs.mean(axis=1) for s, Vs in V.items()}


def fuse(V, Y=None, est_set=EST, binarize=False, optimize_ensemble=False):
    """FUSE (Algorithm 1). Two optional flags:
      binarize          : steps 2-3, apply g_tau* (per-verifier thresholds that
                          minimize empirical TCI) -> binary V~ before the MoM.
      optimize_ensemble : steps 5-7, fit a logistic f_theta on the FUSE posterior
                          (eq. 7) and predict with it instead of the raw posterior.
    Both use no labels -> all four combinations are fully unsupervised.
    """
    est = V[est_set]
    tau = search_g_tau(est) if binarize else None
    if binarize:
        est = apply_g_tau(est, tau)
    params = fit_mom(est)                                    # Thm 2.3 (step 4)

    phat = {}
    for s, Vs in V.items():
        Vs_e = apply_g_tau(Vs, tau) if binarize else Vs
        phat[s] = posterior_triplet_avg(Vs_e, params)        # eq. 9
    if not optimize_ensemble:
        return phat

    # f_theta is fit on / applied to the ORIGINAL (soft) V (eq. 7), not V~
    w, b = _opt_ensemble(V[est_set], phat[est_set])
    return {s: predict_ensemble(Vs, w, b) for s, Vs in V.items()}


def fuse_bin(V, Y=None):        # step 2-3 only
    return fuse(V, Y, binarize=True)


def fuse_ens(V, Y=None):        # step 5-7 only
    return fuse(V, Y, optimize_ensemble=True)


def fuse_full(V, Y=None):       # full Algorithm 1
    return fuse(V, Y, binarize=True, optimize_ensemble=True)


# ---- supervised ------------------------------------------------------------ #
def obv(V, Y):
    """Oracle Best Verifier: per set, the single verifier with best balanced acc."""
    out = {}
    for s, Vs in V.items():
        y = Y[s]
        best_j, best_ba = 0, -1.0
        for j in range(Vs.shape[1]):
            sens, spec = _sens_spec((Vs[:, j] >= 0.5).astype(int), y)
            ba = (sens + spec) / 2
            if ba > best_ba:
                best_ba, best_j = ba, j
        out[s] = Vs[:, best_j]
    return out


def _fit_logreg(X, y, epochs=800, lr=0.5, l2=1e-3):
    n, m = X.shape
    w = np.zeros(m); b = 0.0
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return w, b


def logistic(V, Y, train_set=TRAIN):
    w, b = _fit_logreg(V[train_set], Y[train_set].astype(float))
    return {s: 1.0 / (1.0 + np.exp(-(Vs @ w + b))) for s, Vs in V.items()}


def gaussian_nb(V, Y, train_set=TRAIN, eps=1e-4):
    """Gaussian Naive Bayes: class-conditional Gaussian per verifier (from S_L),
    combined assuming conditional independence."""
    Xtr, ytr = V[train_set], Y[train_set]
    mu1, var1 = Xtr[ytr == 1].mean(0), Xtr[ytr == 1].var(0) + eps
    mu0, var0 = Xtr[ytr == 0].mean(0), Xtr[ytr == 0].var(0) + eps
    logprior = np.log(max((ytr == 1).mean(), eps) / max((ytr == 0).mean(), eps))

    def llr(Vs):
        # sum_j [ logN(v;mu1,var1) - logN(v;mu0,var0) ] + logprior
        t1 = -0.5 * ((Vs - mu1) ** 2 / var1 + np.log(var1))
        t0 = -0.5 * ((Vs - mu0) ** 2 / var0 + np.log(var0))
        z = (t1 - t0).sum(axis=1) + logprior
        return 1.0 / (1.0 + np.exp(-z))

    return {s: llr(Vs) for s, Vs in V.items()}


# ---- oracle ---------------------------------------------------------------- #
def oracle(V, Y):
    """Ceiling: the FUSE posterior with TRUE parameters (eval-set GT) per set."""
    out = {}
    for s, Vs in V.items():
        params = params_from_labels(Vs, 2 * Y[s] - 1)
        out[s] = posterior_triplet_avg(Vs, params)
    return out


# ---- weaver (pluggable) ---------------------------------------------------- #
def weaver(V, Y, train_set=TRAIN):
    """Faithful Weaver: real MeTaL LabelModel + binarize/drop/prior (weaver_ext).
    Skipped gracefully if metal-ama isn't installed."""
    try:
        from .weaver_ext import run_weaver
    except Exception as e:  # noqa: BLE001
        raise NotImplementedError(f"weaver: metal-ama not importable ({e})")
    try:
        return run_weaver(V, Y, train_set=train_set)
    except ImportError as e:
        raise NotImplementedError(f"weaver: metal-ama not installed ({e})")


UNSUPERVISED = {"majority_vote": majority_vote, "naive_ensemble": naive_ensemble,
                "fuse": fuse, "fuse_bin": fuse_bin, "fuse_ens": fuse_ens,
                "fuse_full": fuse_full}
SUPERVISED = {"obv": obv, "logistic": logistic, "gaussian_nb": gaussian_nb, "weaver": weaver}
CEILING = {"oracle": oracle}
