"""Decision-threshold selection for turning a soft score in [0,1] into a 0/1 call.

The benchmark scores every method by thresholding its soft output (default 0.5,
metrics.score_stats). At strong class imbalance a fixed 0.5 is a poor operating
point: a well-calibrated model (e.g. the oracle at 7% malignant prevalence) puts
almost every score below 0.5 and predicts all-negative, so balanced accuracy
collapses to 0.5 even though the RANKING (AUC/AP) is excellent.

These rules pick the threshold tau from a fit set instead of hardcoding 0.5:

  "fixed"               tau = 0.5                          (the default; unchanged)
  "prevalence"          tau = the (1 - pi) quantile of the fit scores, so the top
                        pi fraction is flagged. pi defaults to the fit-set positive
                        rate (the known malignancy prevalence). LABEL-FREE if a
                        prevalence is supplied; uses only the base rate otherwise.
  "youden"              tau = argmax_tau (sensitivity + specificity) on the fit set
                        (= the balanced-accuracy-maximizing point). Uses labels.
  "target_sensitivity"  the lowest tau that still catches >= `target` of the fit
                        positives (default 0.95). Uses labels. The screening choice.

DISCIPLINE: fit tau on a NON-test set (labeled S_L, or unlabeled for prevalence)
and apply it to the held-out test scores. Fitting tau on the test scores/labels
is leakage. choose_threshold() only sees the fit set; the caller applies the
returned tau to the other sets.
"""

from __future__ import annotations

import numpy as np

RULES = ("fixed", "prevalence", "youden", "target_sensitivity")


def choose_threshold(rule: str, fit_score: np.ndarray, fit_y: np.ndarray, *,
                     prevalence: float | None = None,
                     target_sensitivity: float = 0.95) -> float:
    """Return a scalar decision threshold tau in [0,1] for the given rule.

    fit_score/fit_y: soft scores and 0/1 labels of the FIT set (never the test set).
    Degenerate fit sets (no positives or no negatives) fall back to tau=0.5.
    """
    s = np.asarray(fit_score, dtype=float)
    y = np.asarray(fit_y, dtype=int)

    if rule == "fixed":
        return 0.5

    if rule == "prevalence":
        pi = float(y.mean()) if prevalence is None else float(prevalence)
        pi = min(max(pi, 1e-6), 1.0 - 1e-6)
        return float(np.quantile(s, 1.0 - pi))

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:                      # can't fit a sens/spec tradeoff
        return 0.5

    if rule == "youden":
        # candidate cuts = midpoints between sorted unique scores (bounded for cost)
        u = np.unique(s)
        if u.size > 400:
            u = np.quantile(s, np.linspace(0.0, 1.0, 400))
            u = np.unique(u)
        cands = (u[:-1] + u[1:]) / 2.0 if u.size > 1 else u
        best_tau, best_j = 0.5, -np.inf
        P, N = n_pos, n_neg
        for t in cands:
            pred = s >= t
            sens = np.count_nonzero(pred & (y == 1)) / P
            spec = np.count_nonzero((~pred) & (y == 0)) / N
            j = sens + spec - 1.0
            if j > best_j:
                best_j, best_tau = j, float(t)
        return best_tau

    if rule == "target_sensitivity":
        pos = s[y == 1]
        # lowest tau catching >= target of positives = (1 - target) quantile of positives
        tau = float(np.quantile(pos, max(0.0, 1.0 - target_sensitivity)))
        return min(max(tau, 0.0), 1.0)

    raise ValueError(f"unknown threshold rule {rule!r} (valid: {RULES})")
