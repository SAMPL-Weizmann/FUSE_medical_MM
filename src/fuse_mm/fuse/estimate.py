"""FUSE stage-2/3: Method-of-Moments estimation + posterior (Jaffe et al. 2015 /
Candes et al., Thm 2.3, Prop C.1).

Verifier scores are soft in [0,1]; internally mapped to [-1,1] (decision: soft
2v-1). Under class-conditional independence the centered 2nd/3rd moments are
rank-1 in a per-verifier skill u; from (mu, u, b) we get generalized
sensitivity/specificity (eq 3) and the triplet-averaged posterior (Prop C.1).

Parameters can come from three sources -> the three methods share one posterior:
  fit_mom(S_U)          -> FUSE   (unsupervised)
  params_from_labels(y) -> Oracle (eval-set GT) / a labelled estimate
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

EPS = 1e-8


def _to_pm1(V: np.ndarray) -> np.ndarray:
    return 2.0 * V - 1.0


# --------------------------------------------------------------------------- #
# Moments + rank-1 recovery                                                    #
# --------------------------------------------------------------------------- #
def moments(Vpm: np.ndarray):
    mu = Vpm.mean(axis=0)
    Vc = Vpm - mu
    n = Vc.shape[0]
    Sigma = (Vc.T @ Vc) / n
    T = np.einsum("na,nb,nc->abc", Vc, Vc, Vc) / n
    return mu, Sigma, T


def recover_u(Sigma: np.ndarray) -> np.ndarray:
    """Rank-1 factor of the OFF-diagonal of Sigma: Sigma_{ij}=u_i u_j (i!=j).

    u_j^2 = median_{i<k, i,k!=j} Sigma_{ji} Sigma_{jk} / Sigma_{ik}. Sign fixed by
    Assumption 2.1 (majority of verifiers better than random -> majority u_j>0).
    """
    m = Sigma.shape[0]
    u = np.zeros(m)
    for j in range(m):
        ests = []
        for i in range(m):
            if i == j:
                continue
            for k in range(i + 1, m):
                if k == j or abs(Sigma[i, k]) < EPS:
                    continue
                val = Sigma[j, i] * Sigma[j, k] / Sigma[i, k]
                if val > 0:
                    ests.append(val)
        u[j] = np.sqrt(np.median(ests)) if ests else 0.0

    anchor = int(np.argmax(np.abs(u)))
    signs = np.ones(m)
    for j in range(m):
        if j != anchor and abs(Sigma[anchor, j]) > 0:
            signs[j] = np.sign(Sigma[anchor, j])
    u = u * signs
    if (u > 0).sum() < m / 2.0:          # Assumption 2.1
        u = -u
    return u


def recover_b(T: np.ndarray, u: np.ndarray) -> float:
    """Class imbalance b = P(y=1)-P(y=-1) from T_{ijk}=w_i w_j w_k, w=rho*u.

    T_{ijk}/(u_i u_j u_k) = rho^3 (const); rho^3 = -2b/sqrt(1-b^2)  =>  solve b.
    """
    m = len(u)
    ests = []
    for i, j, k in combinations(range(m), 3):
        d = u[i] * u[j] * u[k]
        if abs(d) > EPS:
            ests.append(T[i, j, k] / d)
    if not ests:
        return 0.0
    s = float(np.median(ests))                    # rho^3
    b = -np.sign(s) * np.sqrt(s * s / (4.0 + s * s))
    return float(np.clip(b, -0.98, 0.98))


def psi_eta(mu: np.ndarray, u: np.ndarray, b: float):
    """Generalized sensitivity/specificity, eq (3)."""
    r = np.sqrt((1.0 - b) / (1.0 + b))
    psi = 0.5 * (1.0 + mu + u * r)
    eta = 0.5 * (1.0 - mu + u * r)
    return psi, eta


# --------------------------------------------------------------------------- #
# Parameter sources                                                            #
# --------------------------------------------------------------------------- #
def fit_mom(V_est: np.ndarray) -> dict:
    """Unsupervised MoM on an estimation set (S_U). V_est in [0,1]."""
    Vpm = _to_pm1(V_est)
    mu, Sigma, T = moments(Vpm)
    u = recover_u(Sigma)
    b = recover_b(T, u)
    psi, eta = psi_eta(mu, u, b)
    return {"mu": mu, "u": u, "b": b, "psi": psi, "eta": eta}


def params_from_labels(V: np.ndarray, y_pm1: np.ndarray) -> dict:
    """Parameters from labels (y in {-1,+1}) via class-conditional means.
    Used for the Oracle (eval-set GT). psi=(E[v|y=1]+1)/2, eta=(1-E[v|y=-1])/2."""
    Vpm = _to_pm1(V)
    m1 = Vpm[y_pm1 == 1].mean(axis=0)
    m0 = Vpm[y_pm1 == -1].mean(axis=0)
    psi = (m1 + 1.0) / 2.0
    eta = (1.0 - m0) / 2.0
    b = float(np.clip(y_pm1.mean(), -0.98, 0.98))
    return {"mu": Vpm.mean(axis=0), "u": None, "b": b, "psi": psi, "eta": eta}


# --------------------------------------------------------------------------- #
# Posterior (Prop C.1) — triplet-averaged                                      #
# --------------------------------------------------------------------------- #
def posterior_triplet_avg(V: np.ndarray, params: dict) -> np.ndarray:
    """Return P(y=1 | v) per row, averaged over all C(m,3) verifier triplets.

    Per verifier the eq-(13) factors (log-space, clipped for soft robustness):
      y=+1 term: 1 + v*(2*psi-1);   y=-1 term: 1 + v*(1-2*eta).
    """
    Vpm = _to_pm1(V)
    psi, eta, b = params["psi"], params["eta"], params["b"]
    t1 = np.clip(1.0 + Vpm * (2.0 * psi - 1.0), 1e-6, None)
    tn = np.clip(1.0 + Vpm * (1.0 - 2.0 * eta), 1e-6, None)
    lt1, ltn = np.log(t1), np.log(tn)
    base1, basen = np.log(1.0 + b), np.log(1.0 - b)

    m = V.shape[1]
    acc = np.zeros(V.shape[0])
    n_trip = 0
    for i, j, k in combinations(range(m), 3):
        L1 = base1 + lt1[:, i] + lt1[:, j] + lt1[:, k]
        Ln = basen + ltn[:, i] + ltn[:, j] + ltn[:, k]
        acc += 1.0 / (1.0 + np.exp(-(L1 - Ln)))   # sigmoid(L1-Ln)
        n_trip += 1
    return acc / max(n_trip, 1)
