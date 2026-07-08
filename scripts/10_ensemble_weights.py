"""Learned ensemble predictor weights (fuse_ens): θ per verifier and bias b.

f_θ(V) = σ(θᵀV + b) is fit (on the est set, soft scores) to the FUSE posterior's
confidence-weighted pseudo-labels. This dumps θ_j for each verifier + b, sorted by
|θ|, as a printed table, CSV, and a bar chart — per lambda.

Usage:
    python scripts/10_ensemble_weights.py [--lambdas 0.0 1.0] [--est unlabeled]
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fuse_mm.fuse.estimate import (  # noqa: E402
    fit_mom, posterior_triplet_avg, optimize_ensemble,
)


def fit_weights(V_est):
    p_hat = posterior_triplet_avg(V_est, fit_mom(V_est))
    w, b = optimize_ensemble(V_est, p_hat)
    return w, b


def bar_plot(names, w, b, lam, out):
    order = np.argsort(w)                                  # most negative -> most positive
    plt.figure(figsize=(8, max(6, len(names) * 0.28)))
    colors = ["#c0392b" if w[i] >= 0 else "#2c6fbf" for i in order]
    plt.barh(range(len(names)), w[order], color=colors)
    plt.yticks(range(len(names)), [names[i] for i in order], fontsize=7)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("theta (verifier weight)")
    plt.title(f"ensemble predictor weights  (lambda={lam}, bias b={b:+.3f})", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"ensemble_weights_lambda_{lam}.png"), dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/fuse")
    ap.add_argument("--out", default="artifacts/reports/ensemble_weights")
    ap.add_argument("--est", default="unlabeled")
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 1.0])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for lam in args.lambdas:
        d = np.load(os.path.join(args.dir, f"lambda_{lam}",
                                 f"verifier_scores_{args.est}.npz"), allow_pickle=True)
        V = d["v"].astype(float)
        names = list(d["verifier_names"].astype(str))
        w, b = fit_weights(V)

        # CSV
        fp = os.path.join(args.out, f"ensemble_weights_lambda_{lam}.csv")
        with open(fp, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["verifier", "theta"])
            for i in np.argsort(-np.abs(w)):
                wr.writerow([names[i], f"{w[i]:.4f}"])
            wr.writerow(["__bias_b__", f"{b:.4f}"])
        bar_plot(names, w, b, lam, args.out)

        # print (top by |theta|)
        print(f"\n=== lambda={lam}   bias b = {b:+.4f} ===")
        print(f"{'verifier':28s} {'theta':>8s}")
        for i in np.argsort(-np.abs(w)):
            print(f"{names[i]:28s} {w[i]:>8.3f}")
    print(f"\nwrote CSVs + bar charts to {args.out}/ensemble_weights_lambda_*")


if __name__ == "__main__":
    main()
