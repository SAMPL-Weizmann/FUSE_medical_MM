"""Eigen-decomposition of the 6 conditional correlation matrices (lambda in
{0.0,0.3,1.0} x y in {0,1}).

For each C = Q diag(vals) Qᵀ it saves eigenvalues and eigenvectors. For a
correlation matrix the eigenvalues sum to m (=#verifiers); all-ones means perfect
conditional independence, so a large leading eigenvalue = strong shared structure.

Outputs to artifacts/reports/:
  cond_corr_eigenvalues.csv       rank x 6 matrices (eigenvalues, descending)
  cond_corr_eigvecs_<tag>.csv     verifier x eig index, per matrix
  cond_corr_spectrum.npz          eigvals, eigvecs, names for all 6

Usage: python scripts/11_cond_corr_spectrum.py [--set unlabeled] [--lambdas 0.0 0.3 1.0]
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_scree(eigvals, tags, out):
    plt.figure(figsize=(8.5, 5.5))
    for t in tags:
        v = eigvals[t]
        plt.plot(range(1, len(v) + 1), v, marker="o", ms=3, label=t)
    plt.axhline(1.0, color="k", ls="--", lw=1.2, label="conditional independence (all = 1)")
    plt.xlabel("eigenvalue rank"); plt.ylabel("eigenvalue")
    plt.title("Conditional-correlation eigenvalue spectra (unlabeled)")
    plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out, "cond_corr_scree.png"), dpi=140)
    plt.close()


def plot_leading_eigvec(eigvecs, tags, names, out):
    m, n = len(names), len(tags)
    ncols = 3
    nrows = (n + ncols - 1) // ncols                    # dynamic grid, any #tags
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 5 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for k, t in enumerate(tags):
        ax = flat[k]
        v = eigvecs[t][:, 0].copy()
        if v[np.argmax(np.abs(v))] < 0:                 # fix arbitrary eigvec sign
            v = -v
        ax.barh(range(m), v, color=["#c0392b" if x >= 0 else "#2c6fbf" for x in v])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"leading eigenvector — {t}", fontsize=10)
        ax.set_yticks(range(m))
        ax.set_yticklabels(names if k % ncols == 0 else [], fontsize=4)
        ax.invert_yaxis()
    for k in range(n, len(flat)):
        flat[k].axis("off")
    fig.suptitle("Dominant shared factor (leading eigenvector) per matrix", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(os.path.join(out, "cond_corr_leading_eigvec.png"), dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/fuse")
    ap.add_argument("--out", default="artifacts/reports/conditional_corr")
    ap.add_argument("--set", default="unlabeled")
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.3, 1.0])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tags, eigvals, eigvecs, names = [], {}, {}, None
    for lam in args.lambdas:
        d = np.load(os.path.join(args.dir, f"lambda_{lam}",
                                 f"verifier_scores_{args.set}.npz"), allow_pickle=True)
        V = d["v"].astype(float); y = d["y"].astype(int)
        names = list(d["verifier_names"].astype(str))
        for cls in (0, 1):
            C = np.corrcoef(V[y == cls].T)
            vals, vecs = np.linalg.eigh(C)              # ascending, C symmetric
            order = np.argsort(vals)[::-1]
            vals, vecs = vals[order], vecs[:, order]    # descending
            tag = f"lam{lam}_y{cls}"
            tags.append(tag); eigvals[tag] = vals; eigvecs[tag] = vecs

            # per-matrix eigenvector CSV (rows=verifiers, cols=eig1..eigm)
            fp = os.path.join(args.out, f"cond_corr_eigvecs_{tag}.csv")
            with open(fp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["verifier"] + [f"eig{i+1}" for i in range(len(vals))])
                for j, nm in enumerate(names):
                    w.writerow([nm] + [f"{vecs[j, i]:.4f}" for i in range(len(vals))])

    # combined eigenvalues CSV
    m = len(eigvals[tags[0]])
    with open(os.path.join(args.out, "cond_corr_eigenvalues.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank"] + tags)
        for r in range(m):
            w.writerow([r + 1] + [f"{eigvals[t][r]:.4f}" for t in tags])

    # min / max eigenvalue and their ratio (condition number) per matrix
    with open(os.path.join(args.out, "cond_corr_eig_minmax.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "y", "min_eig", "max_eig", "max_over_min"])
        for t in tags:
            lam_s, y_s = t.split("_")
            v = eigvals[t]
            mn, mx = float(v.min()), float(v.max())
            w.writerow([lam_s[3:], y_s[1:], f"{mn:.4f}", f"{mx:.4f}", f"{mx / mn:.2f}"])

    np.savez_compressed(os.path.join(args.out, "cond_corr_spectrum.npz"),
                        names=np.array(names),
                        **{f"{t}_vals": eigvals[t] for t in tags},
                        **{f"{t}_vecs": eigvecs[t] for t in tags})

    plot_scree(eigvals, tags, args.out)
    plot_leading_eigvec(eigvecs, tags, names, args.out)

    # summary
    print("leading eigenvalues (sum of all = #verifiers = %d):" % m)
    print(f"{'matrix':12s} {'lam1':>7} {'lam2':>7} {'lam3':>7} {'lam4':>7} {'lam5':>7}  {'part.ratio':>10}")
    for t in tags:
        v = eigvals[t]
        pr = (v.sum() ** 2) / (v ** 2).sum()            # participation ratio (eff. rank)
        print(f"{t:12s} " + " ".join(f"{v[i]:7.2f}" for i in range(5)) + f"  {pr:10.1f}")
    print("\nmin / max eigenvalue and ratio (condition number) per matrix:")
    print(f"{'lambda':>7} {'y':>3} {'min':>8} {'max':>8} {'max/min':>9}")
    for t in tags:
        lam_s, y_s = t.split("_")
        v = eigvals[t]
        print(f"{lam_s[3:]:>7} {y_s[1:]:>3} {v.min():8.4f} {v.max():8.3f} {v.max() / v.min():9.2f}")

    print(f"\nwrote eigenvalues CSV, min/max CSV, {len(tags)} eigenvector CSVs, "
          f"and spectrum.npz to {args.out}/")


if __name__ == "__main__":
    main()
