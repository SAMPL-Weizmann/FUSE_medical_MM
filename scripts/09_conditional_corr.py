"""Conditional correlation matrices of the trained verifiers, given the true label.

For each (lambda, class y in {0,1}) it takes the verifier scores on patients with
that label and plots the m x m Pearson correlation heatmap. Under TCI (conditional
independence) the off-diagonals -> 0, so lambda=1 should look paler than lambda=0.
The mean |off-diagonal| (a scalar TCI proxy) is shown in each title.

Usage:
    python scripts/09_conditional_corr.py [--set unlabeled] [--lambdas 0.0 1.0]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def cond_corr(V, y, cls):
    X = V[y == cls]
    C = np.corrcoef(X.T)                       # (m, m)
    off = C[~np.eye(C.shape[0], dtype=bool)]
    return C, float(np.mean(np.abs(off)))


def plot_corr(C, names, title, out_path):
    m = len(names)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)          # diverging, centered at 0
    ax.set_xticks(range(m)); ax.set_yticks(range(m))
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_yticklabels(names, fontsize=5)
    # modality block separators (names sorted -> MG block then US block)
    for k in range(1, m):
        if names[k].split("__")[0] != names[k - 1].split("__")[0]:
            ax.axhline(k - 0.5, color="k", lw=1.0)
            ax.axvline(k - 0.5, color="k", lw=1.0)
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/fuse")
    ap.add_argument("--out", default="artifacts/reports/conditional_corr")
    ap.add_argument("--set", default="unlabeled")
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 1.0])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"mean |off-diagonal| conditional correlation ({args.set}):")
    print(f"{'lambda':>7} {'y=0':>8} {'y=1':>8}")
    for lam in args.lambdas:
        d = np.load(os.path.join(args.dir, f"lambda_{lam}",
                                 f"verifier_scores_{args.set}.npz"), allow_pickle=True)
        V = d["v"].astype(float); y = d["y"].astype(int)
        names = list(d["verifier_names"].astype(str))
        row = [f"{lam:>7}"]
        for cls in (0, 1):
            C, moff = cond_corr(V, y, cls)
            title = (f"conditional corr | y={cls} | lambda={lam}  "
                     f"(mean|off-diag|={moff:.3f})")
            fp = os.path.join(args.out, f"cond_corr_lambda_{lam}_y{cls}.png")
            plot_corr(C, names, title, fp)
            row.append(f"{moff:>8.3f}")
        print(" ".join(row))
    print(f"\nwrote 4 heatmaps to {args.out}/cond_corr_lambda_*_y*.png")


if __name__ == "__main__":
    main()
