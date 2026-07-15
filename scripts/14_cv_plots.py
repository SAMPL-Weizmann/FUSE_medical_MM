"""Plots from the merged cross-validation results.

Reads artifacts/reports/cv/cv_results.json (produced by 12_cv.py per-lambda then
merged by 13_cv_merge.py) and writes presentation figures to the same dir:

  - cv_balanced_acc_vs_lambda.png   test + unlabeled(OOF) bAcc vs TCI weight,
                                    focused method set, fold mean +/- std error bars
  - cv_ranking_test.png             Cleveland dot plot: every method ranked by test
                                    bAcc at a chosen lambda, colored by category
  - cv_soft_vs_binary.png           soft (fuse/fuse_ens) vs binary (fuse_bin/
                                    fuse_full) FUSE variants vs lambda
  - cv_metric_grid.png              small multiples: bAcc / F1 / MCC / AUC dot plots

Design notes: Okabe-Ito colorblind-safe palette, assigned in a FIXED order (never
cycled); line style + marker are redundant with color so identity is never
color-alone; line plots use a zoomed y-axis (honest for lines), the ranking uses
a DOT plot (not truncated bars) so near-equal accuracies read without implying a
false zero baseline.

Usage:
    python scripts/14_cv_plots.py [--results artifacts/reports/cv/cv_results.json]
                                  [--rank-lambda 0.0]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- Okabe-Ito colorblind-safe palette (fixed roles) ----------------------- #
OI = {
    "blue": "#0072B2", "sky": "#56B4E9", "green": "#009E73", "orange": "#E69F00",
    "vermillion": "#D55E00", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000",
}
GRID = "#B8B8B8"

# category of each method (for the ranking colors / grouping)
CATEGORY = {
    "majority_vote": "unsupervised", "naive_ensemble": "unsupervised",
    "fuse": "unsupervised", "fuse_bin": "unsupervised", "fuse_ens": "unsupervised",
    "fuse_full": "unsupervised", "obv": "supervised", "weaver": "supervised",
    "logistic": "supervised", "gaussian_nb": "supervised", "oracle": "oracle",
}
CAT_COLOR = {"unsupervised": OI["blue"], "supervised": OI["orange"], "oracle": "#555555"}

# fixed display order: unsupervised -> supervised -> ceiling (used where methods
# must appear in a stable, category-grouped order rather than sorted by value)
UNSUP = ["majority_vote", "naive_ensemble", "fuse", "fuse_ens", "fuse_bin", "fuse_full"]
SUP = ["logistic", "gaussian_nb", "obv", "weaver"]
CEIL = ["oracle"]
METHOD_ORDER = UNSUP + SUP + CEIL

# focused set for the vs-lambda line plot: (method, color, linestyle, marker, lw)
LINE_SPEC = [
    ("fuse_ens",       OI["blue"],   "-",  "o", 2.8),
    ("fuse",           OI["sky"],    "-",  "s", 1.8),
    ("naive_ensemble", OI["green"],  "-",  "^", 1.8),
    ("majority_vote",  OI["orange"], "-",  "D", 1.8),
    ("weaver",         OI["purple"], "--", "v", 1.8),  # supervised -> dashed
    ("oracle",         "#444444",    "--", "x", 1.8),  # ceiling -> dashed gray
]

SOFT_BIN_SPEC = [
    ("fuse",      OI["blue"],       "-",  "o", 2.2, "soft"),
    ("fuse_ens",  OI["green"],      "-",  "s", 2.2, "soft"),
    ("fuse_bin",  OI["vermillion"], "--", "v", 1.8, "binary"),
    ("fuse_full", OI["purple"],     "--", "D", 1.8, "binary"),
]


def _style(ax):
    ax.grid(True, color=GRID, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _pf(res, lam, method, s, metric):
    """(mean, std) of a per-fold metric."""
    node = res[lam]["methods"][method]["per_fold"][s][metric]
    return node["mean"], node["std"]


# --------------------------------------------------------------------------- #
def plot_vs_lambda(res, lams_f, out):
    lams = [str(l) for l in lams_f]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    handles = None
    for ax, s, title in zip(axes, ("test", "unlabeled"),
                            ("Held-out test", "Unlabeled (out-of-fold)")):
        for method, color, ls, mk, lw in LINE_SPEC:
            means = [_pf(res, l, method, s, "balanced_acc")[0] for l in lams]
            stds = [_pf(res, l, method, s, "balanced_acc")[1] for l in lams]
            ax.errorbar(lams_f, means, yerr=stds, color=color, ls=ls, marker=mk,
                        lw=lw, markersize=6, capsize=2.5, elinewidth=1,
                        markeredgecolor="white", markeredgewidth=0.6, label=method,
                        zorder=3 if method == "fuse_ens" else 2)
        _style(ax)
        ax.set_xlabel(r"TCI weight  $\lambda$")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(lams_f)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Balanced accuracy  (fold mean $\\pm$ std)")
    # one shared legend below both panels (6 series -> legend beats colliding end-labels)
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("FUSE cross-validation: balanced accuracy vs TCI weight",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fp = os.path.join(out, "cv_balanced_acc_vs_lambda.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    return fp


def plot_ranking(res, rank_lam, out):
    lam = str(rank_lam)
    methods = list(res[lam]["methods"].keys())
    rows = []
    for m in methods:
        mean, std = _pf(res, lam, m, "test", "balanced_acc")
        rows.append((m, mean, std, CATEGORY.get(m, "unsupervised")))
    rows.sort(key=lambda r: r[1])  # ascending -> best on top of horizontal axis
    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    colors = [CAT_COLOR[r[3]] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    y = range(len(names))
    ax.errorbar(means, y, xerr=stds, fmt="none", ecolor="#999999",
                elinewidth=1.2, capsize=3, zorder=1)
    ax.scatter(means, y, s=70, c=colors, zorder=3,
               edgecolor="white", linewidth=0.8)
    for yi, (m, mean) in enumerate(zip(names, means)):
        ax.annotate(f"{mean:.3f}", (mean, yi), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#333333")
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Test balanced accuracy  (fold mean $\\pm$ std)")
    ax.set_title(f"Method ranking at $\\lambda$={rank_lam} (no TCI)"
                 if float(rank_lam) == 0 else f"Method ranking at $\\lambda$={rank_lam}",
                 fontsize=12, fontweight="bold")
    _style(ax); ax.grid(axis="y", visible=False)
    # legend by category
    handles = [plt.Line2D([0], [0], marker="o", ls="", markersize=8, color=c,
                          markeredgecolor="white", label=cat)
               for cat, c in CAT_COLOR.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9,
              title="category")
    fig.tight_layout()
    fp = os.path.join(out, "cv_ranking_test.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    return fp


def plot_soft_vs_binary(res, lams_f, out):
    lams = [str(l) for l in lams_f]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for method, color, ls, mk, lw, grp in SOFT_BIN_SPEC:
        means = [_pf(res, l, method, "test", "balanced_acc")[0] for l in lams]
        stds = [_pf(res, l, method, "test", "balanced_acc")[1] for l in lams]
        ax.errorbar(lams_f, means, yerr=stds, color=color, ls=ls, marker=mk,
                    lw=lw, markersize=6, capsize=2.5, elinewidth=1,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=f"{method} ({grp})")
        ax.annotate(method, (lams_f[-1], means[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=color)
    _style(ax)
    ax.set_xlabel(r"TCI weight  $\lambda$")
    ax.set_ylabel("Test balanced accuracy  (fold mean $\\pm$ std)")
    ax.set_title("Soft vs binary FUSE variants (soft dominates)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(lams_f)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fp = os.path.join(out, "cv_soft_vs_binary.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    return fp


def plot_metric_grid(res, rank_lam, out):
    lam = str(rank_lam)
    metrics = [("balanced_acc", "Balanced accuracy"), ("f1", "F1"),
               ("mcc", "MCC"), ("auc", "AUC")]
    methods = list(res[lam]["methods"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        rows = []
        for m in methods:
            mean, std = _pf(res, lam, m, "test", metric)
            rows.append((m, mean, std, CATEGORY.get(m, "unsupervised")))
        rows.sort(key=lambda r: r[1])
        names = [r[0] for r in rows]
        means = [r[1] for r in rows]
        stds = [r[2] for r in rows]
        colors = [CAT_COLOR[r[3]] for r in rows]
        y = range(len(names))
        ax.errorbar(means, y, xerr=stds, fmt="none", ecolor="#999999",
                    elinewidth=1.0, capsize=2.5, zorder=1)
        ax.scatter(means, y, s=48, c=colors, zorder=3,
                   edgecolor="white", linewidth=0.7)
        ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        _style(ax); ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([0], [0], marker="o", ls="", markersize=8, color=c,
                          markeredgecolor="white", label=cat)
               for cat, c in CAT_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(f"Test-set metrics by method at $\\lambda$={rank_lam}",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fp = os.path.join(out, "cv_metric_grid.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    return fp


def plot_metric_grid_2x3(res, out, lambdas=("0.0", "0.1", "1.0"),
                         metrics=(("balanced_acc", "Balanced accuracy"), ("auc", "AUC"))):
    """2 rows (metric) x 3 cols (lambda) of dot plots; methods on the y-axis in a
    fixed unsupervised->supervised order (NOT sorted by value)."""
    lambdas = [l if l in res else str(float(l)) for l in lambdas]
    order = [m for m in METHOD_ORDER if m in res[lambdas[0]]["methods"]]
    yorder = list(reversed(order))          # first method (majority_vote) at TOP
    y = range(len(yorder))
    nrow, ncol = len(metrics), len(lambdas)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 0.42 * len(yorder) + 1.6),
                             sharey=True)
    for r, (metric, mlabel) in enumerate(metrics):
        # common x-range across the row so the 3 lambdas are directly comparable
        ms = [_pf(res, lam, m, "test", metric) for lam in lambdas for m in yorder]
        lo = min(mean - std for mean, std in ms)
        hi = max(mean + std for mean, std in ms)
        pad = 0.03 * (hi - lo + 1e-9)
        for c, lam in enumerate(lambdas):
            ax = axes[r][c]
            means = [_pf(res, lam, m, "test", metric)[0] for m in yorder]
            stds = [_pf(res, lam, m, "test", metric)[1] for m in yorder]
            colors = [CAT_COLOR[CATEGORY.get(m, "unsupervised")] for m in yorder]
            ax.errorbar(means, y, xerr=stds, fmt="none", ecolor="#999999",
                        elinewidth=1.0, capsize=2.5, zorder=1)
            ax.scatter(means, y, s=46, c=colors, zorder=3,
                       edgecolor="white", linewidth=0.7)
            ax.set_xlim(lo - pad, hi + pad)
            _style(ax); ax.grid(axis="y", visible=False)
            if r == 0:
                ax.set_title(f"$\\lambda$ = {float(lam):g}", fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_yticks(list(y)); ax.set_yticklabels(yorder, fontsize=8.5)
                ax.set_ylabel(mlabel, fontsize=11, fontweight="bold")
            if r == nrow - 1:
                ax.set_xlabel("test score", fontsize=9)
    handles = [plt.Line2D([0], [0], marker="o", ls="", markersize=8, color=c,
                          markeredgecolor="white", label=cat)
               for cat, c in CAT_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Test balanced accuracy & AUC by method across TCI weight",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fp = os.path.join(out, "cv_metric_grid_bacc_auc.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fp


def plot_vs_lambda_means(res, lams_f, out):
    """Section-4 request: same as the vs-lambda figure but MEANS ONLY (the
    overlapping std whiskers were uninformative)."""
    lams = [str(l) for l in lams_f]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    handles = None
    for ax, s, title in zip(axes, ("test", "unlabeled"),
                            ("Held-out test", "Unlabeled (out-of-fold)")):
        for method, color, ls, mk, lw in LINE_SPEC:
            means = [_pf(res, l, method, s, "balanced_acc")[0] for l in lams]
            ax.plot(lams_f, means, color=color, ls=ls, marker=mk, lw=lw, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.6, label=method,
                    zorder=3 if method == "fuse_ens" else 2)
        _style(ax)
        ax.set_xlabel(r"TCI weight  $\lambda$"); ax.set_title(title, fontsize=11)
        ax.set_xticks(lams_f)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Balanced accuracy  (fold mean)")
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("FUSE cross-validation: balanced accuracy vs TCI weight (means)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fp = os.path.join(out, "cv_balanced_acc_vs_lambda_means.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    return fp


def plot_vs_lambda_dodge(res, lams_f, out):
    """Aesthetic alternative that keeps the std but avoids overlap: lambda on an
    evenly-spaced ordinal axis, each method's error bars dodged horizontally so
    the whiskers no longer sit on top of one another."""
    lams = [str(l) for l in lams_f]
    xpos = list(range(len(lams)))
    nm = len(LINE_SPEC)
    span = 0.66
    offs = [(i - (nm - 1) / 2) * (span / nm) for i in range(nm)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    handles = None
    for ax, s, title in zip(axes, ("test", "unlabeled"),
                            ("Held-out test", "Unlabeled (out-of-fold)")):
        for (method, color, ls, mk, lw), off in zip(LINE_SPEC, offs):
            means = [_pf(res, l, method, s, "balanced_acc")[0] for l in lams]
            stds = [_pf(res, l, method, s, "balanced_acc")[1] for l in lams]
            xx = [p + off for p in xpos]
            ax.errorbar(xx, means, yerr=stds, color=color, ls=ls, marker=mk, lw=lw,
                        markersize=5.5, capsize=2, elinewidth=0.9,
                        markeredgecolor="white", markeredgewidth=0.5, label=method,
                        zorder=3 if method == "fuse_ens" else 2)
        _style(ax)
        ax.set_xlabel(r"TCI weight  $\lambda$  (evenly spaced)")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xpos); ax.set_xticklabels([f"{float(l):g}" for l in lams])
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Balanced accuracy  (fold mean $\\pm$ std, dodged)")
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("FUSE cross-validation: balanced accuracy vs TCI weight (dodged std)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fp = os.path.join(out, "cv_balanced_acc_vs_lambda_dodged.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/reports/cv/cv_results.json")
    ap.add_argument("--rank-lambda", default="0.0",
                    help="lambda used for the ranking + metric-grid figures")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    res = data["results"]
    lams_f = [float(l) for l in data["lambdas"]]
    out = os.path.dirname(args.results)

    # normalize the rank lambda to the key form actually present
    rank_lam = args.rank_lambda
    if rank_lam not in res:
        rank_lam = str(float(rank_lam))
    if rank_lam not in res:
        raise SystemExit(f"lambda {args.rank_lambda} not in results {list(res)}")

    written = [
        plot_vs_lambda(res, lams_f, out),
        plot_ranking(res, rank_lam, out),
        plot_soft_vs_binary(res, lams_f, out),
        plot_metric_grid(res, rank_lam, out),
        plot_metric_grid_2x3(res, out),
        plot_vs_lambda_means(res, lams_f, out),
        plot_vs_lambda_dodge(res, lams_f, out),
    ]
    print("wrote:")
    for fp in written:
        print("  ", fp)


if __name__ == "__main__":
    main()
