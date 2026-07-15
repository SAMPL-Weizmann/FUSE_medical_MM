"""Compare the FOUR CV setups by lambda and method.

Setups = {10-fold, 20-fold} x {2 answers (NCL, 32 verifiers), 1 answer (16)}:
    cv10-2ans  artifacts/reports/cv/cv_results.json       (n_answers=2)
    cv20-2ans  artifacts/reports/cv20/cv_results.json      (n_answers=2)
    cv10-1ans  artifacts/reports/cv_1ans/cv_results.json   (n_answers=1)
    cv20-1ans  artifacts/reports/cv20_1ans/cv_results.json (n_answers=1)

The 1-answer runs halve the verifier count (one head output instead of an NCL
twin pair). This script asks: does that change the accuracy or the lambda story,
per method? Reads the merged cv_results.json of each setup (12_cv -> 13_cv_merge)
and writes, into artifacts/reports/cv_compare/:

  - cmp_bacc_vs_lambda_by_method.png   small multiples, one panel per method;
                                       all 4 setups overlaid vs lambda (test bAcc)
  - cmp_delta_1ans_vs_2ans.png         two heatmaps (10-/20-fold), method x lambda,
                                       Delta = (1ans - 2ans) test bAcc, in points
  - cmp_headline_fuse_ens.png          fuse_ens alone, 4 setups vs lambda (big)
  - cmp_table_lambda0.{png,csv}        method x setup table at lambda=0 (mean+/-std %)

Visual encoding: answers -> color (2ans blue / 1ans vermillion); folds ->
linestyle+marker (10 solid o / 20 dashed s). Okabe-Ito, colorblind-safe.

Usage:
    python scripts/19_cv_compare.py            # default paths above
    python scripts/19_cv_compare.py --metric auc --rank-lambda 0.0
    # override any setup path: --cv10-2ans PATH --cv20-1ans PATH ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

# --- Okabe-Ito colorblind-safe palette ------------------------------------- #
OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "purple": "#CC79A7", "black": "#000000"}
GRID = "#B8B8B8"

# setup key -> (label, n_answers, n_folds, color, linestyle, marker)
STYLE = {
    "cv10-2ans": ("10-fold, 2 answers (32 verif.)", 2, 10, OI["blue"],       "-",  "o"),
    "cv20-2ans": ("20-fold, 2 answers (32 verif.)", 2, 20, OI["blue"],       "--", "s"),
    "cv10-1ans": ("10-fold, 1 answer (16 verif.)",  1, 10, OI["vermillion"], "-",  "o"),
    "cv20-1ans": ("20-fold, 1 answer (16 verif.)",  1, 20, OI["vermillion"], "--", "s"),
}
SETUP_ORDER = ["cv10-2ans", "cv20-2ans", "cv10-1ans", "cv20-1ans"]

# stable method display order (unsupervised -> supervised -> ceiling)
METHOD_ORDER = ["majority_vote", "naive_ensemble", "fuse", "fuse_ens", "fuse_bin",
                "fuse_full", "logistic", "gaussian_nb", "obv", "weaver", "oracle"]


def _style(ax):
    ax.grid(True, color=GRID, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _load(paths: dict) -> dict:
    """key -> {"res": {...}, "lams": [str,...]} for every setup file that exists."""
    out = {}
    for key, path in paths.items():
        if not path or not os.path.isfile(path):
            print(f"  [skip] {key}: {path} not found")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out[key] = {"res": data["results"],
                    "lams": [str(float(l)) for l in data["lambdas"]]}
    return out


def _get(data, key, lam, method, metric, which="mean"):
    """Return the per-fold test metric (mean/std) or None if absent."""
    res = data[key]["res"]
    if lam not in res or method not in res[lam]["methods"]:
        return None
    node = res[lam]["methods"][method]["per_fold"]["test"].get(metric)
    return None if node is None else node[which]


def _common(data, attr):
    """Methods (or lambdas) present in EVERY loaded setup, in canonical order."""
    if attr == "methods":
        sets = [set(d["res"][next(iter(d["res"]))]["methods"]) for d in data.values()]
        common = set.intersection(*sets) if sets else set()
        return [m for m in METHOD_ORDER if m in common]
    sets = [set(d["lams"]) for d in data.values()]
    common = set.intersection(*sets) if sets else set()
    return sorted(common, key=float)


# --------------------------------------------------------------------------- #
def plot_by_method(data, methods, lams, metric, out):
    lamf = [float(l) for l in lams]
    ncol = 4
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow),
                             sharex=True, squeeze=False)
    handles = labels = None
    for idx, method in enumerate(methods):
        ax = axes[idx // ncol][idx % ncol]
        for key in SETUP_ORDER:
            if key not in data:
                continue
            lbl, _, _, color, ls, mk = STYLE[key]
            means = [_get(data, key, l, method, metric) for l in lams]
            if any(m is None for m in means):
                continue
            ax.plot(lamf, means, color=color, ls=ls, marker=mk, lw=1.9,
                    markersize=5, markeredgecolor="white", markeredgewidth=0.5,
                    label=lbl)
        _style(ax)
        ax.set_title(method, fontsize=10, fontweight="bold")
        ax.set_xticks(lamf)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    for j in range(len(methods), nrow * ncol):        # blank unused panels
        axes[j // ncol][j % ncol].axis("off")
    for r in range(nrow):
        axes[r][0].set_ylabel(f"test {metric}")
    for c in range(ncol):
        axes[nrow - 1][c].set_xlabel(r"TCI weight  $\lambda$")
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                   fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"CV comparison: test {metric} vs $\\lambda$ "
                 f"(1 vs 2 answers, 10 vs 20 folds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fp = os.path.join(out, f"cmp_{metric_slug(metric)}_vs_lambda_by_method.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fp


def plot_delta_heatmaps(data, methods, lams, metric, out):
    """(1ans - 2ans) per method x lambda, one heatmap per fold count."""
    pairs = [(10, "cv10-1ans", "cv10-2ans"), (20, "cv20-1ans", "cv20-2ans")]
    pairs = [(n, a, b) for n, a, b in pairs if a in data and b in data]
    if not pairs:
        print("  [skip] delta heatmaps: need both 1ans and 2ans for a fold count")
        return None
    lamf = [float(l) for l in lams]
    fig, axes = plt.subplots(1, len(pairs), figsize=(1.1 * len(lams) * len(pairs) + 2,
                                                     0.42 * len(methods) + 1.6),
                             squeeze=False)
    # symmetric color scale across both panels (in points)
    alld = []
    for _, a, b in pairs:
        for m in methods:
            for l in lams:
                va, vb = _get(data, a, l, m, metric), _get(data, b, l, m, metric)
                if va is not None and vb is not None:
                    alld.append((va - vb) * 100)
    vmax = max(1e-6, max(abs(x) for x in alld)) if alld else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for ax, (n, a, b) in zip(axes[0], pairs):
        D = np.full((len(methods), len(lams)), np.nan)
        for i, m in enumerate(methods):
            for j, l in enumerate(lams):
                va, vb = _get(data, a, l, m, metric), _get(data, b, l, m, metric)
                if va is not None and vb is not None:
                    D[i, j] = (va - vb) * 100
        im = ax.imshow(D, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{v:g}" for v in lamf])
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=8.5)
        ax.set_xlabel(r"TCI weight  $\lambda$")
        ax.set_title(f"{n}-fold: (1 answer $-$ 2 answers)", fontsize=11, fontweight="bold")
        for i in range(len(methods)):
            for j in range(len(lams)):
                if not np.isnan(D[i, j]):
                    ax.text(j, i, f"{D[i, j]:+.1f}", ha="center", va="center",
                            fontsize=7, color="#111111")
    cbar = fig.colorbar(im, ax=list(axes[0]), fraction=0.03, pad=0.02)
    cbar.set_label(f"$\\Delta$ test {metric}  (points; red = 1ans better)", fontsize=9)
    fig.suptitle(f"Effect of halving the verifiers on test {metric}",
                 fontsize=13, fontweight="bold")
    fp = os.path.join(out, f"cmp_delta_1ans_vs_2ans_{metric_slug(metric)}.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
    return fp


def plot_headline(data, lams, metric, out, method="fuse_ens"):
    lamf = [float(l) for l in lams]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for key in SETUP_ORDER:
        if key not in data:
            continue
        lbl, _, _, color, ls, mk = STYLE[key]
        means = [_get(data, key, l, method, metric) for l in lams]
        stds = [_get(data, key, l, method, metric, "std") for l in lams]
        if any(m is None for m in means):
            continue
        ax.errorbar(lamf, means, yerr=stds, color=color, ls=ls, marker=mk, lw=2.2,
                    markersize=6.5, capsize=2.5, elinewidth=1,
                    markeredgecolor="white", markeredgewidth=0.6, label=lbl)
    _style(ax)
    ax.set_xlabel(r"TCI weight  $\lambda$")
    ax.set_ylabel(f"test {metric}  (fold mean $\\pm$ std)")
    ax.set_xticks(lamf)
    ax.set_title(f"{method}: 1 vs 2 answers, 10 vs 20 folds", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="best")
    fig.tight_layout()
    fp = os.path.join(out, f"cmp_headline_{method}_{metric_slug(metric)}.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    return fp


def write_table(data, methods, metric, rank_lam, out):
    """method x setup table at a fixed lambda (mean +/- std, as %). PNG + CSV."""
    lam = rank_lam
    keys = [k for k in SETUP_ORDER if k in data]
    rows = []
    for m in methods:
        cells = []
        for k in keys:
            mu = _get(data, k, lam, m, metric)
            sd = _get(data, k, lam, m, metric, "std")
            cells.append(None if mu is None else (mu * 100, sd * 100))
        rows.append((m, cells))

    # CSV
    csv_fp = os.path.join(out, f"cmp_table_lambda{lam}_{metric_slug(metric)}.csv")
    with open(csv_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"{k}_mean" for k in keys] + [f"{k}_std" for k in keys])
        for m, cells in rows:
            means = [f"{c[0]:.2f}" if c else "" for c in cells]
            stds = [f"{c[1]:.2f}" if c else "" for c in cells]
            w.writerow([m] + means + stds)

    # PNG table
    fig, ax = plt.subplots(figsize=(1.9 * len(keys) + 2.4, 0.42 * len(rows) + 1.2))
    ax.axis("off")
    col_labels = ["method"] + [STYLE[k][0].split(",")[0] + "\n" + STYLE[k][0].split("(")[1].rstrip(")")
                               for k in keys]
    # per-column max (bold the winner in each setup column)
    best = []
    for ci in range(len(keys)):
        vals = [(ri, rows[ri][1][ci][0]) for ri in range(len(rows)) if rows[ri][1][ci]]
        best.append(max(vals, key=lambda t: t[1])[0] if vals else -1)
    table_rows = []
    for m, cells in rows:
        table_rows.append([m] + [f"{c[0]:.1f}±{c[1]:.1f}" if c else "-" for c in cells])
    tbl = ax.table(cellText=table_rows, colLabels=col_labels, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.35)
    for ci in range(len(keys)):
        if best[ci] >= 0:
            cell = tbl[best[ci] + 1, ci + 1]
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#EAF3FA")
    for ci in range(len(col_labels)):
        tbl[0, ci].set_text_props(fontweight="bold")
        tbl[0, ci].set_facecolor("#DDDDDD")
    ax.set_title(f"Test {metric} at $\\lambda$={lam}  (mean $\\pm$ std %, column-max bold)",
                 fontsize=11, fontweight="bold", pad=12)
    png_fp = os.path.join(out, f"cmp_table_lambda{lam}_{metric_slug(metric)}.png")
    fig.savefig(png_fp, dpi=150, bbox_inches="tight"); plt.close(fig)
    return png_fp, csv_fp


def metric_slug(metric):
    return {"balanced_acc": "bacc"}.get(metric, metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv10-2ans", default="artifacts/reports/cv/cv_results.json")
    ap.add_argument("--cv20-2ans", default="artifacts/reports/cv20/cv_results.json")
    ap.add_argument("--cv10-1ans", default="artifacts/reports/cv_1ans/cv_results.json")
    ap.add_argument("--cv20-1ans", default="artifacts/reports/cv20_1ans/cv_results.json")
    ap.add_argument("--metric", default="balanced_acc")
    ap.add_argument("--rank-lambda", default="0.0", help="lambda for the summary table")
    ap.add_argument("--out-dir", default="artifacts/reports/cv_compare")
    args = ap.parse_args()

    paths = {"cv10-2ans": args.cv10_2ans, "cv20-2ans": args.cv20_2ans,
             "cv10-1ans": args.cv10_1ans, "cv20-1ans": args.cv20_1ans}
    print("loading setups:")
    data = _load(paths)
    if len(data) < 2:
        raise SystemExit("need at least 2 setups present to compare")

    methods = _common(data, "methods")
    lams = _common(data, "lambdas")
    if not methods or not lams:
        raise SystemExit(f"no common methods/lambdas across setups "
                         f"(methods={methods}, lambdas={lams})")
    rank_lam = args.rank_lambda if args.rank_lambda in lams else str(float(args.rank_lambda))
    if rank_lam not in lams:
        rank_lam = lams[0]
    print(f"  setups={list(data)}\n  methods={methods}\n  lambdas={lams}\n  table@lambda={rank_lam}")

    os.makedirs(args.out_dir, exist_ok=True)
    written = [
        plot_by_method(data, methods, lams, args.metric, args.out_dir),
        plot_delta_heatmaps(data, methods, lams, args.metric, args.out_dir),
        plot_headline(data, lams, args.metric, args.out_dir),
        *write_table(data, methods, args.metric, rank_lam, args.out_dir),
    ]
    print("wrote:")
    for fp in written:
        if fp:
            print("  ", fp)


if __name__ == "__main__":
    main()
