"""ROC curves + precision/recall-vs-threshold table for the best CV methods.

Reads the pooled out-of-fold predictions (cv_pooled_predictions.npz, produced by
12_cv.py --dump-predictions) and the merged results (cv_results.json). By default
it plots the SAME methods as cv_table_top3_merged (union of top-3-by-bAcc and
top-3-by-AUC, each at its best lambda), so the ROC matches that table.

Outputs (into the results dir, or --out-dir):
  roc_<tag>.png                 ROC (TPR vs FPR) per method, AUC in the legend,
                                dots marking a few decision thresholds on each curve
  pr_table_<tag>.{png,csv}      recall & precision at a grid of thresholds, per method

A ROC curve already sweeps every threshold (each point = one cut); the marked dots
show where specific thresholds land, and AUC is the threshold-free area.

Usage:
    python scripts/20_roc_pr.py \
        --results     artifacts/reports/malig_cv_1ans/cv_results.json \
        --predictions artifacts/reports/malig_cv_1ans/cv_pooled_predictions.npz
    # overrides: --methods fuse fuse_ens oracle --lambda 0.0
    #            --thresholds 0.05 0.1 0.2 0.3 0.5 0.7 0.9  --tag malig_1ans
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuse_mm.heads.metrics import roc_auc  # noqa: E402

OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#000000"]
MARK = "os^Dv<>"                                    # per-curve marker glyphs
METHOD_ORDER = ["majority_vote", "naive_ensemble", "fuse", "fuse_ens", "fuse_bin",
                "fuse_full", "logistic", "gaussian_nb", "obv", "weaver", "oracle"]


def _best_lambda_per_method(res, lambdas, metric):
    best = {}
    for m in METHOD_ORDER:
        if m not in res[lambdas[0]]["methods"]:
            continue
        best[m] = max(((lam, res[lam]["methods"][m]["per_fold"]["test"][metric]["mean"])
                       for lam in lambdas), key=lambda t: t[1])
    return best


def merged_rows(res):
    """Reproduce cv_table_top3_merged's (method, lambda) rows."""
    lambdas = sorted(res, key=float)
    bacc = _best_lambda_per_method(res, lambdas, "balanced_acc")
    auc = _best_lambda_per_method(res, lambdas, "auc")
    top_b = [m for m, _ in sorted(bacc.items(), key=lambda kv: -kv[1][1])[:3]]
    top_a = [m for m, _ in sorted(auc.items(), key=lambda kv: -kv[1][1])[:3]]
    rows, seen = [], set()
    for m in top_b:
        k = (m, bacc[m][0])
        if k not in seen:
            seen.add(k); rows.append(k)
    for m in top_a:
        k = (m, auc[m][0])
        if k not in seen:
            seen.add(k); rows.append(k)
    return rows                                     # [(method, lambda_str), ...]


def roc_curve(y, score):
    """Return (fpr, tpr) stepped over all thresholds, with (0,0) and (1,1) anchored."""
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    P, N = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])
    return fpr, tpr


def _point_at(y, score, thr):
    pred = score >= thr
    P, N = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    tpr = np.count_nonzero(pred & (y == 1)) / P
    fpr = np.count_nonzero(pred & (y == 0)) / N
    return fpr, tpr


def _recall_precision(y, score, thr):
    pred = score >= thr
    tp = np.count_nonzero(pred & (y == 1))
    fp = np.count_nonzero(pred & (y == 0))
    fn = np.count_nonzero((~pred) & (y == 1))
    recall = tp / max(tp + fn, 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan   # undefined if nothing flagged
    return recall, precision


def _key(preds, lam, method):
    for k in (f"{lam}__{method}", f"{float(lam)}__{method}"):
        if k in preds:
            return k
    raise SystemExit(f"prediction '{lam}__{method}' not in npz "
                     f"(have e.g. {list(preds)[:4]}...). Re-run 12_cv with --dump-predictions.")


# --------------------------------------------------------------------------- #
def plot_roc(rows, preds, y, thresholds, out, tag):
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.plot([0, 1], [0, 1], ls=":", color="#999", lw=1, zorder=1)     # chance
    for i, (m, lam) in enumerate(rows):
        s = preds[_key(preds, lam, m)]
        fpr, tpr = roc_curve(y, s)
        auc = roc_auc(y, s)
        color = OI[i % len(OI)]
        ax.plot(fpr, tpr, color=color, lw=2.0, zorder=3,
                label=f"{m} ($\\lambda$={float(lam):g})  AUC={auc:.3f}")
        # mark where specific thresholds land
        pts = np.array([_point_at(y, s, t) for t in thresholds])
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=26, marker=MARK[i % len(MARK)],
                   edgecolor="white", linewidth=0.5, zorder=4)
        if i == 0:                                   # annotate thresholds on the first curve
            for (fx, ty), t in zip(pts, thresholds):
                ax.annotate(f"{t:g}", (fx, ty), textcoords="offset points",
                            xytext=(5, -6), fontsize=7, color=color)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate  (1 $-$ specificity)")
    ax.set_ylabel("True positive rate  (recall / sensitivity)")
    ax.set_title(f"ROC — malignant vs non-malignant ({tag})\npooled out-of-fold, "
                 "dots = decision thresholds", fontsize=12, fontweight="bold")
    ax.grid(True, color="#B8B8B8", alpha=0.35, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, title="method")
    fig.tight_layout()
    fp = os.path.join(out, f"roc_{tag}.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    return fp


def pr_table(rows, preds, y, thresholds, out, tag):
    # matrix[t][method] = (recall, precision)
    labels = [f"{m} (λ={float(lam):g})" for m, lam in rows]
    data = []
    for t in thresholds:
        cells = []
        for m, lam in rows:
            cells.append(_recall_precision(y, preds[_key(preds, lam, m)], t))
        data.append(cells)

    # CSV
    csv_fp = os.path.join(out, f"pr_table_{tag}.csv")
    with open(csv_fp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        head = ["threshold"]
        for lb in labels:
            head += [f"{lb} recall", f"{lb} precision"]
        w.writerow(head)
        for t, cells in zip(thresholds, data):
            row = [f"{t:g}"]
            for rec, prec in cells:
                row += [f"{rec:.3f}", "" if np.isnan(prec) else f"{prec:.3f}"]
            w.writerow(row)

    # PNG table: rows = thresholds, grouped columns (recall|precision) per method
    nM = len(rows)
    fig_w = 1.2 + 1.7 * nM
    fig_h = 1.1 + 0.34 * len(thresholds)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.axis("off")
    ncol = 1 + 2 * nM
    W, H = ncol, len(thresholds) + 2
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.add_patch(Rectangle((0, 0), W, 2, facecolor="#e9edf2", edgecolor="none"))
    for i in range(len(thresholds)):
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, 2 + i), W, 1, facecolor="#f4f4f6", edgecolor="none"))
    ax.text(0.5, 1.0, "thr", ha="center", va="center", fontsize=10, fontweight="bold")
    for j, (m, lam) in enumerate(rows):
        c0 = 1 + 2 * j
        ax.text(c0 + 1.0, 0.5, f"{m}\n($\\lambda$={float(lam):g})", ha="center",
                va="center", fontsize=9, fontweight="bold", color=OI[j % len(OI)])
        ax.text(c0 + 0.5, 1.5, "recall", ha="center", va="center", fontsize=8, fontstyle="italic")
        ax.text(c0 + 1.5, 1.5, "prec", ha="center", va="center", fontsize=8, fontstyle="italic")
    for i, (t, cells) in enumerate(zip(thresholds, data)):
        yc = 2 + i + 0.5
        ax.text(0.5, yc, f"{t:g}", ha="center", va="center", fontsize=9, fontweight="bold")
        for j, (rec, prec) in enumerate(cells):
            c0 = 1 + 2 * j
            ax.text(c0 + 0.5, yc, f"{rec:.2f}", ha="center", va="center", fontsize=8.5)
            ax.text(c0 + 1.5, yc, "–" if np.isnan(prec) else f"{prec:.2f}",
                    ha="center", va="center", fontsize=8.5)
    for j in range(nM + 1):                           # separators between method groups
        x = 1 + 2 * j
        ax.plot([x, x], [0, H], color="#c4c4c4", lw=0.9)
    ax.plot([0, W], [2, 2], color="#555", lw=1.2)
    ax.set_title(f"Recall / precision vs threshold — {tag}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    png_fp = os.path.join(out, f"pr_table_{tag}.png")
    fig.savefig(png_fp, dpi=160, bbox_inches="tight"); plt.close(fig)
    return png_fp, csv_fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="cv_results.json of the setup")
    ap.add_argument("--predictions", default=None,
                    help="cv_pooled_predictions.npz (default: sibling of --results)")
    ap.add_argument("--out-dir", default=None, help="default: dir of --results")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="override method list (default: cv_table_top3_merged set)")
    ap.add_argument("--lambda", dest="lam", default=None,
                    help="single lambda for --methods (default: each method's best)")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--tag", default=None, help="filename tag (default from results dir)")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        res = json.load(f)["results"]
    out = args.out_dir or os.path.dirname(args.results)
    os.makedirs(out, exist_ok=True)
    npz = args.predictions or os.path.join(os.path.dirname(args.results),
                                           "cv_pooled_predictions.npz")
    if not os.path.isfile(npz):
        raise SystemExit(f"predictions not found: {npz}\n"
                         "Re-run 12_cv.py with --dump-predictions first.")
    preds = dict(np.load(npz))
    y = preds.pop("y").astype(int)
    tag = args.tag or os.path.basename(os.path.dirname(args.results)) or "cv"

    if args.methods:
        lams = sorted(res, key=float)
        rows = [(m, args.lam or _best_lambda_per_method(res, lams, "auc")[m][0])
                for m in args.methods]
    else:
        rows = merged_rows(res)

    print(f"methods (method, lambda): {rows}")
    print(f"pooled OOF patients: {len(y)}  positives: {int((y==1).sum())} "
          f"({100*y.mean():.1f}%)")
    roc_fp = plot_roc(rows, preds, y, args.thresholds, out, tag)
    png_fp, csv_fp = pr_table(rows, preds, y, args.thresholds, out, tag)
    print("wrote:")
    for fp in (roc_fp, png_fp, csv_fp):
        print("  ", fp)


if __name__ == "__main__":
    main()
