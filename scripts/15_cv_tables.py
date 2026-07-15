"""Summary tables from the merged cross-validation results.

Reads artifacts/reports/cv/cv_results.json and writes to the same dir:

  Section 2 -- all methods at a single lambda (default 0):
    cv_table_lambda0.png / .csv
      rows = methods (fixed unsupervised -> supervised -> ceiling order),
      columns = metric (bAcc, acc, AUC, recall, precision, F1), each split into
      two sub-columns S_U (unlabeled/OOF) and test. Values are fold mean +/- std
      shown as PERCENTAGES.

  Section 3 -- top-3 methods across ALL (method, lambda) combinations, chosen once
  by test balanced accuracy and once by test AUC:
    cv_table_top3_by_bacc.png / .csv
    cv_table_top3_by_auc.png  / .csv
      same column structure, plus a "lambda" column giving the lambda each method
      was selected at (each method contributes its single best lambda).

Usage:
    python scripts/15_cv_tables.py [--results artifacts/reports/cv/cv_results.json]
                                   [--lambda0 0.0]
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# fixed display order + category colors (kept in sync with 14_cv_plots.py)
UNSUP = ["majority_vote", "naive_ensemble", "fuse", "fuse_ens", "fuse_bin", "fuse_full"]
SUP = ["logistic", "gaussian_nb", "obv", "weaver"]
CEIL = ["oracle"]
METHOD_ORDER = UNSUP + SUP + CEIL
CATEGORY = {m: "unsupervised" for m in UNSUP}
CATEGORY.update({m: "supervised" for m in SUP})
CATEGORY.update({m: "oracle" for m in CEIL})
CAT_COLOR = {"unsupervised": "#0072B2", "supervised": "#B8860B", "oracle": "#555555"}

# (json key, display label)
METRICS = [("balanced_acc", "bAcc"), ("acc", "acc"), ("auc", "AUC"),
           ("recall", "recall"), ("precision", "precision"), ("f1", "F1")]
SETS = [("unlabeled", "S_U"), ("test", "test")]


def _pf(res, lam, method, s, metric):
    node = res[lam]["methods"][method]["per_fold"][s][metric]
    return node["mean"], node["std"]


def _fmt(mean, std):
    """Percentage mean +/- std, e.g. 91.0±2.6 (no % sign -- it lives in the title)."""
    return f"{100 * mean:.1f}±{100 * std:.1f}"


def _row_cells(res, lam, method):
    """[[S_U, test], ...] formatted strings, one pair per metric."""
    return [[_fmt(*_pf(res, lam, method, s, mkey)) for s, _ in SETS]
            for mkey, _ in METRICS]


# --------------------------------------------------------------------------- #
def render_table(path, title, rows, lead_specs, bold=frozenset(), n_folds=10):
    """rows: list of dicts {lead:[str,...], cells:[[su,test],...], color:hex}.
    lead_specs: list of (header, rel_width).
    bold: set of (row, metric, set) index triples to render bold (column-best)."""
    n_lead = len(lead_specs)
    n_metric = len(METRICS)
    widths = [w for _, w in lead_specs] + [1.0] * (n_metric * 2)
    edges = [0.0]
    for w in widths:
        edges.append(edges[-1] + w)
    W = edges[-1]
    n_rows = len(rows)
    H = 2.0 + n_rows  # 2 header rows + data rows

    fig_w = W * 0.62 + 0.4
    fig_h = H * 0.36 + 0.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

    # backgrounds: header band + zebra striping
    ax.add_patch(Rectangle((0, 0), W, 2, facecolor="#e9edf2", edgecolor="none", zorder=0))
    for i in range(n_rows):
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, 2 + i), W, 1, facecolor="#f4f4f6",
                                   edgecolor="none", zorder=0))

    def col_center(k):
        return (edges[k] + edges[k + 1]) / 2

    # lead headers (centered across both header rows)
    for j, (hdr, _) in enumerate(lead_specs):
        ax.text(col_center(j), 1.0, hdr, ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=3)

    # metric group labels + underline; sub-headers
    for g, (_, mlabel) in enumerate(METRICS):
        c0 = n_lead + 2 * g
        cxl, cxr = edges[c0], edges[c0 + 2]
        ax.text((cxl + cxr) / 2, 0.5, mlabel, ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=3)
        ax.plot([cxl + 0.12, cxr - 0.12], [0.92, 0.92], color="#8a8a8a", lw=0.8, zorder=3)
        for s, (_, slabel) in enumerate(SETS):
            ax.text(col_center(c0 + s), 1.5, slabel, ha="center", va="center",
                    fontsize=9, fontstyle="italic", color="#333", zorder=3)

    # data rows
    for i, row in enumerate(rows):
        yc = 2 + i + 0.5
        for j, val in enumerate(row["lead"]):
            if j == 0:
                ax.text(edges[0] + 0.12, yc, val, ha="left", va="center",
                        fontsize=9.5, fontweight="bold",
                        color=row.get("color", "#222"), zorder=3)
            else:
                ax.text(col_center(j), yc, val, ha="center", va="center",
                        fontsize=9.5, color="#222", zorder=3)
        for g in range(n_metric):
            c0 = n_lead + 2 * g
            for s in range(2):
                is_b = (i, g, s) in bold
                ax.text(col_center(c0 + s), yc, row["cells"][g][s],
                        ha="center", va="center", fontsize=9.0 if is_b else 8.7,
                        fontweight="bold" if is_b else "normal",
                        color="#000" if is_b else "#222", zorder=3)

    # gridlines
    for j in range(1, n_lead + 1):                      # after each lead col
        ax.plot([edges[j]] * 2, [0, H], color="#c4c4c4", lw=0.8, zorder=2)
    for g in range(n_metric + 1):                        # between metric groups
        x = edges[n_lead + 2 * g]
        ax.plot([x, x], [0, H], color="#c4c4c4", lw=0.9, zorder=2)
    for g in range(n_metric):                            # faint split within a metric
        x = edges[n_lead + 2 * g + 1]
        ax.plot([x, x], [2, H], color="#e2e2e2", lw=0.6, zorder=2)
    ax.plot([0, W], [2, 2], color="#555", lw=1.3, zorder=2)   # header/body divider
    for yy in (0, H):
        ax.plot([0, W], [yy, yy], color="#555", lw=1.2, zorder=2)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    note = (f"values are fold mean ± std over {n_folds} CV folds, in %   ·   "
            "S_U = unlabeled (out-of-fold)")
    if bold:
        note += "   ·   bold = best in column"
    fig.text(0.5, 0.008, note, ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows, lead_headers):
    header = list(lead_headers)
    for _, mlabel in METRICS:
        for _, slabel in SETS:
            header.append(f"{mlabel}_{slabel} (%)")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            flat = list(row["lead"])
            for pair in row["cells"]:
                flat.extend(pair)
            w.writerow(flat)


# --------------------------------------------------------------------------- #
def table_single_lambda(res, lam, out, n_folds=10):
    methods = [m for m in METHOD_ORDER if m in res[lam]["methods"]]
    rows = [{"lead": [m], "cells": _row_cells(res, lam, m),
             "color": CAT_COLOR[CATEGORY[m]]} for m in methods]
    title = f"All methods at $\\lambda$ = {float(lam):g}  (%)"
    render_table(os.path.join(out, "cv_table_lambda0.png"), title, rows,
                 lead_specs=[("Method", 2.6)], n_folds=n_folds)
    write_csv(os.path.join(out, "cv_table_lambda0.csv"), rows, ["method"])


def _best_lambda_per_method(res, lambdas, rank_metric):
    """method -> (best_lambda, best_test_mean) maximizing test rank_metric."""
    best = {}
    for m in METHOD_ORDER:
        if m not in res[lambdas[0]]["methods"]:
            continue
        cand = [(lam, _pf(res, lam, m, "test", rank_metric)[0]) for lam in lambdas]
        best[m] = max(cand, key=lambda t: t[1])
    return best


def table_top3(res, lambdas, rank_metric, rank_label, fname, out, n_folds=10):
    """Per method, pick the lambda maximizing the test rank_metric mean; then take
    the top-3 methods by that value. rank done on the held-out TEST set."""
    best = _best_lambda_per_method(res, lambdas, rank_metric)
    top = sorted(best.items(), key=lambda kv: -kv[1][1])[:3]

    rows = []
    for m, (blam, _) in top:
        rows.append({"lead": [m, f"{float(blam):g}"],
                     "cells": _row_cells(res, blam, m),
                     "color": CAT_COLOR[CATEGORY[m]]})
    title = (f"Top-3 methods by test {rank_label} "
             f"(best $\\lambda$ per method, over all $\\lambda$)  (%)")
    render_table(os.path.join(out, fname + ".png"), title, rows,
                 lead_specs=[("Method", 2.6), ("$\\lambda$", 0.8)], n_folds=n_folds)
    write_csv(os.path.join(out, fname + ".csv"), rows, ["method", "lambda"])


def table_merged_top3(res, lambdas, out, n_folds=10):
    """Union of (top-3 by bAcc) and (top-3 by AUC), one row per unique (method,
    lambda); a 'top-3 in' column notes bAcc / AUC / both; the best value in each
    metric sub-column is bolded."""
    bacc = _best_lambda_per_method(res, lambdas, "balanced_acc")
    auc = _best_lambda_per_method(res, lambdas, "auc")
    top_b = [m for m, _ in sorted(bacc.items(), key=lambda kv: -kv[1][1])[:3]]
    top_a = [m for m, _ in sorted(auc.items(), key=lambda kv: -kv[1][1])[:3]]

    order, key2idx = [], {}
    def _add(m, lam, tag):
        k = (m, lam)
        if k in key2idx:
            order[key2idx[k]]["sel"].add(tag)
        else:
            key2idx[k] = len(order)
            order.append({"m": m, "lam": lam, "sel": {tag}})
    for m in top_b:
        _add(m, bacc[m][0], "bAcc")
    for m in top_a:
        _add(m, auc[m][0], "AUC")

    rows, nummat = [], []
    for o in order:
        m, lam = o["m"], o["lam"]
        sel = "both" if o["sel"] == {"bAcc", "AUC"} else next(iter(o["sel"]))
        rows.append({"lead": [m, f"{float(lam):g}", sel],
                     "cells": _row_cells(res, lam, m),
                     "color": CAT_COLOR[CATEGORY[m]]})
        nummat.append([[_pf(res, lam, m, s, mk)[0] for s, _ in SETS]
                       for mk, _ in METRICS])

    # bold the column-max in every metric sub-column
    bold = set()
    nr = len(rows)
    for g in range(len(METRICS)):
        for s in range(len(SETS)):
            bold.add((max(range(nr), key=lambda i: nummat[i][g][s]), g, s))

    title = ("Best methods: top-3 by test bAcc $\\cup$ top-3 by test AUC "
             "(best $\\lambda$ per method)  (%)")
    render_table(os.path.join(out, "cv_table_top3_merged.png"), title, rows,
                 lead_specs=[("Method", 2.6), ("$\\lambda$", 0.7), ("top-3 in", 1.15)],
                 bold=bold, n_folds=n_folds)
    write_csv(os.path.join(out, "cv_table_top3_merged.csv"), rows,
              ["method", "lambda", "top3_in"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/reports/cv/cv_results.json")
    ap.add_argument("--lambda0", default="0.0")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    res = data["results"]
    n_folds = int(data.get("n_folds", 10))
    out = os.path.dirname(args.results)
    lambdas = sorted(res, key=float)

    lam0 = args.lambda0 if args.lambda0 in res else str(float(args.lambda0))
    if lam0 not in res:
        raise SystemExit(f"lambda {args.lambda0} not in results {list(res)}")

    table_single_lambda(res, lam0, out, n_folds)
    table_top3(res, lambdas, "balanced_acc", "balanced accuracy",
               "cv_table_top3_by_bacc", out, n_folds)
    table_top3(res, lambdas, "auc", "AUC", "cv_table_top3_by_auc", out, n_folds)
    table_merged_top3(res, lambdas, out, n_folds)

    print("wrote:")
    for fn in ("cv_table_lambda0", "cv_table_top3_by_bacc", "cv_table_top3_by_auc",
               "cv_table_top3_merged"):
        print(f"   {out}/{fn}.png  +  .csv")


if __name__ == "__main__":
    main()
