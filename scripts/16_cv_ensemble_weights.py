"""Per-fold ensemble-predictor weights (fuse_ens) for one lambda, for the CV folds.

The single-split figure (10_ensemble_weights.py) shows theta_j for the one original
fold. This shows the SAME weights but for the CV's best folds: it retrains the
pipeline per fold (like 12_cv.py), fits f_theta = sigma(theta.V + b) on each fold's
unlabeled/OOF scores, records the fold's fuse_ens TEST balanced accuracy (to rank
folds) and dumps every fold's (theta, b) to JSON. A separate plot mode then draws
the top-3 folds as three thin per-verifier bars (red=best, green=2nd, blue=3rd).

Two stages (training is heavy -> run `dump` on WEXAC; `plot` is cheap/local):

    # 1) retrain per fold, save weights  (WEXAC; CPU is fine)
    python scripts/16_cv_ensemble_weights.py dump --lambda 0.2 [--device cpu]

    # 2) build the two figures from the JSON  (local)
    python scripts/16_cv_ensemble_weights.py plot --lambda 0.2 [--topk 3]

Outputs (artifacts/reports/ensemble_weights/):
    cv_ens_weights_lambda_<L>.json
    cv_ens_weights_lambda_<L>_by_bestfold.png     (verifiers ordered by best fold's theta)
    cv_ens_weights_lambda_<L>_lexicographic.png   (verifiers ordered A->Z)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fuse_mm.fuse.estimate import (  # noqa: E402
    fit_mom, posterior_triplet_avg, optimize_ensemble, predict_ensemble,
)

OUT = "artifacts/reports/ensemble_weights"
# fold-rank colors: best=red, 2nd=green, 3rd=blue (CVD-considerate red/green/blue)
FOLD_COLORS = ["#D55E00", "#009E73", "#0072B2"]
RANK_NAME = ["best", "2nd", "3rd"]


def _json_path(lam):
    return os.path.join(OUT, f"cv_ens_weights_lambda_{lam:g}.json")


# --------------------------------------------------------------------------- #
def dump(lam, device, verbose, config=None):
    """Retrain per fold, fit f_theta on each fold's OOF scores, save all folds.
    `config` = FUSE head config (None -> configs/fuse.yaml, 2 answers). The data
    task (abnormal vs malignant) comes from FUSE_DATA_CONFIG in the environment."""
    from fuse_mm.bench.metrics import score_stats
    from fuse_mm.fuse import load_fuse_config
    from fuse_mm.fuse.bank import (
        build_bank_pooled, load_full_featuresets, resolve_verifiers)
    from fuse_mm.fuse.train import fit_and_score
    from fuse_mm.cv.folds import fold_assignment, load_cv_folds, make_cv_folds
    import copy

    cfg = load_fuse_config(config)
    cfg["train"]["lambda_tci"] = lam
    feats_dir = cfg["io"]["features_dir"]
    verifiers = resolve_verifiers(cfg, feats_dir)
    full_fs = load_full_featuresets(feats_dir, verifiers)
    try:
        folds = load_cv_folds()
    except FileNotFoundError:
        folds = make_cv_folds()
    n_folds = folds["n_folds"]

    records, names_ref = [], None
    for i in range(n_folds):
        fs = fold_assignment(folds, i)
        banks, ys, pids = {}, {}, {}
        for key, s in [("L", "labeled"), ("U", "unlabeled"), ("T", "test")]:
            banks[key], ys[key], pids[key] = build_bank_pooled(full_fs, fs[s], verifiers)
        scores, names, *_ = fit_and_score(banks, ys, pids, cfg, device, verbose)
        names_ref = list(map(str, names))
        V = {s: scores[s][0].astype(float) for s in ("labeled", "unlabeled", "test")}
        Y = {s: scores[s][1] for s in ("labeled", "unlabeled", "test")}

        params = fit_mom(V["unlabeled"])                          # fuse_ens est = unlabeled
        phat = posterior_triplet_avg(V["unlabeled"], params)
        w, b = optimize_ensemble(V["unlabeled"], phat)
        test_pred = predict_ensemble(V["test"], w, b)
        test_bacc = score_stats(test_pred, Y["test"])["balanced_acc"]

        records.append({"fold": i, "test_bacc": float(test_bacc),
                        "bias": float(b), "theta": [float(x) for x in w]})
        print(f"  lambda={lam} fold {i+1}/{n_folds}: "
              f"fuse_ens test bAcc={test_bacc:.4f}, b={b:+.4f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    payload = {"lambda": lam, "verifier_names": names_ref,
               "n_folds": n_folds, "folds": records}
    with open(_json_path(lam), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {_json_path(lam)}  ({n_folds} folds)")


# --------------------------------------------------------------------------- #
def _grouped_barh(names, top, order, lam, title_suffix, fname):
    n = len(names)
    y = np.arange(n)
    h = 0.26
    offs = [0.27, 0.0, -0.27]                                  # rank0 on top of the group
    fig, ax = plt.subplots(figsize=(8.5, max(6.5, n * 0.32)))
    for rank, rec in enumerate(top):
        theta = np.asarray(rec["theta"])[order]
        ax.barh(y + offs[rank], theta, height=h, color=FOLD_COLORS[rank],
                edgecolor="white", linewidth=0.3, zorder=2,
                label=f"fold {rec['fold'] + 1} ({RANK_NAME[rank]}, "
                      f"bAcc={rec['test_bacc']:.3f},  b={rec['bias']:+.3f})")
    ax.axvline(0, color="k", lw=0.8, zorder=3)
    for yb in np.arange(n - 1) + 0.5:                          # separate each verifier's group
        ax.axhline(yb, color="#bdbdbd", lw=0.6, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(r"$\theta$  (verifier weight in $f_\theta$)")
    ax.grid(axis="x", color="#c9c9c9", alpha=0.4, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"fuse_ens ensemble weights — top-3 CV folds  "
                 f"($\\lambda$={lam:g})\n{title_suffix}", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.9,
              title="fold (rank, test bAcc, bias)")
    fig.tight_layout()
    fp = os.path.join(OUT, fname)
    fig.savefig(fp, dpi=150); plt.close(fig)
    return fp


def plot(lam, topk):
    with open(_json_path(lam), encoding="utf-8") as f:
        data = json.load(f)
    names = data["verifier_names"]
    folds = sorted(data["folds"], key=lambda r: -r["test_bacc"])
    top = folds[:topk]
    best_theta = np.asarray(top[0]["theta"])

    # version A: verifiers ordered by the BEST fold's theta (signed, ascending)
    order_best = list(np.argsort(best_theta))
    fp1 = _grouped_barh(names, top, order_best, lam,
                        "verifiers ordered by best fold's weight",
                        f"cv_ens_weights_lambda_{lam:g}_by_bestfold.png")
    # version B: lexicographic verifier order
    order_lex = list(np.argsort(names))
    fp2 = _grouped_barh(names, top, order_lex, lam,
                        "verifiers in lexicographic order",
                        f"cv_ens_weights_lambda_{lam:g}_lexicographic.png")
    print("wrote:\n  ", fp1, "\n  ", fp2)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "plot"])
    ap.add_argument("--lambda", dest="lam", type=float, default=0.2)
    ap.add_argument("--config", default=None,
                    help="FUSE head config (default configs/fuse.yaml, 2 answers; "
                         "pass configs/fuse_1ans.yaml for the 1-answer setup)")
    ap.add_argument("--out-dir", default=OUT,
                    help=f"output dir (default {OUT}). Use a separate dir for the "
                         "1-answer setup so the 2-answer figures aren't overwritten.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    OUT = args.out_dir                                 # _json_path / _grouped_barh read this
    if args.mode == "dump":
        dump(args.lam, args.device, args.verbose, config=args.config)
    else:
        plot(args.lam, args.topk)


if __name__ == "__main__":
    main()
