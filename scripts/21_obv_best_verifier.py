"""How often each verifier is picked as the OBV (Oracle Best Verifier) across folds.

obv (bench.methods.obv) picks, PER SET, the single verifier with the highest
balanced accuracy (scores thresholded at 0.5, using that set's labels). This
script records that argmax for every CV fold and set, then plots a bar chart:
x = verifiers, y = number of folds that crowned each verifier.

Not stored by 12_cv (obv returns the chosen verifier's scores, not its index), so
`dump` retrains the pipeline per fold (like 16_cv_ensemble_weights) and records the
choice; `plot` is cheap/local.

    # 1) retrain per fold, record obv's pick per set  (WEXAC; CPU is fine)
    python scripts/21_obv_best_verifier.py dump --lambda 0 \
        --config configs/fuse_1ans.yaml --out-dir artifacts/reports/obv_1ans
    # 2) bar chart from the JSON  (local)
    python scripts/21_obv_best_verifier.py plot --lambda 0 \
        --out-dir artifacts/reports/obv_1ans --set test

The DATA task (normal-vs-abnormal vs malignant) is chosen by FUSE_DATA_CONFIG in
the environment, exactly like 12_cv / 16, so the same script serves every setup.

Outputs (<out-dir>/):
    obv_best_verifier_lambda_<L>.json                per-fold, per-set picks + names
    obv_best_verifier_lambda_<L>_<set>.png           the histogram (set=all -> grouped)
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

OUT_DEFAULT = "artifacts/reports/obv_best_verifier"
SETS = ["labeled", "unlabeled", "test"]
# modality colors (US generally >> MG in AUC, so worth seeing the split)
MOD_COLOR = {"US": "#0072B2", "MG": "#E69F00"}


def _json_path(out, lam):
    return os.path.join(out, f"obv_best_verifier_lambda_{lam:g}.json")


def _obv_pick(Vs, y):
    """Replicate bench.methods.obv's choice: argmax balanced acc at 0.5, earliest
    on ties. Returns (best_index, best_balanced_acc)."""
    from fuse_mm.bench.metrics import _sens_spec
    best_j, best_ba = 0, -1.0
    for j in range(Vs.shape[1]):
        sens, spec = _sens_spec((Vs[:, j] >= 0.5).astype(int), y)
        ba = (sens + spec) / 2.0
        if ba > best_ba:
            best_ba, best_j = ba, j
    return best_j, float(best_ba)


# --------------------------------------------------------------------------- #
def dump(lam, device, config, out, folds_n, n_test, verbose):
    import copy
    from fuse_mm.fuse import load_fuse_config
    from fuse_mm.fuse.bank import (build_bank_pooled, load_full_featuresets,
                                   resolve_verifiers)
    from fuse_mm.fuse.train import fit_and_score
    from fuse_mm.cv.folds import fold_assignment, load_cv_folds, make_cv_folds

    cfg = load_fuse_config(config)
    cfg["train"]["lambda_tci"] = lam
    feats_dir = cfg["io"]["features_dir"]
    verifiers = resolve_verifiers(cfg, feats_dir)
    full_fs = load_full_featuresets(feats_dir, verifiers)
    try:
        folds = load_cv_folds(n_folds=folds_n)
    except FileNotFoundError:
        folds = make_cv_folds(n_folds=folds_n)
    folds_n = folds["n_folds"]

    names_ref, records = None, []
    for i in range(folds_n):
        fs = fold_assignment(folds, i, n_test=n_test)
        banks, ys, pids = {}, {}, {}
        for key, s in [("L", "labeled"), ("U", "unlabeled"), ("T", "test")]:
            banks[key], ys[key], pids[key] = build_bank_pooled(full_fs, fs[s], verifiers)
        scores, names, *_ = fit_and_score(banks, ys, pids, copy.deepcopy(cfg), device, verbose)
        names_ref = list(map(str, names))
        picks = {}
        for s in SETS:
            j, ba = _obv_pick(scores[s][0].astype(float), scores[s][1])
            picks[s] = {"idx": int(j), "name": names_ref[j], "bacc": ba}
        records.append({"fold": i, "picks": picks})
        print(f"  lambda={lam} fold {i+1}/{folds_n}: "
              + ", ".join(f"{s}={picks[s]['name']}({picks[s]['bacc']:.3f})" for s in SETS),
              flush=True)

    os.makedirs(out, exist_ok=True)
    payload = {"lambda": lam, "config": config, "n_folds": folds_n, "n_test": n_test,
               "verifier_names": names_ref, "folds": records}
    with open(_json_path(out, lam), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {_json_path(out, lam)}  ({folds_n} folds)")


# --------------------------------------------------------------------------- #
def _counts(data, set_name):
    names = data["verifier_names"]
    c = np.zeros(len(names), dtype=int)
    for rec in data["folds"]:
        c[rec["picks"][set_name]["idx"]] += 1
    return names, c


def _modality(name):
    return name.split("__")[0] if "__" in name else "?"


def _bar_single(ax, names, counts, title):
    order = np.argsort(-counts, kind="mergesort")            # most-picked first
    keep = [i for i in order if counts[i] > 0]               # only verifiers ever chosen
    if not keep:
        keep = list(order)
    labels = [names[i] for i in keep]
    vals = [counts[i] for i in keep]
    colors = [MOD_COLOR.get(_modality(names[i]), "#999999") for i in keep]
    x = np.arange(len(keep))
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
    ax.set_ylabel("folds that chose it as best verifier")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", color="#c9c9c9", alpha=0.4, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for xi, v in zip(x, vals):
        ax.annotate(str(v), (xi, v), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=7.5, color="#333")
    ax.set_title(title, fontsize=12, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MOD_COLOR.values()]
    ax.legend(handles, list(MOD_COLOR), title="modality", frameon=False, fontsize=9)


def plot(lam, out, set_name):
    with open(_json_path(out, lam), encoding="utf-8") as f:
        data = json.load(f)
    task = os.path.basename(out.rstrip("/\\"))
    sets = SETS if set_name == "all" else [set_name]
    if len(sets) == 1:
        names, counts = _counts(data, sets[0])
        fig, ax = plt.subplots(figsize=(max(7.0, 0.42 * max(1, int((counts > 0).sum())) + 3), 5.6))
        _bar_single(ax, names, counts,
                    f"OBV best-verifier frequency — {sets[0]} "
                    f"($\\lambda$={lam:g}, {data['n_folds']} folds, {task})")
        fig.tight_layout()
        fp = os.path.join(out, f"obv_best_verifier_lambda_{lam:g}_{sets[0]}.png")
        fig.savefig(fp, dpi=150); plt.close(fig)
        print("wrote", fp)
        return

    # set == all: union of ever-chosen verifiers, grouped bars per set
    names = data["verifier_names"]
    per = {s: _counts(data, s)[1] for s in SETS}
    ever = [i for i in range(len(names)) if any(per[s][i] > 0 for s in SETS)]
    ever.sort(key=lambda i: -sum(per[s][i] for s in SETS))
    x = np.arange(len(ever)); w = 0.26
    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * len(ever) + 3), 5.8))
    setcol = {"labeled": "#009E73", "unlabeled": "#56B4E9", "test": "#D55E00"}
    for k, s in enumerate(SETS):
        ax.bar(x + (k - 1) * w, [per[s][i] for i in ever], width=w, label=s,
               color=setcol[s], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([names[i] for i in ever], rotation=90, fontsize=7.5)
    ax.set_ylabel("folds that chose it as best verifier")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", color="#c9c9c9", alpha=0.4, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(title="set", frameon=False, fontsize=9)
    ax.set_title(f"OBV best-verifier frequency — all sets "
                 f"($\\lambda$={lam:g}, {data['n_folds']} folds, {task})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fp = os.path.join(out, f"obv_best_verifier_lambda_{lam:g}_all.png")
    fig.savefig(fp, dpi=150); plt.close(fig)
    print("wrote", fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "plot"])
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0)
    ap.add_argument("--config", default=None,
                    help="FUSE head config (default configs/fuse.yaml, 2 answers)")
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=1)
    ap.add_argument("--set", dest="set_name", default="test",
                    choices=["labeled", "unlabeled", "test", "all"],
                    help="which set's obv pick to histogram (default test)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.mode == "dump":
        dump(args.lam, args.device, args.config, args.out_dir, args.folds,
             args.n_test, args.verbose)
    else:
        plot(args.lam, args.out_dir, args.set_name)


if __name__ == "__main__":
    main()
