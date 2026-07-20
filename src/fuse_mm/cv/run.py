"""Run the full-pipeline CV: for each lambda and fold, retrain the verifiers on
the fold's S_L+S_U, benchmark all methods on S_L/S_U/Test, and aggregate.

Aggregation per (lambda, method, set): per-fold mean +/- std of every metric,
plus the pooled out-of-fold Test metric (concatenate all folds' Test predictions,
each patient tested exactly once).
"""

from __future__ import annotations

import copy
import csv
import json
import os
from collections import defaultdict

import numpy as np

from ..bench import methods as M
from ..bench.metrics import score_stats
from ..bench.thresholds import choose_threshold
from ..fuse.bank import build_bank_pooled, load_full_featuresets, resolve_verifiers
from ..fuse.train import fit_and_score
from .folds import fold_assignment, load_cv_folds, make_cv_folds

SETS = ["labeled", "unlabeled", "test"]
FIT_SET = "labeled"      # tau is fit here (off-test) then applied to every set


def _agg(dicts):
    keys = list(dicts[0].keys())
    return {k: {"mean": float(np.mean([d[k] for d in dicts])),
                "std": float(np.std([d[k] for d in dicts]))} for k in keys}


def run_cv(cfg, lambdas, n_folds=10, n_test=1, device="cpu", out_dir=None, verbose=False,
           threshold_rules=("fixed",), prevalence=None, target_sensitivity=0.95,
           dump_predictions=False):
    """Full-pipeline CV. `threshold_rules` selects the decision threshold(s) used to
    turn soft scores into 0/1 calls (see bench.thresholds). Each rule is scored in
    the SAME pass (thresholding is post-scoring, so extra rules are ~free). Output:
    one cv_results.json per rule; the default ("fixed" = 0.5) writes to out_dir
    unchanged, other rules write to out_dir/<rule>/ so nothing is overwritten.

    dump_predictions: also save the pooled out-of-fold soft scores (each patient
    tested once) to out_dir/cv_pooled_predictions.npz, keyed "<lambda>__<method>",
    plus shared "y". These are threshold-free -> the input for ROC / PR curves
    (scripts/20_roc_pr.py). Off by default; adds no cost beyond the npz write."""
    rules = list(threshold_rules)
    thr_kw = {"prevalence": prevalence, "target_sensitivity": target_sensitivity}
    feats_dir = cfg["io"]["features_dir"]
    verifiers = resolve_verifiers(cfg, feats_dir)
    full_fs = load_full_featuresets(feats_dir, verifiers)     # all cohort patients, once
    try:
        folds = load_cv_folds(n_folds=n_folds)
    except FileNotFoundError:
        folds = make_cv_folds(n_folds=n_folds)
    n_folds = folds["n_folds"]

    all_methods = {**M.UNSUPERVISED, **M.SUPERVISED, **M.CEILING}
    results = {r: {} for r in rules}                          # rule -> {lam: {...}}
    dump = {}                                                 # "<lam>__<method>" -> pooled OOF score
    dump_y = None                                             # shared pooled OOF labels

    for lam in lambdas:
        lcfg = copy.deepcopy(cfg)
        lcfg["train"]["lambda_tci"] = lam
        # rule -> method -> set -> [metricdict]
        per_fold = {r: defaultdict(lambda: defaultdict(list)) for r in rules}
        # rule -> method -> (y_list, score_list, tau_list) for pooled Test
        pooled = {r: defaultdict(lambda: ([], [], [])) for r in rules}
        skipped = set()

        for i in range(n_folds):
            fs = fold_assignment(folds, i, n_test=n_test)
            banks, ys, pids = {}, {}, {}
            for key, s in [("L", "labeled"), ("U", "unlabeled"), ("T", "test")]:
                banks[key], ys[key], pids[key] = build_bank_pooled(full_fs, fs[s], verifiers)

            scores, names, *_ = fit_and_score(banks, ys, pids, lcfg, device, verbose)
            V = {s: scores[s][0] for s in SETS}
            Y = {s: scores[s][1] for s in SETS}

            for name, fn in all_methods.items():
                try:
                    pred = fn(V, Y)
                except NotImplementedError:
                    skipped.add(name); continue
                for r in rules:
                    tau = choose_threshold(r, pred[FIT_SET], Y[FIT_SET], **thr_kw)
                    for s in SETS:
                        per_fold[r][name][s].append(score_stats(pred[s], Y[s], threshold=tau))
                    yt, pt = Y["test"], pred["test"]
                    pooled[r][name][0].append(yt)
                    pooled[r][name][1].append(pt)
                    pooled[r][name][2].append(np.full(len(pt), tau))   # per-fold tau
            print(f"  lambda={lam} fold {i+1}/{n_folds} done", flush=True)

        for r in rules:
            results[r][str(lam)] = {
                "methods": {
                    name: {
                        "per_fold": {s: _agg(per_fold[r][name][s]) for s in SETS},
                        "pooled_test": score_stats(
                            np.concatenate(pooled[r][name][1]),
                            np.concatenate(pooled[r][name][0]),
                            threshold=np.concatenate(pooled[r][name][2])),
                    } for name in per_fold[r]
                },
                "skipped": sorted(skipped),
            }

        if dump_predictions:                                 # pooled OOF is threshold-free
            r0 = rules[0]
            for name in pooled[r0]:
                if dump_y is None:
                    dump_y = np.concatenate(pooled[r0][name][0])
                dump[f"{lam}__{name}"] = np.concatenate(pooled[r0][name][1])

    base_out = out_dir or os.path.join(os.path.dirname(cfg["io"]["out_dir"]), "reports", "cv")
    for r in rules:
        # default single-rule "fixed" keeps the historical path; others get a subdir
        r_out = base_out if rules == ["fixed"] else os.path.join(base_out, r)
        os.makedirs(r_out, exist_ok=True)
        with open(os.path.join(r_out, "cv_results.json"), "w", encoding="utf-8") as f:
            json.dump({"n_folds": n_folds, "n_test": n_test, "lambdas": lambdas,
                       "threshold_rule": r, "threshold_params": thr_kw,
                       "results": results[r]}, f, indent=2)
        _write_csv(results[r], r_out)
        if len(rules) > 1:
            print(f"\n### threshold rule: {r} ###")
        _print(results[r])
        print(f"\nwrote cv_results.json + cv_summary.csv to {r_out}/")

    if dump_predictions and dump:
        os.makedirs(base_out, exist_ok=True)
        np.savez_compressed(os.path.join(base_out, "cv_pooled_predictions.npz"),
                            y=dump_y, **dump)
        print(f"wrote cv_pooled_predictions.npz ({len(dump)} lambda-method arrays) "
              f"to {base_out}/")
    return results if len(rules) > 1 else results[rules[0]]


def _write_csv(results, out_dir):
    with open(os.path.join(out_dir, "cv_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "method", "set", "metric", "fold_mean", "fold_std", "pooled_test"])
        for lam, r in results.items():
            for name, md in r["methods"].items():
                for s in SETS:
                    for metric, ms in md["per_fold"][s].items():
                        pooled = md["pooled_test"].get(metric, "") if s == "test" else ""
                        w.writerow([lam, name, s, metric, f"{ms['mean']:.4f}",
                                    f"{ms['std']:.4f}",
                                    f"{pooled:.4f}" if pooled != "" else ""])


def _print(results):
    for lam, r in results.items():
        print(f"\n=== lambda={lam}  (test balanced_acc: fold mean+/-std | pooled OOF) ===")
        rows = sorted(r["methods"].items(),
                      key=lambda kv: kv[1]["per_fold"]["test"]["balanced_acc"]["mean"],
                      reverse=True)
        for name, md in rows:
            ba = md["per_fold"]["test"]["balanced_acc"]
            pooled = md["pooled_test"]["balanced_acc"]
            print(f"  {name:16s} {ba['mean']:.3f} +/- {ba['std']:.3f}   | {pooled:.3f}")
