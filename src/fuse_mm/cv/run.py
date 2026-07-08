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
from ..fuse.bank import build_bank_pooled, load_full_featuresets, resolve_verifiers
from ..fuse.train import fit_and_score
from .folds import fold_assignment, load_cv_folds, make_cv_folds

SETS = ["labeled", "unlabeled", "test"]


def _agg(dicts):
    keys = list(dicts[0].keys())
    return {k: {"mean": float(np.mean([d[k] for d in dicts])),
                "std": float(np.std([d[k] for d in dicts]))} for k in keys}


def run_cv(cfg, lambdas, n_folds=10, device="cpu", out_dir=None, verbose=False):
    feats_dir = cfg["io"]["features_dir"]
    verifiers = resolve_verifiers(cfg, feats_dir)
    full_fs = load_full_featuresets(feats_dir, verifiers)     # all cohort patients, once
    try:
        folds = load_cv_folds()
    except FileNotFoundError:
        folds = make_cv_folds(n_folds=n_folds)
    n_folds = folds["n_folds"]

    all_methods = {**M.UNSUPERVISED, **M.SUPERVISED, **M.CEILING}
    results = {}

    for lam in lambdas:
        lcfg = copy.deepcopy(cfg)
        lcfg["train"]["lambda_tci"] = lam
        per_fold = defaultdict(lambda: defaultdict(list))     # method -> set -> [metricdict]
        pooled = defaultdict(lambda: ([], []))                # method -> (y_list, score_list) for Test
        skipped = set()

        for i in range(n_folds):
            fs = fold_assignment(folds, i)
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
                for s in SETS:
                    per_fold[name][s].append(score_stats(pred[s], Y[s]))
                pooled[name][0].append(Y["test"]); pooled[name][1].append(pred["test"])
            print(f"  lambda={lam} fold {i+1}/{n_folds} done", flush=True)

        results[str(lam)] = {
            "methods": {
                name: {
                    "per_fold": {s: _agg(per_fold[name][s]) for s in SETS},
                    "pooled_test": score_stats(np.concatenate(pooled[name][1]),
                                               np.concatenate(pooled[name][0])),
                } for name in per_fold
            },
            "skipped": sorted(skipped),
        }

    out_dir = out_dir or os.path.join(os.path.dirname(cfg["io"]["out_dir"]), "reports", "cv")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump({"n_folds": n_folds, "lambdas": lambdas, "results": results}, f, indent=2)
    _write_csv(results, out_dir)
    _print(results)
    print(f"\nwrote cv_results.json + cv_summary.csv to {out_dir}/")
    return results


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
