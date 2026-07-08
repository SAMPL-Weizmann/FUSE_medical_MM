"""Stratified 10-fold assignment over the cohort for the CV rotation.

The cohort = the union of the existing labeled/unlabeled/test records in
splits.json (no drive access needed; folds are just a re-partition). Rotation:
  iteration i -> S_L = fold i, Test = fold (i+1) mod n, S_U = the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import load_config, project_root
from ..splits import load_splits


def _cv_path(cfg) -> Path:
    return Path(cfg["artifacts"]["splits_path"]).with_name("cv_folds.json")


def make_cv_folds(n_folds: int = 10, seed: int = 42, cfg=None, write: bool = True) -> dict:
    cfg = cfg or load_config()
    split = load_splits(cfg)
    records = [dict(r) for r in split["labeled"] + split["unlabeled"] + split["test"]]

    # dedup by (mg_id, us_id) — a few patients appear twice in the source table,
    # and a duplicate across folds would leak between S_L/S_U/Test.
    seen, unique = set(), []
    for r in records:
        key = (r["mg_id"], r["us_id"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    records = unique

    y = np.array([int(r["label"]) for r in records])
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(records), dtype=int)
    for c in np.unique(y):                       # stratified round-robin
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_folds
    for r, f in zip(records, fold_of):
        r["fold"] = int(f)

    folds = {"n_folds": n_folds, "seed": seed, "records": records,
             "fold_sizes": {int(f): int((fold_of == f).sum()) for f in range(n_folds)}}
    if write:
        with open(_cv_path(cfg), "w", encoding="utf-8") as fh:
            json.dump(folds, fh, indent=2)
    return folds


def load_cv_folds(cfg=None) -> dict:
    cfg = cfg or load_config()
    with open(_cv_path(cfg), "r", encoding="utf-8") as fh:
        return json.load(fh)


def fold_assignment(folds: dict, i: int) -> dict:
    """S_L = fold i, Test = fold (i+1) mod n (circular), S_U = the rest."""
    n = folds["n_folds"]
    test = (i + 1) % n
    recs = folds["records"]
    return {
        "labeled": [r for r in recs if r["fold"] == i],
        "test": [r for r in recs if r["fold"] == test],
        "unlabeled": [r for r in recs if r["fold"] not in (i, test)],
    }
