"""Run stratified patient-level k-fold CV for one (modality, backbone).

Produces patient-level out-of-fold (OOF) probabilities + per-fold and summary
metrics. The OOF probability vector (patients x 1) is the clean signal for the
later verifier decorrelation / combination study, so it is saved alongside the
metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import run_tag
from .data import FeatureSet
from .metrics import (
    binary_metrics, standardize_apply, standardize_fit, stratified_kfold,
)
from .models import train_head


def run_cv(cfg, modality: str, backbone: str, device: str = "cpu") -> dict:
    fs = FeatureSet.load(cfg["io"]["features_dir"],
                         cfg["experiment"]["feature_set"], modality, backbone)
    mode = cfg["aggregation"]["mode"]
    head = cfg["head"]

    patients, y_pat = fs.patients, fs.y_patient
    folds = stratified_kfold(y_pat, cfg["cv"]["folds"], cfg["cv"]["seed"])

    oof = np.full(len(patients), np.nan, dtype=np.float64)
    fold_metrics = []

    for tr, va in folds:
        tr_pat, va_pat = patients[tr], patients[va]

        if mode == "per_patient":
            Xtr, ytr = fs.pooled(tr_pat), y_pat[tr]
            Xva = fs.pooled(va_pat)
            if head["standardize"]:
                mu, sd = standardize_fit(Xtr)
                Xtr, Xva = standardize_apply(Xtr, mu, sd), standardize_apply(Xva, mu, sd)
            predict = train_head(Xtr, ytr, head, fs.n_classes, device)
            prob_va = predict(Xva)[:, 1]
        else:  # per_image: train on images, pool probs per val patient
            Xtr_img, ytr_img, _ = fs.images_of(tr_pat)
            if head["standardize"]:
                mu, sd = standardize_fit(Xtr_img)
                Xtr_img = standardize_apply(Xtr_img, mu, sd)
            predict = train_head(Xtr_img, ytr_img, head, fs.n_classes, device)
            prob_va = np.empty(len(va_pat))
            for k, p in enumerate(va_pat):
                Xp = fs.X[fs.image_indices(p)]
                if head["standardize"]:
                    Xp = standardize_apply(Xp, mu, sd)
                prob_va[k] = predict(Xp)[:, 1].mean()

        oof[va] = prob_va
        fold_metrics.append(binary_metrics(y_pat[va], prob_va))

    summary = {
        "modality": modality,
        "backbone": backbone,
        "aggregation": mode,
        "head_type": head["type"],
        "feat_dim": int(fs.feat_dim),
        "n_patients": int(len(patients)),
        "label_distribution": {int(c): int(n)
                               for c, n in zip(*np.unique(y_pat, return_counts=True))},
        "folds": fold_metrics,
        "mean": {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]},
        "std": {k: float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]},
    }

    _save(cfg, modality, backbone, summary, patients, y_pat, oof)
    return summary


def _save(cfg, modality, backbone, summary, patients, y_pat, oof):
    out_dir = Path(cfg["io"]["results_dir"]) / run_tag(cfg) / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{backbone}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        out_dir / f"{backbone}_oof.npz",
        patient_ids=patients, y_true=y_pat, oof_prob=oof,
    )
