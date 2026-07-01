"""Stratified, patient-level Labeled / Unlabeled / Test split.

The split is computed on the PAIRED cohort, so every patient is assigned to the
same set in both modalities. Stratification by label preserves the class
distribution across all three sets. The result is fully determined by
split.seed and split.ratios in configs/data.yaml.

Output (artifacts/splits.json) stores patient ids only — not pixels — so it is
small, reproducible, and modality-agnostic.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .labels import PatientRecord, build_cohort, label_distribution


def _largest_remainder(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Split `total` items into integer counts that sum exactly to total."""
    raw = {k: total * v for k, v in ratios.items()}
    floored = {k: int(x) for k, x in raw.items()}
    remainder = total - sum(floored.values())
    # hand out the leftover to the largest fractional parts
    frac_order = sorted(raw, key=lambda k: raw[k] - floored[k], reverse=True)
    for k in frac_order[:remainder]:
        floored[k] += 1
    return floored


def make_splits(cfg: dict[str, Any], write: bool = True) -> dict[str, Any]:
    """Compute the stratified split and (optionally) write splits.json."""
    cohort = build_cohort(cfg)
    if not cohort:
        raise RuntimeError("cohort is empty — check dataset paths / drive access")

    ratios = cfg["split"]["ratios"]
    seed = cfg["split"]["seed"]
    rng = random.Random(seed)

    # group patients by label, then split each group with the same ratios
    by_label: dict[int, list[PatientRecord]] = defaultdict(list)
    for r in cohort:
        by_label[r.label].append(r)

    assignment: dict[str, list[PatientRecord]] = {
        "labeled": [], "unlabeled": [], "test": []
    }
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda r: r.mg_id)  # deterministic
        rng.shuffle(group)
        counts = _largest_remainder(len(group), ratios)
        i = 0
        for set_name in ("labeled", "unlabeled", "test"):
            n = counts[set_name]
            assignment[set_name].extend(group[i:i + n])
            i += n

    splits = _serialize(cfg, cohort, assignment)
    if write:
        out = Path(cfg["artifacts"]["splits_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
    return splits


def _serialize(cfg, cohort, assignment) -> dict[str, Any]:
    def pack(records: list[PatientRecord]) -> list[dict[str, Any]]:
        return [
            {"mg_id": r.mg_id, "us_id": r.us_id,
             "label": r.label, "raw_label": r.raw_label,
             "mg_finding_side": r.mg_finding_side,
             "us_finding_side": r.us_finding_side}
            for r in sorted(records, key=lambda r: r.mg_id)
        ]

    return {
        "meta": {
            "scheme": cfg["labels"]["scheme"],
            "class_names": cfg["labels"]["class_names"],
            "ratios": cfg["split"]["ratios"],
            "seed": cfg["split"]["seed"],
            "require_both_modalities": cfg["split"].get("require_both_modalities", True),
            "cohort_size": len(cohort),
            "cohort_label_distribution": label_distribution(cohort),
            "set_sizes": {k: len(v) for k, v in assignment.items()},
            "set_label_distributions": {
                k: label_distribution(v) for k, v in assignment.items()
            },
        },
        "labeled": pack(assignment["labeled"]),
        "unlabeled": pack(assignment["unlabeled"]),
        "test": pack(assignment["test"]),
    }


def load_splits(cfg: dict[str, Any]) -> dict[str, Any]:
    with open(cfg["artifacts"]["splits_path"], "r", encoding="utf-8") as f:
        return json.load(f)
