"""Build aligned per-verifier, per-patient feature matrices for a split set.

Each verifier is a (modality, backbone). For a set (labeled/unlabeled/test), we
pool every patient's images to one vector per verifier (per_patient), aligned to
the split's patient order — MG verifiers keyed by mg_id, US by us_id, both
pointing at the same patient row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..heads.data import FeatureSet


def vkey(modality: str, backbone: str) -> str:
    return f"{modality}__{backbone}"          # ModuleDict-safe (no dots)


def discover_backbones(features_dir, set_name: str, modality: str) -> list[str]:
    d = Path(features_dir) / set_name / modality
    return sorted(p.stem for p in d.glob("*.npz") if not p.stem.endswith("_oof"))


def resolve_verifiers(cfg, features_dir) -> list[tuple[str, str]]:
    mods = cfg["verifiers"]["modalities"]
    bbs = cfg["verifiers"]["backbones"]
    out: list[tuple[str, str]] = []
    for mod in mods:
        blist = bbs or discover_backbones(features_dir, cfg["io"]["labeled_set"], mod)
        out.extend((mod, bb) for bb in blist)
    return out


def build_bank(features_dir, split, set_name: str, verifiers: list[tuple[str, str]]):
    """Returns (bank, y, patient_ids) where bank[vkey] = (n_patients, feat_dim)."""
    rows = split[set_name]
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    patient_ids = np.array([r["mg_id"] for r in rows])   # stable patient key
    bank: dict[str, np.ndarray] = {}
    for mod, bb in verifiers:
        fs = FeatureSet.load(features_dir, set_name, mod, bb)
        ids = np.array([r["mg_id"] if mod == "MG" else r["us_id"] for r in rows])
        bank[vkey(mod, bb)] = fs.pooled(ids).astype(np.float32)
    return bank, y, patient_ids


# --------------------------------------------------------------------------- #
# Cross-validation support: load ALL cohort patients, re-partition per fold    #
# --------------------------------------------------------------------------- #
def load_full_featuresets(features_dir, verifiers,
                          sets=("labeled", "unlabeled", "test")):
    """One FeatureSet per verifier over ALL cohort patients (union of the 3 dirs).
    Features are split-independent, so this lets any fold assignment be built
    without re-extraction."""
    from ..heads.data import load_feature_npz

    full = {}
    for mod, bb in verifiers:
        Xs, pids, ys = [], [], []
        for s in sets:
            X, pid, y = load_feature_npz(features_dir, s, mod, bb)
            Xs.append(X); pids.append(pid); ys.append(y)
        full[vkey(mod, bb)] = FeatureSet(np.concatenate(Xs),
                                         np.concatenate(pids), np.concatenate(ys))
    return full


def build_bank_pooled(full_fs, rows, verifiers):
    """Build (bank, y, patient_ids) for an arbitrary list of patient rows from the
    all-patient FeatureSets (used per CV fold)."""
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    patient_ids = np.array([r["mg_id"] for r in rows])
    bank = {}
    for mod, bb in verifiers:
        fs = full_fs[vkey(mod, bb)]
        ids = np.array([r["mg_id"] if mod == "MG" else r["us_id"] for r in rows])
        bank[vkey(mod, bb)] = fs.pooled(ids).astype(np.float32)
    return bank, y, patient_ids
