"""Load extracted features and expose patient-level views + aggregation.

A patient owns several images (MG views / US frames). Labels are per-patient,
propagated to each image at extraction time. Cross-validation must therefore
split at the PATIENT level in both aggregation modes to avoid leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_feature_npz(features_dir, set_name: str, modality: str, backbone: str):
    fp = Path(features_dir) / set_name / modality / f"{backbone}.npz"
    if not fp.exists():
        raise FileNotFoundError(f"missing features: {fp}")
    d = np.load(fp, allow_pickle=True)
    X = d["X"].astype(np.float32)
    pids = d["patient_ids"].astype(str)
    y = d["labels"].astype(np.int64)
    return X, pids, y


class FeatureSet:
    """Per-image features for one (modality, backbone), indexed by patient."""

    def __init__(self, X: np.ndarray, pids: np.ndarray, y_img: np.ndarray):
        self.X = X
        self.pids = pids
        self.y_img = y_img

        # stable patient order + per-patient image indices + per-patient label
        self.patients: list[str] = []
        self._idx: dict[str, list[int]] = {}
        seen: dict[str, int] = {}
        for i, p in enumerate(pids):
            if p not in seen:
                seen[p] = len(self.patients)
                self.patients.append(p)
                self._idx[p] = []
            self._idx[p].append(i)
        self.patients = np.array(self.patients)
        self.y_patient = np.array(
            [int(self.y_img[self._idx[p][0]]) for p in self.patients], dtype=np.int64
        )
        self.feat_dim = X.shape[1]
        self.n_classes = int(self.y_img.max()) + 1

    @classmethod
    def load(cls, features_dir, set_name, modality, backbone) -> "FeatureSet":
        return cls(*load_feature_npz(features_dir, set_name, modality, backbone))

    def image_indices(self, patient: str) -> list[int]:
        return self._idx[patient]

    def pooled(self, patients: np.ndarray) -> np.ndarray:
        """Mean-pool each patient's image features -> (len(patients), feat_dim)."""
        return np.stack(
            [self.X[self._idx[p]].mean(axis=0) for p in patients], axis=0
        ).astype(np.float32)

    def images_of(self, patients: np.ndarray):
        """Stack all images belonging to `patients`. Returns (X, y, owner_pid)."""
        idx = [i for p in patients for i in self._idx[p]]
        idx = np.array(idx, dtype=np.int64)
        owner = np.array([self.pids[i] for i in idx])
        return self.X[idx], self.y_img[idx], owner
