"""Lazy per-patient access to the .npy images.

Storage layout (per patient folder):
    MG: <patient>/<VIEW>_Series.../<image>.npy   grayscale uint8 (H, W),
        multiple views (L/R x CC/MLO/LM), sizes vary between patients.
    US: <patient>/Series0001/<uid>.npy           uint8, mixed (H, W) or (H, W, 3),
        multiple frames, sizes vary.

This module enumerates and loads raw arrays and normalizes channels to a
canonical (H, W, 3) uint8 layout. Resizing/backbone-specific preprocessing is
applied later by the feature-extraction transform, so this layer stays
numpy-only and free of any deep-learning dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

Modality = str  # "MG" | "US"


def list_patient_npy(patient_dir: Path) -> list[Path]:
    """All .npy files under a patient folder (recurses series/view subdirs)."""
    return sorted(patient_dir.rglob("*.npy"))


def to_hwc3(arr: np.ndarray) -> np.ndarray:
    """Normalize any loaded array to (H, W, 3) uint8.

    Handles grayscale (H, W), single-channel (H, W, 1), and RGB (H, W, 3).
    Higher-bit data is min-max scaled to 0-255 per image.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"unexpected array ndim={a.ndim}, shape={a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.shape[2] == 4:          # RGBA -> drop alpha
        a = a[:, :, :3]
    elif a.shape[2] != 3:
        raise ValueError(f"unexpected channel count {a.shape[2]} (shape={a.shape})")

    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = np.zeros_like(a) if hi <= lo else (a - lo) * (255.0 / (hi - lo))
        a = a.astype(np.uint8)
    return np.ascontiguousarray(a)


def load_image(npy_path: Path) -> np.ndarray:
    """Load one .npy image and return it as (H, W, 3) uint8."""
    return to_hwc3(np.load(npy_path, allow_pickle=False))


class PatientLoader:
    """Enumerate and lazily load all images for one patient/modality.

    Example:
        loader = PatientLoader(Path(rec.mg_dir), "MG")
        for img in loader.images():      # each (H, W, 3) uint8
            ...
    """

    def __init__(self, patient_dir: Path, modality: Modality):
        self.patient_dir = Path(patient_dir)
        self.modality = modality
        self._files: list[Path] | None = None

    @property
    def files(self) -> list[Path]:
        if self._files is None:
            self._files = list_patient_npy(self.patient_dir)
        return self._files

    def __len__(self) -> int:
        return len(self.files)

    def images(self) -> Iterator[np.ndarray]:
        for fp in self.files:
            yield load_image(fp)
