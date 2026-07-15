"""Configuration + path resolution.

Everything tunable (drive paths, label scheme, split ratios) lives in
configs/data.yaml. This module loads it and resolves project-relative paths so
scripts work regardless of the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Repo root = two levels up from this file (src/fuse_mm/config.py)."""
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "configs" / "data.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load configs/data.yaml into a dict.

    Relative artifact paths are resolved against the project root so they land
    in the repo, while dataset paths (absolute, on the Z: drive) are left as-is.

    When `path` is None the file comes from FUSE_DATA_CONFIG if set, else
    configs/data.yaml. The env var exists because several call sites (cv.folds,
    fuse.train) call load_config() with no argument, so an alternative data
    config (e.g. configs/data_malig.yaml, the malignant-vs-rest relabeling)
    can only reach them via the environment:
        export FUSE_DATA_CONFIG=configs/data_malig.yaml
    """
    if path is None:
        path = os.environ.get("FUSE_DATA_CONFIG") or default_config_path()
    cfg_path = Path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _apply_env_overrides(cfg)

    root = project_root()
    art = cfg.setdefault("artifacts", {})
    for key in ("splits_path", "features_dir"):
        if key in art and not os.path.isabs(art[key]):
            art[key] = str(root / art[key])

    _validate(cfg)
    return cfg


# Environment variables let the SAME repo/config run on Windows and on WEXAC
# (Linux) without editing data.yaml — just export the Linux mount paths there.
_ENV_OVERRIDES = {
    "FUSE_DATASET_ROOT": ("dataset", "root"),
    "FUSE_MG_SUBDIR": ("dataset", "mg_subdir"),
    "FUSE_US_SUBDIR": ("dataset", "us_subdir"),
    "FUSE_LABEL_TABLE": ("dataset", "label_table"),
    "FUSE_SPLITS_PATH": ("artifacts", "splits_path"),
    "FUSE_FEATURES_DIR": ("artifacts", "features_dir"),
}


def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    for env_key, (section, field) in _ENV_OVERRIDES.items():
        val = os.environ.get(env_key)
        if val:
            cfg.setdefault(section, {})[field] = val


def _validate(cfg: dict[str, Any]) -> None:
    ratios = cfg["split"]["ratios"]
    total = ratios["labeled"] + ratios["unlabeled"] + ratios["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"split.ratios must sum to 1.0, got {total:.6f} "
            f"(labeled={ratios['labeled']}, unlabeled={ratios['unlabeled']}, "
            f"test={ratios['test']})"
        )
    scheme = cfg["labels"]["scheme"]
    valid = {"binary_abnormal", "binary_malignant", "binary_malignant_vs_rest",
             "multiclass"}
    if scheme not in valid:
        raise ValueError(f"labels.scheme must be one of {valid}, got {scheme!r}")
