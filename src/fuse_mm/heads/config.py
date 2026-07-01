"""Load configs/train.yaml and resolve io paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ..config import project_root


def load_train_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "configs" / "train.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = project_root()
    io = cfg["io"]
    for key in ("features_dir", "results_dir"):
        if not os.path.isabs(io[key]):
            io[key] = str(root / io[key])
    # share the same env override as extraction, so WEXAC can relocate features
    io["features_dir"] = os.environ.get("FUSE_FEATURES_DIR", io["features_dir"])

    _validate(cfg)
    return cfg


def run_tag(cfg: dict[str, Any]) -> str:
    """Stable id for this experiment combo, e.g. 'per_patient_linear' or
    'per_patient_linear_ncl2' when multiple decorrelated outputs are used."""
    tag = f"{cfg['aggregation']['mode']}_{cfg['head']['type']}"
    n_out = int(cfg["head"].get("n_outputs", 1))
    if n_out > 1:
        tag += f"_ncl{n_out}"
    return tag


def _validate(cfg: dict[str, Any]) -> None:
    mode = cfg["aggregation"]["mode"]
    if mode not in {"per_patient", "per_image"}:
        raise ValueError(f"aggregation.mode must be per_patient|per_image, got {mode!r}")
    htype = cfg["head"]["type"]
    if htype not in {"linear", "mlp"}:
        raise ValueError(f"head.type must be linear|mlp, got {htype!r}")
    if htype == "mlp" and not cfg["head"].get("mlp_layers"):
        raise ValueError("head.type == mlp requires non-empty head.mlp_layers")
