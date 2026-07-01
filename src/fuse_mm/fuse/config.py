"""Load configs/fuse.yaml and resolve io paths (shares FUSE_FEATURES_DIR override)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ..config import project_root


def load_fuse_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "configs" / "fuse.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = project_root()
    io = cfg["io"]
    for key in ("features_dir", "out_dir"):
        if not os.path.isabs(io[key]):
            io[key] = str(root / io[key])
    io["features_dir"] = os.environ.get("FUSE_FEATURES_DIR", io["features_dir"])
    return cfg
