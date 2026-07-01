"""Per-patient scan selection — choose which .npy files to use per patient.

Unequal scan counts correlate with the label (especially US: few targeted
clips for findings vs. many sweep frames for normals), so using all images can
leak. This module reduces each patient to a fixed set, mirroring the prior team:

  MG: the 4 canonical views (finding-side; default Left unless finding side is
      'RT'; fall back to the other side), one .npy per view folder. Patients
      lacking the 4 views are invalid (dropped when require_all_views).
  US: k frames (k=1 by default), with sampling + optional repeat-pad so no
      patient is dropped.

Set selection.<mod>.mode to "all" to use every image instead.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

# .npy enumeration is shared with dataset.py's all-files behaviour
from .dataset import list_patient_npy


# --------------------------------------------------------------------------- #
# MG: canonical views                                                          #
# --------------------------------------------------------------------------- #
def mg_side(finding_side: str) -> str:
    """Prior team's rule: Right only when finding side is exactly 'RT'."""
    return "R" if str(finding_side).strip() == "RT" else "L"


def _mg_view_folders(mg_dir: Path, side: str, views: list[str]) -> list[Path]:
    prefixes = [f"{side} {v}" for v in views]
    try:
        subs = sorted(p for p in os.listdir(mg_dir) if (mg_dir / p).is_dir())
    except FileNotFoundError:
        return []
    return [mg_dir / f for f in subs if any(f.startswith(pre) for pre in prefixes)]


def select_mg(mg_dir: Path | None, finding_side: str, cfg: dict[str, Any]) -> list[Path] | None:
    sel = cfg["selection"]["mg"]
    if sel["mode"] == "all":
        return list_patient_npy(mg_dir) if mg_dir else None
    if mg_dir is None:
        return None
    views = sel["views"]
    # try finding side, then the other side
    for side in _ordered_sides(mg_side(finding_side)):
        folders = _mg_view_folders(mg_dir, side, views)
        if len(folders) == len(views):
            npys = []
            for folder in folders:
                files = sorted(p for p in os.listdir(folder) if p.endswith(".npy"))
                if not files:
                    npys = []
                    break
                npys.append(folder / files[0])
            if len(npys) == len(views):
                return npys
    return None  # missing canonical views


def _ordered_sides(side: str) -> list[str]:
    return [side, "L" if side == "R" else "R"]


# --------------------------------------------------------------------------- #
# US: fixed k frames                                                           #
# --------------------------------------------------------------------------- #
def select_us(us_dir: Path | None, finding_side: str, cfg: dict[str, Any]) -> list[Path] | None:
    sel = cfg["selection"]["us"]
    if us_dir is None:
        return None
    files = list_patient_npy(us_dir)
    if sel["mode"] == "all":
        return files or None
    if not files:
        return None

    k = int(sel["k"])
    strategy = sel.get("strategy", "first")
    if len(files) >= k:
        if strategy == "first":
            return files[:k]
        if strategy == "evenly":
            idx = [round(i * (len(files) - 1) / max(k - 1, 1)) for i in range(k)]
            return [files[i] for i in idx]
        if strategy == "random":
            rng = random.Random(sel.get("seed", 42))
            return sorted(rng.sample(files, k))
        raise ValueError(f"unknown us strategy {strategy!r}")
    # fewer than k available
    if sel.get("pad", "repeat") == "repeat":
        out = list(files)
        i = 0
        while len(out) < k:
            out.append(files[i % len(files)])
            i += 1
        return out
    return files  # pad == none: keep what we have


# --------------------------------------------------------------------------- #
# Dispatch + cohort validity                                                   #
# --------------------------------------------------------------------------- #
def select_npys(mg_dir, us_dir, mg_side_str, us_side_str, modality, cfg) -> list[Path] | None:
    if modality == "MG":
        return select_mg(mg_dir, mg_side_str, cfg)
    return select_us(us_dir, us_side_str, cfg)


def is_valid(record, cfg: dict[str, Any]) -> bool:
    """A patient is usable only if the selection succeeds for both modalities."""
    mg_ok = select_mg(record.mg_dir, record.mg_finding_side, cfg) is not None
    us_ok = select_us(record.us_dir, record.us_finding_side, cfg) is not None
    return mg_ok and us_ok
