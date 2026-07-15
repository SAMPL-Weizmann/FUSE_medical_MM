"""Read the paired label table and build the on-disk patient cohort.

The clean_mm.xlsx table has one row per patient with columns:
    idx, MG, US, MG finding side, US finding side, Label, Group, ...
where MG / US are the patient-folder ids and Label is:
    0 = normal (takin), 1 = benign (shafir), 2 = malignant (BC)

We parse the .xlsx with the standard library only (it is a zip of XML), so the
data layer has no pandas/openpyxl dependency.
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# Column headers we rely on (resolved by name, not by position, for safety).
_COL_MG = "MG"
_COL_US = "US"
_COL_LABEL = "Label"
_COL_GROUP = "Group"
_COL_MG_SIDE = "MG finding side"
_COL_US_SIDE = "US finding side"


# --------------------------------------------------------------------------- #
# Minimal .xlsx reader (first worksheet only)                                  #
# --------------------------------------------------------------------------- #
def _col_to_idx(cell_ref: str) -> int:
    """'B7' -> 1 (zero-based column index)."""
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


_CELL_RE = re.compile(
    r'<c r="([A-Z]+\d+)"(?:\s+s="\d+")?(?:\s+t="(\w+)")?\s*'
    r'(?:/>|>(?:<v>(.*?)</v>|<is><t[^>]*>(.*?)</t></is>)?</c>)'
)


def read_worksheet(xlsx_path: str | os.PathLike) -> list[dict[str, str]]:
    """Return the first worksheet as a list of row dicts keyed by header name."""
    with zipfile.ZipFile(xlsx_path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            sx = z.read("xl/sharedStrings.xml").decode("utf-8", "replace")
            for si in re.findall(r"<si>(.*?)</si>", sx, re.S):
                shared.append("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S)))
        sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8", "replace")

    rows_xml = re.findall(r"<row[^>]*>(.*?)</row>", sheet, re.S)

    def decode(row_xml: str) -> dict[int, str]:
        out: dict[int, str] = {}
        for ref, typ, v, inline in _CELL_RE.findall(row_xml):
            val = inline if inline else v
            if typ == "s" and val.isdigit():
                idx = int(val)
                val = shared[idx] if idx < len(shared) else val
            out[_col_to_idx(ref)] = val.strip()
        return out

    if not rows_xml:
        return []

    header = decode(rows_xml[0])
    width = max(header) + 1
    names = [header.get(i, f"col{i}") for i in range(width)]

    records: list[dict[str, str]] = []
    for rx in rows_xml[1:]:
        cells = decode(rx)
        records.append({names[i]: cells.get(i, "") for i in range(width)})
    return records


# --------------------------------------------------------------------------- #
# Label mapping                                                                #
# --------------------------------------------------------------------------- #
def map_label(raw: int, scheme: str) -> int | None:
    """Map native label (0/1/2) to the configured scheme. None => drop patient."""
    if scheme == "multiclass":
        return raw
    if scheme == "binary_abnormal":          # normal vs (benign+malignant)
        return 0 if raw == 0 else 1
    if scheme == "binary_malignant":         # benign vs malignant; DROPS normal
        if raw == 0:
            return None
        return 0 if raw == 1 else 1
    if scheme == "binary_malignant_vs_rest":  # (normal+benign) vs malignant; keeps ALL
        return 1 if raw == 2 else 0
    raise ValueError(f"unknown label scheme {scheme!r}")


@dataclass(frozen=True)
class PatientRecord:
    """One paired patient: ids in both modalities + mapped label."""
    mg_id: str
    us_id: str
    label: int            # mapped to the active scheme
    raw_label: int        # original 0/1/2 from the table
    group: str            # original Train/Validation/Test tag (informational)
    mg_dir: Path | None   # on-disk MG folder, or None if missing
    us_dir: Path | None   # on-disk US folder, or None if missing
    mg_finding_side: str = ""   # 'LT'/'RT'/'ALL'/... drives MG view selection
    us_finding_side: str = ""

    @property
    def has_mg(self) -> bool:
        return self.mg_dir is not None

    @property
    def has_us(self) -> bool:
        return self.us_dir is not None


def read_label_table(cfg: dict[str, Any]) -> list[PatientRecord]:
    """Parse the paired table into PatientRecords (no disk check yet)."""
    ds = cfg["dataset"]
    scheme = cfg["labels"]["scheme"]
    mg_root = Path(ds["root"]) / ds["mg_subdir"]
    us_root = Path(ds["root"]) / ds["us_subdir"]

    records: list[PatientRecord] = []
    for row in read_worksheet(ds["label_table"]):
        mg_id, us_id = row.get(_COL_MG, ""), row.get(_COL_US, "")
        raw = row.get(_COL_LABEL, "")
        if not mg_id or not us_id or not raw.lstrip("-").isdigit():
            continue
        raw_label = int(raw)
        mapped = map_label(raw_label, scheme)
        if mapped is None:
            continue
        mg_dir = mg_root / mg_id
        us_dir = us_root / us_id
        records.append(
            PatientRecord(
                mg_id=mg_id,
                us_id=us_id,
                label=mapped,
                raw_label=raw_label,
                group=row.get(_COL_GROUP, ""),
                mg_dir=mg_dir if mg_dir.is_dir() else None,
                us_dir=us_dir if us_dir.is_dir() else None,
                mg_finding_side=row.get(_COL_MG_SIDE, ""),
                us_finding_side=row.get(_COL_US_SIDE, ""),
            )
        )
    return records


def build_cohort(cfg: dict[str, Any]) -> list[PatientRecord]:
    """Records usable for the split.

    A patient must be present on disk (per require_both_modalities) AND pass the
    scan-selection check — e.g. with selection.mg.mode == canonical_views +
    require_all_views, patients missing the 4 MG views are dropped (this is the
    same culling the prior team did).
    """
    from .selection import is_valid  # local import avoids cycle at module load

    require_both = cfg["split"].get("require_both_modalities", True)
    out: list[PatientRecord] = []
    for r in read_label_table(cfg):
        present = (r.has_mg and r.has_us) if require_both else (r.has_mg or r.has_us)
        if not present:
            continue
        if not is_valid(r, cfg):
            continue
        out.append(r)
    return out


def label_distribution(records: list[PatientRecord]) -> dict[int, int]:
    dist: dict[int, int] = {}
    for r in records:
        dist[r.label] = dist.get(r.label, 0) + 1
    return dict(sorted(dist.items()))
