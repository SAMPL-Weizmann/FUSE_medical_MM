"""FUSE_medical_MM — multimodal (MG + US) breast-imaging data layer.

Public surface:
    load_config()                  -> dict from configs/data.yaml
    read_label_table(cfg)          -> list[PatientRecord]
    build_cohort(cfg)              -> paired patients present on disk, with labels
    make_splits(cfg)               -> stratified Labeled/Unlabeled/Test split
    PatientLoader                  -> lazy per-patient .npy access
"""

from .config import load_config, project_root
from .labels import PatientRecord, read_label_table, build_cohort
from .splits import make_splits, load_splits

__all__ = [
    "load_config",
    "project_root",
    "PatientRecord",
    "read_label_table",
    "build_cohort",
    "make_splits",
    "load_splits",
]
