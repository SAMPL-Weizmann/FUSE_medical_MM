"""Extract and save frozen-backbone feature vectors.

For each (split, modality, backbone) we run every patient image through the
frozen backbone and save a single .npz holding:
    X            (n_images, feat_dim) float32   pooled features
    patient_ids  (n_images,)         str        owning patient id
    labels       (n_images,)         int        patient label (active scheme)
    paths        (n_images,)         str        source .npy path

Per-image features are kept (not pre-aggregated), so downstream head training
is free to pool per patient, weight views, or combine backbones however it
likes. Saved under artifacts/features/<split>/<modality>/<backbone>.npz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..dataset import PatientLoader
from ..labels import PatientRecord
from .backbones import BackboneSpec, build_extractor


def _records_from_split(cfg: dict[str, Any], split: dict[str, Any],
                        set_name: str) -> list[PatientRecord]:
    ds = cfg["dataset"]
    mg_root = Path(ds["root"]) / ds["mg_subdir"]
    us_root = Path(ds["root"]) / ds["us_subdir"]
    recs = []
    for p in split[set_name]:
        mg = mg_root / p["mg_id"]
        us = us_root / p["us_id"]
        recs.append(PatientRecord(
            mg_id=p["mg_id"], us_id=p["us_id"], label=p["label"],
            raw_label=p["raw_label"], group="",
            mg_dir=mg if mg.is_dir() else None,
            us_dir=us if us.is_dir() else None,
        ))
    return recs


def extract_split_modality(cfg, split, set_name: str, modality: str,
                           spec: BackboneSpec, device: str,
                           batch_size: int = 32) -> Path:
    import torch
    from PIL import Image

    ex = build_extractor(spec, device)
    transform, feat_dim = ex.transform, ex.feat_dim
    records = _records_from_split(cfg, split, set_name)

    feats: list[np.ndarray] = []
    pids: list[str] = []
    labels: list[int] = []
    paths: list[str] = []

    batch, meta = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        y = ex.embed(x).float().cpu().numpy()
        feats.append(y)
        for pid, lab, pth in meta:
            pids.append(pid); labels.append(lab); paths.append(pth)
        batch.clear(); meta.clear()

    for rec in records:
        pdir = rec.mg_dir if modality == "MG" else rec.us_dir
        pid = rec.mg_id if modality == "MG" else rec.us_id
        if pdir is None:
            continue
        loader = PatientLoader(pdir, modality)
        for fp, img in zip(loader.files, loader.images()):
            tensor = transform(Image.fromarray(img))   # (3, H, W)
            batch.append(tensor)
            meta.append((pid, rec.label, str(fp)))
            if len(batch) >= batch_size:
                flush()
    flush()

    X = (np.concatenate(feats, axis=0) if feats
         else np.zeros((0, feat_dim), np.float32))
    out_dir = Path(cfg["artifacts"]["features_dir"]) / set_name / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}.npz"
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        patient_ids=np.array(pids),
        labels=np.array(labels, dtype=np.int64),
        paths=np.array(paths),
        feat_dim=np.array(feat_dim),
        backbone=np.array(spec.name),
        model_id=np.array(spec.model_id),
        objective=np.array(spec.objective),
        domain=np.array(spec.domain),
    )
    return out_path
