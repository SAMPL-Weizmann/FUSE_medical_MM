"""US annotation-shortcut probe.

Tests whether burned-in annotations (calipers / measurement text / color-Doppler)
alone explain the high US AUC. Real B-mode ultrasound is grayscale (R==G==B per
pixel); annotations are colored or near-pure-white. We turn each US frame into a
tiny "annotation-content" feature vector and ask whether THAT alone predicts
normal-vs-abnormal, using the same labeled split + patient-level CV as the
backbone heads.

If the annotation-only AUC approaches the backbone US AUC (~0.97), the verifiers
are likely reading the radiologist's marks, not the tissue.

Usage (WEXAC):  python scripts/diagnostics/us_annotation_probe.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fuse_mm import load_config, load_splits  # noqa: E402
from fuse_mm.dataset import PatientLoader  # noqa: E402
from fuse_mm.heads.metrics import (  # noqa: E402
    binary_metrics, roc_auc, standardize_apply, standardize_fit, stratified_kfold,
)
from fuse_mm.heads.models import train_head  # noqa: E402

FEATURES = ["color_frac", "white_frac"]


def frame_stats(img: np.ndarray, color_tau: int, white_tau: int) -> tuple[float, float]:
    """img: (H,W,3) uint8 -> (colored-pixel fraction, near-white-pixel fraction)."""
    a = img.astype(np.int16)
    spread = a.max(axis=2) - a.min(axis=2)          # 0 on truly-gray pixels
    color_frac = float((spread > color_tau).mean())
    white_frac = float((a.min(axis=2) > white_tau).mean())
    return color_frac, white_frac


def build_features(cfg, split, color_tau, white_tau):
    ds = cfg["dataset"]
    us_root = Path(ds["root"]) / ds["us_subdir"]
    rows, y, pids = [], [], []
    for rec in split["labeled"]:
        loader = PatientLoader(us_root / rec["us_id"], "US")
        per_frame = [frame_stats(im, color_tau, white_tau) for im in loader.images()]
        if not per_frame:
            continue
        rows.append(np.mean(per_frame, axis=0))      # patient = mean over frames
        y.append(int(rec["label"]))
        pids.append(rec["us_id"])
    return np.array(rows, dtype=np.float32), np.array(y), np.array(pids)


def annotation_only_cv(X, y, seed=42, folds=5):
    cfg_head = {"type": "linear", "lr": 0.01, "weight_decay": 1e-4,
                "epochs": 300, "batch_size": 64, "class_weight": "balanced"}
    oof = np.full(len(y), np.nan)
    for tr, va in stratified_kfold(y, folds, seed):
        mu, sd = standardize_fit(X[tr])
        predict = train_head(standardize_apply(X[tr], mu, sd), y[tr], cfg_head, 2)
        oof[va] = predict(standardize_apply(X[va], mu, sd))[:, 0]  # single answer col
    return oof


def save_montage(cfg, split, X, pids, out_path, k=8):
    """Top-k most-annotated abnormal frames vs k least-annotated normal frames."""
    try:
        from PIL import Image
    except Exception:
        return
    ds = cfg["dataset"]
    us_root = Path(ds["root"]) / ds["us_subdir"]
    label = {r["us_id"]: int(r["label"]) for r in split["labeled"]}
    score = X.sum(axis=1)
    abn = [pids[i] for i in np.argsort(-score) if label[pids[i]] == 1][:k]
    nor = [pids[i] for i in np.argsort(score) if label[pids[i]] == 0][:k]

    def thumb(pid):
        img = next(PatientLoader(us_root / pid, "US").images())
        return np.array(Image.fromarray(img).resize((224, 224)))

    grid = np.zeros((2 * 224, k * 224, 3), np.uint8)
    for j, pid in enumerate(abn):
        grid[0:224, j * 224:(j + 1) * 224] = thumb(pid)
    for j, pid in enumerate(nor):
        grid[224:448, j * 224:(j + 1) * 224] = thumb(pid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    print(f"montage (top row=abnormal/most-annotated, bottom=normal): {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--color-tau", type=int, default=25)
    ap.add_argument("--white-tau", type=int, default=240)
    ap.add_argument("--no-montage", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    split = load_splits(cfg)
    X, y, pids = build_features(cfg, split, args.color_tau, args.white_tau)

    print("=" * 62)
    print(f"US annotation probe — {len(y)} labeled patients "
          f"(normal={int((y == 0).sum())}, abnormal={int((y == 1).sum())})")
    print("=" * 62)
    print(f"{'feature':12s} {'normal_mean':>12s} {'abn_mean':>10s} {'single AUC':>11s}")
    for j, name in enumerate(FEATURES):
        nm, am = X[y == 0, j].mean(), X[y == 1, j].mean()
        print(f"{name:12s} {nm:12.4f} {am:10.4f} {roc_auc(y, X[:, j]):11.3f}")

    oof = annotation_only_cv(X, y)
    m = binary_metrics(y, oof)
    print("-" * 62)
    print(f"annotation-ONLY linear head, 5-fold CV:  AUC={m['auc']:.3f}  "
          f"AP={m['ap']:.3f}  bACC={m['balanced_acc']:.3f}")
    print("(reference: best backbone US head AUC ~0.97)")
    print("Interpretation: AUC near the backbone score => shortcut is real.")

    if not args.no_montage:
        out = Path(cfg["artifacts"]["splits_path"]).parent / "diagnostics" / "us_annotation_montage.png"
        save_montage(cfg, split, X, pids, out)


if __name__ == "__main__":
    main()
