"""Train + CV-evaluate classifier heads for every (modality, backbone).

Driven entirely by configs/train.yaml. Run different experiments by editing the
config (aggregation.mode, head.type, ...) — results are keyed by that combo and
never overwrite across configs.

Usage:
    .venv/Scripts/python.exe scripts/03_train_heads.py [--config configs/train.yaml]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuse_mm.heads import load_train_config, run_cv  # noqa: E402
from fuse_mm.heads.config import run_tag  # noqa: E402


def discover_backbones(features_dir, feature_set, modality) -> list[str]:
    d = Path(features_dir) / feature_set / modality
    return sorted(p.stem for p in d.glob("*.npz") if not p.stem.endswith("_oof"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu", help="cpu|cuda (heads are tiny; cpu is fine)")
    args = ap.parse_args()

    cfg = load_train_config(args.config)
    tag = run_tag(cfg)
    exp = cfg["experiment"]

    print("=" * 70)
    print(f"head training  |  {tag}  |  feature_set={exp['feature_set']}")
    print(f"aggregation={cfg['aggregation']['mode']}  head={cfg['head']}")
    print("=" * 70)

    rows = []
    for modality in exp["modalities"]:
        backbones = exp["backbones"] or discover_backbones(
            cfg["io"]["features_dir"], exp["feature_set"], modality)
        for bb in backbones:
            s = run_cv(cfg, modality, bb, device=args.device)
            m, sd = s["mean"], s["std"]
            rows.append((modality, bb, m, sd))
            print(f"{modality:3s} {bb:20s}  "
                  f"AUC={m['auc']:.3f}±{sd['auc']:.3f}  "
                  f"AP={m['ap']:.3f}  bACC={m['balanced_acc']:.3f}  F1={m['f1']:.3f}")

    # combined summary table sorted by AUC within modality
    summary_path = Path(cfg["io"]["results_dir"]) / tag / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"modality": mod, "backbone": bb, "mean": m, "std": sd}
             for mod, bb, m, sd in rows], f, indent=2)

    print("-" * 70)
    print("ranked by AUC:")
    for modality in exp["modalities"]:
        sub = sorted([r for r in rows if r[0] == modality],
                     key=lambda r: r[2]["auc"], reverse=True)
        print(f"  [{modality}]")
        for _, bb, m, sd in sub:
            print(f"    {bb:20s} AUC={m['auc']:.3f}±{sd['auc']:.3f}")
    print(f"\nwrote results under {Path(cfg['io']['results_dir']) / tag}")


if __name__ == "__main__":
    main()
