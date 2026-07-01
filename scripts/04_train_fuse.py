"""FUSE stage-1: joint verifier training (L_CE + alpha*L_NCL + lambda*L_TCI).

Usage:
    python scripts/04_train_fuse.py [--config configs/fuse.yaml] [--device cpu]
Outputs under artifacts/fuse/: fuse_model.pt, verifier_scores_{labeled,unlabeled,test}.npz,
train_summary.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuse_mm.fuse import load_fuse_config  # noqa: E402
from fuse_mm.fuse.train import train_fuse  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_fuse_config(args.config)
    print("=" * 68)
    print("FUSE stage-1 joint training")
    print(f"  warm={cfg['train']['warmstart_epochs']} joint={cfg['train']['joint_epochs']} "
          f"alpha={cfg['train']['alpha_ncl']} lambda={cfg['train']['lambda_tci']}")
    print("=" * 68)

    s = train_fuse(cfg, device=args.device)

    print("-" * 68)
    print(f"verifiers: {s['n_verifiers']}   final TCI (unlabeled): {s['tci_unlabeled_final']:.4f}")
    print("per-verifier AUC  (labeled / test):")
    for pv in sorted(s["per_verifier"], key=lambda r: r["auc_test"], reverse=True):
        print(f"  {pv['name']:26s}  {pv['auc_labeled']:.3f} / {pv['auc_test']:.3f}")
    print(f"\nwrote artifacts/fuse/  (model + verifier scores + summary)")


if __name__ == "__main__":
    main()
