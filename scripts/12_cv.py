"""Cross-validate the full FUSE pipeline over a lambda list.

10-fold rotation (S_L = fold i, Test = fold i+1 circular, S_U = rest); retrains
the verifiers per fold, benchmarks all methods on S_L/S_U/Test, aggregates
per-fold mean/std + pooled out-of-fold. Outputs artifacts/reports/cv/.

Usage:
    python scripts/12_cv.py [--lambdas 0 0.1 0.2 0.3 0.5 1] [--folds 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuse_mm.cv import make_cv_folds, run_cv  # noqa: E402
from fuse_mm.cv.folds import load_cv_folds  # noqa: E402
from fuse_mm.fuse import load_fuse_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0, 0.1, 0.2, 0.3, 0.5, 1])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None,
                    help="where to write cv_results.json/cv_summary.csv "
                         "(default artifacts/reports/cv/). Use a per-lambda dir "
                         "when fanning out one lambda per job.")
    ap.add_argument("--remake-folds", action="store_true", help="regenerate cv_folds.json")
    args = ap.parse_args()

    cfg = load_fuse_config(args.config)
    if args.remake_folds:
        make_cv_folds(n_folds=args.folds)
    else:
        try:
            load_cv_folds()
        except FileNotFoundError:
            make_cv_folds(n_folds=args.folds)

    print("=" * 70)
    print(f"FUSE CV  |  folds={args.folds}  lambdas={args.lambdas}")
    print("=" * 70)
    run_cv(cfg, args.lambdas, n_folds=args.folds, device=args.device, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
