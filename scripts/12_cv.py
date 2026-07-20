"""Cross-validate the full FUSE pipeline over a lambda list.

Rotation: S_L = fold i, Test = the next `--n-test` folds (circular), S_U = rest;
retrains the verifiers per fold, benchmarks all methods on S_L/S_U/Test,
aggregates per-fold mean/std + pooled out-of-fold.
  * default  10 folds, n_test=1  -> 10/80/10  (S_L/S_U/Test), cv_folds.json
  * 20 folds, n_test=2           ->  5/85/10, cv_folds_20.json

Usage:
    python scripts/12_cv.py [--lambdas 0 0.1 0.2 0.3 0.5 1] [--folds 10] [--n-test 1]
    # 20-fold 5/85/10:
    python scripts/12_cv.py --folds 20 --n-test 2 --out-dir artifacts/reports/cv20/...
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
    ap.add_argument("--n-test", type=int, default=1,
                    help="number of folds used as Test each iteration (2 for the "
                         "20-fold 5/85/10 split)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None,
                    help="where to write cv_results.json/cv_summary.csv "
                         "(default artifacts/reports/cv/). Use a per-lambda dir "
                         "when fanning out one lambda per job.")
    ap.add_argument("--threshold-rules", nargs="+", default=["fixed"],
                    choices=["fixed", "prevalence", "youden", "target_sensitivity"],
                    help="decision-threshold rule(s) for turning soft scores into "
                         "0/1 calls. Default 'fixed' (0.5) = unchanged. Extra rules "
                         "cost ~nothing (computed in the same pass) and each writes "
                         "to <out-dir>/<rule>/; 'fixed' alone keeps the old path.")
    ap.add_argument("--prevalence", type=float, default=None,
                    help="positive rate for the 'prevalence' rule (default: the "
                         "labeled-set base rate). Pass e.g. 0.073 to stay label-free.")
    ap.add_argument("--target-sensitivity", type=float, default=0.95,
                    help="target recall for the 'target_sensitivity' rule")
    ap.add_argument("--dump-predictions", action="store_true",
                    help="also save pooled out-of-fold soft scores to "
                         "<out-dir>/cv_pooled_predictions.npz (input for ROC/PR, "
                         "scripts/20_roc_pr.py). Off by default.")
    ap.add_argument("--remake-folds", action="store_true", help="regenerate cv_folds.json")
    args = ap.parse_args()

    cfg = load_fuse_config(args.config)
    if args.remake_folds:
        make_cv_folds(n_folds=args.folds)
    else:
        try:
            load_cv_folds(n_folds=args.folds)
        except FileNotFoundError:
            make_cv_folds(n_folds=args.folds)

    print("=" * 70)
    print(f"FUSE CV  |  folds={args.folds}  n_test={args.n_test}  lambdas={args.lambdas}")
    print(f"         |  threshold_rules={args.threshold_rules}")
    print("=" * 70)
    run_cv(cfg, args.lambdas, n_folds=args.folds, n_test=args.n_test,
           device=args.device, out_dir=args.out_dir,
           threshold_rules=args.threshold_rules, prevalence=args.prevalence,
           target_sensitivity=args.target_sensitivity,
           dump_predictions=args.dump_predictions)


if __name__ == "__main__":
    main()
