"""Merge per-lambda CV outputs from the fan-out into one combined report.

The fan-out (scripts/wexac/cv.lsf, one job per lambda) writes each lambda to
its own dir: artifacts/reports/cv/lambda_<L>/cv_results.json. This collects them
into the single artifacts/reports/cv/cv_results.json + cv_summary.csv that the
single-process run would have produced.

Usage:
    python scripts/13_cv_merge.py [--base artifacts/reports/cv]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuse_mm.cv.run import _print, _write_csv  # noqa: E402
from fuse_mm.fuse import load_fuse_config  # noqa: E402


def _default_base(cfg) -> str:
    return os.path.join(os.path.dirname(cfg["io"]["out_dir"]), "reports", "cv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", default=None,
                    help="dir containing the lambda_* subdirs (default artifacts/reports/cv)")
    args = ap.parse_args()

    cfg = load_fuse_config(args.config)
    base = args.base or _default_base(cfg)

    part_files = sorted(glob.glob(os.path.join(base, "lambda_*", "cv_results.json")))
    if not part_files:
        raise SystemExit(f"no lambda_*/cv_results.json found under {base}/ — nothing to merge")

    merged, n_folds, n_test = {}, None, None
    for pf in part_files:
        with open(pf, encoding="utf-8") as fh:
            part = json.load(fh)
        if n_folds is None:
            n_folds, n_test = part["n_folds"], part.get("n_test", 1)
        elif n_folds != part["n_folds"]:
            raise SystemExit(f"n_folds mismatch: {pf} has {part['n_folds']}, expected {n_folds}")
        for lam, res in part["results"].items():
            if lam in merged:
                raise SystemExit(f"duplicate lambda {lam} (seen twice across parts, e.g. {pf})")
            merged[lam] = res

    # numeric lambda order for stable output
    results = {k: merged[k] for k in sorted(merged, key=float)}
    lambdas = [float(k) for k in results]

    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump({"n_folds": n_folds, "n_test": n_test, "lambdas": lambdas,
                   "results": results}, f, indent=2)
    _write_csv(results, base)
    _print(results)
    print(f"\nmerged {len(results)} lambdas from {len(part_files)} parts "
          f"-> {base}/cv_results.json + cv_summary.csv")


if __name__ == "__main__":
    main()
