"""Benchmark FUSE + baselines on a verifier-scores directory (one lambda run).

Usage:
    python scripts/05_benchmark.py artifacts/fuse/lambda_0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuse_mm.bench import run_benchmark  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_dir", help="dir with verifier_scores_{labeled,unlabeled,test}.npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_benchmark(args.scores_dir, args.out)


if __name__ == "__main__":
    main()
