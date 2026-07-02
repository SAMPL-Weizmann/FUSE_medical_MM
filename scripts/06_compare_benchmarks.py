"""Compare two benchmark.json files (e.g. with-TCI vs without-TCI).

Prints Δ = (B minus A) per method x set for accuracy and balanced accuracy.

Usage:
    python scripts/06_compare_benchmarks.py \
        artifacts/fuse/lambda_0.0/benchmark.json  artifacts/fuse/lambda_1.0/benchmark.json \
        --labels no_tci with_tci
"""

from __future__ import annotations

import argparse
import json

SETS = ["labeled", "unlabeled", "test"]


def _load(path):
    return json.load(open(path, encoding="utf-8"))["methods"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--metric", default="balanced_acc", choices=["acc", "balanced_acc"])
    args = ap.parse_args()
    A, B = _load(args.a), _load(args.b)
    la, lb = args.labels
    mt = args.metric

    print(f"Δ = {lb} − {la}   (metric: {mt})")
    print(f"{'method':16s} " + " ".join(f"{s+' '+la:>12s} {s+' '+lb:>10s} {'Δ':>7s}" for s in SETS))
    for name in A:
        if "skipped" in A[name] or name not in B or "skipped" in B[name]:
            continue
        row = f"{name:16s} "
        for s in SETS:
            va, vb = A[name][s][mt], B[name][s][mt]
            row += f" {va:12.3f} {vb:10.3f} {vb-va:+7.3f}"
        print(row)


if __name__ == "__main__":
    main()
