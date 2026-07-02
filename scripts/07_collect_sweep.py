"""Collect a lambda sweep: read artifacts/fuse/lambda_*/benchmark.json and print
each method's metric across lambda (the TCI curve), per set. Flags each method's
best lambda.

Usage:
    python scripts/07_collect_sweep.py [--dir artifacts/fuse] [--metric balanced_acc]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

SETS = ["unlabeled", "test"]     # held-out sets (labeled is in-sample/overfit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/fuse")
    ap.add_argument("--metric", default="balanced_acc", choices=["acc", "balanced_acc"])
    args = ap.parse_args()

    runs = {}
    for path in glob.glob(os.path.join(args.dir, "lambda_*", "benchmark.json")):
        m = re.search(r"lambda_([0-9.]+)", path)
        if m:
            runs[float(m.group(1))] = json.load(open(path, encoding="utf-8"))["methods"]
    if not runs:
        print(f"no lambda_*/benchmark.json under {args.dir}")
        return

    lams = sorted(runs)
    methods = [k for k, v in runs[lams[0]].items() if "skipped" not in v]

    print(f"lambda sweep  (metric: {args.metric})   lambdas: {lams}")
    for s in SETS:
        print(f"\n=== {s} ===")
        head = "method".ljust(16) + " ".join(f"{l:>7}" for l in lams) + "   best_lambda"
        print(head)
        for name in methods:
            vals = [runs[l].get(name, {}).get(s, {}).get(args.metric, float("nan")) for l in lams]
            best = lams[int(max(range(len(vals)), key=lambda i: vals[i]))]
            cells = " ".join(f"{v:7.3f}" for v in vals)
            print(f"{name:16s} {cells}   best_lam={best}")


if __name__ == "__main__":
    main()
