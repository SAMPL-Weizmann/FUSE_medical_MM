"""Run the benchmark: per-verifier stats + methods x sets table.

Consumes verifier scores saved by FUSE stage-1 (verifier_scores_{set}.npz with
v, y, verifier_names). Run on a lambda=0 scores dir now, lambda!=0 later, compare.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import methods as M
from .metrics import score_stats

SETS = ["labeled", "unlabeled", "test"]


def _load_scores(scores_dir: Path):
    V, Y, names = {}, {}, None
    for s in SETS:
        d = np.load(scores_dir / f"verifier_scores_{s}.npz", allow_pickle=True)
        V[s] = d["v"].astype(np.float64)
        Y[s] = d["y"].astype(int)
        names = list(d["verifier_names"].astype(str))
    return V, Y, names


def run_benchmark(scores_dir, out_path=None) -> dict:
    scores_dir = Path(scores_dir)
    V, Y, names = _load_scores(scores_dir)

    # per-verifier stats across sets
    per_verifier = {}
    for j, name in enumerate(names):
        per_verifier[name] = {s: score_stats(V[s][:, j], Y[s]) for s in SETS}

    # methods
    all_methods = {**M.UNSUPERVISED, **M.SUPERVISED, **M.CEILING}
    categories = ({n: "unsupervised" for n in M.UNSUPERVISED}
                  | {n: "supervised" for n in M.SUPERVISED}
                  | {n: "oracle" for n in M.CEILING})
    results = {}
    for name, fn in all_methods.items():
        try:
            scores = fn(V, Y)
            results[name] = {"category": categories[name],
                             **{s: score_stats(scores[s], Y[s]) for s in SETS}}
        except NotImplementedError as e:
            results[name] = {"category": categories[name], "skipped": str(e)}

    summary = {"scores_dir": str(scores_dir), "n_verifiers": len(names),
               "set_sizes": {s: int(len(Y[s])) for s in SETS},
               "methods": results, "per_verifier": per_verifier}

    _print(summary)
    out_path = Path(out_path) if out_path else scores_dir / "benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")
    return summary


def _print(summary):
    print("=" * 78)
    print(f"BENCHMARK  ({summary['scores_dir']})   sizes={summary['set_sizes']}")
    print("=" * 78)
    print("methods (accuracy / balanced-accuracy) per set:")
    print(f"{'method':16s} {'cat':12s} " + " ".join(f"{s:>18s}" for s in SETS))
    for name, r in summary["methods"].items():
        if "skipped" in r:
            print(f"{name:16s} {r['category']:12s}  [skipped: {r['skipped']}]")
            continue
        cells = " ".join(f"{r[s]['acc']:.3f}/{r[s]['balanced_acc']:.3f}".rjust(18) for s in SETS)
        print(f"{name:16s} {r['category']:12s} {cells}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_benchmark(args.scores_dir, args.out)
