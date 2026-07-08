"""Generate plots + tables from the benchmark/sweep results.

Reads artifacts/fuse/lambda_*/benchmark.json and writes to artifacts/reports/:
  - sweep_<set>_<metric>.png      method vs lambda line plots
  - methods_summary.csv           (method, lambda, set) -> acc, balanced_acc
  - per_verifier.csv              (verifier, lambda, set) -> acc, auc, sens, spec, bal_acc
  - verifier_auc_<set>_lambda_<L>.png   per-verifier AUC bar chart
  - report.md                     human-readable summary (FUSE curve + top verifiers)

Usage:
    python scripts/08_report.py [--dir artifacts/fuse] [--out artifacts/reports]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SETS = ["labeled", "unlabeled", "test"]
HELD_OUT = ["unlabeled", "test"]
METRICS = ["acc", "balanced_acc"]


def load_runs(fuse_dir):
    runs = {}
    for path in glob.glob(os.path.join(fuse_dir, "lambda_*", "benchmark.json")):
        m = re.search(r"lambda_([0-9.]+)", path)
        if m:
            runs[float(m.group(1))] = json.load(open(path, encoding="utf-8"))
    return dict(sorted(runs.items()))


# --------------------------------------------------------------------------- #
def plot_sweep(runs, out):
    lams = sorted(runs)
    methods = [k for k, v in runs[lams[0]]["methods"].items() if "skipped" not in v]
    for s in HELD_OUT:
        for metric in METRICS:
            plt.figure(figsize=(7, 5))
            for name in methods:
                ys = [runs[l]["methods"][name][s][metric] for l in lams]
                style = "-o" if name == "fuse" else "--."[0:1] + "o"
                lw = 2.5 if name == "fuse" else 1.2
                plt.plot(lams, ys, marker="o", linewidth=lw, label=name)
            plt.xlabel("lambda_tci"); plt.ylabel(metric)
            plt.title(f"{metric} vs TCI weight  ({s})")
            plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2)
            fp = os.path.join(out, f"sweep_{s}_{metric}.png")
            plt.tight_layout(); plt.savefig(fp, dpi=130); plt.close()


def write_methods_csv(runs, out):
    fp = os.path.join(out, "methods_summary.csv")
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "category", "lambda", "set", "acc", "balanced_acc", "auc"])
        for lam, run in runs.items():
            for name, r in run["methods"].items():
                if "skipped" in r:
                    continue
                for s in SETS:
                    w.writerow([name, r["category"], lam, s,
                                f"{r[s]['acc']:.4f}", f"{r[s]['balanced_acc']:.4f}",
                                f"{r[s]['auc']:.4f}"])


def write_verifier_csv(runs, out):
    fp = os.path.join(out, "per_verifier.csv")
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["verifier", "lambda", "set", "acc", "auc",
                    "sensitivity", "specificity", "balanced_acc"])
        for lam, run in runs.items():
            for vname, sets in run["per_verifier"].items():
                for s in SETS:
                    st = sets[s]
                    w.writerow([vname, lam, s, f"{st['acc']:.4f}", f"{st['auc']:.4f}",
                                f"{st['sensitivity']:.4f}", f"{st['specificity']:.4f}",
                                f"{st['balanced_acc']:.4f}"])


def plot_verifier_bars(runs, out, lam, s="test"):
    if lam not in runs:
        return
    pv = runs[lam]["per_verifier"]
    items = sorted(((n, d[s]["auc"]) for n, d in pv.items()), key=lambda x: x[1])
    names, aucs = zip(*items)
    plt.figure(figsize=(8, max(6, len(names) * 0.28)))
    plt.barh(range(len(names)), aucs, color="#2a9d8f")
    plt.yticks(range(len(names)), names, fontsize=7)
    plt.axvline(0.5, color="k", ls="--", lw=0.8)
    plt.xlabel(f"AUC ({s})"); plt.title(f"per-verifier AUC  (lambda={lam}, {s})")
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"verifier_auc_{s}_lambda_{lam}.png"), dpi=130)
    plt.close()


def write_report_md(runs, out):
    lams = sorted(runs)
    lines = ["# FUSE-Medical-MM — results report", ""]
    lines.append(f"lambda sweep: {lams}\n")
    for s in HELD_OUT:
        lines.append(f"## {s} — balanced accuracy vs lambda\n")
        methods = [k for k, v in runs[lams[0]]["methods"].items() if "skipped" not in v]
        lines.append("| method | " + " | ".join(f"λ={l}" for l in lams) + " |")
        lines.append("|" + "---|" * (len(lams) + 1))
        for name in methods:
            vals = " | ".join(f"{runs[l]['methods'][name][s]['balanced_acc']:.3f}" for l in lams)
            lines.append(f"| {name} | {vals} |")
        lines.append("")
    # top verifiers at lambda=min
    lam0 = lams[0]
    pv = runs[lam0]["per_verifier"]
    top = sorted(pv.items(), key=lambda x: -x[1]["test"]["auc"])[:10]
    lines.append(f"## Top-10 verifiers by test AUC (lambda={lam0})\n")
    lines.append("| verifier | test AUC | test sens | test spec | test bAcc |")
    lines.append("|---|---|---|---|---|")
    for n, d in top:
        t = d["test"]
        lines.append(f"| {n} | {t['auc']:.3f} | {t['sensitivity']:.3f} | "
                     f"{t['specificity']:.3f} | {t['balanced_acc']:.3f} |")
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/fuse")
    ap.add_argument("--out", default="artifacts/reports/benchmark")
    args = ap.parse_args()

    runs = load_runs(args.dir)
    if not runs:
        print(f"no lambda_*/benchmark.json under {args.dir}"); return
    os.makedirs(args.out, exist_ok=True)

    plot_sweep(runs, args.out)
    write_methods_csv(runs, args.out)
    write_verifier_csv(runs, args.out)
    for lam in (min(runs), max(runs)):
        plot_verifier_bars(runs, args.out, lam, "test")
    write_report_md(runs, args.out)

    print(f"wrote report to {args.out}/ :")
    for f in sorted(os.listdir(args.out)):
        print("  ", f)


if __name__ == "__main__":
    main()
