"""Build the Labeled / Unlabeled / Test split and write artifacts/splits.json.

Usage:
    .venv/Scripts/python.exe scripts/01_make_splits.py [--config configs/data.yaml] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make `src` importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuse_mm import build_cohort, load_config, make_splits  # noqa: E402
from fuse_mm.labels import label_distribution, read_label_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to data.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the split summary without writing splits.json")
    args = ap.parse_args()

    cfg = load_config(args.config)

    all_records = read_label_table(cfg)
    cohort = build_cohort(cfg)
    on_disk_both = sum(r.has_mg and r.has_us for r in all_records)

    print("=" * 64)
    print("FUSE_medical_MM — split builder")
    print("=" * 64)
    print(f"label scheme        : {cfg['labels']['scheme']} "
          f"{cfg['labels']['class_names']}")
    print(f"table records (kept): {len(all_records)}")
    print(f"  with both MG+US   : {on_disk_both}")
    print(f"cohort (split input): {len(cohort)}  "
          f"dist={label_distribution(cohort)}")
    print(f"ratios              : {cfg['split']['ratios']}  "
          f"seed={cfg['split']['seed']}")

    splits = make_splits(cfg, write=not args.dry_run)
    meta = splits["meta"]
    print("-" * 64)
    for name in ("labeled", "unlabeled", "test"):
        print(f"{name:>10}: {meta['set_sizes'][name]:>5} patients   "
              f"dist={meta['set_label_distributions'][name]}")
    print("-" * 64)
    if args.dry_run:
        print("dry-run: splits.json NOT written")
    else:
        print(f"wrote {cfg['artifacts']['splits_path']}")


if __name__ == "__main__":
    main()
