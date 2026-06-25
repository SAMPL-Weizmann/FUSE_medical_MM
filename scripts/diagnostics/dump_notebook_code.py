"""Print a Jupyter notebook's code (and optionally markdown) cells, skipping the
huge embedded outputs that make these clinical notebooks hard to read.

Usage:
    python scripts/diagnostics/dump_notebook_code.py "<path to .ipynb>" [--md] [--grep WORD]
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--md", action="store_true", help="also show markdown cells")
    ap.add_argument("--grep", default=None, help="only cells containing this (case-insensitive)")
    args = ap.parse_args()

    nb = json.load(open(args.notebook, encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" or (args.md and cell["cell_type"] == "markdown"):
            src = "".join(cell["source"]).rstrip()
            if not src:
                continue
            if args.grep and args.grep.lower() not in src.lower():
                continue
            print(f"\n===== {cell['cell_type'].upper()} CELL {i} =====")
            print(src)


if __name__ == "__main__":
    main()
