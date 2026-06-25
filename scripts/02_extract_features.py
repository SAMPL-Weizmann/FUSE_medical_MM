"""Extract frozen-backbone feature vectors for the chosen splits/modalities.

REQUIRES a torch-capable env (Python 3.11/3.12). See requirements.txt.

Usage:
    .venv-torch/Scripts/python.exe scripts/02_extract_features.py \
        --sets labeled --modalities MG US --backbones resnet50 efficientnet_b0

Defaults: all backbones in the registry, both modalities, the 'labeled' set
(the set used to train classifier heads).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuse_mm import load_config, load_splits  # noqa: E402
from fuse_mm.features.backbones import get_backbones  # noqa: E402
from fuse_mm.features.extract import extract_split_modality  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sets", nargs="+", default=["labeled"],
                    choices=["labeled", "unlabeled", "test"])
    ap.add_argument("--modalities", nargs="+", default=["MG", "US"],
                    choices=["MG", "US"])
    ap.add_argument("--backbones", nargs="+", default=None,
                    help="subset of registry names; default = all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if omitted)")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    split = load_splits(cfg)
    specs = get_backbones(args.backbones)

    print(f"device={device}  sets={args.sets}  modalities={args.modalities}")
    print(f"backbones={[s.name for s in specs]}")
    for set_name in args.sets:
        for modality in args.modalities:
            for spec in specs:
                print(f"-> {set_name}/{modality}/{spec.name} ...", flush=True)
                out = extract_split_modality(
                    cfg, split, set_name, modality, spec, device,
                    batch_size=args.batch_size,
                )
                print(f"   saved {out}")


if __name__ == "__main__":
    main()
