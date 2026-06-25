# FUSE_medical_MM

Multimodal (mammography + ultrasound) breast-imaging pipeline: build a
Labeled / Unlabeled / Test split, extract frozen pretrained-backbone feature
vectors per modality, and (later) use the Unlabeled Pool for FUSE-based
estimation with Test patients as frozen verifiers.

## Data

Lives on the clinical share `Z:` (read-only). Paths are configured in
[`configs/data.yaml`](configs/data.yaml) — the single place to edit them.

| Modality | Layout | Array |
|---|---|---|
| MG | `MG_CLEAN_NPY/<id>/<VIEW>_Series.../*.npy` | `uint8 (H,W)`, multi-view, sizes vary |
| US | `US_CLEAN_NPY/<id>/Series0001/*.npy` | `uint8 (H,W)` or `(H,W,3)`, multi-frame, sizes vary |

Labels come from `clean_mm.xlsx` (paired MG↔US rows). Native labels:
`0 = normal (takin)`, `1 = benign (shafir)`, `2 = malignant (BC)`.

### Active configuration
- **Cohort:** paired patients with both MG and US on disk (~1831).
- **Label scheme:** `binary_abnormal` → `{0}=normal`, `{1,2}=abnormal`.
- **Split:** patient-level, stratified, `10 / 80 / 10` (Labeled / Unlabeled / Test), seed 42.

Change any of these in `configs/data.yaml` and re-run step 1.

## Repo layout

```
configs/data.yaml          # paths, label scheme, split ratios (single edit point)
src/fuse_mm/
  config.py                # load config + resolve paths
  labels.py                # parse clean_mm.xlsx (stdlib), map labels, build cohort
  dataset.py               # PatientLoader: lazy .npy load -> (H,W,3) uint8
  splits.py                # stratified Labeled/Unlabeled/Test split
  features/backbones.py    # backbone registry (timm / open_clip / hf verifiers)
  features/extract.py      # save per-image feature vectors
scripts/
  01_make_splits.py        # -> artifacts/splits.json
  02_extract_features.py   # -> artifacts/features/<set>/<modality>/<backbone>.npz
```

## Environment

Single venv (`.venv`, Python 3.14 — torch 2.12+ ships 3.14 wheels):
```bash
py -3.14 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**Data layer (Steps 1–4):**
```bash
.venv/Scripts/python.exe scripts/01_make_splits.py --dry-run   # preview
.venv/Scripts/python.exe scripts/01_make_splits.py             # write splits.json
```

**Feature extraction (Steps 5–6):**
```bash
.venv/Scripts/python.exe scripts/02_extract_features.py --sets labeled --modalities MG US
```
> The default PyPI torch wheel is CPU-only. For GPU, reinstall torch from the
> CUDA index (see [requirements.txt](requirements.txt)).

## Backbones (verifiers)

Registry in [src/fuse_mm/features/backbones.py](src/fuse_mm/features/backbones.py),
spread across two decorrelation axes — pre-training **objective** and **domain**.
Some are intentionally correlated (cut-and-try study); add/remove via
`--backbones`. Loaded behind one `Extractor` interface (timm / open_clip / hf).

| name | source | objective | domain | arch |
|---|---|---|---|---|
| `resnet50` | timm | supervised | natural | CNN |
| `efficientnet_b0` | timm | supervised | natural | light CNN |
| `vit_b16` | timm | supervised | natural | ViT-B/16 |
| `convnext_tiny` | timm | supervised | natural | modern CNN |
| `dinov2_vits14` | timm | self-supervised | natural | ViT-S/14 |
| `mae_vit_b16` | timm | self-supervised | natural | MAE ViT-B/16 |
| `biomedclip_vit_b16` | open_clip | vision-language | medical | ViT-B/16 |
| `rad_dino_vit_b14` | hf | self-supervised | medical | ViT-B/14 (CXR) |

```bash
# all 8 (default), or a subset:
.venv/Scripts/python.exe scripts/02_extract_features.py --sets labeled \
    --backbones resnet50 biomedclip_vit_b16 rad_dino_vit_b14
```

## Status
- [x] 1. Dataset structure understood + loaders
- [x] 2. Stratified Labeled/Unlabeled/Test split
- [x] 3. Ratios/scheme fully config-driven
- [x] 4. Feature-extraction code (frozen backbones → saved vectors) — *ready; runs once torch env exists*
- [ ] 5. Classifier heads per modality×backbone
- [ ] 6. FUSE estimation on Unlabeled Pool + TCI with Test verifiers
