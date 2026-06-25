"""Registry of frozen pretrained backbones (verifiers).

Backbones span two decorrelation axes — pre-training OBJECTIVE (supervised /
self-supervised / vision-language) and pre-training DOMAIN (natural / medical).
Some are deliberately correlated (e.g. resnet50 vs efficientnet_b0): the study
is cut-and-try, so we keep all of them and add/remove easily via `--backbones`.

Three loader backends are supported behind one uniform `Extractor` interface:
    "timm"      -> timm.create_model(num_classes=0)        -> pooled features
    "open_clip" -> open_clip image encoder                 -> encode_image()
    "hf"        -> transformers AutoModel (ViT/DINOv2)      -> CLS pooler_output

Each Extractor exposes:
    .transform(PIL.Image) -> Tensor (C, H, W)   model-specific preprocessing
    .embed(Tensor[B,C,H,W]) -> Tensor (B, D)     frozen feature vectors
    .feat_dim                                     D
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BackboneSpec:
    name: str          # short id used in output filenames / --backbones
    source: str        # "timm" | "open_clip" | "hf"
    model_id: str      # backend-specific model identifier
    objective: str     # supervised | self-supervised | vision-language
    domain: str        # natural | medical
    note: str = ""


# --------------------------------------------------------------------------- #
# Registry — edit this list to add/remove verifiers.                          #
# --------------------------------------------------------------------------- #
DEFAULT_BACKBONES: list[BackboneSpec] = [
    # --- natural-image, supervised --------------------------------------- #
    BackboneSpec("resnet50", "timm", "resnet50.a1_in1k",
                 "supervised", "natural", "CNN, IN-1k"),
    BackboneSpec("efficientnet_b0", "timm", "efficientnet_b0.ra_in1k",
                 "supervised", "natural", "light CNN, IN-1k"),
    BackboneSpec("vit_b16", "timm", "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                 "supervised", "natural", "ViT-B/16, IN-21k->IN-1k"),
    BackboneSpec("convnext_tiny", "timm", "convnext_tiny.fb_in22k_ft_in1k",
                 "supervised", "natural", "modern CNN, IN-22k->IN-1k"),
    # --- natural-image, self-supervised ---------------------------------- #
    BackboneSpec("dinov2_vits14", "timm", "vit_small_patch14_dinov2.lvd142m",
                 "self-supervised", "natural", "DINOv2 ViT-S/14"),
    BackboneSpec("mae_vit_b16", "timm", "vit_base_patch16_224.mae",
                 "self-supervised", "natural", "MAE ViT-B/16 (no fine-tune)"),
    # --- medical-domain --------------------------------------------------- #
    BackboneSpec("biomedclip_vit_b16", "open_clip",
                 "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 "vision-language", "medical", "BiomedCLIP ViT-B/16 image enc"),
    BackboneSpec("rad_dino_vit_b14", "hf", "microsoft/rad-dino",
                 "self-supervised", "medical", "RAD-DINO ViT-B/14 (CXR)"),
]


def get_backbones(names: list[str] | None = None) -> list[BackboneSpec]:
    if names is None:
        return DEFAULT_BACKBONES
    by_name = {b.name: b for b in DEFAULT_BACKBONES}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"unknown backbone(s) {missing}; known: {sorted(by_name)}")
    return [by_name[n] for n in names]


# --------------------------------------------------------------------------- #
# Uniform extractor                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Extractor:
    name: str
    transform: Callable          # PIL.Image -> Tensor (C, H, W)
    embed: Callable              # Tensor (B, C, H, W) -> Tensor (B, D)
    feat_dim: int
    model_id: str


def _infer_feat_dim(transform, embed, device) -> int:
    import torch
    from PIL import Image
    dummy = Image.new("RGB", (256, 256), (127, 127, 127))
    x = transform(dummy).unsqueeze(0).to(device)
    with torch.no_grad():
        out = embed(x)
    return int(out.shape[-1])


def build_extractor(spec: BackboneSpec, device: str) -> Extractor:
    if spec.source == "timm":
        transform, embed = _build_timm(spec, device)
    elif spec.source == "open_clip":
        transform, embed = _build_open_clip(spec, device)
    elif spec.source == "hf":
        transform, embed = _build_hf(spec, device)
    else:
        raise ValueError(f"unknown backbone source {spec.source!r}")
    feat_dim = _infer_feat_dim(transform, embed, device)
    return Extractor(spec.name, transform, embed, feat_dim, spec.model_id)


def _build_timm(spec: BackboneSpec, device: str):
    import timm
    import torch

    model = timm.create_model(spec.model_id, pretrained=True, num_classes=0)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)

    def embed(x):
        with torch.no_grad():
            return model(x)

    return transform, embed


def _build_open_clip(spec: BackboneSpec, device: str):
    import open_clip
    import torch

    model, preprocess = open_clip.create_model_from_pretrained(spec.model_id)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    def embed(x):
        with torch.no_grad():
            return model.encode_image(x)

    return preprocess, embed


def _build_hf(spec: BackboneSpec, device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(spec.model_id)
    model = AutoModel.from_pretrained(spec.model_id)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    def transform(pil):
        return processor(images=pil, return_tensors="pt")["pixel_values"][0]

    def embed(x):
        with torch.no_grad():
            out = model(pixel_values=x)
            # ViT/DINOv2: prefer CLS pooler_output, fall back to mean of tokens
            if getattr(out, "pooler_output", None) is not None:
                return out.pooler_output
            return out.last_hidden_state.mean(dim=1)

    return transform, embed
