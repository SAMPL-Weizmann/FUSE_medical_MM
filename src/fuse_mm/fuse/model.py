"""Joint verifier model: one head per (modality, backbone), each emitting
`n_answers` NCL outputs. Verifier scores v in [0,1] = P(abnormal)."""

from __future__ import annotations

from ..heads.models import build_head


class FuseVerifiers:
    """Thin wrapper around a ModuleDict of heads (kept plain to lazy-import torch)."""

    def __init__(self, dims: dict[str, int], n_classes: int, cfg_head: dict):
        import torch.nn as nn

        self.names = sorted(dims)                       # stable verifier-head order
        self.n_answers = int(cfg_head.get("n_outputs", 1))
        self.module = nn.ModuleDict(
            {name: build_head(dims[name], n_classes, cfg_head) for name in self.names}
        )

    def to(self, device):
        self.module.to(device)
        return self

    def parameters(self):
        return self.module.parameters()

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def flat_verifier_names(self) -> list[str]:
        return [f"{name}#a{a}" for name in self.names for a in range(self.n_answers)]

    def __call__(self, feats: dict):
        """feats[name] = (B, dim). Returns logits (B, H, A, C) and scores v (B, H, A)."""
        import torch

        logits = torch.stack([self.module[n](feats[n]) for n in self.names], dim=1)
        v = torch.softmax(logits, dim=-1)[..., 1]       # (B, H, A) = P(class 1)
        return logits, v
