"""Classifier heads, with optional Negative-Correlation-Learning (NCL) outputs.

A head emits `n_outputs` answers for the SAME label. Each answer is a full
classifier trained with cross-entropy; an extra loss penalizes the correlation
between the answers' scores over the batch (output-space decorrelation: cosine
of the centered scores, which equals Pearson r). This yields a diverse pair of
predictors from one backbone. n_outputs == 1 recovers the plain single head.

`linear` == logistic-regression probe (no hidden layer); `mlp` == configurable
hidden layers shared by all answers (a trunk), with a per-answer linear readout.
"""

from __future__ import annotations

import numpy as np


def build_head(in_dim: int, n_classes: int, cfg_head: dict):
    import torch.nn as nn

    n_out = int(cfg_head.get("n_outputs", 1))

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            layers: list = []
            d = in_dim
            if cfg_head["type"] == "mlp":
                dropout = float(cfg_head.get("dropout", 0.0))
                for h in cfg_head["mlp_layers"]:
                    layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                    d = h
            self.trunk = nn.Sequential(*layers)      # shared representation
            self.readout = nn.Linear(d, n_out * n_classes)
            self.n_out, self.n_classes = n_out, n_classes

        def forward(self, x):
            z = self.readout(self.trunk(x))
            return z.view(-1, self.n_out, self.n_classes)   # (B, n_out, n_classes)

    return Head()


def _decorr_loss(scores):
    """Mean squared off-diagonal cosine of centered scores (= Pearson r^2).

    scores: (B, n_out). For n_out==2 this is exactly corr(s0, s1)^2.
    """
    import torch

    k = scores.shape[1]
    if k < 2:                                           # one answer -> no pair to decorrelate
        return scores.new_zeros(())                     # (avoids /(k*(k-1))=0 -> NaN)
    sc = scores - scores.mean(dim=0, keepdim=True)
    normed = sc / sc.norm(dim=0).clamp_min(1e-8)
    corr = normed.t() @ normed                          # (n_out, n_out) cosine
    off = corr - torch.eye(k, device=corr.device, dtype=corr.dtype)
    return (off ** 2).sum() / (k * (k - 1))


def train_head(Xtr, ytr, cfg_head: dict, n_classes: int, device: str = "cpu"):
    """Fit a (possibly multi-output NCL) head. Returns predict(X) -> (N, n_out)
    array of positive-class probabilities, one column per answer."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    n_out = int(cfg_head.get("n_outputs", 1))
    dec = cfg_head.get("decorrelation") or {}
    lam = float(dec.get("weight", 0.0)) if n_out > 1 else 0.0
    dec_on = dec.get("on", "logit")

    model = build_head(Xtr.shape[1], n_classes, cfg_head).to(device)

    weight = None
    if cfg_head.get("class_weight") == "balanced":
        counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
        w = counts.sum() / (n_classes * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float32, device=device)

    crit = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=cfg_head["lr"],
                           weight_decay=cfg_head["weight_decay"])

    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    full_batch = cfg_head["type"] == "linear" or len(Xtr) <= cfg_head["batch_size"]
    bs = len(Xtr) if full_batch else cfg_head["batch_size"]
    rng = torch.Generator().manual_seed(0)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True, generator=rng)

    model.train()
    for _ in range(cfg_head["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)                                  # (B, n_out, C)
            loss = sum(crit(logits[:, a, :], yb) for a in range(n_out))
            if lam > 0 and xb.shape[0] > 2:
                if dec_on == "prob":
                    scores = torch.softmax(logits, dim=-1)[:, :, 1]
                else:                                           # class-1 margin
                    scores = logits[:, :, 1] - logits[:, :, 0]
                loss = loss + lam * _decorr_loss(scores)
            loss.backward()
            opt.step()

    model.eval()

    def predict_proba(X) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32, device=device))
            return torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()  # (N, n_out)

    return predict_proba
