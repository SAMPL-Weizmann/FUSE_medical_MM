"""Classifier heads. `linear` == logistic-regression probe (no hidden layer);
`mlp` == configurable hidden layers. Both share one training/eval path so the
only difference between experiments is config."""

from __future__ import annotations

import numpy as np


def build_head(in_dim: int, n_classes: int, cfg_head: dict):
    import torch.nn as nn

    layers: list = []
    d = in_dim
    if cfg_head["type"] == "mlp":
        dropout = float(cfg_head.get("dropout", 0.0))
        for h in cfg_head["mlp_layers"]:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
    layers += [nn.Linear(d, n_classes)]
    return nn.Sequential(*layers)


def train_head(Xtr, ytr, cfg_head: dict, n_classes: int, device: str = "cpu"):
    """Fit a head on (Xtr, ytr). Returns a predict(X)->prob[:,1..] callable."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    rng = torch.Generator().manual_seed(0)
    model = build_head(Xtr.shape[1], n_classes, cfg_head).to(device)

    # class weights to counter imbalance
    weight = None
    if cfg_head.get("class_weight") == "balanced":
        counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
        w = counts.sum() / (n_classes * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float32, device=device)

    crit = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg_head["lr"],
        weight_decay=cfg_head["weight_decay"],
    )

    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    full_batch = cfg_head["type"] == "linear" or len(Xtr) <= cfg_head["batch_size"]
    bs = len(Xtr) if full_batch else cfg_head["batch_size"]
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True, generator=rng)

    model.train()
    for _ in range(cfg_head["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()

    def predict_proba(X) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32, device=device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    return predict_proba
