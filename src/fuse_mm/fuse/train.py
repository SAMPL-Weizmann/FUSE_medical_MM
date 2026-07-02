"""FUSE stage-1 trainer: warm-start (CE + alpha*NCL on S_L) then joint
(add lambda*TCI on S_U). Saves the verifier model, the per-verifier scalers,
and the verifier score matrices for S_L / S_U / Test (stage-2/3 consume these)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import load_config
from ..heads.metrics import binary_metrics, standardize_apply, standardize_fit
from ..splits import load_splits
from .bank import build_bank, resolve_verifiers, vkey
from .losses import ce_loss, ncl_loss, tci_loss
from .model import FuseVerifiers


def train_fuse(cfg, device: str = "cpu", verbose: bool = True) -> dict:
    import torch

    # fixed seed so a lambda-sweep varies only lambda, not weight init / shuffling
    torch.manual_seed(0)
    np.random.seed(0)

    feats_dir = cfg["io"]["features_dir"]
    split = load_splits(load_config())
    verifiers = resolve_verifiers(cfg, feats_dir)

    banks, ys, pids = {}, {}, {}
    for key, set_name in [("L", cfg["io"]["labeled_set"]),
                          ("U", cfg["io"]["unlabeled_set"]),
                          ("T", cfg["io"]["test_set"])]:
        banks[key], ys[key], pids[key] = build_bank(feats_dir, split, set_name, verifiers)

    # standardize each verifier, fit on S_L, apply to all
    scalers = {}
    for name in banks["L"]:
        mu, sd = standardize_fit(banks["L"][name])
        scalers[name] = (mu, sd)
        for key in banks:
            banks[key][name] = standardize_apply(banks[key][name], mu, sd)

    dims = {name: banks["L"][name].shape[1] for name in banks["L"]}
    ch = cfg["head"]
    cfg_head = {"type": ch["type"], "mlp_layers": ch["mlp_layers"],
                "dropout": ch["dropout"], "n_outputs": ch["n_answers"]}
    model = FuseVerifiers(dims, 2, cfg_head).to(device)

    def tensors(key):
        return {n: torch.tensor(banks[key][n], dtype=torch.float32, device=device)
                for n in banks[key]}
    tL, tU, tT = tensors("L"), tensors("U"), tensors("T")
    yL = torch.tensor(ys["L"], dtype=torch.long, device=device)

    cw = None
    if ch["class_weight"] == "balanced":
        counts = np.bincount(ys["L"], minlength=2).astype(float)
        w = counts.sum() / (2 * np.maximum(counts, 1))
        cw = torch.tensor(w, dtype=torch.float32, device=device)

    tr = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    warm, joint = tr["warmstart_epochs"], tr["joint_epochs"]

    history = []
    for epoch in range(warm + joint):
        model.train()
        opt.zero_grad()
        logits_L, v_L = model(tL)
        l_ce = ce_loss(logits_L, yL, cw)
        l_ncl = ncl_loss(v_L)
        lam = tr["lambda_tci"] if epoch >= warm else 0.0
        l_tci = torch.tensor(0.0, device=device)
        if lam > 0:
            _, v_U = model(tU)
            l_tci = tci_loss(v_U.reshape(v_U.shape[0], -1), tr["tci_eps"])
        loss = l_ce + tr["alpha_ncl"] * l_ncl + lam * l_tci
        loss.backward()
        opt.step()

        if verbose and (epoch % tr["log_every"] == 0 or epoch == warm + joint - 1):
            phase = "warm" if epoch < warm else "joint"
            print(f"[{phase} {epoch:4d}] CE={l_ce.item():.3f} NCL={l_ncl.item():.3f} "
                  f"TCI={float(l_tci):.4f}", flush=True)
            history.append({"epoch": epoch, "ce": float(l_ce), "ncl": float(l_ncl),
                            "tci": float(l_tci)})

    # final verifier scores
    model.eval()
    def scores(t):
        with torch.no_grad():
            _, v = model(t)
        return v.reshape(v.shape[0], -1).cpu().numpy()
    vL, vU, vT = scores(tL), scores(tU), scores(tT)
    names = model.flat_verifier_names()

    summary = _finalize(cfg, model, scalers, verifiers, names,
                        {"labeled": (vL, ys["L"], pids["L"]),
                         "unlabeled": (vU, ys["U"], pids["U"]),
                         "test": (vT, ys["T"], pids["T"])},
                        history, device)
    return summary


def _finalize(cfg, model, scalers, verifiers, names, sets, history, device):
    import torch

    # key by lambda so lambda=0 (baseline) and lambda!=0 (TCI) runs coexist
    out = Path(cfg["io"]["out_dir"]) / f"lambda_{cfg['train']['lambda_tci']}"
    out.mkdir(parents=True, exist_ok=True)

    for set_name, (v, y, pid) in sets.items():
        np.savez_compressed(out / f"verifier_scores_{set_name}.npz",
                            v=v, y=y, patient_ids=pid, verifier_names=np.array(names))

    torch.save({"state_dict": model.module.state_dict(),
                "scalers": {k: (mu.tolist(), sd.tolist()) for k, (mu, sd) in scalers.items()},
                "verifiers": verifiers, "verifier_names": names,
                "cfg": cfg}, out / "fuse_model.pt")

    # sanity: per-verifier AUC on labeled + test, and final TCI on unlabeled
    vU = sets["unlabeled"][0]
    from .losses import tci_loss
    with torch.no_grad():
        tci_final = float(tci_loss(torch.tensor(vU, dtype=torch.float32), cfg["train"]["tci_eps"]))
    vL, yL, _ = sets["labeled"]
    vT, yT, _ = sets["test"]
    per_verifier = [{"name": names[j],
                     "auc_labeled": binary_metrics(yL, vL[:, j])["auc"],
                     "auc_test": binary_metrics(yT, vT[:, j])["auc"]}
                    for j in range(len(names))]
    summary = {"n_verifiers": len(names), "verifiers": verifiers,
               "tci_unlabeled_final": tci_final,
               "per_verifier": per_verifier, "history": history}
    with open(out / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
