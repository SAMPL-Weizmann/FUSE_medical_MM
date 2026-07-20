"""Verifier Grad-CAM with role-based columns (per-patient CV membership).

Companion to `17_saliency.py` (input-gradient d(margin)/dx). Grad-CAM taps a
SPATIAL feature layer inside the backbone and weights its activation channels
A^k by the GAP'd gradient of the abnormal margin  m = logit_1 - logit_0:

    CAM = ReLU( sum_k  ( GAP_hw  dm/dA^k )  * A^k )

CNN target = last conv map; ViT target = last block's PRE-ATTENTION LayerNorm
(`norm1`) — the block output's patch tokens feed nothing under CLS pooling, so
their gradient is 0 and the CAM is blank; norm1's patch tokens DO reach CLS
through that block's attention. CNN-vs-ViT is auto-detected from activation rank.

WHAT'S NEW vs the first cut: the three columns are no longer fixed folds. Each
patient sits in ONE home fold, so across CV iterations it plays three different
roles. We show one column per ROLE, chosen per-patient:
    col 1  labeled   — the (unique) fold where the patient is in S_L
    col 2  unlabeled — the BEST-performing fold where it is in S_U
    col 3  test      — the fold where it is held out (10-fold: unique; 20-fold:
                       2 such folds -> the better-performing one)
plus a leftmost 'original' reference column (raw frame, no overlay).
"Performance" of a fold = that CV iteration's fuse_ens test balanced accuracy,
computed here in Stage A (MoM -> ensemble -> score), so selection is
self-contained for any (n_folds, n_test, lambda).

Two configs (pick with --config; both runnable on WEXAC):
    cv10:  10 folds, n_test=1, lambda=0.2  -> artifacts/reports/gradcam/
    cv20:  20 folds, n_test=2, lambda=0.0  -> artifacts/reports/gradcam20/

Same 4 patients as the saliency figure (reuses its manifest when present) for
cross-figure comparability; their home fold / roles are recomputed per config.

Two stages (compute trains heads + runs live backbones -> WEXAC; plot is cheap):
    python scripts/18_gradcam.py compute --config {cv10,cv20} [--device cuda]
    python scripts/18_gradcam.py plot    --config {cv10,cv20}

Outputs (<out>/):
    gcam_<k>_<pid>.npz          per-sample frames + CAM maps (orig coords)
    gradcam_manifest.json       config, per-fold bAcc, per-sample roles + items
    gcam_<k>_<pid>_US.png / _MG.png   contact sheets (rows=16 verifiers)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# reuse 17_saliency's geometry / sample-selection / plotting helpers verbatim so
# the two figures share overlay math and pick the same patients (module name has
# a leading digit, so load it by path).
_H = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("sal17", os.path.join(_H, "17_saliency.py"))
sal17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sal17)

MODALITIES = sal17.MODALITIES    # ["US", "MG"]
N_MG_VIEWS = sal17.N_MG_VIEWS    # 1 shown view (score still pools 4)
SETS = ["labeled", "unlabeled", "test"]
SAL_MANIFEST = os.path.join(sal17.OUT, "saliency_manifest.json")

# --- config presets -------------------------------------------------------- #
# `config` = the FUSE head config (omitted -> configs/fuse.yaml, 2 answers). The
# DATA config (normal-vs-abnormal vs malignant) is chosen by FUSE_DATA_CONFIG in
# the environment, not here, so a preset works for either task.
PRESETS = {
    "cv10": dict(n_folds=10, n_test=1, lam=0.2, out="artifacts/reports/gradcam"),
    "cv20": dict(n_folds=20, n_test=2, lam=0.0, out="artifacts/reports/gradcam20"),
    # normal-vs-abnormal, 1-answer heads (16 verifiers -> 8 backbone rows), lambda=0
    "cv10_1ans": dict(n_folds=10, n_test=1, lam=0.0, config="configs/fuse_1ans.yaml",
                      out="artifacts/reports/gradcam_1ans"),
}


# --------------------------------------------------------------------------- #
# target-layer resolution + activation/grad capture
# --------------------------------------------------------------------------- #
def _resolve_target(model, spec):
    """The nn.Module whose output is the Grad-CAM activation, per backbone.

    CNNs are avg-pooled, so their last conv map is the natural target. The ViTs
    here all pool the feature from the CLS token, and the LAST block's *patch*
    tokens feed nothing downstream (only CLS survives pooling) -> their gradient
    is identically 0 and the CAM comes out blank. So for ViTs we target the last
    block's PRE-ATTENTION LayerNorm (`norm1`): its patch tokens DO reach the CLS
    token through that block's attention, so their gradients are nonzero. (Must
    be norm1, not norm2 — the post-norm2 MLP is token-wise, so norm2's patch
    gradients would also vanish.)"""
    if spec.source == "hf":                       # rad_dino: hf Dinov2 encoder
        return model.encoder.layer[-1].norm1
    if spec.source == "open_clip":                # biomedclip: timm ViT trunk
        return model.visual.trunk.blocks[-1].norm1
    # timm
    if spec.name == "resnet50":
        return model.layer4
    if spec.name == "efficientnet_b0":
        return model.conv_head
    if spec.name == "convnext_tiny":
        return model.stages[-1]
    if hasattr(model, "blocks"):                  # timm ViTs (vit_b16 / dinov2 / mae)
        return model.blocks[-1].norm1
    raise ValueError(f"no Grad-CAM target for backbone {spec.name!r}")


def _pooling_desc(model, spec):
    """Best-effort one-line description of how the backbone pools its feature —
    printed during compute so the CLS-vs-avg split is confirmed, not assumed."""
    try:
        if spec.source == "hf":
            return "hf pooler_output(CLS)"
        if spec.source == "open_clip":
            gp = getattr(model.visual.trunk, "global_pool", "?")
            return f"open_clip trunk.global_pool={gp!r}"
        gp = getattr(model, "global_pool", "?")
        gp = getattr(gp, "pool_type", gp)          # SelectAdaptivePool2d -> str
        return f"timm global_pool={gp!r}"
    except Exception as e:                          # never let a probe break compute
        return f"<pooling probe failed: {e}>"


class CamHook:
    """Forward hook: stashes the target layer's output and keeps its grad."""

    def __init__(self, module):
        self.act = None
        self._h = module.register_forward_hook(self._fwd)

    def _fwd(self, _m, _inp, out):
        out = out[0] if isinstance(out, tuple) else out   # hf layers return a tuple
        out.retain_grad()
        self.act = out

    def remove(self):
        self._h.remove()


def build_gradcam_backbone(spec, device):
    """Like sal17.build_grad_backbone but also registers a CamHook on the target
    layer and returns it (plus the resize/crop geometry for inverse-mapping)."""
    import torch

    if spec.source == "timm":
        import timm
        model = timm.create_model(spec.model_id, pretrained=True, num_classes=0)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        cfg = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**cfg, is_training=False)
        embed = lambda x: model(x)
        geom = sal17._geom_from_compose(transform)
    elif spec.source == "open_clip":
        import open_clip
        model, preprocess = open_clip.create_model_from_pretrained(spec.model_id)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        transform = preprocess
        embed = lambda x: model.encode_image(x)
        geom = sal17._geom_from_compose(preprocess)
    elif spec.source == "hf":
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(spec.model_id)
        model = AutoModel.from_pretrained(spec.model_id).eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)

        def transform(pil):
            return proc(images=pil, return_tensors="pt")["pixel_values"][0]

        def embed(x):
            out = model(pixel_values=x)
            po = getattr(out, "pooler_output", None)
            return po if po is not None else out.last_hidden_state.mean(dim=1)

        geom = sal17._geom_from_processor(proc)
    else:
        raise ValueError(f"unknown source {spec.source!r}")

    target = _resolve_target(model, spec)
    hook = CamHook(target)
    hook.pooling = _pooling_desc(model, spec)
    hook.target_desc = type(target).__name__
    return transform, embed, geom, hook


def compute_cam(act, grad):
    """Vanilla Grad-CAM. act/grad: (Nv,C,H,W) CNN or (Nv,N,D) ViT tokens.
    Returns ReLU'd CAM per view, (Nv, g, g)."""
    import torch

    if act.ndim == 3:                              # ViT: tokens -> square grid
        nv, n, d = act.shape
        g = int(math.isqrt(n))                     # prefix tokens (CLS/registers)
        p = n - g * g                              # sit in front -> drop first p
        act = act[:, p:, :].transpose(1, 2).reshape(nv, d, g, g)
        grad = grad[:, p:, :].transpose(1, 2).reshape(nv, d, g, g)
    alpha = grad.mean(dim=(2, 3), keepdim=True)    # GAP over spatial -> channel weights
    cam = torch.relu((alpha * act).sum(dim=1))     # (Nv, g, g)
    return cam.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# per-patient role -> column-fold selection
# --------------------------------------------------------------------------- #
def _role_columns(home_fold, n_folds, n_test, fold_bacc):
    """Map a patient's single home fold to the three role columns. `fold_bacc[i]`
    ranks CV iterations by fuse_ens test balanced accuracy; ties broken by index.
      labeled   = the one iteration i with S_L == fold i  ->  i == home_fold
      test      = iterations whose Test set contains home_fold -> best of those
      unlabeled = everything else -> best of those"""
    n = n_folds
    labeled = home_fold
    test_iters = [(home_fold - 1 - j) % n for j in range(n_test)]
    unlabeled_iters = [i for i in range(n) if i != labeled and i not in test_iters]
    best = lambda cands: max(cands, key=lambda i: (fold_bacc[i], -i))
    return {"labeled": labeled,
            "unlabeled": best(unlabeled_iters),
            "test": best(test_iters)}


def _pick_samples(models, scalers, full_fs, records, device, override, best_fold):
    """Same 4 patients as the saliency figure: reuse its manifest if present,
    else fall back to 17's selector anchored on the best-performing fold."""
    if override:
        by_us = {r["us_id"]: r for r in records}
        return [{"mg_id": by_us[u]["mg_id"], "us_id": u, "label": int(by_us[u]["label"])}
                for u in override if u in by_us]
    if os.path.exists(SAL_MANIFEST):
        with open(SAL_MANIFEST, encoding="utf-8") as f:
            sal = json.load(f)
        print(f"reusing samples from {SAL_MANIFEST}", flush=True)
        return [{"mg_id": s["mg_id"], "us_id": s["us_id"], "label": int(s["label"])}
                for s in sal["samples"]]
    # fallback: 17's selector, but anchor the reference verifier on the best fold
    saved = list(sal17.FOLDS)
    sal17.FOLDS[:] = [best_fold] + saved[1:]
    try:
        return sal17._select_samples(models, scalers, full_fs, records, device, None)
    finally:
        sal17.FOLDS[:] = saved


# --------------------------------------------------------------------------- #
# compute (WEXAC): retrain ALL folds, rank them, Grad-CAM the role columns
# --------------------------------------------------------------------------- #
def compute(preset, device, override):
    import copy
    import torch
    from PIL import Image

    from fuse_mm.bench.metrics import score_stats
    from fuse_mm.dataset import load_image
    from fuse_mm.features.backbones import get_backbones
    from fuse_mm.fuse import load_fuse_config
    from fuse_mm.fuse.bank import (build_bank_pooled, load_full_featuresets,
                                   resolve_verifiers, vkey)
    from fuse_mm.fuse.estimate import (fit_mom, optimize_ensemble,
                                       posterior_triplet_avg, predict_ensemble)
    from fuse_mm.fuse.train import fit_and_score
    from fuse_mm.cv.folds import fold_assignment, load_cv_folds, make_cv_folds

    n_folds, n_test, lam, out = (preset["n_folds"], preset["n_test"],
                                 preset["lam"], preset["out"])
    cfg = load_fuse_config(preset.get("config"))     # None -> fuse.yaml (2 answers)
    cfg["train"]["lambda_tci"] = lam
    feats_dir = cfg["io"]["features_dir"]
    verifiers = resolve_verifiers(cfg, feats_dir)
    full_fs = load_full_featuresets(feats_dir, verifiers)
    try:
        folds = load_cv_folds(n_folds=n_folds)
    except FileNotFoundError:
        folds = make_cv_folds(n_folds=n_folds)
    records = folds["records"]
    home_fold = {str(r["us_id"]): int(r["fold"]) for r in records}
    n_ans = int(cfg["head"]["n_answers"])

    # --- Stage A: retrain EVERY fold's heads + rank by fuse_ens test bAcc ---- #
    models, scalers, fold_bacc = {}, {}, {}
    for i in range(n_folds):
        fs = fold_assignment(folds, i, n_test=n_test)
        banks, ys, pids = {}, {}, {}
        for k, s in [("L", "labeled"), ("U", "unlabeled"), ("T", "test")]:
            banks[k], ys[k], pids[k] = build_bank_pooled(full_fs, fs[s], verifiers)
        scores, _names, model, sc, _ = fit_and_score(
            banks, ys, pids, copy.deepcopy(cfg), device, False)
        model.eval()
        V = {s: scores[s][0].astype(float) for s in SETS}
        Y = {s: scores[s][1] for s in SETS}
        params = fit_mom(V["unlabeled"])                     # fuse_ens est on S_U
        phat = posterior_triplet_avg(V["unlabeled"], params)
        w, b = optimize_ensemble(V["unlabeled"], phat)
        bacc = float(score_stats(predict_ensemble(V["test"], w, b), Y["test"])["balanced_acc"])
        models[i], scalers[i], fold_bacc[i] = model, sc, bacc
        print(f"  fold {i+1}/{n_folds}: fuse_ens test bAcc={bacc:.4f} (lambda={lam})", flush=True)

    best_fold = max(range(n_folds), key=lambda i: fold_bacc[i])
    samples = _pick_samples(models, scalers, full_fs, records, device, override, best_fold)
    for s in samples:                                        # per-patient role columns
        hf = home_fold[str(s["us_id"])]
        s["home_fold"] = hf
        s["roles"] = _role_columns(hf, n_folds, n_test, fold_bacc)
    print("selected samples:", [(s["us_id"], s["label"], s["roles"]) for s in samples], flush=True)

    path_map = {m: sal17._load_paths_map(feats_dir, m) for m in MODALITIES}
    specs = {s.name: s for s in get_backbones(cfg["verifiers"]["backbones"])}
    os.makedirs(out, exist_ok=True)

    store = {si: {} for si in range(len(samples))}
    frames_saved = {si: set() for si in range(len(samples))}
    manifest = {"method": "grad-cam", "n_folds": n_folds, "n_test": n_test, "lambda": lam,
                "fold_bacc": {str(i): fold_bacc[i] for i in range(n_folds)},
                "best_fold": best_fold,
                "samples": [{"idx": si, **s} for si, s in enumerate(samples)],
                "items": []}

    # --- Stage B: live backbone -> Grad-CAM for each sample's 3 role folds --- #
    for bb, spec in specs.items():
        transform, embed, (S, size), hook = build_gradcam_backbone(spec, device)
        print(f"  {bb}: target={hook.target_desc} pooling[{hook.pooling}]", flush=True)
        for mod in MODALITIES:
            key = vkey(mod, bb)
            pid_field = "us_id" if mod == "US" else "mg_id"
            for si, samp in enumerate(samples):
                pid = samp[pid_field]
                paths = path_map[mod].get(pid, [])
                if not paths:
                    continue
                role_of = {samp["roles"][r]: r for r in ("labeled", "unlabeled", "test")}
                imgs = [load_image(p) for p in paths]                 # (H,W,3) uint8
                x = torch.stack([transform(Image.fromarray(im)) for im in imgs]).to(device)
                x.requires_grad_(True)
                feats = embed(x)                                      # fires hook
                pooled = feats.mean(dim=0, keepdim=True)              # (1, D) mean-pool
                n_show = 1 if mod == "US" else min(N_MG_VIEWS, len(imgs))
                for vi in range(n_show):
                    fk = f"frame__{mod}__v{vi}"
                    if fk not in frames_saved[si]:
                        store[si][fk] = sal17._downsample(imgs[vi].astype(np.uint8))
                        frames_saved[si].add(fk)
                for i in sorted(role_of):                             # this sample's 3 folds
                    mu, sd = scalers[i][key]
                    fstd = (pooled - torch.tensor(mu, dtype=torch.float32, device=device)) \
                        / torch.tensor(sd, dtype=torch.float32, device=device)
                    logits = models[i].module[key](fstd)             # (1, n_ans, 2)
                    for a in range(n_ans):
                        m = logits[0, a, 1] - logits[0, a, 0]
                        v = torch.sigmoid(m).item()
                        if hook.act.grad is not None:
                            hook.act.grad = None
                        m.backward(retain_graph=True)
                        cam = compute_cam(hook.act, hook.act.grad)   # (Nv, g, g)
                        for vi in range(n_show):
                            full = sal17.map_to_original(cam[vi], imgs[vi].shape[:2], S, size)
                            store[si][f"map__{mod}__{bb}__a{a}__f{i}__v{vi}"] = sal17._downsample(full)
                        manifest["items"].append(
                            {"sample": si, "modality": mod, "backbone": bb, "answer": a,
                             "fold": i, "role": role_of[i], "n_views_pooled": int(cam.shape[0]),
                             "n_views_shown": n_show, "v": float(v), "margin": float(m.item())})
                del x, feats, pooled
        hook.remove()
        del embed, transform
        print(f"  backbone {bb}: grad-cam done", flush=True)

    for si, samp in enumerate(samples):
        np.savez_compressed(os.path.join(out, f"gcam_{si}_{samp['us_id']}.npz"), **store[si])
    with open(os.path.join(out, "gradcam_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {len(samples)} sample npz + manifest to {out}/")


# --------------------------------------------------------------------------- #
# plot: per-(sample, modality) sheet — cols = [original, labeled, unlabeled, test]
# --------------------------------------------------------------------------- #
ROLE_ORDER = ["labeled", "unlabeled", "test"]


def _sheet(store, manifest, si, mod, path, allowed=None):
    backbones = sorted({it["backbone"] for it in manifest["items"]})
    if allowed:                                            # slide-legible subset
        backbones = [b for b in backbones if b in allowed]
    n_ans = 1 + max(it["answer"] for it in manifest["items"])
    rows = [(bb, a) for bb in backbones for a in range(n_ans)]
    frame = store.get(f"frame__{mod}__v0")
    samp = manifest["samples"][si]
    roles = samp["roles"]                                   # role -> fold index
    fbac = manifest["fold_bacc"]
    cols = [("original", None)] + [(r, roles[r]) for r in ROLE_ORDER]
    ncol, nrow = len(cols), len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol + 1.8, 1.7 * nrow),
                             squeeze=False)
    vmeta = {(it["backbone"], it["answer"], it["fold"]): it for it in manifest["items"]
             if it["modality"] == mod and it["sample"] == si}
    for r, (bb, a) in enumerate(rows):
        for c, (role, fold) in enumerate(cols):
            ax = axes[r][c]
            if frame is None:
                ax.axis("off"); continue
            if role == "original":
                sal17._overlay(ax, frame, None)             # raw frame, no heat
            else:
                sal17._overlay(ax, frame, store.get(f"map__{mod}__{bb}__a{a}__f{fold}__v0"))
                meta = vmeta.get((bb, a, fold))
                if meta:
                    ax.set_xlabel(f"v={meta['v']:.2f}", fontsize=7, labelpad=1)
            if r == 0:
                if role == "original":
                    ax.set_title("original", fontsize=11, fontweight="bold")
                else:
                    ax.set_title(f"{role}\nfold {fold+1}  (bAcc {fbac[str(fold)]:.2f})",
                                 fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{bb}#a{a}", fontsize=7.5, rotation=0,
                              ha="right", va="center", labelpad=34)
    cls = "abnormal" if samp["label"] == 1 else "normal"
    lam = manifest["lambda"]
    fig.suptitle(f"Grad-CAM  ReLU(Σ αₖ Aₖ)  —  {mod}  —  {manifest['n_folds']}-fold CV "
                 f"(λ={lam:g})  —  sample {si} ({cls}, us={samp['us_id']})\n"
                 f"columns = this patient's role per fold "
                 f"(home fold {samp['home_fold']+1};  labeled / unlabeled / test)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.05, 0, 1, 0.975))
    fig.savefig(path, dpi=115, bbox_inches="tight"); plt.close(fig)


def plot(preset, subset=None):
    out = preset["out"]
    with open(os.path.join(out, "gradcam_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    suffix = "_crop" if subset else ""                     # keep full sheets intact
    written = 0
    for s in manifest["samples"]:
        si = s["idx"]
        small = os.path.join(out, f"gcam_{si}_{s['us_id']}.small.npz")
        big = os.path.join(out, f"gcam_{si}_{s['us_id']}.npz")
        store = sal17._load_npz(small if os.path.exists(small) else big)
        for mod in MODALITIES:
            fp = os.path.join(out, f"gcam_{si}_{s['us_id']}_{mod}{suffix}.png")
            _sheet(store, manifest, si, mod, fp, allowed=subset)
            written += 1
    print(f"wrote {written} {'cropped ' if subset else ''}contact sheets to {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compute", "plot"])
    ap.add_argument("--config", choices=list(PRESETS), default="cv10")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--patients", nargs="*", default=None,
                    help="override auto-selection with these US ids")
    ap.add_argument("--backbones", nargs="*", default=None,
                    help="plot only these backbones -> slide-legible '_crop' sheets")
    args = ap.parse_args()
    preset = PRESETS[args.config]
    if args.mode == "compute":
        compute(preset, args.device, args.patients)
    else:
        plot(preset, subset=args.backbones)


if __name__ == "__main__":
    main()
