"""Verifier input-gradient saliency: d(margin)/dx over the original scan.

For the trained verifiers (the "after-train" stage, before FUSE estimate/infer),
show which pixels of the input scan drive each verifier's score, via the input
gradient of the class-1 margin  m = logit_1 - logit_0  (the de-saturated dv/dx,
since v = sigmoid(m)), overlaid on the ORIGINAL scan frame.

Settled scope (lambda=0.2):
  * all 32 verifiers = 8 backbones x {US, MG} x 2 NCL answers
  * US = 1 frame -> 1 map;  MG = 4 canonical views, mean-pooled before the head,
    so autograd yields one d m / d x_i per view (each scaled 1/N by the pooling)
  * folds: the CV top-3 by fuse_ens (7, 3, 6); per-fold heads are retrained
  * 4 samples (2 abnormal + 2 normal), auto-selected, same set everywhere
  * overlay target: the raw .npy frame (saliency inverse-mapped through resize+crop)

Two stages (compute trains heads + runs live backbones -> WEXAC; plot is local):

    python scripts/17_saliency.py compute [--device cpu] [--patients US_ID ...]
    python scripts/17_saliency.py plot

Outputs (artifacts/reports/saliency/):
    sal_<k>_<pid>.npz         per-sample frames + saliency maps (orig coords)
    saliency_manifest.json    samples + per-(verifier,fold) v / margin / label
    sal_<k>_<pid>_fold<f>_US.png / _MG.png   contact sheets
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUT = "artifacts/reports/saliency"
LAMBDA = 0.2
FOLDS = [6, 2, 5]                 # 0-based -> display folds 7, 3, 6 (top-3 fuse_ens)
MODALITIES = ["US", "MG"]
N_MG_VIEWS = 1                    # MG score still pools all 4 views (faithful); we
                                 # only SAVE/show this many views' saliency (from v0)
N_SAMPLES = 2                     # per class (2 abnormal + 2 normal = 4)
SEL_VERIFIER = ("US", "biomedclip_vit_b16", 1)   # (modality, backbone, answer) for auto-select
MAXSAVE = 320                     # cap saved map/frame resolution (mammograms are
                                  # ~2200px; panels are ~280px, so full-res is wasted
                                  # and makes the .npz enormous over the network share)


def _downsample(a, maxdim=MAXSAVE):
    """Nearest-neighbour downsample by striding (keeps the NaN mask; frame & map
    downsample identically since they share the original H,W -> stay aligned)."""
    h, w = a.shape[:2]
    step = max(1, int(round(max(h, w) / maxdim)))
    return a[::step, ::step]


# --------------------------------------------------------------------------- #
# geometry: recover (resize shorter-side S, center-crop size) so a saliency map
# at model-input resolution can be placed back on the original frame.
# --------------------------------------------------------------------------- #
def _geom_from_compose(tf):
    from torchvision.transforms import CenterCrop, Resize
    S = size = None
    for t in getattr(tf, "transforms", []):
        if isinstance(t, Resize):
            s = t.size
            S = s if isinstance(s, int) else min(s)
        elif isinstance(t, CenterCrop):
            s = t.size
            size = s if isinstance(s, int) else s[0]
    if S is None:
        S = size
    if size is None:
        size = S
    return int(S), int(size)


def _geom_from_processor(proc):
    size = getattr(proc, "size", {}) or {}
    S = size.get("shortest_edge") or size.get("height") or size.get("width")
    crop = getattr(proc, "crop_size", None)
    if crop and getattr(proc, "do_center_crop", False):
        csize = crop.get("height") or crop.get("shortest_edge")
        return int(S), int(csize)
    return int(S), int(S)


def build_grad_backbone(spec, device):
    """Like features.backbones but WITHOUT torch.no_grad so grads reach the input."""
    if spec.source == "timm":
        import timm
        model = timm.create_model(spec.model_id, pretrained=True, num_classes=0)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        cfg = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**cfg, is_training=False)
        return transform, (lambda x: model(x)), _geom_from_compose(transform)
    if spec.source == "open_clip":
        import open_clip
        model, preprocess = open_clip.create_model_from_pretrained(spec.model_id)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        return preprocess, (lambda x: model.encode_image(x)), _geom_from_compose(preprocess)
    if spec.source == "hf":
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

        return transform, embed, _geom_from_processor(proc)
    raise ValueError(f"unknown source {spec.source!r}")


def _resize2d(a, out_hw):
    from PIL import Image
    im = Image.fromarray(a.astype(np.float32), mode="F")
    im = im.resize((out_hw[1], out_hw[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def map_to_original(mag, orig_hw, S, size):
    """Place a (size,size) map back onto the original frame via the inverse of
    'resize shorter side to S' + 'center-crop size'. Outside the crop -> NaN."""
    H0, W0 = orig_hw
    scale = S / min(H0, W0)
    H1, W1 = round(H0 * scale), round(W0 * scale)
    top, left = (H1 - size) // 2, (W1 - size) // 2
    bH = bW = int(round(size / scale))
    box = _resize2d(mag, (bH, bW))
    t, l = int(round(top / scale)), int(round(left / scale))
    full = np.full((H0, W0), np.nan, dtype=np.float32)
    t2, l2 = max(t, 0), max(l, 0)
    bt, bl = t2 - t, l2 - l
    hh, ww = min(bH - bt, H0 - t2), min(bW - bl, W0 - l2)
    if hh > 0 and ww > 0:
        full[t2:t2 + hh, l2:l2 + ww] = box[bt:bt + hh, bl:bl + ww]
    return full


# --------------------------------------------------------------------------- #
# compute (WEXAC): retrain per-fold heads, then backprop margins to the images
# --------------------------------------------------------------------------- #
def _load_paths_map(features_dir, modality, ref_backbone="resnet50"):
    """patient_id -> [source .npy paths] (selection is backbone-independent)."""
    from pathlib import Path
    m = {}
    for s in ("labeled", "unlabeled", "test"):
        d = np.load(Path(features_dir) / s / modality / f"{ref_backbone}.npz", allow_pickle=True)
        for pid, pth in zip(d["patient_ids"].astype(str), d["paths"].astype(str)):
            m.setdefault(pid, []).append(pth)
    return m


def _select_samples(models, scalers, full_fs, records, device, override):
    """Pick 2 abnormal + 2 normal patients, confidently correct under the best
    fold's reference US verifier. Returns list of {mg_id, us_id, label}."""
    import torch
    from fuse_mm.fuse.bank import vkey

    by_us = {r["us_id"]: r for r in records}
    if override:
        return [{"mg_id": by_us[u]["mg_id"], "us_id": u, "label": int(by_us[u]["label"])}
                for u in override if u in by_us]

    mod, bb, a = SEL_VERIFIER
    key = vkey(mod, bb)
    fs = full_fs[key]
    mu, sd = scalers[FOLDS[0]][key]
    head = models[FOLDS[0]].module[key]
    X = (fs.pooled(fs.patients) - mu) / sd
    with torch.no_grad():
        logits = head(torch.tensor(X, dtype=torch.float32, device=device))
        v = torch.softmax(logits, dim=-1)[:, a, 1].cpu().numpy()
    y = fs.y_patient
    us_ids = fs.patients
    pos = us_ids[y == 1][np.argsort(-v[y == 1])][:N_SAMPLES]      # confident abnormal
    neg = us_ids[y == 0][np.argsort(v[y == 0])][:N_SAMPLES]       # confident normal
    out = []
    for u in list(pos) + list(neg):
        r = by_us.get(str(u))
        if r:
            out.append({"mg_id": r["mg_id"], "us_id": r["us_id"], "label": int(r["label"])})
    return out


def compute(device, override):
    import copy
    import torch
    from PIL import Image

    from fuse_mm.dataset import load_image
    from fuse_mm.features.backbones import get_backbones
    from fuse_mm.fuse import load_fuse_config
    from fuse_mm.fuse.bank import (build_bank_pooled, load_full_featuresets,
                                   resolve_verifiers, vkey)
    from fuse_mm.fuse.train import fit_and_score
    from fuse_mm.cv.folds import fold_assignment, load_cv_folds, make_cv_folds

    cfg = load_fuse_config(None)
    cfg["train"]["lambda_tci"] = LAMBDA
    feats_dir = cfg["io"]["features_dir"]
    verifiers = resolve_verifiers(cfg, feats_dir)
    full_fs = load_full_featuresets(feats_dir, verifiers)
    try:
        folds = load_cv_folds()
    except FileNotFoundError:
        folds = make_cv_folds()
    records = folds["records"]
    n_ans = int(cfg["head"]["n_answers"])

    # --- Stage A: retrain per-fold heads at lambda=0.2 ---------------------- #
    models, scalers = {}, {}
    for i in FOLDS:
        fs = fold_assignment(folds, i)
        banks, ys, pids = {}, {}, {}
        for k, s in [("L", "labeled"), ("U", "unlabeled"), ("T", "test")]:
            banks[k], ys[k], pids[k] = build_bank_pooled(full_fs, fs[s], verifiers)
        _, _, model, sc, _ = fit_and_score(banks, ys, pids, copy.deepcopy(cfg), device, False)
        model.eval()
        models[i], scalers[i] = model, sc
        print(f"  trained fold {i+1} heads (lambda={LAMBDA})", flush=True)

    samples = _select_samples(models, scalers, full_fs, records, device, override)
    print("selected samples:", [(s["us_id"], s["label"]) for s in samples], flush=True)

    path_map = {m: _load_paths_map(feats_dir, m) for m in MODALITIES}
    specs = {s.name: s for s in get_backbones(cfg["verifiers"]["backbones"])}
    os.makedirs(OUT, exist_ok=True)

    # per-sample stores + manifest
    store = {si: {} for si in range(len(samples))}       # si -> {arr_key: array}
    frames_saved = {si: set() for si in range(len(samples))}
    manifest = {"lambda": LAMBDA, "folds_display": [i + 1 for i in FOLDS],
                "samples": [{"idx": si, **s} for si, s in enumerate(samples)],
                "items": []}

    # --- Stage B: live backbone -> d(margin)/dx, one backbone at a time ----- #
    for bb, spec in specs.items():
        transform, embed, (S, size) = build_grad_backbone(spec, device)
        for mod in MODALITIES:
            key = vkey(mod, bb)
            pid_field = "us_id" if mod == "US" else "mg_id"
            for si, samp in enumerate(samples):
                pid = samp[pid_field]
                paths = path_map[mod].get(pid, [])
                if not paths:
                    continue
                imgs = [load_image(p) for p in paths]                 # (H,W,3) uint8
                x = torch.stack([transform(Image.fromarray(im)) for im in imgs]).to(device)
                x.requires_grad_(True)
                feats = embed(x)                                      # (Nv, D)
                pooled = feats.mean(dim=0, keepdim=True)              # (1, D) mean-pool
                n_show = 1 if mod == "US" else min(N_MG_VIEWS, len(imgs))
                # save original frames once per (sample, modality)
                for vi in range(n_show):
                    fk = f"frame__{mod}__v{vi}"
                    if fk not in frames_saved[si]:
                        store[si][fk] = _downsample(imgs[vi].astype(np.uint8))
                        frames_saved[si].add(fk)
                for i in FOLDS:
                    mu, sd = scalers[i][key]
                    fstd = (pooled - torch.tensor(mu, dtype=torch.float32, device=device)) \
                        / torch.tensor(sd, dtype=torch.float32, device=device)
                    logits = models[i].module[key](fstd)             # (1, n_ans, 2)
                    for a in range(n_ans):
                        m = logits[0, a, 1] - logits[0, a, 0]
                        v = torch.sigmoid(m).item()
                        if x.grad is not None:
                            x.grad = None
                        m.backward(retain_graph=True)
                        g = x.grad.detach()                          # (Nv, 3, Hin, Win)
                        mag = g.pow(2).sum(dim=1).sqrt().cpu().numpy()   # (Nv, Hin, Win)
                        for vi in range(n_show):                     # save only shown views
                            full = map_to_original(mag[vi], imgs[vi].shape[:2], S, size)
                            store[si][f"map__{mod}__{bb}__a{a}__f{i}__v{vi}"] = _downsample(full)
                        manifest["items"].append(
                            {"sample": si, "modality": mod, "backbone": bb, "answer": a,
                             "fold": i, "n_views_pooled": int(mag.shape[0]),
                             "n_views_shown": n_show,
                             "v": float(v), "margin": float(m.item())})
                del x, feats, pooled
        del embed, transform
        print(f"  backbone {bb}: saliency done", flush=True)

    for si, samp in enumerate(samples):
        np.savez_compressed(os.path.join(OUT, f"sal_{si}_{samp['us_id']}.npz"), **store[si])
    with open(os.path.join(OUT, "saliency_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {len(samples)} sample npz + manifest to {OUT}/")


# --------------------------------------------------------------------------- #
# plot (local): per-(sample, modality) sheet — rows = 16 verifiers, cols = folds
# --------------------------------------------------------------------------- #
def _heat_cmap():
    # transparent -> warm, so low saliency lets the gray frame show through
    return LinearSegmentedColormap.from_list(
        "sal", [(0, 0, 0, 0), (0.85, 0.2, 0.05, 0.55), (1.0, 0.85, 0.1, 0.95)])


def _overlay(ax, frame, sal, maxdim=256):
    # panels are ~280px; downsample by nearest striding (fast, keeps the NaN mask)
    g = frame.mean(axis=2) if frame.ndim == 3 else frame
    step = max(1, round(max(g.shape) / maxdim))
    g = g[::step, ::step]
    ax.imshow(g, cmap="gray", interpolation="nearest")
    if sal is not None:
        s = sal[::step, ::step].astype(np.float32)
        finite = np.isfinite(s)
        if finite.any():
            hi = np.nanpercentile(s[finite], 99) or 1.0
            s = np.clip(s / (hi + 1e-12), 0, 1)
        ax.imshow(s, cmap=_heat_cmap(), interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])


def _sheet(store, manifest, si, mod, folds_disp, path):
    backbones = sorted({it["backbone"] for it in manifest["items"]})
    n_ans = 1 + max(it["answer"] for it in manifest["items"])
    rows = [(bb, a) for bb in backbones for a in range(n_ans)]      # 16 verifiers
    frame = store.get(f"frame__{mod}__v0")                          # single shown view
    ncol, nrow = len(folds_disp), len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol + 1.8, 1.7 * nrow),
                             squeeze=False)
    vmeta = {(it["backbone"], it["answer"], it["fold"]): it for it in manifest["items"]
             if it["modality"] == mod and it["sample"] == si}
    for r, (bb, a) in enumerate(rows):
        for c, fd in enumerate(folds_disp):
            fold = fd - 1
            ax = axes[r][c]
            if frame is None:
                ax.axis("off"); continue
            _overlay(ax, frame, store.get(f"map__{mod}__{bb}__a{a}__f{fold}__v0"))
            meta = vmeta.get((bb, a, fold))
            if meta:
                ax.set_xlabel(f"v={meta['v']:.2f}", fontsize=7, labelpad=1)
            if r == 0:
                ax.set_title(f"fold {fd}", fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{bb}#a{a}", fontsize=7.5, rotation=0,
                              ha="right", va="center", labelpad=34)
    s = manifest["samples"][si]
    cls = "abnormal" if s["label"] == 1 else "normal"
    fig.suptitle(f"Saliency  ∂(margin)/∂x  —  {mod}  —  sample {si} "
                 f"({cls}, us={s['us_id']})  —  λ={manifest['lambda']}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.05, 0, 1, 0.985))
    fig.savefig(path, dpi=115, bbox_inches="tight"); plt.close(fig)


def _load_npz(fp):
    # the .npz lives on a slow network share; np.load random-accesses each member
    # (very slow over the network). Stream the whole file into memory once, then
    # decompress locally.
    import io
    with open(fp, "rb") as fh:
        buf = io.BytesIO(fh.read())
    npz = np.load(buf, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def shrink():
    """One-off: compact the big full-resolution .npz (from an older compute run)
    into small display-resolution .small.npz so re-plotting is fast."""
    with open(os.path.join(OUT, "saliency_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    for s in manifest["samples"]:
        si = s["idx"]
        out = os.path.join(OUT, f"sal_{si}_{s['us_id']}.small.npz")
        if os.path.exists(out):
            print(f"  skip {os.path.basename(out)} (exists)", flush=True)
            continue
        big = os.path.join(OUT, f"sal_{si}_{s['us_id']}.npz")
        print(f"  reading {os.path.basename(big)} ...", flush=True)
        store = _load_npz(big)
        small = {k: _downsample(v) for k, v in store.items()}
        np.savez_compressed(out, **small)
        print(f"  wrote {os.path.basename(out)}", flush=True)
    print("shrink done")


def plot():
    with open(os.path.join(OUT, "saliency_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    folds_disp = manifest["folds_display"]
    written = 0
    for s in manifest["samples"]:
        si = s["idx"]
        small = os.path.join(OUT, f"sal_{si}_{s['us_id']}.small.npz")
        big = os.path.join(OUT, f"sal_{si}_{s['us_id']}.npz")
        store = _load_npz(small if os.path.exists(small) else big)
        for mod in MODALITIES:
            fp = os.path.join(OUT, f"sal_{si}_{s['us_id']}_{mod}.png")
            _sheet(store, manifest, si, mod, folds_disp, fp)
            written += 1
    print(f"wrote {written} contact sheets to {OUT}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compute", "plot", "shrink"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--patients", nargs="*", default=None,
                    help="override auto-selection with these US ids")
    args = ap.parse_args()
    if args.mode == "compute":
        compute(args.device, args.patients)
    elif args.mode == "shrink":
        shrink()
    else:
        plot()


if __name__ == "__main__":
    main()
