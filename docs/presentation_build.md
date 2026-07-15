# FUSE-Medical-MM — presentation build doc

Self-contained slide-by-slide spec to rebuild the deck from the original 9-slide
version, updated with everything we actually implemented + the results. Paste
this into your slide tool (or hand it to another AI). Every figure path, on-slide
text, and caption is here — no external lookup needed.

**Design language (keep from the original):** navy (#0E2233) serif titles, teal
(#0E9384) accent, soft rounded pastel cards (blue / mint / purple), lots of white
space. **Minimal text** — short bullets or a scheme, one takeaway line per
results slide. Figures are the content; captions carry the message.

**Deck map (≈15 slides):**
`1 Title · 2 Problem · 3 Split · 4 Verifiers · 5 TCI · 6 Objective · 7 Pipeline`
→ results → `R1 Operating point · R2 Method comparison · R3 Soft≫binary ·
R4 Ensemble weights · R5 20-fold robustness · R6 Grad-CAM 10-fold ·
R7 Grad-CAM 20-fold · R8 Takeaways`. (Original slides 8 & 9 removed.)

Key project facts used throughout: **US + Mammography**, binary **normal vs
abnormal (breast)**, cohort **≈1594 paired patients** (~38% normal / 62%
abnormal), **8 backbones × 2 modalities × 2 NCL answers = 32 verifiers**, split
**S_L/S_U/Test = 10/80/10** (20-fold variant 5/85/10).

---

## PART 1 — keep slides 1–7, tweaked

### Slide 1 — Title  *(keep as-is)*
- **FUSE - Medical MM** / *Unsupervised Verifier Ensembling · Multimodal Medical Imaging*.
- ⚠️ verify the arXiv id (`2604.18547` looks like a placeholder).

### Slide 2 — The Problem  *(narrow to the real instantiation)*
ON SLIDE (bullets, left) + the modality-stack scheme (right, keep the visual):
- Many trained classifiers, **quality varies**
- **Ground-truth labels are scarce** on new patients
- Here: **Ultrasound + Mammography**, task = **normal vs abnormal (breast)**
- Right-side stack: keep **Ultrasound** + **Mammography** solid; gray **X-ray / CT / …** as "extensible" → **?**
- Bottom band: *Goal: combine verifiers optimally — with **zero labels** on the target patient.*

### Slide 3 — Three-Way Data Split  *(put in real numbers)*
Keep the 3-card scheme (S_L / S_U / Test). Edit the sizes/bullets:
- **S_L — Labeled** · *Size: ~10%* · train classifier heads; stratified sample.
- **S_U — Unlabeled pool** · *Size: ~80%* · TCI + FUSE estimation.
- **Test** · *one patient at a time* · frozen verifiers, estimated statistics.
- Add a thin footer: *Evaluated by full cross-validation (10-fold rotation); robustness variant 20-fold = 5/85/10.*

### Slide 4 — What are the verifiers?  *(use the ACTUAL 8 backbones + NCL head)*
Keep the Modality → Backbone → Head scheme. Edit contents:
- **Modality:** Ultrasound (1 frame) · Mammography (4 views, pooled).
- **Pre-trained backbone (frozen), 8 spanning objective × domain:** resnet50,
  efficientnet_b0, convnext_tiny, vit_b16 (supervised/natural); dinov2, mae
  (self-sup/natural); **biomedclip, rad_dino (medical)**.
- **Head = NCL, 2 "answers" per verifier** (twin heads kept decorrelated, α_NCL=3);
  output vⱼ = σ(ℓ₁−ℓ₀) ∈ [0,1].
- Footer: **32 verifiers = 8 backbones × 2 modalities × 2 answers.**

### Slide 5 — TCI: Diversity Condition  *(keep the math as-is)*
- vₐ ⊥ v_b ⊥ v_c | diagnostic; the L_TCI formula; computed on S_U.
- *(Optional one-liner to foreshadow R1: "We test whether enforcing this actually helps →")*

### Slide 6 — Training Objective  *(ADD the NCL term)*
- Headline formula, updated: **L = L_CE(S_L) + α·L_NCL(S_L) + λ·L_TCI(S_U)**
- **L_CE — on S_L:** cross-entropy, trains the heads (small set, just needs to be accurate).
- **L_NCL — on S_L:** keeps the 2 answers per verifier decorrelated (negative-correlation learning).
- **L_TCI — on S_U:** variance of normalized triplet covariances; penalizes correlated verifiers.

### Slide 7 — Pipeline (Train → Estimate → Infer)  *(keep; one addition)*
- Keep the 3-stage scheme verbatim.
- In ESTIMATE, note the predictor: **fuse_ens = method-of-moments → posterior → f_θ ensemble** (our best FUSE variant).

*(Remove original slides 8 "heatmap schematic" and 9 "Key Properties".)*

---

## PART 2 — Results

> Legend used on R6/R7 (put once, small, on the first Grad-CAM slide):
> *Columns = the same patient's CV role: **labeled** (head trained on this
> patient's label) · **unlabeled** (seen, unlabeled) · **test** (fully held out).
> Warm = Grad-CAM evidence for "abnormal"; v = verifier score.*

### R1 — The honest operating point
- **Figure:** `artifacts/reports/cv/cv_balanced_acc_vs_lambda_means.png`
- ON SLIDE title: *Does the TCI regularizer help?*
- **Takeaway:** fuse_ens is essentially **flat in λ**; the λ=0.1–0.2 bump (.917) over
  λ=0 (.910±.026) is within ~1 SE → **λ = 0 is the honest operating point.**

### R2 — Method comparison (zero labels ≈ supervised)
- **Figures (two panels):** `artifacts/reports/cv/cv_metric_grid_bacc_auc.png`
  (left) + `artifacts/reports/cv/cv_table_top3_merged.png` (right).
- ON SLIDE title: *Zero target labels, supervised-level accuracy.*
- **Takeaway:** **fuse_ens ≈ weaver ≈ oracle** on test bAcc (~.91–.92), and beats
  naive/majority by ~1.5–3 pts — the robust ensemble already reaches the
  supervised/ceiling band with **no labels on the target**.

### R3 — Soft ≫ binary
- **Figure:** `artifacts/reports/cv/cv_soft_vs_binary.png`
- ON SLIDE title: *Keep the verifier scores soft.*
- **Takeaway:** binarizing verifiers (`fuse_bin` .83–.85 / `fuse_full` .85–.88)
  clearly hurts vs soft (~.91). The trained heads carry informative confidence.

### R4 — What the ensemble learns
- **Figure:** `artifacts/reports/ensemble_weights/cv_ens_weights_lambda_0.2_lexicographic.png`
- ON SLIDE title: *Learned ensemble weights (f_θ).*
- **Takeaway:** the **NCL twins act as ± contrasts** (e.g. biomedclip #a1 +, #a0 −)
  consistently across folds; weight mass concentrates on the stronger US verifiers.

### R5 — Robustness: halve the labeled set (20-fold, 5/85/10)
- **Figures (three, small):** `artifacts/reports/cv20/cv_balanced_acc_vs_lambda_means.png`
  + `artifacts/reports/cv20/cv_table_lambda0.png` + `artifacts/reports/cv20/cv_table_top3_merged.png`
- ON SLIDE title: *Same story at 5% labels.*
- **Takeaway:** ranking stays (fuse_ens/fuse/weaver/oracle on top); scores drop and
  fold variance ~2× (fuse_ens **91.7±1.6 → 87.0±3.7**), best λ shifts **0.2 → 0** —
  reinforces **λ = 0** as the honest choice.

### R6 — Interpretability: Grad-CAM (10-fold, λ=0.2)
- **Figures (cropped, slide-legible):**
  `artifacts/reports/gradcam/gcam_0_1205002637591_US_crop.png` (abnormal, US) +
  `.../gcam_0_1205002637591_MG_crop.png` (abnormal, MG); and
  `.../gcam_3_1205004142062_US_crop.png` + `.../gcam_3_1205004142062_MG_crop.png` (normal).
  *(Cropped to convnext_tiny / biomedclip / resnet50 × 2 answers. Full 16-verifier
  sheets — same names without `_crop` — go in a backup slide.)*
- ON SLIDE title: *Where does each verifier look?*
- **Takeaway:** strong verifiers localize the lesion; localization is **stable
  across the labeled vs held-out (test) columns** — not a memorization artifact.

### R7 — Interpretability holds under the harder regime (20-fold, λ=0)
- **Figures (cropped):** `artifacts/reports/gradcam20/gcam_0_1205002637591_US_crop.png`
  + `_MG_crop.png`; `.../gcam_3_1205004142062_US_crop.png` + `_MG_crop.png`.
- ON SLIDE title: *Same localization at λ=0 / 5% labels.*
- **Takeaway:** the lesion-focused maps persist with no TCI and half the labels —
  interpretability tracks the quantitative robustness in R5.

### R8 — Takeaways  *(new closing slide, replaces old "Key Properties")*
Four short cards:
- **Zero-label ensembling works** — fuse_ens reaches supervised/ceiling accuracy with no labels on the target patient.
- **λ = 0 is honest** — the L_TCI regularizer is *not* the source of the gain (holds at 10- and 20-fold).
- **Soft ≫ binary** — keep verifier confidences.
- **Interpretable** — Grad-CAM shows verifiers attend to the lesion, stably across folds.

---

## Backup / appendix slides (optional)
- Full 16-verifier Grad-CAM sheets (`gcam_*_{US,MG}.png`, no `_crop`).
- 10-fold CV verdict table (test bAcc, fold mean±std):

  | method | λ=0 | λ=0.1 | λ=0.2 | λ=1.0 |
  |---|---|---|---|---|
  | fuse_ens | .910±.026 | .917±.015 | .917±.016 | .910±.018 |
  | weaver | **.916**±.018 | .901 | .895 | .894 |
  | naive_ensemble | .895 | .893 | .891 | .890 |
  | oracle | .890 | .891 | .893 | .902 |

## What changed vs the original deck
- Modalities narrowed to US+MG; concrete task + cohort added (slide 2–3).
- Real 8 backbones + **NCL 2-answer head → 32 verifiers** (slide 4).
- **α·L_NCL term added** to the objective (slide 6).
- Removed the schematic heatmap (old 8) and Key Properties (old 9).
- Added 8 results slides (R1–R8) — the CV verdict, ablations, and Grad-CAM.
