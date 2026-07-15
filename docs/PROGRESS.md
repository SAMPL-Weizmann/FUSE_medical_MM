# FUSE-Medical-MM — project status & handoff

Single-file catch-up for a fresh session. Companion to `docs/FUSE_TCI_spec.md`
(the math). Everything below is implemented and on `main` unless marked TODO.

## ⇢ LATEST SESSION HANDOFF (2026-07-14) — START HERE
**Run everything on WEXAC** (unchanged, [[run-training-on-wexac]]); local `.venv`
only for tiny no-I/O checks. NOTE: the Grad-CAM `.npz` are now small (~11 MB, the
≤320 px cap) so `18_gradcam.py plot` runs fine LOCALLY — no LSF needed for figures.

**DONE this session — Grad-CAM (`18_gradcam.py`), both CV configs run on WEXAC**
- Grad-CAM `ReLU(Σ αₖ Aₖ)`, class-discriminative companion to saliency (`17`).
  CNN target = last conv map; **ViT target = last block's PRE-ATTENTION LN
  `norm1`**. Targeting the block *output* gave identically-BLANK ViT maps (energy
  0.000): under CLS pooling the last block's patch tokens feed nothing, so their
  gradient is 0. `norm1`'s patch tokens reach CLS via that block's attention →
  nonzero (norm2 wouldn't — its MLP is token-wise). Confirmed: pooling log shows
  all 5 ViTs token/CLS + 3 CNNs avg; ViT CAM energy **0.000 → ~0.67** (≈ CNN 0.73).
- **Role-based columns**: each patient has ONE home fold, so its CV role varies by
  iteration. Columns = `original` + this patient's **labeled / best-unlabeled /
  test** fold. Fold "performance" = that iteration's fuse_ens test bAcc, computed
  in Stage A (which retrains ALL folds). cv20 test col = better of the 2 held-out.
- Two `--config` presets, both ran + rendered: **cv10** (10-fold, n_test=1, λ=0.2
  → `artifacts/reports/gradcam/`, `gradcam.lsf`) and **cv20** (20-fold, n_test=2,
  λ=0 → `artifacts/reports/gradcam20/`, `gradcam20.lsf`). Figures =
  `gradcam{,20}_figures.lsf` or local `plot`.
- LSF gotcha fixed: `bsub A && bsub B` only chains the *submission* (bsub returns
  at queue time), so figure jobs ran before compute wrote the npz (stale/empty
  output). Figure jobs now carry `#BSUB -w "done(fuse_gcam10|20)"` to wait.
- Qualitative: CNNs (convnext/resnet50) clean focal localization on the lesion;
  ViTs alive but diffuse (dinov2/vit_b16 speckly; biomedclip#a1/mae/rad_dino
  coherent); efficientnet blocky/least reliable. Strong verifiers localize to the
  SAME place in labeled vs test columns → stable, not a memorization artifact.

**PENDING (next session)**
- Saliency (`17`) figures: compute ran, PNG render not confirmed this session.
- Optional deeper look: Grad-CAM MG sheets + normal-class samples (2, 3).

## PRIOR SESSION (2026-07-13)
**Run everything on WEXAC** (updated preference, [[run-training-on-wexac]]): the
Windows-mounted isi share is slow and STALLS on large files (a 615 MB `.npz`
`np.load` took ~567 s and wedged a process in unkillable kernel I/O). Local
`.venv` is ONLY for tiny no-I/O checks (`py_compile`, parsing a small json/npz).
Anything that trains, runs backbones, or reads/writes sizeable artifacts →
LSF job under `scripts/wexac/`.

**DONE this session**
- **CV figures + tables (10-fold)**: `14_cv_plots.py` → `artifacts/reports/cv/*.png`
  (vs-λ [+ means / dodged-std variants], ranking dot plot, soft-vs-binary, metric
  grids; Okabe-Ito palette, dot plots not truncated bars). `15_cv_tables.py` →
  `cv_table_{lambda0,top3_by_bacc,top3_by_auc,top3_merged}.{png,csv}` (mean±std %).
- **Merged top-3 table** (`table_merged_top3` in `15_cv_tables.py`): union of
  top-3-by-bAcc and top-3-by-AUC, deduped by (method,λ), `top-3 in`=bAcc/AUC/both,
  **column-max bolded**. Generated for BOTH `cv/` and `cv20/`.
- **Per-fold ensemble weights** `16_cv_ensemble_weights.py` (`dump` WEXAC
  `cv_ens_weights.lsf`, `plot` local): top-3 folds (7,3,6) at λ=0.2, 2 orderings
  (best-fold θ / lexicographic), per-verifier dividers →
  `artifacts/reports/ensemble_weights/cv_ens_weights_lambda_0.2_*.png`. Confirms
  NCL twins as ± contrasts (biomedclip #a1 +, #a0 −) consistently across folds.
- **20-fold CV (5/85/10)**: rotation parametrized (`--folds 20 --n-test 2`),
  isolated to `cv_folds_20.json` + `artifacts/reports/cv20/`. Ran on WEXAC
  (`cv20.lsf` fan-out → `13_cv_merge --base cv20` → `cv20_report.lsf`). n_test=2
  ⇒ repeated OOF (each patient in pooled test 2×). Result below.
- **n_classes provenance investigated**: verifiers use `n_classes=2` (softmax+CE,
  `v=σ(ℓ₁−ℓ₀)`). Git/docs show it was an UNEXAMINED default (n_classes inferred
  from #labels at scaffolding), never user-approved nor a deliberated verdict.
  Supervisor intended `n_classes=1` (single-logit sigmoid). **NOT changed.**

**IN PROGRESS / PENDING (do first next session)**
- **Saliency figures NOT yet rendered.** `17_saliency.py compute` already ran on
  WEXAC (`saliency.lsf`) → `artifacts/reports/saliency/sal_<i>_<us>.npz`
  (currently ~615 MB each — MG maps were saved at full mammogram res; `compute`
  now downsamples ≤320 px so future runs are small) + `saliency_manifest.json`.
  **TO GET THE 8 FIGURES: `bsub < scripts/wexac/saliency_figures.lsf`** (plots on
  WEXAC where the big-npz read is fast) → `sal_<i>_<us>_{US,MG}.png` (rows=16
  verifiers, cols=folds 7/3/6). `shrink` mode / `.small.npz` only for local reads.
- Optional: confirm 20-fold λ-shape from `cv20/cv_balanced_acc_vs_lambda.png`;
  record the 5/85/10 finding in [[fuse-status-findings]].

**KEY 20-FOLD RESULT (from `cv20/cv_table_top3_merged`)**
- `fuse_ens` best λ shifted **0.2 → 0** (TCI helps even less with smaller S_L).
  Scores drop & fold variance ~2× vs 10-fold (fuse_ens test bAcc 91.7±1.6 →
  87.0±3.7) — expected from S_L 10%→5% (~159→~80 labeled patients; heads are
  label-trained). Ranking stable: fuse_ens/fuse/weaver/oracle top; oracle (ceiling)
  owns AUC/recall. Reinforces: **λ=0 is the honest operating point.**

**DECISIONS / RULED OUT**
- **λ=0 honest operating point** (10- AND 20-fold): TCI (λ>0) gives no robust gain.
- **Saliency**: signal = `∂(ℓ₁−ℓ₀)/∂x` (de-saturated `∂v/∂x`), model n_classes=2;
  overlay = Option A (raw frame, inverse resize+crop); scope λ=0.2, all 32
  verifiers, MG shows view-0 but pools 4 for the score, folds 7/3/6, 4 auto-picked
  samples (2 abnormal + 2 normal).
- `n_classes=1` switch = open supervisor call; would need a full pipeline re-run
  (numbers shift slightly — different parameterization + weight decay + init).

## Goal
Unsupervised **verifier ensembling** (FUSE — Candès/Lee et al.; Jaffe et al. 2015
for the estimator) on **multimodal breast imaging** (mammography **MG** +
ultrasound **US**). Verifiers = (modality, backbone) + a trained head → `v∈[0,1]`;
combine them with **zero labels** on the target, using conditional-independence
structure. Task = binary **normal vs abnormal** per patient.

## Environment (see [[wexac-environment-setup]], [[run-training-on-wexac]])
- **Everything runs on WEXAC** via `bsub` (conda env `fuse_mm`, Py3.12,
  torch 2.12+cu126) — training, compute, AND report/plot generation. Local
  Windows `.venv` (Py3.14) is ONLY for tiny no-I/O checks. See the handoff block
  above (slow/stalling share) and [[run-training-on-wexac]].
- Repo on WEXAC: `/home/projects/yonina/safitl/FUSE_medical_MM` (= Windows `X:`,
  shared storage — edits are live on both instantly, no push/pull needed to run).
- Data: `/home/hsd/...` (Windows `Z:`), via env vars in `scripts/wexac/env.sh`.
- Weaver needs `metal-ama` (installed via `scripts/wexac/setup_weaver.sh`).

## Data & cohort
- 8 backbones × 2 modalities. **Selection** (`src/fuse_mm/selection.py`, mirrors
  the prior team): MG = 4 canonical views, US = 1 frame — neutralizes the
  label-correlated scan-count confound. Cohort ≈ **1594 unique** paired patients
  (binary_abnormal ~38% normal / 62% abnormal).
- Features extracted ONCE per patient (frozen backbones, split-independent) →
  `artifacts/features/{labeled,unlabeled,test}/{MG,US}/<backbone>.npz`.

## Verifiers
- 8 backbones: resnet50, efficientnet_b0, vit_b16, convnext_tiny, dinov2_vits14,
  mae_vit_b16, biomedclip (open_clip), rad_dino (hf). US generally >> MG in AUC.
- Each (modality×backbone) head emits **n_answers=2** (NCL). With **α_NCL=3**
  the two answers stay distinct (corr ~0.1); α=0.1 collapsed them (corr ~0.98).
- **32 verifiers** = 8 × 2 modalities × 2 answers.

## Pipeline & scripts
- `01_make_splits` → splits.json (single 10/80/10 split).
- `02_extract_features` (WEXAC GPU) → per-set feature npz.
- `03_train_heads` → independent per-head CV baselines (Step-5 era).
- `04_train_fuse` (**stage-1**): joint training of all 32 verifiers,
  `L = L_CE(S_L) + α·L_NCL(S_L) + λ·L_TCI(S_U)`. Output λ-keyed:
  `artifacts/fuse/lambda_<L>/verifier_scores_{labeled,unlabeled,test}.npz`.
- `05_benchmark` → methods × sets table on a scores dir.
- `06_compare_benchmarks`, `07_collect_sweep` → sweep tooling.
- `08_report` → `artifacts/reports/benchmark/` (tables + sweep plots).
- `09_conditional_corr`, `11_cond_corr_spectrum` → `artifacts/reports/conditional_corr/`.
- `10_ensemble_weights` (single-split) + `16_cv_ensemble_weights` (per-fold f_θ
  weights for the CV's top folds; `dump` on WEXAC, `plot` local) →
  `artifacts/reports/ensemble_weights/`.
- `17_saliency` (verifier input-gradient saliency `∂(ℓ₁−ℓ₀)/∂x` at λ=0.2, all 32
  verifiers, folds 7/3/6, 4 samples; MG pools 4 views but shows view-0; overlaid
  on the raw frame via inverse resize+crop; `compute` on WEXAC = `saliency.lsf`,
  `plot` local) → `artifacts/reports/saliency/`.
- `18_gradcam` (verifier **Grad-CAM** `ReLU(Σ αₖ Aₖ)`, class-discriminative,
  companion to `17`; taps each backbone's last conv map, ViTs the last block's
  PRE-ATTENTION LN `norm1` — targeting the block *output* gives blank ViT maps
  because CLS pooling zeros the patch-token gradient; CNN-vs-ViT auto-detected
  from activation rank). **Role-based columns**: each patient has one home fold,
  so its CV role varies by iteration — columns are `original` + the patient's
  **labeled** / best **unlabeled** / **test** fold (fold "performance" = that
  iteration's fuse_ens test bAcc, computed in Stage A which retrains ALL folds).
  Two `--config` presets: **cv10** (10-fold, n_test=1, λ=0.2, WEXAC `gradcam.lsf`
  → `artifacts/reports/gradcam/`) and **cv20** (20-fold, n_test=2, λ=0, WEXAC
  `gradcam20.lsf` → `artifacts/reports/gradcam20/`; test column = better of the 2
  held-out folds). Reuses `17`'s selector/geometry/overlay + the SAME 4 patients
  (via `saliency_manifest.json`). `plot` = `gradcam{,20}_figures.lsf` or local.
- `12_cv` (**CV**) + `13_cv_merge` + `14_cv_plots` (CV figures) + `15_cv_tables`
  (CV summary tables, PNG+CSV) → `artifacts/reports/cv/`. CV rotation is
  parametrized: `--folds N --n-test T` (S_L=1 fold, Test=next T folds, S_U=rest).
  Default 10/n_test=1 → 10/80/10 (`cv_folds.json`); **20/n_test=2 → 5/85/10**
  (`cv_folds_20.json`, out `artifacts/reports/cv20/`, WEXAC `cv20.lsf`). n_test>1
  is a repeated OOF (each patient in the pooled test T times).
- FUSE **stage-2/3** = `src/fuse_mm/fuse/estimate.py` (MoM → posterior); the
  methods live in `src/fuse_mm/bench/methods.py`.

## FIGURE CATALOG (what each plot in `artifacts/reports/<dir>/` is)
`<L>` = a λ value; `<k>_<pid>` = Grad-CAM sample (k=0,1 abnormal · 2,3 normal;
pid=US id: 0=1205002637591, 1=1205002616396, 2=1205003172923, 3=1205004142062).

**`benchmark/`** — single 10/80/10 split (`08_report`; SUPERSEDED by CV, kept for history):
- `sweep_{test,unlabeled}_{acc,balanced_acc}.png` — method × λ line sweeps on the
  single split (accuracy / balanced accuracy, on Test / S_U). Source of the old
  (retracted) "λ=0.3 sweet spot" claim.
- `verifier_auc_test_lambda_{0.0,1.0}.png` — per-verifier Test AUC bars at λ=0 / 1
  (shows US ≫ MG, and how AUC shifts with λ).

**`conditional_corr/`** — does TCI reach conditional independence? (`09`, `11`):
- `cond_corr_lambda_<L>_y{0,1}.png` — verifier×verifier correlation *given y* at each
  λ, per class. **Worsens with λ** → TCI does NOT achieve CI.
- `cond_corr_leading_eigvec.png` — leading eigenvector vs λ.
- `cond_corr_scree.png` — eigenvalue spectrum vs λ (effective rank shrinks as λ↑).

**`ensemble_weights/`** — learned f_θ verifier weights:
- `ensemble_weights_lambda_<L>.png` — single-split f_θ weights at each λ (`10`).
- `cv_ens_weights_lambda_0.2_by_bestfold.png` / `_lexicographic.png` — per-fold f_θ
  for the CV top-3 folds (7,3,6), verifiers ordered by best-fold θ / alphabetically
  (`16`). **NCL twins = ± contrasts** (biomedclip #a1 +, #a0 −).

**`cv/`** — 10-fold CV, S_L/S_U/Test = 10/80/10 (`14` plots, `15` tables):
- `cv_balanced_acc_vs_lambda.png` — Test bAcc vs λ per method (error-bar lines).
- `cv_balanced_acc_vs_lambda_dodged.png` — same, dodged error bars (uncluttered).
- `cv_balanced_acc_vs_lambda_means.png` — same, **means only** (clean, slide-ready).
- `cv_metric_grid.png` — methods × ALL metrics heat/grid at the ranking λ.
- `cv_metric_grid_bacc_auc.png` — compact grid, **bAcc & AUC only** (slide-ready).
- `cv_ranking_test.png` — Cleveland dot plot: every method ranked by Test metric.
- `cv_soft_vs_binary.png` — soft vs g_τ-binarized FUSE (**soft ≫ binary**).
- `cv_table_lambda0.png` — method table at λ=0 (mean±std %).
- `cv_table_top3_by_bacc.png` / `_by_auc.png` — top-3 methods by Test bAcc / AUC.
- `cv_table_top3_merged.png` — union of the two top-3s, **column-max bolded**.

**`cv20/`** — identical figure set for the **20-fold 5/85/10** rotation (λ=0 emphasis).

**`gradcam/`** — Grad-CAM, **10-fold λ=0.2** (`18 --config cv10`):
- `gcam_<k>_<pid>_{US,MG}.png` — full **16-verifier** sheet; columns = `original` +
  this patient's **labeled / unlabeled / test** fold (per-patient role-based).
- `gcam_<k>_<pid>_{US,MG}_crop.png` — cropped to convnext_tiny/biomedclip/resnet50
  (6 rows, **slide-legible**).

**`gradcam20/`** — same Grad-CAM sheets for **20-fold λ=0** (`18 --config cv20`;
Test column = better of the patient's 2 held-out folds).

**`saliency/`** — input-gradient `∂(ℓ₁−ℓ₀)/∂x` sheets, 10-fold λ=0.2 (`17`).
⚠️ **compute done, PNGs NOT rendered** — would be `sal_<k>_<pid>_{US,MG}.png` via
`bsub < scripts/wexac/saliency_figures.lsf` (npz still ~615 MB → render on WEXAC).

## Methods benchmarked (all on the frozen verifier scores)
Unsupervised: majority_vote, naive_ensemble, **fuse**, **fuse_bin**, **fuse_ens**,
**fuse_full**. Supervised (use S_L labels): obv, weaver (real metal-ama),
logistic, gaussian_nb. Ceiling: oracle. FUSE has two flags (Algorithm 1):
- `binarize` (g_τ, steps 2-3): per-verifier thresholds minimizing TCI → binary Ṽ.
- `optimize_ensemble` (f_θ, steps 5-7): fit a logistic ensemble to the FUSE
  posterior. **eq (7) is a ranking objective that degenerates in classification**
  → implemented as **confidence-weighted logistic regression to the hard
  pseudo-labels 1[p̂>0.5], weighted by |2p̂−1|** (the well-posed surrogate; can
  beat p̂, unlike a distillation-to-p̂ which just copies it).

## KEY EMPIRICAL FINDINGS (see [[fuse-status-findings]])
1. **Soft > binary.** Trained heads emit informative soft probs; `g_τ`
   binarization (`fuse_bin`/`fuse_full`) *hurts*. So use soft `2v−1`.
2. **`fuse_ens` is the best FUSE variant** (beats naive/majority by ~1.5–3 pts,
   ≈ weaver ≈ oracle). ⚠️ SUPERSEDED: the single-split "IMPROVES with λ, sweet
   spot λ=0.3" claim did NOT survive CV — see **CV VERDICT** below. λ=0 is the
   honest operating point (10- and 20-fold). Retained only to flag the reversal.
3. **TCI training lowers the `L_TCI` *metric* (1.25→0.056) but does NOT achieve
   true conditional independence** — the direct conditional correlation (given y)
   *worsens* monotonically with λ: leading eigenvalue grows (y=1: 4.28→5.75),
   effective rank shrinks (19.3→16.1), condition number grows. `L_TCI` is a
   marginal-moment surrogate (necessary, not sufficient). So the win comes from
   the **robust ensemble predictor**, not from achieving CI.
4. NCL twins are used as +/− **contrasts** in the learned f_θ weights; weight mass
   broadens from US-concentrated (λ=0) → balanced (λ=0.3) as verifiers decorrelate.

## CURRENT STATE
- **Cross-validation DONE and merged** (`artifacts/reports/cv/cv_results.json` +
  `cv_summary.csv`, merged from the 6 per-λ fan-out jobs via `13_cv_merge`).
  10-fold rotation (S_L=fold i, Test=fold i+1 circular, S_U=rest → full
  out-of-fold), retrains the pipeline per fold, benchmarks all methods on
  S_L/S_U/Test, aggregates per-fold mean±std + pooled OOF. λ ∈ {0,0.1,0.2,0.3,0.5,1}.
- **Expanded metrics** (`bench/metrics.score_stats`): TP/TN/FP/FN, acc,
  balanced_acc, sensitivity/recall, specificity, precision, NPV, F1, MCC,
  Youden's J, AUC, AP.
- `fit_and_score` core extracted from `train_fuse` so CV reuses training.
- Reports reorganized into `artifacts/reports/{benchmark,conditional_corr,ensemble_weights,cv}/`.

## CV VERDICT (the robust with/without-TCI answer)
Under honest 10-fold OOF, **TCI training (λ>0) gives no robust improvement** — the
single-split "sweet spot λ=0.3" does NOT survive CV. Test balanced_acc (fold
mean±std), best methods:

| method         | λ=0        | λ=0.1      | λ=0.2      | λ=0.3      | λ=0.5      | λ=1.0      |
|----------------|------------|------------|------------|------------|------------|------------|
| **fuse_ens**   | .910±.026  | **.917**±.015 | **.917**±.016 | .913±.016 | .908±.014 | .910±.018 |
| fuse           | .907±.022  | .907±.017  | .905±.014  | .900±.016  | .902±.015  | .898±.020  |
| weaver         | **.916**±.018 | .901 | .895 | .897 | .891 | .894 |
| naive_ensemble | .895       | .893       | .891       | .889       | .891       | .890       |
| oracle         | .890       | .891       | .893       | .894       | .898       | .902       |

- `fuse_ens` peaks at **λ=0.1–0.2** (0.917) but the bump over λ=0 (0.910) is ~1 SE
  (SE≈std/√10≈0.006) — **within noise**. On **unlabeled OOF** it's worse: fuse_ens
  *monotonically declines* 0.902(λ=0)→0.892(λ=1); bare `fuse` declines on both sets.
  The earlier "unlabeled 0.917 @ λ=0.3" was in-sample S_U, not OOF.
- **Robustly true:** (a) `fuse_ens` is the best FUSE variant and beats naive/majority
  by ~1.5–3 pts; (b) **soft ≫ binary** — `fuse_bin` (.83–.85) / `fuse_full` (.85–.88)
  clearly worse; (c) `fuse_ens`(λ=0) ≈ `weaver`(λ=0) ≈ `oracle`, i.e. the robust
  ensemble predictor already reaches the supervised/ceiling band **with zero TCI**.
- Consistent with finding #3 (TCI lowers its own surrogate metric but doesn't buy
  CI): here it also doesn't buy accuracy. **The win is the ensemble predictor, not λ.**

## Open / next
- **FIRST: `bsub < scripts/wexac/saliency_figures.lsf`** — render the 8 saliency
  sheets (compute already done on WEXAC; only plotting is left). See handoff.
- Confirm the 20-fold λ-shape (`cv20/cv_balanced_acc_vs_lambda.png`) and record
  the 5/85/10 finding.
- Decide the headline framing: FUSE-style robust ensembling works (zero labels,
  ≈ supervised), but the L_TCI regularizer is not the source of the gain. Consider
  presenting λ=0 as the honest recommended operating point.
- Possible: α×λ mini-grid; more f_θ families (softmax-avg, MLP); mask US
  annotations if the shortcut ever matters (probe said it's minor).
- Optional: store per-fold arrays (not just mean/std) so paired significance tests
  on the λ deltas are possible.
