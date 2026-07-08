# FUSE-Medical-MM — project status & handoff

Single-file catch-up for a fresh session. Companion to `docs/FUSE_TCI_spec.md`
(the math). Everything below is implemented and on `main` unless marked TODO.

## Goal
Unsupervised **verifier ensembling** (FUSE — Candès/Lee et al.; Jaffe et al. 2015
for the estimator) on **multimodal breast imaging** (mammography **MG** +
ultrasound **US**). Verifiers = (modality, backbone) + a trained head → `v∈[0,1]`;
combine them with **zero labels** on the target, using conditional-independence
structure. Task = binary **normal vs abnormal** per patient.

## Environment (see [[wexac-environment-setup]], [[run-training-on-wexac]])
- All training/compute on **WEXAC** via `bsub` (conda env `fuse_mm`, Py3.12,
  torch 2.12+cu126). Local Windows `.venv` (Py3.14) is only for quick data-layer
  / numpy checks and report generation. **Never train locally.**
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
- `10_ensemble_weights` → `artifacts/reports/ensemble_weights/`.
- `12_cv` (**CV**) + `13_cv_merge` → `artifacts/reports/cv/`.
- FUSE **stage-2/3** = `src/fuse_mm/fuse/estimate.py` (MoM → posterior); the
  methods live in `src/fuse_mm/bench/methods.py`.

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
2. **`fuse_ens` is the best FUSE variant and IMPROVES with λ**, unlike the bare
   `fuse` posterior which declines. Sweet spot ~**λ=0.3** (unlabeled bAcc ~0.917;
   test ~0.89–0.90). Competitive with / above the supervised baselines.
3. **TCI training lowers the `L_TCI` *metric* (1.25→0.056) but does NOT achieve
   true conditional independence** — the direct conditional correlation (given y)
   *worsens* monotonically with λ: leading eigenvalue grows (y=1: 4.28→5.75),
   effective rank shrinks (19.3→16.1), condition number grows. `L_TCI` is a
   marginal-moment surrogate (necessary, not sufficient). So the win comes from
   the **robust ensemble predictor**, not from achieving CI.
4. NCL twins are used as +/− **contrasts** in the learned f_θ weights; weight mass
   broadens from US-concentrated (λ=0) → balanced (λ=0.3) as verifiers decorrelate.

## CURRENT STATE (in progress)
- **Cross-validation** just implemented: 10-fold rotation (S_L=fold i,
  Test=fold i+1 circular, S_U=rest → full out-of-fold), retrains the pipeline per
  fold, benchmarks all methods on S_L/S_U/Test, aggregates per-fold mean±std +
  pooled OOF. Runs on WEXAC (`scripts/wexac/cv.lsf`, one job per λ via
  `LAMBDAS`, then `13_cv_merge`). λ ∈ {0,0.1,0.2,0.3,0.5,1}. **Smoke-tested OK.**
- **Expanded metrics** (`bench/metrics.score_stats`): TP/TN/FP/FN, acc,
  balanced_acc, sensitivity/recall, specificity, precision, NPV, F1, MCC,
  Youden's J, AUC, AP.
- `fit_and_score` core extracted from `train_fuse` so CV reuses training.
- Reports reorganized into `artifacts/reports/{benchmark,conditional_corr,ensemble_weights,cv}/`.

## Open / next
- Run + read the CV (mean±std + OOF across λ) — the robust with/without-TCI verdict.
- Possible: α×λ mini-grid; more f_θ families (softmax-avg, MLP); mask US
  annotations if the shortcut ever matters (probe said it's minor).
