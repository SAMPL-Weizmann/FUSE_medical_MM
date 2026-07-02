# FUSE-Medical-MM — TCI training & FUSE estimation spec

Reference for the unsupervised verifier-ensembling stage. Based on the
supervisor's deck (FUSE — Lee et al., arXiv:2604.18547). This captures the
formulation and the six confirmed design decisions.

## Objects
- Patient `x`, latent binary label `Y ∈ {0,1}` (normal / abnormal).
- `m` **verifiers** `v₁…v_m`: each a (modality, backbone) with a linear head
  `vⱼ = σ(wⱼᵀz + bⱼ) ∈ [0,1]` on frozen features `z`.
  Verifier set = **8 backbones × 2 modalities × 2 NCL answers** (decision 4).
- Splits: `S_L` (labeled, small), `S_U` (unlabeled, large), `Test` (one-at-a-time).

## TCI = class-conditional independence (the diversity condition)
`v_a ⊥ v_b ⊥ v_c | Y`. Under CI with binary `Y`, the centered-score moments are
**rank-1** in a per-verifier "signed skill" `qⱼ`, which is what the loss exploits.

### Score & moments (on `S_U`)
Map to `[-1,1]` then center over `S_U` (decision 2):
```
uⱼ  = 2·vⱼ − 1
ũⱼ  = uⱼ − mean_{S_U}(uⱼ)
Σ_{a,b}   = mean_{S_U}( ũ_a ũ_b )        # a ≠ b
T_{a,b,c} = mean_{S_U}( ũ_a ũ_b ũ_c )    # distinct a,b,c
```
Under CI: `Σ_{a,b} = q_a q_b` and `T_{a,b,c} = κ·q_a q_b q_c` (κ = skew of Y), so
```
T_{a,b,c} / Σ_{a,b} = κ · q_c      ← constant over (a,b) for fixed c
```

### L_TCI (on `S_U`, minimized over heads)
Paper convention (Candès et al., Prop 2.4 / eq 4): for each `c`, over pairs `1 ≤ a < b < c`:
```
r_{a,b,c} = T_{a,b,c} / denom(Σ_{a,b}),   denom = sign(Σ)·max(|Σ|, ε)   # ε-floor, decision 5
L_TCI     = Σ_{c=1}^{m}  Var_{(a,b)}( r_{a,b,c} )
```
`Var = 0 ⟺ TCI holds`; its magnitude measures the violation. This is **not**
marginal decorrelation — under CI `Σ_{a,b}=q_a q_b ≠ 0`, so the shared `Y`-signal
is preserved; only the rank-1 3rd-moment *structure* is enforced.

## Training objective
```
L = L_CE(S_L)  +  α · L_NCL(S_L)  +  λ · L_TCI(S_U)
```
- `L_CE`: cross-entropy per verifier/answer on labeled features (keeps them accurate).
- `L_NCL` (**kept, in addition** — decision 3): marginal cosine decorrelation of each
  head's 2 answers on `S_L`. Prevents the two answers from collapsing to identical,
  so they are legitimate *separate* verifiers. Keep `α` **gentle** — NCL is marginal
  while TCI is conditional, and two accurate same-label predictors are necessarily
  marginally correlated, so a large `α` fights accuracy and the shared signal.
- `L_TCI`: the real diversity condition — class-conditional independence across all
  `m=32` verifiers on `S_U`.
- Trainable: all heads `{wⱼ, bⱼ}` **jointly** (L_TCI couples them). Backbones frozen.
- **Warm-start** from the CE(+NCL)-trained heads, then switch on `L_TCI` (decision 6).

## Pipeline
1. **Train** (`S_L` + `S_U`): joint loss above → learn heads.
2. **Estimate** (`S_U`) — Method of Moments (Jaffe et al. 2015, Thm 2.3):
   - `μ = E[v]`; `Σ, T` = 2nd/3rd marginal moments over S_U.
   - Rank-1 factor `Σ_offdiag = u uᵀ`, `u = √(1−b²)(2π−1)`; sign fixed by Assumption 2.1
     (majority of verifiers better than random → most `2π−1 > 0`).
   - `T_offdiag = w⊗w⊗w`, `w = (−2b(1−b²))^{1/3}(2π−1)`; ratio `w/u` (const) → class
     imbalance `b`; then balanced accuracies `π` from `u`.
   - Sensitivity/specificity (eq 3): `ψ = ½(1+μ+u√((1−b)/(1+b)))`,
     `η = ½(1−μ+u√((1−b)/(1+b)))`.
   - Posterior (Prop C.1, eq 13): per triplet `P(y|v_{j1},v_{j2},v_{j3}) ∝ (1+by)∏[…]`,
     averaged over all `C(m,3)` triplets → `p̂(r)`.
   - Predictor: naive-Bayes MLE `ŷ = sign(Σ_j v_j·log(ψ_j(1−ψ_j)/(η_j(1−η_j))) + …)`,
     or the fitted predictor maximizing estimated accuracy (paper p8).
3. **Infer** (Test, inductive, one patient): apply frozen verifiers → posterior → predict + uncertainty.

## Confirmed decisions
1. Loss pairs: for each `c`, all `(a,b)` with `a<b`, both `≠ c`.
2. Continuous scores mapped `2v−1 ∈ [-1,1]`, centered (not thresholded).
3. NCL is kept **in addition** to CE and TCI: `L = L_CE + α·L_NCL(S_L) + λ·L_TCI(S_U)`.
   NCL (marginal, S_L) keeps each head's 2 answers distinct → legitimate separate
   verifiers; TCI (conditional, S_U) enforces the diversity condition. Keep α gentle.
4. Verifiers = all 8 backbones × 2 modalities × 2 NCL answers.
5. ε-floor `|Σ_{a,b}|`; do not drop pairs.
6. Warm-start from CE-trained heads, then joint with `L_TCI`.

## Resolved from the FUSE paper (Candès et al. / Jaffe et al. 2015)
- Loss convention: **`1 ≤ a < b < c`** (Prop 2.4) — implemented.
- Stage-2 estimator: **Method of Moments** as above (Thm 2.3, Prop C.1) — to build.

## Remaining adaptation nuance
- **Binary vs continuous verifiers.** The paper's Step-1 *binarizes* verifier scores via
  thresholds `g_τ → {±1}`, minimizing TCI over `τ`; Thm 2.3 / posterior assume ±1 outputs.
  Our (supervisor's) adaptation instead **trains the heads** to minimize TCI and keeps
  `v ∈ [0,1]`. For stage-2 we therefore either (a) binarize `v → ±1` before MoM, or
  (b) use the soft `2v−1 ∈ [−1,1]` moments (ψ/η become generalized conditional means).
  Confirm with supervisor. Also: `b` and the Assumption-2.1 sign can be **anchored by
  S_L labels** (we have them) for robustness, rather than recovered purely unsupervised.
