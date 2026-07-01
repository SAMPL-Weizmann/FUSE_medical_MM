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
For each `c`, over **all pairs `(a,b)` with `a<b`, `a≠c`, `b≠c`** (decision 1):
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
2. **Estimate** (`S_U`): freeze; estimate `Σ, T` → per-verifier sensitivity/specificity
   (from the rank-1 `q`, sign/scale anchored by `S_L`/Test), class prior, posterior
   `P(Y|v)`; fit the combined predictor (CI likelihood / weighted ensemble).
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

## Open items to confirm with supervisor
- Exact `1≤a<b<c` vs "all pairs ≠ c" (we use the latter).
- Estimator specifics in stage 2 (how `q` → sens/spec and the predictor form).
