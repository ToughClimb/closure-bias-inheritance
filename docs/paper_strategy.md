# Paper Strategy

## Current Benchmark Snapshot

The current neural closure learner is viable as a numerical recovery method, but the baseline picture is much stronger than the original paper framing assumed.

Representative single-seed comparisons with the default restricted polynomial library (`diffusion_degree=2`, `reaction_degree=3`):

| Case | Observation | Neural `(ErrD, ErrR, unseen)` | Strong poly `(ErrD, ErrR, unseen)` | Weak poly `(ErrD, ErrR, unseen)` |
| --- | --- | --- | --- | --- |
| `case_a` | clean | `6.70e-02, 3.07e-01, 4.50e-03` | `4.29e-04, 2.20e-03, 1.83e-05` | `8.45e-03, 3.66e-02, 7.84e-04` |
| `case_a` | `5%` noise | `8.63e-01, 2.66e+00, 1.02e-01` | `9.65e-01, 7.62e+00, 9.91e-02` | `2.12e-02, 2.16e-01, 2.25e-03` |
| `case_b` | clean | `1.01e-01, 1.82e-01, 3.63e-03` | `5.02e-04, 3.99e-03, 1.99e-05` | `8.71e-03, 4.72e-02, 9.06e-04` |
| `case_exp` | clean | `4.78e-02, 4.81e-01, 3.05e-03` | `7.79e-02, 5.54e-01, 2.70e-03` | `6.06e-02, 1.70e-01, 2.33e-03` |
| `case_exp` | `5%` noise | `8.33e-01, 4.82e+00, 9.52e-02` | `9.68e-01, 1.40e+01, 9.50e-02` | `8.10e-02, 1.23e+00, 4.41e-03` |

Small multi-seed benchmark on `case_exp` confirms the same pattern:

- clean: neural beats `strong_poly`, but `weak_poly` remains clearly better on `ErrR` and rollout.
- noisy and sparse settings: `weak_poly` remains the most robust method in the current codebase.

## What These Results Mean

1. A paper framed as "the weak-form neural method outperforms sparse or polynomial baselines everywhere" is not supported.
2. On in-library polynomial truth, `weak_poly` behaves like an oracle baseline. The neural model should not be expected to beat it unconditionally.
3. The exploratory `case_exp` regime is the right place to justify a neural surrogate, because the true closure is outside the default low-order polynomial library.
4. The value of the neural stage is therefore not architecture superiority by itself. Its value is to provide a flexible numerical constitutive surrogate that can later be compressed into a restricted symbolic law.

## Recommended Paper Positioning

The strongest near-term neural-symbolic narrative is:

> Neural-symbolic closure discovery is most useful under function-class mismatch or limited prior knowledge: first recover a numerical constitutive surrogate, then compress it into a restricted symbolic law, and finally validate that symbolic law by forward simulation.

Under this framing:

- `weak_poly` remains a serious oracle comparator in matched-library regimes
- the neural surrogate is a middle layer, not the final scientific claim
- the paper's central method is the full chain from numerical recovery to symbolic compression to forward validation

## Neural-Symbolic Pipeline

The intended three-stage pipeline is:

1. Learn numerical closures `\hat D(u)` and `\hat R(u)` with the weak-form neural surrogate.
2. Fit restricted symbolic families to those learned closures rather than performing unrestricted symbolic search directly on raw trajectory data.
3. Reinsert the compressed symbolic laws into the PDE and validate them through unseen rollout and cross-discretization checks.

The key methodological point is:

> low residual or low rollout error is not enough; a symbolic law only counts if it survives the compress-and-reinsert loop.

## Theory Spine

The theory section should not be sold as a full uniqueness-and-convergence result. The safer and more accurate framing is:

- weak-form consistency and mass-balance consistency justify the loss construction
- matched-library identification explains why `weak_poly` is close to an oracle in `case_a` and `case_b`
- limited excitation explains why low residual does not imply true closure recovery
- function-class mismatch explains why a neural surrogate can matter in `case_exp`
- symbolic compression explains why `BIR_D` and `BIR_R` stay near `1`
- finite-time PDE stability explains why symbolic rollout remains close to neural rollout

These points are written out in [theory_outline.md](theory_outline.md). The current strongest theory-backed claim is not "the neural model uniquely discovers the true law", but rather:

> under limited excitation and function-class mismatch, the main bottleneck lies in numerical constitutive recovery, while the symbolic stage mostly preserves the bias already present in the learned surrogate

## Plausible Next Moves

### Option A: Protocol-first paper

Keep `case_a` and `case_b` as the main evidence package. Emphasize:

- low residual versus true closure recovery
- excitation coverage as an identifiability variable
- seed/subset stability
- anti-inverse-crime validation

This is the safest publishable route with the current results.

### Option B: Hybrid neural-symbolic paper

Only pursue this after adding restricted symbolic fitting and showing:

- the neural numerical closure can be compressed into a compact family
- the symbolic closure preserves forward rollout quality
- the symbolic stage remains meaningful in function-class-mismatch settings such as `case_exp`

Without that last point, the symbolic story will still look weaker than the baseline.

### Option C: Architecture-centric method paper

Not recommended yet. The present benchmarks do not justify a "neural method beats the alternatives" claim.

## Immediate Execution Priority

1. Finish the identifiability protocol figures and tables for `case_a` and `case_b`.
2. Use `case_exp` as the main stress test for function-class mismatch.
3. Evaluate symbolic compression by comparing neural surrogate errors, symbolic compression errors, and post-compression rollout side by side.
4. Draft the theory section around non-identifiability, approximation bias, and bias-preserving symbolic compression instead of around architecture novelty.
5. Use [abstract_drafts.md](abstract_drafts.md) and [manuscript_outline.md](manuscript_outline.md) as the writing baseline, so the introduction, theory, and experiments all carry the same claim.
