# Manuscript Outline

## Working Title

Neural-Symbolic Closure Discovery Under Function-Class Mismatch

## One-Sentence Paper Claim

When PDE structure is known but constitutive closures are unknown, the main bottleneck is numerical closure recovery under limited excitation and mismatched prior libraries; restricted symbolic compression can preserve learned dynamics, but it does not automatically repair constitutive bias inherited from the neural stage.

## Paper Type

- methods paper with theory-supported mechanism analysis
- not an architecture paper
- not a full uniqueness-and-convergence theory paper

## Core Message Hierarchy

### Message 1

Low weak residual and good rollout are not sufficient evidence of true closure recovery.

### Message 2

In matched-library settings, weak polynomial baselines are near-oracle and should be treated as strong comparators.

### Message 3

In mismatch settings, neural surrogates are useful as flexible numerical constitutive approximators.

### Message 4

Restricted symbolic compression is a bias-preserving distillation stage: it preserves learned constitutive behavior with little extra rollout degradation, but it does not correct Stage 1 bias.

## Evidence Anchors From Current Results

### Case A

- clean matched-library regime:
  - `strong_poly` is best: `ErrD = 4.10e-04 +/- 1.14e-04`, `ErrR = 2.11e-03 +/- 8.86e-04`, `unseen = 1.80e-05 +/- 5.13e-06`
  - `weak_poly` is also strong: `ErrD = 8.15e-03 +/- 1.05e-03`, `ErrR = 3.09e-02 +/- 9.03e-03`, `unseen = 8.31e-04 +/- 1.35e-04`
  - neural surrogate is worse: `ErrD = 6.93e-02 +/- 4.01e-03`, `ErrR = 3.13e-01 +/- 1.89e-02`, `unseen = 4.83e-03 +/- 3.14e-04`
- symbolic stage preserves the neural closure:
  - clean `BIR_D = 9.893e-01 +/- 4.688e-03`
  - clean `BIR_R = 9.999e-01 +/- 7.003e-05`
- under sparse space or sparse space-time settings, symbolic compression mildly smooths diffusion error but still does not fundamentally repair reaction bias:
  - `sparse x` `BIR_D = 9.231e-01 +/- 3.457e-02`
  - `sparse x+t` `BIR_D = 9.239e-01 +/- 3.650e-02`

### Case Exp

- clean mismatch regime:
  - neural surrogate improves diffusion error relative to `strong_poly` but still trails `weak_poly` on reaction and rollout:
  - neural: `ErrD = 4.726e-02 +/- 3.732e-04`, `ErrR = 4.698e-01 +/- 8.209e-03`, `unseen = 3.162e-03 +/- 9.275e-05`
  - `strong_poly`: `ErrD = 7.664e-02 +/- 2.370e-03`, `ErrR = 3.953e-01 +/- 1.225e-01`, `unseen = 2.383e-03 +/- 2.411e-04`
  - `weak_poly`: `ErrD = 6.395e-02 +/- 2.375e-03`, `ErrR = 1.036e-01 +/- 4.733e-02`, `unseen = 2.137e-03 +/- 1.435e-04`
- symbolic compression is extremely faithful:
  - clean symbolic surrogate errors: `ErrD = 1.872e-03 +/- 5.583e-05`, `ErrR = 1.664e-04 +/- 8.905e-06`
  - clean `BIR_D = 9.965e-01 +/- 4.780e-04`
  - clean `BIR_R = 1.000e+00 +/- 7.288e-07`
- the same pattern persists under noise and sparse observations:
  - noise 5%: `BIR_D = 1.000e+00 +/- 4.183e-06`, `BIR_R = 9.995e-01 +/- 5.436e-04`
  - sparse x+t: `BIR_D = 9.990e-01 +/- 1.465e-03`, `BIR_R = 1.000e+00 +/- 3.719e-07`

## Section Outline

### 1. Introduction

Goal:

- motivate closure discovery under known PDE structure
- explain why full PDE discovery is harder and less identifiable
- position the paper around validation, not architecture novelty

Must contain:

- known structure, unknown closures
- low residual does not imply true law recovery
- matched-library and mismatch regimes are different scientific settings
- symbolic formulas only matter if they survive reinsertion into the PDE

Last paragraph contribution bullets should be:

1. formulate closure discovery as a weak-form inverse problem with explicit identifiability and mismatch considerations
2. propose a neural-symbolic pipeline that separates numerical recovery, symbolic compression, and forward validation
3. show empirically and theoretically that symbolic compression preserves, rather than repairs, constitutive bias
4. benchmark the method against strong and weak polynomial baselines across clean, noisy, and sparse settings

### 2. Problem Formulation

Need:

- target PDE
- closure pair notation `f = (D, R)`
- observation setting
- training, symbolic, and rollout evaluation stages

Keep it narrow:

- 1D only in the main paper
- periodic boundary conditions
- case families labeled clearly

### 3. Method

Subsections:

1. weak-form residual and mass-balance consistency
2. numerical closure surrogate
3. restricted symbolic compression
4. forward re-simulation validation

Important wording:

- call the symbolic stage `restricted symbolic compression of learned constitutive surrogates`
- do not say `directly discover the exact symbolic law from raw data`

### 4. Theory

Use the exact ordering from [theory_outline.md](theory_outline.md):

1. weak-form consistency
2. matched-library identification
3. non-identifiability under limited excitation
4. approximation bias under function-class mismatch
5. symbolic compression as bias-preserving distillation
6. rollout preservation under small closure perturbations

This section should end with the unified error decomposition:

`true symbolic error <= model-class bias + identification error + symbolic compression error`

### 5. Experiments

Recommended order:

1. baseline map: matched-library vs mismatch
2. symbolic compression benchmark
3. rollout preservation and bias inheritance
4. optional appendix: stability and anti-inverse-crime protocols

This order keeps the paper centered on the actual claim rather than on the full backlog of protocol experiments.

### 6. Discussion

Must explicitly answer three questions:

1. why neural does not beat `weak_poly` in matched-library regimes
2. why symbolic compression should not be expected to repair a biased surrogate
3. where the real bottleneck lies for future work

The key sentence of the discussion should be:

> the symbolic stage is not the main bottleneck; once compression error is small, constitutive bias is dominated by model-class mismatch and numerical identification error in Stage 1

### 7. Conclusion

Keep it short:

- summarize the three-stage view
- restate the bottleneck
- point to future work as improved Stage 1 recovery, broader PDE families, and more realistic observation models

## Figure Plan

### Figure 1

Pipeline overview:

- spatiotemporal observations
- weak-form neural recovery
- symbolic compression
- PDE reinsertion and rollout validation

### Figure 2

Baseline map from [baseline_summary.md](../results/paper_artifacts/tables/baseline_summary.md):

- `Case A` and `Case Exp`
- settings: `clean`, `noise 5%`, `sparse x+t`
- methods: `strong_poly`, `weak_poly`, `neural`, `neural+symbolic`

Primary purpose:

- show matched-library oracle behavior
- show mismatch regime and where neural surrogate sits

### Figure 3

Neural-to-symbolic error propagation from [symbolic_error_propagation.png](../results/paper_artifacts/figures/symbolic_error_propagation.png)

Primary purpose:

- Stage 2 error remains small
- Stage 3 error tracks Stage 1

### Figure 4

Rollout preservation and bias inheritance from [symbolic_rollout_and_bir.png](../results/paper_artifacts/figures/symbolic_rollout_and_bir.png)

Primary purpose:

- symbolic unseen rollout nearly equals neural unseen rollout
- `BIR_D` and `BIR_R` stay near `1`

## Table Plan

### Table 1

Primary symbolic benchmark table from [symbolic_summary.md](../results/paper_artifacts/tables/symbolic_summary.md)

Columns:

- case
- setting
- neural true errors
- symbolic surrogate errors
- symbolic true errors
- neural unseen
- symbolic unseen
- `BIR_D`, `BIR_R`

### Table 2

Baseline comparison table from [baseline_summary.md](../results/paper_artifacts/tables/baseline_summary.md)

Columns:

- case
- setting
- method
- `ErrD`
- `ErrR`
- `unseen`

## What Not To Claim

Do not claim:

- exact symbolic recovery from raw trajectory data
- unconditional superiority of neural methods over weak polynomial baselines
- full identifiability from finite noisy data
- KAN as a central contribution

## Minimal Remaining Work Before Draft Submission

1. Write the introduction and discussion around the current message, not around architecture novelty.
2. Decide whether to keep one defensive appendix experiment, preferably capacity ablation on `case_exp`.
3. Package the figures with caption text that explicitly states `preserves but does not repair bias`.

## Immediate Writing Order

1. Abstract
2. Introduction
3. Results section with Figure 2 to Figure 4
4. Theory section from the existing outline
5. Method details
6. Discussion and conclusion
