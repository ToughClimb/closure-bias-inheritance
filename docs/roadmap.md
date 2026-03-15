# Implementation Roadmap

## Design Principles

- Keep the PDE structure explicit.
- Separate data generation, closure learning, weak-form residuals, and symbolic fitting.
- Make every stage independently testable.
- Delay 2D and KAN until the 1D weak-form loop is reliable.
- Treat identifiability and anti-inverse-crime checks as first-class deliverables, not later polish.

## Phase 1: 1D Protocol Stabilization

Goal: establish an end-to-end closure discovery loop on noise-free synthetic data and verify that closure recovery, not only residual minimization, is stable.

Deliverables:

- 1D reaction-diffusion solver with periodic boundary conditions
- Case A and Case B closures
- random smooth initial condition generator
- observation downsampling utilities for coarse identification
- weak-form residual implementation
- MLP-based closure model with positive diffusion
- training script for Case A
- closure-level, rollout-level, and excitation-coverage evaluation
- at least one cross-resolution or cross-time-step validation setting

Success criteria:

- low relative error for both `D(u)` and `R(u)` on Case A under high-excitation training
- stable recovery across multiple random seeds
- clear performance gap between low-excitation and high-excitation regimes
- correct OOD trend on unseen initial conditions

## Phase 2: Robustness Experiments

Goal: support the main experimental claims in the paper.

Deliverables:

- Gaussian noise injection
- temporal and spatial downsampling
- weak-form sparse baseline
- strong-form sparse baseline
- evaluation table for noise and sparsity

Success criteria:

- weak-form learner remains stable under moderate noise
- baselines degrade earlier than the proposed method

## Phase 3: Symbolic Refinement

Goal: convert learned closures into compact analytic expressions.

Deliverables:

- candidate-family fitting for polynomials and rational forms
- complexity-aware model selection
- forward simulation with recovered symbolic closures

Success criteria:

- Case A exact or near-exact symbolic recovery
- Case B compact approximate recovery with valid rollout

## Phase 4: Model Variants And Ablations

Goal: demonstrate what parts of the method actually matter.

Deliverables:

- KAN closure module
- no-weak-form ablation
- no-constraint ablation
- no-symbolic-refinement ablation

Success criteria:

- each major module has a measurable contribution

## Phase 5: 2D Extension

Goal: improve visual quality and strengthen the paper's scope.

Deliverables:

- 2D solver
- 2D weak-form residual
- one visually strong pattern-forming case
- rollout comparison figures

Success criteria:

- recovered closures remain valid when reinserted into 2D simulation

## Immediate Work Queue

1. Add dataset serialization and reproducible configs.
2. Add restricted symbolic fitting on the learned numerical closures.
3. Turn the current cross-resolution benchmark into a stronger anti-inverse-crime table with more settings.
4. Add OOD unseen-rollout settings beyond the training amplitude range.
5. Benchmark a non-polynomial stress case against the restricted polynomial baselines and decide how it should affect the paper narrative.

The frozen research scope is defined in [v1_protocol.md](v1_protocol.md).
