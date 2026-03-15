# V1 Research Protocol

## Frozen Scope

The first paper version is intentionally frozen to:

- 1D scalar reaction-diffusion systems
- periodic boundary conditions
- Case A and Case B closures only
- MLP closure surrogate as the main model
- weak-form training plus forward rollout consistency
- restricted numerical closure recovery as the primary target
- symbolic recovery only as a restricted post-processing stage

The following are explicitly out of scope for `v1`:

- 2D pattern cases
- Case C or unrestricted rational discovery as a main claim
- KAN as a required component
- unconstrained global symbolic regression

Exploratory stress cases that sit outside the main evidence package are allowed if they are clearly labeled as such. Their role is to test baseline mismatch, not to replace the core Case A and Case B protocol.

## Core Research Questions

`v1` is organized around three questions.

1. Under a known PDE structure, when is closure recovery identifiable from spatiotemporal data?
2. Does a low weak-form residual imply correct constitutive recovery?
3. What validation protocol is needed to distinguish true closure recovery from discretization-specific fitting?

## Main Hypotheses

- Diverse excitation coverage improves closure identifiability.
- Weak-form fitting alone is not sufficient evidence of correct closure recovery.
- Forward re-simulation and cross-discretization checks reduce false confidence caused by inverse crime.

## Hard Acceptance Criteria

An experiment does not count as a successful recovery unless all of the following hold.

1. The learned numerical closures achieve low relative error on a held-out support grid.
2. Forward simulation with the learned numerical closures reproduces unseen trajectories.
3. The recovered closures are stable across random seeds and training subsets.
4. Cross-resolution or cross-time-step validation does not collapse performance.

## Evaluation Levels

Every main experiment should report all three levels.

### Level 1: Trajectory

- weak residual
- one-step rollout MSE
- multi-step rollout MSE on unseen initial conditions

### Level 2: Closure

- relative `L2` error of `D(u)`
- relative `L2` error of `R(u)`
- error on a common evaluation support independent of training batches

### Level 3: Identifiability

- variance of recovered closures across seeds
- variance across training subsets
- recovery stability under changes in excitation family

## Excitation Coverage Protocol

The training set should not be described as "random initial conditions" only. It should be characterized quantitatively.

Required coverage summaries:

- state coverage over the target `u` interval
- gradient magnitude coverage
- curvature magnitude coverage
- weak diffusion energy proxy

At minimum, compare two regimes:

- low-excitation regime: smooth, small-amplitude initial conditions near equilibrium
- high-excitation regime: multi-frequency, higher-amplitude initial conditions with sharper transitions

The goal is to show whether identifiability depends on data diversity rather than only on model capacity.

## Anti-Inverse-Crime Protocol

At least one anti-inverse-crime setting is required in `v1`. Preferred order:

1. cross-resolution identification
2. cross-time-step validation
3. cross-solver validation

The minimum practical setting is:

- generate trajectories on a fine grid
- identify closures on a coarsened observation grid
- validate rollout with a third discretization setting

## Model Policy

The main model is a simple positive-diffusion closure surrogate:

- one scalar network for `D(u)`
- one scalar network for `R(u)`
- positivity constraint on `D(u)`
- smoothness regularization on both closures

KAN is allowed only as:

- an ablation
- an interpretable replacement backbone after the baseline protocol is stable

## Symbolic Policy

Symbolic extraction is not a primary success metric in `v1`.

Allowed symbolic families:

- low-order polynomials
- low-order rational functions

The symbolic stage is accepted only if:

- the symbolic closure remains close to the learned numerical closure
- forward simulation with the symbolic closure remains valid

## Immediate Experimental Order

1. Case A, noise-free, high-excitation and low-excitation comparison.
2. Case A, seed stability and training-subset stability.
3. Case A, cross-resolution or cross-time-step anti-inverse-crime setting.
4. Case B under the same protocol.
5. Restricted symbolic fitting only after the numerical closure protocol is stable.

## Paper Narrative

The recommended `v1` paper narrative is:

> We study closure discovery under known PDE structure as an identifiable inverse problem, show that low residual alone is insufficient, and introduce a weak-form neural closure framework with explicit excitation and anti-inverse-crime validation.
