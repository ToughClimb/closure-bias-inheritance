# Feasibility Assessment

## Bottom Line

This paper direction is feasible and publishable as a synthetic-methods paper if it is executed as a closure-discovery problem rather than a full PDE-discovery problem.

The reason is structural: `D(u)` and `R(u)` are one-dimensional constitutive functions, so the hypothesis space is much smaller than discovering an unrestricted PDE library. That makes the problem scientifically meaningful and technically tractable.

## What Is Strong About The Proposal

- The governing structure is known, which improves identifiability.
- The output is scientific insight, not only predictive accuracy.
- Weak-form training is appropriate for noisy spatiotemporal data.
- Synthetic experiments can be made complete enough for a solid first paper.

## Main Technical Risks

### 1. Identifiability Can Still Fail

If the dataset contains too few initial conditions or too narrow a state range, multiple closure pairs can explain the same trajectories locally. The fix is not a more complex network. The fix is better excitation:

- multiple trajectories
- broad amplitude range
- diverse spatial frequencies
- evaluation on unseen initial conditions

For the first paper, this should be elevated from an implementation detail to a research variable. The training data must report excitation coverage rather than only the number of trajectories.

### 2. KAN Is Not The Core Scientific Contribution

KAN may help interpretability, but weak-form closure discovery is the central contribution. If the project depends on KAN from day one, the engineering burden increases and the paper story becomes less clean. The safer path is:

- build the pipeline with a scalar MLP closure first
- add KAN later as an interpretable surrogate variant
- present KAN vs MLP as an ablation, not as the only workable model

### 3. Symbolic Regression Is The Most Brittle Stage

Neural closure learning and symbolic recovery are different problems. Even when the learned function is accurate, exact symbolic structure recovery can fail because:

- sampling range is too small
- closures are nearly collinear in the observed region
- rational fits are numerically unstable

For the first milestone, symbolic extraction should be treated as a post-processing module with a restricted candidate family rather than unconstrained global search.

### 4. Case C And 2D Are Not First-Milestone Tasks

Both add real value, but both also raise failure risk:

- rational closures are harder to recover stably
- 2D makes data generation, training cost, and visualization pipelines heavier

The paper is still viable without them in the first implementation cycle.

### 5. Low Residual Does Not Guarantee Correct Closure Recovery

Weak-form loss or short-horizon rollout loss can decrease even when the recovered constitutive laws remain inaccurate. This is an inverse-problem failure mode rather than a training bug.

For `v1`, every experiment should separate:

- trajectory consistency
- closure accuracy
- identifiability stability across seeds and subsets

### 6. Inverse Crime Must Be Addressed Explicitly

If data generation and identification use nearly identical discretizations, recovery can look stronger than it really is. A convincing first paper should include at least one anti-inverse-crime setting such as cross-resolution or cross-time-step validation.

## Recommended Minimum Publishable Version

The minimum credible version is:

- 1D only
- periodic boundary condition
- Cases A and B
- weak-form residual
- MLP closure surrogate with positive diffusion constraint
- baselines: strong-form sparse regression, weak-form sparse regression, MLP without symbolic stage
- experiments: excitation coverage, base recovery, seed stability, anti-inverse-crime validation, noise, sparse sampling, OOD initial conditions

This is enough to establish the paper's main claim:

> When the PDE structure is known but constitutive closures are unknown, robust recovery depends on excitation coverage and validation protocol, and a weak-form physics-constrained learner can recover the underlying nonlinear laws more reliably than sparse-regression or black-box alternatives.

## What Would Make It Stronger Later

- symbolic formula extraction with restricted candidate families
- KAN closure surrogate
- 2D pattern-forming examples
- rational and saturation closures
- a real-data constitutive discovery case

## Recommendation

Proceed. The direction is strong enough for a first paper, but only if the implementation is staged and the first round avoids over-ambitious scope.

The practical `v1` protocol is documented in [v1_protocol.md](v1_protocol.md).
