# Theory Outline

## Positioning

This note defines the theory spine for the current paper direction.

It is not a complete identifiability theory for nonlinear PDE inverse problems. It is a disciplined analysis framework that does three things:

- justifies the weak-form loss and mass-balance constraints
- explains the observed gap between low residual and true closure recovery
- explains why symbolic compression preserves learned dynamics but does not automatically repair constitutive bias

The intended paper claim is therefore:

> the theory is strong enough to explain the observed mechanisms and to organize the experiments, but not yet strong enough to be sold as a full uniqueness-and-convergence paper

## Setting

We study the 1D reaction-diffusion system

```math
u_t = \partial_x(D(u) u_x) + R(u), \qquad x \in \Omega, \ t \in [0, T],
```

with periodic boundary conditions in the frozen `v1` protocol.

Write the closure pair as

```math
f = (D, R).
```

The data consist of a finite set of trajectories generated from initial conditions in a bounded state interval `U`.

For the theory below, the working assumptions are:

- `D(u) >= d_min > 0` on `U`
- `D` and `R` are bounded and Lipschitz on `U`
- the true and learned trajectories remain inside `U` on `[0, T]`
- the weak-form test functions are fixed in advance
- the reported weak residual is the discrete projection of the continuous weak formulation

These assumptions are enough for a method-paper theory section. They are not intended as the final sharpest assumptions.

## Proposition 1: Weak-Form Consistency

Assume

```math
u \in L^2(0, T; H^1(\Omega)), \qquad u_t \in L^2(0, T; H^{-1}(\Omega)).
```

Then for any admissible test function `phi`,

```math
\int_0^T \langle u_t, \phi \rangle \, dt
+ \int_0^T \int_\Omega D(u) u_x \phi_x \, dx dt
- \int_0^T \int_\Omega R(u) \phi \, dx dt = 0.
```

Under periodic or no-flux boundaries, spatial integration also gives

```math
\frac{d}{dt} \int_\Omega u \, dx = \int_\Omega R(u) \, dx.
```

Role in the paper:

- legitimizes the weak residual as a physically consistent training objective
- legitimizes the mass-balance penalty

Remark:

This proposition should be described as weak-form consistency, not as a blanket equivalence claim, unless the final paper explicitly introduces the needed function-space assumptions.

## Proposition 2: Matched-Library Identification Under Sufficient Excitation

Suppose the true closures lie in known finite dictionaries:

```math
D(u) = \sum_{i=1}^{p_D} a_i \psi_i(u), \qquad
R(u) = \sum_{j=1}^{p_R} b_j \chi_j(u).
```

After weak-form projection and discrete quadrature, the identification problem becomes

```math
\Phi \theta = Y,
```

where `theta` stacks the coefficients `(a, b)`, `Phi` depends on the observed trajectories and test functions, and `Y` is induced by the projected time derivative.

If the Gram matrix

```math
G = \Phi^T \Phi
```

is positive definite, equivalently `lambda_min(G) > 0`, then the least-squares fit is unique.

Role in the paper:

- explains why `weak_poly` behaves like an oracle in `case_a` and `case_b`
- converts "excitation coverage" into a rank condition on the induced design matrix

Interpretation:

- state coverage determines which parts of the closure domain are sampled
- gradient and curvature coverage determine whether the diffusion columns of `Phi` are informative
- if these ingredients are weak, `G` becomes ill-conditioned or rank-deficient

## Proposition 3: Low Residual Does Not Imply True Closure Recovery

Define the weak-form observation operator on the finite training trajectories `T_data` by

```math
A_T(D, R).
```

If `A_T` is not injective on the admissible closure class, then there exist nonzero perturbations `(delta D, delta R)` such that

```math
A_T(D + \delta D, R + \delta R) \approx A_T(D, R).
```

Therefore, low weak residual on the observed trajectories does not imply unique or correct recovery of the true closure pair.

Role in the paper:

- formalizes the central inverse-problem warning behind the whole project
- explains why low weak loss and good short-horizon rollout can coexist with large closure error

Interpretation:

This non-identifiability is expected when the data occupy only a narrow state interval, when the gradients are too small, when the trajectories are too close to equilibrium, or when diffusion and reaction contributions can compensate each other on the observed support.

This proposition is the theory counterpart of the excitation-coverage experiments.

## Proposition 4: Approximation Bias Under Function-Class Mismatch

Let `F` be the hypothesis class used by the identifier. If the true closure `f^*` does not belong to `F`, define the best-in-class target

```math
f^\dagger = \arg\min_{f \in F} L(f),
```

for the chosen population loss `L`.

Then the recovery error can be decomposed as

```math
\|f^* - \hat f\|
\le
\|f^* - f^\dagger\|
+ \|f^\dagger - \hat f\|.
```

The first term is model-class bias. The second term is identification, optimization, and finite-sample error.

Role in the paper:

- explains why polynomial baselines are nearly ideal in matched-library settings
- explains why `case_exp` is the right stress test for a neural surrogate
- explains the value of the neural stage without claiming unconditional superiority

Interpretation:

The neural surrogate matters when the restricted symbolic or polynomial library is misspecified. Its advantage is reduced approximation bias, not a guarantee of unbiased constitutive recovery.

## Proposition 5: Symbolic Compression Preserves Bias Up To Compression Error

Let `f^*` be the true closure, `hat f` the learned numerical surrogate, and `f_sym` the compressed symbolic closure.

Then

```math
\big| \|f^* - f_{sym}\| - \|f^* - \hat f\| \big|
\le
\|\hat f - f_{sym}\|.
```

Equivalently,

```math
\|f^* - f_{sym}\|
\le
\|f^* - \hat f\| + \|\hat f - f_{sym}\|,
```

and

```math
\|f^* - f_{sym}\|
\ge
\|f^* - \hat f\| - \|\hat f - f_{sym}\|.
```

Role in the paper:

- explains why `BIR_D` and `BIR_R` stay near `1`
- formalizes the statement that symbolic compression preserves, rather than repairs, constitutive bias

Interpretation:

If the compression error `\|\hat f - f_sym\|` is small, then the symbolic true error must remain close to the neural true error. The symbolic stage cannot create a large correction unless it first departs substantially from the neural surrogate.

This is the cleanest theoretical explanation for the current benchmark results.

## Proposition 6: Rollout Preservation Under Small Closure Perturbations

Let `u_1` and `u_2` solve the same initial-value problem on `[0, T]` with closure pairs `f_1 = (D_1, R_1)` and `f_2 = (D_2, R_2)`.

Under uniform parabolicity and bounded-Lipschitz assumptions on both closures, a finite-time stability estimate of Gronwall type gives

```math
\sup_{t \in [0, T]} \|u_1(t) - u_2(t)\|_{L^2}
\le
C_T \left(
\|D_1 - D_2\|_U + \|R_1 - R_2\|_U
\right),
```

where `C_T` depends on the time horizon and the bounded solution set.

Role in the paper:

- explains why symbolic rollout remains close to neural rollout when symbolic compression error is small
- provides the PDE-stability bridge from closure-space error to trajectory-space error

Interpretation:

This proposition is finite-time and local to the bounded state range actually visited by the trajectories. It should not be overstated as a global uniform result.

## Unified Error Decomposition

The final symbolic recovery error can be organized as

```math
\|f^* - f_{sym}\|
\le
\underbrace{\|f^* - f^\dagger\|}_{model\text{-}class\ bias}
+ \underbrace{\|f^\dagger - \hat f\|}_{identification\ and\ optimization\ error}
+ \underbrace{\|\hat f - f_{sym}\|}_{symbolic\ compression\ error}.
```

This decomposition is the cleanest way to connect the whole paper:

- matched-library experiments mainly probe the second term
- mismatch experiments such as `case_exp` expose the first term
- symbolic-compression benchmarks isolate the third term

The current evidence says the third term is small, while the first two dominate.

## Recommended Theory Chapter Structure

The theory section should be organized in the following order.

1. Weak-form closure identification and mass balance.
2. Matched-library identification under sufficient excitation.
3. Non-identifiability under limited excitation.
4. Approximation bias under function-class mismatch.
5. Symbolic compression as bias-preserving distillation.
6. Rollout preservation under small closure perturbations.

This order mirrors the scientific story of the experiments.

## What Is Safe To Claim

Safe claims:

- the weak-form objective is physically consistent
- excitation coverage controls identifiability through the conditioning of the induced observation operator
- matched-library baselines are expected to be very strong
- function-class mismatch creates approximation bias
- symbolic compression preserves learned constitutive behavior up to compression error
- small compression error implies small additional rollout degradation over finite horizons

Claims that should not be made yet:

- global uniqueness of closure recovery from finite noisy trajectories
- convergence of the neural optimizer to the true closure
- full noise-to-error bounds for the implemented estimator
- automatic discovery of the true symbolic law from raw data without a surrogate stage

## Practical Writing Guidance

For the current paper, the theory should support the method and explain the benchmarks. It should not be marketed as the main novelty unless Proposition 3 and Proposition 4 are developed carefully enough to become central methodological contributions.

The strongest present theory-led sentence is:

> low residual and good rollout are not sufficient evidence of true constitutive recovery; under limited excitation and function-class mismatch, the main bottleneck lies in numerical closure identification, while symbolic compression mostly preserves the bias already present in the learned surrogate
