# Abstract Drafts

## Draft A: Balanced Main Abstract

We study closure discovery in nonlinear reaction-diffusion systems when the governing PDE structure is known but the constitutive laws remain unknown. Our goal is to recover the diffusion and reaction closures from spatiotemporal observations, while avoiding the usual failure mode in which low residuals or short-horizon predictions are mistaken for correct physical recovery. We propose a weak-form neural-symbolic pipeline with three stages: numerical constitutive recovery, restricted symbolic compression, and forward re-simulation validation. The weak-form stage learns numerical surrogates for the unknown closures under positivity and smoothness constraints; the symbolic stage compresses the learned surrogates into restricted polynomial, rational, and saturation-style families; the final stage reinserts the symbolic closures into the PDE and evaluates unseen rollout behavior. Experiments on 1D reaction-diffusion systems show two distinct regimes. In matched-library settings, weak polynomial baselines behave like near-oracle estimators and should not be expected to be uniformly outperformed by neural surrogates. In function-class-mismatch settings, neural surrogates provide flexible numerical closures that can be compressed into compact symbolic laws with negligible additional surrogate error and little rollout degradation. However, symbolic compression does not automatically repair constitutive bias: across clean, noisy, and sparse observation settings, the symbolic true closure error closely tracks the neural true closure error, and the bias inheritance ratio remains near one. These results show that the main bottleneck in neural-symbolic closure discovery lies in Stage 1 numerical constitutive recovery rather than in symbolic compression, and that closure claims should be supported by explicit forward validation rather than residual minimization alone.

## Draft B: More Theory-Led Abstract

Closure discovery from spatiotemporal data is an inverse problem: low residual or low rollout error does not by itself imply recovery of the true constitutive law. We study this issue for 1D nonlinear reaction-diffusion systems of the form `u_t = \partial_x(D(u)u_x) + R(u)`, where the PDE structure is known but the diffusion and reaction closures are unknown. We develop a weak-form neural-symbolic framework that separates three stages of the problem: numerical closure identification, symbolic compression, and forward PDE validation. The theory spine of the paper links weak-form consistency, excitation-dependent identifiability, approximation bias under function-class mismatch, and bias-preserving symbolic compression. Empirically, matched-library weak polynomial baselines are nearly oracle on in-library cases, while neural surrogates become useful when the true closures fall outside the restricted library. In that mismatch regime, restricted symbolic families compress the learned neural closures with very small additional approximation error and nearly unchanged rollout behavior, but do not remove constitutive bias inherited from the neural stage. The results support a practical conclusion for scientific machine learning: the main challenge is not symbolic distillation itself, but reliable numerical constitutive recovery under limited excitation and misspecified prior libraries.

## Draft C: Shorter Venue-Agnostic Abstract

We investigate neural-symbolic closure discovery for reaction-diffusion systems with known PDE structure and unknown constitutive laws. Our pipeline first learns numerical diffusion and reaction closures from spatiotemporal data in weak form, then compresses the learned surrogates into restricted symbolic families, and finally validates the symbolic laws by forward simulation. The experiments reveal two regimes. In matched-library cases, weak polynomial baselines are near-oracle and should not be treated as weak comparators. In function-class-mismatch cases, neural surrogates provide flexible numerical closures that can be compressed into symbolic laws with minimal extra surrogate error and little additional rollout degradation. However, symbolic compression does not correct constitutive bias inherited from the neural stage: symbolic true closure errors closely track neural true closure errors, with bias inheritance ratios near one across clean, noisy, and sparse settings. The main bottleneck in neural-symbolic closure discovery is therefore numerical constitutive recovery, not symbolic compression.

## Title Candidates

1. Neural-Symbolic Closure Discovery Under Function-Class Mismatch
2. From Neural Constitutive Surrogates to Symbolic Closure Laws
3. Weak-Form Neural-Symbolic Closure Discovery Beyond Restricted Polynomial Libraries
4. Symbolic Compression Preserves but Does Not Repair Constitutive Bias in Closure Discovery
5. Closure Discovery Under Limited Excitation: A Weak-Form Neural-Symbolic Framework

## Current Recommendation

The safest title and abstract pairing at the moment is:

- title: `Neural-Symbolic Closure Discovery Under Function-Class Mismatch`
- abstract: Draft A

This pairing matches the current evidence package without overclaiming architecture superiority or exact symbolic recovery.
