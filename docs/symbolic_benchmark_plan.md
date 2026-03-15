# Symbolic Compression Benchmark Plan

## Goal

Turn the neural-to-symbolic pipeline into a benchmark that answers one concrete question:

> when does symbolic compression faithfully preserve a learned neural surrogate, and how much true constitutive bias is inherited from the neural stage?

The main benchmark target is `case_exp`, because it stresses function-class mismatch. `case_a` is used only as an oracle-style reference.

## Benchmark Settings

Run the following six settings on `case_exp`:

1. `clean`
2. `noise_1`
3. `noise_5`
4. `sparse_space`
5. `sparse_time`
6. `sparse_both`

Recommended run count:

- `3` seeds per setting for the paper table
- `1` seed for smoke tests

## Per-Run Output Fields

The CSV produced by the benchmark should contain:

- `case`
- `setting`
- `seed`
- `noise_level`
- `space_stride`
- `time_stride`
- `backbone`
- `neural_true_ErrD`
- `neural_true_ErrR`
- `neural_unseen`
- `symbolic_family_D`
- `symbolic_family_R`
- `symbolic_expr_D`
- `symbolic_expr_R`
- `symbolic_complexity_D`
- `symbolic_complexity_R`
- `symbolic_surrogate_ErrD`
- `symbolic_surrogate_ErrR`
- `symbolic_true_ErrD`
- `symbolic_true_ErrR`
- `symbolic_unseen`
- `bir_D`
- `bir_R`

Here

```math
\mathrm{BIR}_D = \frac{\mathrm{symbolic\ true\ ErrD}}{\mathrm{neural\ true\ ErrD}}, \qquad
\mathrm{BIR}_R = \frac{\mathrm{symbolic\ true\ ErrR}}{\mathrm{neural\ true\ ErrR}}.
```

Values near `1` indicate that symbolic compression mostly inherits constitutive bias from the neural stage.

## Core Table

The main paper table should aggregate each setting across seeds and report:

- neural true closure errors
- symbolic surrogate compression errors
- symbolic true closure errors
- neural unseen rollout
- symbolic unseen rollout
- `BIR_D`, `BIR_R`

This table is the most compact statement of the full three-stage story.

## Core Figures

### Figure 1: Neural-to-Symbolic Error Propagation

For each setting, plot:

- neural true closure error
- symbolic surrogate compression error
- symbolic true closure error

Use separate panels for diffusion and reaction, or use a left/right split.

Expected pattern:

- symbolic surrogate compression error stays low
- symbolic true closure error tracks neural true closure error

### Figure 2: Rollout Preservation

For each setting, compare:

- neural unseen rollout
- symbolic unseen rollout

Expected pattern:

- symbolic rollout remains close to neural rollout
- additional rollout degradation from compression is small

### Figure 3: Bias Inheritance Ratio

Plot `BIR_D` and `BIR_R` across settings.

Expected pattern:

- ratios remain near `1`
- symbolic compression preserves, rather than repairs, constitutive bias

## Main Claims Supported By This Benchmark

1. Restricted symbolic families can compress learned neural constitutive surrogates with low additional approximation error.
2. Forward rollout quality is largely preserved after symbolic compression.
3. True constitutive bias is inherited from the neural stage rather than corrected by the symbolic stage.
4. Therefore, the main bottleneck in neural-symbolic closure discovery is Stage 1 numerical constitutive recovery.

## Abstract-Level Summary

A concise abstract-compatible summary is:

> We study neural-symbolic closure discovery under function-class mismatch. Our pipeline first learns numerical constitutive surrogates from spatiotemporal data, then compresses them into restricted symbolic families, and finally validates the symbolic laws through forward simulation. Experiments show that symbolic compression faithfully preserves learned neural dynamics with little additional rollout degradation, but does not automatically repair constitutive bias inherited from the neural surrogate.
