# Closure Discovery

Reaction-diffusion closure discovery with a weak-form, physics-constrained learning pipeline.

The current repository is scoped to a frozen `v1` research protocol:

- 1D scalar reaction-diffusion systems
- Case A and Case B only
- Known PDE structure, unknown `D(u)` and `R(u)`
- Weak-form training objective
- Interpretable scalar neural surrogates for closures
- Explicit evaluation of identifiability and excitation coverage
- Forward re-simulation as a hard validation criterion
- Cross-resolution and cross-time-step checks to reduce inverse-crime risk

The intended target problem is

```math
u_t = \partial_x \left(D(u)\partial_x u\right) + R(u).
```

## Why This Scope

The proposed paper direction is feasible, but only if the first implementation avoids two common traps:

1. Starting from 2D and rational closures immediately.
2. Treating KAN as a prerequisite instead of an ablation or later upgrade.

This repository therefore starts from a stable 1D baseline with a clean module interface. Once the weak-form discovery loop is reliable, KAN, symbolic refinement, sparse sampling, and 2D experiments can be added without rewriting the core pipeline.

The first paper is framed around a stronger question than pure loss minimization:

> when does a low weak-form residual imply correct constitutive recovery, and how should that recovery be validated?

## Repository Layout

```text
closure_discovery/
├─ docs/
│  ├─ feasibility.md
│  ├─ roadmap.md
│  └─ v1_protocol.md
├─ experiments/
│  ├─ benchmark_polynomial_baselines.py
│  ├─ compare_excitation_protocol.py
│  ├─ compare_polynomial_baselines.py
│  ├─ cross_resolution_benchmark.py
│  ├─ cross_resolution_protocol.py
│  ├─ evaluate_stability_protocol.py
│  ├─ run_case_protocol.py
│  └─ run_case_a_mvp.py
├─ src/
│  └─ closure_discovery/
│     ├─ baselines/
│     │  └─ polynomial.py
│     ├─ data_generation/
│     │  ├─ cases.py
│     │  ├─ datasets.py
│     │  ├─ observations.py
│     │  └─ rd_solver_1d.py
│     ├─ evaluation/
│     │  ├─ metrics.py
│     │  └─ rollout.py
│     ├─ models/
│     │  ├─ mlp_closure.py
│     │  └─ tabulated_closure.py
│     ├─ pipelines/
│     │  └─ train_1d_closure.py
│     └─ weak_form/
│        ├─ test_functions.py
│        └─ weak_residual.py
└─ pyproject.toml
```

## Quick Start

The current environment already has `numpy`, `torch`, and `matplotlib`. The first smoke test can be run directly:

```bash
PYTHONPATH=src python experiments/run_case_a_mvp.py --epochs 5 --num-trajectories 4
```

This script:

- generates synthetic 1D trajectories for Case A
- optionally applies observation degradation through noise or downsampling
- builds periodic weak-form test functions
- trains an MLP closure model for a few epochs
- reports weak loss, rollout loss, closure errors, and excitation coverage summary

Useful protocol-oriented runs:

```bash
PYTHONPATH=src python experiments/run_case_protocol.py --case case_a --num-trajectories 16
PYTHONPATH=src python experiments/run_case_protocol.py --case case_b --num-trajectories 8
PYTHONPATH=src python experiments/run_case_protocol.py --case case_a --num-trajectories 16 --backbone kan --kan-grid-size 16
PYTHONPATH=src python experiments/compare_polynomial_baselines.py --case case_a
PYTHONPATH=src python experiments/compare_polynomial_baselines.py --case case_a --backbone kan --kan-grid-size 16
PYTHONPATH=src python experiments/benchmark_polynomial_baselines.py --case case_a --num-seeds 3
PYTHONPATH=src python experiments/compare_polynomial_baselines.py --case case_exp --diffusion-degree 2 --reaction-degree 3
PYTHONPATH=src python experiments/benchmark_polynomial_baselines.py --case case_a --num-seeds 3 --output-csv results/polynomial_baselines_case_a.csv
PYTHONPATH=src python experiments/benchmark_polynomial_baselines.py --case case_exp --num-seeds 3 --output-csv results/polynomial_baselines_case_exp.csv
PYTHONPATH=src python experiments/run_symbolic_compression.py --case case_exp --backbone mlp
PYTHONPATH=src python experiments/build_paper_artifacts.py --symbolic-csv results/symbolic_compression_case_a.csv --symbolic-csv results/symbolic_compression_case_exp.csv --baseline-csv results/polynomial_baselines_case_a.csv --baseline-csv results/polynomial_baselines_case_exp.csv --output-dir results/paper_artifacts
PYTHONPATH=src python experiments/compare_excitation_protocol.py --case case_b --num-seeds 3 --output-csv results/excitation_case_b.csv
PYTHONPATH=src python experiments/evaluate_stability_protocol.py --case case_a --num-seed-runs 3 --num-subset-runs 3
PYTHONPATH=src python experiments/cross_resolution_protocol.py --case case_a --space-stride 2 --time-stride 2
PYTHONPATH=src python experiments/cross_resolution_benchmark.py --case case_a --num-seeds 3 --output-csv results/cross_resolution_case_a.csv
```

The frozen `v1` protocol is documented in [docs/v1_protocol.md](docs/v1_protocol.md).
The exploratory `case_exp` setting is available for out-of-library stress tests, but it is not part of the frozen `v1` acceptance criteria.

## Immediate Next Steps

1. Make the 1D weak-form loop reliable on Cases A/B with stable closure recovery, not just low residual.
2. Add restricted-family symbolic fitting on learned closure samples.
3. Extend anti-inverse-crime evaluation into a larger benchmark matrix.
4. Add OOD unseen-rollout settings beyond the training amplitude range.
5. Evaluate the implemented KAN backbone as an ablation only after the current surrogate and baselines are fully benchmarked.

The detailed feasibility assessment and milestone plan are in [docs/feasibility.md](docs/feasibility.md), [docs/roadmap.md](docs/roadmap.md), [docs/v1_protocol.md](docs/v1_protocol.md), and [docs/paper_strategy.md](docs/paper_strategy.md).
