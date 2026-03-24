# closure-bias-inheritance

Code for neural-symbolic closure discovery in reaction-diffusion systems with known PDE structure and unknown constitutive laws.

The repository studies the problem

```math
u_t = \partial_x \left( D(u)\,u_x \right) + R(u),
```

with a focus on one paper-facing mechanism claim:
restricted symbolic compression usually preserves the constitutive bias already present in the learned numerical surrogate, rather than automatically repairing it.

## Scope

- 1D periodic reaction-diffusion solver
- minimal 2D periodic appendix benchmark for dimensionality checking
- weak-form-driven hybrid Stage-1 closure identification
- polynomial baselines for matched-library reference cases
- restricted symbolic compression and forward rollout validation
- multi-case Stage-1 noise-sensitivity sweeps
- excitation and cross-resolution diagnostics
- optional PySR-based Stage-2 replacement check

The current paper-facing synthetic cases are:

- `case_a`: matched-library reference case
- `case_b`: second matched-library reference case
- `case_exp`: smooth non-polynomial mismatch stress test

## Repository Layout

```text
github_release/
├─ src/closure_discovery/   # core library code
├─ experiments/            # experiment and benchmark entry points
├─ paper/                  # public manuscript source and compiled PDF
├─ results/paper_artifacts/# paper-facing table summaries and figure assets
├─ pyproject.toml          # package metadata
├─ LICENSE
└─ CITATION.cff
```

## Installation

Core dependencies are listed in `pyproject.toml`.

```bash
python -m pip install -e .
```

The PySR benchmark is optional and may require extra local setup beyond the core dependencies.

## Public Manuscript Material

The repository also includes the current public manuscript source and selected paper-facing artifacts:

- `paper/main.tex`: single-file manuscript source
- `paper/references.bib`: bibliography database
- `paper/main.pdf`: compiled PDF snapshot
- `paper_bundle_latest.zip`: downloadable zip snapshot of the entire `paper/` directory
- `paper/noise_sweep_all_cases.png`: appendix multi-case noise-sweep figure
- `paper/case_exp_observation_breakdown.png`: appendix observation-breakdown figure
- `results/paper_artifacts/tables/`: CSV/Markdown table summaries used for paper figures/tables
- `results/paper_artifacts/figures/`: exported paper-facing figures
- `results/noise_sweep_case_{a,b,exp}.{csv,md}`: detailed Stage-1 noise-sweep summaries

## Quick Start

Smoke test:

```bash
PYTHONPATH=src python experiments/run_case_a_mvp.py --epochs 5 --num-trajectories 4
```

Representative paper-facing runs:

```bash
PYTHONPATH=src python experiments/benchmark_polynomial_baselines.py --case case_exp --num-seeds 3
PYTHONPATH=src python experiments/benchmark_symbolic_compression.py --case case_exp --num-seeds 3
PYTHONPATH=src python experiments/benchmark_noise_sweep.py --case case_exp --num-seeds 3
PYTHONPATH=src python experiments/compare_excitation_protocol.py --case case_exp --num-seeds 3
PYTHONPATH=src python experiments/cross_resolution_benchmark.py --case case_exp --num-seeds 3
PYTHONPATH=src python experiments/objective_ablation.py --case case_exp --num-seeds 3
PYTHONPATH=src python experiments/benchmark_2d_bias_inheritance.py --cases case_a case_exp --num-seeds 3
```

Optional PySR Stage-2 replacement check:

```bash
PYTHONPATH=src python experiments/run_pysr_symbolic_compression.py --case case_exp --seed 0
```

## Notes

- The paper-facing Stage-1 learner is a weak-form-driven hybrid objective, not pure weak-form training.
- The main benchmark suite is 1D; the public repository also includes a minimal 2D appendix check showing the same bias-inheritance mechanism on small periodic runs.
- Experiment outputs are generated locally and are ignored by default.
- This repository is the public code-and-manuscript release; large training outputs beyond the paper-facing artifacts remain excluded.

## License

MIT, see `LICENSE`.
