# closure-bias-inheritance

Companion code and reproducibility artifacts for neural-symbolic closure discovery in 1D reaction-diffusion systems, focusing on **bias inheritance**: restricted symbolic compression tends to preserve (not automatically repair) constitutive bias learned by a neural surrogate.

We consider the PDE

```math
u_t = \partial_x \left( D(u)\,u_x \right) + R(u),
```

where the PDE structure is known but the constitutive laws `D(u)` and `R(u)` are unknown.

## What This Repo Contains

- `src/closure_discovery/`: solver, weak-form residuals, models, symbolic restricted fitting, and evaluation metrics.
- `experiments/`: benchmark scripts (polynomial baselines, excitation, cross-resolution, symbolic compression).
- `results/`: saved benchmark CSVs used by the manuscript.
- `paper_bundle/`: self-contained LaTeX draft (includes an Elsevier/JCP-style `elsarticle` entry) plus copied figures/CSV artifacts.

Implemented synthetic cases:

- `case_a`, `case_b`: matched-library polynomial closures.
- `case_exp` (paper stress test) and `case_c`: smooth non-polynomial closures for mismatch checks.

## Quick Start

Run a short smoke test:

```bash
PYTHONPATH=src python experiments/run_case_a_mvp.py --epochs 5 --num-trajectories 4
```

## Build The Paper PDF

Draft (article class):

```bash
./paper_bundle/build.sh
```

Elsevier/JCP-style entry (`elsarticle`):

```bash
./paper_bundle/build.sh main_elsarticle.tex
```

## License

MIT, see `LICENSE`.

