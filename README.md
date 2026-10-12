# closure-bias-inheritance

Neural and symbolic closure discovery for periodic reaction-diffusion systems with known PDE structure and unknown constitutive laws:

```math
u_t = \partial_x(D(u)\,u_x) + R(u).
```

The experiments distinguish **faithful compression of a frozen surrogate** from **refitting a symbolic expression to observations**. Faithful compression can inherit a surrogate's bias; this does not imply that symbolic methods cannot correct it. Historical clean Case Exp experiments reduced mean reaction error from approximately 0.441 to 0.0092 after an observation-only forward refit. These are archived CPU results, not new results from the repaired implementation.

## Layout

- `src/closure_discovery/`: solvers, observations, neural models, weak objectives, polynomial fits and restricted symbolic expressions.
- `experiments/`: original benchmarks and optional PySR comparison.
- `revision/`: resumable experiment runners, observation refits, validation, portable exports, reports and provenance.
- `tests/`: numerical invariants, optimizer resume checks and experiment-integrity regressions.
- `results/paper_artifacts/`: historical selected tables and figures.
- `CODE_AUDIT_FIXES.md`: defect mapping, validation and compatibility notes.

Revision tools use this repository's core implementation. Generated outputs and checkpoints are ignored by Git. Manuscript and review correspondence are maintained separately.

## Install and test

Linux commands for a fresh checkout:

```bash
uv venv
uv pip install --python .venv/bin/python -e ".[dev,revision]"
.venv/bin/python -m pytest
```

If a virtual environment already exists in the parent workspace, keep it there and use `../.venv/bin/python` in place of `.venv/bin/python`. The current local workspace retains that environment and its CUDA Torch installation. SciPy 1.17.0 and pandas 2.2.3 were selected to match the validated migration environment; the package extras allow compatible newer versions.

Optional PySR requires its Python package and a working Julia runtime:

```bash
uv pip install --python .venv/bin/python -e ".[pysr]"
```

The regression suite tests PySR failure handling and orchestration with a simulated backend. It does not validate a new Julia search. Existing 30-search results can be read and summarized without repeating searches.

## Small runs

Original pipeline smoke check:

```bash
.venv/bin/python experiments/run_case_a_mvp.py --epochs 5 --num-trajectories 4
```

Prepare a small revision task configuration:

```json
{"tasks": [{"group": "ablation", "case": "case_exp", "setting": "clean", "seed": 0, "epochs": 5, "classification": "runtime_pilot"}]}
```

Save it as `pilot.json`, then run:

```bash
.venv/bin/python revision/run_suite.py --config pilot.json --workers 1 --device cpu --work-root outputs/pilot
```

Repeat the same command to resume an interrupted run or check an existing completed result. Configurations, observations, device, implementation hashes and required outputs must match. For CUDA, select `--device cuda`; `auto` chooses CUDA when available. Use one worker for a single GPU.

CUDA checks are opt-in:

```bash
CLOSURE_TEST_CUDA=1 .venv/bin/python -m pytest -m cuda
```

## Campaign and resume policy

A manifest can be generated without training:

```bash
.venv/bin/python revision/run_suite.py --group all --manifest-only --work-root outputs/new-campaign
```

The repaired declaration contains 445 valid 1D configurations. The historical declaration had 450, including five invalid nx8 runs with unresolved Nyquist sine tests. Those remain diagnostic historical inputs; the valid nx8 arm requests three Fourier modes. Nominal nx64 test functions are unchanged.

Checkpoint schema 2 atomically stores model, Adam state, RNG state, complete epoch history and requested 1D curves. History and curve exports can be reconstructed from the checkpoint after an interrupted write. Restored parameters and optimizer tensors must be finite. Missing requested intermediate checkpoints are errors.

Historical inputs retain their original source identity. Use a new work root when changing code or protocol. Legacy checkpoints require explicit `EAAI_ALLOW_LEGACY_CHECKPOINT=1` and matching configuration, observations, complete external history and finite optimizer state. This is tested for compatible 1D optimizer checkpoints; old model-only files and legacy 2D files without required metadata are insufficient for safe resume.

Failed neural or PySR tasks remain failures, including when cached, and cause a nonzero command exit. `--retry-failed` explicitly retries them while archiving the earlier failure. Successful PySR cache reuse checks the frozen target, source, search budget and protocol. Viewing help and reading a cache do not initialize Julia.

Validation commands:

```bash
.venv/bin/python revision/run_refit_rollout.py --case case_exp --groups main --refit-pde --workers 1 --work-root outputs/new-campaign
.venv/bin/python revision/run_short_validation.py --workers 1 --work-root outputs/new-campaign
.venv/bin/python revision/run_convergence_validation.py --workers 1 --work-root outputs/new-campaign
.venv/bin/python revision/run_2d.py --epochs 150 --workers 1 --work-root outputs/new-campaign
.venv/bin/python revision/run_pysr_suite.py --work-root outputs/new-campaign
```

These commands launch experiments; they are not required to rebuild already completed results. The 2D campaign checks isotropic scalar diffusion and reaction on periodic domains.

## Rebuild existing evidence

Point `--input-root` at an existing project containing `evidence/` and, optionally, binary outputs. Write regenerated reports into a separate work root:

```bash
.venv/bin/python revision/rebuild_reports.py --input-root /path/to/existing-project --work-root outputs/rebuilt
.venv/bin/python revision/audit_bundle.py --input-root /path/to/existing-project --work-root outputs/rebuilt --require-full
.venv/bin/python revision/record_provenance.py --input-root /path/to/existing-project --work-root outputs/rebuilt
```

Portable CSV curves and histories suffice for report reconstruction. Model training and validation use local outputs in their chosen work root; historical input paths are read-only. The bundle audit checks expected configuration identities, final epochs, histories, source hashes and curve metrics, and identifies the five historical invalid Nyquist runs separately.

`verify_portability.py` compares a complete representative Case Exp rerun against a complete reference, including all 37 closure checks and 56 validation checks. Missing methods, truncated series or incomplete coverage fail the comparison.

The original optional PySR runner includes case, observation setting, seed, operators, training protocol and search identity in default CSV paths. Summaries group by these identities and reject duplicate seeds. Legacy rows lack enough metadata to verify protocol equivalence; they are labeled unverified, and modern restricted references must match the exact frozen target.

## Numerical scope

- Spatial strides must divide every periodic grid dimension and leave at least two cells; at least three saved times are required for central time derivatives.
- Time-weak objectives use trapezoidal endpoint quadrature. The primary spatial-weak objective is unchanged.
- Rational expressions record both polynomial degrees. Historical five-coefficient degree-2/degree-2 expressions remain supported.
- Strict validation infers resolution from the initial state and saves the exact requested final time.
- Validation failures retain status and stability metadata; partial rollout errors do not establish successful completion.
- Local Linux/WSL CUDA checks cover short 1D and 2D optimizer resume. Full GPU campaigns and native Windows execution remain unvalidated.

MIT license; see [LICENSE](LICENSE).

Additional reviewer experiments and coefficient-refit commands are described in [REVIEWER_EXPERIMENTS.md](REVIEWER_EXPERIMENTS.md).
