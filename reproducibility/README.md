# Reproduction guide

Run commands from the repository root unless stated otherwise. Use a new output
directory whenever the numerical source, environment or configuration changes.
The scripts retain failed and undetermined cases; a completed process is not
automatically a successful scientific result.

## Frozen theory results

This is the shortest complete reproduction of the added theory diagnostics. It
uses only files in this checkout and needs no GPU, network access during the run,
manuscript source or external experiment workspace.

```bash
uv venv .venv-theory
uv pip install --python .venv-theory/bin/python -r reproducibility/requirements-theory.txt
.venv-theory/bin/python research/theory-pipeline-2026-10-15/reproduce_numerics.py --output outputs/theory-recomputed --workers 4
```

`--output` must not already exist. Inputs resolve from the script location;
relative output paths resolve from the caller's working directory. The recorded
validation used Python 3.12 on Linux. The multiprocessing driver uses `fork`.

The driver checks all 81 input files before computation and then recomputes:

| Calculation | Expected coverage |
| --- | --- |
| Response enclosures | 20 seed/initial-field pairs; 60 endpoints for each state predictor |
| Time refinement | Two designated critical seed/field pairs |
| Observation ranking | Five Case A and five Case Exp seed records |
| Complete PDE response metrics | Five seeds at three horizons; 15 matrices |
| Nonlinear projection controls | Two amplitudes for each of five seed-specific directions |
| Presentation | Two LaTeX tables and a three-panel figure in PDF, PNG and SVG |

All response-count dictionaries must match the frozen references. Ranking
interval comparisons use `rtol=2e-8, atol=2e-13`; selected matrix diagnostics use
`rtol=2e-7, atol=2e-10`. The driver additionally verifies derivative, Jacobian,
interval and nonlinear-control inequalities. Successful completion writes
`outputs/theory-recomputed/REPRODUCTION.json` with `status: passed`.

The Case A second-state intervals determine 59 of 60 endpoint directions, with
no incorrect determined direction. One endpoint remains undetermined. All five
Case Exp intervals with approximation remainders straddle zero. The 60 endpoints
reuse five independent seed units. Synthetic truth supplies the diagnostic
consistency budget; this is not an inferred noise guarantee. Padded float64
enclosures are numerically audited, not formally verified rounding proofs.

### Input identity

`theory/inputs/` contains the native weights, exact symbolic formulas, saved
observations, quadratic objective matrices and previously computed variational
fields used by the published diagnostics. Numeric NPZ files retain their original
bytes. Machine-specific paths in JSON provenance were replaced by repository
relative artifact identifiers; no numerical values were changed.

`theory/INPUT_MANIFEST.json` records both original and public SHA-256 digests,
byte lengths and whether metadata paths were normalized. Historical source
digests inside archived records describe the original experiment. Newly produced
records contain the public implementation's own digests. The two identities must
not be conflated.

`theory/expected/` retains the reference JSON results. The recomputation writes to
a separate directory and does not replace these references. The independent
release check and numerical receipt are in `VALIDATION.json` and
`THEORY_REPRODUCTION.json`.

## Install the experiment and test environment

```bash
uv venv
uv pip install --python .venv/bin/python -r reproducibility/requirements-tests.txt -e ".[dev,revision]"
.venv/bin/python -m pytest -q
```

`requirements-tests.txt` pins the direct dependencies used for validation.
`requirements-linux-tested.txt` records the complete fresh Linux test environment,
including the installed CUDA Torch dependencies. It is a platform-specific
snapshot, not a requirement to use a GPU for CPU tests. The lighter
`requirements-theory.txt` suffices for the frozen theory calculation.

On a compatible CUDA machine, also run:

```bash
CLOSURE_TEST_CUDA=1 .venv/bin/python -m pytest -q -m cuda
```

The base 1D/2D campaign and refit commands are documented in the root README and
`REVIEWER_EXPERIMENTS.md`. Those commands generate their own observations, seeds,
protocols, checkpoint identities and failure records.

## Additional PDE and rheology campaigns

These are complete experiment commands, so the neural training stages can take
substantially longer than the frozen theory check. Their configurations and
budgets are declared in the respective driver sources. Publication does not
imply that every long campaign was retrained during the release checks.

Analytical PDE controls:

```bash
.venv/bin/python experiments/basic_pde_controls.py --output outputs/basic-pde
```

Linear PDE operator-learning campaign:

```bash
.venv/bin/python experiments/linear_pde_extension.py --output outputs/linear-pde
.venv/bin/python experiments/linear_pde_forward_audit.py --output outputs/linear-pde
.venv/bin/python experiments/linear_pde_projection_measure.py --output outputs/linear-pde
.venv/bin/python experiments/linear_pde_report.py --output outputs/linear-pde
```

Viscous Burgers flux campaign:

```bash
.venv/bin/python experiments/burgers_campaign.py train --output outputs/burgers
.venv/bin/python experiments/burgers_campaign.py evaluate --output outputs/burgers
.venv/bin/python experiments/burgers_campaign.py report --output outputs/burgers
```

The rheology input is the versioned, CC BY 4.0 dataset
[Rheological Data For Syntactic Foam](https://doi.org/10.5281/zenodo.19699588)
by Caleb Beckwith and Nikhil Gupta. The download helper checks the recorded
archive byte count and checksum and retains the source attribution.

```bash
.venv/bin/python experiments/download_rheology_data.py --output outputs/rheology-raw
.venv/bin/python experiments/rheology_data.py --raw outputs/rheology-raw --output outputs/rheology
cp research/rheology-engineering-2026-10-13/PROTOCOL.json outputs/rheology/
cp research/rheology-engineering-2026-10-13/COMPLEXITY_FOLLOWUP_PROTOCOL.json outputs/rheology/
.venv/bin/python experiments/rheology_train.py --output outputs/rheology --device cuda
.venv/bin/python experiments/rheology_evaluate.py --output outputs/rheology
.venv/bin/python experiments/rheology_complexity.py --output outputs/rheology
.venv/bin/python experiments/rheology_monotone.py --source outputs/rheology --output outputs/rheology-monotone
.venv/bin/python experiments/rheology_report.py --output outputs/rheology
```

## Weight and objective studies

Study directories retain distinct protocols and cohort identities. Generate new
manifests from the current checkout; historical source hashes are not substituted
for the current implementation.

A representative 150-epoch CUDA prefix of the sensitivity study can be run with:

```bash
.venv/bin/python research/reviewer-sensitivity-2026-10-15-v2/make_manifest.py --output outputs/sensitivity/manifest.json
.venv/bin/python research/reviewer-sensitivity-2026-10-15-v2/prepare_inputs.py --output-root outputs/sensitivity
.venv/bin/python research/reviewer-sensitivity-2026-10-15-v2/run_study.py run --manifest outputs/sensitivity/manifest.json --output-root outputs/sensitivity --device cuda --workers 1 --ids fresh_case_a_clean_s0_default --stop-epoch 150
```

This is a prefix check, not a completed 1500-epoch task. Only declared evaluation
epochs are accepted. Use `--phase all` without `--ids` for the full declared
weight sweep. Do not merge CPU and CUDA optimizer trajectories.

The separate `reviewer-sensitivity-2026-10-15-gpu` and
`reviewer-forward-2026-10-15-gpu` drivers preserve the paired GPU cohort and its
resource constraints. Their manifests depend on the explicitly named upstream
CPU prefix metadata, observation tables and objective bundles. Complete the
upstream campaigns before invoking these downstream audits; the frozen theory
reproduction above already includes its required exports.

The objective controls are prepared with the original sensitivity protocol:

```bash
.venv/bin/python research/reviewer-sensitivity-2026-10-15/make_manifest.py
.venv/bin/python research/reviewer-sensitivity-2026-10-15/prepare_inputs.py
.venv/bin/python research/reviewer-objective-controls-2026-10-15/run_controls.py --mode shared --workers 1
```

After all 20 shared bundles are complete, the regularization intervention uses
those saved inputs. It snapshots their current hashes before solving:

```bash
.venv/bin/python research/reviewer-regularization-scaling-2026-10-15/run_study.py --output-root outputs/regularization
.venv/bin/python research/reviewer-regularization-scaling-2026-10-15/report_study.py --output-root outputs/regularization
```

Use `--input-root` and `--bundle-root` on the intervention driver for other
locations. `--prepare-only` validates and records the 20-bundle input snapshot
without solving. The selector audit similarly accepts `--historical-root` and
`--same-observation-root` for independently regenerated campaign evidence.

## Publication checks

The repository contains scientific code and selected numerical fixtures.
Manuscripts, review correspondence, local delivery tooling, credentials and
machine environment dumps are excluded. A staged-content scan also opens the
string metadata inside NPZ files:

```bash
.venv/bin/python reproducibility/audit_public_files.py
```

The scan reads the Git index, so stage intended changes before invoking it. It
reports file names and rule names without printing matched private values.
