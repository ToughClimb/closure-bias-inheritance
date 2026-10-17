# Numerical records for the 2026-10-16 manuscript revision

This directory contains the 76 frozen CSV/JSON/NPZ files used by the revised
paper: full-precision coefficients, per-dataset measurements, statistical
summaries, forward checks, and rheology diagnostics. `MANIFEST.json` lists
every data file, its size, SHA-256, CSV columns and row count, and the hash
of the original audited record.

## Files cited by name in the paper

| File | Contents | Paper location |
| --- | --- | --- |
| [sameobs_model_descriptors.json](sameobs_model_descriptors.json) | Ten Exp datasets and 100 model descriptors, including neural weights, selected symbolic expressions, and observation-based coefficient refits | Appendix F, complete coefficients |
| [campaign_bir_summary.csv](campaign_bir_summary.csv) | 576 branch/metric summaries, each with five datasets, mean, sample SD, and 95% Student-t interval | Appendices C and D, BIR and uncertainty |
| [campaign_metrics_summary.csv](campaign_metrics_summary.csv) | Closure errors and their sample SDs and confidence intervals for the expanded campaign | Appendices C and D |
| [campaign_metrics_raw.csv](campaign_metrics_raw.csv) | Per-seed closure metrics underlying the expanded-campaign summaries | Appendices C and D |

## Additional records

- Appendix B: `campaign_observation_diagnostics.csv`,
  `campaign_noise_objective_diagnostic.csv`, and
  `campaign_loss_magnitudes_raw.csv` contain the observation-noise and
  objective-component diagnostics.
- Appendices A and D: `campaign_2d_metrics_raw.csv`,
  `campaign_2d_metrics_summary.csv`, `campaign_convergence_dataset_summary.csv`,
  and the `campaign_metrics_*` files contain the expanded dimensionality,
  convergence, and baseline records.
- Appendix E: `campaign_pysr_metrics_summary.csv`,
  `campaign_thresholds_raw.csv`, `campaign_thresholds_summary.csv`, and
  `mechanism_projection_checks.json` contain the PySR, family-selection,
  and projection diagnostics. The family tasks also appear in
  `campaign_metrics_*` and `campaign_bir_summary.csv`.
- Appendix F: `sameobs_*`, `anchor_*`, and the remaining `mechanism_*.json`
  files contain the exact-observation comparisons, training budgets,
  paired effects, state coverage, physical checks, and anchor controls.
  `sameobs_forward_details.csv` retains individual pattern/seed outcomes.
- Appendix H: `rheology/` contains six tables of per-test, per-model,
  common-support, complexity, and constrained-fit results. The delivered `inputs/` include normalized original measurements,
  training/held-out row IDs and both checkpoints. `source_attribution.json`
  identifies Beckwith and Gupta's CC-BY-4.0 data source (Zenodo 19699588).
  The original source files can also be obtained with
  `experiments/download_rheology_data.py`.
- Appendix G uses the separately frozen inputs and expected results in
  [`../theory/`](../theory/).

The historical expanded campaign and the exact-observation supplement retain
their separate cohort identities. Ratios and paired differences are formed
within datasets before aggregation. Full precision is retained in these
records; manuscript tables round the displayed values.

Ten `source` fields in `sameobs_model_descriptors.json` identify historical
run records under `outputs/` using repository-relative paths. The included
model descriptors contain the coefficients and weights directly. This path
normalization preserves all numeric values and is recorded in the manifest.

## Verify the checked-in records

From the repository root, using Python 3 and its standard library:

```bash
python3 reproducibility/verify_paper_records.py
```

The command verifies every manifest file hash, CSV shapes, JSON record counts,
the 100-model cohort, and all 576 BIR confidence intervals. It reads the
published records without training models or changing any files.

The manuscript source package includes an identical copy of this directory
as `LaTeX/Evidence/`.

## Targeted completion of the saved evidence (2026-10-17)

[`TABLE_RECORD_INDEX.csv`](TABLE_RECORD_INDEX.csv) maps all 15 manuscript
tables to their records, using the manuscript's stable table labels and
displayed numbers. The 41 files published previously retain their hashes.

- `twod/` supplies all 25 original task JSONs and 200 per-seed validation
  batches, including four failed strong-polynomial batches. Endpoint errors
  are aggregated only for completed trajectories. Neural/frozen/weak
  polynomial completion is 50/50 each, and strong polynomial is 46/50.
  Actual training length is 150 epochs.
- `original_tables/` preserves the main-table aggregate CSVs. Original
  Table 2 models were not serialized by its driver; the narrow-support
  model-level diagnostic in Appendix F is an independently generated cohort.
- `pysr_original/` preserves the six original Exp clean/5%-noise seed CSVs,
  its aggregate table and its verified-versus-unrecorded protocol metadata.
  These operators omit division. The five-seed clean/3%-noise expansion
  uses a distinct cohort and adds division.
- `rheology/constraint_fit_records.json` contains all 150 original convex-fit
  outputs: 28 coefficients per fit, 36 Bernstein margins, dual multipliers,
  exact row IDs/bounds, and optimizer/primal/dual/KKT diagnostics.
  The unconstrained/complexity/constrained model descriptors, original teacher
  weights, observations and splits permit direct re-evaluation.
- `rheology/comparison_by_fold.csv` averages the two teacher initializations
  within each held-out physical test; `comparison_summary.csv` reports the
  50-test means and descriptive sample SDs used in Table H.15.

Recheck all 150 constrained fits, their held-out relative L2 errors, processing
window discrepancies, Bernstein margins, and KKT/primal-dual residuals:

```bash
python reproducibility/verify_rheology_records.py
```

This second check uses the research environment (NumPy, pandas, SciPy and
PyTorch), reads saved arrays, and calls no optimizer or neural training.
`TARGETED_PROVENANCE.json` records original-file hashes for the added inputs.
