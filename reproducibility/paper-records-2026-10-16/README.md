# Numerical records for the 2026-10-16 manuscript revision

This directory contains the 41 frozen CSV/JSON records used by the revised
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
  common-support, complexity, and constrained-fit results. Raw experimental
  data are obtained with `experiments/download_rheology_data.py`.
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

The command verifies all 41 file hashes, CSV shapes, JSON record counts,
the 100-model cohort, and all 576 BIR confidence intervals. It reads the
published records without training models or changing any files.

The manuscript source package includes an identical copy of this directory
as `LaTeX/Evidence/`.
