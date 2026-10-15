"""Describe all GPU forward directions in function-only norm groups."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
REPORTS = REPO / "outputs" / STUDY.name / "reports"
OUTPUT = REPORTS / "function_norm_sensitivity"
SOURCE = REPORTS / "function_forward_links_by_seed.csv"


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    function_path = OUTPUT / "function_norm_robustness_by_checkpoint.csv"
    functions, links = read(function_path), read(SOURCE)
    assert len(functions) == 200 and len(links) == 4800
    background = {row["pair_id"] for row in functions}
    assert len(background) == 200
    assert {row["pair_id"] for row in links} == background
    selected = {
        "all_200_background": background,
        "original_256_discrete_both_improve": {
            row["pair_id"] for row in functions
            if row["original_256_discrete_both_function_errors_improve"] == "True"},
        "all_five_norms_grids_both_improve": {
            row["pair_id"] for row in functions
            if row["all_grid_norms_both_function_errors_improve"] == "True"}}
    keys = ("case", "setting", "variant", "epoch", "pattern", "horizon", "metric")
    cohort_keys = sorted({tuple(row[key] for key in keys) for row in links})
    rows, overall = [], []
    for name, identifiers in selected.items():
        chosen = [row for row in links if row["pair_id"] in identifiers]
        counts = Counter(row["prediction_direction"] for row in chosen)
        overall.append(dict(group=name, selected_unique_checkpoint_pairs=len(identifiers),
            all_unique_checkpoint_pairs=200, paired_IC_horizon_metric_comparisons=len(chosen),
            frozen_better=counts["better"], frozen_worse=counts["worse"],
            tied=counts["tied_within_reporting_tolerance"], unreached=counts["unreached_pair"],
            resolved_frozen_better=sum(row["prediction_direction"] == "better" and row["temporal_resolution"] == "resolved_on_declared_refined_seed_subset" for row in chosen),
            resolved_frozen_worse=sum(row["prediction_direction"] == "worse" and row["temporal_resolution"] == "resolved_on_declared_refined_seed_subset" for row in chosen)))
        for key in cohort_keys:
            group = [row for row in links if tuple(row[field] for field in keys) == key]
            assert len(group) == 5
            subset = [row for row in group if row["pair_id"] in identifiers]
            counts = Counter(row["prediction_direction"] for row in subset)
            rows.append(dict(group=name, **dict(zip(keys, key)), background_data_seeds=5,
                selected_data_seeds=len(subset), frozen_better=counts["better"], frozen_worse=counts["worse"],
                tied=counts["tied_within_reporting_tolerance"], unreached=counts["unreached_pair"],
                numerically_resolved_selected_comparisons=sum(row["temporal_resolution"] == "resolved_on_declared_refined_seed_subset" for row in subset)))
    assert len(rows) == 2880
    output = OUTPUT / "robust_group_forward_direction_by_cohort_IC_horizon.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status="passed", complete=True, cohort="fresh_GPU150_1500_only",
        snapshot_utc=datetime.now(timezone.utc).isoformat(),
        group_definition="Uniform function-only selection applied to all200 fresh GPU checkpoint pairs; forward-error magnitude and direction never enter selection.",
        source_function_sha256=digest(function_path), source_forward_sha256=digest(SOURCE),
        script_sha256=digest(Path(__file__)), output_sha256=digest(output), groups=overall,
        cohort_IC_horizon_metric_group_rows=len(rows), new_fits_or_integrations=0,
        interpretation="All ICs, horizons, metrics, epochs and cohorts are retained, including zero selected-seed cells. Counts reuse paired checkpoints and data seeds and are descriptive comparisons, not independent replicates. No illustrative winner is selected.")
    (OUTPUT / "ROBUST_GROUP_FORWARD_COUNTS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
