"""Describe all forward directions in uniformly defined function-norm groups."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

STUDY=Path(__file__).resolve().parent
REPO=STUDY.parents[1]
OUTPUT=REPO/"outputs"/STUDY.name
SOURCE=REPO/"outputs/reviewer-forward-2026-10-15-v2/reports/function_forward_links_by_seed.csv"

def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    function_path=OUTPUT/"function_norm_robustness_by_checkpoint.csv"
    functions=read(function_path); links=read(SOURCE)
    assert len(functions)==100 and len(links)==2400
    selected={
        "all_100_background":{row["pair_id"] for row in functions},
        "original_256_discrete_both_improve":{row["pair_id"] for row in functions if row["original_256_discrete_both_function_errors_improve"]=="True"},
        "all_five_norms_grids_both_improve":{row["pair_id"] for row in functions if row["all_grid_norms_both_function_errors_improve"]=="True"}}
    assert [len(selected[name]) for name in selected]==[100,68,47]
    keys=("case","setting","variant","epoch","pattern","horizon","metric")
    cohort_keys=sorted({tuple(row[key] for key in keys) for row in links})
    rows=[]; overall=[]
    for name,ids in selected.items():
        chosen=[row for row in links if row["pair_id"] in ids]
        summary=Counter(row["prediction_direction"] for row in chosen)
        overall.append(dict(group=name, selected_unique_checkpoint_pairs=len(ids),
            all_unique_checkpoint_pairs=100, paired_IC_horizon_metric_comparisons=len(chosen),
            frozen_better=summary["better"],frozen_worse=summary["worse"],
            tied=summary["tied_within_reporting_tolerance"],unreached=summary["unreached_pair"],
            resolved_frozen_better=sum(row["prediction_direction"]=="better" and row["temporal_resolution"]=="resolved_on_declared_refined_seed_subset" for row in chosen),
            resolved_frozen_worse=sum(row["prediction_direction"]=="worse" and row["temporal_resolution"]=="resolved_on_declared_refined_seed_subset" for row in chosen)))
        for key in cohort_keys:
            group=[row for row in links if tuple(row[field] for field in keys)==key]
            assert len(group)==5
            subset=[row for row in group if row["pair_id"] in ids]
            counts=Counter(row["prediction_direction"] for row in subset)
            rows.append(dict(group=name,**dict(zip(keys,key)),background_data_seeds=5,
                selected_data_seeds=len(subset),frozen_better=counts["better"],frozen_worse=counts["worse"],
                tied=counts["tied_within_reporting_tolerance"],unreached=counts["unreached_pair"],
                numerically_resolved_selected_comparisons=sum(row["temporal_resolution"]=="resolved_on_declared_refined_seed_subset" for row in subset)))
    assert len(rows)==1440
    output=OUTPUT/"robust_group_forward_direction_by_cohort_IC_horizon.csv"
    with output.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(status="passed",complete=True,snapshot_utc=datetime.now(timezone.utc).isoformat(),
        group_definition="Uniform function-only selection applied to all100CPU150 checkpoint pairs; no forward-error magnitude or direction enters selection.",
        source_function_sha256=digest(function_path),source_forward_sha256=digest(SOURCE),
        script_sha256=digest(Path(__file__)),output_sha256=digest(output),groups=overall,
        cohort_IC_horizon_metric_group_rows=len(rows),new_fits_or_integrations=0,
        interpretation="All ICs, horizons, metrics and cohorts retained, including zero selected-seed cells. Counts reuse paired data seeds/checkpoints and are descriptive comparisons, not independent replicates. The original illustrative example remains norm-sensitive and is not replaced.")
    (OUTPUT/"ROBUST_GROUP_FORWARD_COUNTS.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(overall,indent=2))

if __name__=="__main__":
    main()
