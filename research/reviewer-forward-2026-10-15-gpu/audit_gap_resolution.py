"""Join saved-field temporal audits to paired forward error differences.

Each available seed-zero comparison is audited in endpoint/space-time and
relative/absolute norms. Other seeds/references remain explicitly unaudited.
No training, fitting, integration, or result selection is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent
REPORTS = STUDY.parents[1] / "outputs" / STUDY.name / "reports"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def flag(value):
    return value == "True"


def key(row):
    return row["unit_id"], row["pattern"], float(row["horizon"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    paths = {name: REPORTS / filename for name,filename in [
        ("pairs", "forward_paired_by_seed.csv"), ("absolute", "forward_absolute_by_seed.csv"),
        ("temporal", "temporal_trajectory_by_seed.csv")]}
    paired, absolute, temporal = [read(paths[name]) for name in ("pairs", "absolute", "temporal")]
    absolute_index, temporal_index = {key(row):row for row in absolute}, {key(row):row for row in temporal}
    assert len(absolute_index) == len(absolute)
    assert len(temporal_index) == len(temporal) == 504
    complete = len(absolute) == 6000 and len(paired) == 8880
    if args.require_complete and not complete:
        raise RuntimeError("Final gap audit requires6000 absolute rows and8880 planned paired contrasts")
    rows = []
    for pair in paired:
        common = dict(contrast=pair["contrast"], case=pair["case"], setting=pair["setting"],
            data_seed=int(pair["data_seed"]), variant=pair["variant"], epoch=pair["epoch"],
            method=pair["method"], pattern=pair["pattern"], horizon=float(pair["horizon"]),
            first_unit_id=pair["first_unit_id"], second_unit_id=pair["second_unit_id"])
        indexes = [(pair[label], pair["pattern"], float(pair["horizon"])) for label in ("first_unit_id", "second_unit_id")]
        metric_rows, temporal_rows = [absolute_index.get(index) for index in indexes], [temporal_index.get(index) for index in indexes]
        for metric, metric_field, delta_field, truth_field, normalization, expected_normalization in [
            ("endpoint", "endpoint_error", "endpoint_absolute_rms_difference", "truth_endpoint_rms", "endpoint_normalization", "relative_L2"),
            ("space_time", "space_time_error", "space_time_absolute_rms_difference", "truth_space_time_rms", "space_time_normalization", "relative_space_time_L2")]:
            for norm in ("reported_relative_or_fallback", "absolute_RMS"):
                out = dict(**common, metric=metric, norm=norm, both_scored=False,
                    error_second_minus_first=None, summed_temporal_discrepancy=None,
                    absolute_gap_over_temporal_margin=None, status="no_pairwise_temporal_audit")
                if not all(metric_rows):
                    out["status"] = "metric_snapshot_pending"
                    rows.append(out); continue
                both_scored = all(flag(row["reached"]) for row in metric_rows)
                out["both_scored"] = both_scored
                if not both_scored:
                    out["status"] = "unresolved_unreached_horizon"
                    rows.append(out); continue
                field = metric + "_absolute_rms" if norm == "absolute_RMS" else metric_field
                gap = float(metric_rows[1][field]) - float(metric_rows[0][field])
                out["error_second_minus_first"] = gap
                if not all(temporal_rows) or any(row["snapshot_status"] != "evaluated" for row in temporal_rows):
                    rows.append(out); continue
                if any(not flag(row["both_reached"]) or flag(row["failed_comparison_unresolved"]) or not flag(row["endpoint_temporal_tolerance_passed"]) for row in temporal_rows):
                    out["status"] = "unresolved_refinement_failure"
                    rows.append(out); continue
                assert metric_rows[0][normalization] == metric_rows[1][normalization]
                assert float(metric_rows[0][truth_field]) == float(metric_rows[1][truth_field])
                denominator = float(metric_rows[0][truth_field]) if norm != "absolute_RMS" and metric_rows[0][normalization] == expected_normalization else 1.
                margin = sum(float(row[delta_field]) for row in temporal_rows) / denominator
                out.update(summed_temporal_discrepancy=margin,
                    absolute_gap_over_temporal_margin=abs(gap) / margin if margin > 0 else None,
                    status="resolved_on_declared_refined_seed_subset" if abs(gap) > margin else "numerically_unresolved_gap")
                rows.append(out)
    output = REPORTS / "forward_gap_resolution_by_seed.csv"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(output)
    summary = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), complete=complete,
        paired_contrasts_evaluated=len(paired), planned_paired_contrasts=8880,
        metric_norm_comparisons=len(rows), planned_metric_norm_comparisons=35520,
        status_counts=dict(Counter(row["status"] for row in rows)),
        source_sha256={name:digest(path) for name,path in paths.items()},
        script_sha256=digest(Path(__file__)), output_sha256=digest(output),
        fits_or_integrations_performed=0,
        rule="A method error gap is resolved only if both full primary/refined trajectories complete, endpoint tolerance checks pass, and the gap exceeds summed temporal field discrepancies in the same norm. This is an observed refinement comparison, not a rigorous error bound or spatial continuum certificate.",
        replication_rule="Metric/norm/horizon/IC/checkpoint/representation contrasts reuse paired data seeds and are not new independent scientific replicates.")
    path = REPORTS / "GAP_RESOLUTION_AUDIT.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key:summary[key] for key in ("complete", "paired_contrasts_evaluated", "metric_norm_comparisons", "status_counts")}, indent=2))


if __name__ == "__main__":
    main()
