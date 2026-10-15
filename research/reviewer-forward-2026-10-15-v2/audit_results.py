"""Audit saved forward results without fitting, integrating, or changing the protocol.

This reads a point-in-time subset of the frozen 500-state manifest. It is safe to
run while the deterministic readiness queue is active: results are atomically
published, missing units stay pending, and no formal result is rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
OUTPUT = REPO / "outputs" / STUDY.name


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csv_write(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def statistics(values):
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--verify-field-hashes", action="store_true")
    args = parser.parse_args()
    plan_path = STUDY / "MANIFEST.json"
    plan = json.loads(plan_path.read_text())
    protocol = plan["protocol"]
    assert digest(STUDY / "run_forward.py") == plan["driver_sha256"]
    assert digest(STUDY / "protocol.py") == plan["protocol_source_sha256"]
    assert canonical(protocol) == plan["protocol_sha256"]
    input_manifest_path = REPO / "outputs" / protocol["upstream_study"] / "INPUT_MANIFEST.json"
    assert digest(input_manifest_path) == plan["input_manifest_sha256"]
    input_manifest = json.loads(input_manifest_path.read_text())
    input_hashes = {row["id"]: row["file_sha256"] for row in input_manifest["inputs"]}
    patterns = [row["id"] for row in protocol["initial_conditions"]]
    horizons = protocol["horizons"]
    records, missing, result_hashes, rows = {}, [], {}, []
    primary_status, primary_reasons, refined_status = Counter(), Counter(), Counter()
    primary_by_kind, support_by_kind = defaultdict(Counter), defaultdict(Counter)
    temporal_rows, native_verification, source_fit_ids = [], [], set()
    failure_details, departure_details = [], []
    for unit in plan["units"]:
        path = OUTPUT / "runs" / unit["id"] / "result.json"
        if not path.exists():
            missing.append(unit["id"])
            continue
        record = json.loads(path.read_text())
        identity = record["identity"]
        assert identity["unit"] == unit, unit["id"]
        assert identity["protocol_sha256"] == plan["protocol_sha256"]
        assert identity["driver_sha256"] == plan["driver_sha256"]
        input_id = unit.get("input_id", f"{unit['case']}_{unit['setting']}_s{unit['data_seed']}")
        assert identity["input_file_sha256"] == input_hashes[input_id]
        assert canonical(record["metadata"]) == identity["model_metadata_sha256"]
        assert record["status"] == "evaluated"
        assert [entry["pattern"] for entry in record["entries"]] == patterns
        assert sum(entry["info"]["status"] != "complete" for entry in record["entries"]) == record["numerical_failure_patterns"]
        if args.verify_field_hashes:
            assert digest(path.parent / "fields.npz") == record["fields_sha256"]
        result_hashes[unit["id"]] = digest(path)
        records[unit["id"]] = record
        if unit["kind"] == "baseline":
            source_fit_ids.add((unit["input_id"], unit["method"]))
        if unit["kind"] == "model":
            verify = record["metadata"]["model_verification"]
            assert verify["evaluation_dtype"] == "float64" and verify["outside_support_probe"]
            native_verification.append(dict(unit_id=unit["id"], **verify))
        for entry in record["entries"]:
            info, audit = entry["info"], entry["info"]["audit"]
            assert info["state_clipping"] is False
            assert info["support_departure_is_failure"] is False
            assert [score["horizon"] for score in entry["scores"]] == horizons
            primary_status[info["status"]] += 1
            primary_by_kind[unit["kind"]][info["status"]] += 1
            if info["status"] != "complete":
                reason = audit["guard"]["reason"] if audit["guard"] else info["reason"]
                primary_reasons[reason] += 1
                failure_details.append(dict(unit_id=unit["id"], kind=unit["kind"], pattern=entry["pattern"],
                    numerical_status=info["status"], reached_time=info["reached_time"], reason=reason, guard=audit["guard"]))
            if any(audit[key] is not None for key in ("first_accepted_outside", "first_trial_outside", "first_accepted_negative", "first_trial_negative")):
                departure_details.append(dict(unit_id=unit["id"], kind=unit["kind"], pattern=entry["pattern"],
                    numerical_status=info["status"], **{key:audit[key] for key in ("first_accepted_outside", "first_trial_outside", "first_accepted_negative", "first_trial_negative")}))
            support_by_kind[unit["kind"]]["attempted"] += 1
            support_by_kind[unit["kind"]]["accepted_departures"] += audit["first_accepted_outside"] is not None
            support_by_kind[unit["kind"]]["trial_departures"] += audit["first_trial_outside"] is not None
            support_by_kind[unit["kind"]]["accepted_negative"] += audit["first_accepted_negative"] is not None
            support_by_kind[unit["kind"]]["trial_negative"] += audit["first_trial_negative"] is not None
            comparisons = {q["horizon"]: q for q in entry.get("temporal_comparison", [])}
            if unit["refinement"]:
                assert "refined_info" in entry and list(comparisons) == horizons
                refined_status[entry["refined_info"]["status"]] += 1
                for comparison in comparisons.values():
                    temporal_rows.append(dict(unit_id=unit["id"], kind="model", case=unit["case"],
                        setting=unit["setting"], data_seed=unit["data_seed"], variant=unit["variant"],
                        epoch=unit["epoch"], method=unit["method"], pattern=entry["pattern"],
                        primary_status=info["status"], refined_status=entry["refined_info"]["status"], **comparison))
            else:
                assert not comparisons and "refined_info" not in entry
            for score in entry["scores"]:
                for metric in ("endpoint_error", "space_time_error"):
                    assert (score[metric] is not None) == score["reached"]
                    if score["reached"]:
                        assert np.isfinite(score[metric]) and score[metric] >= 0
                for kind in ("accepted", "trial"):
                    outside, negative = audit["first_" + kind + "_outside"], audit["first_" + kind + "_negative"]
                    assert score[kind + "_support_valid"] == (score["reached"] and (outside is None or outside["time"] > score["horizon"] + 1e-12))
                    assert score[kind + "_nonnegative"] == (score["reached"] and (negative is None or negative > score["horizon"] + 1e-12))
                comparison = comparisons.get(score["horizon"])
                rows.append(dict(unit_id=unit["id"], kind=unit["kind"], case=unit["case"], setting=unit["setting"],
                    data_seed=unit["data_seed"], variant=unit["variant"], epoch=unit["epoch"], method=unit["method"],
                    pattern=entry["pattern"], numerical_status=info["status"], **score,
                    refined_numerical_status=entry.get("refined_info", {}).get("status"),
                    endpoint_temporal_discrepancy=comparison["difference_relative_to_truth"] if comparison else None,
                    temporal_tolerance_passed=comparison["temporal_tolerance_passed"] if comparison else None))
    if args.require_complete and missing:
        raise RuntimeError(f"Still missing {len(missing)} declared forward states; complete export refused")
    truth_status, truth_comparisons = Counter(), []
    truth_hashes = {}
    for case in protocol["cases"]:
        for seed in protocol["data_seeds"]:
            path = OUTPUT / "truth" / f"{case}_s{seed}" / "result.json"
            truth = json.loads(path.read_text())
            assert truth["identity"]["case"] == case and truth["identity"]["data_seed"] == seed
            assert truth["identity"]["protocol_sha256"] == plan["protocol_sha256"]
            assert truth["identity"]["driver_sha256"] == plan["driver_sha256"]
            assert [entry["pattern"] for entry in truth["entries"]] == patterns
            if args.verify_field_hashes:
                assert digest(path.parent / "fields.npz") == truth["fields_sha256"]
            truth_hashes[f"{case}_s{seed}"] = digest(path)
            for entry in truth["entries"]:
                for label in ("primary", "refined"):
                    truth_status[label + "_" + entry[label]["status"]] += 1
                for comparison in entry["temporal_comparison"]:
                    truth_comparisons.append(dict(case=case, data_seed=seed, pattern=entry["pattern"], **comparison))
    # A checkpoint is reused by native and frozen representations, never an
    # additional training replicate. Their model metadata must be identical.
    checkpoint_pairs = 0
    for unit_id, record in records.items():
        unit = record["identity"]["unit"]
        if unit["kind"] == "model" and unit["method"] == "native":
            partner = records.get(unit_id.removesuffix("__native") + "__frozen")
            if partner:
                assert record["metadata"] == partner["metadata"]
                assert record["identity"]["input_file_sha256"] == partner["identity"]["input_file_sha256"]
                checkpoint_pairs += 1
    # Contrasts are deterministic complete-case paired differences. Their
    # denominators remain explicit; they do not turn ICs/checkpoints into seeds.
    keyed = {(row["case"], row["setting"], row["data_seed"], row["variant"], row["epoch"], row["method"], row["pattern"], row["horizon"]): row for row in rows}
    pairs = []
    for b in rows:
        key = [b[field] for field in ("case", "setting", "data_seed", "variant", "epoch", "method", "pattern", "horizon")]
        contrasts = []
        if b["kind"] == "model":
            if b["method"] == "frozen":
                akey = key.copy(); akey[5] = "native"
                contrasts.append(("frozen_minus_native", akey))
            if b["epoch"] == 1500:
                akey = key.copy(); akey[4] = 150
                contrasts.append(("long_minus_short", akey))
            if b["variant"] != "default":
                akey = key.copy(); akey[3] = "default"
                contrasts.append((b["variant"] + "_minus_default", akey))
        elif b["kind"] == "global_reference" and b["variant"] == "drop_strong":
            akey = key.copy(); akey[3] = "default"
            contrasts.append(("global_drop_strong_minus_default", akey))
        for contrast, akey in contrasts:
            a = keyed.get(tuple(akey))
            if a is None:
                continue
            completed = a["reached"] and b["reached"]
            discrepancy_available = a["endpoint_temporal_discrepancy"] is not None and b["endpoint_temporal_discrepancy"] is not None
            temporal_pass = discrepancy_available and a["temporal_tolerance_passed"] and b["temporal_tolerance_passed"] and all(
                row["numerical_status"] == "complete" and row["refined_numerical_status"] == "complete" for row in (a, b))
            gap = b["endpoint_error"] - a["endpoint_error"] if completed else None
            margin = a["endpoint_temporal_discrepancy"] + b["endpoint_temporal_discrepancy"] if discrepancy_available else None
            resolved = completed and temporal_pass and abs(gap) > margin
            pairs.append(dict(contrast=contrast, case=b["case"], setting=b["setting"], data_seed=b["data_seed"],
                variant=b["variant"], epoch=b["epoch"], method=b["method"], pattern=b["pattern"], horizon=b["horizon"],
                first_unit_id=a["unit_id"], second_unit_id=b["unit_id"], both_reached=completed,
                both_accepted_support_valid=a["accepted_support_valid"] and b["accepted_support_valid"],
                both_accepted_nonnegative=a["accepted_nonnegative"] and b["accepted_nonnegative"],
                endpoint_error_second_minus_first=gap,
                space_time_error_second_minus_first=b["space_time_error"] - a["space_time_error"] if completed else None,
                endpoint_temporal_margin=margin,
                endpoint_gap_status=("resolved_on_refined_seed_subset" if resolved else "unresolved_refinement_or_gap" if discrepancy_available else "no_pairwise_temporal_audit")))
    group_fields = ("case", "setting", "variant", "epoch", "method", "pattern", "horizon")
    grouped = defaultdict(list)
    for unit in plan["units"]:
        for pattern in patterns:
            for horizon in horizons:
                grouped[tuple(unit[key] for key in group_fields[:-2]) + (pattern, horizon)]
    for row in rows:
        grouped[tuple(row[key] for key in group_fields)].append(row)
    groups = []
    for key, group in grouped.items():
        assert len({row["data_seed"] for row in group}) == len(group)
        groups.append(dict(zip(group_fields, key), planned_seeds=5, evaluated_seeds=len(group),
            completed_seeds=sum(row["reached"] for row in group),
            accepted_support_valid_seeds=sum(row["accepted_support_valid"] for row in group),
            trial_support_valid_seeds=sum(row["trial_support_valid"] for row in group),
            accepted_nonnegative_seeds=sum(row["accepted_nonnegative"] for row in group),
            trial_nonnegative_seeds=sum(row["trial_nonnegative"] for row in group),
            endpoint_error=statistics([row["endpoint_error"] for row in group if row["reached"]]),
            space_time_error=statistics([row["space_time_error"] for row in group if row["reached"]])))
    pair_fields = ("contrast",) + group_fields
    grouped_pairs = defaultdict(list)
    for unit in plan["units"]:
        contrasts = []
        if unit["kind"] == "model":
            if unit["method"] == "frozen":
                contrasts.append("frozen_minus_native")
            if unit["epoch"] == 1500:
                contrasts.append("long_minus_short")
            if unit["variant"] != "default":
                contrasts.append(unit["variant"] + "_minus_default")
        elif unit["kind"] == "global_reference" and unit["variant"] == "drop_strong":
            contrasts.append("global_drop_strong_minus_default")
        for contrast in contrasts:
            for pattern in patterns:
                for horizon in horizons:
                    grouped_pairs[(contrast,) + tuple(unit[key] for key in group_fields[:-2]) + (pattern, horizon)]
    for row in pairs:
        grouped_pairs[tuple(row[key] for key in pair_fields)].append(row)
    pair_groups = []
    for key, group in grouped_pairs.items():
        assert len({row["data_seed"] for row in group}) == len(group)
        pair_groups.append(dict(zip(pair_fields, key), planned_seed_pairs=5, evaluated_seed_pairs=len(group),
            completed_seed_pairs=sum(row["both_reached"] for row in group),
            accepted_support_valid_seed_pairs=sum(row["both_accepted_support_valid"] for row in group),
            endpoint_pair_difference=statistics([row["endpoint_error_second_minus_first"] for row in group if row["both_reached"]]),
            space_time_pair_difference=statistics([row["space_time_error_second_minus_first"] for row in group if row["both_reached"]]),
            endpoint_temporal_gap_status_counts=dict(Counter(row["endpoint_gap_status"] for row in group))))
    discrepancies = [row["difference_relative_to_truth"] for row in temporal_rows if row["both_reached"]]
    truth_discrepancies = [row["difference_relative_to_truth"] for row in truth_comparisons if row["both_reached"]]
    verified = {branch: max((row["maximum_absolute_difference"][branch] for row in native_verification), default=None) for branch in ("D", "R")}
    summary = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), complete=not missing,
        manifest_sha256=digest(plan_path), protocol_sha256=plan["protocol_sha256"], analysis_source_sha256=digest(__file__),
        field_hashes_verified=args.verify_field_hashes, planned_states=500, evaluated_states=len(records),
        evaluated_states_by_kind=dict(Counter(record["identity"]["unit"]["kind"] for record in records.values())),
        missing_units=missing, planned_integrations=2208,
        evaluated_integrations=sum(primary_status.values()) + sum(refined_status.values()) + sum(truth_status.values()),
        primary_integrations=dict(primary_status), primary_by_kind={key:dict(value) for key,value in primary_by_kind.items()},
        primary_failure_reasons=dict(primary_reasons), support_by_kind={key:dict(value) for key,value in support_by_kind.items()},
        primary_failure_details=failure_details, support_departure_details=departure_details,
        model_refinement_integrations=dict(refined_status), truth_integrations=dict(truth_status),
        truth_temporal_comparisons=dict(planned=120, evaluated=len(truth_comparisons), passed=sum(row["temporal_tolerance_passed"] for row in truth_comparisons), maximum_relative_discrepancy=max(truth_discrepancies, default=None)),
        model_temporal_comparisons=dict(planned=384, evaluated=len(temporal_rows), both_reached=sum(row["both_reached"] for row in temporal_rows), passed=sum(row["temporal_tolerance_passed"] for row in temporal_rows), maximum_relative_discrepancy=max(discrepancies, default=None)),
        native_vs_torch_maximum_absolute_difference=verified, paired_checkpoint_metadata_checks=checkpoint_pairs,
        direct_reference_fits=dict(planned=60, reused_existing_fits=len(source_fit_ids), new_fits_performed_by_this_analysis=0),
        groups=groups, paired_contrasts=pair_groups,
        interpretation=["Five data seeds are the replication unit; ICs, epochs and representations are paired reuse.",
            "Error means and SDs describe reached-horizon subsets, with planned/evaluated/completed/support-valid denominators.",
            "Endpoint gap resolution is assessed only where both saved refinement comparisons exist and pass; no full-cohort temporal certificate is inferred from seed zero.",
            "Space-time errors use the fixed .01 trapezoidal grid; endpoint refinement checks do not certify quadrature resolution.",
            "Support departure is recorded extrapolation and does not stop the solve; a trial/guard failure is not proof of physical blow-up.",
            "The common nx64 spatial formulation is not a continuum spatial-resolution certificate."])
    reports = OUTPUT / "reports"
    reports.mkdir(exist_ok=True)
    csv_write(reports / "forward_audited_by_seed.csv", rows)
    csv_write(reports / "forward_paired_by_seed.csv", pairs)
    csv_write(reports / "forward_temporal_by_seed.csv", temporal_rows)
    write(reports / "FORWARD_AUDIT.json", summary)
    write(reports / "FORWARD_AUDIT_SOURCES.json", dict(snapshot_utc=summary["snapshot_utc"],
        manifest_sha256=summary["manifest_sha256"], analysis_source_sha256=summary["analysis_source_sha256"],
        result_sha256=result_hashes, truth_result_sha256=truth_hashes))
    print(json.dumps({key:summary[key] for key in ("complete", "evaluated_states", "evaluated_states_by_kind", "evaluated_integrations", "primary_integrations", "primary_failure_reasons", "model_temporal_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
