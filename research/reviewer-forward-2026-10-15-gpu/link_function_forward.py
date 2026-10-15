"""Link actual-weight function/BIR errors to paired saved forward results.

Exploratory postprocessing requested after the forward design was frozen. It
evaluates existing native exports and exact frozen expressions on the common
256-point interval. It never fits, trains, integrates, or selects a model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import shutil

import run_forward as forward
import lineage_checks
import numpy as np

FUNCTION_DIRECTION_TOLERANCE = 1e-10
FORWARD_DIRECTION_TOLERANCE = 1e-12


def direction(delta, tolerance):
    return "worse" if delta > tolerance else "better" if delta < -tolerance else "tied_within_reporting_tolerance"


def csv_write(path, rows):
    if rows:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        temporary.replace(path)


def physical_category(info):
    if info["status"] == "complete":
        return "completed"
    guard = info["audit"]["guard"]
    reason = guard["reason"] if guard else info["reason"]
    if reason in ("nonpositive_diffusion_at_queried_state", "nonremovable_pole_in_field_enclosure"):
        return "initial_constitutive_admissibility_guard" if guard["time"] == 0 else "encountered_constitutive_admissibility_guard"
    return "numerical_solver_finiteness_or_safeguard_stop"




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    plan, core = forward.read_plan(), forward.load_core()
    from reliability_core import NumpyMLPClosure
    records, result_hashes, pending_states = {}, {}, []
    for unit in plan["units"]:
        path = forward.OUTPUT / "runs" / unit["id"] / "result.json"
        if not path.exists():
            pending_states.append(unit["id"])
            continue
        record = json.loads(path.read_text())
        assert record["identity"]["unit"] == unit
        lineage_checks.verify_record(record, path, plan, unit=unit)
        assert forward.canonical(record["metadata"]) == record["identity"]["model_metadata_sha256"]
        records[unit["id"]], result_hashes[unit["id"]] = record, forward.digest(path)
    physical, physical_totals = [], Counter()
    for identifier, record in records.items():
        unit = record["identity"]["unit"]
        for entry in record["entries"]:
            info, audit = entry["info"], entry["info"]["audit"]
            category = physical_category(info)
            physical_totals[category] += 1
            physical.append(dict(unit_id=identifier, kind=unit["kind"], case=unit["case"], setting=unit["setting"],
                data_seed=unit["data_seed"], variant=unit["variant"], epoch=unit["epoch"], method=unit["method"],
                pattern=entry["pattern"], category=category, numerical_status=info["status"],
                guard_reason=audit["guard"]["reason"] if audit["guard"] else info["reason"],
                guard_time=audit["guard"]["time"] if audit["guard"] else None,
                accepted_support_departure=audit["first_accepted_outside"] is not None,
                trial_support_departure=audit["first_trial_outside"] is not None,
                accepted_negative=audit["first_accepted_negative"] is not None,
                trial_negative=audit["first_trial_negative"] is not None))
    paired_physical = []
    for unit in plan["units"]:
        if unit["kind"] != "model" or unit["method"] != "native":
            continue
        first, second = unit["id"], unit["id"].removesuffix("__native") + "__frozen"
        if first not in records or second not in records:
            continue
        for a,b in zip(records[first]["entries"], records[second]["entries"]):
            aa, ba = a["info"]["audit"], b["info"]["audit"]
            paired_physical.append(dict(training_id=unit["training_id"], epoch=unit["epoch"],
                case=unit["case"], setting=unit["setting"], data_seed=unit["data_seed"], variant=unit["variant"], pattern=a["pattern"],
                native_category=physical_category(a["info"]), frozen_category=physical_category(b["info"]),
                native_status=a["info"]["status"], frozen_status=b["info"]["status"],
                native_accepted_support_departure=aa["first_accepted_outside"] is not None,
                frozen_accepted_support_departure=ba["first_accepted_outside"] is not None,
                native_accepted_negative=aa["first_accepted_negative"] is not None,
                frozen_accepted_negative=ba["first_accepted_negative"] is not None,
                native_first_accepted_support_departure=aa["first_accepted_outside"]["time"] if aa["first_accepted_outside"] else None,
                frozen_first_accepted_support_departure=ba["first_accepted_outside"]["time"] if ba["first_accepted_outside"] else None))
    # Optional common-norm resolution results are independently derived from
    # the committed arrays. Endpoint resolution can also be checked directly
    # from each original result's frozen temporal metadata.
    gap_path = forward.OUTPUT / "reports/forward_gap_resolution_by_seed.csv"
    gap_index = {}
    if gap_path.exists():
        with gap_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["contrast"] == "frozen_minus_native" and row["norm"] == "reported_relative_or_fallback":
                    gap_index[(row["first_unit_id"], row["second_unit_id"], row["pattern"], float(row["horizon"]), row["metric"])] = row
    function_rows, prediction_rows, pending_pairs, export_sources = [], [], [], {}
    input_cache = {}
    for unit in plan["units"]:
        if unit["kind"] != "model" or unit["method"] != "native":
            continue
        native_id, frozen_id = unit["id"], unit["id"].removesuffix("__native") + "__frozen"
        identifier = f"{unit['training_id']}_e{unit['epoch']}"
        npz = forward.OUTPUT / "native_exports" / (identifier + ".npz")
        sidecar_path = npz.with_suffix(".json")
        if native_id not in records or frozen_id not in records or not sidecar_path.exists():
            pending_pairs.append(identifier)
            continue
        native_record, frozen_record = records[native_id], records[frozen_id]
        assert native_record["metadata"] == frozen_record["metadata"]
        assert native_record["identity"]["input_file_sha256"] == frozen_record["identity"]["input_file_sha256"]
        sidecar = json.loads(sidecar_path.read_text())
        assert forward.digest(npz) == sidecar["export_sha256"]
        assert sidecar["identity"]["checkpoint_sha256"] == native_record["metadata"]["checkpoint_sha256"]
        assert sidecar["identity"]["model_metadata_sha256"] == native_record["identity"]["model_metadata_sha256"]
        assert sidecar["identity"]["input_file_sha256"] == native_record["identity"]["input_file_sha256"]
        assert sidecar["frozen_expressions"] == native_record["metadata"]["frozen_expressions"]
        export_sources[identifier] = dict(npz_sha256=sidecar["export_sha256"], sidecar_sha256=forward.digest(sidecar_path))
        input_id = f"{unit['case']}_{unit['setting']}_s{unit['data_seed']}"
        if input_id not in input_cache:
            _, interval, metadata = forward.load_input(core, input_id)
            input_cache[input_id] = interval, metadata
        interval, input_metadata = input_cache[input_id]
        assert input_metadata["file_sha256"] == native_record["identity"]["input_file_sha256"]
        assert input_metadata["observation_signature"] == native_record["metadata"]["observation_signature"]
        native = NumpyMLPClosure.load(npz)
        expressions = [core.SymbolicExpression(**dict(row, parameters=tuple(row["parameters"]))) for row in sidecar["frozen_expressions"]]
        case, grid = core.extra_case(unit["case"]), np.linspace(*interval, 256)
        function = dict(pair_id=identifier, training_id=unit["training_id"], case=unit["case"], setting=unit["setting"],
            data_seed=unit["data_seed"], variant=unit["variant"], epoch=unit["epoch"],
            input_file_sha256=input_metadata["file_sha256"])
        for label, truth_law, native_law, expression in zip(("D", "R"), (case.diffusion, case.reaction),
                (native.diffusion, native.reaction), expressions):
            true, a, b = truth_law(grid), native_law(grid), expression.evaluate(grid)
            denominator = float(np.linalg.norm(true))
            assert denominator > 0 and np.isfinite(a).all() and np.isfinite(b).all()
            en, es = float(np.linalg.norm(a - true) / (denominator+1e-12)), float(np.linalg.norm(b - true) / (denominator+1e-12))
            function.update({"native_error_" + label:en, "frozen_error_" + label:es,
                "frozen_minus_native_" + label:es - en, "BIR_" + label:es / en if en > 0 else None,
                "direction_" + label:direction(es - en, FUNCTION_DIRECTION_TOLERANCE)})
        function["native_R0"] = float(native.reaction(np.array([0.]))[0])
        function["frozen_R0"] = float(expressions[1].evaluate(np.array([0.]))[0])
        function["both_function_errors_lower"] = all(function["direction_" + label] == "better" for label in ("D", "R"))
        function["both_function_errors_higher"] = all(function["direction_" + label] == "worse" for label in ("D", "R"))
        function_rows.append(function)
        for a, b in zip(native_record["entries"], frozen_record["entries"]):
            assert a["pattern"] == b["pattern"]
            at = {row["horizon"]:row for row in a.get("temporal_comparison", [])}
            bt = {row["horizon"]:row for row in b.get("temporal_comparison", [])}
            for sa, sb in zip(a["scores"], b["scores"]):
                assert sa["horizon"] == sb["horizon"]
                for metric, field in (("endpoint", "endpoint_error"), ("space_time", "space_time_error")):
                    reached = sa["reached"] and sb["reached"]
                    delta = sb[field] - sa[field] if reached else None
                    resolution = gap_index.get((native_id, frozen_id, a["pattern"], sa["horizon"], metric), {}).get("status", "no_pairwise_temporal_audit")
                    if metric == "endpoint" and sa["horizon"] in at and sa["horizon"] in bt:
                        ca, cb = at[sa["horizon"]], bt[sa["horizon"]]
                        full = all(entry["info"]["status"] == "complete" and entry.get("refined_info", {}).get("status") == "complete" for entry in (a,b))
                        if reached and full and ca["temporal_tolerance_passed"] and cb["temporal_tolerance_passed"]:
                            margin = ca["difference_relative_to_truth"] + cb["difference_relative_to_truth"]
                            resolution = "resolved_on_declared_refined_seed_subset" if abs(delta) > margin else "numerically_unresolved_gap"
                        else:
                            resolution = "unresolved_refinement_failure"
                    prediction_direction = direction(delta, FORWARD_DIRECTION_TOLERANCE) if reached else "unreached_pair"
                    prediction_rows.append(dict(**function, pattern=a["pattern"], horizon=sa["horizon"], metric=metric,
                        native_numerical_status=a["info"]["status"], frozen_numerical_status=b["info"]["status"],
                        both_reached=reached, native_prediction_error=sa[field], frozen_prediction_error=sb[field],
                        frozen_minus_native_prediction_error=delta, prediction_direction=prediction_direction,
                        temporal_resolution=resolution,
                        joint_function_improvement_forward_worsening=function["both_function_errors_lower"] and prediction_direction == "worse",
                        joint_function_worsening_forward_improvement=function["both_function_errors_higher"] and prediction_direction == "better",
                        native_accepted_support_valid=sa["accepted_support_valid"], frozen_accepted_support_valid=sb["accepted_support_valid"],
                        native_accepted_nonnegative=sa["accepted_nonnegative"], frozen_accepted_nonnegative=sb["accepted_nonnegative"]))
    complete = not pending_states and not pending_pairs
    reports = forward.OUTPUT / "reports"
    reports.mkdir(exist_ok=True)
    csv_write(reports / "function_errors_native_frozen_by_checkpoint.csv", function_rows)
    csv_write(reports / "function_forward_links_by_seed.csv", prediction_rows)
    csv_write(reports / "physical_numerical_status_by_trajectory.csv", physical)
    csv_write(reports / "native_frozen_physical_by_trajectory.csv", paired_physical)
    counterexamples = [row for row in prediction_rows if row["joint_function_improvement_forward_worsening"] or row["joint_function_worsening_forward_improvement"]]
    cohorts = defaultdict(list)
    for row in prediction_rows:
        cohorts[(row["case"], row["setting"], row["variant"], row["epoch"], row["metric"])].append(row)
    cohort_rows = []
    for key, group in cohorts.items():
        cohort_rows.append(dict(zip(("case", "setting", "variant", "epoch", "metric"), key),
            planned_data_seeds=5, evaluated_data_seeds=len({row["data_seed"] for row in group}),
            planned_IC_horizon_pairs=60, evaluated_IC_horizon_pairs=len(group),
            joint_function_improvement_forward_worsening=sum(row["joint_function_improvement_forward_worsening"] for row in group),
            resolved_joint_function_improvement_forward_worsening=sum(row["joint_function_improvement_forward_worsening"] and row["temporal_resolution"] == "resolved_on_declared_refined_seed_subset" for row in group),
            joint_function_worsening_forward_improvement=sum(row["joint_function_worsening_forward_improvement"] for row in group)))
    report = dict(cohort="fresh_GPU_only", function_norm_epsilon=1e-12, lineage_checks_sha256=forward.digest(forward.STUDY/"lineage_checks.py"), snapshot_utc=datetime.now(timezone.utc).isoformat(), complete=complete,
        selected_counterexample=None,
        planned_states=500, evaluated_states=len(records), planned_unique_checkpoint_pairs=200,
        evaluated_unique_checkpoint_pairs=len(function_rows), planned_metric_IC_horizon_pairs=4800,
        evaluated_metric_IC_horizon_pairs=len(prediction_rows), pending_states=pending_states, pending_pairs=pending_pairs,
        function_direction_absolute_tolerance=FUNCTION_DIRECTION_TOLERANCE,
        forward_direction_absolute_tolerance=FORWARD_DIRECTION_TOLERANCE,
        function_grid_points=256, evaluated_trajectory_status_categories=dict(physical_totals),
        sampled_physical_observations=dict(accepted_support_departures=sum(row["accepted_support_departure"] for row in physical),
            accepted_negative=sum(row["accepted_negative"] for row in physical),
            paired_native_frozen_trajectories=len(paired_physical), planned_paired_native_frozen_trajectories=800,
            native_completes_frozen_does_not=sum(row["native_status"] == "complete" and row["frozen_status"] != "complete" for row in paired_physical),
            native_stays_in_support_frozen_leaves=sum(not row["native_accepted_support_departure"] and row["frozen_accepted_support_departure"] for row in paired_physical),
            native_nonnegative_frozen_negative=sum(not row["native_accepted_negative"] and row["frozen_accepted_negative"] for row in paired_physical)),
        sampled_physical_observations_by_kind={kind:dict(evaluated_trajectories=sum(row["kind"] == kind for row in physical),
            accepted_support_departures=sum(row["kind"] == kind and row["accepted_support_departure"] for row in physical),
            accepted_negative=sum(row["kind"] == kind and row["accepted_negative"] for row in physical)) for kind in ("model", "baseline", "global_reference")},
        counterexample_counts=dict(joint_function_improvement_forward_worsening=sum(row["joint_function_improvement_forward_worsening"] for row in prediction_rows),
            resolved_joint_function_improvement_forward_worsening=sum(row["joint_function_improvement_forward_worsening"] and row["temporal_resolution"] == "resolved_on_declared_refined_seed_subset" for row in prediction_rows),
            joint_function_worsening_forward_improvement=sum(row["joint_function_worsening_forward_improvement"] for row in prediction_rows)),
        counterexamples=counterexamples, cohorts=cohort_rows,
        source_manifest_sha256=forward.digest(forward.STUDY / "MANIFEST.json"), script_sha256=forward.digest(__file__),
        source_result_sha256=result_hashes, source_exports=export_sources,
        source_gap_resolution_sha256=forward.digest(gap_path) if gap_path.exists() else None,
        fits_or_integrations_performed=0,
        interpretation=["This is an exploratory postprocessing link, not a pre-registered endpoint or a model-selection rule.",
            "Function errors compare the actual float64 native formulas and exact frozen expressions on the same256-point equal-weight U grid, using norm(true)+1e-12; BIR is frozen/native true function error and is undefined for a zero teacher error.",
            "A joint function-improvement counterexample requires lower D and R function errors, yet a larger paired forward error at the same IC/horizon. Individual branch reversals are retained but not promoted to this stronger category.",
            "Counts of IC/horizon/metric comparisons reuse checkpoints and five data seeds, not independent scientific replicates.",
            "Joint function-improvement claims are norm-specific until the comprehensive quadrature/grid sensitivity audit confirms their scope.",
            "Only explicitly resolved comparisons have the declared temporal refinement evidence. No new spatial or continuous-time certificate is inferred.",
            "Initial negative-diffusion/pole guards do not complete a forward trajectory. Sampled negative states are reported physical violations; support departures are extrapolation; solver/finiteness/safeguard stops are not proof of physical blow-up."])
    forward.write(reports / "FUNCTION_FORWARD_LINK.json", report)
    print(json.dumps({key:report[key] for key in ("complete", "evaluated_states", "evaluated_unique_checkpoint_pairs", "counterexample_counts", "sampled_physical_observations", "evaluated_trajectory_status_categories")}, indent=2))
    if args.require_complete and not complete:
        raise RuntimeError("Full function/forward link export is still pending")


if __name__ == "__main__":
    main()
