"""Audit saved forward arrays; never fit, train, or integrate a PDE.

Pending snapshots are allowed. --require-complete writes the audit and then exits
with status 2 if any of the 500 planned states or 168 refinement trajectories is
pending. Temporal comparisons concern the saved nx64 formulation only. Comparing
the existing .02 and .01 sampling grids measures trapezoidal quadrature
sensitivity; it supplies no .005 sampling or spatial-continuum certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np
import lineage_checks

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
OUTPUT = REPO / "outputs" / STUDY.name


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path, rows, empty_fieldnames=None):
    check(bool(rows) or bool(empty_fieldnames), f"No CSV schema for {path}")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class Sources:
    def __init__(self):
        self.rows = {}

    def read(self, path, role):
        path = Path(path).resolve()
        raw = path.read_bytes()
        row = dict(path=str(path), role=role, sha256=sha(raw), bytes=len(raw))
        if str(path) in self.rows:
            check(self.rows[str(path)]["sha256"] == row["sha256"], f"Source changed during audit: {path}")
        self.rows[str(path)] = row
        return raw

    def json(self, path, role):
        return json.loads(self.read(path, role))

    def arrays(self, path, role, expected_sha):
        raw = self.read(path, role)
        check(sha(raw) == expected_sha, f"NPZ hash mismatch: {path}")
        with np.load(io.BytesIO(raw), allow_pickle=False) as stored:
            return {key: stored[key].copy() for key in stored.files}


def initial_fields(protocol, interval, seed):
    """Reconstruct only the declared initial data, using the frozen formula."""
    x = np.arange(protocol["nx"]) / protocol["nx"]
    phase = np.random.default_rng(seed + 20261015).uniform(0, 2 * np.pi, 2)
    shape = np.sin(2 * np.pi * x + phase[0]) + .25 * np.sin(6 * np.pi * x + phase[1])
    shape /= np.max(np.abs(shape))
    width = interval[1] - interval[0]
    return {row["id"]: interval[0] + row["mean_fraction"] * width
            + row["half_amplitude_fraction"] * width * shape
            for row in protocol["initial_conditions"]}


def validate_trajectory(times, fields, initial, grid, nx, status, name):
    check(times.ndim == 1 and len(times) > 0, f"Invalid time array: {name}")
    check(fields.shape == (len(times), nx), f"Invalid field shape: {name}")
    check(np.isfinite(times).all() and np.isfinite(fields).all(), f"Nonfinite saved array: {name}")
    check(len(times) <= len(grid) and np.array_equal(times, grid[:len(times)]), f"Time-grid mismatch: {name}")
    check(np.array_equal(fields[0], initial), f"Initial condition mismatch: {name}")
    check(status in {"complete", "guard_failed", "solver_failed", "arithmetic_failed"}, f"Unknown solve status: {name}")
    if status == "complete":
        check(len(times) == len(grid), f"Complete solve lacks saved times: {name}")


def metrics(times, prediction, truth, horizon, stride=1):
    """Absolute RMS and the original normalization, from identical saved fields."""
    n = int(round(horizon / (times[1] - times[0])))
    check(len(times) > n and math.isclose(float(times[n]), horizon, abs_tol=1e-12), "Unreached metric horizon")
    indexes = np.arange(0, n + 1, stride)
    check(indexes[-1] == n, "Quadrature grid must contain the horizon")
    t = times[indexes]
    error = prediction[indexes] - truth[indexes]
    true = truth[indexes]
    endpoint_absolute = float(np.sqrt(np.mean(error[-1] ** 2)))
    endpoint_truth = float(np.sqrt(np.mean(true[-1] ** 2)))
    numerator = float(np.trapezoid(np.mean(error ** 2, axis=-1), t))
    denominator = float(np.trapezoid(np.mean(true ** 2, axis=-1), t))
    st_absolute = math.sqrt(numerator / horizon)
    st_truth = math.sqrt(denominator / horizon)
    endpoint_relative = endpoint_truth > 1e-12
    st_relative = denominator > horizon * 1e-24
    return dict(endpoint_absolute_rms=endpoint_absolute,
                space_time_absolute_rms=st_absolute,
                truth_endpoint_rms=endpoint_truth,
                truth_space_time_rms=st_truth,
                endpoint_relative_l2=endpoint_absolute / endpoint_truth if endpoint_relative else None,
                space_time_relative_l2=math.sqrt(numerator / denominator) if st_relative else None,
                endpoint_error=endpoint_absolute / (endpoint_truth if endpoint_relative else 1.),
                space_time_error=math.sqrt(numerator / (denominator if st_relative else horizon)),
                endpoint_normalization="relative_L2" if endpoint_relative else "absolute_RMS",
                space_time_normalization="relative_space_time_L2" if st_relative else "absolute_space_time_RMS")


def temporal_row(identity, pattern, horizon, primary=None, refined=None, truth=None, comparison=None):
    row = dict(**identity, pattern=pattern, horizon=horizon, snapshot_status="pending",
               primary_status=None, refined_status=None, both_reached=False,
               primary_saved_points=0, refined_saved_points=0, common_saved_points_through_horizon=0,
               endpoint_absolute_rms_difference=None, endpoint_difference_relative_to_truth=None,
               max_absolute_rms_difference_over_saved_grid=None,
               max_difference_relative_to_truth_over_saved_grid=None,
               space_time_absolute_rms_difference=None, space_time_difference_relative_to_truth=None,
               endpoint_temporal_tolerance_passed=None, failed_comparison_unresolved=None)
    if primary is None:
        return row
    t, a, pstatus = primary
    rt, b, rstatus = refined
    tt, true = truth
    row.update(snapshot_status="evaluated", primary_status=pstatus, refined_status=rstatus,
               primary_saved_points=len(t), refined_saved_points=len(rt))
    stop = int(round(horizon / .01))
    common = min(len(t), len(rt), len(tt), stop + 1)
    check(np.array_equal(t[:common], rt[:common]) and np.array_equal(t[:common], tt[:common]), "Temporal grids differ")
    delta = np.sqrt(np.mean((a[:common] - b[:common]) ** 2, axis=-1))
    scale = np.maximum(np.sqrt(np.mean(true[:common] ** 2, axis=-1)), 1e-12)
    row.update(common_saved_points_through_horizon=common,
               max_absolute_rms_difference_over_saved_grid=float(np.max(delta)),
               max_difference_relative_to_truth_over_saved_grid=float(np.max(delta / scale)),
               failed_comparison_unresolved=pstatus != "complete" or rstatus != "complete")
    reached = common == stop + 1
    row["both_reached"] = reached
    check(comparison["both_reached"] == reached, "Stored temporal reached flag mismatch")
    if reached:
        rms = float(delta[-1])
        relative = float(delta[-1] / scale[-1])
        st_num = float(np.trapezoid(np.mean((a[:common] - b[:common]) ** 2, axis=-1), t[:common]))
        st_den = float(np.trapezoid(np.mean(true[:common] ** 2, axis=-1), t[:common]))
        passed = rms <= 1e-7 + 1e-6 * float(scale[-1])
        check(math.isclose(rms, comparison["rms_difference"], rel_tol=2e-13, abs_tol=2e-15), "Stored temporal absolute discrepancy mismatch")
        check(math.isclose(relative, comparison["difference_relative_to_truth"], rel_tol=2e-13, abs_tol=2e-15), "Stored temporal relative discrepancy mismatch")
        check(comparison["temporal_tolerance_passed"] == passed, "Stored temporal tolerance flag mismatch")
        row.update(endpoint_absolute_rms_difference=rms, endpoint_difference_relative_to_truth=relative,
                   space_time_absolute_rms_difference=math.sqrt(st_num / horizon),
                   space_time_difference_relative_to_truth=math.sqrt(st_num / max(st_den, horizon * 1e-24)),
                   endpoint_temporal_tolerance_passed=passed)
    else:
        check(comparison["rms_difference"] is None and comparison["difference_relative_to_truth"] is None,
              "Failed temporal comparison must have null endpoint discrepancies")
        check(comparison["temporal_tolerance_passed"] is False, "Failed temporal comparison cannot pass")
        row["endpoint_temporal_tolerance_passed"] = False
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    started = utc()
    sources = Sources()
    sources.read(__file__, "array_audit_script")
    sources.read(STUDY / "lineage_checks.py", "lineage_check_script")
    plan = sources.json(STUDY / "MANIFEST.json", "frozen_forward_manifest")
    protocol = sources.json(STUDY / "PROTOCOL.json", "frozen_forward_protocol")
    check(protocol == plan["protocol"] and canonical(protocol) == plan["protocol_sha256"], "Protocol/manifest mismatch")
    check(plan["cohort"] == "fresh_GPU_only" and plan["formal_execution_eligible"] is True
          and plan["design_only"] is False, "GPU forward plan is not frozen for formal execution")
    check(sha(sources.read(STUDY / "run_forward.py", "frozen_forward_driver")) == plan["driver_sha256"], "Driver changed")
    check(sha(sources.read(STUDY / "protocol.py", "frozen_protocol_source")) == plan["protocol_source_sha256"], "Protocol source changed")
    check(sha(sources.read(STUDY / "cohort_io.py", "frozen_cohort_io_source")) == plan["cohort_io_sha256"], "Cohort I/O source changed")
    check(sha(sources.read(STUDY / "RESOURCE_POLICY.json", "frozen_resource_policy")) == plan["resource_policy_sha256"], "Resource policy changed")
    lineage_checks.imports(plan, sources)
    training_path = REPO / "outputs" / protocol["training_output_directory"] / "LAUNCHED_MANIFEST.json"
    training_research_path = REPO / "research" / protocol["training_manifest_directory"] / "MANIFEST.json"
    training = sources.json(training_path, "training_launched_manifest")
    training_research = sources.json(training_research_path, "training_research_manifest")
    check(sources.rows[str(training_path.resolve())]["sha256"] == plan["upstream_manifest_sha256"], "Training launch manifest changed")
    check(sources.rows[str(training_research_path.resolve())]["sha256"] == plan["training_research_manifest_sha256"], "Training research manifest changed")
    check(training == training_research and canonical(training) == plan["training_manifest_canonical_sha256"], "Training manifests differ")
    input_root = REPO / "outputs" / protocol["upstream_study"]
    input_path = input_root / "INPUT_MANIFEST.json"
    input_manifest = sources.json(input_path, "observation_input_manifest")
    check(sources.rows[str(input_path.resolve())]["sha256"] == plan["input_manifest_sha256"], "Input manifest changed")
    inputs = {}
    intervals = {}
    for row in input_manifest["inputs"]:
        inputs[row["id"]] = row
        arrays = sources.arrays(input_root / row["file"], "observation_input_arrays", row["file_sha256"])
        interval = arrays["value_range"]
        check(np.array_equal(interval, np.asarray(row["value_range"])), "Input interval metadata mismatch")
        key = (row["case"], row["data_seed"])
        if key in intervals:
            check(np.array_equal(intervals[key], interval), "Noise settings use different intervals")
        intervals[key] = interval
    check(len(inputs) == input_manifest["count"] == 20, "Unexpected observation input count")

    units = {unit["id"]: unit for unit in plan["units"]}
    patterns = [row["id"] for row in protocol["initial_conditions"]]
    horizons = protocol["horizons"]
    dt = protocol["save_dt"]
    check(dt == .01 and protocol["numerical_comparison"]["absolute_rms_tolerance"] == 1e-7
          and protocol["numerical_comparison"]["relative_to_truth_tolerance"] == 1e-6,
          "This audit requires the frozen V2 sampling/tolerances")
    grid = np.linspace(0., max(horizons), int(round(max(horizons) / dt)) + 1)
    check(len(units) == plan["unit_count"] == 500, "Unexpected planned state count")
    check(sum(unit["refinement"] for unit in units.values()) * len(patterns) == plan["model_refinement_integrations"] == 128,
          "Unexpected model refinement count")
    check(plan["primary_model_integrations"] + plan["primary_baseline_integrations"]
          + plan["primary_global_reference_integrations"] == 2000
          and plan["unique_truth_integrations"] == plan["truth_refinement_integrations"] == 40
          and plan["total_declared_integrations"] == 2208, "Manifest integration denominator mismatch")
    check(plan["new_integrations_to_execute"] == 1728 and plan["existing_integration_records_reused"] == 480
          and plan["new_reference_fits"] == 0, "New-versus-imported integration denominator mismatch")
    run_paths = sorted((OUTPUT / "runs").glob("*/result.json"))
    check(not ({path.parent.name for path in run_paths} - set(units)), "Unplanned result directories")
    truth_keys = [(case, seed) for case in protocol["cases"] for seed in protocol["data_seeds"]]
    truth_paths = {key: OUTPUT / "truth" / f"{key[0]}_s{key[1]}" / "result.json" for key in truth_keys}
    # Freeze result contents before inspecting arrays; later atomic publications
    # belong to the next audit, rather than silently changing this snapshot.
    records = {path.parent.name: sources.json(path, "forward_result") for path in run_paths}
    truth_records = {key: sources.json(path, "truth_result") for key, path in truth_paths.items() if path.exists()}
    captured = utc()
    field_paths_at_capture = sorted((OUTPUT / "runs").glob("*/fields.npz")) + sorted((OUTPUT / "truth").glob("*/fields.npz"))
    expected_used_fields = {str((OUTPUT / "runs" / uid / "fields.npz").resolve()) for uid in records}
    expected_used_fields.update(str((path.parent / "fields.npz").resolve()) for key, path in truth_paths.items() if key in truth_records)
    uncommitted_fields = [str(path.resolve()) for path in field_paths_at_capture if str(path.resolve()) not in expected_used_fields]
    truth_arrays = {}
    temporal_rows = []
    counts = Counter()
    status_counts = Counter()
    maxima = dict(endpoint_metric_recomputation_absolute_difference=0., space_time_metric_recomputation_absolute_difference=0.,
                  space_time_absolute_rms_quadrature_change=0., space_time_reported_metric_quadrature_change=0.)
    for key in truth_keys:
        case, seed = key
        identity = dict(kind="truth", unit_id=f"{case}_s{seed}", case=case, setting=None,
                        data_seed=seed, variant=None, epoch=None, method="truth")
        if key not in truth_records:
            for pattern in patterns:
                temporal_rows.extend(temporal_row(identity, pattern, h) for h in horizons)
            continue
        record = truth_records[key]
        check(record["identity"]["case"] == case and record["identity"]["data_seed"] == seed, "Truth identity mismatch")
        lineage_checks.verify_record(record, truth_paths[key], plan, truth_id=f"{case}_s{seed}", sources=sources)
        check(record["status"] == "complete" and [e["pattern"] for e in record["entries"]] == patterns, "Truth record incomplete")
        arrays = sources.arrays(truth_paths[key].parent / "fields.npz", "truth_fields", record["fields_sha256"])
        truth_arrays[key] = arrays
        expected = initial_fields(protocol, intervals[key], seed)
        expected_keys = {pattern + suffix for pattern in patterns for suffix in ("_times", "_fields", "_refined_fields")}
        check(set(arrays) == expected_keys, "Unexpected truth NPZ keys")
        for entry in record["entries"]:
            pattern = entry["pattern"]
            t, fields, refined = arrays[pattern + "_times"], arrays[pattern + "_fields"], arrays[pattern + "_refined_fields"]
            check(entry["primary"]["settings"] == protocol["solver"]
                  and entry["refined"]["settings"] == protocol["refined_solver"], "Truth solver metadata changed")
            check(sha(expected[pattern].tobytes()) == entry["initial_sha256"], "Truth initial hash mismatch")
            check(math.isclose(float(expected[pattern].mean()), entry["initial_mean"], abs_tol=2e-15)
                  and math.isclose(float(np.ptp(expected[pattern]) / 2), entry["initial_half_amplitude"], abs_tol=2e-15),
                  "Truth initial summary mismatch")
            validate_trajectory(t, fields, expected[pattern], grid, protocol["nx"], entry["primary"]["status"], f"truth {key} {pattern}")
            validate_trajectory(t, refined, expected[pattern], grid, protocol["nx"], entry["refined"]["status"], f"truth refined {key} {pattern}")
            check(entry["primary"]["status"] == entry["refined"]["status"] == "complete", "Failed truth cannot support scoring")
            check([c["horizon"] for c in entry["temporal_comparison"]] == horizons, "Truth temporal horizons mismatch")
            for h, comparison in zip(horizons, entry["temporal_comparison"]):
                temporal_rows.append(temporal_row(identity, pattern, h, (t, fields, "complete"),
                                                  (t, refined, "complete"), (t, refined), comparison))
            counts["truth_primary_trajectories"] += 1
            counts["truth_refined_trajectories"] += 1

    absolute_rows = []
    for uid, unit in units.items():
        identity = {key: unit[key] for key in ("kind", "case", "setting", "data_seed", "variant", "epoch", "method")}
        identity["unit_id"] = uid
        # Keep one column order for truth and model refinement rows.
        identity = {key: identity[key] for key in ("kind", "unit_id", "case", "setting", "data_seed", "variant", "epoch", "method")}
        if uid not in records:
            if unit["refinement"]:
                for pattern in patterns:
                    temporal_rows.extend(temporal_row(identity, pattern, h) for h in horizons)
            continue
        record = records[uid]
        check(record["identity"]["unit"] == unit, f"Unit identity mismatch: {uid}")
        lineage_checks.verify_record(record, OUTPUT / "runs" / uid / "result.json", plan, unit=unit, sources=sources)
        check(canonical(record["metadata"]) == record["identity"]["model_metadata_sha256"], f"Model metadata changed: {uid}")
        input_id = unit.get("input_id", f"{unit['case']}_{unit['setting']}_s{unit['data_seed']}")
        check(record["identity"]["input_file_sha256"] == inputs[input_id]["file_sha256"], f"Run input hash mismatch: {uid}")
        check(record["status"] == "evaluated" and [entry["pattern"] for entry in record["entries"]] == patterns, f"Incomplete record: {uid}")
        check(sum(e["info"]["status"] != "complete" for e in record["entries"]) == record["numerical_failure_patterns"], f"Failure count mismatch: {uid}")
        arrays = sources.arrays(OUTPUT / "runs" / uid / "fields.npz", "forward_fields", record["fields_sha256"])
        suffixes = ("_times", "_fields", "_refined_times", "_refined_fields") if unit["refinement"] else ("_times", "_fields")
        check(set(arrays) == {pattern + suffix for pattern in patterns for suffix in suffixes}, f"Unexpected run NPZ keys: {uid}")
        key = (unit["case"], unit["data_seed"])
        check(key in truth_arrays, f"Run exists without committed truth: {uid}")
        expected = initial_fields(protocol, intervals[key], unit["data_seed"])
        for entry in record["entries"]:
            pattern = entry["pattern"]
            t, fields = arrays[pattern + "_times"], arrays[pattern + "_fields"]
            true_t, true = truth_arrays[key][pattern + "_times"], truth_arrays[key][pattern + "_fields"]
            info = entry["info"]
            check(info["settings"] == protocol["solver"], f"Run solver metadata changed: {uid}")
            validate_trajectory(t, fields, expected[pattern], grid, protocol["nx"], info["status"], f"{uid} {pattern}")
            check(np.array_equal(fields[0], true[0]), f"Run/truth IC mismatch: {uid} {pattern}")
            check(info["state_clipping"] is False and info["support_departure_is_failure"] is False, "Clipping/support rule changed")
            check([score["horizon"] for score in entry["scores"]] == horizons, f"Score horizons mismatch: {uid}")
            counts[unit["kind"] + "_primary_trajectories"] += 1
            status_counts[unit["kind"] + ":" + info["status"]] += 1
            for score in entry["scores"]:
                h = score["horizon"]
                reached = len(t) > int(round(h / dt))
                check(score["reached"] == reached, f"Reached score mismatch: {uid} {pattern} {h}")
                guard = info["audit"]["guard"] or {}
                out = dict(**identity, pattern=pattern, horizon=h, reached=reached,
                           numerical_status=info["status"], guard_reason=guard.get("reason"), guard_time=guard.get("time"))
                for kind in ("accepted", "trial"):
                    outside = info["audit"]["first_" + kind + "_outside"]
                    negative = info["audit"]["first_" + kind + "_negative"]
                    check(score[kind + "_support_valid"] == (reached and (outside is None or outside["time"] > h + 1e-12)), "Support score mismatch")
                    check(score[kind + "_nonnegative"] == (reached and (negative is None or negative > h + 1e-12)), "Nonnegative score mismatch")
                    out[kind + "_support_valid"] = score[kind + "_support_valid"]
                    out[kind + "_nonnegative"] = score[kind + "_nonnegative"]
                metric_names = ("endpoint_absolute_rms", "space_time_absolute_rms", "truth_endpoint_rms", "truth_space_time_rms",
                                "endpoint_relative_l2", "space_time_relative_l2", "endpoint_error", "space_time_error",
                                "endpoint_normalization", "space_time_normalization")
                out.update({name: None for name in metric_names})
                out.update(space_time_absolute_rms_dt02=None, space_time_error_dt02=None, space_time_normalization_dt02=None,
                           space_time_absolute_rms_dt02_minus_dt01=None, space_time_error_dt02_minus_dt01=None)
                if reached:
                    fine = metrics(t, fields, true, h)
                    coarse = metrics(t, fields, true, h, stride=2)
                    out.update(fine)
                    for name in ("endpoint_error", "space_time_error"):
                        difference = abs(fine[name] - score[name])
                        maxima[name.replace("_error", "_metric_recomputation_absolute_difference")] = max(maxima[name.replace("_error", "_metric_recomputation_absolute_difference")], difference)
                        check(math.isclose(fine[name], score[name], rel_tol=2e-13, abs_tol=2e-15), f"Metric mismatch: {uid} {pattern} {h} {name}")
                    check(fine["endpoint_normalization"] == score["endpoint_normalization"]
                          and fine["space_time_normalization"] == score["space_time_normalization"], "Metric normalization mismatch")
                    abs_change = coarse["space_time_absolute_rms"] - fine["space_time_absolute_rms"]
                    metric_change = coarse["space_time_error"] - fine["space_time_error"]
                    out.update(space_time_absolute_rms_dt02=coarse["space_time_absolute_rms"], space_time_error_dt02=coarse["space_time_error"],
                               space_time_normalization_dt02=coarse["space_time_normalization"],
                               space_time_absolute_rms_dt02_minus_dt01=abs_change, space_time_error_dt02_minus_dt01=metric_change)
                    maxima["space_time_absolute_rms_quadrature_change"] = max(maxima["space_time_absolute_rms_quadrature_change"], abs(abs_change))
                    maxima["space_time_reported_metric_quadrature_change"] = max(maxima["space_time_reported_metric_quadrature_change"], abs(metric_change))
                    counts["completed_horizon_metrics_recomputed"] += 1
                else:
                    check(all(score[name] is None for name in ("endpoint_error", "space_time_error", "endpoint_normalization", "space_time_normalization")), "Unreached metric must be null")
                    counts["unreached_horizon_metrics_checked_null"] += 1
                absolute_rows.append(out)
            if unit["refinement"]:
                check("refined_info" in entry and "temporal_comparison" in entry, f"Missing planned refinement: {uid}")
                check(entry["refined_info"]["settings"] == protocol["refined_solver"], f"Refined solver metadata changed: {uid}")
                rt, refined = arrays[pattern + "_refined_times"], arrays[pattern + "_refined_fields"]
                validate_trajectory(rt, refined, expected[pattern], grid, protocol["nx"], entry["refined_info"]["status"], f"{uid} refined {pattern}")
                check([c["horizon"] for c in entry["temporal_comparison"]] == horizons, "Model temporal horizons mismatch")
                for h, comparison in zip(horizons, entry["temporal_comparison"]):
                    temporal_rows.append(temporal_row(identity, pattern, h, (t, fields, info["status"]),
                                                      (rt, refined, entry["refined_info"]["status"]), (true_t, true), comparison))
                counts["model_refined_trajectories"] += 1
                status_counts["model_refined:" + entry["refined_info"]["status"]] += 1

    missing = [uid for uid in units if uid not in records]
    missing_truth = [f"{key[0]}_s{key[1]}" for key in truth_keys if key not in truth_records]
    planned_kind = Counter(unit["kind"] for unit in units.values())
    evaluated_kind = Counter(units[uid]["kind"] for uid in records)
    planned = dict(states=500, states_by_kind=dict(planned_kind), primary_trajectories=2000,
                   truth_primary_trajectories=40, truth_refined_trajectories=40,
                   model_refined_trajectories=128, total_integrations=2208,
                   horizon_metric_rows=6000, temporal_horizon_rows=504)
    check(len(temporal_rows) == planned["temporal_horizon_rows"], "Planned temporal rows missing")
    check(len(absolute_rows) == len(records) * len(patterns) * len(horizons), "Metric row count mismatch")
    complete = not missing and not missing_truth and counts["model_refined_trajectories"] == 128
    temporal_summary = {}
    for kind in ("truth", "model"):
        rows = [row for row in temporal_rows if row["kind"] == kind]
        evaluated = [row for row in rows if row["snapshot_status"] == "evaluated"]
        full = [row for row in evaluated if row["both_reached"]]
        temporal_summary[kind] = dict(planned_trajectories=len(rows) // len(horizons),
            evaluated_trajectories=len(evaluated) // len(horizons), planned_horizon_rows=len(rows),
            evaluated_horizon_rows=len(evaluated), both_reached_horizon_rows=len(full),
            endpoint_tolerance_passed_horizon_rows=sum(row["endpoint_temporal_tolerance_passed"] is True for row in evaluated),
            failed_comparison_unresolved_horizon_rows=sum(row["failed_comparison_unresolved"] is True for row in evaluated),
            maximum_saved_grid_relative_discrepancy=max((row["max_difference_relative_to_truth_over_saved_grid"] for row in evaluated), default=None),
            maximum_endpoint_relative_discrepancy=max((row["endpoint_difference_relative_to_truth"] for row in full), default=None),
            maximum_space_time_relative_discrepancy=max((row["space_time_difference_relative_to_truth"] for row in full), default=None))
    reports = OUTPUT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    sources_path = reports / "ARRAY_AUDIT_SOURCES.json"
    temporal_path = reports / "temporal_trajectory_by_seed.csv"
    absolute_path = reports / "forward_absolute_by_seed.csv"
    atomic_json(sources_path, dict(snapshot_started_utc=started, result_capture_finished_utc=captured,
                                  sources=sorted(sources.rows.values(), key=lambda row: row["path"])))
    atomic_csv(temporal_path, temporal_rows)
    # A completely empty pending snapshot retains the same CSV column schema.
    atomic_csv(absolute_path, absolute_rows, empty_fieldnames=[
        "kind", "unit_id", "case", "setting", "data_seed", "variant", "epoch", "method", "pattern", "horizon", "reached",
        "numerical_status", "guard_reason", "guard_time", "accepted_support_valid", "accepted_nonnegative",
        "trial_support_valid", "trial_nonnegative", "endpoint_absolute_rms", "space_time_absolute_rms",
        "truth_endpoint_rms", "truth_space_time_rms", "endpoint_relative_l2", "space_time_relative_l2",
        "endpoint_error", "space_time_error", "endpoint_normalization", "space_time_normalization",
        "space_time_absolute_rms_dt02", "space_time_error_dt02", "space_time_normalization_dt02",
        "space_time_absolute_rms_dt02_minus_dt01", "space_time_error_dt02_minus_dt01"])
    result_paths_now = {str(path.resolve()) for path in (OUTPUT / "runs").glob("*/result.json")}
    report = dict(cohort="fresh_GPU_only", new_model_integrations_planned=1728, imported_integration_records_reused=480, status="passed", classification="saved_array_postprocessing_only", snapshot_started_utc=started,
        result_capture_finished_utc=captured, audit_finished_utc=utc(), snapshot_complete=complete,
        require_complete_requested=args.require_complete, completeness_requirement_passed=complete if args.require_complete else None,
        command=[sys.executable, *sys.argv], runtime=dict(python=platform.python_version(), numpy=np.__version__),
        protocol_sha256=plan["protocol_sha256"], planned=planned,
        resolved_repository_root=str(REPO),
        path_resolution="Resolve protocol-declared input/training directories relative to the current repository, retaining original immutable manifest byte hashes. Absolute paths inside old metadata are provenance labels, not file lookup requirements.",
        evaluated=dict(states=len(records), states_by_kind=dict(evaluated_kind), truth_units=len(truth_records),
                       total_integration_results=sum(counts[key] for key in ("model_primary_trajectories", "baseline_primary_trajectories",
                           "global_reference_primary_trajectories", "truth_primary_trajectories", "truth_refined_trajectories", "model_refined_trajectories")),
                       counts=dict(counts), status_counts=dict(status_counts), metric_rows=len(absolute_rows)),
        missing_state_ids=missing, missing_truth_ids=missing_truth,
        result_paths_added_after_snapshot=sorted(result_paths_now - {str(path.resolve()) for path in run_paths}),
        array_paths_without_captured_result=uncommitted_fields,
        array_inventory=dict(captured_field_npz_paths=len(field_paths_at_capture), validated_committed_field_npz=len(expected_used_fields),
                             observation_input_npz=len(inputs), source_files_hashed=len(sources.rows)),
        checks=dict(source_identities_and_sha256="passed", all_captured_committed_npz_sha256="passed",
                    initial_field_reconstruction_and_run_truth_bitwise_equality="passed",
                    saved_time_grids_field_shapes_and_finite_values="passed",
                    stored_endpoint_and_space_time_metrics_and_normalizations="passed",
                    unreached_metrics_null_and_support_nonnegative_flags="passed",
                    existing_endpoint_temporal_comparisons="passed"),
        validation_maxima=maxima, temporal=temporal_summary,
        artifact_sha256={path.name: sha(path.read_bytes()) for path in (sources_path, temporal_path, absolute_path)},
        fields=dict(truth_npz="<pattern>_times, <pattern>_fields, <pattern>_refined_fields",
                    run_npz="<pattern>_times, <pattern>_fields; selected states additionally <pattern>_refined_times, <pattern>_refined_fields",
                    shape="times: (n,); fields: (n,64); complete n=61; failed solves retain the saved-grid prefix"),
        interpretation=dict(temporal="Only the predeclared subset has model refinement evidence. Saved-grid comparisons do not certify all RK stages, continuous time, unrefined states, or a spatial continuum solution.",
            failure="Scientific guard failures remain evaluated integrations with null unreached metrics. A guard is not proof of continuum blow-up. Failed primary/refined comparisons remain unresolved.",
            absolute="Absolute endpoint RMS and sqrt(trapezoid(mean(error**2),t)/horizon) accompany relative metrics so changes in truth normalization remain visible.",
            quadrature="dt02 columns select every second existing dt01 saved sample, retaining both endpoints. Their differences measure trapezoidal quadrature sensitivity on the same fields, not .005 sampling accuracy or a new PDE integration.",
            snapshot="Result JSON contents are captured before field analysis; later atomic publications are excluded. Arrays without a captured result may be in progress and are listed but not treated as evaluated or validated.",
            independence="Data seeds are the replicates; reused truth, references, ICs and checkpoints do not increase the independent sample count."))
    atomic_json(reports / "ARRAY_AUDIT.json", report)
    print(json.dumps({key: report[key] for key in ("status", "snapshot_complete", "evaluated", "validation_maxima", "temporal")}, indent=2))
    if args.require_complete and not complete:
        print(f"Incomplete frozen cohort: {len(missing)} states and {len(missing_truth)} truth units pending; audit artifacts were written.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
