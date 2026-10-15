"""Comprehensive norm/grid sensitivity of the fresh GPU-150/1500 forward cohort.

Evaluate all200 existing native/frozen pairs; never fit, train, integrate or pick
a replacement example. The original256-point equal-weight norm is retained next
to trapezoidal estimates on256/1025/4097/8193 points. All inputs must belong
to the frozen GPU cohort and its authenticated native-export manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
OUTPUT = REPO / "outputs" / STUDY.name / "reports/function_norm_sensitivity"
SOURCE_STUDY = REPO / "research/reviewer-forward-2026-10-15-gpu"
SOURCE_OUTPUT = REPO / "outputs/reviewer-forward-2026-10-15-gpu"
TOLERANCE = 1e-10
NORM_EPSILON = 1e-12
CONFIGURATIONS = [("discrete_equal_weight", 256), ("trapezoidal", 256),
                  ("trapezoidal", 1025), ("trapezoidal", 4097), ("trapezoidal", 8193)]


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def direction(delta):
    return "better" if delta < -TOLERANCE else "worse" if delta > TOLERANCE else "tied_within_reporting_tolerance"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-memory-cap", action="store_true")
    args = parser.parse_args()
    cgroup = next(row.split("::", 1)[1] for row in Path("/proc/self/cgroup").read_text().splitlines() if row.startswith("0::"))
    memory_path = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "memory.max"
    limit = memory_path.read_text().strip()
    if args.require_memory_cap:
        assert limit != "max" and int(limit) <= 2 * 1024**3, "Run this audit in a MemoryMax<=2G cgroup"
    spec = importlib.util.spec_from_file_location("gpu_original_forward", SOURCE_STUDY / "run_forward.py")
    sys.path.insert(0, str(SOURCE_STUDY))
    forward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forward)
    plan = forward.read_plan()
    import lineage_checks
    OUTPUT.mkdir(parents=True, exist_ok=True)
    core = forward.load_core(OUTPUT)
    from reliability_core import NumpyMLPClosure
    assert core.torch.get_num_threads() == 1
    units = [unit for unit in plan["units"] if unit["kind"] == "model" and unit["method"] == "native"]
    assert len(units) == 200
    units_by_id = {unit["id"]: unit for unit in plan["units"]}
    package_path = SOURCE_OUTPUT / "native_exports/NATIVE_EXPORTS.json"
    package = json.loads(package_path.read_text())
    package_index = {row["id"]: row for row in package["records"]}
    planned_pairs = {f"{unit['training_id']}_e{unit['epoch']}": unit for unit in units}
    assert package["cohort"] == "fresh_GPU_only" and package["source_training_device"] == "cuda"
    assert package["complete"] and not package["pending"]
    assert package["source_forward_manifest_sha256"] == digest(SOURCE_STUDY / "MANIFEST.json")
    assert package["evaluator_source_sha256"] == digest(REPO / "revision/reliability_core.py")
    assert len(package_index) == len(package["records"]) == package["exported_unique_checkpoints"] == package["planned_unique_checkpoints"] == 200
    assert set(package_index) == set(planned_pairs)
    rows, index, source_hashes, prediction_scores = [], {}, {}, {}
    for unit in units:
        pair_id = f"{unit['training_id']}_e{unit['epoch']}"
        native_path = SOURCE_OUTPUT / "runs" / unit["id"] / "result.json"
        frozen_id = unit["id"].removesuffix("__native") + "__frozen"
        frozen_path = SOURCE_OUTPUT / "runs" / frozen_id / "result.json"
        native_record, frozen_record = json.loads(native_path.read_text()), json.loads(frozen_path.read_text())
        lineage_checks.verify_record(native_record, native_path, plan, unit=unit)
        lineage_checks.verify_record(frozen_record, frozen_path, plan, unit=units_by_id[frozen_id])
        assert native_record["metadata"] == frozen_record["metadata"]
        assert native_record["identity"]["input_file_sha256"] == frozen_record["identity"]["input_file_sha256"]
        for entry in native_record["entries"]:
            for score in entry["scores"]:
                prediction_scores[(pair_id, entry["pattern"], score["horizon"], "native")] = score
        for entry in frozen_record["entries"]:
            for score in entry["scores"]:
                prediction_scores[(pair_id, entry["pattern"], score["horizon"], "frozen")] = score
        export = SOURCE_OUTPUT / "native_exports" / (pair_id + ".npz")
        sidecar_path = export.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        packaged = package_index[pair_id]
        assert packaged["export_file"] == export.name and packaged["metadata_file"] == sidecar_path.name
        assert packaged["metadata_sha256"] == digest(sidecar_path) and packaged["export_sha256"] == digest(export)
        expected_identity = dict(training_id=unit["training_id"], epoch=unit["epoch"],
            checkpoint_sha256=native_record["metadata"]["checkpoint_sha256"],
            input_file_sha256=native_record["identity"]["input_file_sha256"], protocol_sha256=plan["protocol_sha256"],
            driver_sha256=plan["driver_sha256"], model_metadata_sha256=native_record["identity"]["model_metadata_sha256"])
        assert sidecar["identity"] == expected_identity
        assert digest(export) == sidecar["export_sha256"]
        assert sidecar["export_file"] == export.name and sidecar["export_bytes"] == export.stat().st_size == packaged["bytes"]
        assert sidecar["source_forward_result_sha256"] == digest(native_path)
        proof_path = native_path.parent / "CHECKPOINT_DEVICE.json"
        assert sidecar["source_training_device"] == "cuda"
        assert sidecar["source_device_proof_sha256"] == digest(proof_path)
        assert sidecar["source_device_proof"] == json.loads(proof_path.read_text())
        assert sidecar["frozen_expressions"] == native_record["metadata"]["frozen_expressions"]
        assert sidecar["model_verification"] == native_record["metadata"]["model_verification"]
        assert sidecar["fits_or_integrations_performed"] == 0
        assert sidecar["identity"]["checkpoint_sha256"] == native_record["metadata"]["checkpoint_sha256"]
        assert sidecar["identity"]["model_metadata_sha256"] == native_record["identity"]["model_metadata_sha256"]
        native = NumpyMLPClosure.load(export)
        assert native.metadata == sidecar["native_metadata"] == native_record["metadata"]["native_metadata"]
        expressions = [core.SymbolicExpression(**dict(item, parameters=tuple(item["parameters"]))) for item in native_record["metadata"]["frozen_expressions"]]
        case, interval = core.extra_case(unit["case"]), tuple(sidecar["value_range"])
        assert interval == tuple(case.value_range)
        source_hashes[pair_id] = dict(native_result_sha256=digest(native_path), frozen_result_sha256=digest(frozen_path),
            native_export_sha256=digest(export), sidecar_sha256=digest(sidecar_path),
            checkpoint_sha256=native_record["metadata"]["checkpoint_sha256"], curves_sha256=native_record["metadata"]["curves_sha256"],
            native_device_proof_sha256=digest(proof_path), frozen_device_proof_sha256=digest(frozen_path.parent / "CHECKPOINT_DEVICE.json"),
            input_file_sha256=native_record["identity"]["input_file_sha256"])
        for norm,points in CONFIGURATIONS:
            grid = np.linspace(*interval, points)
            for branch, true_law, native_law, expression in zip(("D", "R"), (case.diffusion, case.reaction), (native.diffusion, native.reaction), expressions):
                true, a, b = true_law(grid), native_law(grid), expression.evaluate(grid)
                assert np.isfinite(true).all() and np.isfinite(a).all() and np.isfinite(b).all()
                if norm == "discrete_equal_weight":
                    truth_norm = float(np.linalg.norm(true))
                    native_norm, frozen_norm = float(np.linalg.norm(a-true)), float(np.linalg.norm(b-true))
                else:
                    truth_norm = float(np.sqrt(np.trapezoid(true**2, grid)))
                    native_norm = float(np.sqrt(np.trapezoid((a-true)**2, grid)))
                    frozen_norm = float(np.sqrt(np.trapezoid((b-true)**2, grid)))
                en, es = native_norm/(truth_norm+NORM_EPSILON), frozen_norm/(truth_norm+NORM_EPSILON)
                row = dict(pair_id=pair_id, training_id=unit["training_id"], case=unit["case"], setting=unit["setting"],
                    data_seed=unit["data_seed"], variant=unit["variant"], epoch=unit["epoch"],
                    norm=norm, grid_points=points, branch=branch, truth_norm=truth_norm,
                    native_absolute_error_norm=native_norm, frozen_absolute_error_norm=frozen_norm,
                    native_relative_error=en, frozen_relative_error=es, frozen_minus_native_relative_error=es-en,
                    BIR=frozen_norm/native_norm if native_norm else None, direction=direction(es-en))
                rows.append(row); index[(pair_id,norm,points,branch)] = row
    assert len(rows) == 2000
    assert len(index) == 2000, "Duplicate function-norm branch combinations"
    pair_summary, joint = [], {}
    for unit in units:
        pair_id = f"{unit['training_id']}_e{unit['epoch']}"
        flags = {}
        for norm,points in CONFIGURATIONS:
            d, r = [index[(pair_id,norm,points,branch)] for branch in ("D", "R")]
            flags[(norm,points)] = d["direction"] == r["direction"] == "better"
            joint[(pair_id,norm,points)] = flags[(norm,points)]
        original = flags[("discrete_equal_weight",256)]
        robust = all(flags.values())
        pair_summary.append(dict(pair_id=pair_id, case=unit["case"], setting=unit["setting"], data_seed=unit["data_seed"], variant=unit["variant"], epoch=unit["epoch"],
            original_256_discrete_both_function_errors_improve=original,
            all_grid_norms_both_function_errors_improve=robust,
            trapezoid_8193_both_function_errors_improve=flags[("trapezoidal",8193)],
            both_function_improvement_is_norm_sensitive=len(set(flags.values()))>1,
            D_direction_changes=index[(pair_id,"discrete_equal_weight",256,"D")]["direction"] != index[(pair_id,"trapezoidal",8193,"D")]["direction"],
            R_direction_changes=index[(pair_id,"discrete_equal_weight",256,"R")]["direction"] != index[(pair_id,"trapezoidal",8193,"R")]["direction"]))
    link_path = SOURCE_OUTPUT / "reports/function_forward_links_by_seed.csv"
    with link_path.open(newline="") as stream:
        links = list(csv.DictReader(stream))
    assert len(links) == 4800 and len({row["pair_id"] for row in links}) == 200
    expected_links = {(pair, pattern["id"], float(horizon), metric) for pair in planned_pairs
                      for pattern in plan["protocol"]["initial_conditions"] for horizon in plan["protocol"]["horizons"]
                      for metric in ("endpoint", "space_time")}
    observed_links = {(row["pair_id"], row["pattern"], float(row["horizon"]), row["metric"]) for row in links}
    assert len(observed_links) == len(links) and observed_links == expected_links, "Duplicate, missing or unplanned forward-link combinations"
    for row in links:
        unit = planned_pairs[row["pair_id"]]
        assert all(row[key] == str(unit[key]) for key in ("case", "setting", "data_seed", "variant", "epoch"))
        for method in ("native", "frozen"):
            score = prediction_scores[(row["pair_id"], row["pattern"], float(row["horizon"]), method)]
            expected_error = score[row["metric"] + "_error"]
            actual_error = row[method + "_prediction_error"]
            assert (not actual_error and expected_error is None) or (bool(actual_error) and float(actual_error) == expected_error), "Forward-link score differs from the authenticated GPU result"
    expanded = []
    robust_by_pair = {row["pair_id"]:row["all_grid_norms_both_function_errors_improve"] for row in pair_summary}
    for original in links:
        for norm,points in CONFIGURATIONS:
            row = {key:original[key] for key in ("pair_id", "case", "setting", "data_seed", "variant", "epoch", "pattern", "horizon", "metric", "prediction_direction", "temporal_resolution", "native_prediction_error", "frozen_prediction_error")}
            row.update(norm=norm, grid_points=points,
                both_function_errors_improve=joint[(original["pair_id"],norm,points)],
                both_function_errors_improve_on_all_norms_grids=robust_by_pair[original["pair_id"]],
                joint_function_improvement_forward_worsening=joint[(original["pair_id"],norm,points)] and original["prediction_direction"] == "worse")
            expanded.append(row)
    assert len(expanded) == 24000
    assert len({(row["pair_id"], row["pattern"], float(row["horizon"]), row["metric"], row["norm"], row["grid_points"])
                for row in expanded}) == 24000, "Duplicate linked norm/grid combinations"
    csv_write(OUTPUT / "function_norm_sensitivity_by_checkpoint.csv", rows)
    csv_write(OUTPUT / "function_norm_robustness_by_checkpoint.csv", pair_summary)
    csv_write(OUTPUT / "function_forward_norm_sensitivity_by_seed.csv", expanded)
    aggregate = []
    for norm,points in CONFIGURATIONS:
        matches = [row for row in expanded if row["norm"] == norm and row["grid_points"] == points]
        aggregate.append(dict(norm=norm, grid_points=points, evaluated_checkpoint_pairs=200,
            both_function_errors_improve=sum(joint[(f"{unit['training_id']}_e{unit['epoch']}",norm,points)] for unit in units),
            joint_function_improvement_forward_worsening=sum(row["joint_function_improvement_forward_worsening"] for row in matches),
            resolved_joint_function_improvement_forward_worsening=sum(row["joint_function_improvement_forward_worsening"] and row["temporal_resolution"] == "resolved_on_declared_refined_seed_subset" for row in matches)))
    summary = dict(status="passed", complete=True, cohort="fresh_GPU150_1500_only", source_forward_study=SOURCE_STUDY.name,
        snapshot_utc=datetime.now(timezone.utc).isoformat(), unique_checkpoint_pairs=200,
        function_norm_branch_rows=len(rows), linked_IC_horizon_metric_norm_rows=len(expanded),
        original_discrete_both_function_improvements=sum(row["original_256_discrete_both_function_errors_improve"] for row in pair_summary),
        robust_both_function_improvements_all_tested_norms_grids=sum(row["all_grid_norms_both_function_errors_improve"] for row in pair_summary),
        norm_sensitive_both_function_improvement_pairs=sum(row["both_function_improvement_is_norm_sensitive"] for row in pair_summary),
        selected_example_pair=None, aggregates=aggregate,
        norm_epsilon=NORM_EPSILON, direction_absolute_tolerance=TOLERANCE, configurations=CONFIGURATIONS,
        source_forward_manifest_sha256=digest(SOURCE_STUDY / "MANIFEST.json"), source_link_csv_sha256=digest(link_path),
        source_native_export_manifest_sha256=digest(package_path), lineage_checks_sha256=digest(STUDY / "lineage_checks.py"),
        source_reliability_core_sha256=digest(REPO / "revision/reliability_core.py"), source_models=source_hashes,
        script_sha256=digest(__file__), artifacts={path.name:digest(path) for path in OUTPUT.glob("*.csv")},
        resources=dict(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, cgroup_memory_max=limit,
            threads_per_numerical_library=1, processes=1), new_fits_or_integrations=0,
        interpretation=["This comprehensive audit retains the originally declared256-point equal-weight norm and adds prespecified sensitivity grids without selecting a different example or refitting.",
            "Dense-grid trapezoidal values estimate an interval L2 norm; they are not a rigorous continuum/quadrature error certificate.",
            "A both-functions-improve statement must state its norm. If a branch changes direction under quadrature weighting, the original discrete-norm counterexample is not a norm-robust joint-improvement counterexample.",
            "Forward fields/errors are unchanged. ICs, horizons, metrics and norm grids reuse the same200 checkpoint pairs and20 observations, with five data seeds per case/setting/variant."])
    write(OUTPUT / "FUNCTION_NORM_SENSITIVITY_AUDIT.json", summary)
    print(json.dumps({key:summary[key] for key in ("status", "unique_checkpoint_pairs", "original_discrete_both_function_improvements", "robust_both_function_improvements_all_tested_norms_grids", "norm_sensitive_both_function_improvement_pairs", "aggregates", "resources")}, indent=2))


if __name__ == "__main__":
    main()
