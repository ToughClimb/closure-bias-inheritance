"""Isolated CPU forward validation of the fresh sensitivity-study cohort."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.integrate import RK45

from protocol import PROTOCOL

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
OUTPUT = REPO / "outputs" / STUDY.name
INPUT_ROOT = REPO / "outputs" / PROTOCOL["upstream_study"]
TRAINING_ROOT = REPO / "outputs" / PROTOCOL["training_output_directory"]
UPSTREAM_MANIFEST = REPO / "research" / PROTOCOL["upstream_study"] / "MANIFEST.json"
GLOBAL_ROOT = REPO / "outputs/reviewer-objective-controls-2026-10-15/shared/bundles"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise FloatingPointError("Nonfinite JSON field")
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".{os.getpid()}.tmp")
    staging.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")
    staging.replace(path)


def load_core(output=OUTPUT):
    os.environ["EAAI_WORK_ROOT"] = str(Path(output).resolve())
    os.environ["EAAI_INPUT_ROOT"] = str(INPUT_ROOT)
    os.environ["EAAI_DEVICE"] = "cpu"
    sys.path.insert(0, str(REPO / "revision"))
    core = importlib.import_module("revision_core")
    if core.ROOT != Path(output).resolve() or core.INPUT_ROOT != INPUT_ROOT or core.DEVICE_OPTION != "cpu":
        raise ValueError("Already-imported revision core points to another artifact root/device")
    core.torch.set_num_threads(1)
    return core


def load_input(core, identifier):
    manifest = json.loads((INPUT_ROOT / "INPUT_MANIFEST.json").read_text())
    row = next(item for item in manifest["inputs"] if item["id"] == identifier)
    path = INPUT_ROOT / row["file"]
    if digest(path) != row["file_sha256"] or row["core_source_signature"] != core.source_signature():
        raise ValueError("Input byte/source identity changed")
    with np.load(path, allow_pickle=False) as arrays:
        observed = {key: arrays[key].copy() for key in ("u", "x", "t", "u0")}
        interval = tuple(map(float, arrays["value_range"]))
    if core.observation_signature(observed, interval) != row["observation_signature"]:
        raise ValueError("Observed-field array identity changed")
    return observed, interval, row


def initial_fields(interval, seed):
    x = np.arange(PROTOCOL["nx"]) / PROTOCOL["nx"]
    phase = np.random.default_rng(seed + 20261015).uniform(0, 2 * np.pi, 2)
    shape = np.sin(2 * np.pi * x + phase[0]) + .25 * np.sin(6 * np.pi * x + phase[1])
    shape /= np.max(np.abs(shape))
    result = {}
    for row in PROTOCOL["initial_conditions"]:
        mean = interval[0] + row["mean_fraction"] * (interval[1] - interval[0])
        amplitude = row["half_amplitude_fraction"] * (interval[1] - interval[0])
        values = mean + amplitude * shape
        result[row["id"]] = values
        np.testing.assert_allclose([values.mean(), (values.max() - values.min()) / 2], [mean, amplitude], atol=2e-15)
    return result


class ForwardGuard(RuntimeError):
    def __init__(self, reason, time_value, **details):
        self.details = dict(reason=reason, time=float(time_value), **details)
        super().__init__(reason)


def strict_integrate(D, R, u0, interval, settings, poles=()):
    """One solve to the maximum horizon, retaining every reached requested time."""
    started = time.perf_counter()
    target = max(PROTOCOL["horizons"])
    save_times = np.linspace(0., target, int(round(target / PROTOCOL["save_dt"])) + 1)
    state = np.asarray(u0, dtype=float).copy()
    dx = 1 / state.size
    audit = dict(rhs_calls=0, accepted_steps=0, accepted_min=float(state.min()), accepted_max=float(state.max()),
                 trial_min=float(state.min()), trial_max=float(state.max()), first_accepted_outside=None,
                 first_trial_outside=None, first_accepted_negative=None, first_trial_negative=None,
                 min_D=None, max_D=None, guard=None)

    def track(values, t, accepted):
        label = "accepted" if accepted else "trial"
        if not np.isfinite(values).all():
            raise ForwardGuard("nonfinite_" + label + "_state", t)
        lo, hi = float(values.min()), float(values.max())
        audit[label + "_min"] = min(audit[label + "_min"], lo)
        audit[label + "_max"] = max(audit[label + "_max"], hi)
        negative_key = "first_" + label + "_negative"
        if lo < -settings["range_tolerance"] and (audit[negative_key] is None or t < audit[negative_key]):
            audit[negative_key] = float(t)
        if (lo < interval[0] - settings["range_tolerance"] or hi > interval[1] + settings["range_tolerance"]):
            key = "first_" + label + "_outside"
            if audit[key] is None or t < audit[key]["time"]:
                audit[key] = dict(time=float(t), minimum=lo, maximum=hi)
        if max(abs(lo), abs(hi)) > settings["absolute_state_guard"]:
            raise ForwardGuard("absolute_state_guard", t, minimum=lo, maximum=hi)
        for pole in poles:
            if lo <= pole <= hi:
                raise ForwardGuard("nonremovable_pole_in_field_enclosure", t, pole=float(pole), minimum=lo, maximum=hi)

    def law(values, t):
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            d = np.broadcast_to(np.asarray(D(values), dtype=float), values.shape)
            r = np.broadcast_to(np.asarray(R(values), dtype=float), values.shape)
        if not np.isfinite(d).all() or not np.isfinite(r).all():
            raise ForwardGuard("nonfinite_constitutive_values", t)
        if float(d.min()) <= 0:
            raise ForwardGuard("nonpositive_diffusion_at_queried_state", t, minimum_D=float(d.min()))
        audit["min_D"] = float(d.min()) if audit["min_D"] is None else min(audit["min_D"], float(d.min()))
        audit["max_D"] = float(d.max()) if audit["max_D"] is None else max(audit["max_D"], float(d.max()))
        return d, r

    def rhs(t, values):
        audit["rhs_calls"] += 1
        if audit["rhs_calls"] > settings["max_rhs_evaluations"]:
            raise ForwardGuard("rhs_evaluation_budget", t)
        track(values, t, False)
        try:
            d, r = law(values, t)
        except (FloatingPointError, OverflowError) as exc:
            raise ForwardGuard("constitutive_arithmetic", t, detail=str(exc)) from exc
        flux = .5 * (d + np.roll(d, -1)) * (np.roll(values, -1) - values) / dx
        derivative = (flux - np.roll(flux, 1)) / dx + r
        if not np.isfinite(derivative).all():
            raise ForwardGuard("nonfinite_rhs", t)
        return derivative

    times, fields = [0.], [state.copy()]
    status, reason, solver, next_save = "complete", None, None, 1
    reached_time = 0.
    try:
        track(state, 0., True)
        d, _ = law(state, 0.)
        cfl_step = settings["cfl"] * dx * dx / float(d.max())
        solver = RK45(rhs, 0., state, target, rtol=settings["rtol"], atol=settings["atol"],
                      max_step=min(settings["max_step"], cfl_step))
        while solver.status == "running":
            d, _ = law(solver.y, solver.t)
            solver.max_step = min(settings["max_step"], settings["cfl"] * dx * dx / float(d.max()))
            message = solver.step()
            if solver.status == "failed":
                status, reason = "solver_failed", str(message)
                break
            audit["accepted_steps"] += 1
            reached_time = float(solver.t)
            track(solver.y, solver.t, True)
            dense = solver.dense_output()
            while next_save < len(save_times) and save_times[next_save] <= solver.t + 1e-13:
                values = dense(float(save_times[next_save]))
                if not np.isfinite(values).all():
                    raise ForwardGuard("nonfinite_dense_output", save_times[next_save])
                track(values, save_times[next_save], True)
                law(values, save_times[next_save])
                times.append(float(save_times[next_save])); fields.append(values)
                next_save += 1
    except ForwardGuard as exc:
        status, reason, audit["guard"] = "guard_failed", str(exc), exc.details
    except (FloatingPointError, OverflowError) as exc:
        status, reason = "arithmetic_failed", str(exc)
        audit["guard"] = dict(reason=str(exc), time=float(solver.t) if solver is not None else 0.)
    if status == "complete" and next_save != len(save_times):
        raise ArithmeticError("Completed solve is missing requested dense-output snapshots")
    info = dict(status=status, reason=reason, reached_time=reached_time, settings=settings, audit=audit,
                seconds=time.perf_counter() - started, state_clipping=False, support_departure_is_failure=False,
                accepted_audit_scope=PROTOCOL["accepted_audit_scope"], diffusion_scope="Queried RHS/accepted-step-CFL/saved states only; not a global U positivity certificate")
    return np.asarray(times), np.asarray(fields), info


def expression_poles(expression):
    """Exact cancellation before floating real-root extraction (degrees <=2)."""
    if expression.family not in ("rational", "saturation", "u_saturation"):
        return []
    import sympy as sp
    u = sp.Symbol("u")
    p = [sp.Rational(float(value)) for value in expression.parameters]
    if expression.family == "rational":
        degree = expression.numerator_degree if expression.numerator_degree is not None else 2
        numerator = sum(value * u**j for j, value in enumerate(p[:degree + 1]))
        denominator = 1 + sum(value * u**j for j, value in enumerate(p[degree + 1:], 1))
    else:
        first, second, rate = p
        denominator = 1 + rate * u
        numerator = first * denominator + second if expression.family == "saturation" else u * (first * denominator + second)
    reduced = sp.cancel(numerator / denominator)
    denominator = sp.Poly(sp.denom(reduced), u)
    if denominator.degree() <= 0:
        return []
    roots = np.roots([float(v) for v in denominator.all_coeffs()])
    return sorted(float(value.real) for value in roots if abs(value.imag) <= 1e-10 * max(1., abs(value.real)))


def scores(times, prediction, truth_times, truth, info):
    rows = []
    for horizon in PROTOCOL["horizons"]:
        at = np.flatnonzero(np.isclose(times, horizon, rtol=0, atol=1e-12))
        truth_at = np.flatnonzero(np.isclose(truth_times, horizon, rtol=0, atol=1e-12))
        reached = bool(len(at))
        row = dict(horizon=horizon, reached=reached, endpoint_error=None, space_time_error=None,
                   endpoint_normalization=None, space_time_normalization=None)
        for kind in ("accepted", "trial"):
            outside = info["audit"]["first_" + kind + "_outside"]
            negative = info["audit"]["first_" + kind + "_negative"]
            row[kind + "_support_valid"] = reached and (outside is None or outside["time"] > horizon + 1e-12)
            row[kind + "_nonnegative"] = reached and (negative is None or negative > horizon + 1e-12)
        if reached:
            j, k = int(at[0]), int(truth_at[0])
            np.testing.assert_allclose(times[:j + 1], truth_times[:k + 1], atol=1e-12, rtol=0)
            error, true = prediction[:j + 1] - truth[:k + 1], truth[:k + 1]
            rms_true = float(np.sqrt(np.mean(true[-1] ** 2)))
            row["endpoint_error"] = float(np.sqrt(np.mean(error[-1] ** 2))) / (rms_true if rms_true > 1e-12 else 1.)
            row["endpoint_normalization"] = "relative_L2" if rms_true > 1e-12 else "absolute_RMS"
            denominator = float(np.trapezoid(np.mean(true ** 2, axis=-1), times[:j + 1]))
            numerator = float(np.trapezoid(np.mean(error ** 2, axis=-1), times[:j + 1]))
            relative = denominator > horizon * 1e-24
            row["space_time_error"] = float(np.sqrt(numerator / (denominator if relative else horizon)))
            row["space_time_normalization"] = "relative_space_time_L2" if relative else "absolute_space_time_RMS"
        rows.append(row)
    return rows


def compare_solutions(primary, refined, truth):
    t, a, _ = primary
    rt, b, _ = refined
    tt, true, _ = truth
    output = []
    for h in PROTOCOL["horizons"]:
        indexes = [np.flatnonzero(np.isclose(q, h, atol=1e-12, rtol=0)) for q in (t, rt, tt)]
        entry = dict(horizon=h, both_reached=all(len(q) for q in indexes), rms_difference=None,
                     difference_relative_to_truth=None, temporal_tolerance_passed=False)
        if entry["both_reached"]:
            i, j, k = [int(q[0]) for q in indexes]
            delta = float(np.sqrt(np.mean((a[i] - b[j]) ** 2)))
            scale = max(float(np.sqrt(np.mean(true[k] ** 2))), 1e-12)
            entry.update(rms_difference=delta, difference_relative_to_truth=delta / scale,
                         temporal_tolerance_passed=delta <= PROTOCOL["numerical_comparison"]["absolute_rms_tolerance"] + PROTOCOL["numerical_comparison"]["relative_to_truth_tolerance"] * scale)
        output.append(entry)
    return output


def plan():
    upstream = json.loads(UPSTREAM_MANIFEST.read_text())
    rows = [task for task in upstream["tasks"] if task["variant"] in PROTOCOL["variants"]]
    assert len(rows) == 100 and all(task["longer"] for task in rows)
    inputs = json.loads((INPUT_ROOT / "INPUT_MANIFEST.json").read_text())
    units = []
    for task in rows:
        for epoch in PROTOCOL["training_checkpoints"]:
            for method in PROTOCOL["methods"]:
                units.append(dict(id=f"{task['id']}__e{epoch}__{method}", kind="model", training_id=task["id"],
                                  case=task["case"], setting=task["setting"], data_seed=task["data_seed"],
                                  variant=task["variant"], epoch=epoch, method=method,
                                  refinement=task["data_seed"] == 0 and task["variant"] in ["default", "drop_strong"]))
    for row in inputs["inputs"]:
        for method in PROTOCOL["baseline_methods"]:
            units.append(dict(id=f"{row['id']}__{method}", kind="baseline", input_id=row["id"], case=row["case"],
                              setting=row["setting"], data_seed=row["data_seed"], method=method, epoch=None, variant=None, refinement=False))
        for variant in PROTOCOL["global_reference_variants"]:
            units.append(dict(id=f"{row['id']}__global_{variant}", kind="global_reference", input_id=row["id"], case=row["case"],
                              setting=row["setting"], data_seed=row["data_seed"], method="global_positive_poly", epoch=None,
                              variant=variant, refinement=False))
    record = dict(study_id=PROTOCOL["study_id"], protocol=PROTOCOL, protocol_sha256=canonical(PROTOCOL),
                  upstream_manifest_sha256=digest(UPSTREAM_MANIFEST), input_manifest_sha256=digest(INPUT_ROOT / "INPUT_MANIFEST.json"),
                  input_root=str(INPUT_ROOT), training_root=str(TRAINING_ROOT),
                  global_bundle_sha256={row["id"]:digest(GLOBAL_ROOT / (row["id"] + ".json")) for row in inputs["inputs"]},
                  driver_sha256=digest(__file__), protocol_source_sha256=digest(STUDY / "protocol.py"),
                  model_units=400, baseline_units=60, global_reference_units=40, units=units, unit_count=len(units),
                  primary_model_integrations=1600, primary_baseline_integrations=240,
                  primary_global_reference_integrations=160,
                  unique_truth_integrations=40, truth_refinement_integrations=40, model_refinement_integrations=128,
                  total_declared_integrations=2208, formal_execution_started=False)
    assert len(units) == 500 and len({unit["id"] for unit in units}) == 500
    # Draft/pilot plans remain recoverable when an explicit new plan is frozen.
    for filename in ("PROTOCOL.json", "MANIFEST.json"):
        previous = STUDY / filename
        if previous.exists():
            archive = STUDY / "design_versions" / (previous.stem + "_" + digest(previous) + ".json")
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                shutil.copy2(previous, archive)
    write(STUDY / "PROTOCOL.json", PROTOCOL)
    write(STUDY / "MANIFEST.json", record)
    print(json.dumps({key:record[key] for key in ("unit_count", "total_declared_integrations", "protocol_sha256")}, indent=2))
    return record


def read_plan():
    record = json.loads((STUDY / "MANIFEST.json").read_text())
    expected = {"protocol_sha256": canonical(PROTOCOL), "upstream_manifest_sha256": digest(UPSTREAM_MANIFEST),
                "input_manifest_sha256": digest(INPUT_ROOT / "INPUT_MANIFEST.json"), "driver_sha256": digest(__file__),
                "protocol_source_sha256": digest(STUDY / "protocol.py")}
    for key, value in expected.items():
        if record[key] != value:
            raise ValueError(f"Frozen forward plan changed: {key}; inspect before explicitly regenerating the plan")
    return record


def truth_path(case, seed, output=OUTPUT):
    return Path(output) / "truth" / f"{case}_s{seed}"


def run_truth(case_name, seed, output=OUTPUT):
    core = load_core(output)
    record = read_plan()
    case = core.extra_case(case_name)
    target = truth_path(case_name, seed, output)
    identity = dict(protocol_sha256=record["protocol_sha256"], driver_sha256=record["driver_sha256"],
                    core_source_signature=core.source_signature(), case=case_name, data_seed=seed)
    if (target / "result.json").exists():
        prior = json.loads((target / "result.json").read_text())
        if prior["identity"] != identity:
            raise ValueError("Truth cache identity changed")
        return prior
    arrays, entries = {}, []
    for label, u0 in initial_fields(case.value_range, seed).items():
        primary = strict_integrate(case.diffusion, case.reaction, u0, case.value_range, PROTOCOL["solver"])
        refined = strict_integrate(case.diffusion, case.reaction, u0, case.value_range, PROTOCOL["refined_solver"])
        comparison = compare_solutions(primary, refined, refined)
        if primary[2]["status"] != "complete" or refined[2]["status"] != "complete" or not all(row["temporal_tolerance_passed"] for row in comparison):
            write(target / "failure.json", dict(identity=identity, pattern=label, primary=primary[2], refined=refined[2], comparison=comparison))
            raise ArithmeticError("Truth failed its declared temporal check; formal model scoring is blocked")
        arrays[label + "_times"], arrays[label + "_fields"] = primary[:2]
        arrays[label + "_refined_fields"] = refined[1]
        entries.append(dict(pattern=label, initial_mean=float(u0.mean()), initial_half_amplitude=float(np.ptp(u0) / 2),
                            initial_sha256=hashlib.sha256(u0.tobytes()).hexdigest(), primary=primary[2], refined=refined[2], temporal_comparison=comparison))
    core.npz_write(target / "fields.npz", **arrays)
    result = dict(identity=identity, status="complete", entries=entries, fields_sha256=digest(target / "fields.npz"))
    write(target / "result.json", result)
    return result


def load_models(core, task, epoch, interval, run_directory=None):
    run = TRAINING_ROOT / "runs" / task["id"] if run_directory is None else Path(run_directory)
    path, curve_path = run / f"checkpoint_{epoch}.pt", run / f"curves_{epoch}.npz"
    saved = core.torch.load(path, weights_only=False, map_location="cpu")
    if saved["epoch"] != epoch or saved["source_signature"] != core.source_signature():
        raise ValueError("Checkpoint epoch/source mismatch")
    if run_directory is None:
        upstream = json.loads(UPSTREAM_MANIFEST.read_text())
        expected = dict(study_id=upstream["study_id"], config_sha256=canonical(task),
                        protocol_sha256=upstream["protocol_sha256"], implementation_signature=upstream["implementation_signature"])
        for key, value in expected.items():
            if saved[key] != value:
                raise ValueError("Checkpoint identity mismatch: " + key)
    model = core.ReactionDiffusionClosure(input_range=interval).cpu()
    model.load_state_dict(saved["model"])
    core.assert_finite_state(model)
    from reliability_core import NumpyMLPClosure
    native = NumpyMLPClosure.from_model(model)
    verification = native.verify(model, interval)
    with np.load(curve_path, allow_pickle=False) as curves:
        original = [curves[branch] for branch in ("D", "R")]
        actual = [native.diffusion(curves["grid"]), native.reaction(curves["grid"])]
        for a, b in zip(actual, original):
            np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-7)
        verification["float32_curve_export_maximum_absolute_difference"] = {key:float(np.max(np.abs(a - b))) for key,a,b in zip(("D","R"),actual,original)}
    snapshot = saved["evaluations"][str(epoch)]
    expressions = [core.SymbolicExpression(**dict(value, parameters=tuple(value["parameters"]))) for value in snapshot["symbols"]]
    methods = {"native": (native.diffusion, native.reaction, []),
               "frozen": (expressions[0].evaluate, expressions[1].evaluate, sorted(set(sum([expression_poles(e) for e in expressions], []))))}
    metadata = dict(checkpoint_sha256=digest(path), curves_sha256=digest(curve_path), observation_signature=saved["observation_signature"],
                    model_verification=verification, frozen_expressions=snapshot["symbols"], native_metadata=native.metadata)
    return methods, metadata


def polynomial(core, method, observed, interval):
    if method == "spacetime_poly":
        closure, mse = core.spacetime_polynomial(observed, interval)
    elif method == "strong_poly":
        closure, mse = core.fit_strong_form_polynomial_closure(observed, value_range=interval)
    elif method == "weak_poly":
        closure, mse = core.fit_weak_form_polynomial_closure(observed, value_range=interval)
    else:
        raise ValueError(method)
    metadata = dict(coefficients_D=closure.diffusion_coefficients, coefficients_R=closure.reaction_coefficients, training_mse=mse,
                    continuous_polynomial_min_D=polynomial_minimum(closure.diffusion_coefficients, interval))
    return (closure.diffusion, closure.reaction, []), metadata


def polynomial_minimum(coefficients, interval):
    coefficients = np.asarray(coefficients, dtype=float)
    candidates = list(interval)
    roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coefficients))
    candidates.extend(float(root.real) for root in roots if abs(root.imag) <= 1e-12 and interval[0] <= root.real <= interval[1])
    values = np.polynomial.polynomial.polyval(candidates, coefficients)
    index = int(np.argmin(values))
    return dict(value=float(values[index]), state=float(candidates[index]), interval=interval,
                method="Endpoints and all real derivative roots, evaluated in floating point")


def global_reference(unit, observed, interval, input_metadata, frozen_plan):
    path = GLOBAL_ROOT / (unit["input_id"] + ".json")
    if digest(path) != frozen_plan["global_bundle_sha256"][unit["input_id"]]:
        raise ValueError("Global coefficient source bundle changed after the forward plan was frozen")
    bundle = json.loads(path.read_text())
    array = np.ascontiguousarray(observed["u"])
    observed_hash = hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()
    if bundle["job"]["input_sha256"] != input_metadata["file_sha256"] or bundle["observed_array_hash"] != observed_hash:
        raise ValueError("Global reference observations differ from the shared NPZ")
    if tuple(bundle["source_metadata"]["value_range"]) != interval:
        raise ValueError("Global reference interval changed")
    matching = [row for row in bundle["results"] if row["variant"] == unit["variant"] and row["constraint"] == "positive_diffusion"
                and not row["oracle"] and row["oracle_arm"] == "observed_features_observed_targets"]
    if len(matching) != 1 or matching[0]["status"] != "passed":
        raise ValueError("Global coefficient fit is missing or did not pass its source KKT checks")
    fit = matching[0]
    if fit["case"] != unit["case"] or fit["data_seed"] != unit["data_seed"] or fit["source_input_id"] != unit["input_id"]:
        raise ValueError("Global reference case/seed/input identity changed")
    coefficients = np.asarray(fit["solver"]["coefficients"], dtype=float)
    if coefficients.shape != (7,) or not np.isfinite(coefficients).all():
        raise ValueError("Invalid global-reference coefficient vector")
    unconstrained = [row for row in bundle["results"] if row["variant"] == unit["variant"] and row["constraint"] == "unconstrained"
                     and not row["oracle"] and row["oracle_arm"] == "observed_features_observed_targets"]
    metadata = dict(source_bundle=str(path), source_bundle_sha256=digest(path), fit=fit,
                    unconstrained_static_diagnostics=unconstrained,
                    continuous_polynomial_min_D=polynomial_minimum(coefficients[:3], interval),
                    scope="Global optimum within the declared seven-coefficient polynomial objective; not a global neural-class optimum")
    return (lambda u:np.polynomial.polynomial.polyval(u, coefficients[:3]),
            lambda u:np.polynomial.polynomial.polyval(u, coefficients[3:]), []), metadata


def run_unit(unit, output=OUTPUT):
    core, frozen_plan = load_core(output), read_plan()
    input_id = unit.get("input_id", f"{unit['case']}_{unit['setting']}_s{unit['data_seed']}")
    observed, interval, input_metadata = load_input(core, input_id)
    if unit["kind"] == "model":
        upstream = json.loads(UPSTREAM_MANIFEST.read_text())
        task = next(task for task in upstream["tasks"] if task["id"] == unit["training_id"])
        methods, metadata = load_models(core, task, unit["epoch"], interval)
        if metadata["observation_signature"] != input_metadata["observation_signature"]:
            raise ValueError("Checkpoint was trained on different observations")
        closure = methods[unit["method"]]
    elif unit["kind"] == "global_reference":
        closure, metadata = global_reference(unit, observed, interval, input_metadata, frozen_plan)
    else:
        closure, metadata = polynomial(core, unit["method"], observed, interval)
    target = Path(output) / "runs" / unit["id"]
    identity = dict(protocol_sha256=frozen_plan["protocol_sha256"], driver_sha256=frozen_plan["driver_sha256"],
                    unit=unit, input_file_sha256=input_metadata["file_sha256"], model_metadata_sha256=canonical(safe(metadata)))
    if (target / "result.json").exists():
        prior = json.loads((target / "result.json").read_text())
        if prior["identity"] != identity:
            raise ValueError("Forward cache identity changed")
        return unit["id"], "cached"
    truth_root = truth_path(unit["case"], unit["data_seed"], output)
    truth_record = json.loads((truth_root / "result.json").read_text())
    if digest(truth_root / "fields.npz") != truth_record["fields_sha256"]:
        raise ValueError("Truth fields hash changed")
    truth_arrays = np.load(truth_root / "fields.npz", allow_pickle=False)
    arrays, entries = {}, []
    for label, u0 in initial_fields(interval, unit["data_seed"]).items():
        true = (truth_arrays[label + "_times"], truth_arrays[label + "_fields"], {"status":"complete"})
        primary = strict_integrate(*closure[:2], u0, interval, PROTOCOL["solver"], poles=closure[2])
        entry = dict(pattern=label, info=primary[2], scores=scores(*primary[:2], *true[:2], primary[2]))
        arrays[label + "_times"], arrays[label + "_fields"] = primary[:2]
        if unit["refinement"]:
            refined = strict_integrate(*closure[:2], u0, interval, PROTOCOL["refined_solver"], poles=closure[2])
            entry["refined_info"] = refined[2]
            entry["temporal_comparison"] = compare_solutions(primary, refined, true)
            arrays[label + "_refined_times"], arrays[label + "_refined_fields"] = refined[:2]
        entries.append(entry)
    truth_arrays.close()
    core.npz_write(target / "fields.npz", **arrays)
    record = dict(identity=identity, status="evaluated", numerical_failure_patterns=sum(row["info"]["status"] != "complete" for row in entries),
                  entries=entries, metadata=metadata, fields_sha256=digest(target / "fields.npz"))
    write(target / "result.json", record)
    return unit["id"], record["numerical_failure_patterns"]


def self_check():
    """Meaningful guard and metric checks, never classified as study evidence."""
    solver = PROTOCOL["solver"]
    zero = lambda u: np.zeros_like(u)
    positive = lambda u: np.full_like(u, .03)
    u0 = np.full(PROTOCOL["nx"], .4)
    constant = strict_integrate(positive, zero, u0, (0., 1.), solver)
    assert constant[2]["status"] == "complete"
    np.testing.assert_allclose(constant[1], .4, atol=1e-14)
    metric = scores(*constant[:2], *constant[:2], constant[2])
    assert all(row["reached"] and row["endpoint_error"] == 0 and row["space_time_error"] == 0 for row in metric)
    nonzero = scores(constant[0], constant[1] + .04, *constant[:2], constant[2])
    np.testing.assert_allclose([[row["endpoint_error"], row["space_time_error"]] for row in nonzero], .1, atol=1e-14)
    absolute = scores(constant[0], np.full_like(constant[1], .04), constant[0], np.zeros_like(constant[1]), constant[2])
    np.testing.assert_allclose([[row["endpoint_error"], row["space_time_error"]] for row in absolute], .04, atol=1e-14)
    assert all(row["endpoint_normalization"] == "absolute_RMS" for row in absolute)
    negative = strict_integrate(lambda u: -positive(u), zero, u0, (0., 1.), solver)
    assert negative[2]["status"] == "guard_failed" and len(negative[0]) == 1
    outward = strict_integrate(positive, lambda u: np.ones_like(u), np.full_like(u0, .95), (0., 1.), solver)
    assert outward[2]["status"] == "complete" and outward[2]["audit"]["first_accepted_outside"] is not None
    assert not scores(*outward[:2], *outward[:2], outward[2])[-1]["accepted_support_valid"]
    pole = strict_integrate(positive, zero, np.linspace(0., 1., PROTOCOL["nx"]), (0., 1.), solver, poles=[.5])
    assert pole[2]["status"] == "guard_failed" and pole[2]["audit"]["guard"]["reason"] == "nonremovable_pole_in_field_enclosure"
    from types import SimpleNamespace
    removable = SimpleNamespace(family="rational", numerator_degree=1, parameters=(-1., 1., -1.))
    nonremovable = SimpleNamespace(family="rational", numerator_degree=1, parameters=(1., 0., -1.))
    assert expression_poles(removable) == [] and expression_poles(nonremovable) == [1.]
    result = dict(classification="driver_guard_tests", status="passed", tests=["constant exact solution", "zero metric", "nonzero endpoint/space-time metrics", "zero-truth absolute fallback", "negative D rejection", "support extrapolation without clipping", "pole-domain rejection", "exact rational pole cancellation"], formal_evidence=False)
    write(OUTPUT / "self_check.json", result)
    print(json.dumps(result, indent=2))


def pilot(launch=None):
    """Bounded CPU benchmark using an existing runtime pilot, not fresh inference."""
    core, frozen_plan = load_core(), read_plan()
    started = time.perf_counter()
    pilot_run = INPUT_ROOT / "runtime-pilot-cuda/runs/runtime_pilot_case_exp_clean_s0_default"
    observed, interval, input_metadata = load_input(core, "case_exp_clean_s0")
    methods, metadata = load_models(core, {"id": pilot_run.name}, 150, interval, pilot_run)
    if metadata["observation_signature"] != input_metadata["observation_signature"]:
        raise ValueError("Runtime pilot observations differ from shared fresh observations")
    case = core.extra_case("case_exp")
    entries = []
    for label, u0 in initial_fields(interval, 0).items():
        truth = strict_integrate(case.diffusion, case.reaction, u0, interval, PROTOCOL["solver"])
        truth_refined = strict_integrate(case.diffusion, case.reaction, u0, interval, PROTOCOL["refined_solver"])
        entries.append(dict(category="truth", pattern=label, primary=truth[2], refined=truth_refined[2], comparison=compare_solutions(truth, truth_refined, truth_refined)))
        for name in ("native", "frozen"):
            closure = methods[name]
            primary = strict_integrate(*closure[:2], u0, interval, PROTOCOL["solver"], poles=closure[2])
            refined = strict_integrate(*closure[:2], u0, interval, PROTOCOL["refined_solver"], poles=closure[2])
            entries.append(dict(category=name, pattern=label, primary=primary[2], refined=refined[2], scores=scores(*primary[:2], *truth[:2], primary[2]), comparison=compare_solutions(primary, refined, truth)))
    # One noisy polynomial reference verifies the independent input-fit interface.
    noisy, noisy_interval, _ = load_input(core, "case_exp_noise3_s0")
    for method in PROTOCOL["baseline_methods"]:
        closure, fit = polynomial(core, method, noisy, noisy_interval)
        u0 = initial_fields(noisy_interval, 0)["low"]
        solution = strict_integrate(*closure[:2], u0, noisy_interval, PROTOCOL["solver"])
        entries.append(dict(category=method, pattern="low", fit=fit, primary=solution[2]))
    report = dict(classification="runtime_pilot", formal_evidence=False, protocol_sha256=frozen_plan["protocol_sha256"],
                  driver_sha256=digest(__file__), protocol_source_sha256=digest(STUDY / "protocol.py"),
                  frozen_plan_sha256=digest(STUDY / "MANIFEST.json"), pilot_checkpoint_metadata=metadata,
                  entries=entries, wall_seconds=time.perf_counter() - started,
                  note="Existing runtime-pilot teacher only; formal680study checkpoints were not selected or scored.")
    report["primary_mean_seconds_by_category"] = {name:float(np.mean([row["primary"]["seconds"] for row in entries if row["category"] == name])) for name in sorted({row["category"] for row in entries})}
    write(OUTPUT / "pilot/PILOT_REPORT.json", report)
    if launch is not None:
        write(launch / "PILOT_REPORT.json", report)
    print(json.dumps({key:report[key] for key in ("wall_seconds", "primary_mean_seconds_by_category")}, indent=2))


def summarize():
    """Export all evaluated units, with missing/failed denominators intact."""
    plan_record = read_plan()
    rows, missing = [], []
    for unit in plan_record["units"]:
        path = OUTPUT / "runs" / unit["id"] / "result.json"
        if not path.exists():
            missing.append(unit["id"])
            continue
        result = json.loads(path.read_text())
        if result["identity"]["protocol_sha256"] != plan_record["protocol_sha256"]:
            raise ValueError("Mixed protocol in forward report")
        for entry in result["entries"]:
            comparisons = {row["horizon"]:row for row in entry.get("temporal_comparison", [])}
            for score in entry["scores"]:
                comparison = comparisons.get(score["horizon"])
                rows.append(dict(unit_id=unit["id"], case=unit["case"], setting=unit["setting"], data_seed=unit["data_seed"],
                                 variant=unit["variant"], epoch=unit["epoch"], method=unit["method"], pattern=entry["pattern"],
                                 numerical_status=entry["info"]["status"], **score,
                                 endpoint_temporal_discrepancy=comparison["difference_relative_to_truth"] if comparison else None,
                                 temporal_tolerance_passed=comparison["temporal_tolerance_passed"] if comparison else None))
    reports = OUTPUT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if rows:
        with (reports / "forward_by_seed.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    group_fields = ("case", "setting", "variant", "epoch", "method", "pattern", "horizon")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    aggregate = []
    for key, group in groups.items():
        assert len({row["data_seed"] for row in group}) == len(group), "Repeated seed in a reported group"
        summary = dict(zip(group_fields, key), planned_seeds=5, evaluated_seeds=len(group),
                       completed=sum(row["reached"] for row in group),
                       accepted_support_valid=sum(row["accepted_support_valid"] for row in group),
                       trial_support_valid=sum(row["trial_support_valid"] for row in group),
                       accepted_nonnegative=sum(row["accepted_nonnegative"] for row in group))
        for metric in ("endpoint_error", "space_time_error"):
            values = [row[metric] for row in group if row["reached"]]
            summary[metric + "_successful_subset_mean"] = float(np.mean(values)) if values else None
            summary[metric + "_successful_subset_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregate.append(summary)
    write(reports / "FORWARD_SUMMARY.json", dict(protocol_sha256=plan_record["protocol_sha256"],
          expected_units=plan_record["unit_count"], evaluated_units=plan_record["unit_count"] - len(missing), missing_units=missing, groups=aggregate,
          inference_rule="Groups contain data seeds, not repeated ICs/checkpoints. Successful-subset errors are accompanied by planned/evaluated/completed/valid denominators.",
          ranking_rule=PROTOCOL["numerical_comparison"]["gap_rule"], formal_launch_or_completion_inferred=False))
    print(json.dumps(dict(evaluated_units=plan_record["unit_count"]-len(missing), missing_units=len(missing), metric_rows=len(rows)), indent=2))


def environment(stage):
    launch = OUTPUT / "launches" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"_{os.getpid()}_{stage}")
    record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv, python=sys.version,
                  executable=sys.executable, platform=platform.platform(), cpu_only=True,
                  driver_sha256=digest(__file__), protocol_sha256=canonical(PROTOCOL), stage=stage,
                  input_root=str(INPUT_ROOT), training_root=str(TRAINING_ROOT))
    write(launch / "runtime.json", record)
    (launch / "git_status.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True))
    packages = sorted(f"{item.metadata['Name']}=={item.version}" for item in importlib.metadata.distributions())
    (launch / "packages.txt").write_text("\n".join(packages) + "\n")
    for filename in ("run_forward.py", "protocol.py", "PROTOCOL.json", "MANIFEST.json"):
        shutil.copy2(STUDY / filename, launch / filename)
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "self-check", "pilot", "truth", "run", "summarize"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--phase", choices=["all", "models", "coefficients", "baselines", "global"], default="all")
    parser.add_argument("--available-only", action="store_true", help="Evaluate available declared checkpoints; missing checkpoints remain pending")
    parser.add_argument("--execute", action="store_true", help="Explicitly launch the reviewed formal computation")
    args = parser.parse_args()
    if args.workers not in (1, 2):
        parser.error("CPU pilot/formal workers must be1or2")
    if args.action == "plan":
        plan(); return
    if args.action == "self-check":
        self_check(); return
    if args.action == "pilot":
        pilot(environment("runtime_pilot")); return
    if args.action == "summarize":
        summarize(); return
    if not args.execute:
        parser.error("Formal execution is disabled until the concrete plan has been reviewed; pass --execute afterward")
    record = read_plan()
    launch = environment(args.action + "_" + args.phase)
    if args.action == "truth":
        try:
            for case in PROTOCOL["cases"]:
                for seed in PROTOCOL["data_seeds"]:
                    run_truth(case, seed)
                    print(json.dumps(dict(truth=case, data_seed=seed, status="complete")), flush=True)
        except Exception as exc:
            write(launch / "RESULT.json", dict(status="failed", error=str(exc), traceback=traceback.format_exc()))
            raise
        write(launch / "RESULT.json", dict(status="complete", truth_units=10, primary_integrations=40, refined_integrations=40))
        return
    units = record["units"]
    kinds = {"all": {"model", "baseline", "global_reference"}, "models": {"model"},
             "coefficients": {"baseline", "global_reference"}, "baselines": {"baseline"}, "global": {"global_reference"}}
    units = [unit for unit in units if unit["kind"] in kinds[args.phase]]
    if args.ids:
        requested = set(args.ids)
        units = [unit for unit in units if unit["id"] in requested]
        if len(units) != len(requested):
            raise ValueError("Unknown requested unit")
    available, missing = [], []
    for unit in units:
        if unit["kind"] == "model":
            path = TRAINING_ROOT / "runs" / unit["training_id"]
            exists = all((path / f"{prefix}_{unit['epoch']}.{suffix}").exists() for prefix,suffix in [("checkpoint","pt"),("curves","npz")])
        else:
            exists = True
        (available if exists else missing).append(unit)
    if missing and not args.available_only:
        raise FileNotFoundError(f"{len(missing)} declared units lack immutable checkpoint/curve files; use --available-only to leave them pending")
    if missing:
        print(json.dumps(dict(pending_declared_units=len(missing), available_units=len(available))), flush=True)
    units = available
    # Spawn avoids importing a revision_core bound to another output root.
    import multiprocessing
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_unit, unit): unit["id"] for unit in units}
        try:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed.append(dict(unit=futures[future], result=result))
                print(json.dumps(completed[-1]), flush=True)
        except Exception as exc:
            for future in futures:
                future.cancel()
            write(launch / "RESULT.json", dict(status="software_failed", completed=completed, error=str(exc), traceback=traceback.format_exc()))
            raise
    write(launch / "RESULT.json", dict(status="complete", evaluated_units=len(completed), pending_declared_units=len(missing), completed=completed))


if __name__ == "__main__":
    main()
