"""Independently verify the tangent/homotopy claims in the returned theory insert.

Uses the immutable handoff's actual native weights and exact frozen formulas.
No training, expression fitting, clipping, or truth-dependent model selection.
"""
from __future__ import annotations

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.integrate import RK45
from scipy.special import expit

WORKSPACE = Path(__file__).resolve().parents[2]
PACKAGE = WORKSPACE / "reproducibility/theory/inputs"
AUDIT = WORKSPACE / "outputs/theory-variations"
TIMES = np.linspace(0.0, 0.6, 61)
AMPLITUDES = (0.125, 0.25, 0.5, 1.0)
HORIZONS = (0.1, 0.3, 0.6)
PATTERNS = {"low": (0.05, 0.04), "midlow": (0.30, 0.20),
            "midhigh": (0.65, 0.20), "high": (0.90, 0.08)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeLaw:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            self.arrays = {k: np.asarray(data[k], dtype=float) for k in data.files if k != "metadata"}
        assert self.metadata["backbone"] == "mlp"

    def scalar(self, branch, values, derivative=False):
        spec = self.metadata["branches"][branch]
        factor = 2.0 / max(spec["upper"] - spec["lower"], 1e-6)
        scaled = (values - spec["lower"]) * factor - 1.0
        coefficients = self.arrays[branch + "_coefficients"]
        polynomial = np.polynomial.polynomial.polyval(scaled, coefficients)
        hidden = scaled[..., None]
        if derivative:
            hidden_prime = np.full_like(hidden, factor)
        for i in range(len(spec["layers"])):
            weight = self.arrays[f"{branch}_weight_{i}"]
            hidden = hidden @ weight.T + self.arrays[f"{branch}_bias_{i}"]
            if derivative:
                hidden_prime = hidden_prime @ weight.T
            if i + 1 < len(spec["layers"]):
                hidden = np.tanh(hidden)
                if derivative:
                    hidden_prime *= 1.0 - hidden * hidden
        result = polynomial + spec["residual_scale"] * hidden[..., 0]
        if not derivative:
            return result
        polynomial_prime = factor * np.polynomial.polynomial.polyval(
            scaled, np.polynomial.polynomial.polyder(coefficients))
        return result, polynomial_prime + spec["residual_scale"] * hidden_prime[..., 0]

    def values(self, values):
        logits = self.scalar("D", values)
        sigmoid = 1.0 / (1.0 + np.exp(-logits)) if np.iscomplexobj(logits) else expit(logits)
        diffusion = self.metadata["min_diffusion"] + np.exp(self.arrays["log_diffusion_scale"]) * sigmoid
        return diffusion, self.scalar("R", values)

    def values_derivatives(self, values):
        logits, logits_prime = self.scalar("D", values, derivative=True)
        sigmoid = expit(logits)
        scale = np.exp(self.arrays["log_diffusion_scale"])
        d = self.metadata["min_diffusion"] + scale * sigmoid
        dp = scale * sigmoid * (1.0 - sigmoid) * logits_prime
        r, rp = self.scalar("R", values, derivative=True)
        return d, r, dp, rp


def symbolic_value(expression, values):
    p = np.asarray(expression["parameters"])
    if expression["family"] == "polynomial":
        return np.polynomial.polynomial.polyval(values, p)
    if expression["family"] == "rational":
        degree = expression["numerator_degree"] if expression["numerator_degree"] is not None else 2
        numerator = np.polynomial.polynomial.polyval(values, p[:degree + 1])
        denominator = np.polynomial.polynomial.polyval(values, np.r_[1.0, p[degree + 1:]])
        return numerator / denominator
    raise ValueError("Unexpected expression family in the declared A/clean cohort")


class Problem:
    def __init__(self, seed, nx):
        self.seed, self.nx = seed, nx
        name = f"gpu_case_a_clean_s{seed}_default_e1500"
        self.weights_path = PACKAGE / "data/native_weights_gpu" / (name + ".npz")
        self.metadata_path = self.weights_path.with_suffix(".json")
        self.metadata = json.loads(self.metadata_path.read_text())
        assert sha256(self.weights_path) == self.metadata["export_sha256"]
        assert self.metadata["identity"]["training_id"] == name.removesuffix("_e1500")
        self.native = NativeLaw(self.weights_path)
        self.expressions = self.metadata["frozen_expressions"]
        self.interval = self.metadata["value_range"]
        self.dx = 1.0 / nx

    def symbolic(self, values):
        return tuple(symbolic_value(e, values) for e in self.expressions)

    def coefficients(self, values, alpha):
        d, r = self.native.values(values)
        sd, sr = self.symbolic(values)
        return d + alpha * (sd - d), r + alpha * (sr - r)

    def initial(self, pattern):
        # The normalization is always the ORIGINAL nx64 one, even on refined grids.
        # Consequently these are samples of one fixed continuous initial field.
        phase = np.random.default_rng(self.seed + 20261015).uniform(0.0, 2 * np.pi, 2)
        def shape(x):
            return np.sin(2 * np.pi * x + phase[0]) + 0.25 * np.sin(6 * np.pi * x + phase[1])
        normalization = np.max(np.abs(shape(np.arange(64) / 64.0)))
        mean_fraction, amplitude_fraction = PATTERNS[pattern]
        lo, hi = self.interval
        return lo + mean_fraction * (hi - lo) + amplitude_fraction * (hi - lo) * shape(np.arange(self.nx) / self.nx) / normalization

    def divergence(self, face_flux):
        return (face_flux - np.roll(face_flux, 1)) / self.dx

    def rhs_values(self, values, alpha):
        d, r = self.coefficients(values, alpha)
        flux = 0.5 * (d + np.roll(d, -1)) * (np.roll(values, -1) - values) / self.dx
        return self.divergence(flux) + r

    def tangent_values(self, values, tangent):
        d, r, dp, rp = self.native.values_derivatives(values)
        sd, sr = self.symbolic(values)
        jump = (np.roll(values, -1) - values) / self.dx
        flux = 0.5 * (d + np.roll(d, -1)) * jump
        dstate = dp * tangent
        tangent_flux = 0.5 * (dstate + np.roll(dstate, -1)) * jump
        tangent_flux += 0.5 * (d + np.roll(d, -1)) * (np.roll(tangent, -1) - tangent) / self.dx
        cd = sd - d
        tangent_flux += 0.5 * (cd + np.roll(cd, -1)) * jump
        return np.r_[self.divergence(flux) + r,
                     self.divergence(tangent_flux) + rp * tangent + sr - r]


def integrate(problem, initial, kind, alpha=0.0, refined_time=False):
    """Original accepted-step diffusion CFL and dense-output sampling."""
    nx, dx = problem.nx, problem.dx
    state = np.r_[initial, np.zeros(nx)] if kind == "tangent" else initial.copy()
    statistics = {"rhs_calls": 0, "accepted_steps": 0, "state_min": float(initial.min()),
                  "state_max": float(initial.max()), "minimum_diffusion": None}

    def law(values):
        if kind == "truth":
            return 0.01 + 0.05 * values, values * (1.0 - values)
        if kind == "tangent":
            return problem.native.values(values)
        return problem.coefficients(values, alpha)

    def check(values):
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 10.0:
            raise FloatingPointError("Nonfinite or guarded state")
        d, r = law(values)
        if not np.isfinite(d).all() or not np.isfinite(r).all() or np.min(d) <= 0.0:
            raise FloatingPointError("Invalid constitutive query")
        statistics["state_min"] = min(statistics["state_min"], float(values.min()))
        statistics["state_max"] = max(statistics["state_max"], float(values.max()))
        minimum = float(np.min(d))
        statistics["minimum_diffusion"] = minimum if statistics["minimum_diffusion"] is None else min(minimum, statistics["minimum_diffusion"])
        return d, r

    def rhs(t, combined):
        statistics["rhs_calls"] += 1
        if statistics["rhs_calls"] > 600000:
            raise RuntimeError("RHS evaluation budget exceeded")
        values = combined[:nx]
        d, r = check(values)
        if kind == "tangent":
            result = problem.tangent_values(values, combined[nx:])
        else:
            flux = 0.5 * (d + np.roll(d, -1)) * (np.roll(values, -1) - values) / dx
            result = problem.divergence(flux) + r
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite RHS")
        return result

    cfl, max_step = (0.175, 0.005) if refined_time else (0.35, 0.01)
    rtol, atol = (1e-7, 1e-10) if refined_time else (1e-6, 1e-9)
    d, _ = check(initial)
    solver = RK45(rhs, 0.0, state, 0.6, rtol=rtol, atol=atol,
                  max_step=min(max_step, cfl * dx * dx / float(d.max())))
    fields, next_index = [state.copy()], 1
    while solver.status == "running":
        d, _ = check(solver.y[:nx])
        solver.max_step = min(max_step, cfl * dx * dx / float(d.max()))
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError("RK45 failed: " + str(message))
        statistics["accepted_steps"] += 1
        check(solver.y[:nx])
        dense = solver.dense_output()
        while next_index < len(TIMES) and TIMES[next_index] <= solver.t + 1e-13:
            value = dense(float(TIMES[next_index]))
            check(value[:nx])
            fields.append(value)
            next_index += 1
    assert next_index == len(TIMES)
    return np.asarray(fields), statistics


def derivative_checks():
    spec = importlib.util.spec_from_file_location("provided_native", PACKAGE / "tools/native_numpy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    errors = {"native_function_max_abs": 0.0, "D_derivative_max_abs": 0.0,
              "R_derivative_max_abs": 0.0, "semidiscrete_tangent_max_abs": 0.0}
    for seed in range(5):
        problem = Problem(seed, 64)
        supplied = module.NumpyMLPClosure.load(problem.weights_path)
        grid = np.linspace(*problem.interval, 257)
        d, r, dp, rp = problem.native.values_derivatives(grid)
        for actual, expected in [(d, supplied.diffusion(grid)), (r, supplied.reaction(grid))]:
            np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=3e-15)
            errors["native_function_max_abs"] = max(errors["native_function_max_abs"], float(np.max(np.abs(actual - expected))))
        eps = 1e-30
        complex_d, complex_r = problem.native.values(grid.astype(complex) + 1j * eps)
        for label, actual, expected in [("D", dp, complex_d.imag / eps), ("R", rp, complex_r.imag / eps)]:
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
            errors[label + "_derivative_max_abs"] = max(errors[label + "_derivative_max_abs"], float(np.max(np.abs(actual - expected))))
        state = problem.initial("midlow")
        direction = 0.1 * np.cos(2 * np.pi * np.arange(64) / 64.0 + 0.3)
        analytic = problem.tangent_values(state, direction)[64:]
        complex_tangent = problem.rhs_values(state.astype(complex) + 1j * eps * direction, 1j * eps).imag / eps
        np.testing.assert_allclose(analytic, complex_tangent, rtol=2e-10, atol=2e-12)
        errors["semidiscrete_tangent_max_abs"] = max(errors["semidiscrete_tangent_max_abs"], float(np.max(np.abs(analytic - complex_tangent))))
    return {"status": "passed", "seeds": 5, "complex_step": 1e-30, **errors}


def run_one(job):
    seed, pattern, nx, output, refined_time = job
    output = Path(output)
    key = f"s{seed}_{pattern}_nx{nx}" + ("_timefine" if refined_time else "")
    started = time.perf_counter()
    try:
        problem = Problem(seed, nx)
        initial = problem.initial(pattern)
        truth, truth_info = integrate(problem, initial, "truth", refined_time=refined_time)
        tangent, tangent_info = integrate(problem, initial, "tangent", refined_time=refined_time)
        native, response = tangent[:, :nx], tangent[:, nx:]
        fields = {"times": TIMES, "truth": truth, "native": native, "tangent": response}
        rows, orders, integrations = [], [], {"truth": truth_info, "tangent": tangent_info}
        for alpha in AMPLITUDES:
            values, info = integrate(problem, initial, "homotopy", alpha, refined_time=refined_time)
            fields[f"alpha_{alpha:g}"] = values
            integrations[f"alpha_{alpha:g}"] = info
            for horizon in HORIZONS:
                i = int(round(horizon / 0.01))
                normalization = np.linalg.norm(truth[i])
                assert normalization > 1e-12
                residual = (native[i] - truth[i]) / normalization
                z = response[i] / normalization
                actual_error = (values[i] - truth[i]) / normalization
                remainder = (values[i] - native[i] - alpha * response[i]) / normalization
                A, B = float(z @ z), float(2 * (residual @ z))
                Q = alpha * B + alpha * alpha * A
                actual_change = float(actual_error @ actual_error - residual @ residual)
                delta = float(np.linalg.norm(remainder))
                bound = float(2 * (np.linalg.norm(residual) + alpha * np.linalg.norm(z)) * delta + delta * delta)
                assert abs(actual_change - Q) <= bound + 2e-16
                displacement = float(np.linalg.norm((values[i] - native[i]) / normalization))
                rows.append(dict(seed=seed, pattern=pattern, nx=nx, refined_time=refined_time,
                                 horizon=horizon, alpha=alpha, A=A, B=B, predicted_change=Q,
                                 actual_change=actual_change, prediction_agrees=bool(np.sign(Q) == np.sign(actual_change)),
                                 measured_remainder=delta, measured_remainder_bound=bound,
                                 margin_exceeds_measured_bound=bool(abs(Q) > bound),
                                 remainder_relative_to_displacement=delta / displacement,
                                 remainder_relative_to_linear_displacement=delta / (alpha * np.linalg.norm(z)),
                                 native_error=float(np.linalg.norm(residual)), actual_error=float(np.linalg.norm(actual_error)),
                                 predicted_positive_crossing=-B / A if A > 0.0 and B < 0.0 else None))
        for horizon in HORIZONS:
            cells = {row["alpha"]: row for row in rows if row["horizon"] == horizon}
            for alpha in (0.25, 0.5, 1.0):
                order = float(np.log2(cells[alpha]["measured_remainder"] / cells[alpha / 2]["measured_remainder"]))
                orders.append(dict(seed=seed, pattern=pattern, nx=nx, refined_time=refined_time,
                                   horizon=horizon, alpha=alpha, halving_order=order))
        np.savez_compressed(output / (key + ".npz"), **fields)
        result = dict(status="complete", key=key, seed=seed, pattern=pattern, nx=nx,
                      refined_time=refined_time, rows=rows, orders=orders, integrations=integrations,
                      weights_sha256=sha256(problem.weights_path), metadata_sha256=sha256(problem.metadata_path),
                      driver_sha256=sha256(__file__), elapsed_seconds=time.perf_counter() - started)
        write_json(output / (key + ".json"), result)
        return result
    except Exception:
        write_json(output / (key + ".FAILED.json"), dict(status="failed", traceback=traceback.format_exc(), job=list(job)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal", "refinement"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--refined-time", action="store_true")
    args = parser.parse_args()
    output = AUDIT / (args.mode + f"_nx{args.nx}" + ("_timefine" if args.refined_time else ""))
    output.mkdir(exist_ok=True)
    assert not (output / "SUMMARY.json").exists(), "Preserve completed audit output"
    write_json(output / "DERIVATIVE_CHECKS.json", derivative_checks())
    jobs = [(0, "low")] if args.mode == "pilot" else ([(s, p) for s in range(5) for p in PATTERNS] if args.mode == "formal" else [(0, "low"), (3, "high")])
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [pool.submit(run_one, (s, p, args.nx, str(output), args.refined_time)) for s, p in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed.append(result)
            print(json.dumps({"done": len(completed), "total": len(jobs), "key": result["key"],
                              "seconds": result["elapsed_seconds"]}), flush=True)
    rows = [row for result in completed for row in result["rows"]]
    orders = [row for result in completed for row in result["orders"]]
    for name, table in [("endpoint_checks.csv", rows), ("remainder_orders.csv", orders)]:
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(sorted(table, key=lambda r: (r["seed"], r["pattern"], r["horizon"], r["alpha"])))
    full = [row for row in rows if row["alpha"] == 1.0]
    upstream = {}
    with (PACKAGE / "data/forward_gpu/forward_by_seed.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case"] == "case_a" and row["setting"] == "clean" and row["variant"] == "default" and row["epoch"] == "1500" and row["method"] in ("native", "frozen"):
                upstream[(int(row["data_seed"]), row["pattern"], float(row["horizon"]), row["method"])] = float(row["endpoint_error"])
    discrepancies = [abs(row[field] - upstream[row["seed"], row["pattern"], row["horizon"], method])
                     for row in full for field, method in [("native_error", "native"), ("actual_error", "frozen")]]
    summary = dict(status="complete", mode=args.mode, nx=args.nx, refined_time=args.refined_time,
                   checked_utc=datetime.now(timezone.utc).isoformat(), driver_sha256=sha256(__file__),
                   seeds=sorted({row["seed"] for row in full}), jobs=len(completed), endpoint_pairs=len(full),
                   sign_agreements=sum(row["prediction_agrees"] for row in full),
                   disagreements=[row for row in full if not row["prediction_agrees"]],
                   per_seed=[dict(seed=s, pairs=sum(r["seed"] == s for r in full), agrees=sum(r["seed"] == s and r["prediction_agrees"] for r in full)) for s in sorted({r["seed"] for r in full})],
                   halving_order_min=float(min(row["halving_order"] for row in orders)),
                   halving_order_median=float(np.median([row["halving_order"] for row in orders])),
                   halving_order_max=float(max(row["halving_order"] for row in orders)),
                   maximum_remainder_relative_to_displacement=max(row["remainder_relative_to_displacement"] for row in rows),
                   maximum_remainder_relative_to_linear_displacement=max(row["remainder_relative_to_linear_displacement"] for row in rows),
                   comparisons_with_margin_exceeding_measured_bound=sum(row["margin_exceeds_measured_bound"] for row in full),
                   maximum_endpoint_error_difference_from_archived_nx64_metrics=max(discrepancies),
                   limitations="IC/horizon/amplitude rows repeat five data seeds. Measured remainder bound is a posteriori. Spatial refinements compare different discretizations, not archived-metric reproduction. No teachers retrained.")
    write_json(output / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
