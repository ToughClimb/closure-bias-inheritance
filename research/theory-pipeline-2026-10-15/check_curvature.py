"""Check the full second variation, including the residual-weighted curvature term.

This is a post-review diagnostic of a mathematical omission from the full Taylor
coefficient; it does not select or change any neural/symbolic model.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.integrate import RK45
from scipy.special import expit
import check_tangent as first


class NativeJet(first.NativeLaw):
    def scalar_jet(self, branch, values):
        spec = self.metadata["branches"][branch]
        scale = 2.0 / max(spec["upper"] - spec["lower"], 1e-6)
        x = (values - spec["lower"]) * scale - 1.0
        coefficients = self.arrays[branch + "_coefficients"]
        polynomial = [np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coefficients, m=k)) * scale**k for k in range(3)]
        h, hp, hpp = x[..., None], np.full_like(x[..., None], scale), np.zeros_like(x[..., None])
        for i in range(len(spec["layers"])):
            weight = self.arrays[f"{branch}_weight_{i}"]
            h, hp, hpp = h @ weight.T + self.arrays[f"{branch}_bias_{i}"], hp @ weight.T, hpp @ weight.T
            if i + 1 < len(spec["layers"]):
                h = np.tanh(h)
                hpp = (1.0 - h * h) * (hpp - 2.0 * h * hp * hp)
                hp = (1.0 - h * h) * hp
        return tuple(p + spec["residual_scale"] * value[..., 0] for p, value in zip(polynomial, (h, hp, hpp)))

    def jet(self, values):
        g, gp, gpp = self.scalar_jet("D", values)
        s = 1.0 / (1.0 + np.exp(-g)) if np.iscomplexobj(g) else expit(g)
        scale = np.exp(self.arrays["log_diffusion_scale"])
        d = self.metadata["min_diffusion"] + scale * s
        dp = scale * s * (1.0 - s) * gp
        dpp = scale * s * (1.0 - s) * (gpp + (1.0 - 2.0 * s) * gp * gp)
        r, rp, rpp = self.scalar_jet("R", values)
        return d, r, dp, rp, dpp, rpp


def symbolic_prime(expression, values):
    coefficients = np.asarray(expression["parameters"])
    if expression["family"] == "polynomial":
        return np.polynomial.polynomial.polyval(values, np.polynomial.polynomial.polyder(coefficients))
    if expression["family"] == "rational":
        degree = expression["numerator_degree"] if expression["numerator_degree"] is not None else 2
        numerator, denominator = coefficients[:degree + 1], np.r_[1.0, coefficients[degree + 1:]]
        n, d = np.polynomial.polynomial.polyval(values, numerator), np.polynomial.polynomial.polyval(values, denominator)
        np_, dp = np.polynomial.polynomial.polyval(values, np.polynomial.polynomial.polyder(numerator)), np.polynomial.polynomial.polyval(values, np.polynomial.polynomial.polyder(denominator))
        return (np_ * d - n * dp) / d**2
    raise ValueError("Unexpected declared expression family")


class CurvatureProblem(first.Problem):
    def __init__(self, seed, nx):
        super().__init__(seed, nx)
        self.native = NativeJet(self.weights_path)

    @staticmethod
    def average(values):
        return 0.5 * (values + np.roll(values, -1))

    def jump(self, values):
        return (np.roll(values, -1) - values) / self.dx

    def first_variation(self, u, z, alpha):
        d, r, dp, rp, _, _ = self.native.jet(u)
        sd, sr = self.symbolic(u)
        sdp, srp = tuple(symbolic_prime(e, u) for e in self.expressions)
        cd, cr, cdp, crp = sd - d, sr - r, sdp - dp, srp - rp
        da, dpa, rpa = d + alpha * cd, dp + alpha * cdp, rp + alpha * crp
        flux = self.average(dpa * z) * self.jump(u) + self.average(da) * self.jump(z) + self.average(cd) * self.jump(u)
        return self.divergence(flux) + rpa * z + cr

    def second_variation(self, u, z, v):
        d, r, dp, rp, dpp, rpp = self.native.jet(u)
        sd, sr = self.symbolic(u)
        sdp, srp = tuple(symbolic_prime(e, u) for e in self.expressions)
        cd, cr, cdp, crp = sd - d, sr - r, sdp - dp, srp - rp
        flux = self.average(d) * self.jump(u)
        flux_z = self.average(dp * z) * self.jump(u) + self.average(d) * self.jump(z) + self.average(cd) * self.jump(u)
        flux_v = self.average(dp * v + dpp * z * z) * self.jump(u)
        flux_v += 2.0 * self.average(dp * z) * self.jump(z) + self.average(d) * self.jump(v)
        flux_v += 2.0 * self.average(cdp * z) * self.jump(u) + 2.0 * self.average(cd) * self.jump(z)
        return np.r_[self.divergence(flux) + r,
                     self.divergence(flux_z) + rp * z + cr,
                     self.divergence(flux_v) + rp * v + rpp * z * z + 2.0 * crp * z]


def checks():
    maximum = {"D_second_derivative_abs": 0.0, "R_second_derivative_abs": 0.0, "semidiscrete_second_variation_abs": 0.0}
    for seed in range(5):
        problem = CurvatureProblem(seed, 64)
        grid = np.linspace(*problem.interval, 257)
        d, r, dp, rp, dpp, rpp = problem.native.jet(grid)
        eps = 1e-30
        complex_jet = problem.native.jet(grid.astype(complex) + 1j * eps)
        for label, computed, expected in [("D", dpp, complex_jet[2].imag / eps), ("R", rpp, complex_jet[3].imag / eps)]:
            np.testing.assert_allclose(computed, expected, rtol=2e-11, atol=2e-12)
            maximum[label + "_second_derivative_abs"] = max(maximum[label + "_second_derivative_abs"], float(np.max(np.abs(computed - expected))))
        u = problem.initial("midlow")
        phase = 2 * np.pi * np.arange(64) / 64.0
        z, v = 0.1 * np.cos(phase + 0.3), 0.03 * np.sin(phase + 0.2)
        actual = problem.second_variation(u, z, v)[128:]
        expected = problem.first_variation(u.astype(complex) + 1j * eps * z, z.astype(complex) + 1j * eps * v, 1j * eps).imag / eps
        np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-12)
        maximum["semidiscrete_second_variation_abs"] = max(maximum["semidiscrete_second_variation_abs"], float(np.max(np.abs(actual - expected))))
    return dict(status="passed", seeds=5, complex_step=1e-30, **maximum)


def integrate(problem, initial):
    nx = problem.nx
    def rhs(t, state):
        u, z, v = state[:nx], state[nx:2 * nx], state[2 * nx:]
        d, r = problem.native.values(u)
        if not np.isfinite(state).all() or np.max(np.abs(u)) > 10.0 or np.min(d) <= 0.0 or not np.isfinite(r).all():
            raise FloatingPointError("Invalid curvature integration state")
        return problem.second_variation(u, z, v)
    state = np.r_[initial, np.zeros(2 * nx)]
    d, _ = problem.native.values(initial)
    solver = RK45(rhs, 0.0, state, 0.6, rtol=1e-6, atol=1e-9,
                  max_step=min(0.01, 0.35 * problem.dx**2 / float(d.max())))
    fields, index = [state], 1
    while solver.status == "running":
        d, _ = problem.native.values(solver.y[:nx])
        solver.max_step = min(0.01, 0.35 * problem.dx**2 / float(d.max()))
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError(message)
        dense = solver.dense_output()
        while index < len(first.TIMES) and first.TIMES[index] <= solver.t + 1e-13:
            fields.append(dense(float(first.TIMES[index])))
            index += 1
    assert index == len(first.TIMES)
    return np.asarray(fields)


def run_one(job):
    seed, pattern, nx, mode, output = job
    output = Path(output)
    key = f"s{seed}_{pattern}_nx{nx}"
    start = time.perf_counter()
    try:
        problem = CurvatureProblem(seed, nx)
        fields = integrate(problem, problem.initial(pattern))
        reference_path = first.AUDIT / f"{mode}_nx{nx}" / (key + ".npz")
        with np.load(reference_path, allow_pickle=False) as values:
            reference = {k: values[k] for k in values.files}
        u, z, v = fields[:, :nx], fields[:, nx:2 * nx], fields[:, 2 * nx:]
        max_difference = float(np.max(np.abs(u - reference["native"])))
        np.testing.assert_allclose(u, reference["native"], atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(z, reference["tangent"], atol=2e-11, rtol=2e-10)
        rows = []
        for alpha in first.AMPLITUDES:
            actual = reference[f"alpha_{alpha:g}"]
            for horizon in first.HORIZONS:
                i = int(round(horizon / 0.01))
                scale = np.linalg.norm(reference["truth"][i])
                r, zz, vv = (u[i] - reference["truth"][i]) / scale, z[i] / scale, v[i] / scale
                actual_error = (actual[i] - reference["truth"][i]) / scale
                actual_change = float(actual_error @ actual_error - r @ r)
                B, A, curvature = float(2 * r @ zz), float(zz @ zz), float(r @ vv)
                gn = alpha * B + alpha**2 * A
                taylor = alpha * B + alpha**2 * (A + curvature)
                displacement = alpha * zz + 0.5 * alpha**2 * vv
                squared_second_trajectory = float(2 * r @ displacement + displacement @ displacement)
                remainder = float(np.linalg.norm((actual[i] - u[i] - alpha * z[i] - 0.5 * alpha**2 * v[i]) / scale))
                bound = float(2 * (np.linalg.norm(r) + np.linalg.norm(displacement)) * remainder + remainder**2)
                assert abs(actual_change - squared_second_trajectory) <= bound + 2e-16
                rows.append(dict(seed=seed, pattern=pattern, nx=nx, horizon=horizon, alpha=alpha,
                                 actual_change=actual_change, gauss_newton_prediction=gn,
                                 full_taylor_prediction=taylor, squared_second_trajectory_prediction=squared_second_trajectory,
                                 curvature_coefficient=curvature,
                                 gn_agrees=bool(np.sign(gn) == np.sign(actual_change)),
                                 full_taylor_agrees=bool(np.sign(taylor) == np.sign(actual_change)),
                                 second_trajectory_agrees=bool(np.sign(squared_second_trajectory) == np.sign(actual_change)),
                                 measured_third_order_remainder=remainder, measured_second_trajectory_bound=bound))
        orders = []
        for horizon in first.HORIZONS:
            cells = {r["alpha"]: r for r in rows if r["horizon"] == horizon}
            for alpha in (0.25, 0.5, 1.0):
                orders.append(dict(seed=seed, pattern=pattern, nx=nx, horizon=horizon, alpha=alpha,
                                   order=float(np.log2(cells[alpha]["measured_third_order_remainder"] / cells[alpha / 2]["measured_third_order_remainder"]))))
        np.savez_compressed(output / (key + ".npz"), times=first.TIMES, native=u, tangent=z, curvature=v)
        result = dict(status="complete", seed=seed, pattern=pattern, nx=nx, rows=rows, orders=orders,
                      reference_sha256=first.sha256(reference_path), maximum_native_array_difference=max_difference,
                      driver_sha256=first.sha256(__file__), elapsed_seconds=time.perf_counter() - start)
        first.write_json(output / (key + ".json"), result)
        return result
    except Exception:
        first.write_json(output / (key + ".FAILED.json"), dict(status="failed", traceback=traceback.format_exc(), job=list(job)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("formal", "refinement"), required=True)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = first.AUDIT / f"curvature_{args.mode}_nx{args.nx}"
    output.mkdir(exist_ok=True)
    assert not (output / "SUMMARY.json").exists(), "Preserve completed curvature audit"
    first.write_json(output / "DERIVATIVE_CHECKS.json", checks())
    cells = [(s, p) for s in range(5) for p in first.PATTERNS] if args.mode == "formal" else [(0, "low"), (3, "high")]
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, len(cells))) as pool:
        futures = [pool.submit(run_one, (s, p, args.nx, args.mode, str(output))) for s, p in cells]
        for future in concurrent.futures.as_completed(futures):
            result = future.result(); completed.append(result)
            print(json.dumps(dict(done=len(completed), total=len(cells), seed=result["seed"], pattern=result["pattern"], seconds=result["elapsed_seconds"])), flush=True)
    rows = [r for result in completed for r in result["rows"]]
    orders = [r for result in completed for r in result["orders"]]
    for filename, table in [("endpoint_checks.csv", rows), ("remainder_orders.csv", orders)]:
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    full = [r for r in rows if r["alpha"] == 1.0]
    summary = dict(status="complete", mode=args.mode, nx=args.nx, checked_utc=datetime.now(timezone.utc).isoformat(),
                   driver_sha256=first.sha256(__file__), endpoint_pairs=len(full),
                   gauss_newton_agreements=sum(r["gn_agrees"] for r in full),
                   full_taylor_agreements=sum(r["full_taylor_agrees"] for r in full),
                   second_trajectory_agreements=sum(r["second_trajectory_agrees"] for r in full),
                   halving_order_min=min(r["order"] for r in orders),
                   halving_order_median=float(np.median([r["order"] for r in orders])),
                   halving_order_max=max(r["order"] for r in orders),
                   critical_case=[r for r in full if r["seed"] == 3 and r["pattern"] == "high" and r["horizon"] == 0.6],
                   scope="Post-review second-variation diagnostic on the same saved five-seed cohort; not independent population-level predictive validation.")
    first.write_json(output / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
