"""Residual-based response enclosures for the actual frozen nonlinear expressions.

No symbolic rollout, truth array, or training loss is read by ``enclose``.
The exact-arithmetic bounds are proved in the accompanying TeX. This implementation
uses padded float64 interval arithmetic; it is not a machine-verified proof.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

WORK = Path(__file__).resolve().parents[2]
DATA = WORK / "reproducibility/theory"
AUDIT = DATA / "inputs/variations"
REFERENCE = DATA / "expected"
OUT = WORK / "outputs/theory-pipeline-2026-10-15"
import check_curvature as old

ARITHMETIC_FLOOR = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class I:
    """Array intervals, padded after each operation; no hidden sampled maxima."""
    __array_priority__ = 1000

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = self.lo if hi is None else np.asarray(hi, dtype=float)
        assert np.all(self.lo <= self.hi)

    @staticmethod
    def padded(lo, hi):
        pad = 32 * np.finfo(float).eps * (1 + np.maximum(np.abs(lo), np.abs(hi)))
        return I(np.nextafter(lo - pad, -np.inf), np.nextafter(hi + pad, np.inf))

    @staticmethod
    def cast(x):
        return x if isinstance(x, I) else I(x)

    def __add__(self, other):
        other = I.cast(other)
        return I.padded(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-I.cast(other))

    def __rsub__(self, other):
        return I.cast(other) - self

    def __mul__(self, other):
        other = I.cast(other)
        values = np.broadcast_arrays(self.lo * other.lo, self.lo * other.hi,
                                     self.hi * other.lo, self.hi * other.hi)
        return I.padded(np.minimum.reduce(values), np.maximum.reduce(values))

    __rmul__ = __mul__

    def reciprocal(self):
        if np.any((self.lo <= 0) & (self.hi >= 0)):
            raise FloatingPointError("Interval denominator contains zero")
        return I.padded(1 / self.hi, 1 / self.lo)

    def __truediv__(self, other):
        return self * I.cast(other).reciprocal()

    def __rtruediv__(self, other):
        return I.cast(other) / self

    def square(self):
        lower = np.minimum(self.lo**2, self.hi**2)
        lower = np.where((self.lo <= 0) & (self.hi >= 0), 0, lower)
        return I.padded(lower, np.maximum(self.lo**2, self.hi**2))

    def tanh(self):
        return I.padded(np.tanh(self.lo), np.tanh(self.hi))

    def exp(self):
        return I.padded(np.exp(self.lo), np.exp(self.hi))

    def affine(self, weight, bias=0):
        positive, negative = np.maximum(weight, 0), np.minimum(weight, 0)
        lo = self.lo @ positive.T + self.hi @ negative.T + bias
        hi = self.hi @ positive.T + self.lo @ negative.T + bias
        magnitude = np.maximum(np.abs(self.lo), np.abs(self.hi)) @ np.abs(weight).T
        # Standard dot-product error majorant, in addition to interval padding.
        pad = 8 * weight.shape[1] * np.finfo(float).eps * (1 + magnitude + np.abs(bias))
        return I.padded(lo - pad, hi + pad)

    def absmax(self):
        return np.maximum(np.abs(self.lo), np.abs(self.hi))


def polyval(coefficients, x):
    value = I(0)
    for coefficient in np.asarray(coefficients)[::-1]:
        value = value * x + coefficient
    return value


def symbolic_jet(expression, x):
    p = np.asarray(expression["parameters"])
    if expression["family"] == "polynomial":
        return tuple(polyval(np.polynomial.polynomial.polyder(p, m=i), x) for i in range(3))
    if expression["family"] == "rational":
        degree = expression.get("numerator_degree")
        degree = 2 if degree is None else degree
        n, d = p[:degree + 1], np.r_[1.0, p[degree + 1:]]
        n0, n1, n2 = (polyval(np.polynomial.polynomial.polyder(n, m=i), x) for i in range(3))
        d0, d1, d2 = (polyval(np.polynomial.polynomial.polyder(d, m=i), x) for i in range(3))
        r0 = n0 / d0
        r1 = (n1 - r0 * d1) / d0
        r2 = (n2 - 2 * r1 * d1 - r0 * d2) / d0
        return r0, r1, r2
    raise ValueError(expression["family"])


def native_jet(law, x):
    result = {}
    for branch in ("D", "R"):
        spec = law.metadata["branches"][branch]
        scale = 2 / max(spec["upper"] - spec["lower"], 1e-6)
        xx = (x - spec["lower"]) * scale - 1
        coefficients = law.arrays[branch + "_coefficients"]
        poly = [polyval(np.polynomial.polynomial.polyder(coefficients, m=i), xx) * scale**i for i in range(3)]
        h = I(xx.lo[..., None], xx.hi[..., None])
        hp, hpp = I(np.ones_like(h.lo) * scale), I(np.zeros_like(h.lo))
        for i in range(len(spec["layers"])):
            weight = law.arrays[f"{branch}_weight_{i}"]
            h = h.affine(weight, law.arrays[f"{branch}_bias_{i}"])
            hp, hpp = hp.affine(weight), hpp.affine(weight)
            if i + 1 < len(spec["layers"]):
                h = h.tanh()
                hpp = (1 - h.square()) * (hpp - 2 * h * hp.square())
                hp = (1 - h.square()) * hp
        values = [poly[i] + spec["residual_scale"] * I(q.lo[..., 0], q.hi[..., 0]) for i, q in enumerate((h, hp, hpp))]
        if branch == "D":
            g, gp, gpp = values
            sigmoid = 1 / (1 + (-g).exp())
            ds = float(np.exp(law.arrays["log_diffusion_scale"]))
            result[branch] = (law.metadata["min_diffusion"] + ds * sigmoid,
                              ds * sigmoid * (1 - sigmoid) * gp,
                              ds * sigmoid * (1 - sigmoid) * (gpp + (1 - 2 * sigmoid) * gp.square()))
        else:
            result[branch] = tuple(values)
    return result["D"][0], result["R"][0], result["D"][1], result["R"][1], result["D"][2], result["R"][2]


def derivative_constants(problem, mode):
    # The preset guard contains the entire tube; it is not a visited-state sample.
    edges = np.linspace(-0.1, 1.5, 1601)
    interval = I(edges[:-1], edges[1:])
    if mode == "native":
        jets = native_jet(problem.native, interval)
    else:
        d, r = (symbolic_jet(e, interval) for e in problem.expressions)
        jets = d[0], r[0], d[1], r[1], d[2], r[2]
    constants = {key: float(np.max(jet.absmax())) for key, jet in zip(("D0", "R0", "D1", "R1", "D2", "R2"), jets)}
    constants.update(lower=-0.1, upper=1.5, interval_cells=1600)
    # Independent interior probes verify the interval implementation, not its proof.
    points = edges[:-1] + np.diff(edges) * 0.371
    if mode == "native":
        point_jets = problem.native.jet(points)
    else:
        point_jets = tuple(q.lo for q in (symbolic_jet(problem.expressions[0], I(points))[0],
                                         symbolic_jet(problem.expressions[1], I(points))[0],
                                         symbolic_jet(problem.expressions[0], I(points))[1],
                                         symbolic_jet(problem.expressions[1], I(points))[1],
                                         symbolic_jet(problem.expressions[0], I(points))[2],
                                         symbolic_jet(problem.expressions[1], I(points))[2]))
    for interval_jet, values in zip(jets, point_jets):
        assert np.all(values >= interval_jet.lo) and np.all(values <= interval_jet.hi)
    return constants


def hermite_coefficients(values, derivatives, step=.01):
    p0, p1 = values[:-1], values[1:]
    d0, d1 = derivatives[:-1] * step, derivatives[1:] * step
    return np.stack((p0, d0, 3 * (p1 - p0) - 2 * d0 - d1,
                     2 * (p0 - p1) + d0 + d1), axis=1)


def evaluate(coefficients, s, degree=0, step=.01):
    c = coefficients
    for _ in range(degree):
        c = c[:, 1:] * np.arange(1, c.shape[1])[None, :, None]
    value = np.zeros((len(c), len(s), c.shape[-1]))
    for i in reversed(range(c.shape[1])):
        value = value * s[None, :, None] + c[:, i, None, :]
    return value.reshape(-1, c.shape[-1]) / step**degree


def rhs(problem, values, mode):
    if mode == "native":
        d, r = problem.native.values(values)
    else:
        d, r = problem.symbolic(values)
    flux = .5 * (d + np.roll(d, -1, axis=-1)) * (np.roll(values, -1, axis=-1) - values) / problem.dx
    return (flux - np.roll(flux, 1, axis=-1)) / problem.dx + r


def point_jet(problem, values, mode):
    if mode == "native":
        return problem.native.jet(values)
    d, r = (symbolic_jet(e, I(values)) for e in problem.expressions)
    # Point interval midpoint is used only with the explicit arithmetic floor.
    return tuple((q.lo + q.hi) * .5 for q in (d[0], r[0], d[1], r[1], d[2], r[2]))


def poly_ranges(coefficients, subdivisions, degree=0):
    """Enclose cubic polynomials and derivatives on every fine time slab."""
    c = coefficients
    for _ in range(degree):
        c = c[:, 1:] * np.arange(1, c.shape[1])[None, :, None]
    mid = (np.arange(subdivisions) + .5) / subdivisions
    radius = .5 / subdivisions
    values = evaluate(coefficients, mid, degree).reshape(len(c), subdivisions, -1)
    error = np.zeros_like(values)
    for k in range(1, c.shape[1]):
        differentiated = c.copy()
        for _ in range(k):
            differentiated = differentiated[:, 1:] * np.arange(1, differentiated.shape[1])[None, :, None]
        q = np.zeros_like(values)
        for i in reversed(range(differentiated.shape[1])):
            q = q * mid[None, :, None] + differentiated[:, i, None, :]
        error += np.abs(q) * radius**k / math.factorial(k) / .01**degree
    return values.reshape(-1, values.shape[-1]), error.reshape(-1, error.shape[-1]) + ARITHMETIC_FLOOR


def envelope_data(problem, coefficients, subdivisions, mode, constants):
    step = .01 / subdivisions
    s = np.linspace(0, 1, subdivisions + 1)
    endpoints = evaluate(coefficients, s)
    endpoint_derivatives = evaluate(coefficients, s, 1)
    defect = rhs(problem, endpoints, mode) - endpoint_derivatives
    defect = defect.reshape(60, subdivisions + 1, problem.nx)
    q_endpoint = np.maximum(np.abs(defect[:, :-1]), np.abs(defect[:, 1:])).reshape(-1, problem.nx)
    del endpoints, endpoint_derivatives, defect
    values, radii = poly_ranges(coefficients, subdivisions)
    v1, r1 = poly_ranges(coefficients, subdivisions, 1)
    v2, r2 = poly_ranges(coefficients, subdivisions, 2)
    v3, r3 = poly_ranges(coefficients, subdivisions, 3)
    # Spatial differences are formed before interval evaluation to preserve their
    # common time variable and the dissipative spatial cancellation.
    jump_coefficients = (np.roll(coefficients, -1, axis=-1) - coefficients) / problem.dx
    g0, gr0 = poly_ranges(jump_coefficients, subdivisions)
    g1, gr1 = poly_ranges(jump_coefficients, subdivisions, 1)
    g2, gr2 = poly_ranges(jump_coefficients, subdivisions, 2)
    d, r, dp, rp, _, _ = point_jet(problem, values, mode)
    d_abs = np.abs(d) + constants["D1"] * radii
    dp_abs = np.abs(dp) + constants["D2"] * radii
    rp_abs = np.abs(rp) + constants["R2"] * radii
    av = lambda a: .5 * (a + np.roll(a, -1, axis=-1))
    x1, x2 = np.abs(v1) + r1, np.abs(v2) + r2
    face = av(constants["D2"] * x1**2 + dp_abs * x2) * (np.abs(g0) + gr0)
    face += 2 * av(dp_abs * x1) * (np.abs(g1) + gr1)
    face += av(d_abs) * (np.abs(g2) + gr2)
    defect_second = (face + np.roll(face, 1, axis=-1)) / problem.dx
    defect_second += constants["R2"] * x1**2 + rp_abs * x2 + np.abs(v3) + r3
    q = np.max(q_endpoint + step**2 * defect_second / 8, axis=-1) + ARITHMETIC_FLOOR
    lap_coefficients = (np.roll(coefficients, -1, axis=-1) + np.roll(coefficients, 1, axis=-1) - 2 * coefficients) / problem.dx**2
    lap, lap_radius = poly_ranges(lap_coefficients, subdivisions)
    return dict(values=values, radii=radii, d=d, dp=dp, rp=rp,
                g0=g0, gr0=gr0, lap=lap, lap_radius=lap_radius,
                q=q, step=step, mode=mode,
                defect_second_max=float(np.max(defect_second)),
                endpoint_defect_max=float(np.max(q_endpoint)))


def propagate(ell, q, step):
    # Scalar affine recurrence, vectorized using its positive integrating factor.
    cumulative = np.r_[0., np.cumsum(ell * step)]
    if np.max(cumulative) - np.min(cumulative) > 300:
        raise FloatingPointError("Comparison bound overflow: uninformative tube")
    factor = np.exp(cumulative)
    phi = np.where(np.abs(ell) < 1e-10, step,
                   np.expm1(ell * step) / np.where(np.abs(ell) < 1e-10, 1, ell))
    result = factor[1:] * np.cumsum(q * phi / factor[1:])
    return np.r_[0., result]


def enclose(problem, coefficients, subdivisions, mode, constants):
    """Build an enclosure without a truth or full symbolic solution argument."""
    data = envelope_data(problem, coefficients, subdivisions, mode, constants)
    rho = 1e-10
    iterations = []
    for iteration in range(12):
        radius = data["radii"] + rho
        lower, upper = float(np.min(data["values"] - radius)), float(np.max(data["values"] + radius))
        if lower < constants["lower"] or upper > constants["upper"]:
            raise FloatingPointError("Tube leaves the declared closure interval")
        diffusion_lower = data["d"] - constants["D1"] * radius
        jump = problem.dx * (np.abs(data["g0"]) + data["gr0"]) + 2 * rho
        # Sufficient nonnegative off-diagonal Jacobian entries on the whole tube.
        cooperative_margin = diffusion_lower + np.roll(diffusion_lower, -1, axis=-1) - constants["D1"] * jump
        if np.min(cooperative_margin) <= 0:
            raise FloatingPointError("Cooperative Jacobian condition could not be enclosed")
        dp_lo = data["dp"] - constants["D2"] * radius
        dp_hi = data["dp"] + constants["D2"] * radius
        lap_lo = data["lap"] - data["lap_radius"] - 4 * rho / problem.dx**2
        lap_hi = data["lap"] + data["lap_radius"] + 4 * rho / problem.dx**2
        product = np.maximum.reduce((dp_lo * lap_lo, dp_lo * lap_hi, dp_hi * lap_lo, dp_hi * lap_hi))
        # Row sum: R' + D'_i Delta_h u_i + a D'' remainder.
        ell_nodes = data["rp"] + constants["R2"] * radius + product
        ell_nodes += .5 * constants["D2"] * (jump**2 + np.roll(jump, 1, axis=-1)**2) / problem.dx**2
        ell = np.max(ell_nodes, axis=-1) + ARITHMETIC_FLOOR
        beta = propagate(ell, data["q"], data["step"])
        maximum = float(np.max(beta))
        iterations.append(dict(radius=rho, maximum_bound=maximum))
        if maximum < rho:
            endpoints = {str(t): float(beta[int(round(t / data["step"]))]) for t in (.1, .3, .6)}
            return dict(status="enclosed", mode=mode, subdivisions=subdivisions,
                        endpoints=endpoints, tube_radius=rho, maximum_bound=maximum,
                        maximum_logarithmic_norm=float(np.max(ell)), minimum_diffusion=float(np.min(diffusion_lower)),
                        minimum_cooperative_margin=float(np.min(cooperative_margin)),
                        maximum_defect=float(np.max(data["q"])),
                        maximum_defect_second_derivative=data["defect_second_max"],
                        endpoint_defect_max=data["endpoint_defect_max"], iterations=iterations)
        rho = 2 * max(maximum, 1e-10)
        if rho > .02:
            raise FloatingPointError("No self-consistent tube below the declared 0.02 guard")
    raise FloatingPointError("Tube iteration did not converge")


def run_cell(job):
    seed, pattern, subdivisions, folder = job
    start = time.perf_counter()
    folder = Path(folder)
    key = f"s{seed}_{pattern}"
    try:
        problem = old.CurvatureProblem(seed, 64)
        input_path = AUDIT / f"curvature_formal_nx64/{key}_nx64.npz"
        with np.load(input_path, allow_pickle=False) as saved:
            u, z, w = (saved[k] for k in ("native", "tangent", "curvature"))
        jets = np.array([problem.second_variation(a, b, c).reshape(3, 64) for a, b, c in zip(u, z, w)])
        coefs = {"native": hermite_coefficients(u, jets[:, 0]),
                 "first": hermite_coefficients(u + z, jets[:, 0] + jets[:, 1]),
                 "second": hermite_coefficients(u + z + .5 * w, jets[:, 0] + jets[:, 1] + .5 * jets[:, 2])}
        constants = {mode: derivative_constants(problem, mode) for mode in ("native", "symbolic")}
        bounds = {kind: enclose(problem, coefficients, subdivisions,
                                "native" if kind == "native" else "symbolic",
                                constants["native" if kind == "native" else "symbolic"])
                  for kind, coefficients in coefs.items()}
        # Truth and full symbolic trajectories are opened only AFTER all bounds
        # have been computed and frozen. They are used solely for evaluation.
        reference_path = AUDIT / f"formal_nx64/{key}_nx64.npz"
        with np.load(reference_path, allow_pickle=False) as saved:
            reference, symbolic = saved["truth"], saved["alpha_1"]
        rows = []
        rms = lambda x: float(np.sqrt(np.mean(x**2)))
        for kind in ("first", "second"):
            approximate = u + z + (.5 * w if kind == "second" else 0)
            for horizon in (.1, .3, .6):
                k = int(round(horizon / .01))
                b, v, target = u[k], approximate[k], reference[k]
                delta_s = bounds[kind]["endpoints"][str(horizon)]
                delta_n = bounds["native"]["endpoints"][str(horizon)]
                q = float(np.mean((v - target)**2) - np.mean((b - target)**2))
                error_bound = 2 * rms(v - target) * delta_s + delta_s**2 + 2 * rms(b - target) * delta_n + delta_n**2
                actual = float(np.mean((symbolic[k] - target)**2) - np.mean((b - target)**2))
                observed_remainder_inf = float(np.max(np.abs(symbolic[k] - v)))
                assert observed_remainder_inf <= delta_s + 2e-10, (key, kind, horizon, observed_remainder_inf, delta_s)
                assert abs(q - actual) <= error_bound + 2e-12
                certified = bool(abs(q) > error_bound)
                assert not certified or np.sign(q) == np.sign(actual)
                eps_max = max(0., (abs(q) - error_bound) / (2 * (rms(v - b) + delta_s + delta_n)))
                rows.append(dict(seed=seed, pattern=pattern, horizon=horizon, kind=kind,
                                 predicted_change=q, actual_change=actual, bound=error_bound,
                                 interval_lower=q - error_bound, interval_upper=q + error_bound,
                                 certified=certified, prediction_sign_agrees=bool(np.sign(q) == np.sign(actual)),
                                 symbolic_state_bound=delta_s, neural_state_bound=delta_n,
                                 observed_remainder_inf=observed_remainder_inf,
                                 bound_to_observed_remainder=delta_s / max(observed_remainder_inf, 1e-30),
                                 reference_rms_uncertainty_radius=eps_max,
                                 certified_at_1e_6_reference_rms=bool(eps_max > 1e-6),
                                 certified_at_1e_5_reference_rms=bool(eps_max > 1e-5),
                                 certified_at_1e_4_reference_rms=bool(eps_max > 1e-4)))
        result = dict(status="complete", seed=seed, pattern=pattern, subdivisions=subdivisions,
                      source_sha256={"response_arrays": sha(input_path), "evaluation_arrays": sha(reference_path),
                                     "weights": sha(problem.weights_path), "metadata": sha(problem.metadata_path)},
                      driver_sha256=sha(__file__), constants=constants, bounds=bounds, rows=rows,
                      seconds=time.perf_counter() - start)
        save(folder / (key + ".json"), result)
        return result
    except Exception:
        save(folder / (key + ".FAILED.json"), dict(status="failed", job=list(job), traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal", "refine"), required=True)
    parser.add_argument("--subdivisions", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    folder = OUT / f"{args.mode}_dt{args.subdivisions}"
    folder.mkdir(exist_ok=False)
    cells = [(s, p) for s in range(5) for p in old.first.PATTERNS] if args.mode == "formal" else [(0, "low"), (3, "high")]
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, len(cells))) as pool:
        futures = [pool.submit(run_cell, (s, p, args.subdivisions, str(folder))) for s, p in cells]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            completed.append(item)
            print(json.dumps(dict(done=len(completed), total=len(cells), seed=item["seed"], pattern=item["pattern"], seconds=item["seconds"])), flush=True)
    rows = [r for c in completed for r in c["rows"]]
    with (folder / "checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = dict(status="complete", mode=args.mode, cells=len(cells), seed_units=len(set(s for s, _ in cells)),
                   driver_sha256=sha(__file__), checked_utc=datetime.now(timezone.utc).isoformat(),
                   by_predictor={kind: dict(pairs=sum(r["kind"] == kind for r in rows),
                         certified=sum(r["kind"] == kind and r["certified"] for r in rows),
                         sign_agreements=sum(r["kind"] == kind and r["prediction_sign_agrees"] for r in rows),
                         at_reference_1e_6=sum(r["kind"] == kind and r["certified_at_1e_6_reference_rms"] for r in rows),
                         at_reference_1e_5=sum(r["kind"] == kind and r["certified_at_1e_5_reference_rms"] for r in rows),
                         at_reference_1e_4=sum(r["kind"] == kind and r["certified_at_1e_4_reference_rms"] for r in rows))
                         for kind in ("first", "second")},
                   arithmetic_scope="Padded float64 evaluation of analytic continuous-time enclosures; not formally verified rounding arithmetic.",
                   critical=[r for r in rows if r["seed"] == 3 and r["pattern"] == "high" and r["horizon"] == .6])
    save(folder / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
