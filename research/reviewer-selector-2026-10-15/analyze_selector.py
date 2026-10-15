"""Reproduce and diagnose frozen symbolic selection, without neural training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
HIST = REPO / "outputs/revision"
SAME = REPO / "outputs/reviewer-supplement-2026-10-12"
VENDOR = REPO / "src"
sys.path.insert(0, str(VENDOR))
import numpy as np
from scipy.special import expit
from scipy.stats import t as student_t
from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.symbolic.restricted_fit import (
    fit_diffusion_candidates, fit_reaction_candidates, select_symbolic_expression,
)

RELATIVE = [0., .001, .003, .01, .02, .03, .05, .075, .1, .15, .2, .3, .5, .75, 1.]
ABSOLUTE = [0., 1e-12, 3e-12, 1e-11, 3e-11, 1e-10, 3e-10, 1e-9, 2e-9,
            3e-9, 5e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
DEFAULT = (.05, 1e-8)
REPRO_TOL = 2e-8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":"))
                             if isinstance(v, (dict, list, tuple)) else v for k, v in r.items()})


def input_hashes(paths):
    return [{"path": str(p), "sha256": sha(p)} for p in paths]


def historical_teachers():
    for case in ["case_a", "case_b", "case_exp"]:
        for setting in ["clean", "noise1", "noise3", "noise5", "sparse"]:
            for seed in range(5):
                ident = f"main_{case}_{setting}_{seed}_default"
                yield dict(id=ident, cohort="historical_main_150", case=case,
                           setting=setting, seed=seed, epoch=150,
                           fit_curve_path=HIST / f"evidence/surrogate_curves/{ident}_e150.csv",
                           reference_path=HIST / f"evidence/raw/{ident}.json")


def sameobs_teachers():
    for setting in ["clean", "noise3"]:
        for seed in range(5):
            ident = f"sameobs_case_exp_{setting}_{seed}"
            yield dict(id=ident + "_e6000", source_id=ident, cohort="sameobs_6000",
                       case="case_exp", setting=setting, seed=seed, epoch=6000,
                       fit_curve_path=SAME / f"outputs/runs/{ident}/neural_adam_6000.npz",
                       eval_curve_path=SAME / f"outputs/runs/{ident}/neural_adam_6000_curves.npz",
                       reference_path=SAME / f"evidence/training/{ident}.json")


def native_values(path, grid):
    """Evaluate the original native export without importing a training stack.

    Mirrors the archived NumpyMLPClosure.scalar/diffusion/reaction routines;
    load_teacher also checks it against every saved 256-state value.
    """
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        if metadata["backbone"] != "mlp":
            raise ValueError("Unsupported native teacher backbone")
        values = {}
        for branch in ["D", "R"]:
            specification = metadata["branches"][branch]
            span = max(specification["upper"] - specification["lower"], 1e-6)
            scaled = 2 * (grid - specification["lower"]) / span - 1
            polynomial = np.polynomial.polynomial.polyval(scaled, arrays[branch + "_coefficients"])
            hidden = scaled[..., None]
            for index in range(len(specification["layers"])):
                hidden = hidden @ arrays[f"{branch}_weight_{index}"].T + arrays[f"{branch}_bias_{index}"]
                if index + 1 < len(specification["layers"]):
                    hidden = np.tanh(hidden)
            scalar = polynomial + specification["residual_scale"] * hidden[..., 0]
            values[branch] = (metadata["min_diffusion"] + np.exp(arrays["log_diffusion_scale"]) * expit(scalar)
                              if branch == "D" else scalar)
        return values


def load_teacher(config):
    p = config["fit_curve_path"]
    case = CASE_BUILDERS[config["case"]]()
    if p.suffix == ".csv":
        values = np.genfromtxt(p, delimiter=",", names=True)
        fit = {"grid": values["u"], "D": values["D"], "R": values["R"]}
    else:
        grid512 = np.linspace(*case.value_range, 512)
        fit = {"grid": grid512, **native_values(p, grid512)}
    assert len(fit["grid"]) == 512 and np.all(np.diff(fit["grid"]) > 0)
    assert all(np.isfinite(v).all() for v in fit.values())
    reference = json.loads(config["reference_path"].read_text())
    if config["cohort"] == "historical_main_150":
        grid = np.linspace(*case.value_range, 256)
        teacher = {b: np.interp(grid, fit["grid"], fit[b]) for b in ["D", "R"]}
        expected = reference["epochs"]["150"]
        prior_grid = reference["thresholds"]
    else:
        with np.load(config["eval_curve_path"]) as arrays:
            grid = arrays["grid"].copy()
            teacher = {b: arrays["neural_" + b].copy() for b in ["D", "R"]}
        reconstructed = native_values(p, grid)
        for branch in ["D", "R"]:
            np.testing.assert_allclose(reconstructed[branch], teacher[branch], rtol=2e-12, atol=2e-13)
        expected = reference["stages"]["neural_adam_6000"]
        prior_grid = []
    assert len(grid) == 256
    truth = {"D": case.diffusion(grid), "R": case.reaction(grid)}
    paths = [config["fit_curve_path"], config["reference_path"]]
    if "eval_curve_path" in config:
        paths.append(config["eval_curve_path"])
    return fit, grid, teacher, truth, expected, prior_grid, input_hashes(paths), case.value_range


def real_roots(coefficients, interval):
    c = np.asarray(coefficients, dtype=float)
    while len(c) > 1 and c[-1] == 0:
        c = c[:-1]
    if len(c) <= 1:
        return []
    roots = np.polynomial.polynomial.polyroots(c)
    lo, hi = interval
    return sorted(float(r.real) for r in roots
                  if abs(r.imag) <= 1e-9 * max(1., abs(r.real))
                  and lo - 1e-12 <= r.real <= hi + 1e-12)


def rational_polynomials(candidate):
    p = candidate.parameters
    if candidate.family == "polynomial":
        return np.array(p), np.array([1.])
    if candidate.family == "rational":
        split = (len(p) + 1) // 2
        return np.array(p[:split]), np.array([1., *p[split:]])
    if candidate.family == "saturation":
        a, b, k = p
        return np.array([a + b, a * k]), np.array([1., k])
    if candidate.family == "u_saturation":
        a, b, k = p
        return np.array([0., a + b, a * k]), np.array([1., k])
    return None


def exact_quadratic_range(coefficients, interval):
    """Exact extrema of degree<=2 polynomial with binary64 coefficients."""
    c = [Fraction(float(v)) for v in coefficients]
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    if len(c) > 3:
        raise ValueError("Exact range helper is only for quadratic denominators")
    lo, hi = [Fraction(float(v)) for v in interval]
    points = [lo, hi]
    if len(c) == 3 and c[2] != 0:
        vertex = -c[1] / (2 * c[2])
        if lo <= vertex <= hi:
            points.append(vertex)
    values = [sum(a * x**i for i, a in enumerate(c)) for x in points]
    return min(values), max(values)


def upward(value):
    return float(np.nextafter(float(value) * (1 + 1e-12), np.inf))


def coefficient_bound(coefficients, interval, derivative=False):
    u = max(abs(Fraction(float(v))) for v in interval)
    c = [Fraction(float(v)) for v in coefficients]
    if derivative:
        return sum(i * abs(a) * u**(i - 1) for i, a in enumerate(c) if i > 0)
    return sum(abs(a) * u**i for i, a in enumerate(c))


def symbolic_lipschitz_bound(candidate, interval):
    pair = rational_polynomials(candidate)
    if pair is not None:
        p, q = pair
        qmin, qmax = exact_quadratic_range(q, interval)
        if qmin <= 0 <= qmax:
            return None
        m = min(abs(qmin), abs(qmax))
        bound = (coefficient_bound(p, interval, True) / m
                 + coefficient_bound(p, interval) * coefficient_bound(q, interval, True) / m**2)
        return upward(bound)
    a, b, k = candidate.parameters
    maximum_exp = math.exp(max(k * interval[0], k * interval[1]))
    if candidate.family == "exp_decay":
        return upward(abs(b * k) * maximum_exp)
    if candidate.family == "u_exp_decay":
        return upward(abs(a) + abs(b) * maximum_exp * (1 + abs(k) * max(abs(v) for v in interval)))
    raise ValueError("Unknown symbolic derivative family")


def teacher_uniform_context(config, fit, interval):
    fine = np.unique(np.concatenate([np.linspace(*interval, 8193), fit["grid"]]))
    coarse = np.unique(np.concatenate([np.linspace(*interval, 4097), fit["grid"]]))
    coarse_indices = np.searchsorted(fine, coarse)
    np.testing.assert_array_equal(fine[coarse_indices], coarse)
    if config["cohort"] == "historical_main_150":
        values = {b: np.interp(fine, fit["grid"], fit[b]) for b in ["D", "R"]}
        lipschitz = {b: upward(np.max(np.abs(np.diff(fit[b]) / np.diff(fit["grid"])))) for b in ["D", "R"]}
        lower_D = float(np.min(fit["D"]))
    else:
        values = native_values(config["fit_curve_path"], fine)
        with np.load(config["fit_curve_path"], allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"]))
            lipschitz = {}
            for b in ["D", "R"]:
                spec = metadata["branches"][b]
                poly_derivative = sum(i * abs(float(a)) for i, a in enumerate(arrays[b + "_coefficients"]))
                residual_bound = float(np.prod([np.linalg.norm(arrays[f"{b}_weight_{i}"], ord="fro")
                                                for i in range(len(spec["layers"]))]))
                scalar = (poly_derivative + abs(spec["residual_scale"]) * residual_bound) * 2 / max(spec["upper"] - spec["lower"], 1e-6)
                lipschitz[b] = upward(scalar * float(np.exp(arrays["log_diffusion_scale"])) / 4 if b == "D" else scalar)
            lower_D = metadata["min_diffusion"]
    case = CASE_BUILDERS[config["case"]]()
    truth = {"D": case.diffusion(fine), "R": case.reaction(fine)}
    return dict(grid=fine, teacher=values, truth=truth, coarse_indices=coarse_indices,
                maximum_grid_gap=float(np.max(np.diff(fine))), teacher_lipschitz=lipschitz,
                teacher_global_diffusion_lower_bound=lower_D)


def uniform_diagnostics(candidate, branch, interval, context):
    grid = context["grid"]
    with np.errstate(all="ignore"):
        symbolic = candidate.evaluate(grid)
    finite = bool(np.isfinite(symbolic).all())
    result = dict(uniform_diagnostic_points=len(grid), uniform_diagnostic_maximum_gap=context["maximum_grid_gap"],
                  uniform_diagnostic_status="finite_samples" if finite else "nonfinite_samples",
                  teacher_lipschitz_upper=context["teacher_lipschitz"][branch],
                  teacher_global_diffusion_lower_bound=context["teacher_global_diffusion_lower_bound"] if branch == "D" else None)
    if not finite:
        return dict(**result, sampled_compression_sup=None, sampled_compression_sup_coarse=None,
                    sampled_sup_refinement_increment=None, sampled_teacher_true_sup=None,
                    sampled_symbolic_true_sup=None, symbolic_lipschitz_upper=None,
                    compression_sup_lipschitz_upper=None, sufficient_diffusion_positivity_by_perturbation_bound=None)
    compression = np.abs(symbolic - context["teacher"][branch])
    maximum = float(np.max(compression))
    coarse = float(np.max(compression[context["coarse_indices"]]))
    Ls = symbolic_lipschitz_bound(candidate, interval)
    bound = upward(maximum + (Ls + context["teacher_lipschitz"][branch]) * context["maximum_grid_gap"] / 2) if Ls is not None else None
    return dict(**result, sampled_compression_sup=maximum, sampled_compression_sup_coarse=coarse,
                sampled_sup_refinement_increment=maximum - coarse,
                sampled_teacher_true_sup=float(np.max(np.abs(context["teacher"][branch] - context["truth"][branch]))),
                sampled_symbolic_true_sup=float(np.max(np.abs(symbolic - context["truth"][branch]))),
                symbolic_lipschitz_upper=Ls, compression_sup_lipschitz_upper=bound,
                sufficient_diffusion_positivity_by_perturbation_bound=(bound < context["teacher_global_diffusion_lower_bound"]
                                                                     if branch == "D" and bound is not None else None))


def physical_diagnostics(candidate, branch, interval):
    lo, hi = interval
    points = [lo, hi]
    polynomial_pair = rational_polynomials(candidate)
    poles = []
    roots = []
    denominator_min = 1.
    root_slopes = []
    if polynomial_pair is not None:
        p, q = polynomial_pair
        poles = real_roots(q, interval)
        q_points = [lo, hi, *real_roots(np.polynomial.polynomial.polyder(q), interval)]
        denominator_min = 0. if poles else float(np.min(np.abs(np.polynomial.polynomial.polyval(q_points, q))))
        qmin, qmax = exact_quadratic_range(q, interval)
        denominator_zero_exact = qmin <= 0 <= qmax
        denominator_min = 0. if denominator_zero_exact else float(min(abs(qmin), abs(qmax)))
        derivative = np.polynomial.polynomial.polysub(
            np.polynomial.polynomial.polymul(np.polynomial.polynomial.polyder(p), q),
            np.polynomial.polynomial.polymul(p, np.polynomial.polynomial.polyder(q)))
        points += real_roots(derivative, interval)
        roots = real_roots(p, interval)
        for z in roots:
            den = float(np.polynomial.polynomial.polyval(z, q))
            if den != 0:
                root_slopes.append(float(np.polynomial.polynomial.polyval(z, derivative) / den**2))
            else:
                root_slopes.append(None)
    elif candidate.family == "exp_decay":
        a, b, k = candidate.parameters
        if b != 0 and -a / b > 0:
            z = math.log(-a / b) / k
            if lo <= z <= hi:
                roots = [z]
                root_slopes = [float(b * k * math.exp(k * z))]
    elif candidate.family == "u_exp_decay":
        a, b, k = candidate.parameters
        if lo <= 0 <= hi:
            roots.append(0.)
            root_slopes.append(float(a + b))
        if b != 0 and -a / b > 0:
            z = math.log(-a / b) / k
            if lo <= z <= hi and z != 0:
                roots.append(z)
                root_slopes.append(float(a + b * math.exp(k * z) * (1 + k * z)))
        # Numerical bounded minimization is intentionally not presented as a
        # global certificate for u*exp(k*u). Diffusion never uses this family.
    with np.errstate(all="ignore"):
        predictions = candidate.evaluate(np.array(points))
        low = candidate.evaluate(np.array([0., .01]))
    denominator_zero = denominator_zero_exact if polynomial_pair is not None else False
    continuous_min = (float(np.min(predictions)) if not denominator_zero and np.isfinite(predictions).all() else None)
    return dict(denominator_zero_in_support=bool(denominator_zero), denominator_roots=poles,
                minimum_denominator_magnitude=denominator_min,
                minimum_D_support=continuous_min if branch == "D" else None,
                nonpositive_D_support=(continuous_min <= 0 if branch == "D" and continuous_min is not None else None),
                R_at_zero=float(low[0]) if branch == "R" and np.isfinite(low[0]) else None,
                R_at_001=float(low[1]) if branch == "R" and np.isfinite(low[1]) else None,
                negative_small_state_reaction=bool(low[1] < 0) if branch == "R" and np.isfinite(low[1]) else None,
                reaction_roots=roots if branch == "R" else [],
                reaction_root_slopes=root_slopes if branch == "R" else [],
                physical_check_method="float64 polynomial roots/critical points and analytic exponential structure; not interval certification")


def geometry(candidate, grid, teacher, truth):
    with np.errstate(all="ignore"):
        symbolic = candidate.evaluate(grid)
    if not np.isfinite(symbolic).all():
        return dict(numerical_status="nonfinite_evaluation", eN=None, eC=None, eS=None,
                    BIR=None, rho=None, cosine_alignment=None, error_identity_residual=None,
                    bir_squared_identity_residual=None, reverse_triangle_residual=None,
                    teacher_relative_compression=None, symbolic_min_on_eval=None)
    a = teacher - truth
    c = symbolic - teacher
    eN = float(np.linalg.norm(a))
    eC = float(np.linalg.norm(c))
    eS = float(np.linalg.norm(symbolic - truth))
    dot = float(np.dot(a, c))
    cosine = dot / (eN * eC) if eN > 0 and eC > 0 else None
    bir = eS / eN if eN > 0 else None
    rho = eC / eN if eN > 0 else None
    identity = eS**2 - eN**2 - eC**2 - 2 * dot
    squared_identity = bir**2 - 1 - rho**2 - 2 * dot / eN**2 if eN > 0 else None
    bound_residual = abs(bir - 1) - rho if eN > 0 else None
    return dict(numerical_status="finite", eN=eN, eC=eC, eS=eS, BIR=bir, rho=rho,
                cosine_alignment=cosine, cross_term=2 * dot,
                error_identity_residual=identity,
                bir_squared_identity_residual=squared_identity,
                reverse_triangle_residual=bound_residual,
                teacher_relative_compression=eC / (float(np.linalg.norm(teacher)) + 1e-12),
                symbolic_min_on_eval=float(np.min(symbolic)),
                teacher_min_on_eval=float(np.min(teacher)),
                true_relative_error=eS / (float(np.linalg.norm(truth)) + 1e-12),
                teacher_true_relative_error=eN / (float(np.linalg.norm(truth)) + 1e-12))


def select_index(candidates, relative, absolute):
    chosen = select_symbolic_expression(candidates, relative_tolerance=relative, absolute_tolerance=absolute)
    return next(i for i, c in enumerate(candidates) if c is chosen)


def reproduction(candidates, expected, prior, branch, metrics):
    i = select_index(candidates, *DEFAULT)
    chosen = candidates[i]
    before = expected["symbols"][0 if branch == "D" else 1]
    same_structure = (chosen.family, chosen.complexity) == (before["family"], before["complexity"])
    parameters_match = (same_structure and np.allclose(chosen.parameters, before["parameters"], rtol=1e-7, atol=1e-9))
    parameter_delta = (float(np.max(np.abs(np.asarray(chosen.parameters) - before["parameters"])))
                       if len(chosen.parameters) == len(before["parameters"]) else None)
    mse_delta = abs(chosen.fit_mse - before["fit_mse"])
    checked = []
    for t in prior:
        j = select_index(candidates, t["relative"], t["absolute"])
        c = candidates[j]
        bir = metrics[j]["BIR"]
        difference = abs(bir - t["BIR_" + branch]) if bir is not None else None
        passed = ((c.family, c.complexity) == (t["family_" + branch], t["complexity_" + branch])
                  and difference is not None and difference <= REPRO_TOL)
        checked.append(dict(relative=t["relative"], absolute=t["absolute"],
                            expected_family=t["family_" + branch], restored_family=c.family,
                            expected_complexity=t["complexity_" + branch], restored_complexity=c.complexity,
                            BIR_absolute_difference=difference, passed=bool(passed)))
    if "compression" in expected:
        expected_bir = expected["compression"][branch]["BIR"]
    else:
        expected_bir = expected["frozen_metrics"][branch] / expected["metrics"][branch]
    bir_delta = abs(metrics[i]["BIR"] - expected_bir)
    passed = bool(same_structure and parameters_match and mse_delta <= 1e-12
                  and bir_delta <= REPRO_TOL and all(x["passed"] for x in checked))
    return dict(passed=passed, default_candidate_index=i, default_family=chosen.family,
                default_complexity=chosen.complexity, parameters_match=bool(parameters_match),
                parameter_max_absolute_difference=parameter_delta, fit_MSE_absolute_difference=mse_delta,
                default_BIR_absolute_difference=bir_delta, old_grid_cells=len(checked), old_grid_checks=checked)


def sweep_row(base, candidates, cached, relative, absolute, kind="dense_grid"):
    best = min(c.fit_mse for c in candidates)
    i = select_index(candidates, relative, absolute)
    c = candidates[i]
    extra = relative * best + absolute
    return dict(**base, grid_kind=kind, relative=relative, absolute=absolute,
                candidate_index=i, family=c.family, complexity=c.complexity,
                best_mse=best, selected_mse=c.fit_mse,
                relative_slack=relative * best, absolute_slack=absolute,
                absolute_share_of_added_slack=absolute / extra if extra > 0 else None,
                eligible_candidates=sum(x.fit_mse <= (1 + relative) * best + absolute for x in candidates),
                **cached[i])


def separated_probes(boundary, ordered, index, upper, arithmetic_scale):
    before = ordered[index - 1] if index > 0 else 0.
    after = ordered[index + 1] if index + 1 < len(ordered) else upper
    resolution = 32 * abs(float(np.spacing(arithmetic_scale)))
    eps = max(abs(boundary) * 1e-6, resolution, 1e-20)
    low_gap, high_gap = boundary - before, after - boundary
    low = boundary - min(eps, low_gap / 4) if low_gap > 0 else boundary
    high = boundary + min(eps, high_gap / 4) if high_gap > 0 else boundary
    return float(max(0., low)), float(min(upper, high)), resolution


def breakpoints(base, candidates, cached):
    best = min(c.fit_mse for c in candidates)
    lines = [dict(**base, candidate_index=i, family=c.family, complexity=c.complexity,
                  candidate_mse=c.fit_mse, best_mse=best,
                  absolute_intercept=c.fit_mse - best, absolute_slope=-best,
                  required_absolute_at_default_relative=c.fit_mse - 1.05 * best,
                  admitted_at_zero_relative_zero_absolute=c.fit_mse <= best)
             for i, c in enumerate(candidates)]
    events = []
    probes = []
    for axis, fixed_values, upper in [("absolute", RELATIVE, 1e-4), ("relative", ABSOLUTE, 1.)]:
        for fixed in fixed_values:
            boundaries = defaultdict(list)
            for i, c in enumerate(candidates):
                value = (c.fit_mse - (1 + fixed) * best if axis == "absolute"
                         else (c.fit_mse - fixed) / best - 1 if best > 0 else None)
                if value is not None and 0 < value <= upper:
                    boundaries[float(value)].append(i)
            ordered = sorted(boundaries)
            for n, value in enumerate(ordered):
                arithmetic_scale = (1 + fixed) * best if axis == "absolute" else 1 + value
                low, high, resolution = separated_probes(value, ordered, n, upper, arithmetic_scale)
                points = [("below_nextafter", float(np.nextafter(value, -np.inf))),
                          ("at", value), ("above_nextafter", float(np.nextafter(value, np.inf))),
                          ("below_separated", low), ("above_separated", high)]
                chosen = {}
                for side, point in points:
                    if not 0 <= point <= upper:
                        continue
                    relative, absolute = (fixed, point) if axis == "absolute" else (point, fixed)
                    row = sweep_row(base, candidates, cached, relative, absolute, "boundary_probe")
                    probes.append(dict(**row, boundary_axis=axis, fixed_other_value=fixed,
                                       analytic_boundary=value, boundary_side=side))
                    chosen[side] = row["candidate_index"]
                before = chosen.get("below_separated")
                after = chosen.get("above_separated")
                events.append(dict(**base, boundary_axis=axis, fixed_other_value=fixed,
                                   analytic_boundary=value, admitted_candidate_indices=boundaries[value],
                                   before_candidate_index=before, after_candidate_index=after,
                                   before_family=candidates[before].family if before is not None else None,
                                   after_family=candidates[after].family if after is not None else None,
                                   before_complexity=candidates[before].complexity if before is not None else None,
                                   after_complexity=candidates[after].complexity if after is not None else None,
                                   selected_expression_changes=(before != after if before is not None and after is not None else None),
                                   arithmetic_probe_resolution=resolution,
                                   separated_before_value=low, separated_after_value=high,
                                   probe_selection_indices=chosen))
    return lines, events, probes


def stats(values):
    finite = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = len(finite)
    if n == 0:
        return dict(n=0, mean=None, sd=None, ci95_low=None, ci95_high=None)
    mean = float(finite.mean())
    sd = float(finite.std(ddof=1)) if n > 1 else None
    half = float(student_t.ppf(.975, n - 1) * sd / math.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, sd=sd, ci95_low=mean - half if half is not None else None,
                ci95_high=mean + half if half is not None else None)


def aggregate(rows):
    keys = ["cohort", "case", "setting", "epoch", "branch", "relative", "absolute"]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summarized = []
    for key, values in sorted(groups.items()):
        base = dict(zip(keys, key))
        common = dict(total_teachers=len(values), dataset_seeds=sorted(r["seed"] for r in values),
                      families=dict(Counter(r["family"] for r in values)),
                      denominator_zero_count=sum(r["denominator_zero_in_support"] for r in values),
                      nonpositive_diffusion_count=sum(r["nonpositive_D_support"] is True for r in values),
                      negative_small_reaction_count=sum(r["negative_small_state_reaction"] is True for r in values),
                      diffusion_positivity_from_sup_bound_count=sum(r.get("sufficient_diffusion_positivity_by_perturbation_bound") is True for r in values),
                      nonfinite_count=sum(r["numerical_status"] != "finite" for r in values))
        for metric in ["BIR", "rho", "cosine_alignment", "eN", "eC", "eS", "complexity", "minimum_D_support",
                       "R_at_zero", "R_at_001", "absolute_share_of_added_slack", "sampled_compression_sup",
                       "compression_sup_lipschitz_upper", "sampled_sup_refinement_increment"]:
            summarized.append(dict(**base, **common, metric=metric, **stats([r.get(metric) for r in values])))
    return summarized


def main():
    global HIST, SAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/reviewer-selector-2026-10-15")
    parser.add_argument("--cohort", choices=["historical", "sameobs", "both"], default="both")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--historical-root", type=Path, default=HIST)
    parser.add_argument("--same-observation-root", type=Path, default=SAME)
    args = parser.parse_args()
    args.output = args.output.resolve()
    HIST = args.historical_root.resolve()
    SAME = args.same_observation_root.resolve()
    configs = []
    if args.cohort in ["historical", "both"]:
        configs += list(historical_teachers())
    if args.cohort in ["sameobs", "both"]:
        configs += list(sameobs_teachers())
    if args.limit is not None:
        configs = configs[:args.limit]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, *sys.argv]
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    code_paths = [Path(__file__),
                  VENDOR / "closure_discovery/symbolic/restricted_fit.py",
                  VENDOR / "closure_discovery/data_generation/cases.py",
                  VENDOR / "closure_discovery/evaluation/metrics.py",
                  REPO / "revision/reliability_core.py",
                  REPO / "experiments/run_reliability_supplement.py"]
    provenance = dict(started_utc=started, command=command, cwd=str(Path.cwd()),
                      mode="validation_only" if args.validation_only else "full_diagnostics",
                      relative=RELATIVE, absolute=ABSOLUTE, default=list(DEFAULT),
                      source_files=input_hashes(code_paths),
                      git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                      git_status=subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
                      environment=dict(python=sys.version, platform=platform.platform(),
                                       packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                                       thread_options={k: os.environ.get(k) for k in
                                                       ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]}),
                      planned_teachers=len(configs), truth_usage="evaluation only; no candidate fitting or selection uses truth")
    write_json(output / "PROTOCOL.json", provenance)
    validation_rows = []
    dense = []
    lines = []
    events = []
    probes = []
    failures = []
    inputs = []
    for index, config in enumerate(configs):
        serializable_config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        try:
            fit, grid, teacher, truth, expected, prior, teacher_inputs, interval = load_teacher(config)
            inputs += teacher_inputs
            uniform_context = teacher_uniform_context(config, fit, interval) if not args.validation_only else None
            candidate_record = dict(config=serializable_config, input_hashes=teacher_inputs,
                                    evaluation_policy="native256" if config["cohort"] == "sameobs_6000" else "interpolated256",
                                    fitting_points=512, evaluation_points=256, support=list(interval), branches={})
            valid = True
            for branch, function in [("D", fit_diffusion_candidates), ("R", fit_reaction_candidates)]:
                candidates = function(fit["grid"], fit[branch])
                assert candidates and all(math.isfinite(c.fit_mse) for c in candidates)
                cached = [{**geometry(c, grid, teacher[branch], truth[branch]),
                           **physical_diagnostics(c, branch, interval)} for c in candidates]
                if uniform_context is not None:
                    for i, candidate in enumerate(candidates):
                        cached[i].update(uniform_diagnostics(candidate, branch, interval, uniform_context))
                validation = reproduction(candidates, expected, prior, branch, cached)
                valid = valid and validation["passed"]
                base = {k: config[k] for k in ["id", "cohort", "case", "setting", "seed", "epoch"]}
                base["branch"] = branch
                validation_rows.append(dict(**base, **{k: v for k, v in validation.items() if k != "old_grid_checks"}))
                candidate_record["branches"][branch] = dict(validation=validation,
                                                           candidates=[dict(index=i, **asdict(c), metrics=cached[i])
                                                                       for i, c in enumerate(candidates)])
                if not args.validation_only and validation["passed"]:
                    dense += [sweep_row(base, candidates, cached, r, a) for r in RELATIVE for a in ABSOLUTE]
                    ls, es, ps = breakpoints(base, candidates, cached)
                    lines += ls
                    events += es
                    probes += ps
            write_json(output / "candidate_records" / f"{config['id']}.json", candidate_record)
            if not valid:
                raise ValueError("Default or original 25-grid reproduction gate failed; see candidate record.")
            print(f"{index + 1}/{len(configs)} {config['id']} reproduction PASS", flush=True)
        except Exception as exc:
            failure = dict(config=serializable_config, error=str(exc), traceback=traceback.format_exc(),
                           stage="reconstruction_or_diagnostics")
            failures.append(failure)
            write_json(output / "failures" / f"{config['id']}.json", failure)
            print(f"{index + 1}/{len(configs)} {config['id']} FAILED: {exc}", flush=True)
    write_csv(output / "reproduction.csv", validation_rows)
    write_json(output / "INPUTS.json", list({r["path"]: r for r in inputs}.values()))
    report = dict(mode=provenance["mode"], started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  seconds=time.perf_counter() - begin, planned_teachers=len(configs), failed_teachers=len(failures),
                  reproduction_branch_rows=len(validation_rows),
                  reproduced_branches=sum(r["passed"] for r in validation_rows),
                  old_grid_branch_cells=sum(r["old_grid_cells"] for r in validation_rows),
                  maximum_default_parameter_difference=max((r["parameter_max_absolute_difference"] or 0 for r in validation_rows), default=None),
                  maximum_default_BIR_difference=max((r["default_BIR_absolute_difference"] for r in validation_rows), default=None),
                  status="failed" if failures else "complete", failures=failures)
    if not args.validation_only:
        write_csv(output / "dense_grid.csv", dense)
        write_csv(output / "dense_grid_summary.csv", aggregate(dense))
        write_csv(output / "candidate_admission_lines.csv", lines)
        write_csv(output / "boundary_events.csv", events)
        write_csv(output / "boundary_probes.csv", probes)
        write_csv(output / "default_cells.csv", [r for r in dense if (r["relative"], r["absolute"]) == DEFAULT])
        write_csv(output / "selection_transitions.csv", [r for r in events if r["selected_expression_changes"] is True])
        report.update(dense_rows=len(dense), candidate_admission_lines=len(lines),
                      boundary_events=len(events), boundary_probe_rows=len(probes),
                      selected_expression_changes=sum(r["selected_expression_changes"] is True for r in events),
                      nonfinite_dense_rows=sum(r["numerical_status"] != "finite" for r in dense),
                      physical_diagnostics_note="Function roots and critical points in float64; denominator sign ranges use exact rational binary64-coefficient arithmetic. Sup maxima use nested grids plus conservative analytic Lipschitz bounds. No integration claim or interval-arithmetic certificate.",
                      scope="75 historical main 150-epoch teachers plus ten same-observation 6000-epoch teachers, distinct cohorts. No new neural-weight-study teachers are pooled here.",
                      archived_failed_attempt="attempts/float32_snapshot_fit retains the first incorrect-input reconstruction attempt in full.",
                      inference_note="All tolerance cells reuse fixed teachers. Cohorts remain separate. Truth evaluates only.")
        finite = [r for r in dense if r["numerical_status"] == "finite"]
        report.update(maximum_reverse_triangle_violation=max((r["reverse_triangle_residual"] for r in finite), default=None),
                      maximum_squared_identity_residual=max((abs(r["bir_squared_identity_residual"]) for r in finite), default=None),
                      maximum_sampled_sup_bound_violation=max((r["sampled_compression_sup"] - r["compression_sup_lipschitz_upper"]
                                                               for r in finite if r["compression_sup_lipschitz_upper"] is not None), default=None))
    write_json(output / ("VALIDATION.json" if args.validation_only else "RESULTS.json"), report)
    generated = [p for p in output.rglob("*") if p.is_file() and p.name != "MANIFEST.json"]
    write_json(output / "MANIFEST.json", dict(files=input_hashes(generated), source_files=input_hashes(code_paths)))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
