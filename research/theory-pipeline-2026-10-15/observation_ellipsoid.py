"""Sharp constitutive ranking intervals from the actual observation objective.

The seven-function consistency ellipsoid contains synthetic Case A truth at its
measured consistency budget. That budget is an oracle diagnostic, not an inferred
noise guarantee. Neither neural nor rational symbolic functions must lie in V.
"""
from __future__ import annotations
import response_bounds as rb
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import block_diag
from function_basis import FunctionBasis

sys.path.insert(0, str(rb.WORK / "src"))
from closure_discovery.weak_form.test_functions import make_test_functions_1d

PACKAGE = rb.old.first.PACKAGE
WEIGHTS = dict(weak=1., rollout=.1, mass=1., strong=1., anchor=1., reg=1e-4)


def operators(law, data):
    u, x, t = data["u"], data["x"], data["t"]
    dx, dt = x[1] - x[0], t[1] - t[0]
    phi, grad = make_test_functions_1d(x, 4)
    d, r = law(u)
    mid = u[:, 1:-1]
    ux = (np.roll(mid, -1, axis=-1) - np.roll(mid, 1, axis=-1)) / (2 * dx)
    flux = .5 * (d + np.roll(d, -1, axis=-1)) * (np.roll(u, -1, axis=-1) - u) / dx
    full_rhs = (flux - np.roll(flux, 1, axis=-1)) / dx + r
    weak = dx * (-np.einsum("kx,btx->btk", grad, d[:, 1:-1] * ux)
                 + np.einsum("kx,btx->btk", phi, r[:, 1:-1]))
    support = np.linspace(*data["value_range"], 256)
    derivatives = law(support, derivative=1)
    return dict(weak=weak.ravel(), rollout=(dt * full_rhs[:, :-1]).ravel(),
                mass=r[:, 1:-1].mean(axis=-1).ravel(), strong=full_rhs[:, 1:-1].ravel(),
                anchor=np.asarray(law(np.array([data["value_range"][0]]))[1]).ravel(),
                reg=np.r_[derivatives[0], derivatives[1]])


def targets(data):
    u, x, t = data["u"], data["x"], data["t"]
    dx, dt = x[1] - x[0], t[1] - t[0]
    phi, _ = make_test_functions_1d(x, 4)
    ut = (u[:, 2:] - u[:, :-2]) / (2 * dt)
    return dict(weak=(dx * np.einsum("kx,btx->btk", phi, ut)).ravel(),
                rollout=(u[:, 1:] - u[:, :-1]).ravel(), mass=ut.mean(axis=-1).ravel(),
                strong=ut.ravel(), anchor=np.zeros(1), reg=np.zeros(512))


def polynomial_law(coefficients):
    def law(values, derivative=0):
        return tuple(np.polynomial.polynomial.polyval(values,
                      np.polynomial.polynomial.polyder(p, m=derivative))
                     for p in (coefficients[:3], coefficients[3:]))
    return law


def run(seed, output):
    problem = rb.old.CurvatureProblem(seed, 64)
    basis = FunctionBasis(problem)
    input_path = PACKAGE / f"data/shared_observations/case_a_clean_s{seed}.npz"
    with np.load(input_path, allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    b_raw = targets(data)
    n = {key: 256 if key == "reg" else len(value) for key, value in b_raw.items()}
    rows = []
    for i in range(7):
        law = polynomial_law(np.eye(7)[i])
        rows.append(operators(law, data))
    terms = {key: np.column_stack([r[key] for r in rows]) / np.sqrt(n[key]) for key in b_raw}
    target_terms = {key: v / np.sqrt(n[key]) for key, v in b_raw.items()}
    bundle_path = PACKAGE / f"data/global_objective/quadratic_bundles/shared/case_a_clean_s{seed}.json"
    bundle = json.loads(bundle_path.read_text())["matrices"]["observed_features_observed_targets__space_weak"]
    gram_discrepancies = []
    for key in terms:
        augmented = np.column_stack([terms[key], -target_terms[key]])
        qr = np.asarray(bundle[key]["augmented_qr"])
        got, expected = augmented.T @ augmented, qr.T @ qr
        np.testing.assert_allclose(got, expected, atol=1e-10, rtol=5e-11)
        gram_discrepancies.append(float(np.linalg.norm(got - expected) / max(np.linalg.norm(expected), 1e-30)))
    T = block_diag(basis.transforms["D"], basis.transforms["R"])
    A = np.vstack([np.sqrt(WEIGHTS[key]) * terms[key] @ T for key in terms])
    b = np.concatenate([np.sqrt(WEIGHTS[key]) * target_terms[key] for key in terms])
    theta_dagger = np.linalg.lstsq(A, b, rcond=None)[0]
    H = A.T @ A
    minimum_loss = float(np.sum((A @ theta_dagger - b)**2))
    truth_theta = basis.truth_coordinates[:7]
    truth_budget_squared = float(np.sum((A @ truth_theta - b)**2))
    radius = np.sqrt(max(0, truth_budget_squared - minimum_loss))
    delta = truth_theta - theta_dagger
    np.testing.assert_allclose(delta @ H @ delta, radius**2, atol=2e-14, rtol=2e-10)
    bd, br = basis.values(basis.grid)
    d_N, r_N = problem.native.values(basis.grid)
    d_S, r_S = problem.symbolic(basis.grid)
    d_0, r_0 = bd[:, :7] @ theta_dagger, br[:, :7] @ theta_dagger
    c_D, c_R = d_S - d_N, r_S - r_N
    g = (bd[:, :7].T @ c_D + br[:, :7].T @ c_R) / 512
    score = float(np.mean((d_S - d_0)**2 + (r_S - r_0)**2 - (d_N - d_0)**2 - (r_N - r_0)**2))
    dual = float(np.sqrt(g @ np.linalg.solve(H, g)))
    width = 2 * radius * dual
    d_true, r_true = .01 + .05 * basis.grid, basis.grid * (1 - basis.grid)
    actual = float(np.mean((d_S - d_true)**2 + (r_S - r_true)**2 - (d_N - d_true)**2 - (r_N - r_true)**2))
    assert abs(actual - score) <= width + 1e-13
    extremizers = []
    for sign in (-1, 1):
        theta = theta_dagger + sign * radius * np.linalg.solve(H, g) / dual
        d_t, r_t = bd[:, :7] @ theta, br[:, :7] @ theta
        gain = float(np.mean((d_S - d_t)**2 + (r_S - r_t)**2 - (d_N - d_t)**2 - (r_N - r_t)**2))
        expected_gain = score - sign * width
        np.testing.assert_allclose(gain, expected_gain, rtol=1e-10, atol=2e-14)
        np.testing.assert_allclose(np.linalg.norm(A @ theta - b)**2, truth_budget_squared, rtol=1e-9, atol=1e-12)
        extremizers.append(dict(sign=sign, theta=theta.tolist(), gain=gain))
    native_operator = operators(lambda x, derivative=0: (problem.native.values(x) if derivative == 0
                       else problem.native.jet(x)[2:4]), data)
    r_native = np.concatenate([np.sqrt(WEIGHTS[key] / n[key]) * (native_operator[key] - b_raw[key]) for key in terms])
    theta_N = basis.teacher_coordinates[:7]
    Ah = r_native + b - A @ theta_N
    corrected_residual = r_native - Ah
    sigma = float(np.linalg.svd(A, compute_uv=False)[-1])
    tail = np.sqrt(sum(q**2 for q in basis.residual_norms.values()))
    recovery_bound = float(np.sqrt(tail**2 + ((np.linalg.norm(corrected_residual) + np.sqrt(truth_budget_squared)) / sigma)**2))
    actual_recovery = float(np.sqrt(np.mean((d_N - d_true)**2 + (r_N - r_true)**2)))
    assert actual_recovery <= recovery_bound
    critical_budget = float(np.sqrt(minimum_loss + (abs(score) / (2 * dual))**2))
    record = dict(status="complete", seed=seed, driver_sha256=rb.sha(__file__),
        source_sha256={"weights": rb.sha(problem.weights_path), "metadata": rb.sha(problem.metadata_path),
                       "input": rb.sha(input_path), "quadratic_bundle": rb.sha(bundle_path)},
        basis=basis.record(), maximum_augmented_gram_relative_discrepancy=max(gram_discrepancies),
        minimum_objective=minimum_loss, minimum_consistency_budget=np.sqrt(minimum_loss),
        oracle_truth_consistency_budget=np.sqrt(truth_budget_squared), ellipsoid_radius=radius,
        reference_coefficients=theta_dagger.tolist(), center_gain=score, dual_sensitivity=dual,
        interval_halfwidth=width, interval_lower=score - width, interval_upper=score + width,
        actual_constitutive_gain=actual, robust_sign=bool(abs(score) > width),
        critical_consistency_budget=critical_budget, extremizers=extremizers,
        minimum_observation_singular_value=sigma, actual_neural_recovery_error=actual_recovery,
        neural_recovery_bound=recovery_bound, native_polynomial_tail_norm=tail,
        native_full_objective=float(r_native @ r_native),
        native_residual_corrected_norm=float(np.linalg.norm(corrected_residual)),
        scope="Actual frozen MLP and nonlinear selected expressions; truth assumed in V. Oracle budget evaluates the theorem, not a truth-free budget estimator; priors and discretization are included in consistency error.")
    rb.save(output / f"seed{seed}.json", record)
    np.savez_compressed(output / f"seed{seed}.npz", H=H, g=g, reference_coefficients=theta_dagger,
                        truth_coordinates=truth_theta, teacher_coordinates=theta_N, projection_matrix=T)
    return record


if __name__ == "__main__":
    output = rb.OUT / "observation_ellipsoid"; output.mkdir(exist_ok=False)
    records = [run(seed, output) for seed in range(5)]
    with (output / "checks.csv").open("w", newline="") as stream:
        keys = ["seed", "minimum_consistency_budget", "oracle_truth_consistency_budget", "center_gain",
                "interval_lower", "interval_upper", "actual_constitutive_gain", "robust_sign",
                "critical_consistency_budget", "minimum_observation_singular_value",
                "actual_neural_recovery_error", "neural_recovery_bound"]
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        writer.writerows({key: r[key] for key in keys} for r in records)
    summary = dict(status="complete", seed_units=5, all_sharp_support_intervals_verified=True,
        robust_rankings_at_oracle_budget=sum(r["robust_sign"] for r in records),
        matrix_reconstruction_max_relative_error=max(r["maximum_augmented_gram_relative_discrepancy"] for r in records),
        driver_sha256=rb.sha(__file__), completed_utc=datetime.now(timezone.utc).isoformat(),
        limitation="Consistency budget is measured from synthetic truth; no truth-free calibration claimed.")
    rb.save(output / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
