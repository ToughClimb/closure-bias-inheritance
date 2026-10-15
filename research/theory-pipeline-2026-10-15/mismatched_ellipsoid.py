"""Evaluate the approximation-remainder ranking interval for frozen Exp models."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import json
import numpy as np
from scipy.linalg import block_diag
import response_bounds as rb
import observation_ellipsoid as obs


def run(seed, output):
    root = obs.PACKAGE
    name = f"gpu_case_exp_clean_s{seed}_default_e1500"
    weights = root / f"data/native_weights_gpu/{name}.npz"
    metadata_path = weights.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    assert rb.sha(weights) == metadata["export_sha256"]
    native = rb.old.NativeJet(weights)
    expressions = metadata["frozen_expressions"]
    input_path = root / f"data/shared_observations/case_exp_clean_s{seed}.npz"
    with np.load(input_path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    grid = np.linspace(0, 1.8, 512)
    transforms, orthogonal = [], []
    for degree in (2, 3):
        matrix = np.polynomial.polynomial.polyvander(grid, degree)
        _, factor = np.linalg.qr(matrix / np.sqrt(512), mode="reduced")
        transform = np.linalg.solve(factor, np.eye(degree + 1))
        transforms.append(transform); orthogonal.append(matrix @ transform)
    B = block_diag(*orthogonal)
    T = block_diag(*transforms)
    truth = np.r_[.01 + .035 * (1 - np.exp(-2.5 * grid)), grid * (1.1 * np.exp(-1.4 * grid) - .22)]
    theta_star = B.T @ truth / 512
    truth_tail = truth - B @ theta_star
    zeta = np.linalg.norm(truth_tail) / np.sqrt(512)
    np.testing.assert_allclose(B.T @ truth_tail / 512, np.zeros(7), atol=3e-14)
    b_raw = obs.targets(data)
    n = {key: 256 if key == "reg" else len(value) for key, value in b_raw.items()}
    columns = [obs.operators(obs.polynomial_law(np.eye(7)[i]), data) for i in range(7)]
    terms = {key: np.column_stack([c[key] for c in columns]) / np.sqrt(n[key]) for key in b_raw}
    target_terms = {key: value / np.sqrt(n[key]) for key, value in b_raw.items()}
    bundle_path = root / f"data/global_objective/quadratic_bundles/shared/case_exp_clean_s{seed}.json"
    bundle = json.loads(bundle_path.read_text())["matrices"]["observed_features_observed_targets__space_weak"]
    for key in terms:
        q = np.column_stack([terms[key], -target_terms[key]])
        archived = np.asarray(bundle[key]["augmented_qr"])
        np.testing.assert_allclose(q.T @ q, archived.T @ archived, rtol=5e-11, atol=1e-10)
    A = np.vstack([np.sqrt(obs.WEIGHTS[key]) * terms[key] @ T for key in terms])
    b = np.concatenate([np.sqrt(obs.WEIGHTS[key]) * target_terms[key] for key in terms])
    theta0 = np.linalg.lstsq(A, b, rcond=None)[0]
    H = A.T @ A
    minimum_loss = float(np.linalg.norm(A @ theta0 - b)**2)
    epsilon_squared = float(np.linalg.norm(A @ theta_star - b)**2)
    radius = np.sqrt(max(0, epsilon_squared - minimum_loss))
    f_N = np.concatenate(native.values(grid))
    f_S = np.concatenate([rb.old.first.symbolic_value(e, grid) for e in expressions])
    c = f_S - f_N
    g = B.T @ c / 512
    c_tail = c - B @ g
    reference = B @ theta0
    center = float((np.linalg.norm(f_S - reference)**2 - np.linalg.norm(f_N - reference)**2) / 512)
    observation_width = float(2 * radius * np.sqrt(g @ np.linalg.solve(H, g)))
    approximation_width = float(2 * zeta * np.linalg.norm(c_tail) / np.sqrt(512))
    width = observation_width + approximation_width
    actual = float((np.linalg.norm(f_S - truth)**2 - np.linalg.norm(f_N - truth)**2) / 512)
    assert abs(actual - center) <= width + 1e-12
    # Attaining both independent support terms checks the sharp outer product set.
    extremizers = []
    for sign in (-1, 1):
        theta = theta0 + sign * radius * np.linalg.solve(H, g) / np.sqrt(g @ np.linalg.solve(H, g))
        tail = sign * zeta * c_tail / (np.linalg.norm(c_tail) / np.sqrt(512))
        proposed = B @ theta + tail
        gain = float((np.linalg.norm(f_S - proposed)**2 - np.linalg.norm(f_N - proposed)**2) / 512)
        np.testing.assert_allclose(gain, center - sign * width, rtol=1e-9, atol=2e-13)
        extremizers.append(dict(sign=sign, gain=gain))
    record = dict(status="complete", case="Exp", seed=seed, driver_sha256=rb.sha(__file__),
        source_sha256={"weights": rb.sha(weights), "metadata": rb.sha(metadata_path),
                       "input": rb.sha(input_path), "quadratic_bundle": rb.sha(bundle_path)},
        approximation_remainder_norm=float(zeta), minimum_consistency_budget=np.sqrt(minimum_loss),
        oracle_projected_truth_consistency_budget=np.sqrt(epsilon_squared),
        center_gain=center, observation_halfwidth=observation_width,
        approximation_halfwidth=approximation_width, interval_lower=center - width,
        interval_upper=center + width, actual_constitutive_gain=actual,
        robust_sign=bool(abs(center) > width), sharp_outer_product_extremizers=extremizers,
        selected_families=[e["family"] for e in expressions],
        scope="Both true Exp branches lie outside the polynomial reference. Approximation and projected-consistency budgets are oracle diagnostics; candidate functions and models remain frozen.")
    rb.save(output / f"seed{seed}.json", record)
    return record


if __name__ == "__main__":
    output = rb.OUT / "mismatched_ellipsoid"; output.mkdir(exist_ok=False)
    records = [run(seed, output) for seed in range(5)]
    summary = dict(status="complete", case="Exp", seeds=5,
        all_five_intervals_contain_actual_gains=True,
        robust_rankings_at_oracle_budgets=sum(r["robust_sign"] for r in records),
        all_ten_outer_product_extremizers_attain_bounds=True,
        approximation_remainder_norm=records[0]["approximation_remainder_norm"],
        driver_sha256=rb.sha(__file__), completed_utc=datetime.now(timezone.utc).isoformat(),
        limitation="Budgets are evaluated from synthetic truth, not inferred by the algorithm.")
    rb.save(output / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
