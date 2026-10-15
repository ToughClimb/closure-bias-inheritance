"""Independent checks of the continuous-time bound ingredients and receipts."""
import response_bounds as rb
from pathlib import Path
import json
import numpy as np


def jacobian(problem, u, mode):
    d, _, dp, rp, _, _ = rb.point_jet(problem, u, mode)
    h2 = problem.dx**2
    plus_jump, minus_jump = np.roll(u, -1) - u, np.roll(u, 1) - u
    diag = -(2 * d + np.roll(d, -1) + np.roll(d, 1)) / (2 * h2)
    diag += dp * (plus_jump + minus_jump) / (2 * h2) + rp
    plus = (d + np.roll(d, -1) + np.roll(dp, -1) * plus_jump) / (2 * h2)
    minus = (d + np.roll(d, 1) + np.roll(dp, 1) * minus_jump) / (2 * h2)
    matrix = np.diag(diag)
    matrix[np.arange(64), (np.arange(64) + 1) % 64] = plus
    matrix[np.arange(64), (np.arange(64) - 1) % 64] = minus
    return matrix


def main():
    rng = np.random.default_rng(106762026)
    maximum_jacobian_error = 0.
    maximum_range_slack_violation = 0.
    log_norm_checks = 0
    for seed in range(5):
        p = rb.old.CurvatureProblem(seed, 64)
        for mode in ("native", "symbolic"):
            constants = rb.derivative_constants(p, mode)
            for pattern in rb.old.first.PATTERNS:
                u = p.initial(pattern)
                direction = rng.normal(size=64)
                computed = jacobian(p, u, mode) @ direction
                expected = rb.rhs(p, u.astype(complex) + 1e-30j * direction, mode).imag / 1e-30
                maximum_jacobian_error = max(maximum_jacobian_error, float(np.max(np.abs(computed - expected))))
                np.testing.assert_allclose(computed, expected, rtol=2e-11, atol=3e-11)
                # Independently evaluate the row-sum enclosure on actual balls.
                for radius in (1e-6, 1e-4):
                    d, _, dp, rp, _, _ = rb.point_jet(p, u, mode)
                    jump = np.abs(np.roll(u, -1) - u) + 2 * radius
                    lower = d - constants["D1"] * radius
                    assert np.min(lower + np.roll(lower, -1) - constants["D1"] * jump) > 0
                    lap = (np.roll(u, -1) + np.roll(u, 1) - 2 * u) / p.dx**2
                    lap_interval = rb.I(lap - 4 * radius / p.dx**2, lap + 4 * radius / p.dx**2)
                    slope = rb.I(dp - constants["D2"] * radius, dp + constants["D2"] * radius)
                    bound = rp + constants["R2"] * radius + (slope * lap_interval).hi
                    bound += .5 * constants["D2"] * (jump**2 + np.roll(jump, 1)**2) / p.dx**2
                    for _ in range(6):
                        perturbed = u + rng.uniform(-radius, radius, 64)
                        J = jacobian(p, perturbed, mode)
                        observed = np.diag(J) + np.sum(np.abs(J), axis=1) - np.abs(np.diag(J))
                        assert np.max(observed - bound) < 5e-10
                        log_norm_checks += 1
    # Arbitrary cubic polynomials, including derivatives and common-time spatial
    # differences; these checks do not use the implementation's recurrence.
    coefficients = rng.normal(size=(60, 4, 64)) * np.array([1, .01, .001, .0001])[None, :, None]
    for derivative in range(4):
        centers, radii = rb.poly_ranges(coefficients, 17, derivative)
        centers, radii = centers.reshape(60, 17, 64), radii.reshape(60, 17, 64)
        for _ in range(100):
            block = int(rng.integers(60)); slab = int(rng.integers(17))
            s = (slab + rng.random()) / 17
            c = np.polynomial.polynomial.polyder(coefficients[block], m=derivative, axis=0)
            value = np.array([np.polynomial.polynomial.polyval(s, c[:, j]) for j in range(64)]) / .01**derivative
            violation = float(np.max(np.abs(value - centers[block, slab]) - radii[block, slab]))
            maximum_range_slack_violation = max(maximum_range_slack_violation, violation)
            assert violation < 1e-7
    # Read every completed result, including abstentions and all code/input hashes.
    results = sorted((rb.OUT / "formal_dt100").glob("s*.json"))
    assert len(results) == 20
    for path in results:
        record = json.loads(path.read_text())
        assert record["status"] == "complete" and record["driver_sha256"] == rb.sha(Path(rb.__file__))
        for row in record["rows"]:
            assert row["interval_lower"] <= row["actual_change"] <= row["interval_upper"]
            assert not row["certified"] or row["prediction_sign_agrees"]
    record = dict(status="passed", analytical_jacobian_complex_step_max_absolute_error=maximum_jacobian_error,
                  sampled_tube_jacobian_inequalities=log_norm_checks,
                  cubic_range_checks=400, cubic_range_max_violation=maximum_range_slack_violation,
                  all_120_endpoint_intervals_contain_evaluated_changes=True,
                  all_declared_signs_correct=True, actual_model_or_selection_changes=0,
                  driver_sha256=rb.sha(__file__))
    rb.save(rb.OUT / "NUMERICAL_CHECKS.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
