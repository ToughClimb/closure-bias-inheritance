import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from slit_flow_core import (central_shear, exact_fluidity, exact_velocity, exact_flow_rate,
                            weighted_projection, polynomial, observation_fit, integration_nodes,
                            oscillatory_counterexample, oscillatory_derivative, geometry)


def test_analytic_velocity_and_central_difference_bias():
    y = np.linspace(0, 1, 33)
    pressure, alpha = 0.75, 0.6
    observed = exact_velocity(y, pressure, alpha)
    assert observed[-1] == 0
    expected = exact_fluidity(pressure * y[1:-1], alpha) + alpha * pressure**3 * y[1:-1] / 32**2
    np.testing.assert_allclose(central_shear(y, observed), expected, atol=4e-15)
    recovered, _ = observation_fit(y, exact_velocity(y, alpha=alpha))
    np.testing.assert_allclose(recovered, [1, alpha], atol=2e-15)
    reconstructed = np.array([quad(lambda s: float(exact_fluidity(pressure * s, alpha)), x, 1)[0] for x in y])
    np.testing.assert_allclose(reconstructed, observed, atol=2e-15)


def test_projection_removes_only_the_orthogonal_component():
    x, w = leggauss(96)
    x, w = (x + 1) / 2, w / 2
    raw = x * np.sin(6 * np.pi * x)
    perpendicular = raw - polynomial(x, weighted_projection(x, raw, w))
    truth = exact_fluidity(x)
    teacher = truth + 0.02 * x + 0.03 * perpendicular
    compressed = polynomial(x, weighted_projection(x, teacher, w))
    np.testing.assert_allclose(compressed, truth + 0.02 * x, atol=2e-15)
    result = geometry(truth, teacher, compressed, w)
    assert result['geometry_identity_residual'] < 1e-17


def test_training_quadrature_matches_known_integral_operator():
    y = np.linspace(0, 1, 19)
    points, weights = integration_nodes(y, pressure=1.25, order=32)
    stress = torch.from_numpy(points)
    prediction = ((stress + 0.6 * stress**3) * torch.from_numpy(weights)).sum(dim=1).numpy()
    np.testing.assert_allclose(prediction, exact_velocity(y, 1.25), atol=4e-16)
    rate = 2 * quad(lambda s: s * float(exact_fluidity(1.25 * s)), 0, 1)[0]
    assert abs(rate - exact_flow_rate(1.25)) < 3e-15


@pytest.mark.parametrize('cycles',[4,16,64,256])
def test_small_positive_counterexample_loses_incremental_monotonicity(cycles):
    stress = np.linspace(0, 1, 16385)
    candidate = oscillatory_counterexample(stress, cycles)
    assert candidate[0] == 0 and np.min(candidate[1:]) > 0
    # This explicit witness is a global failure certificate, not a sampled assertion.
    assert abs(float(oscillatory_derivative(np.array([1 / (2 * cycles)]), cycles)[0]) + 0.9) < 1e-14
    frequency = 2 * np.pi * cycles
    rate = 2 * quad(lambda t: t * float(oscillatory_counterexample(t, cycles)), 0, 1,
                    points=np.linspace(0,1,2*cycles+1)[1:-1],epsabs=1e-12,epsrel=1e-12,limit=1024)[0]
    assert abs(rate - (2 * 1.1 / 3 - 4 / frequency**2)) < 1e-12
