"""Controlled inverse constitutive experiment for steady planar slit flow.

The learned function is inverse rheology: shear rate Phi(stress), not velocity
or a reaction--diffusion coefficient. Truth is restricted to manufactured data
and evaluation. All training objectives receive saved observations only.
"""
from __future__ import annotations

import math
import numpy as np
from numpy.polynomial.legendre import leggauss
import torch


def exact_fluidity(stress, alpha=0.6):
    stress = np.asarray(stress, dtype=np.float64)
    return stress + alpha * stress**3


def exact_velocity(y, pressure=1.0, alpha=0.6):
    y = np.asarray(y, dtype=np.float64)
    return pressure * (1 - y**2) / 2 + alpha * pressure**3 * (1 - y**4) / 4


def exact_flow_rate(pressure=1.0, alpha=0.6):
    return 2 * pressure / 3 + 2 * alpha * pressure**3 / 5


def central_shear(y, velocity):
    y = np.asarray(y, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    h = np.diff(y)
    if y.ndim != 1 or len(y) < 5 or velocity.shape != y.shape:
        raise ValueError('Need matching one-dimensional observations on at least five nodes')
    if not np.isfinite(velocity).all() or not np.allclose(h, h[0], rtol=1e-12, atol=1e-14) or h[0] <= 0:
        raise ValueError('Need a finite velocity and a uniform increasing grid')
    return -(velocity[2:] - velocity[:-2]) / (2 * h[0])


def basis(stress, degree=3):
    if degree not in (1, 3):
        raise ValueError('The declared families are linear and odd cubic')
    stress = np.asarray(stress, dtype=np.float64)
    return np.column_stack([stress**p for p in range(1, degree + 1, 2)])


def velocity_basis(y, pressure=1.0, degree=3):
    y = np.asarray(y, dtype=np.float64)
    return np.column_stack([pressure**p * (1 - y**(p + 1)) / (p + 1)
                            for p in range(1, degree + 1, 2)])


def polynomial(stress, coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    return basis(stress, degree=2 * len(coefficients) - 1) @ coefficients


def polynomial_derivative(stress, coefficients):
    stress = np.asarray(stress, dtype=np.float64)
    return sum((2 * i + 1) * c * stress**(2 * i) for i, c in enumerate(coefficients))


def weighted_projection(stress, values, weights, degree=3):
    design = basis(stress, degree)
    root = np.sqrt(np.asarray(weights))
    coefficients, _, rank, _ = np.linalg.lstsq(root[:, None] * design, root * values, rcond=None)
    if rank != design.shape[1]:
        raise ValueError('Projection design is rank deficient')
    return coefficients


def observation_fit(y, velocity, degree=3):
    """Global fixed-family observation refit; also the direct symbolic fit.

    This is one estimator, not two independent methods. It does not receive
    the manufactured alpha or the teacher coefficients.
    """
    design = velocity_basis(y, degree=degree)
    coefficients, _, rank, singular = np.linalg.lstsq(design, velocity, rcond=None)
    if rank != design.shape[1]:
        raise ValueError('Observation design is rank deficient')
    return coefficients, singular


def integration_nodes(y, pressure=1.0, order=32):
    nodes, weights = leggauss(order)
    y = np.asarray(y, dtype=np.float64)
    points = y[:, None] + (1 - y[:, None]) * (nodes[None, :] + 1) / 2
    quadrature = (1 - y[:, None]) * weights[None, :] / 2
    return pressure * points, quadrature


class FluidityNetwork(torch.nn.Module):
    """Hard zero response and positive fluidity for positive stress.

    Incremental monotonicity is not imposed; it is evaluated separately.
    """
    def __init__(self, width=16):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(1, width), torch.nn.Tanh(),
                                           torch.nn.Linear(width, width), torch.nn.Tanh(),
                                           torch.nn.Linear(width, 1))

    def forward(self, stress):
        original = stress.shape
        raw = self.network((2 * stress.reshape(-1, 1) - 1)).reshape(original)
        return stress * (torch.nn.functional.softplus(raw) + 1e-4)


def train_teacher(y, velocity, *, objective, init_seed, epochs=1500, learning_rate=0.005, width=16, order=32):
    if objective not in ('differential', 'integral'):
        raise ValueError(objective)
    torch.manual_seed(init_seed)
    torch.set_num_threads(1)
    model = FluidityNetwork(width).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if objective == 'differential':
        inputs = torch.as_tensor(np.asarray(y)[1:-1], dtype=torch.float64)
        target = torch.as_tensor(central_shear(y, velocity), dtype=torch.float64)
        quadrature = None
    else:
        points, weights = integration_nodes(y, order=order)
        inputs = torch.as_tensor(points, dtype=torch.float64)
        quadrature = torch.as_tensor(weights, dtype=torch.float64)
        target = torch.as_tensor(velocity, dtype=torch.float64)

    def loss():
        prediction = model(inputs)
        if quadrature is not None:
            prediction = (prediction * quadrature).sum(dim=1)
        return (prediction - target).square().mean()

    history = []
    for epoch in range(epochs + 1):
        if epoch:
            optimizer.zero_grad()
            current = loss()
            current.backward()
            if not torch.isfinite(current) or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite training loss or gradient')
            optimizer.step()
        if epoch % 300 == 0 or epoch == epochs:
            optimizer.zero_grad()
            current = loss()
            current.backward()
            gradient = math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters()))
            history.append({'epoch':epoch, 'loss':float(current.detach()), 'gradient_norm':gradient})
    return model, history


def network_values(model, stress, *, derivative=False):
    inputs = torch.tensor(np.asarray(stress), dtype=torch.float64, requires_grad=derivative)
    if derivative:
        values = model(inputs)
        slope = torch.autograd.grad(values.sum(), inputs)[0]
        return values.detach().numpy(), slope.detach().numpy()
    with torch.no_grad():
        return model(inputs).numpy()


def velocity_prediction(function, y, pressure=1.0, order=64):
    points, weights = integration_nodes(y, pressure, order)
    return np.sum(function(points.ravel()).reshape(points.shape) * weights, axis=1)


def flow_prediction(function, pressure=1.0, order=128):
    nodes, weights = leggauss(order)
    y, weights = (nodes + 1) / 2, weights / 2
    return float(2 * np.sum(weights * y * function(pressure * y)))


def geometry(truth, teacher, compressed, weights):
    a = teacher - truth
    c = compressed - teacher
    sq = lambda x: float(np.sum(weights * x**2))
    en, ec, es = math.sqrt(sq(a)), math.sqrt(sq(c)), math.sqrt(sq(a + c))
    alignment = float(np.sum(weights * a * c)) / (en * ec) if en * ec > 1e-14 else None
    return dict(teacher_error=en, compression_error=ec, symbolic_error=es,
                rho=ec / en if en > 1e-14 else None, bir=es / en if en > 1e-14 else None,
                alignment=alignment, geometry_identity_residual=abs(es**2 - en**2 - ec**2 - 2 * float(np.sum(weights * a * c))))


def oscillatory_counterexample(stress, cycles, teacher_slope=1.1):
    frequency = 2 * np.pi * cycles
    amplitude = 2 / frequency
    stress = np.asarray(stress, dtype=np.float64)
    return teacher_slope * stress + amplitude * np.sin(frequency * stress)


def oscillatory_derivative(stress, cycles, teacher_slope=1.1):
    return teacher_slope + 2 * np.cos(2 * np.pi * cycles * np.asarray(stress))
