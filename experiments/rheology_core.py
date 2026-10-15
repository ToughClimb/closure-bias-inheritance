"""Constitutive reliability controls for measured steady-shear rheometry.

Viscosity uses Pa s, shear rate 1/s, temperature Celsius at the public API.
No routine treats a held-out measurement as exact constitutive ground truth.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import torch


def features(temperature, rate, temperature_bounds=(35., 125.), rate_bounds=(1., 50.)):
    temperature, rate = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(rate, float))
    if not np.isfinite(temperature).all() or not np.isfinite(rate).all() or np.any(rate <= 0) or np.any(temperature <= -273.15):
        raise ValueError('Finite positive absolute temperatures and shear rates required')
    inv_bounds = 1 / (np.array(temperature_bounds, dtype=float) + 273.15)
    lx_bounds = np.log(rate_bounds)
    x = (2 / (temperature + 273.15) - inv_bounds.sum()) / (inv_bounds[0] - inv_bounds[1])
    z = (2 * np.log(rate) - lx_bounds.sum()) / (lx_bounds[1] - lx_bounds[0])
    return np.stack([x, z], axis=-1)


def powers(degree):
    if degree not in (1, 2, 3, 4, 6, 8):
        raise ValueError('Declared initial degrees 1/2/3 and complexity follow-up 4/6/8')
    return [(i, d - i) for d in range(degree + 1) for i in range(d + 1)]


def design(x, degree):
    x = np.asarray(x, float)
    return np.stack([x[..., 0]**i * x[..., 1]**j for i, j in powers(degree)], axis=-1)


def fit_log_polynomial(x, log_viscosity, degree):
    a = design(x, degree)
    coef, _, rank, singular = np.linalg.lstsq(a, np.asarray(log_viscosity), rcond=None)
    if rank != a.shape[-1]:
        raise ValueError('Unidentifiable polynomial coefficients')
    return coef, singular


def polynomial_log_viscosity(x, coefficients, degree):
    return design(x, degree) @ np.asarray(coefficients)


def polynomial_stress_margin(x, coefficients, degree, rate_bounds=(1., 50.)):
    """tau'(gamma)/eta = 1 + partial log(eta)/partial log(gamma)."""
    x = np.asarray(x, float)
    dz = 2 / np.log(rate_bounds[1] / rate_bounds[0])
    return 1 + sum(c * j * x[..., 0]**i * x[..., 1]**(j-1) * dz
                   for c, (i, j) in zip(coefficients, powers(degree)) if j)


class ViscosityNetwork(torch.nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.Tanh(),
                                      torch.nn.Linear(width, width), torch.nn.Tanh(),
                                      torch.nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def whole_replicate_split(replicate_ids, held_out):
    ids = np.asarray(replicate_ids)
    test = ids == held_out
    if not test.any() or test.all():
        raise ValueError('Need observations in both held-out and training replicates')
    return ~test, test


def threshold_crossing(temperatures, viscosities, threshold=.4):
    """Piecewise log-viscosity interpolation, only inside measured brackets.

    Multiple crossings remain explicit and are not arbitrarily selected.
    A value exactly on the threshold is retained once.
    """
    t, eta = np.asarray(temperatures, float), np.asarray(viscosities, float)
    if len(t) < 2 or t.shape != eta.shape or np.any(np.diff(t) <= 0) or np.any(eta <= 0) or not np.isfinite(eta).all():
        raise ValueError('Need increasing measured temperatures and positive finite viscosity')
    r = np.log(eta / threshold)
    crossings = [float(ti) for ti, ri in zip(t, r) if ri == 0]
    for i in range(len(t)-1):
        if r[i] * r[i+1] < 0:
            crossings.append(float(t[i] - r[i] * (t[i+1] - t[i]) / (r[i+1] - r[i])))
    crossings = sorted(set(crossings))
    status = 'unique' if len(crossings) == 1 else 'unbracketed' if not crossings else 'multiple'
    return {'status': status, 'crossings_celsius': crossings}


def interpolate_log_rate(rates, viscosities, target_rate=20.):
    r, eta = np.asarray(rates, float), np.asarray(viscosities, float)
    if np.any(np.diff(r) <= 0) or np.any(eta <= 0) or not np.isfinite(eta).all():
        raise ValueError('Need increasing shear rates and positive finite viscosity')
    if target_rate < r[0] or target_rate > r[-1]:
        raise ValueError('Shear-rate extrapolation forbidden')
    return float(np.exp(np.interp(np.log(target_rate), np.log(r), np.log(eta))))


@lru_cache(maxsize=16)
def _legendre_rule(order):
    return np.polynomial.legendre.leggauss(order)


def slit_flow_band(log_eta, *, min_rate, max_rate, wall_stress, half_gap=.0005, quadrature_order=128):
    """Conditional Q/band for the unsupported low-shear central region.

    Requires monotone stress on the measured-rate interval (sampled check).
    Outside that interval, assumes a monotone inverse extension Phi(0)=0.
    No constant-viscosity or power-law extrapolation is silently imposed.
    No measured flow observation is available in the source dataset.
    """
    rates = np.geomspace(min_rate, max_rate, 2049)
    stress = rates * np.exp(log_eta(rates))
    if not np.isfinite(stress).all() or np.any(np.diff(stress) <= 0):
        return {'status': 'nonmonotone_stress', 'sampled_monotonicity_points': len(rates)}
    tau_min, tau_max = float(stress[0]), float(stress[-1])
    if not (tau_min < wall_stress <= tau_max):
        return {'status': 'wall_stress_outside_supported_interval', 'stress_min': tau_min, 'stress_max': tau_max}

    def inverse(tau):
        return brentq(lambda g: g * np.exp(float(log_eta(np.asarray(g)))) - tau,
                      min_rate, max_rate, xtol=1e-11, rtol=1e-13)

    independent, integration_error = quad(lambda z: z * inverse(z), tau_min, wall_stress,
                                         epsabs=1e-10, epsrel=2e-10, limit=150)
    initial_difference = None
    while True:
        qn, qw = _legendre_rule(quadrature_order)
        tau = tau_min + (qn + 1) * (wall_stress - tau_min) / 2
        phi = np.array([inverse(z) for z in tau])
        moment = float(np.sum(qw * tau * phi) * (wall_stress - tau_min) / 2)
        difference = abs(moment-independent)/max(abs(independent),1e-30)
        if initial_difference is None:
            initial_difference = difference
        if difference <= 2e-9 or quadrature_order >= 4096:
            break
        quadrature_order *= 2
    factor = 2 * half_gap**2 / wall_stress**2
    lower = factor * moment
    gap_width = half_gap**2 * min_rate * (tau_min / wall_stress)**2
    return {'status': 'valid_conditional_band', 'Q_lower_m2_per_s': lower,
            'Q_upper_m2_per_s': lower + gap_width, 'Q_unmeasured_core_width': gap_width,
            'stress_min': tau_min, 'stress_max': tau_max, 'wall_stress': float(wall_stress),
            'relative_quadrature_difference': difference,
            'initial_128_relative_quadrature_difference': initial_difference,
            'final_gauss_order': quadrature_order,
            'adaptive_integration_error': float(integration_error),
            'sampled_monotonicity_points': len(rates)}
