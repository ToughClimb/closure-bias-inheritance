from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ExcitationSummary:
    state_min: float
    state_max: float
    state_bin_coverage: float
    gradient_rms: float
    curvature_rms: float
    weak_diffusion_energy: float


def relative_l2_error(truth: Array, prediction: Array) -> float:
    numerator = np.linalg.norm(truth - prediction)
    denominator = np.linalg.norm(truth) + 1.0e-12
    return float(numerator / denominator)


def mean_squared_error(truth: Array, prediction: Array) -> float:
    return float(np.mean((truth - prediction) ** 2))


def pairwise_relative_l2_dispersion(values: Array) -> float:
    values = np.asarray(values)
    if values.shape[0] < 2:
        return 0.0

    discrepancies = []
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[0]):
            discrepancies.append(relative_l2_error(values[left], values[right]))
    return float(np.mean(discrepancies))


def _periodic_gradient(u: Array, dx: float) -> Array:
    return (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2.0 * dx)


def _periodic_curvature(u: Array, dx: float) -> Array:
    return (np.roll(u, -1, axis=-1) - 2.0 * u + np.roll(u, 1, axis=-1)) / (dx**2)


def state_bin_coverage(
    values: Array,
    value_range: tuple[float, float] | None = None,
    num_bins: int = 32,
) -> float:
    flat = np.asarray(values).reshape(-1)
    if value_range is None:
        lower = float(np.min(flat))
        upper = float(np.max(flat))
    else:
        lower, upper = value_range

    if upper <= lower:
        return 1.0

    hist, _ = np.histogram(flat, bins=num_bins, range=(lower, upper))
    return float(np.count_nonzero(hist) / num_bins)


def summarize_excitation(
    u: Array,
    dx: float,
    value_range: tuple[float, float] | None = None,
    num_bins: int = 32,
) -> ExcitationSummary:
    """
    Summarize how strongly a dataset excites the closure terms.

    Parameters
    ----------
    u:
        Array with shape [..., space].
    """

    u = np.asarray(u)
    gradient = _periodic_gradient(u, dx=dx)
    curvature = _periodic_curvature(u, dx=dx)
    diffusion_energy = np.mean(np.abs(gradient * curvature))

    return ExcitationSummary(
        state_min=float(np.min(u)),
        state_max=float(np.max(u)),
        state_bin_coverage=state_bin_coverage(u, value_range=value_range, num_bins=num_bins),
        gradient_rms=float(np.sqrt(np.mean(gradient**2))),
        curvature_rms=float(np.sqrt(np.mean(curvature**2))),
        weak_diffusion_energy=float(diffusion_energy),
    )
