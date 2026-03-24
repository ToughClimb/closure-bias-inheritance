from __future__ import annotations

import numpy as np


Array = np.ndarray


def _normalize_rows(values: Array, dx: float) -> tuple[Array, Array]:
    norms = np.sqrt(np.sum(values**2, axis=1, keepdims=True) * dx)
    norms = np.where(norms > 0.0, norms, 1.0)
    return values / norms, norms


def make_test_functions_1d(
    xs: Array,
    num_modes: int = 4,
    boundary: str = "periodic",
    include_constant: bool = True,
    num_bumps: int = 4,
) -> tuple[Array, Array]:
    """Return test functions phi_k and their spatial derivatives."""

    dx = xs[1] - xs[0]
    length = xs[-1] - xs[0] + dx

    if boundary == "periodic":
        phi_list = []
        grad_list = []
        if include_constant:
            phi_list.append(np.ones_like(xs))
            grad_list.append(np.zeros_like(xs))
        for mode in range(1, num_modes + 1):
            wave = 2.0 * np.pi * mode / length
            sin_values = np.sin(wave * xs)
            cos_values = np.cos(wave * xs)
            phi_list.extend([sin_values, cos_values])
            grad_list.extend([wave * cos_values, -wave * sin_values])

        if num_bumps > 0:
            sigma = length / max(2.0 * num_bumps, 1.0)
            for center in np.linspace(0.0, length, num_bumps, endpoint=False):
                offset = ((xs - center + 0.5 * length) % length) - 0.5 * length
                bump = np.exp(-0.5 * (offset / sigma) ** 2)
                grad_bump = -(offset / (sigma**2)) * bump
                phi_list.append(bump)
                grad_list.append(grad_bump)
        phi = np.stack(phi_list, axis=0)
        grad_phi = np.stack(grad_list, axis=0)
    elif boundary == "neumann":
        phi_list = []
        grad_list = []
        if include_constant:
            phi_list.append(np.ones_like(xs))
            grad_list.append(np.zeros_like(xs))
        for mode in range(1, num_modes + 1):
            wave = np.pi * mode / length
            cos_values = np.cos(wave * xs)
            phi_list.append(cos_values)
            grad_list.append(-wave * np.sin(wave * xs))

        if num_bumps > 0:
            sigma = length / max(2.0 * num_bumps, 1.0)
            for center in np.linspace(0.0, length, num_bumps + 2, endpoint=True)[1:-1]:
                offset = xs - center
                bump = np.exp(-0.5 * (offset / sigma) ** 2)
                grad_bump = -(offset / (sigma**2)) * bump
                phi_list.append(bump)
                grad_list.append(grad_bump)
        phi = np.stack(phi_list, axis=0)
        grad_phi = np.stack(grad_list, axis=0)
    else:
        raise ValueError(f"Unsupported boundary condition: {boundary}")

    phi, norms = _normalize_rows(phi, dx)
    grad_phi = grad_phi / norms
    return phi, grad_phi


def make_time_test_functions_1d(
    ts: Array,
    num_modes: int = 4,
) -> tuple[Array, Array]:
    """Return temporal test functions psi_m and their time derivatives."""

    if ts.ndim != 1:
        raise ValueError("ts must be one-dimensional")
    if len(ts) < 2:
        raise ValueError("at least two saved times are required")
    if num_modes < 1:
        raise ValueError("num_modes must be positive")

    dt = float(ts[1] - ts[0])
    duration = float(ts[-1] - ts[0])
    shifted = ts - ts[0]
    psi_list = []
    dpsi_list = []
    for mode in range(1, num_modes + 1):
        wave = np.pi * mode / max(duration, dt)
        psi = np.sin(wave * shifted)
        dpsi = wave * np.cos(wave * shifted)
        psi_list.append(psi)
        dpsi_list.append(dpsi)

    psi = np.stack(psi_list, axis=0)
    dpsi = np.stack(dpsi_list, axis=0)
    psi, norms = _normalize_rows(psi, dt)
    dpsi = dpsi / norms
    return psi, dpsi
