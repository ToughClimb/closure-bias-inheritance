from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from closure_discovery.data_generation.cases import ReactionDiffusionCase
from closure_discovery.data_generation.observations import apply_observation_model
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D, generate_dataset
from closure_discovery.evaluation.metrics import relative_l2_error, summarize_excitation
from closure_discovery.evaluation.rollout import compare_cases_on_shared_initial_conditions
from closure_discovery.pipelines.train_1d_closure import ObservationConfig, UnseenRolloutConfig
from closure_discovery.weak_form.test_functions import make_test_functions_1d


Array = np.ndarray


@dataclass(frozen=True)
class PolynomialClosure:
    diffusion_coefficients: Array
    reaction_coefficients: Array
    value_range: tuple[float, float]
    name: str
    description: str

    def diffusion(self, u: Array) -> Array:
        u = np.asarray(u)
        output = np.zeros_like(u, dtype=np.float64)
        for degree, coefficient in enumerate(self.diffusion_coefficients):
            output += coefficient * u**degree
        return output

    def reaction(self, u: Array) -> Array:
        u = np.asarray(u)
        output = np.zeros_like(u, dtype=np.float64)
        for degree, coefficient in enumerate(self.reaction_coefficients):
            output += coefficient * u**degree
        return output

    def to_case(self) -> ReactionDiffusionCase:
        return ReactionDiffusionCase(
            name=self.name,
            description=self.description,
            diffusion=self.diffusion,
            reaction=self.reaction,
            value_range=self.value_range,
        )


def _central_time_derivative(u: Array, dt: float) -> Array:
    return (u[:, 2:, :] - u[:, :-2, :]) / (2.0 * dt)


def _spatial_gradient(u: Array, dx: float, boundary: str = "periodic") -> Array:
    if boundary == "periodic":
        return (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2.0 * dx)
    if boundary == "neumann":
        gradient = np.zeros_like(u)
        gradient[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * dx)
        return gradient
    raise ValueError(f"Unsupported boundary condition: {boundary}")


def _diffusion_basis_term(u: Array, power: int, dx: float, boundary: str = "periodic") -> Array:
    basis_values = u**power
    if boundary == "periodic":
        right_u = np.roll(u, -1, axis=-1)
        right_basis = np.roll(basis_values, -1, axis=-1)
        flux_right = 0.5 * (basis_values + right_basis) * (right_u - u) / dx
        flux_left = np.roll(flux_right, 1, axis=-1)
        return (flux_right - flux_left) / dx
    if boundary == "neumann":
        flux = np.zeros((*u.shape[:-1], u.shape[-1] + 1), dtype=u.dtype)
        left_u = u[..., :-1]
        right_u = u[..., 1:]
        left_basis = basis_values[..., :-1]
        right_basis = basis_values[..., 1:]
        flux[..., 1:-1] = 0.5 * (left_basis + right_basis) * (right_u - left_u) / dx
        return (flux[..., 1:] - flux[..., :-1]) / dx
    raise ValueError(f"Unsupported boundary condition: {boundary}")


def _solve_ridge_least_squares(design: Array, target: Array, ridge: float) -> tuple[Array, float]:
    column_scales = np.linalg.norm(design, axis=0)
    column_scales = np.where(column_scales > 1.0e-12, column_scales, 1.0)
    scaled_design = design / column_scales

    if ridge > 0.0:
        identity = np.sqrt(ridge) * np.eye(scaled_design.shape[1], dtype=scaled_design.dtype)
        augmented_design = np.concatenate([scaled_design, identity], axis=0)
        augmented_target = np.concatenate([target, np.zeros(scaled_design.shape[1], dtype=target.dtype)], axis=0)
    else:
        augmented_design = scaled_design
        augmented_target = target

    scaled_coefficients, _, _, _ = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)
    coefficients = scaled_coefficients / column_scales
    residual = design @ coefficients - target
    return coefficients, float(np.mean(residual**2))


def fit_strong_form_polynomial_closure(
    dataset: dict[str, Array],
    *,
    value_range: tuple[float, float],
    diffusion_degree: int = 2,
    reaction_degree: int = 3,
    boundary: str = "periodic",
    ridge: float = 1.0e-8,
    name: str = "strong_poly_baseline",
) -> tuple[PolynomialClosure, float]:
    dx = float(dataset["x"][1] - dataset["x"][0])
    dt = float(dataset["t"][1] - dataset["t"][0])
    u = np.asarray(dataset["u"], dtype=np.float64)
    u_mid = u[:, 1:-1, :]
    ut = _central_time_derivative(u, dt)

    design_columns = []
    for degree in range(diffusion_degree + 1):
        design_columns.append(_diffusion_basis_term(u_mid, power=degree, dx=dx, boundary=boundary))
    for degree in range(reaction_degree + 1):
        design_columns.append(u_mid**degree)

    design = np.stack(design_columns, axis=-1).reshape(-1, diffusion_degree + reaction_degree + 2)
    target = ut.reshape(-1)
    coefficients, fit_mse = _solve_ridge_least_squares(design, target, ridge=ridge)

    return (
        PolynomialClosure(
            diffusion_coefficients=coefficients[: diffusion_degree + 1],
            reaction_coefficients=coefficients[diffusion_degree + 1 :],
            value_range=value_range,
            name=name,
            description="Strong-form polynomial closure baseline",
        ),
        fit_mse,
    )


def fit_weak_form_polynomial_closure(
    dataset: dict[str, Array],
    *,
    value_range: tuple[float, float],
    diffusion_degree: int = 2,
    reaction_degree: int = 3,
    boundary: str = "periodic",
    ridge: float = 1.0e-8,
    num_test_modes: int = 4,
    num_bump_functions: int = 4,
    name: str = "weak_poly_baseline",
) -> tuple[PolynomialClosure, float]:
    dx = float(dataset["x"][1] - dataset["x"][0])
    dt = float(dataset["t"][1] - dataset["t"][0])
    u = np.asarray(dataset["u"], dtype=np.float64)
    u_mid = u[:, 1:-1, :]
    ut = _central_time_derivative(u, dt)
    ux = _spatial_gradient(u_mid, dx=dx, boundary=boundary)
    phi, grad_phi = make_test_functions_1d(
        dataset["x"],
        num_modes=num_test_modes,
        boundary=boundary,
        num_bumps=num_bump_functions,
    )

    term_ut = np.einsum("kx,btx->btk", phi, ut) * dx
    design_columns = []
    for degree in range(diffusion_degree + 1):
        diffusion_basis = (u_mid**degree) * ux
        term_diff = np.einsum("kx,btx->btk", grad_phi, diffusion_basis) * dx
        design_columns.append(term_diff)
    for degree in range(reaction_degree + 1):
        term_react = np.einsum("kx,btx->btk", phi, u_mid**degree) * dx
        design_columns.append(-term_react)

    design = np.stack(design_columns, axis=-1).reshape(-1, diffusion_degree + reaction_degree + 2)
    target = (-term_ut).reshape(-1)
    coefficients, fit_mse = _solve_ridge_least_squares(design, target, ridge=ridge)

    return (
        PolynomialClosure(
            diffusion_coefficients=coefficients[: diffusion_degree + 1],
            reaction_coefficients=coefficients[diffusion_degree + 1 :],
            value_range=value_range,
            name=name,
            description="Weak-form polynomial closure baseline",
        ),
        fit_mse,
    )


def run_polynomial_baseline(
    *,
    method: str,
    case,
    simulation_config: SimulationConfig1D,
    num_trajectories: int,
    amplitude_range: tuple[float, float],
    num_initial_modes: int = 4,
    observation_config: ObservationConfig | None = None,
    unseen_rollout_config: UnseenRolloutConfig | None = None,
    diffusion_degree: int = 2,
    reaction_degree: int = 3,
    ridge: float = 1.0e-8,
    seed: int = 0,
    raw_dataset: dict[str, Array] | None = None,
    num_test_modes: int = 4,
    num_bump_functions: int = 4,
) -> dict[str, Any]:
    observation_config = observation_config or ObservationConfig()
    unseen_rollout_config = unseen_rollout_config or UnseenRolloutConfig()

    if raw_dataset is None:
        dataset = generate_dataset(
            case=case,
            config=simulation_config,
            num_trajectories=num_trajectories,
            seed=seed,
            amplitude_range=amplitude_range,
            num_modes=num_initial_modes,
        )
    else:
        dataset = {
            "x": raw_dataset["x"].copy(),
            "t": raw_dataset["t"].copy(),
            "u0": raw_dataset["u0"].copy(),
            "u": raw_dataset["u"].copy(),
        }

    observed = apply_observation_model(
        dataset,
        space_stride=observation_config.space_stride,
        time_stride=observation_config.time_stride,
        noise_level=observation_config.noise_level,
        seed=seed,
        clip_range=case.value_range,
    )
    dx = float(observed["x"][1] - observed["x"][0])
    excitation = summarize_excitation(observed["u"], dx=dx, value_range=case.value_range)

    if method == "strong":
        closure, fit_mse = fit_strong_form_polynomial_closure(
            observed,
            value_range=case.value_range,
            diffusion_degree=diffusion_degree,
            reaction_degree=reaction_degree,
            boundary=simulation_config.boundary,
            ridge=ridge,
        )
    elif method == "weak":
        closure, fit_mse = fit_weak_form_polynomial_closure(
            observed,
            value_range=case.value_range,
            diffusion_degree=diffusion_degree,
            reaction_degree=reaction_degree,
            boundary=simulation_config.boundary,
            ridge=ridge,
            num_test_modes=num_test_modes,
            num_bump_functions=num_bump_functions,
        )
    else:
        raise ValueError(f"Unsupported polynomial baseline method: {method}")

    learned_case = closure.to_case()
    evaluation_grid = np.linspace(case.value_range[0], case.value_range[1], 256, dtype=np.float64)
    diffusion_true = case.diffusion(evaluation_grid)
    reaction_true = case.reaction(evaluation_grid)
    diffusion_pred = closure.diffusion(evaluation_grid)
    reaction_pred = closure.reaction(evaluation_grid)

    rollout_config = unseen_rollout_config.simulation_config or simulation_config
    rollout_amplitude_range = unseen_rollout_config.amplitude_range or amplitude_range
    rollout = compare_cases_on_shared_initial_conditions(
        true_case=case,
        predicted_case=learned_case,
        config=rollout_config,
        num_trajectories=unseen_rollout_config.num_trajectories,
        seed=seed + unseen_rollout_config.seed_offset,
        amplitude_range=rollout_amplitude_range,
        num_modes=unseen_rollout_config.num_modes,
    )

    metrics = {
        "fit_mse": fit_mse,
        "relative_error_D": relative_l2_error(diffusion_true, diffusion_pred),
        "relative_error_R": relative_l2_error(reaction_true, reaction_pred),
        "unseen_rollout_mse": rollout.mse,
        "unseen_rollout_relative_l2": rollout.relative_l2,
    }

    return {
        "method": method,
        "case_name": case.name,
        "seed": seed,
        "dataset": observed,
        "excitation": excitation,
        "closure": closure,
        "evaluation_grid": evaluation_grid,
        "diffusion_true": diffusion_true,
        "diffusion_pred": diffusion_pred,
        "reaction_true": reaction_true,
        "reaction_pred": reaction_pred,
        "metrics": metrics,
        "rollout": rollout,
    }
