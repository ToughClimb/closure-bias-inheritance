from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

import numpy as np
import torch

from closure_discovery.data_generation.rd_solver_2d import (
    SimulationConfig2D,
    generate_dataset,
)
from closure_discovery.evaluation.metrics import relative_l2_error, state_bin_coverage
from closure_discovery.evaluation.rollout_2d import compare_cases_on_shared_initial_conditions_2d
from closure_discovery.models.mlp_closure import ReactionDiffusionClosure
from closure_discovery.models.tabulated_closure import tabulated_closure_from_model
from closure_discovery.weak_form.test_functions_2d import make_test_functions_2d
from closure_discovery.weak_form.weak_residual_2d import (
    mass_balance_loss_2d,
    one_step_rollout_loss_2d,
    strong_residual_loss_2d,
    weak_loss_2d,
)


@dataclass(frozen=True)
class ObservationConfig2D:
    noise_level: float = 0.0
    space_stride: int = 1
    time_stride: int = 1


@dataclass(frozen=True)
class TrainingConfig2D:
    epochs: int = 100
    lr: float = 1.0e-3
    rollout_weight: float = 0.1
    mass_weight: float = 1.0
    strong_weight: float = 1.0
    reaction_anchor_weight: float = 1.0
    reg_weight: float = 1.0e-4
    hidden_width: int = 64
    hidden_depth: int = 2
    diffusion_degree: int = 2
    reaction_degree: int = 3
    residual_scale: float = 0.1
    num_test_modes: int = 1


@dataclass(frozen=True)
class UnseenRolloutConfig2D:
    enabled: bool = True
    num_trajectories: int = 3
    seed_offset: int = 1000
    amplitude_range: tuple[float, float] | None = None
    initial_clip_range: tuple[float, float] | None = None
    num_modes: int = 3
    simulation_config: SimulationConfig2D | None = None


def make_paper_training_config_2d(
    *,
    epochs: int = 100,
    lr: float = 1.0e-3,
    hidden_width: int = 64,
    hidden_depth: int = 2,
    diffusion_degree: int = 2,
    reaction_degree: int = 3,
    residual_scale: float = 0.1,
    num_test_modes: int = 1,
) -> TrainingConfig2D:
    return TrainingConfig2D(
        epochs=epochs,
        lr=lr,
        rollout_weight=0.1,
        mass_weight=1.0,
        strong_weight=1.0,
        reaction_anchor_weight=1.0,
        reg_weight=1.0e-4,
        hidden_width=hidden_width,
        hidden_depth=hidden_depth,
        diffusion_degree=diffusion_degree,
        reaction_degree=reaction_degree,
        residual_scale=residual_scale,
        num_test_modes=num_test_modes,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_observation_model_2d(
    dataset: dict[str, np.ndarray],
    observation_config: ObservationConfig2D,
    *,
    seed: int,
    clip_range: tuple[float, float] | None,
) -> dict[str, np.ndarray]:
    stride = observation_config.space_stride
    observed_u = dataset["u"][:, :: observation_config.time_stride, ::stride, ::stride].copy()
    if observation_config.noise_level > 0.0:
        rng = np.random.default_rng(seed)
        scale = observation_config.noise_level * np.std(observed_u)
        observed_u = observed_u + rng.normal(loc=0.0, scale=scale, size=observed_u.shape)
        if clip_range is not None:
            observed_u = np.clip(observed_u, clip_range[0], clip_range[1])
    return {
        "x": dataset["x"][::stride].copy(),
        "y": dataset["y"][::stride].copy(),
        "t": dataset["t"][:: observation_config.time_stride].copy(),
        "u0": observed_u[:, 0, :, :].copy(),
        "u": observed_u,
    }


def run_closure_identification_2d(
    *,
    case,
    simulation_config: SimulationConfig2D,
    num_trajectories: int,
    amplitude_range: tuple[float, float],
    num_initial_modes: int = 3,
    observation_config: ObservationConfig2D | None = None,
    training_config: TrainingConfig2D | None = None,
    unseen_rollout_config: UnseenRolloutConfig2D | None = None,
    initial_clip_range: tuple[float, float] | None = None,
    seed: int = 0,
    device: str | torch.device | None = None,
    raw_dataset: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    observation_config = observation_config or ObservationConfig2D()
    training_config = training_config or make_paper_training_config_2d()
    unseen_rollout_config = unseen_rollout_config or UnseenRolloutConfig2D()
    set_seed(seed)

    if raw_dataset is None:
        dataset = generate_dataset(
            case=case,
            config=simulation_config,
            num_trajectories=num_trajectories,
            seed=seed,
            amplitude_range=amplitude_range,
            num_modes=num_initial_modes,
            initial_clip_range=initial_clip_range,
        )
    else:
        dataset = {
            "x": raw_dataset["x"].copy(),
            "y": raw_dataset["y"].copy(),
            "t": raw_dataset["t"].copy(),
            "u0": raw_dataset["u0"].copy(),
            "u": raw_dataset["u"].copy(),
        }

    observed = apply_observation_model_2d(
        dataset,
        observation_config,
        seed=seed,
        clip_range=case.value_range,
    )

    dx = float(observed["x"][1] - observed["x"][0])
    dy = float(observed["y"][1] - observed["y"][0])
    dt = float(observed["t"][1] - observed["t"][0])
    phi_np, grad_phi_x_np, grad_phi_y_np = make_test_functions_2d(
        observed["x"],
        observed["y"],
        num_modes=training_config.num_test_modes,
        boundary=simulation_config.boundary,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    u = torch.tensor(observed["u"], dtype=torch.float32, device=device)
    phi = torch.tensor(phi_np, dtype=torch.float32, device=device)
    grad_phi_x = torch.tensor(grad_phi_x_np, dtype=torch.float32, device=device)
    grad_phi_y = torch.tensor(grad_phi_y_np, dtype=torch.float32, device=device)

    model = ReactionDiffusionClosure(
        hidden_width=training_config.hidden_width,
        hidden_depth=training_config.hidden_depth,
        input_range=case.value_range,
        diffusion_degree=training_config.diffusion_degree,
        reaction_degree=training_config.reaction_degree,
        residual_scale=training_config.residual_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)

    u_min = float(case.value_range[0])
    u_max = float(case.value_range[1])
    support = torch.linspace(u_min, u_max, 256, device=device, dtype=torch.float32).requires_grad_(True)
    lower_anchor = torch.tensor([u_min], device=device, dtype=torch.float32)

    history: list[dict[str, float]] = []
    for epoch in range(1, training_config.epochs + 1):
        optimizer.zero_grad()
        weak = weak_loss_2d(
            u=u,
            dt=dt,
            dx=dx,
            dy=dy,
            phi=phi,
            grad_phi_x=grad_phi_x,
            grad_phi_y=grad_phi_y,
            closure_model=model,
            boundary=simulation_config.boundary,
        )
        rollout = one_step_rollout_loss_2d(
            u=u,
            dt=dt,
            dx=dx,
            dy=dy,
            closure_model=model,
            boundary=simulation_config.boundary,
        )
        mass = mass_balance_loss_2d(u=u, dt=dt, closure_model=model)
        strong = strong_residual_loss_2d(
            u=u,
            dt=dt,
            dx=dx,
            dy=dy,
            closure_model=model,
            boundary=simulation_config.boundary,
        )
        reaction_anchor = model.reaction(lower_anchor).pow(2).mean()
        reg = model.smoothness_penalty(support)
        loss = (
            weak
            + training_config.rollout_weight * rollout
            + training_config.mass_weight * mass
            + training_config.strong_weight * strong
            + training_config.reaction_anchor_weight * reaction_anchor
            + training_config.reg_weight * reg
        )
        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.item()),
                "weak_loss": float(weak.item()),
                "rollout_loss": float(rollout.item()),
                "mass_loss": float(mass.item()),
                "strong_loss": float(strong.item()),
                "reaction_anchor_loss": float(reaction_anchor.item()),
                "reg_loss": float(reg.item()),
            }
        )

    evaluation_grid = np.linspace(u_min, u_max, 256, dtype=np.float32)
    evaluation_tensor = torch.tensor(evaluation_grid, device=device)
    with torch.no_grad():
        diffusion_pred = model.diffusion(evaluation_tensor).cpu().numpy()
        reaction_pred = model.reaction(evaluation_tensor).cpu().numpy()

    diffusion_true = case.diffusion(evaluation_grid)
    reaction_true = case.reaction(evaluation_grid)
    metrics = {
        "relative_error_D": relative_l2_error(diffusion_true, diffusion_pred),
        "relative_error_R": relative_l2_error(reaction_true, reaction_pred),
        "final_loss": history[-1]["loss"],
        "final_weak_loss": history[-1]["weak_loss"],
        "final_rollout_loss": history[-1]["rollout_loss"],
        "final_mass_loss": history[-1]["mass_loss"],
        "final_strong_loss": history[-1]["strong_loss"],
        "final_reaction_anchor_loss": history[-1]["reaction_anchor_loss"],
        "final_reg_loss": history[-1]["reg_loss"],
        "state_bin_coverage": state_bin_coverage(observed["u"], value_range=case.value_range),
    }

    learned_tabulated = tabulated_closure_from_model(
        model=model,
        value_range=case.value_range,
        num_points=512,
        device=device,
        name=f"{case.name}_2d_learned_seed_{seed}",
    )
    learned_case = learned_tabulated.to_case(description="2D learned closure exported from neural surrogate")

    unseen_rollout = None
    if unseen_rollout_config.enabled:
        rollout_config = unseen_rollout_config.simulation_config or simulation_config
        rollout_amplitude_range = unseen_rollout_config.amplitude_range or amplitude_range
        rollout_initial_clip_range = unseen_rollout_config.initial_clip_range or rollout_amplitude_range
        unseen_rollout = compare_cases_on_shared_initial_conditions_2d(
            true_case=case,
            predicted_case=learned_case,
            config=rollout_config,
            num_trajectories=unseen_rollout_config.num_trajectories,
            seed=seed + unseen_rollout_config.seed_offset,
            amplitude_range=rollout_amplitude_range,
            initial_clip_range=rollout_initial_clip_range,
            num_modes=unseen_rollout_config.num_modes,
        )
        metrics["unseen_rollout_mse"] = unseen_rollout.mse
        metrics["unseen_rollout_relative_l2"] = unseen_rollout.relative_l2

    return {
        "case_name": case.name,
        "seed": seed,
        "simulation_config": asdict(simulation_config),
        "observation_config": asdict(observation_config),
        "training_config": asdict(training_config),
        "amplitude_range": amplitude_range,
        "initial_clip_range": initial_clip_range or amplitude_range,
        "dataset": observed,
        "history": history,
        "evaluation_grid": evaluation_grid,
        "diffusion_true": diffusion_true,
        "diffusion_pred": diffusion_pred,
        "reaction_true": reaction_true,
        "reaction_pred": reaction_pred,
        "metrics": metrics,
        "tabulated_closure": learned_tabulated,
        "unseen_rollout": unseen_rollout,
        "model": model,
    }
