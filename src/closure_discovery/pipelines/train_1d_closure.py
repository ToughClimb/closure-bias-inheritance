from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

import numpy as np
import torch

from closure_discovery.data_generation.observations import apply_observation_model
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D, generate_dataset
from closure_discovery.evaluation.metrics import relative_l2_error, summarize_excitation
from closure_discovery.evaluation.rollout import compare_cases_on_shared_initial_conditions
from closure_discovery.models.mlp_closure import ReactionDiffusionClosure
from closure_discovery.models.tabulated_closure import tabulated_closure_from_model
from closure_discovery.weak_form.test_functions import make_test_functions_1d
from closure_discovery.weak_form.weak_residual import (
    mass_balance_loss,
    one_step_rollout_loss,
    strong_residual_loss,
    weak_loss_1d,
)


@dataclass(frozen=True)
class ObservationConfig:
    noise_level: float = 0.0
    space_stride: int = 1
    time_stride: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 150
    lr: float = 1.0e-3
    rollout_weight: float = 0.1
    mass_weight: float = 1.0
    strong_weight: float = 1.0
    reaction_anchor_weight: float = 1.0
    reg_weight: float = 1.0e-4
    backbone: str = "mlp"
    hidden_width: int = 64
    hidden_depth: int = 2
    diffusion_degree: int = 2
    reaction_degree: int = 3
    residual_scale: float = 0.1
    kan_grid_size: int = 16
    num_test_modes: int = 4
    num_bump_functions: int = 4


@dataclass(frozen=True)
class UnseenRolloutConfig:
    enabled: bool = True
    num_trajectories: int = 4
    seed_offset: int = 1000
    amplitude_range: tuple[float, float] | None = None
    num_modes: int = 4
    simulation_config: SimulationConfig1D | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_closure_identification(
    *,
    case,
    simulation_config: SimulationConfig1D,
    num_trajectories: int,
    amplitude_range: tuple[float, float] = (0.2, 0.8),
    num_initial_modes: int = 4,
    observation_config: ObservationConfig | None = None,
    training_config: TrainingConfig | None = None,
    unseen_rollout_config: UnseenRolloutConfig | None = None,
    seed: int = 0,
    device: str | torch.device | None = None,
    raw_dataset: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    observation_config = observation_config or ObservationConfig()
    training_config = training_config or TrainingConfig()
    unseen_rollout_config = unseen_rollout_config or UnseenRolloutConfig()
    set_seed(seed)

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
    dt = float(observed["t"][1] - observed["t"][0])
    phi_np, grad_phi_np = make_test_functions_1d(
        observed["x"],
        num_modes=training_config.num_test_modes,
        boundary=simulation_config.boundary,
        num_bumps=training_config.num_bump_functions,
    )
    excitation = summarize_excitation(observed["u"], dx=dx, value_range=case.value_range)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    u = torch.tensor(observed["u"], dtype=torch.float32, device=device)
    phi = torch.tensor(phi_np, dtype=torch.float32, device=device)
    grad_phi = torch.tensor(grad_phi_np, dtype=torch.float32, device=device)

    model = ReactionDiffusionClosure(
        hidden_width=training_config.hidden_width,
        hidden_depth=training_config.hidden_depth,
        input_range=case.value_range,
        backbone=training_config.backbone,
        diffusion_degree=training_config.diffusion_degree,
        reaction_degree=training_config.reaction_degree,
        residual_scale=training_config.residual_scale,
        kan_grid_size=training_config.kan_grid_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)

    u_min = float(case.value_range[0])
    u_max = float(case.value_range[1])
    support = torch.linspace(u_min, u_max, 256, device=device, dtype=torch.float32).requires_grad_(True)
    lower_anchor = torch.tensor([u_min], device=device, dtype=torch.float32)

    history: list[dict[str, float]] = []
    for epoch in range(1, training_config.epochs + 1):
        optimizer.zero_grad()
        weak = weak_loss_1d(
            u=u,
            dt=dt,
            dx=dx,
            phi=phi,
            grad_phi=grad_phi,
            closure_model=model,
            boundary=simulation_config.boundary,
        )
        rollout = one_step_rollout_loss(
            u=u,
            dt=dt,
            dx=dx,
            closure_model=model,
            boundary=simulation_config.boundary,
        )
        mass = mass_balance_loss(
            u=u,
            dt=dt,
            closure_model=model,
        )
        strong = strong_residual_loss(
            u=u,
            dt=dt,
            dx=dx,
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
    }
    learned_tabulated = tabulated_closure_from_model(
        model=model,
        value_range=case.value_range,
        num_points=512,
        device=device,
        name=f"{case.name}_learned_seed_{seed}",
    )
    learned_case = learned_tabulated.to_case(description="Learned closure exported from neural surrogate")

    unseen_rollout = None
    if unseen_rollout_config.enabled:
        rollout_config = unseen_rollout_config.simulation_config or simulation_config
        rollout_amplitude_range = unseen_rollout_config.amplitude_range or amplitude_range
        unseen_rollout = compare_cases_on_shared_initial_conditions(
            true_case=case,
            predicted_case=learned_case,
            config=rollout_config,
            num_trajectories=unseen_rollout_config.num_trajectories,
            seed=seed + unseen_rollout_config.seed_offset,
            amplitude_range=rollout_amplitude_range,
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
        "dataset": observed,
        "excitation": excitation,
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
