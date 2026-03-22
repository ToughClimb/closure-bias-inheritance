from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D, generate_dataset
from closure_discovery.evaluation.metrics import mean_squared_error


Array = np.ndarray


@dataclass(frozen=True)
class RolloutComparison:
    mse: float
    relative_l2: float
    true_dataset: dict[str, Array]
    predicted_dataset: dict[str, Array]


def relative_trajectory_l2(true_u: Array, predicted_u: Array) -> float:
    numerator = np.linalg.norm(true_u - predicted_u)
    denominator = np.linalg.norm(true_u) + 1.0e-12
    return float(numerator / denominator)


def compare_cases_on_shared_initial_conditions(
    true_case,
    predicted_case,
    config: SimulationConfig1D,
    num_trajectories: int,
    seed: int,
    amplitude_range: tuple[float, float] = (0.2, 0.8),
    initial_clip_range: tuple[float, float] | None = None,
    num_modes: int = 4,
) -> RolloutComparison:
    clip_range = initial_clip_range or amplitude_range
    true_dataset = generate_dataset(
        case=true_case,
        config=config,
        num_trajectories=num_trajectories,
        seed=seed,
        amplitude_range=amplitude_range,
        initial_clip_range=clip_range,
        num_modes=num_modes,
    )
    predicted_dataset = generate_dataset(
        case=predicted_case,
        config=config,
        num_trajectories=num_trajectories,
        seed=seed,
        amplitude_range=amplitude_range,
        initial_clip_range=clip_range,
        num_modes=num_modes,
    )

    return RolloutComparison(
        mse=mean_squared_error(true_dataset["u"], predicted_dataset["u"]),
        relative_l2=relative_trajectory_l2(true_dataset["u"], predicted_dataset["u"]),
        true_dataset=true_dataset,
        predicted_dataset=predicted_dataset,
    )
