from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from closure_discovery.data_generation.cases import ReactionDiffusionCase


Array = np.ndarray


@dataclass(frozen=True)
class TabulatedClosure:
    u_grid: Array
    diffusion_values: Array
    reaction_values: Array
    name: str = "tabulated_closure"
    value_range: tuple[float, float] = (0.0, 1.0)

    def diffusion(self, u: Array) -> Array:
        return np.interp(
            np.asarray(u),
            self.u_grid,
            self.diffusion_values,
            left=float(self.diffusion_values[0]),
            right=float(self.diffusion_values[-1]),
        )

    def reaction(self, u: Array) -> Array:
        return np.interp(
            np.asarray(u),
            self.u_grid,
            self.reaction_values,
            left=float(self.reaction_values[0]),
            right=float(self.reaction_values[-1]),
        )

    def to_case(self, description: str | None = None) -> ReactionDiffusionCase:
        return ReactionDiffusionCase(
            name=self.name,
            description=description or "Tabulated numerical closure",
            diffusion=self.diffusion,
            reaction=self.reaction,
            value_range=self.value_range,
        )


def tabulated_closure_from_model(
    model,
    value_range: tuple[float, float],
    num_points: int = 512,
    device: str | torch.device | None = None,
    name: str = "learned_closure",
) -> TabulatedClosure:
    lower, upper = value_range
    u_grid = np.linspace(lower, upper, num_points, dtype=np.float32)

    if device is None:
        try:
            first_param = next(model.parameters())
            device = first_param.device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    u_tensor = torch.tensor(u_grid, dtype=torch.float32, device=device)
    with torch.no_grad():
        diffusion_values = model.diffusion(u_tensor).cpu().numpy()
        reaction_values = model.reaction(u_tensor).cpu().numpy()

    return TabulatedClosure(
        u_grid=u_grid,
        diffusion_values=diffusion_values,
        reaction_values=reaction_values,
        name=name,
        value_range=value_range,
    )

