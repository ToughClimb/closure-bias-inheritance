from __future__ import annotations

import math

import torch
from torch import nn


class ResidualMLP(nn.Module):
    """A small residual network used on top of explicit polynomial features."""

    def __init__(self, hidden_width: int = 64, hidden_depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = 1
        for _ in range(hidden_depth):
            layers.append(nn.Linear(input_dim, hidden_width))
            layers.append(nn.Tanh())
            input_dim = hidden_width
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.network(u.unsqueeze(-1)).squeeze(-1)


class PolynomialResidualScalarNet(nn.Module):
    """
    Scalar surrogate with an explicit polynomial backbone and a small residual MLP.

    The polynomial part makes low-order constitutive laws easy to recover, while the
    residual network keeps the model flexible for mildly non-polynomial cases.
    """

    def __init__(
        self,
        degree: int,
        input_range: tuple[float, float],
        hidden_width: int = 64,
        hidden_depth: int = 2,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.degree = degree
        self.register_buffer("input_lower", torch.tensor(float(input_range[0]), dtype=torch.float32))
        self.register_buffer("input_upper", torch.tensor(float(input_range[1]), dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))
        self.residual_scale = residual_scale
        self.residual_net = ResidualMLP(hidden_width=hidden_width, hidden_depth=hidden_depth)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

        final_linear = self.residual_net.network[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def initialize_constant(self, value: float) -> None:
        with torch.no_grad():
            self.coefficients.zero_()
            self.coefficients[0] = float(value)

    def _normalize(self, u: torch.Tensor) -> torch.Tensor:
        span = torch.clamp(self.input_upper - self.input_lower, min=1.0e-6)
        return 2.0 * (u - self.input_lower) / span - 1.0

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        scaled_u = self._normalize(u)
        basis = [torch.ones_like(scaled_u)]
        for power in range(1, self.degree + 1):
            basis.append(scaled_u**power)
        stacked_basis = torch.stack(basis, dim=-1)
        polynomial = torch.einsum("...k,k->...", stacked_basis, self.coefficients)
        residual = self.residual_scale * self.residual_net(scaled_u)
        return polynomial + residual


class KANScalarNet(nn.Module):
    """
    Lightweight scalar KAN branch built from a spline expansion.

    For one-dimensional closure discovery, a KAN-style representation reduces to a
    learned univariate spline over the state variable. This keeps the interface
    compatible with the current closure pipeline while allowing a more local basis
    than the polynomial-residual surrogate.
    """

    def __init__(
        self,
        input_range: tuple[float, float],
        grid_size: int = 16,
    ) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError("KAN grid_size must be at least 2.")

        self.grid_size = grid_size
        self.register_buffer("input_lower", torch.tensor(float(input_range[0]), dtype=torch.float32))
        self.register_buffer("input_upper", torch.tensor(float(input_range[1]), dtype=torch.float32))
        self.register_buffer("centers", torch.linspace(0.0, float(grid_size - 1), grid_size, dtype=torch.float32))
        self.spline_coefficients = nn.Parameter(torch.zeros(grid_size, dtype=torch.float32))
        self.base_weight = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.base_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def initialize_constant(self, value: float) -> None:
        with torch.no_grad():
            self.spline_coefficients.fill_(float(value))
            self.base_weight.zero_()
            self.base_bias.zero_()

    def _normalize(self, u: torch.Tensor) -> torch.Tensor:
        span = torch.clamp(self.input_upper - self.input_lower, min=1.0e-6)
        scaled = (u - self.input_lower) / span
        scaled = scaled.clamp(0.0, 1.0)
        return scaled * float(self.grid_size - 1)

    def _hat_basis(self, scaled_u: torch.Tensor) -> torch.Tensor:
        distances = torch.abs(scaled_u.unsqueeze(-1) - self.centers)
        return torch.clamp(1.0 - distances, min=0.0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        scaled_u = self._normalize(u)
        basis = self._hat_basis(scaled_u)
        spline = torch.einsum("...k,k->...", basis, self.spline_coefficients)
        centered = 2.0 * scaled_u / float(self.grid_size - 1) - 1.0
        base = centered * self.base_weight + self.base_bias
        return spline + base.squeeze(-1)


def _flux_divergence_periodic_torch(u: torch.Tensor, diffusion: torch.Tensor, dx: float) -> torch.Tensor:
    right_u = torch.roll(u, shifts=-1, dims=-1)
    right_d = torch.roll(diffusion, shifts=-1, dims=-1)
    d_face = 0.5 * (diffusion + right_d)
    flux_right = d_face * (right_u - u) / dx
    flux_left = torch.roll(flux_right, shifts=1, dims=-1)
    return (flux_right - flux_left) / dx


def _flux_divergence_neumann_torch(u: torch.Tensor, diffusion: torch.Tensor, dx: float) -> torch.Tensor:
    flux = torch.zeros(*u.shape[:-1], u.shape[-1] + 1, device=u.device, dtype=u.dtype)
    left_u = u[..., :-1]
    right_u = u[..., 1:]
    left_d = diffusion[..., :-1]
    right_d = diffusion[..., 1:]
    d_face = 0.5 * (left_d + right_d)
    flux[..., 1:-1] = d_face * (right_u - left_u) / dx
    return (flux[..., 1:] - flux[..., :-1]) / dx


class ReactionDiffusionClosure(nn.Module):
    """Interpretable scalar surrogates for diffusion and reaction closures."""

    def __init__(
        self,
        hidden_width: int = 64,
        hidden_depth: int = 2,
        min_diffusion: float = 1.0e-4,
        input_range: tuple[float, float] = (0.0, 1.0),
        backbone: str = "mlp",
        diffusion_degree: int = 2,
        reaction_degree: int = 3,
        residual_scale: float = 0.1,
        kan_grid_size: int = 16,
        initial_diffusion_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if backbone == "mlp":
            self.diffusion_net = PolynomialResidualScalarNet(
                degree=diffusion_degree,
                input_range=input_range,
                hidden_width=hidden_width,
                hidden_depth=hidden_depth,
                residual_scale=residual_scale,
            )
            self.reaction_net = PolynomialResidualScalarNet(
                degree=reaction_degree,
                input_range=input_range,
                hidden_width=hidden_width,
                hidden_depth=hidden_depth,
                residual_scale=residual_scale,
            )
        elif backbone == "kan":
            self.diffusion_net = KANScalarNet(
                input_range=input_range,
                grid_size=kan_grid_size,
            )
            self.reaction_net = KANScalarNet(
                input_range=input_range,
                grid_size=kan_grid_size,
            )
        else:
            raise ValueError(f"Unsupported closure backbone: {backbone}")

        self.min_diffusion = min_diffusion
        self.log_diffusion_scale = nn.Parameter(torch.tensor(math.log(float(initial_diffusion_scale))))
        self._reset_output_bias()

    def _reset_output_bias(self) -> None:
        initial_ratio = 0.1
        initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
        self.diffusion_net.initialize_constant(initial_logit)
        self.reaction_net.initialize_constant(0.0)

    def diffusion(self, u: torch.Tensor) -> torch.Tensor:
        diffusion_scale = torch.exp(self.log_diffusion_scale)
        return self.min_diffusion + diffusion_scale * torch.sigmoid(self.diffusion_net(u))

    def reaction(self, u: torch.Tensor) -> torch.Tensor:
        return self.reaction_net(u)

    def rhs(self, u: torch.Tensor, dx: float, boundary: str = "periodic") -> torch.Tensor:
        diffusion = self.diffusion(u)
        if boundary == "periodic":
            diffusion_term = _flux_divergence_periodic_torch(u, diffusion, dx)
        elif boundary == "neumann":
            diffusion_term = _flux_divergence_neumann_torch(u, diffusion, dx)
        else:
            raise ValueError(f"Unsupported boundary condition: {boundary}")
        return diffusion_term + self.reaction(u)

    def smoothness_penalty(self, u_support: torch.Tensor) -> torch.Tensor:
        if not u_support.requires_grad:
            u_support = u_support.clone().detach().requires_grad_(True)

        diffusion = self.diffusion(u_support)
        reaction = self.reaction(u_support)

        ones_d = torch.ones_like(diffusion)
        ones_r = torch.ones_like(reaction)
        d_du = torch.autograd.grad(diffusion, u_support, grad_outputs=ones_d, create_graph=True)[0]
        r_du = torch.autograd.grad(reaction, u_support, grad_outputs=ones_r, create_graph=True)[0]
        return d_du.pow(2).mean() + r_du.pow(2).mean()
