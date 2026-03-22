from __future__ import annotations

import torch
from torch.nn import functional as F


def central_time_derivative(u: torch.Tensor, dt: float) -> torch.Tensor:
    return (u[:, 2:, :] - u[:, :-2, :]) / (2.0 * dt)


def spatial_gradient(u: torch.Tensor, dx: float, boundary: str = "periodic") -> torch.Tensor:
    if boundary == "periodic":
        return (torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1)) / (2.0 * dx)
    if boundary == "neumann":
        grad = torch.zeros_like(u)
        grad[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * dx)
        grad[..., 0] = 0.0
        grad[..., -1] = 0.0
        return grad
    raise ValueError(f"Unsupported boundary condition: {boundary}")


def weak_residual_1d(
    u: torch.Tensor,
    dt: float,
    dx: float,
    phi: torch.Tensor,
    grad_phi: torch.Tensor,
    closure_model,
    boundary: str = "periodic",
) -> torch.Tensor:
    """
    Compute spatial weak-form residuals.

    Parameters
    ----------
    u:
        Tensor with shape [batch, time, space].
    phi, grad_phi:
        Tensors with shape [num_test_functions, space].
    """

    u_mid = u[:, 1:-1, :]
    ut = central_time_derivative(u, dt)
    ux = spatial_gradient(u_mid, dx, boundary=boundary)

    diffusion = closure_model.diffusion(u_mid)
    reaction = closure_model.reaction(u_mid)

    term_ut = torch.einsum("kx,btx->btk", phi, ut) * dx
    term_diff = torch.einsum("kx,btx->btk", grad_phi, diffusion * ux) * dx
    term_react = torch.einsum("kx,btx->btk", phi, reaction) * dx
    return term_ut + term_diff - term_react


def weak_loss_1d(
    u: torch.Tensor,
    dt: float,
    dx: float,
    phi: torch.Tensor,
    grad_phi: torch.Tensor,
    closure_model,
    boundary: str = "periodic",
) -> torch.Tensor:
    residual = weak_residual_1d(
        u=u,
        dt=dt,
        dx=dx,
        phi=phi,
        grad_phi=grad_phi,
        closure_model=closure_model,
        boundary=boundary,
    )
    return residual.pow(2).mean()


def one_step_rollout_loss(
    u: torch.Tensor,
    dt: float,
    dx: float,
    closure_model,
    boundary: str = "periodic",
) -> torch.Tensor:
    current = u[:, :-1, :]
    target = u[:, 1:, :]
    prediction = current + dt * closure_model.rhs(current, dx=dx, boundary=boundary)
    return F.mse_loss(prediction, target)


def mass_balance_loss(
    u: torch.Tensor,
    dt: float,
    closure_model,
) -> torch.Tensor:
    """
    Reweight the spatially averaged balance:

        d/dt ∫ u dx = ∫ R(u) dx

    for periodic or no-flux boundaries, where the diffusion contribution vanishes.
    This is the constant-test-function relation already contained in the weak form,
    isolated here as an extra scalar penalty so it can receive its own optimization weight.
    """

    u_mid = u[:, 1:-1, :]
    ut = central_time_derivative(u, dt)
    reaction = closure_model.reaction(u_mid)
    mean_ut = ut.mean(dim=-1)
    mean_reaction = reaction.mean(dim=-1)
    return F.mse_loss(mean_ut, mean_reaction)


def strong_residual_loss(
    u: torch.Tensor,
    dt: float,
    dx: float,
    closure_model,
    boundary: str = "periodic",
) -> torch.Tensor:
    """
    Auxiliary pointwise consistency term on the observed grid.

    This is useful in the clean synthetic regime, where the weak loss alone may admit
    degenerate closure pairs with similar trajectory-level behavior.
    """

    u_mid = u[:, 1:-1, :]
    ut = central_time_derivative(u, dt)
    rhs = closure_model.rhs(u_mid, dx=dx, boundary=boundary)
    return F.mse_loss(ut, rhs)
