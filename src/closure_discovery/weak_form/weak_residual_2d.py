from __future__ import annotations

import torch
from torch.nn import functional as F


def central_time_derivative_2d(u: torch.Tensor, dt: float) -> torch.Tensor:
    return (u[:, 2:, :, :] - u[:, :-2, :, :]) / (2.0 * dt)


def spatial_gradients_2d(u: torch.Tensor, dx: float, dy: float, boundary: str = "periodic") -> tuple[torch.Tensor, torch.Tensor]:
    if boundary != "periodic":
        raise ValueError(f"Unsupported boundary condition for 2D weak residuals: {boundary}")
    grad_x = (torch.roll(u, shifts=-1, dims=-2) - torch.roll(u, shifts=1, dims=-2)) / (2.0 * dx)
    grad_y = (torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1)) / (2.0 * dy)
    return grad_x, grad_y


def flux_divergence_periodic_2d(u: torch.Tensor, diffusion: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    right_x_u = torch.roll(u, shifts=-1, dims=-2)
    right_y_u = torch.roll(u, shifts=-1, dims=-1)
    right_x_d = torch.roll(diffusion, shifts=-1, dims=-2)
    right_y_d = torch.roll(diffusion, shifts=-1, dims=-1)

    d_face_x = 0.5 * (diffusion + right_x_d)
    d_face_y = 0.5 * (diffusion + right_y_d)

    flux_x_right = d_face_x * (right_x_u - u) / dx
    flux_y_right = d_face_y * (right_y_u - u) / dy

    flux_x_left = torch.roll(flux_x_right, shifts=1, dims=-2)
    flux_y_left = torch.roll(flux_y_right, shifts=1, dims=-1)
    return (flux_x_right - flux_x_left) / dx + (flux_y_right - flux_y_left) / dy


def rhs_2d(u: torch.Tensor, closure_model, dx: float, dy: float, boundary: str = "periodic") -> torch.Tensor:
    if boundary != "periodic":
        raise ValueError(f"Unsupported boundary condition for 2D rhs: {boundary}")
    diffusion = closure_model.diffusion(u)
    diffusion_term = flux_divergence_periodic_2d(u, diffusion, dx=dx, dy=dy)
    return diffusion_term + closure_model.reaction(u)


def weak_residual_2d(
    u: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    phi: torch.Tensor,
    grad_phi_x: torch.Tensor,
    grad_phi_y: torch.Tensor,
    closure_model,
    boundary: str = "periodic",
) -> torch.Tensor:
    u_mid = u[:, 1:-1, :, :]
    ut = central_time_derivative_2d(u, dt)
    ux, uy = spatial_gradients_2d(u_mid, dx=dx, dy=dy, boundary=boundary)

    diffusion = closure_model.diffusion(u_mid)
    reaction = closure_model.reaction(u_mid)

    cell_area = dx * dy
    term_ut = torch.einsum("kxy,btxy->btk", phi, ut) * cell_area
    term_diff_x = torch.einsum("kxy,btxy->btk", grad_phi_x, diffusion * ux) * cell_area
    term_diff_y = torch.einsum("kxy,btxy->btk", grad_phi_y, diffusion * uy) * cell_area
    term_react = torch.einsum("kxy,btxy->btk", phi, reaction) * cell_area
    return term_ut + term_diff_x + term_diff_y - term_react


def weak_loss_2d(
    u: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    phi: torch.Tensor,
    grad_phi_x: torch.Tensor,
    grad_phi_y: torch.Tensor,
    closure_model,
    boundary: str = "periodic",
) -> torch.Tensor:
    residual = weak_residual_2d(
        u=u,
        dt=dt,
        dx=dx,
        dy=dy,
        phi=phi,
        grad_phi_x=grad_phi_x,
        grad_phi_y=grad_phi_y,
        closure_model=closure_model,
        boundary=boundary,
    )
    return residual.pow(2).mean()


def one_step_rollout_loss_2d(
    u: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    closure_model,
    boundary: str = "periodic",
) -> torch.Tensor:
    current = u[:, :-1, :, :]
    target = u[:, 1:, :, :]
    prediction = current + dt * rhs_2d(current, closure_model=closure_model, dx=dx, dy=dy, boundary=boundary)
    return F.mse_loss(prediction, target)


def mass_balance_loss_2d(
    u: torch.Tensor,
    dt: float,
    closure_model,
) -> torch.Tensor:
    u_mid = u[:, 1:-1, :, :]
    ut = central_time_derivative_2d(u, dt)
    reaction = closure_model.reaction(u_mid)
    mean_ut = ut.mean(dim=(-2, -1))
    mean_reaction = reaction.mean(dim=(-2, -1))
    return F.mse_loss(mean_ut, mean_reaction)


def strong_residual_loss_2d(
    u: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    closure_model,
    boundary: str = "periodic",
) -> torch.Tensor:
    u_mid = u[:, 1:-1, :, :]
    ut = central_time_derivative_2d(u, dt)
    rhs = rhs_2d(u_mid, closure_model=closure_model, dx=dx, dy=dy, boundary=boundary)
    return F.mse_loss(ut, rhs)
