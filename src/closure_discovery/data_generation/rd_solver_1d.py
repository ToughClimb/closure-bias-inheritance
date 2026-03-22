from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cases import ReactionDiffusionCase


Array = np.ndarray


@dataclass(frozen=True)
class SimulationConfig1D:
    nx: int = 128
    length: float = 1.0
    dt: float = 1.0e-4
    t_final: float = 0.2
    save_every: int = 20
    boundary: str = "periodic"

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def num_steps(self) -> int:
        return int(round(self.t_final / self.dt))

    @property
    def save_stride(self) -> int:
        return max(self.save_every, 1)

    @property
    def saved_dt(self) -> float:
        return self.dt * self.save_stride

    @property
    def last_saved_step(self) -> int:
        return (self.num_steps // self.save_stride) * self.save_stride

    @property
    def last_saved_time(self) -> float:
        return self.last_saved_step * self.dt


def make_grid(config: SimulationConfig1D) -> Array:
    return np.linspace(0.0, config.length, config.nx, endpoint=False)


def random_fourier_initial_condition(
    xs: Array,
    rng: np.random.Generator,
    amplitude_range: tuple[float, float] = (0.2, 0.8),
    num_modes: int = 4,
    clip_range: tuple[float, float] | None = None,
) -> Array:
    """Construct a smooth, positive random field from low-frequency Fourier modes."""

    base = np.zeros_like(xs)
    length = xs[-1] - xs[0] + (xs[1] - xs[0])
    for mode in range(1, num_modes + 1):
        sin_weight = rng.normal(scale=0.25 / mode)
        cos_weight = rng.normal(scale=0.25 / mode)
        angle = 2.0 * np.pi * mode * xs / length
        base += sin_weight * np.sin(angle) + cos_weight * np.cos(angle)

    base -= base.min()
    if base.max() > 0.0:
        base /= base.max()

    lower, upper = amplitude_range
    if upper < lower:
        raise ValueError(f"Expected amplitude_range[0] <= amplitude_range[1], got {amplitude_range}.")
    field = lower + (upper - lower) * base
    clip_lower, clip_upper = clip_range or amplitude_range
    return np.clip(field, clip_lower, clip_upper)


def _flux_divergence_periodic(u: Array, diffusion: callable, dx: float) -> Array:
    right = np.roll(u, -1)
    d_face = 0.5 * (diffusion(u) + diffusion(right))
    flux_right = d_face * (right - u) / dx
    flux_left = np.roll(flux_right, 1)
    return (flux_right - flux_left) / dx


def _flux_divergence_neumann(u: Array, diffusion: callable, dx: float) -> Array:
    flux = np.zeros(u.shape[0] + 1, dtype=u.dtype)
    left = u[:-1]
    right = u[1:]
    d_face = 0.5 * (diffusion(left) + diffusion(right))
    flux[1:-1] = d_face * (right - left) / dx
    return (flux[1:] - flux[:-1]) / dx


def rhs(
    u: Array,
    case: ReactionDiffusionCase,
    dx: float,
    boundary: str = "periodic",
) -> Array:
    if boundary == "periodic":
        diffusion_term = _flux_divergence_periodic(u, case.diffusion, dx)
    elif boundary == "neumann":
        diffusion_term = _flux_divergence_neumann(u, case.diffusion, dx)
    else:
        raise ValueError(f"Unsupported boundary condition: {boundary}")
    return diffusion_term + case.reaction(u)


def rk4_step(
    u: Array,
    dt: float,
    rhs_fn,
) -> Array:
    k1 = rhs_fn(u)
    k2 = rhs_fn(u + 0.5 * dt * k1)
    k3 = rhs_fn(u + 0.5 * dt * k2)
    k4 = rhs_fn(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_trajectory(
    case: ReactionDiffusionCase,
    config: SimulationConfig1D,
    u0: Array,
) -> tuple[Array, Array]:
    """Advance a single trajectory and return saved times and states."""

    save_stride = config.save_stride
    num_saves = config.num_steps // save_stride + 1
    trajectory = np.zeros((num_saves, config.nx), dtype=np.float64)
    times = np.zeros(num_saves, dtype=np.float64)

    state = u0.astype(np.float64).copy()
    trajectory[0] = state

    rhs_fn = lambda current: rhs(current, case=case, dx=config.dx, boundary=config.boundary)
    save_idx = 1
    for step in range(1, config.num_steps + 1):
        state = rk4_step(state, config.dt, rhs_fn)
        state = np.clip(state, -0.25, 2.0)
        if step % save_stride == 0:
            trajectory[save_idx] = state
            times[save_idx] = step * config.dt
            save_idx += 1

    return times, trajectory


def generate_dataset(
    case: ReactionDiffusionCase,
    config: SimulationConfig1D,
    num_trajectories: int,
    seed: int = 0,
    amplitude_range: tuple[float, float] = (0.2, 0.8),
    num_modes: int = 4,
    initial_clip_range: tuple[float, float] | None = None,
) -> dict[str, Array]:
    rng = np.random.default_rng(seed)
    xs = make_grid(config)

    all_u0 = np.zeros((num_trajectories, config.nx), dtype=np.float64)
    all_trajectories = []
    times = None

    for index in range(num_trajectories):
        u0 = random_fourier_initial_condition(
            xs,
            rng=rng,
            amplitude_range=amplitude_range,
            num_modes=num_modes,
            clip_range=initial_clip_range,
        )
        current_times, trajectory = simulate_trajectory(case=case, config=config, u0=u0)
        all_u0[index] = u0
        all_trajectories.append(trajectory)
        if times is None:
            times = current_times

    return {
        "x": xs,
        "t": times,
        "u0": all_u0,
        "u": np.stack(all_trajectories, axis=0),
    }
