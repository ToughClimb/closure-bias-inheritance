from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cases import ReactionDiffusionCase


Array = np.ndarray


@dataclass(frozen=True)
class SimulationConfig2D:
    nx: int = 32
    ny: int = 32
    length_x: float = 1.0
    length_y: float = 1.0
    dt: float = 1.0e-4
    t_final: float = 0.03
    save_every: int = 5
    boundary: str = "periodic"

    @property
    def dx(self) -> float:
        return self.length_x / self.nx

    @property
    def dy(self) -> float:
        return self.length_y / self.ny

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


def make_grid(config: SimulationConfig2D) -> tuple[Array, Array]:
    xs = np.linspace(0.0, config.length_x, config.nx, endpoint=False)
    ys = np.linspace(0.0, config.length_y, config.ny, endpoint=False)
    return xs, ys


def random_fourier_initial_condition_2d(
    xs: Array,
    ys: Array,
    rng: np.random.Generator,
    amplitude_range: tuple[float, float] = (0.2, 0.8),
    num_modes: int = 3,
    clip_range: tuple[float, float] | None = None,
) -> Array:
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    length_x = xs[-1] - xs[0] + dx
    length_y = ys[-1] - ys[0] + dy
    field = np.zeros_like(xx)
    for mode_x in range(1, num_modes + 1):
        for mode_y in range(1, num_modes + 1):
            scale = 0.18 / (mode_x + mode_y - 1)
            angle_x = 2.0 * np.pi * mode_x * xx / length_x
            angle_y = 2.0 * np.pi * mode_y * yy / length_y
            field += rng.normal(scale=scale) * np.sin(angle_x) * np.sin(angle_y)
            field += rng.normal(scale=scale) * np.sin(angle_x) * np.cos(angle_y)
            field += rng.normal(scale=scale) * np.cos(angle_x) * np.sin(angle_y)
            field += rng.normal(scale=scale) * np.cos(angle_x) * np.cos(angle_y)

    field -= field.min()
    max_value = field.max()
    if max_value > 0.0:
        field /= max_value

    lower, upper = amplitude_range
    if upper < lower:
        raise ValueError(f"Expected amplitude_range[0] <= amplitude_range[1], got {amplitude_range}.")
    field = lower + (upper - lower) * field
    clip_lower, clip_upper = clip_range or amplitude_range
    return np.clip(field, clip_lower, clip_upper)


def _flux_divergence_periodic(u: Array, diffusion: callable, dx: float, dy: float) -> Array:
    right_x = np.roll(u, -1, axis=0)
    right_y = np.roll(u, -1, axis=1)

    d_face_x = 0.5 * (diffusion(u) + diffusion(right_x))
    d_face_y = 0.5 * (diffusion(u) + diffusion(right_y))

    flux_x_right = d_face_x * (right_x - u) / dx
    flux_y_right = d_face_y * (right_y - u) / dy

    flux_x_left = np.roll(flux_x_right, 1, axis=0)
    flux_y_left = np.roll(flux_y_right, 1, axis=1)
    return (flux_x_right - flux_x_left) / dx + (flux_y_right - flux_y_left) / dy


def rhs(
    u: Array,
    case: ReactionDiffusionCase,
    dx: float,
    dy: float,
    boundary: str = "periodic",
) -> Array:
    if boundary != "periodic":
        raise ValueError(f"Unsupported boundary condition for the 2D solver: {boundary}")
    diffusion_term = _flux_divergence_periodic(u, case.diffusion, dx=dx, dy=dy)
    return diffusion_term + case.reaction(u)


def rk4_step(u: Array, dt: float, rhs_fn) -> Array:
    k1 = rhs_fn(u)
    k2 = rhs_fn(u + 0.5 * dt * k1)
    k3 = rhs_fn(u + 0.5 * dt * k2)
    k4 = rhs_fn(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_trajectory(
    case: ReactionDiffusionCase,
    config: SimulationConfig2D,
    u0: Array,
) -> tuple[Array, Array]:
    save_stride = config.save_stride
    num_saves = config.num_steps // save_stride + 1
    trajectory = np.zeros((num_saves, config.nx, config.ny), dtype=np.float64)
    times = np.zeros(num_saves, dtype=np.float64)

    state = u0.astype(np.float64).copy()
    trajectory[0] = state

    rhs_fn = lambda current: rhs(current, case=case, dx=config.dx, dy=config.dy, boundary=config.boundary)
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
    config: SimulationConfig2D,
    num_trajectories: int,
    seed: int = 0,
    amplitude_range: tuple[float, float] = (0.2, 0.8),
    num_modes: int = 3,
    initial_clip_range: tuple[float, float] | None = None,
) -> dict[str, Array]:
    rng = np.random.default_rng(seed)
    xs, ys = make_grid(config)

    all_u0 = np.zeros((num_trajectories, config.nx, config.ny), dtype=np.float64)
    all_trajectories = []
    times = None

    for index in range(num_trajectories):
        u0 = random_fourier_initial_condition_2d(
            xs,
            ys,
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
        "y": ys,
        "t": times,
        "u0": all_u0,
        "u": np.stack(all_trajectories, axis=0),
    }
