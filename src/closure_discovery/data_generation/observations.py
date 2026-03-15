from __future__ import annotations

from typing import Any

import numpy as np


Array = np.ndarray


def subsample_dataset(
    dataset: dict[str, Array],
    space_stride: int = 1,
    time_stride: int = 1,
) -> dict[str, Array]:
    if space_stride < 1 or time_stride < 1:
        raise ValueError("space_stride and time_stride must be positive integers")

    u = dataset["u"][:, ::time_stride, ::space_stride]
    return {
        "x": dataset["x"][::space_stride].copy(),
        "t": dataset["t"][::time_stride].copy(),
        "u0": u[:, 0, :].copy(),
        "u": u.copy(),
    }


def add_gaussian_noise(
    dataset: dict[str, Array],
    noise_level: float,
    seed: int = 0,
    clip_range: tuple[float, float] | None = None,
) -> dict[str, Array]:
    if noise_level < 0.0:
        raise ValueError("noise_level must be non-negative")

    if noise_level == 0.0:
        return {
            "x": dataset["x"].copy(),
            "t": dataset["t"].copy(),
            "u0": dataset["u0"].copy(),
            "u": dataset["u"].copy(),
        }

    rng = np.random.default_rng(seed)
    u = dataset["u"]
    scale = noise_level * np.std(u)
    noisy_u = u + rng.normal(loc=0.0, scale=scale, size=u.shape)
    if clip_range is not None:
        noisy_u = np.clip(noisy_u, clip_range[0], clip_range[1])

    return {
        "x": dataset["x"].copy(),
        "t": dataset["t"].copy(),
        "u0": noisy_u[:, 0, :].copy(),
        "u": noisy_u.copy(),
    }


def apply_observation_model(
    dataset: dict[str, Array],
    space_stride: int = 1,
    time_stride: int = 1,
    noise_level: float = 0.0,
    seed: int = 0,
    clip_range: tuple[float, float] | None = None,
) -> dict[str, Array]:
    observed = subsample_dataset(dataset, space_stride=space_stride, time_stride=time_stride)
    observed = add_gaussian_noise(
        observed,
        noise_level=noise_level,
        seed=seed,
        clip_range=clip_range,
    )
    return observed
