"""Geometry checks shared by the uniform-grid observation and weak-form code."""
from __future__ import annotations

import operator
import numpy as np


def positive_stride(value: int, name: str) -> int:
    try:
        stride = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, (bool, np.bool_)) or stride < 1:
        raise ValueError(f"{name} must be a positive integer")
    return stride


def uniform_spacing(values, name: str, *, minimum: int = 2) -> float:
    coordinates = np.asarray(values)
    if coordinates.ndim != 1 or len(coordinates) < minimum:
        raise ValueError(f"{name} requires at least {minimum} coordinates in one dimension")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} coordinates must be finite")
    differences = np.diff(coordinates.astype(np.float64))
    if np.any(differences <= 0):
        raise ValueError(f"{name} coordinates must be strictly increasing")
    epsilon = np.finfo(coordinates.dtype).eps if np.issubdtype(coordinates.dtype, np.floating) else np.finfo(float).eps
    tolerance = 8 * epsilon * max(float(np.max(np.abs(coordinates))), float(differences[0]))
    if not np.allclose(differences, differences[0], rtol=1e-10, atol=tolerance):
        raise ValueError(f"{name} requires uniformly spaced coordinates")
    return float(differences[0])


def validate_subsampling(dataset, space_stride: int, time_stride: int) -> tuple[int, int]:
    space_stride = positive_stride(space_stride, "space_stride")
    time_stride = positive_stride(time_stride, "time_stride")
    uniform_spacing(dataset["t"], "time")
    for axis in ("x", "y"):
        if axis not in dataset:
            continue
        uniform_spacing(dataset[axis], axis)
        if len(dataset[axis]) % space_stride:
            raise ValueError(f"space_stride={space_stride} must divide the periodic {axis} grid size {len(dataset[axis])}")
        if len(dataset[axis]) // space_stride < 2:
            raise ValueError(f"Subsampling must leave at least two points on the {axis} grid")
    if len(dataset["t"][::time_stride]) < 3:
        raise ValueError("Subsampling must leave at least three saved times for time derivatives")
    return space_stride, time_stride
