from __future__ import annotations

import numpy as np


Array = np.ndarray


def make_test_functions_2d(
    xs: Array,
    ys: Array,
    num_modes: int = 1,
    boundary: str = "periodic",
) -> tuple[Array, Array, Array]:
    if boundary != "periodic":
        raise ValueError(f"Unsupported boundary condition for 2D test functions: {boundary}")

    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    length_x = xs[-1] - xs[0] + dx
    length_y = ys[-1] - ys[0] + dy
    phi_list = [np.ones_like(xx)]
    grad_x_list = [np.zeros_like(xx)]
    grad_y_list = [np.zeros_like(xx)]

    for mode in range(1, num_modes + 1):
        freq_x = 2.0 * np.pi * mode / length_x
        freq_y = 2.0 * np.pi * mode / length_y
        sin_x = np.sin(freq_x * xx)
        cos_x = np.cos(freq_x * xx)
        sin_y = np.sin(freq_y * yy)
        cos_y = np.cos(freq_y * yy)

        phi_list.extend([sin_x, cos_x, sin_y, cos_y])
        grad_x_list.extend([freq_x * cos_x, -freq_x * sin_x, np.zeros_like(xx), np.zeros_like(xx)])
        grad_y_list.extend([np.zeros_like(xx), np.zeros_like(xx), freq_y * cos_y, -freq_y * sin_y])

        mixed_terms = [
            (sin_x * sin_y, freq_x * cos_x * sin_y, freq_y * sin_x * cos_y),
            (sin_x * cos_y, freq_x * cos_x * cos_y, -freq_y * sin_x * sin_y),
            (cos_x * sin_y, -freq_x * sin_x * sin_y, freq_y * cos_x * cos_y),
            (cos_x * cos_y, -freq_x * sin_x * cos_y, -freq_y * cos_x * sin_y),
        ]
        for phi, grad_x, grad_y in mixed_terms:
            phi_list.append(phi)
            grad_x_list.append(grad_x)
            grad_y_list.append(grad_y)

    phi = np.stack(phi_list, axis=0).astype(np.float64)
    grad_x = np.stack(grad_x_list, axis=0).astype(np.float64)
    grad_y = np.stack(grad_y_list, axis=0).astype(np.float64)
    return phi, grad_x, grad_y
