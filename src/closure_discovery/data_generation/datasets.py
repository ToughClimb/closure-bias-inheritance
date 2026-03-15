from __future__ import annotations

import numpy as np


Array = np.ndarray


def select_trajectories(dataset: dict[str, Array], indices: Array | list[int]) -> dict[str, Array]:
    index_array = np.asarray(indices, dtype=int)
    return {
        "x": dataset["x"].copy(),
        "t": dataset["t"].copy(),
        "u0": dataset["u0"][index_array].copy(),
        "u": dataset["u"][index_array].copy(),
    }

