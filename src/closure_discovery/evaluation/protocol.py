"""Identify training protocols and frozen targets without using truth errors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def configuration_signature(configuration) -> str:
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def package_signature() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def training_metadata(result, *, num_initial_modes: int) -> dict:
    configuration = dict(simulation=result["simulation_config"], observation=result["observation_config"],
                         training=result["training_config"], amplitude_range=result["amplitude_range"],
                         initial_clip_range=result["initial_clip_range"], num_initial_modes=num_initial_modes,
                         num_trajectories=int(result["dataset"]["u"].shape[0]),
                         numpy=np.__version__, torch=torch.__version__, implementation=package_signature(),
                         device=str(next(result["model"].parameters()).device))
    digest = hashlib.sha256()
    for key in ("evaluation_grid", "diffusion_pred", "reaction_pred"):
        values = np.ascontiguousarray(result[key], dtype=np.float64)
        digest.update(key.encode()); digest.update(str(values.shape).encode()); digest.update(values.tobytes())
    return dict(protocol_id=configuration_signature(configuration), surrogate_signature=digest.hexdigest(),
                device=configuration["device"], epochs=result["training_config"]["epochs"])
