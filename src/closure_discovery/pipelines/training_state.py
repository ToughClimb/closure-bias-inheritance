"""Finite-state checks and atomic, self-contained optimizer checkpoints."""
from __future__ import annotations

import os
import re
import json
from pathlib import Path
import tempfile
import warnings

import torch


def assert_finite_state(model, optimizer=None, *, gradients=False, epoch=None):
    label = f" at epoch {epoch}" if epoch is not None else ""
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Nonfinite parameter {name}{label}")
        if gradients and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Nonfinite gradient {name}{label}")
    if optimizer is not None:
        for index, state in enumerate(optimizer.state.values()):
            for name, value in state.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite optimizer state {index}/{name}{label}")


def validate_history(history, epoch):
    if [row.get("epoch") for row in history] != list(range(1, epoch + 1)):
        raise ValueError(f"Checkpoint history does not contain every epoch through {epoch}")
    for row in history:
        for name, value in row.items():
            if isinstance(value, (int, float)) and not torch.isfinite(torch.tensor(value, dtype=torch.float64)):
                raise FloatingPointError(f"Nonfinite history field {name} at epoch {row['epoch']}")


def atomic_torch_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def checkpoint_payload(model, optimizer, history, *, snapshots=None, **metadata):
    epoch = metadata["epoch"]
    validate_history(history, epoch)
    assert_finite_state(model, optimizer, epoch=epoch)
    device = next(model.parameters()).device
    return dict(metadata, checkpoint_schema=2, model=model.state_dict(), optimizer=optimizer.state_dict(),
                history=history, snapshots=snapshots or {}, rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_checkpoint(path, model, optimizer, expected, *, legacy_history=None, legacy_history_path=None):
    """Restore trusted local state; derived exports never override embedded history."""
    # Adam's non-capturable step counters stay on CPU; load_state_dict moves
    # parameter/moment tensors onto their appropriate devices.
    saved = torch.load(path, weights_only=False, map_location="cpu")
    declared = re.fullmatch(r"checkpoint_(\d+)", Path(path).stem)
    if declared is not None and saved.get("epoch") != int(declared.group(1)):
        raise ValueError("Checkpoint epoch does not match its filename")
    legacy = saved.get("checkpoint_schema", 0) < 2
    allow_legacy = os.environ.get("EAAI_ALLOW_LEGACY_CHECKPOINT") == "1"
    if legacy and not allow_legacy:
        raise ValueError("Legacy checkpoint requires explicit EAAI_ALLOW_LEGACY_CHECKPOINT=1; keep historical evidence in its original work root")
    for key, value in expected.items():
        if legacy and key == "source_signature" and key not in saved:
            continue
        if saved.get(key) != value:
            label = "observation signature" if key == "observation_signature" else "configuration"
            raise ValueError(f"Checkpoint {label} mismatch: {key}")
    if legacy and legacy_history is None and legacy_history_path is not None and Path(legacy_history_path).exists():
        legacy_history = json.loads(Path(legacy_history_path).read_text())[:saved["epoch"]]
    history = saved.get("history") if not legacy else legacy_history
    if history is None:
        raise ValueError("Checkpoint is missing its complete training history")
    validate_history(history, saved["epoch"])
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    assert_finite_state(model, optimizer, epoch=saved["epoch"])
    if saved.get("rng_state") is not None:
        torch.set_rng_state(saved["rng_state"].cpu())
    device = next(model.parameters()).device
    if device.type == "cuda" and saved.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu(), device)
    if legacy:
        warnings.warn("Restoring a legacy checkpoint: source identity is unavailable; configuration, observations, history and finite optimizer state were checked", RuntimeWarning)
    return saved, history
