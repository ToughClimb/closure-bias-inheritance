"""Lazy repository imports and input identities for the isolated follow-up."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
DEFAULT_OUTPUT = REPO / "outputs" / STUDY.name


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_core(output_root, device="cpu"):
    """Never point the revision reader at historical output/cache directories."""
    output_root = Path(output_root).resolve()
    os.environ["EAAI_WORK_ROOT"] = str(output_root)
    os.environ["EAAI_INPUT_ROOT"] = str(output_root)
    os.environ["EAAI_DEVICE"] = device
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    sys.path.insert(0, str(REPO / "revision"))
    core = importlib.import_module("revision_core")
    if core.ROOT != output_root or core.INPUT_ROOT != output_root or core.DEVICE_OPTION != device:
        raise ValueError("Repository import already points to a different work root/device; use a fresh process")
    return core


def input_id(case, setting, seed):
    return f"{case}_{setting}_s{seed}"


def load_input(core, output_root, case, setting, seed):
    root = Path(output_root)
    ident = input_id(case, setting, seed)
    meta = json.loads((root / "inputs" / f"{ident}.json").read_text())
    path = root / "inputs" / f"{ident}.npz"
    if file_hash(path) != meta["file_sha256"]:
        raise ValueError(f"Input file hash mismatch: {ident}")
    if meta["core_source_signature"] != core.source_signature():
        raise ValueError(f"Input generation source changed: {ident}; preserve these inputs and use a new study root")
    with core.np.load(path) as arrays:
        observed = {k: arrays[k].copy() for k in ("u", "x", "t", "u0")}
        value_range = tuple(float(v) for v in arrays["value_range"])
    if core.observation_signature(observed, value_range) != meta["observation_signature"]:
        raise ValueError(f"Input array identity mismatch: {ident}")
    return observed, value_range, meta
