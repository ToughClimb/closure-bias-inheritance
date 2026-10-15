"""Generate and hash the twenty fresh paired datasets; never fit a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from study_io import STUDY, DEFAULT_OUTPUT, canonical_hash, file_hash, input_id, load_core, load_input


def prepare(protocol, root):
    core = load_core(root, "cpu")
    source = core.source_signature()
    protocol_hash = canonical_hash(protocol)
    directory = root / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name in protocol["cases"]:
        case = core.extra_case(name)
        simulation = core.SimulationConfig1D(**protocol["simulation"])
        for seed in protocol["data_seeds"]:
            raw = None
            for setting, noise in protocol["settings"].items():
                ident = input_id(name, setting, seed)
                path = directory / f"{ident}.npz"
                metadata_path = directory / f"{ident}.json"
                if path.exists() or metadata_path.exists():
                    _, _, prior = load_input(core, root, name, setting, seed)
                    if prior["protocol_sha256"] != protocol_hash:
                        raise ValueError(f"Input protocol mismatch: {ident}")
                    records.append(prior)
                    continue
                if raw is None:
                    raw = core.generate_dataset(case, simulation, protocol["num_trajectories"], seed=seed,
                                                amplitude_range=case.value_range,
                                                initial_clip_range=case.value_range,
                                                num_modes=protocol["num_initial_modes"])
                observed = core.apply_observation_model(raw, noise_level=noise, space_stride=1, time_stride=1,
                                                       seed=seed, clip_range=case.value_range)
                if not core.np.isfinite(raw["u"]).all() or not core.np.isfinite(observed["u"]).all():
                    raise FloatingPointError(f"Nonfinite generated observations: {ident}")
                core.npz_write(path, **observed, clean_u=raw["u"], clean_u0=raw["u0"],
                               value_range=core.np.asarray(case.value_range, dtype=core.np.float64))
                meta = dict(id=ident, case=name, setting=setting, data_seed=seed,
                            classification="fresh_input", protocol_sha256=protocol_hash,
                            core_source_signature=source, file=str(path.relative_to(root)),
                            file_sha256=file_hash(path),
                            data_hash=hashlib.sha256(observed["u"].tobytes()).hexdigest(),
                            clean_data_hash=hashlib.sha256(raw["u"].tobytes()).hexdigest(),
                            observation_signature=core.observation_signature(observed, case.value_range),
                            shape=list(observed["u"].shape), value_range=list(case.value_range),
                            simulation=protocol["simulation"], num_trajectories=protocol["num_trajectories"],
                            observation=core.observation_diagnostics(raw, observed, case, seed, noise),
                            generation_clipping=protocol["generation_clip"],
                            generated_utc=datetime.now(timezone.utc).isoformat())
                core.json_write(metadata_path, meta)
                records.append(meta)
                print(ident, meta["data_hash"], flush=True)
    assert len(records) == 20 and len({r["id"] for r in records}) == 20
    for name in protocol["cases"]:
        for seed in protocol["data_seeds"]:
            hashes = {r["clean_data_hash"] for r in records if r["case"] == name and r["data_seed"] == seed}
            assert len(hashes) == 1
    manifest = dict(study_id=protocol["study_id"], protocol_sha256=protocol_hash,
                    core_source_signature=source, array_schema={
                        "u": "observed field [8,101,64], float64",
                        "clean_u": "paired clean synthetic field, float64",
                        "x": "periodic saved spatial coordinates", "t": "saved times",
                        "u0": "first observed snapshot", "clean_u0": "clean initial fields",
                        "value_range": "two-element closure/support interval"},
                    inputs=records, count=len(records), training_performed=False,
                    selection_rule=protocol["selection_rule"])
    core.json_write(root / "INPUT_MANIFEST.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=STUDY / "PROTOCOL.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare(json.loads(args.protocol.read_text()), args.output_root.resolve())
    print(json.dumps({"inputs": manifest["count"], "manifest": str(args.output_root.resolve() / "INPUT_MANIFEST.json"),
                      "training_performed": False}, indent=2))


if __name__ == "__main__":
    main()
