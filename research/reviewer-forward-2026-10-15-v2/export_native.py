"""Export each saved forward checkpoint once as verified native NumPy weights.

No training, symbolic fitting, interpolation, or PDE integration is performed.
Only already-evaluated forward states can supply the comparison metadata.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import run_forward as forward

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Verify packaged NPZ/JSON integrity and forward provenance without original PT files")
    args = parser.parse_args()
    plan = forward.read_plan()
    core = forward.load_core()
    from reliability_core import NumpyMLPClosure
    upstream = json.loads(forward.UPSTREAM_MANIFEST.read_text())
    tasks = {task["id"]: task for task in upstream["tasks"]}
    target = forward.OUTPUT / "native_exports"
    target.mkdir(parents=True, exist_ok=True)
    records, pending = [], []
    for unit in plan["units"]:
        if unit["kind"] != "model" or unit["method"] != "native":
            continue
        identifier = f"{unit['training_id']}_e{unit['epoch']}"
        result_path = forward.OUTPUT / "runs" / unit["id"] / "result.json"
        if not result_path.exists():
            pending.append(identifier)
            continue
        result = json.loads(result_path.read_text())
        assert result["identity"]["unit"] == unit
        assert result["identity"]["protocol_sha256"] == plan["protocol_sha256"]
        assert result["identity"]["driver_sha256"] == plan["driver_sha256"]
        source = forward.TRAINING_ROOT / "runs" / unit["training_id"] / f"checkpoint_{unit['epoch']}.pt"
        checkpoint_sha = result["metadata"]["checkpoint_sha256"]
        if not args.verify_only:
            assert forward.digest(source) == checkpoint_sha
        identity = dict(training_id=unit["training_id"], epoch=unit["epoch"], checkpoint_sha256=checkpoint_sha,
            input_file_sha256=result["identity"]["input_file_sha256"], protocol_sha256=plan["protocol_sha256"],
            driver_sha256=plan["driver_sha256"], model_metadata_sha256=result["identity"]["model_metadata_sha256"])
        export_path, metadata_path = target / (identifier + ".npz"), target / (identifier + ".json")
        if metadata_path.exists():
            record = json.loads(metadata_path.read_text())
            assert record["identity"] == identity, identifier
            assert export_path.exists() and forward.digest(export_path) == record["export_sha256"]
            assert record["frozen_expressions"] == result["metadata"]["frozen_expressions"]
            assert record["model_verification"] == result["metadata"]["model_verification"]
            assert forward.digest(result_path) == record["source_forward_result_sha256"]
            assert record["reload_arrays_bitwise_equal"] and record["reload_evaluation_bitwise_equal"]
            loaded = NumpyMLPClosure.load(export_path)
            assert loaded.metadata == record["native_metadata"]
            probe = np.linspace(record["value_range"][0] - .1, record["value_range"][1] + .1, 257)
            assert np.isfinite(loaded.diffusion(probe)).all() and (loaded.diffusion(probe) > 0).all()
            assert np.isfinite(loaded.reaction(probe)).all()
        else:
            if args.verify_only:
                pending.append(identifier)
                continue
            _, interval, input_metadata = forward.load_input(core, f"{unit['case']}_{unit['setting']}_s{unit['data_seed']}")
            assert input_metadata["file_sha256"] == identity["input_file_sha256"]
            methods, metadata = forward.load_models(core, tasks[unit["training_id"]], unit["epoch"], interval)
            metadata = forward.safe(metadata)
            assert metadata == result["metadata"], identifier
            native = methods["native"][0].__self__
            assert isinstance(native, NumpyMLPClosure)
            native.save(export_path)
            reloaded = NumpyMLPClosure.load(export_path)
            assert native.metadata == reloaded.metadata
            assert native.arrays.keys() == reloaded.arrays.keys()
            for key in native.arrays:
                np.testing.assert_array_equal(native.arrays[key], reloaded.arrays[key])
            probe = np.linspace(interval[0] - .1, interval[1] + .1, 257)
            for branch in ("diffusion", "reaction"):
                np.testing.assert_array_equal(getattr(native, branch)(probe), getattr(reloaded, branch)(probe))
            record = dict(identity=identity, export_file=export_path.name, export_sha256=forward.digest(export_path),
                export_bytes=export_path.stat().st_size, source_checkpoint=str(source),
                source_forward_result=str(result_path), source_forward_result_sha256=forward.digest(result_path),
                model_verification=metadata["model_verification"], native_metadata=metadata["native_metadata"],
                reload_arrays_bitwise_equal=True, reload_evaluation_bitwise_equal=True,
                frozen_expressions=metadata["frozen_expressions"],
                value_range=list(interval), fits_or_integrations_performed=0,
                scope="Same trained polynomial-residual MLP weights used by native forward evaluation, including outside U; no saved-curve interpolation. Frozen expression parameters are exact stored binary-float JSON values, not rounded display strings.")
            forward.write(metadata_path, record)
        records.append(dict(id=identifier, export_file=export_path.name, metadata_file=metadata_path.name,
            export_sha256=record["export_sha256"], metadata_sha256=forward.digest(metadata_path),
            checkpoint_sha256=checkpoint_sha, bytes=record["export_bytes"]))
    assert len(records) + len(pending) == 200
    manifest = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), complete=not pending,
        verification_only=args.verify_only, original_checkpoint_files_required=not args.verify_only,
        existing_exports_numpy_probe_verified=True,
        planned_unique_checkpoints=200, exported_unique_checkpoints=len(records), pending=pending,
        source_forward_manifest_sha256=forward.digest(forward.STUDY / "MANIFEST.json"),
        export_script_sha256=forward.digest(__file__),
        evaluator_source="revision/reliability_core.py:NumpyMLPClosure",
        evaluator_source_sha256=forward.digest(forward.REPO / "revision/reliability_core.py"),
        total_export_bytes=sum(record["bytes"] for record in records), records=records,
        statistical_rule="These 200 files are portable representations of the existing 200 unique checkpoints; native/frozen evaluation does not create new training replicates.")
    if args.verify_only:
        forward.write(forward.OUTPUT / "reports/NATIVE_EXPORT_VERIFICATION.json", manifest)
    else:
        forward.write(target / "NATIVE_EXPORTS.json", manifest)
        (target / "README.txt").write_text("Native function exports (no table interpolation)\n\n"
        "Each NPZ stores actual float64 copies of the original float32-trained MLP weights, polynomial backbone, normalization and diffusion transform.\n"
        "Use revision/reliability_core.py:NumpyMLPClosure.load(path); closure.diffusion(u) and closure.reaction(u) evaluate the formulas at any u.\n"
        "These APIs reproduce the current forward calculation without the much larger training PT files. See research/reviewer-forward-2026-10-15-v2/run_forward.py:strict_integrate for the frozen PDE discretization/guards.\n"
        "Each adjacent JSON links the original checkpoint/input SHA, actual forward metadata, outside-support Torch verification, bitwise NPZ reload verification and frozen symbolic expressions.\n"
        "The symbolic parameters are exact stored JSON numbers; the human-readable expression field is rounded and must not be parsed for evaluation.\n"
        "NATIVE_EXPORTS.json gives all planned/pending/complete counts and source hashes. Reused representations are not independent scientific replicates.\n"
        "Verify packaged exports without original PT files: python -B research/reviewer-forward-2026-10-15-v2/export_native.py --verify-only --require-complete\n"
        "This verification checks NPZ/sidecar/forward identities and the recorded original Torch verification chain; regenerating exports or repeating Torch-vs-original-PT verification requires the original PT files.\n")
    print(json.dumps({key:manifest[key] for key in ("complete", "planned_unique_checkpoints", "exported_unique_checkpoints", "total_export_bytes")}, indent=2))
    if args.require_complete and pending:
        raise RuntimeError(f"Still missing {len(pending)} native exports; complete export refused")


if __name__ == "__main__":
    main()
