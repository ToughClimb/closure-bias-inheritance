"""Verify native exports in a second tree containing no PT or saved curves.

The child interpreter rejects every file-open under the original repository.
CPU150 is a separately scoped100-checkpoint archive; GPU expects all200.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("cpu150", "gpu"), default="gpu")
    args = parser.parse_args()
    study_name = "reviewer-forward-2026-10-15-v2" if args.cohort == "cpu150" else STUDY.name
    source_study = REPO / "research" / study_name
    source_output = REPO / "outputs" / study_name
    target_output = (REPO / "outputs/reviewer-forward-2026-10-15-cpu150-audit"
                     if args.cohort == "cpu150" else source_output / "reports")
    expected = 100 if args.cohort == "cpu150" else 200
    plan = read(source_study / "MANIFEST.json")
    units = [unit for unit in plan["units"] if unit["kind"] == "model"
             and unit["method"] == "native"
             and (args.cohort != "cpu150" or unit["epoch"] == 150)]
    assert len(units) == expected
    copied = {}
    with tempfile.TemporaryDirectory(prefix="closure-native-portable-") as directory:
        portable = Path(directory) / "independent-checkout"

        def copy(path):
            relative = path.relative_to(REPO)
            destination = portable / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            copied[str(relative)] = digest(path)
            assert digest(destination) == copied[str(relative)]

        for source_root in (REPO / "revision", REPO / "src"):
            for path in source_root.rglob("*.py"):
                copy(path)
        for path in source_study.iterdir():
            if path.is_file() and path.suffix in (".py", ".json"):
                copy(path)
        protocol = read(source_study / "PROTOCOL.json")
        copy(REPO / "outputs" / protocol["upstream_study"] / "INPUT_MANIFEST.json")
        copy(REPO / "research" / protocol["training_manifest_directory"] / "MANIFEST.json")
        copy(REPO / "outputs" / protocol["training_output_directory"] / "LAUNCHED_MANIFEST.json")
        if args.cohort == "gpu":
            copy(source_output / "SHARED_IMPORTS.json")
            copy(source_output / "native_exports/NATIVE_EXPORTS.json")
        for unit in units:
            run = source_output / "runs" / unit["id"]
            copy(run / "result.json")
            if args.cohort == "gpu":
                copy(run / "CHECKPOINT_DEVICE.json")
            identifier = f"{unit['training_id']}_e{unit['epoch']}"
            copy(source_output / "native_exports" / (identifier + ".npz"))
            copy(source_output / "native_exports" / (identifier + ".json"))
        assert not list(portable.rglob("*.pt"))
        assert not list(portable.rglob("curves*.npz"))
        assert len(list((portable / "outputs" / study_name / "native_exports").glob("*.npz"))) == expected
        # runpy needs an explicit script search path. The audit hook makes an
        # accidental absolute source-tree dependency fail the portability gate.
        child = """import os, pathlib, runpy, sys
original = pathlib.Path(sys.argv[1]).resolve()
script = pathlib.Path(sys.argv[2]).resolve()
arguments = sys.argv[3:]
def audit(event, values):
    if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
        path = pathlib.Path(os.fsdecode(values[0])).resolve()
        if path == original or original in path.parents:
            raise RuntimeError('Portable verification attempted original-repository read: ' + str(path))
sys.addaudithook(audit)
sys.path.insert(0, str(script.parent))
sys.argv = [str(script)] + arguments
runpy.run_path(str(script), run_name='__main__')
"""
        command = [sys.executable, "-B", "-c", child, str(REPO),
                   str(portable / "research" / study_name / "export_native.py"), "--verify-only"]
        if args.cohort == "gpu":
            command.append("--require-complete")
        environment = dict(os.environ)
        environment.update(PYTHONPATH="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           NUMEXPR_NUM_THREADS="1", EAAI_DEVICE="cpu")
        result = subprocess.run(command, cwd=portable, env=environment, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError("Portable child failed:\n" + result.stdout + result.stderr)
        verification = read(portable / "outputs" / study_name / "reports/NATIVE_EXPORT_VERIFICATION.json")
        assert verification["exported_unique_checkpoints"] == expected
        assert verification["verification_only"] and not verification["original_checkpoint_files_required"]
        assert verification["existing_exports_numpy_probe_verified"]
        if args.cohort == "gpu":
            assert verification["complete"] and not verification["pending"]
        else:
            assert not verification["complete"] and len(verification["pending"]) == 100
            assert all(identifier.endswith("_e1500") for identifier in verification["pending"])
        report = dict(status="passed", complete=True, cohort=args.cohort,
            snapshot_utc=datetime.now(timezone.utc).isoformat(), expected_unique_checkpoints=expected,
            verified_unique_checkpoints=expected, source_protocol_planned_native_checkpoints=200,
            source_protocol_complete=verification["complete"], excluded_CPU1500_checkpoints=100 if args.cohort == "cpu150" else 0,
            independent_tree_without_PT_or_saved_curves=True, original_repository_file_opens_blocked=True,
            export_npz_numpy_probes_verified=True, verification_only=True, new_fits_or_integrations=0,
            original_pt_files_in_relocated_copy=0, verified_existing_native_exports=expected,
            numpy_inside_outside_probe_verified=True, verifier_exit_code=result.returncode,
            original_checkpoint_files_required=False, temporary_tree_removed=True,
            source_manifest_sha256=digest(source_study / "MANIFEST.json"),
            verifier_script_sha256=digest(Path(__file__)), copied_source_and_artifact_sha256=copied,
            portable_verification=verification,
            interpretation="NPZ weights, sidecars, source and forward identities reproduce verification from a separate tree. The recorded original Torch/PT verification chain is preserved; this test evaluates the packaged NumPy functions and does not claim to rerun Torch against absent PT files.")
    target_output.mkdir(parents=True, exist_ok=True)
    path = target_output / "NATIVE_EXPORT_PORTABILITY_CHECK.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "complete", "cohort", "verified_unique_checkpoints", "original_repository_file_opens_blocked")}))


if __name__ == "__main__":
    main()
