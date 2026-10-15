"""One-worker regularization intervention using frozen residual QR matrices."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
SOURCE_STUDY = REPO / "research/reviewer-objective-controls-2026-10-15"
sys.path.insert(0, str(SOURCE_STUDY))
import numpy as np
from objective_controls import (Term, TERMS, DEFAULT_WEIGHTS, file_hash, write_json,
    load_core, read_input, least_squares, positive_diffusion, diffusion_minimum, evaluation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs" / STUDY.name)
    parser.add_argument("--input-root", type=Path, default=REPO / "outputs/reviewer-sensitivity-2026-10-15/inputs")
    parser.add_argument("--bundle-root", type=Path, default=REPO / "outputs/reviewer-objective-controls-2026-10-15/shared/bundles")
    parser.add_argument("--prepare-only", action="store_true", help="Record and verify the current inputs without solving")
    args = parser.parse_args()
    out = args.output_root.resolve()
    protocol = json.loads((STUDY / "PROTOCOL.json").read_text())
    inputs, bundles = args.input_root.resolve(), args.bundle_root.resolve()
    bundle_paths = sorted(bundles.glob("*.json"))
    if len(bundle_paths) != 20:
        raise ValueError("Expected all 20 paired objective bundles before starting this study")
    paths = {"source/PROTOCOL.json": STUDY / "PROTOCOL.json",
             "source/objective_controls.py": SOURCE_STUDY / "objective_controls.py"}
    for path in bundle_paths:
        bundle = json.loads(path.read_text())
        required = {"observed_features_observed_targets__" + variant
                    for variant in ("space_weak", "space_time_weak")}
        if not required.issubset(bundle["matrices"]):
            raise ValueError("Incomplete objective bundle: run the shared controls with all declared variants")
        paths["bundles/" + path.name] = path
        for suffix in (".npz", ".json"):
            paths["inputs/" + path.stem + suffix] = inputs / (path.stem + suffix)
    freeze = {"files": {name: file_hash(path) for name, path in paths.items()},
              "scope": "Current public-checkout sources and supplied inputs, frozen before solving"}
    out.mkdir(parents=True, exist_ok=True)
    if (out / "RUN_MANIFEST.json").exists():
        raise FileExistsError("Preserve this run; choose a new output root for any rerun")
    snapshot_path = out / "INPUT_SNAPSHOT.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != freeze:
        raise ValueError("Frozen input snapshot differs; use a new output directory")
    write_json(snapshot_path, freeze)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "files": len(paths)}))
        return
    core = load_core(out)
    environment = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
        packages=sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
        cpu_workers=1, blas_threads=1, device="cpu",
        git_status=subprocess.run(["git", "status", "--short"], cwd=REPO,
            text=True, capture_output=True, check=True).stdout)
    write_json(out / "ENVIRONMENT.json", environment)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        classification=protocol["classification"], conditions=320,
        repeated_reference_conditions=80, new_interventions=240,
        protocol_sha256=file_hash(STUDY / "PROTOCOL.json"), freeze_sha256=file_hash(snapshot_path),
        runner_sha256=file_hash(Path(__file__)), imported_solver_sha256=file_hash(SOURCE_STUDY / "objective_controls.py"),
        core_source_signature=core.source_signature(), source_files=freeze["files"])
    write_json(out / "RUN_MANIFEST.json", manifest)
    write_json(out / "COMPLETION.json", dict(status="running", completed=0, failed=0))
    start = time.perf_counter()
    records = []
    for path in bundle_paths:
        bundle = json.loads(path.read_text())
        meta = bundle["source_metadata"]
        input_path = inputs / (meta["id"] + ".npz")
        arrays, input_meta = read_input(input_path)
        if (input_meta["file_sha256"] != bundle["job"]["input_sha256"]
                or input_meta["observation_signature"] != meta["observation_signature"]):
            raise ValueError("Bundle/input identity mismatch")
        case = core.extra_case(meta["case"])
        interval = tuple(meta["value_range"])
        if tuple(case.value_range) != interval:
            raise ValueError("Evaluation interval changed")
        matrices = {variant: {key: Term(np.asarray(value["augmented_qr"], dtype=float),
                    value["rows"], value["normalization_count"])
                    for key, value in bundle["matrices"]["observed_features_observed_targets__" + variant].items()}
                    for variant in ("space_weak", "space_time_weak")}
        for variant in protocol["variants"]:
            weak_variant = "space_weak" if variant == "reduced_space_weak" else "space_time_weak"
            terms = matrices[weak_variant]
            reference_weights = dict(protocol["other_weights"], reg=protocol["reference_regularization_weight"])
            for factor in protocol["regularization_factors"]:
                weights = dict(reference_weights, reg=reference_weights["reg"] * factor)
                for constraint in protocol["constraint_arms"]:
                    ident = f"{meta['id']}__{variant}__reg{factor:g}__{constraint}"
                    record = dict(id=ident, source_input_id=meta["id"], case=meta["case"],
                        setting=meta["setting"], data_seed=meta["data_seed"],
                        data_hash=meta["data_hash"], observation_signature=meta["observation_signature"],
                        source_bundle_sha256=file_hash(path), variant=variant, weak_variant=weak_variant,
                        reg_factor=factor, weights=weights, constraint=constraint,
                        repeated_reference=(factor == 1), status="running")
                    try:
                        if constraint == "unconstrained":
                            theta, solver, _ = least_squares(terms, weights)
                        else:
                            theta, solver = positive_diffusion(terms, weights, interval)
                        record["solver"] = solver
                        record["coefficients"] = theta.tolist()
                        record["evaluation"] = evaluation(theta, case, arrays["u"])
                        losses = {key: terms[key].loss(theta) for key in TERMS}
                        weighted = {key: weights[key] * losses[key] for key in TERMS}
                        original_losses = {key: matrices["space_weak"][key].loss(theta) for key in TERMS}
                        objective = sum(weighted.values())
                        record.update(losses=losses, weighted_losses=weighted, objective=objective,
                            no_reg_reduced_objective=sum(v for k, v in weighted.items() if k != "reg"),
                            regularizer_fraction=weighted["reg"] / objective if objective else None,
                            common_same_variant_default_reg_objective=sum(reference_weights[k] * losses[k] for k in TERMS),
                            common_original_hybrid_objective=sum(DEFAULT_WEIGHTS[k] * original_losses[k] for k in TERMS),
                            original_space_weak_losses=original_losses,
                            continuous_minimum_diffusion=diffusion_minimum(theta, interval))
                        if solver["status"] != "passed":
                            raise ArithmeticError(f"Solver certificate failed: {solver['status']}")
                        if factor == 1:
                            refs = [r for r in bundle["results"] if r["variant"] == variant
                                    and r["constraint"] == constraint and not r["oracle"]]
                            if len(refs) != 1:
                                raise ValueError("Nonunique original factor-one reference")
                            reference = refs[0]
                            comparisons = dict(coefficients=float(np.max(np.abs(theta - reference["solver"]["coefficients"])
                                / np.maximum(1., np.abs(reference["solver"]["coefficients"])))),
                                losses=max(abs(losses[k] - reference["solver"]["losses"][k])
                                    / max(1., abs(reference["solver"]["losses"][k])) for k in TERMS),
                                evaluation=max(abs(record["evaluation"][k] - reference["evaluation"][k])
                                    / max(1., abs(reference["evaluation"][k])) for k in ("E_D", "E_R")))
                            record["reference_reproduction"] = comparisons
                            if max(comparisons.values()) > 1e-10:
                                raise AssertionError(f"Factor-one reproduction failed: {comparisons}")
                        record["status"] = "passed"
                    except Exception as exc:
                        record.update(status="failed", error=str(exc), traceback=traceback.format_exc())
                    write_json(out / "conditions" / (ident + ".json"), record)
                    records.append(record)
    assert len(records) == 320
    failed = sum(r["status"] != "passed" for r in records)
    repeated = [r for r in records if r["repeated_reference"]]
    summary = dict(status="passed" if not failed else "completed_with_failures", total=320,
        passed=320 - failed, failed=failed, repeated_reference_count=len(repeated), new_interventions=240,
        elapsed_seconds=time.perf_counter() - start,
        maximum_reference_reproduction_difference=max((max(r.get("reference_reproduction", {"failure": 1.}).values()) for r in repeated), default=None),
        finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "COMPLETION.json", summary)
    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
