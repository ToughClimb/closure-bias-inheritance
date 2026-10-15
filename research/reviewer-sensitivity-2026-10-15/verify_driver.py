"""Numerical checks: diagnostics preserve updates and resumed Adam matches fresh Adam.

These three-epoch test runs are runtime_pilot artifacts and cannot enter formal
study summaries. They use one declared shared observation array.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from run_study import read_manifest, run_task
from study_io import STUDY, DEFAULT_OUTPUT, file_hash, load_core, load_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--variant", choices=("default", "reduced_space_weak", "space_time_weak"), default="default")
    args = parser.parse_args()
    root = (args.output_root or args.input_root / ("driver-verification-" + args.device)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root == args.input_root.resolve():
        raise ValueError("Verification artifacts require an isolated root")
    inputs = root / "inputs"; inputs.mkdir(exist_ok=True)
    for suffix in ("npz", "json"):
        name = f"case_exp_clean_s0.{suffix}"
        shutil.copyfile(args.input_root / "inputs" / name, inputs / name)
    manifest = read_manifest(STUDY / "MANIFEST.json")
    original = next(t for t in manifest["tasks"] if t["case"] == "case_exp" and t["setting"] == "clean" and t["data_seed"] == 0 and t["variant"] == args.variant)
    first = dict(copy.deepcopy(original), id="verification_uninterrupted", classification="runtime_pilot", max_epochs=3, evaluation_epochs=[1, 3], longer=False)
    second = dict(copy.deepcopy(original), id="verification_resumed", classification="runtime_pilot", max_epochs=3, evaluation_epochs=[1, 3], longer=False)
    core = load_core(root, args.device)
    core.json_write(root / "VERIFICATION_MANIFEST.json", dict(manifest, tasks=[first, second], classification="runtime_pilot"))
    observed, interval, meta = load_input(core, root, "case_exp", "clean", 0)
    _, baseline_history, baseline_snaps = core.train(observed, interval, epochs=3, init_seed=0,
                                                    weights=first["weights"], variant=first["weak_variant"], checkpoints=[1, 3])
    run_task(first, manifest, root, args.device)
    run_task(second, manifest, root, args.device, stop_epoch=1)
    run_task(second, manifest, root, args.device, stop_epoch=3)
    u = root / "runs" / first["id"]; r = root / "runs" / second["id"]
    fresh = json.loads((u / "result.json").read_text()); resumed = json.loads((r / "result.json").read_text())
    history = json.loads((u / "history.json").read_text()); resumed_history = json.loads((r / "history.json").read_text())
    with core.np.load(u / "curves_3.npz") as arrays:
        curves = {k: arrays[k].copy() for k in ("D", "R")}
    with core.np.load(r / "curves_3.npz") as arrays:
        resumed_curves = {k: arrays[k].copy() for k in ("D", "R")}
    checks = dict(
        exact_match_to_existing_training_loop=all(core.np.array_equal(curves[k], baseline_snaps[3][k]) for k in ("D", "R")),
        exact_loss_history_match_to_existing_loop=all(a["loss"] == b["loss"] for a, b in zip(history, baseline_history)),
        resumed_curves_bitwise_equal=all(core.np.array_equal(curves[k], resumed_curves[k]) for k in ("D", "R")),
        resumed_history_equal=(history == resumed_history),
        resumed_metrics_equal=(fresh["epochs"] == resumed["epochs"]),
        resumed_gradient_diagnostics_equal=(fresh["gradient_diagnostics"] == resumed["gradient_diagnostics"]),
        initial_model_hashes_equal=(fresh["initial_model_sha256"] == resumed["initial_model_sha256"]),
        both_true_input_hashes_equal=(fresh["data_hash"] == resumed["data_hash"] == meta["data_hash"]),
        all_D_R_parameter_blocks_present=all(set(row["parameter_blocks"]) == {"all", "D", "R"} for row in fresh["gradient_diagnostics"].values()),
        no_formal_evidence=all(value["classification"] == "runtime_pilot" for value in (fresh, resumed)),
    )
    report = dict(classification="runtime_pilot", formal_evidence=False, device=args.device,
                  all_checks_passed=all(checks.values()), checks=checks,
                  input_data_hash=meta["data_hash"], initial_model_sha256=fresh["initial_model_sha256"],
                  implementation_signature=manifest["implementation_signature"],
                  driver_sha256=file_hash(STUDY / "run_study.py"),
                  variant=args.variant,
                  validation_epochs=3)
    core.json_write(root / "DRIVER_VERIFICATION.json", report)
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise AssertionError("Numerical driver verification failed")


if __name__ == "__main__":
    main()
