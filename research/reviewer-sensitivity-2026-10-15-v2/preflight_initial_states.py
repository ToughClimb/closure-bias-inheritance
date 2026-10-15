"""Check all 34 weight/objective configurations on each of 20 inputs, with no fitting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import traceback

from run_study import read_manifest, context, diagnose, initial_model_hash, record_environment
from study_io import STUDY, DEFAULT_OUTPUT, canonical_hash, load_core, load_input


def one(task, manifest, root):
    core = load_core(root, "cpu"); torch = core.torch; torch.set_num_threads(1)
    path = root / "preflight" / (task["id"] + ".json")
    row = dict(id=task["id"], config=task, classification="numerical_preflight", training_performed=False,
               implementation_signature=manifest["implementation_signature"], status="running")
    try:
        observed, interval, meta = load_input(core, root, task["case"], task["setting"], task["data_seed"])
        row["data_hash"] = meta["data_hash"]
        torch.manual_seed(task["init_seed"])
        model = core.ReactionDiffusionClosure(input_range=interval)
        row["initial_model_sha256"] = initial_model_hash(model)
        ctx = context(core, model, observed, interval, manifest["protocol"], task["weak_variant"], torch.device("cpu"))
        row["diagnostic"] = diagnose(core, ctx, task["weights"], 0)
        row["status"] = "passed"
    except Exception as exc:
        row.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    core.json_write(path, row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.output_root.resolve(); manifest = read_manifest(STUDY / "MANIFEST.json")
    core = load_core(root, "cpu")
    record_environment(core, root, "cpu", dict(classification="numerical_preflight", training_performed=False,
                                               workers=args.workers, selected_task_count=len(manifest["tasks"]),
                                               protocol_sha256=manifest["protocol_sha256"], manifest_sha256=canonical_hash(manifest)))
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(one, task, manifest, root) for task in manifest["tasks"]]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            if len(records) % 50 == 0:
                print(len(records), dict(Counter(r["status"] for r in records)), flush=True)
    initial, data = defaultdict(set), defaultdict(set)
    for row in records:
        key = (row["config"]["case"], row["config"]["setting"], row["config"]["data_seed"])
        if row["status"] == "passed":
            initial[key].add(row["initial_model_sha256"]); data[key].add(row["data_hash"])
    checks = dict(all_680_states_checked=len(records) == 680,
                  all_high_precision_gates_pass=all(r["status"] == "passed" for r in records),
                  same_initial_model_state_within_input=len(initial) == 20 and all(len(v) == 1 for v in initial.values()),
                  same_observations_within_input=len(data) == 20 and all(len(v) == 1 for v in data.values()),
                  no_training_performed=all(r["training_performed"] is False for r in records))
    report = dict(classification="numerical_preflight", training_performed=False,
                  implementation_signature=manifest["implementation_signature"],
                  numerical_policy_sha256=manifest["numerical_policy_sha256"],
                  checks=checks, all_checks_passed=all(checks.values()), status_counts=dict(Counter(r["status"] for r in records)),
                  f32_legacy_heuristic_failures=sum(not r["diagnostic"]["legacy_f32_heuristic_pass"] for r in records if r["status"] == "passed"),
                  maximum_f64_linearity_error=max((r["diagnostic"]["high_precision_validation"]["gradient_linearity_error"] for r in records if r["status"] == "passed"), default=None),
                  failed_ids=[r["id"] for r in records if r["status"] != "passed"])
    core.json_write(root / "PREFLIGHT_INITIAL_STATES.json", report)
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise RuntimeError("Initial-state gate failed; inspect retained preflight records")


if __name__ == "__main__":
    main()
