"""Four fixed complete runtime pilots for comparing independent-worker throughput."""
from __future__ import annotations

import argparse
import copy
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import shutil
import time

from run_study import read_manifest, record_environment, run_task
from study_io import STUDY, DEFAULT_OUTPUT, file_hash, load_core

SPECS = [("case_a", "clean", 0), ("case_exp", "clean", 1),
         ("case_a", "noise3", 0), ("case_exp", "noise3", 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = (args.output_root or args.input_root / ("throughput-pilot-" + args.device + "4")).resolve()
    if root == args.input_root.resolve():
        raise ValueError("Runtime pilots must use isolated output directories")
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"; inputs.mkdir(exist_ok=True)
    manifest = read_manifest(STUDY / "MANIFEST.json")
    tasks = []
    for case, setting, seed in SPECS:
        original = next(t for t in manifest["tasks"] if t["case"] == case and t["setting"] == setting and t["data_seed"] == seed and t["variant"] == "default")
        task = dict(copy.deepcopy(original), id=f"runtime_pilot_{case}_{setting}_s{seed}_default",
                    classification="runtime_pilot", max_epochs=150, evaluation_epochs=[150], longer=False)
        tasks.append(task)
        for suffix in ("npz", "json"):
            name = f"{case}_{setting}_s{seed}.{suffix}"
            source, target = args.input_root / "inputs" / name, inputs / name
            if target.exists() and file_hash(source) != file_hash(target):
                raise ValueError("Existing pilot input hash differs")
            if not target.exists():
                shutil.copyfile(source, target)
    manifest = dict(manifest, tasks=tasks, classification="runtime_pilot", task_count=4)
    core = load_core(root, args.device)
    core.json_write(root / "PILOT_MANIFEST.json", manifest)
    record_environment(core, root, args.device)
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_task, task, manifest, root, args.device) for task in tasks]
        completed = [future.result() for future in concurrent.futures.as_completed(futures)]
    wall = time.perf_counter() - started
    reports = []
    for task in tasks:
        row = json.loads((root / "runs" / task["id"] / "result.json").read_text())
        if row["status"] != "complete" or row["completed_epoch"] != 150:
            raise ValueError("Runtime pilot did not finish its complete budget")
        reports.append(dict(id=row["id"], seconds=row["seconds"], initial_model_sha256=row["initial_model_sha256"],
                            data_hash=row["data_hash"], gpu_peak_allocated_mb=row.get("gpu_peak_allocated_mb")))
    report = dict(classification="runtime_pilot", formal_evidence=False, device=args.device, workers=4,
                  complete_runs=len(completed), epochs_per_run=150, total_wall_seconds=wall,
                  seconds_per_run_throughput=wall / len(completed), tasks=reports,
                  implementation_signature=manifest["implementation_signature"],
                  driver_sha256=file_hash(STUDY / "run_study.py"),
                  caution="Runtime routing diagnostic on the present shared machine; not a scientific method comparison")
    core.json_write(root / "THROUGHPUT_PILOT.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
