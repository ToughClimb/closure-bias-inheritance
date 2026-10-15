"""One approved CUDA/CPU runtime pilot, isolated from all formal result rows."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

from run_study import read_manifest, record_environment, run_task
from study_io import STUDY, DEFAULT_OUTPUT, file_hash, load_core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=STUDY / "MANIFEST.json")
    args = parser.parse_args()
    root = (args.output_root or args.input_root / ("runtime-pilot-" + args.device)).resolve()
    if root == args.input_root.resolve():
        raise ValueError("Pilot outputs must use a separate subdirectory")
    manifest = read_manifest(args.manifest)
    task = copy.deepcopy(next(t for t in manifest["tasks"] if t["case"] == "case_exp" and t["setting"] == "clean" and t["data_seed"] == 0 and t["variant"] == "default"))
    task.update(id="runtime_pilot_case_exp_clean_s0_default", classification="runtime_pilot",
                max_epochs=150, evaluation_epochs=[150], longer=False)
    manifest = dict(manifest, classification="runtime_pilot", tasks=[task], task_count=1)
    inputs = root / "inputs"; inputs.mkdir(parents=True, exist_ok=True)
    for suffix in ("npz", "json"):
        source = args.input_root / "inputs" / f"case_exp_clean_s0.{suffix}"
        destination = inputs / source.name
        if destination.exists() and file_hash(destination) != file_hash(source):
            raise ValueError("Existing pilot inputs differ from the declared shared input")
        if not destination.exists():
            shutil.copyfile(source, destination)
    shutil.copyfile(args.input_root / "INPUT_MANIFEST.json", root / "INPUT_MANIFEST.json")
    core = load_core(root, args.device)
    record_environment(core, root, args.device)
    core.json_write(root / "PILOT_MANIFEST.json", manifest)
    started = time.perf_counter()
    ident, status = run_task(task, manifest, root, args.device)
    result = json.loads((root / "runs" / ident / "result.json").read_text())
    report = dict(classification="runtime_pilot", id=ident, status=status,
                  wall_seconds=time.perf_counter() - started, device=args.device,
                  data_hash=result["data_hash"], gradient_epochs=sorted(map(int, result["gradient_diagnostics"])),
                  source_signature=result["identity"]["source_signature"],
                  implementation_signature=manifest["implementation_signature"],
                  run_study_sha256=file_hash(STUDY / "run_study.py"),
                  pilot_driver_sha256=file_hash(Path(__file__)),
                  final_metrics=result["epochs"]["150"],
                  gpu_peak_allocated_mb=result.get("gpu_peak_allocated_mb"),
                  formal_evidence=False)
    core.json_write(root / "PILOT_REPORT.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "final_metrics"}, indent=2))


if __name__ == "__main__":
    main()
