"""Generate the frozen fresh-study manifest without importing numerical libraries."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def implementation_signature():
    files = sorted((REPO / "src").rglob("*.py")) + sorted((REPO / "revision").glob("*.py"))
    files += [STUDY / name for name in ("make_manifest.py", "run_study.py", "study_io.py", "prepare_inputs.py")]
    files.append(STUDY / "NUMERICAL_POLICY.json")
    files.append(STUDY / "GPU_COHORT_AMENDMENT.json")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_manifest(protocol):
    default = protocol["default_weights"]
    configs = {}

    def add(name, weights, phase, factors, weak_variant="space_weak"):
        key = (weak_variant,) + tuple(weights[k] for k in default)
        if key in configs:
            return
        configs[key] = dict(variant=name, weights=weights, phase=phase, factors=factors, weak_variant=weak_variant)

    add("default", default.copy(), "core", {})
    for term in default:
        for factor in protocol["core_oat_factors"]:
            if factor == 1:
                continue
            weights = default.copy()
            weights[term] *= factor
            name = "drop_" + term if factor == 0 else f"{term}_x{factor:g}"
            add(name, weights, "core", {term: factor})
    for term, factors in protocol["dense_oat_additional_factors"].items():
        for factor in factors:
            weights = default.copy()
            weights[term] *= factor
            add(f"{term}_x{factor:g}", weights, "dense", {term: factor})
    first, second = protocol["interaction_terms"]
    for a in protocol["interaction_factors"]:
        for b in protocol["interaction_factors"]:
            weights = default.copy()
            weights[first] *= a
            weights[second] *= b
            add(f"joint_{first}_x{a:g}_{second}_x{b:g}", weights, "joint", {first: a, second: b})
    for name, arm in protocol["temporal_control_variants"].items():
        weights = default.copy()
        for term in arm["weights_zero"]:
            weights[term] = 0.0
        add(name, weights, "weak", {}, weak_variant=arm["weak_variant"])
    tasks = []
    for case in protocol["cases"]:
        for setting in protocol["settings"]:
            for seed in protocol["data_seeds"]:
                for config in configs.values():
                    longer = config["variant"] in protocol["longer_variants"]
                    epochs = protocol["longer_checkpoints"] if longer else [protocol["fixed_budget"]]
                    tasks.append(dict(config, id=f"fresh_{case}_{setting}_s{seed}_{config['variant']}",
                                      case=case, setting=setting, data_seed=seed, init_seed=seed,
                                      classification="fresh_followup", max_epochs=max(epochs),
                                      evaluation_epochs=epochs, longer=longer))
    counts = dict(Counter(task["phase"] for task in tasks))
    assert len(configs) == 34 and len(tasks) == 680, (len(configs), len(tasks))
    assert counts == {"core": 380, "dense": 180, "joint": 80, "weak": 40}, counts
    assert len({task["id"] for task in tasks}) == len(tasks)
    parent_manifest = json.loads((REPO / "research/reviewer-sensitivity-2026-10-15-v2/MANIFEST.json").read_text())
    parent_tasks = {task["id"]: task for task in parent_manifest["tasks"]}
    gpu_tasks = []
    for task in tasks:
        if not task["longer"]:
            continue
        original_id = task["id"]
        if parent_tasks[original_id] != task:
            raise ValueError("GPU scientific task differs from the frozen CPU protocol")
        parent_path = REPO / "outputs/reviewer-sensitivity-2026-10-15-v2/runs" / original_id / "result.json"
        parent = json.loads(parent_path.read_text())
        if parent["config"] != task or parent["identity"]["implementation_signature"] != parent_manifest["implementation_signature"]:
            raise ValueError("CPU reference metadata identity mismatch")
        gpu_tasks.append(dict(task, id=original_id.replace("fresh_", "gpu_", 1),
                              cpu_reference_task_id=original_id,
                              paired_cpu_initial_model_sha256=parent["initial_model_sha256"],
                              cpu_reference_metadata_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                              required_device="cuda", cohort="fresh_cuda_long"))
    tasks = gpu_tasks
    counts = dict(Counter(task["phase"] for task in tasks))
    assert len(tasks) == 100 and all(task["max_epochs"] == 1500 for task in tasks)
    assert {task["variant"] for task in tasks} == set(protocol["longer_variants"])
    amendment = json.loads((STUDY / "GPU_COHORT_AMENDMENT.json").read_text())
    return dict(study_id=amendment["cohort_id"], protocol=protocol,
                cohort_amendment=amendment, required_device="cuda",
                cohort_amendment_sha256=hashlib.sha256((STUDY / "GPU_COHORT_AMENDMENT.json").read_bytes()).hexdigest(),
                parent_cpu_implementation_signature=parent_manifest["implementation_signature"],
                numerical_policy_sha256=hashlib.sha256((STUDY / "NUMERICAL_POLICY.json").read_bytes()).hexdigest(),
                protocol_sha256=canonical_hash(protocol), implementation_signature=implementation_signature(),
                inputs=20, unique_weight_objective_configurations=5,
                unique_weight_vectors=len({tuple(c["weights"][k] for k in default) for c in tasks}),
                task_count=len(tasks),
                phase_counts=counts, longer_task_count=sum(task["longer"] for task in tasks),
                fixed_budget_optimizer_steps=sum(min(protocol["fixed_budget"], task["max_epochs"]) for task in tasks),
                full_declared_optimizer_steps=sum(task["max_epochs"] for task in tasks), tasks=tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=STUDY / "PROTOCOL.json")
    parser.add_argument("--output", type=Path, default=STUDY / "MANIFEST.json")
    args = parser.parse_args()
    manifest = make_manifest(json.loads(args.protocol.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key not in ("tasks", "protocol")}, indent=2))


if __name__ == "__main__":
    main()
