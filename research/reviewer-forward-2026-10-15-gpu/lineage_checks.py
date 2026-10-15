"""Independent postprocessing checks for GPU records and original-byte imports.

This helper reads JSON receipts only; it does not import model/PDE code or open
original checkpoints. Paths for the GPU training manifest are derived from the
current checkout so a relocated native-export package remains verifiable.
"""
import hashlib
import json
from pathlib import Path

STUDY=Path(__file__).resolve().parent
REPO=STUDY.parents[1]
OUTPUT=REPO/"outputs"/STUDY.name

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, role, sources=None):
    return sources.json(path, role) if sources else json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def imports(plan,sources=None):
    path=OUTPUT/"SHARED_IMPORTS.json"
    value=read(path,"shared_import_index",sources)
    check(digest(path)==plan["shared_imports_sha256"],"Shared import index hash changed")
    check(value["status"]=="passed" and value["complete"] is True,"Shared import index incomplete")
    check(value["target_study"]==plan["study_id"] and value["input_manifest_sha256"]==plan["input_manifest_sha256"],"Shared import cohort mismatch")
    check(value["reference_states"]==plan["shared_imported_states"]==100
          and value["total_integration_records_reused"]==plan["existing_integration_records_reused"]==480
          and value["new_fits"]==value["new_integrations"]==0,"Shared import denominators changed")
    check(value["original_result_and_fields_bytes_preserved"] is True
          and value["original_metadata_source_sha_preserved"] is True,"Imported identities were rewritten")
    physics=value["physics_equivalence"]
    check(physics["status"]=="passed" and physics["all_11_function_bodies_byte_and_ast_identical"] is True
          and physics["target_driver_sha256"]==plan["driver_sha256"]
          and not physics["changed_physics_fields"],"Imported physics-equivalence receipt mismatch")
    lookup={(row["kind"],row["unit_id"]):row for row in value["entries"]}
    expected={("imported_reference",unit["id"]) for unit in plan["units"] if unit["kind"]!="model"}
    expected.update(("imported_truth",f"{case}_s{seed}") for case in plan["protocol"]["cases"] for seed in plan["protocol"]["data_seeds"])
    check(len(lookup)==len(value["entries"])==110 and set(lookup)==expected,"Duplicate, missing or unplanned import receipts")
    return lookup

def verify_record(record,path,plan,unit=None,truth_id=None,sources=None):
    path=Path(path)
    check((unit is None)!=(truth_id is None),"Specify exactly one model/reference unit or truth ID")
    if unit is not None and unit["kind"]=="model":
        check(record["identity"]["unit"]==unit,"GPU model unit identity mismatch")
        check(record["identity"]["protocol_sha256"]==plan["protocol_sha256"]
              and record["identity"]["driver_sha256"]==plan["driver_sha256"],"GPU model protocol/driver mismatch")
        check(canonical(record["metadata"])==record["identity"]["model_metadata_sha256"],"GPU metadata hash mismatch")
        proof_path=path.parent/"CHECKPOINT_DEVICE.json"
        proof=read(proof_path,"GPU_checkpoint_device_proof",sources)
        check(proof["source_training_device"]=="cuda" and proof["source_training_dtype"]=="float32","GPU checkpoint lacks CUDA/float32 proof")
        check(proof["GPU_scratch_task_id"]==unit["training_id"] and proof["CPU_model_or_optimizer_imported"] is False,"GPU task/CPU-import proof mismatch")
        check(proof["source_checkpoint_sha256"]==record["metadata"]["checkpoint_sha256"],"GPU checkpoint SHA proof mismatch")
        check(proof["training_study"]==plan["protocol"]["training_output_directory"]
              and proof["forward_cohort"]==plan["cohort"]=="fresh_GPU_only"
              and proof["forward_evaluation_device"]=="cpu","GPU proof cohort/routing mismatch")
        training_path=REPO/"outputs"/plan["protocol"]["training_output_directory"]/"LAUNCHED_MANIFEST.json"
        training=read(training_path,"GPU_training_launched_manifest",sources)
        check(digest(training_path)==plan["upstream_manifest_sha256"]
              and canonical(training)==plan["training_manifest_canonical_sha256"],"GPU training manifest changed")
        tasks={task["id"]:task for task in training["tasks"]}
        check(len(tasks)==len(training["tasks"])==100 and training["required_device"]=="cuda","GPU training task denominator/device mismatch")
        task=tasks[unit["training_id"]]
        check(task["required_device"]=="cuda" and all(task[key]==unit[key] for key in ("case","setting","data_seed","variant")),"GPU training task identity mismatch")
        check(proof["original_initial_state_sha256"]==task["paired_cpu_initial_model_sha256"],"GPU scratch initial-state proof mismatch")
        return
    lookup=imports(plan,sources)
    kind="imported_truth" if truth_id is not None else "imported_reference"
    identifier=truth_id if truth_id is not None else unit["id"]
    entry=lookup[(kind,identifier)]
    wrapper_path=path.parent/"IMPORT.json"
    wrapper=read(wrapper_path,"original_byte_import_wrapper",sources)
    check(digest(wrapper_path)==entry["wrapper_sha256"] and digest(path)==entry["result_sha256"],"Imported wrapper/result hash mismatch")
    check(wrapper=={key:value for key,value in entry.items() if key not in ("relative_directory","wrapper_sha256")},"Import wrapper/index disagree")
    check(wrapper["original_identity_preserved"] is True and wrapper["new_fits"]==wrapper["new_integrations"]==0,"Imported result was recomputed or relabelled")
    check(wrapper["target_study"]==plan["study_id"] and wrapper["kind"]==kind and wrapper["unit_id"]==identifier,"Imported wrapper cohort/unit mismatch")
    expected_directory=("truth/" if truth_id is not None else "runs/")+identifier
    check(entry["relative_directory"]==expected_directory and path.parent.resolve()==(OUTPUT/expected_directory).resolve(),"Imported result path mismatch")
    check(record["identity"]["protocol_sha256"]==entry["source_protocol_sha256"]
          and record["identity"]["driver_sha256"]==entry["source_driver_sha256"]
          and record["fields_sha256"]==entry["fields_sha256"],"Original imported identity/fields SHA mismatch")
    if unit is not None:
        check(record["identity"]["unit"]==unit==entry["unit"],"Imported reference unit mismatch")
    else:
        check(f"{record['identity']['case']}_s{record['identity']['data_seed']}"==truth_id,"Imported truth identity mismatch")
