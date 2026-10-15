"""GPU-cohort identity, original-byte reference imports, and resource guards.

No fitting, training, or PDE calculation lives in this module.
"""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from protocol import PROTOCOL

STUDY=Path(__file__).resolve().parent
REPO=STUDY.parents[1]
OUTPUT=REPO/"outputs"/STUDY.name
SOURCE_STUDY=REPO/"research/reviewer-forward-2026-10-15-v2"
SOURCE_OUTPUT=REPO/"outputs/reviewer-forward-2026-10-15-v2"
PHYSICS_FUNCTIONS=["strict_integrate","initial_fields","scores","compare_solutions","expression_poles","polynomial","polynomial_minimum","global_reference","run_truth","load_models","run_unit"]
IO_PROTOCOL_KEYS={"study_id","classification","upstream_study","training_output_directory","training_manifest_directory","training_version_rule","baseline_rule","global_reference_source","formal_launch"}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temporary.replace(path)

def read(path):
    return json.loads(Path(path).read_text())

def physics_equivalence():
    records=[]
    parsed=[]
    for path in (SOURCE_STUDY/"run_forward.py",STUDY/"run_forward.py"):
        source=path.read_text();tree=ast.parse(source)
        parsed.append({node.name:(ast.get_source_segment(source,node),ast.dump(node,include_attributes=False))
                       for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))})
    for name in PHYSICS_FUNCTIONS+["ForwardGuard"]:
        a,b=parsed[0][name],parsed[1][name]
        assert a==b,"Numerical source changed: "+name
        records.append(dict(name=name,source_bytes_sha256=hashlib.sha256(a[0].encode()).hexdigest(),
            ast_sha256=hashlib.sha256(a[1].encode()).hexdigest(),byte_identical=True,ast_identical=True))
    original=read(SOURCE_STUDY/"PROTOCOL.json")
    for key,value in original.items():
        if key not in IO_PROTOCOL_KEYS:
            assert PROTOCOL[key]==value,"Scientific protocol changed: "+key
    common={key:value for key,value in original.items() if key not in IO_PROTOCOL_KEYS}
    report=dict(status="passed",all_11_function_bodies_byte_and_ast_identical=True,
        source_driver_sha256=digest(SOURCE_STUDY/"run_forward.py"),target_driver_sha256=digest(STUDY/"run_forward.py"),
        common_scientific_protocol_sha256=canonical(common),functions=records,
        metadata_only_protocol_differences=sorted(key for key in original if original[key]!=PROTOCOL.get(key)),
        changed_physics_fields=[],source_outputs_preserved=True)
    return report

def copy_original(source,destination):
    source,destination=Path(source),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        assert digest(destination)==digest(source),"Existing imported bytes differ: "+str(destination)
    else:
        temporary=destination.with_suffix(destination.suffix+".tmp")
        shutil.copy2(source,temporary);temporary.replace(destination)
    assert digest(destination)==digest(source)

def import_shared():
    """Copy reference/truth bytes and add wrappers; never rewrite source identity."""
    index_path=OUTPUT/"SHARED_IMPORTS.json"
    if index_path.exists():
        verify_shared_imports();return read(index_path)
    equivalence=physics_equivalence();source_plan=read(SOURCE_STUDY/"MANIFEST.json")
    assert source_plan["driver_sha256"]==digest(SOURCE_STUDY/"run_forward.py")
    input_root=REPO/"outputs"/PROTOCOL["upstream_study"]
    assert digest(input_root/"INPUT_MANIFEST.json")==source_plan["input_manifest_sha256"]
    inputs=read(input_root/"INPUT_MANIFEST.json")
    for item in inputs["inputs"]:
        assert digest(input_root/item["file"])==item["file_sha256"]
    entries=[]
    source_units=[unit for unit in source_plan["units"] if unit["kind"]!="model"]
    for unit in source_units:
        source=SOURCE_OUTPUT/"runs"/unit["id"]
        result=read(source/"result.json")
        assert result["identity"]["unit"]==unit
        assert result["identity"]["protocol_sha256"]==source_plan["protocol_sha256"]
        assert result["identity"]["driver_sha256"]==source_plan["driver_sha256"]
        assert canonical(result["metadata"])==result["identity"]["model_metadata_sha256"]
        assert digest(source/"fields.npz")==result["fields_sha256"]
        destination=OUTPUT/"runs"/unit["id"]
        for filename in ("result.json","fields.npz"):
            copy_original(source/filename,destination/filename)
        wrapper=dict(kind="imported_reference",target_study=STUDY.name,unit_id=unit["id"],unit=unit,
            source_study=SOURCE_STUDY.name,source_directory=str(source.relative_to(REPO)),
            source_protocol_sha256=source_plan["protocol_sha256"],source_driver_sha256=source_plan["driver_sha256"],
            result_sha256=digest(source/"result.json"),fields_sha256=digest(source/"fields.npz"),
            original_identity_preserved=True,new_fits=0,new_integrations=0)
        write(destination/"IMPORT.json",wrapper)
        entries.append(dict(wrapper,relative_directory=str(destination.relative_to(OUTPUT)),wrapper_sha256=digest(destination/"IMPORT.json")))
    for case in PROTOCOL["cases"]:
        for seed in PROTOCOL["data_seeds"]:
            identifier=f"{case}_s{seed}";source=SOURCE_OUTPUT/"truth"/identifier;result=read(source/"result.json")
            assert result["identity"]["protocol_sha256"]==source_plan["protocol_sha256"]
            assert result["identity"]["driver_sha256"]==source_plan["driver_sha256"]
            assert result["identity"]["case"]==case and result["identity"]["data_seed"]==seed
            assert digest(source/"fields.npz")==result["fields_sha256"]
            destination=OUTPUT/"truth"/identifier
            for filename in ("result.json","fields.npz"):
                copy_original(source/filename,destination/filename)
            wrapper=dict(kind="imported_truth",target_study=STUDY.name,unit_id=identifier,
                source_study=SOURCE_STUDY.name,source_directory=str(source.relative_to(REPO)),
                source_protocol_sha256=source_plan["protocol_sha256"],source_driver_sha256=source_plan["driver_sha256"],
                result_sha256=digest(source/"result.json"),fields_sha256=digest(source/"fields.npz"),
                original_identity_preserved=True,new_fits=0,new_integrations=0)
            write(destination/"IMPORT.json",wrapper)
            entries.append(dict(wrapper,relative_directory=str(destination.relative_to(OUTPUT)),wrapper_sha256=digest(destination/"IMPORT.json")))
    assert len(source_units)==100 and len(entries)==110
    index=dict(status="passed",complete=True,target_study=STUDY.name,source_study=SOURCE_STUDY.name,
        source_manifest_sha256=digest(SOURCE_STUDY/"MANIFEST.json"),input_manifest_sha256=source_plan["input_manifest_sha256"],
        original_result_and_fields_bytes_preserved=True,original_metadata_source_sha_preserved=True,
        reference_states=100,direct_reference_fits_reused=60,global_reference_solutions_reused=40,
        reference_primary_integrations_reused=400,truth_primary_integrations_reused=40,truth_refinement_integrations_reused=40,
        total_integration_records_reused=480,new_fits=0,new_integrations=0,physics_equivalence=equivalence,entries=entries)
    write(index_path,index);verify_shared_imports();return index

def verify_shared_imports():
    index=read(OUTPUT/"SHARED_IMPORTS.json")
    assert index["complete"] and len(index["entries"])==110
    for entry in index["entries"]:
        directory=OUTPUT/entry["relative_directory"]
        assert digest(directory/"IMPORT.json")==entry["wrapper_sha256"]
        assert digest(directory/"result.json")==entry["result_sha256"]
        assert digest(directory/"fields.npz")==entry["fields_sha256"]
    return index

def build_plan(driver_path,freeze=False):
    training_root=REPO/"outputs"/PROTOCOL["training_output_directory"]
    research_path=REPO/"research"/PROTOCOL["training_manifest_directory"]/"MANIFEST.json"
    launched_path=training_root/"LAUNCHED_MANIFEST.json"
    upstream=read(research_path)
    assert upstream["task_count"]==100 and len(upstream["tasks"])==100 and upstream["required_device"]=="cuda"
    assert all(task["longer"] and task["id"].startswith("gpu_") and task["required_device"]=="cuda" for task in upstream["tasks"])
    if freeze:
        assert launched_path.exists(),"GPU launched manifest not yet available"
        assert canonical(read(launched_path))==canonical(upstream),"GPU launched/research manifests differ"
    imports=verify_shared_imports();source_plan=read(SOURCE_STUDY/"MANIFEST.json")
    units=[]
    for task in upstream["tasks"]:
        for epoch in PROTOCOL["training_checkpoints"]:
            for method in PROTOCOL["methods"]:
                units.append(dict(id=f"{task['id']}__e{epoch}__{method}",kind="model",training_id=task["id"],
                    case=task["case"],setting=task["setting"],data_seed=task["data_seed"],variant=task["variant"],epoch=epoch,method=method,
                    refinement=task["data_seed"]==0 and task["variant"] in ["default","drop_strong"]))
    units.extend(unit for unit in source_plan["units"] if unit["kind"]!="model")
    assert len(units)==500 and len({unit["id"] for unit in units})==500
    record=dict(study_id=STUDY.name,cohort="fresh_GPU_only",design_only=not freeze,formal_execution_eligible=freeze,
        protocol=PROTOCOL,protocol_sha256=canonical(PROTOCOL),driver_sha256=digest(driver_path),protocol_source_sha256=digest(STUDY/"protocol.py"),
        cohort_io_sha256=digest(__file__),
        upstream_manifest_sha256=digest(launched_path) if freeze else None,
        training_manifest_path=str(launched_path),training_research_manifest_path=str(research_path),
        training_research_manifest_sha256=digest(research_path),training_manifest_canonical_sha256=canonical(upstream),training_manifest_json_equivalent=True,
        input_manifest_sha256=source_plan["input_manifest_sha256"],input_root=str(REPO/"outputs"/PROTOCOL["upstream_study"]),training_root=str(training_root),
        shared_imports_sha256=digest(OUTPUT/"SHARED_IMPORTS.json"),shared_imported_states=100,shared_truth_units=10,
        global_bundle_sha256=source_plan["global_bundle_sha256"],model_units=400,baseline_units=60,global_reference_units=40,
        unit_count=500,units=units,primary_model_integrations=1600,primary_baseline_integrations=240,primary_global_reference_integrations=160,
        unique_truth_integrations=40,truth_refinement_integrations=40,model_refinement_integrations=128,total_declared_integrations=2208,
        new_integrations_to_execute=1728,existing_integration_records_reused=480,new_reference_fits=0,
        formal_execution_started=False,resource_policy_sha256=digest(STUDY/"RESOURCE_POLICY.json"),
        lineage_rule="CPU150 states are excluded. Imported100references and80truth integrations retain original V2 bytes/identities; they are reused evidence, never new independent replicates.")
    write(STUDY/"PROTOCOL.json",PROTOCOL)
    destination=STUDY/("MANIFEST.json" if freeze else "MANIFEST_DRAFT.json")
    if freeze and destination.exists():
        assert read(destination)==record,"Frozen GPU forward manifest already differs"
    else:
        write(destination,record)
    write(STUDY/"PHYSICS_EQUIVALENCE.json",physics_equivalence())
    return record

def read_frozen_plan(driver_path):
    record=read(STUDY/"MANIFEST.json")
    assert record["formal_execution_eligible"] and not record["design_only"]
    research_path=REPO/"research"/PROTOCOL["training_manifest_directory"]/"MANIFEST.json"
    launched_path=REPO/"outputs"/PROTOCOL["training_output_directory"]/"LAUNCHED_MANIFEST.json"
    expected={"driver_sha256":digest(driver_path),"protocol_source_sha256":digest(STUDY/"protocol.py"),
        "cohort_io_sha256":digest(__file__),
        "protocol_sha256":canonical(PROTOCOL),"upstream_manifest_sha256":digest(launched_path),
        "training_research_manifest_sha256":digest(research_path),
        "input_manifest_sha256":digest(REPO/"outputs"/PROTOCOL["upstream_study"]/"INPUT_MANIFEST.json"),
        "shared_imports_sha256":digest(OUTPUT/"SHARED_IMPORTS.json"),"resource_policy_sha256":digest(STUDY/"RESOURCE_POLICY.json")}
    for key,value in expected.items():
        assert record[key]==value,"Frozen GPU forward identity changed: "+key
    assert canonical(read(launched_path))==canonical(read(research_path))
    return record

def load_cohort_result(unit,plan):
    directory=OUTPUT/"runs"/unit["id"];result=read(directory/"result.json")
    assert result["identity"]["unit"]==unit
    if unit["kind"]=="model":
        assert result["identity"]["protocol_sha256"]==plan["protocol_sha256"]
        assert result["identity"]["driver_sha256"]==plan["driver_sha256"]
    else:
        index=read(OUTPUT/"SHARED_IMPORTS.json")
        entry=next(row for row in index["entries"] if row["kind"]=="imported_reference" and row["unit_id"]==unit["id"])
        assert digest(directory/"result.json")==entry["result_sha256"]
        assert digest(directory/"IMPORT.json")==entry["wrapper_sha256"]
        assert result["identity"]["protocol_sha256"]==entry["source_protocol_sha256"]
        assert result["identity"]["driver_sha256"]==entry["source_driver_sha256"]
    return result

def require_resource_limits():
    row=next(row for row in Path("/proc/self/cgroup").read_text().splitlines() if row.startswith("0::"))
    cgroup=Path("/sys/fs/cgroup")/row.split("::",1)[1].lstrip("/")
    limit=(cgroup/"memory.max").read_text().strip()
    assert limit!="max" and int(limit)<=2*1024**3,"Formal CPU forward requires MemoryMax<=2G"
    available=int(next(row.split()[1] for row in Path("/proc/meminfo").read_text().splitlines() if row.startswith("MemAvailable:")))*1024
    assert available>=12*1024**3,"Host MemAvailable below12GiB launch threshold"
    return dict(cgroup_memory_max=int(limit),host_available_bytes=available,cpu_workers_max=2,BLAS_threads=1)
