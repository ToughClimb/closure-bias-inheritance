"""Archive the completed CPU150 scope without changing its500-state source plan."""
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

STUDY=Path(__file__).resolve().parent;REPO=STUDY.parents[1]
OUTPUT=REPO/"outputs"/STUDY.name
SOURCE_STUDY=REPO/"research/reviewer-forward-2026-10-15-v2"
SOURCE=REPO/"outputs/reviewer-forward-2026-10-15-v2"

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def main():
    plan=read(SOURCE_STUDY/"MANIFEST.json")
    assert sha(SOURCE_STUDY/"run_forward.py")==plan["driver_sha256"]
    selected=[unit for unit in plan["units"] if unit["kind"]!="model" or unit["epoch"]==150]
    excluded=[unit for unit in plan["units"] if unit not in selected]
    assert len(selected)==300 and len(excluded)==200 and all(unit["kind"]=="model" and unit["epoch"]==1500 for unit in excluded)
    assert not any((SOURCE/"runs"/unit["id"]/"result.json").exists() for unit in excluded)
    result_hashes={};export_hashes={};kind=Counter();primary=Counter();model_refinements=0
    for unit in selected:
        directory=SOURCE/"runs"/unit["id"];record=read(directory/"result.json")
        assert record["identity"]["unit"]==unit and record["identity"]["protocol_sha256"]==plan["protocol_sha256"]
        assert record["identity"]["driver_sha256"]==plan["driver_sha256"]
        assert canonical(record["metadata"])==record["identity"]["model_metadata_sha256"]
        assert sha(directory/"fields.npz")==record["fields_sha256"]
        result_hashes[unit["id"]]=sha(directory/"result.json");kind[unit["kind"]]+=1
        for entry in record["entries"]:
            primary[entry["info"]["status"]]+=1
            if unit["refinement"]:
                assert entry["refined_info"]["status"]=="complete";model_refinements+=1
        if unit["kind"]=="model" and unit["method"]=="native":
            identifier=f"{unit['training_id']}_e150";npz=SOURCE/"native_exports"/(identifier+".npz");sidecar=read(npz.with_suffix(".json"))
            assert sha(npz)==sidecar["export_sha256"]
            assert sidecar["identity"]["model_metadata_sha256"]==record["identity"]["model_metadata_sha256"]
            assert sidecar["identity"]["checkpoint_sha256"]==record["metadata"]["checkpoint_sha256"]
            assert sidecar["source_forward_result_sha256"]==result_hashes[unit["id"]]
            assert sidecar["model_verification"]==record["metadata"]["model_verification"]
            assert sidecar["reload_arrays_bitwise_equal"] and sidecar["reload_evaluation_bitwise_equal"]
            export_hashes[identifier]=dict(npz_sha256=sha(npz),sidecar_sha256=sha(npz.with_suffix(".json")))
    assert dict(kind)=={"model":200,"baseline":60,"global_reference":40} and len(export_hashes)==100 and model_refinements==64
    array=read(SOURCE/"reports/ARRAY_AUDIT.json")
    assert array["status"]=="passed" and array["evaluated"]["states"]==300 and array["evaluated"]["metric_rows"]==3600
    assert array["validation_maxima"]["endpoint_metric_recomputation_absolute_difference"]==0
    assert array["validation_maxima"]["space_time_metric_recomputation_absolute_difference"]==0
    norms=read(OUTPUT/"FUNCTION_NORM_SENSITIVITY_AUDIT.json")
    assert norms["complete"] and norms["unique_checkpoint_pairs"]==100
    report=dict(status="passed",complete=True,cohort="CPU150_only",snapshot_utc=datetime.now(timezone.utc).isoformat(),
        expected_states=300,evaluated_states=300,states_by_kind=dict(kind),expected_unique_native_exports=100,verified_native_exports=100,
        expected_metric_rows=3600,evaluated_metric_rows=3600,total_integration_records=1344,primary_integration_status=dict(primary),
        model_refinement_integrations=64,truth_primary_integrations=40,truth_refinement_integrations=40,
        source_protocol_planned_states=500,source_protocol_complete=False,excluded_unexecuted_CPU1500_states=200,
        scope_change_authority="User required CUDA neural computation and host-memory protection; interrupted CPU-long states are excluded without modifying the original500-state protocol or selecting on errors.",
        CPU_GPU_states_mixed=False,source_results_unchanged=True,new_fits_or_integrations=0,
        source_manifest_sha256=sha(SOURCE_STUDY/"MANIFEST.json"),source_array_audit_sha256=sha(SOURCE/"reports/ARRAY_AUDIT.json"),
        source_norm_audit_sha256=sha(OUTPUT/"FUNCTION_NORM_SENSITIVITY_AUDIT.json"),script_sha256=sha(Path(__file__)),
        source_result_sha256=result_hashes,native_export_sha256=export_hashes,
        interpretation="The completed300-state CPU150 scope is separate from the fresh500-state GPU forward cohort. Shared references/truth are reused evidence. The original V2 source plan remains incomplete at500 and its remaining200 CPU1500 states were never executed. Norm sensitivity of the original illustrative example is retained explicitly.")
    (OUTPUT/"CPU150_COHORT_STATUS.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:report[key] for key in ["status","complete","cohort","evaluated_states","verified_native_exports","evaluated_metric_rows","total_integration_records","primary_integration_status"]},indent=2))

if __name__=="__main__":main()
