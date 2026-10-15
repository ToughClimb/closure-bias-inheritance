"""Run the preregistered global polynomial controls, using at most two CPU workers."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
from objective_controls import (STUDY,REPO,DEFAULT_OUTPUT,SHARED_OUTPUT,TERMS,array_hash,
    compress_rows,design_rows,diffusion_minimum,evaluation,file_hash,least_squares,
    load_core,observation_at_noise,positive_diffusion,read_input,write_json)


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def configurations(path):
    manifest=json.loads(Path(path).read_text())
    unique={}
    for task in manifest["tasks"]:
        conf={key:task[key] for key in ("variant","weights","weak_variant","phase")}
        if conf["variant"] in unique and unique[conf["variant"]]!=conf:
            raise ValueError("Configuration changes across inputs")
        unique[conf["variant"]]=conf
    if len(unique)!=34:raise ValueError(f"Expected 34 configurations, got {len(unique)}")
    return unique


def execute_bundle(job):
    out=Path(job["output_root"])
    path=out/job["mode"]/"bundles"/(job["id"]+".json")
    identity=canonical_hash(job)
    if path.exists():
        old=json.loads(path.read_text())
        if old["identity"]!=identity:raise ValueError(f"Existing bundle identity changed: {path}")
        return old["summary"]
    start=time.time()
    core=load_core(out)
    try:
        arrays,meta=read_input(job["input_path"])
        case=core.extra_case(meta["case"])
        if job["mode"]=="shared":
            observed=arrays["u"]
        else:
            observed=observation_at_noise(arrays,job["noise_level"],meta["data_seed"],
                                           job["clipping"]=="original_clipped")
            # Verify the same draw and clipping convention against the shared noise3 input.
            paired_path=Path(job["input_path"]).with_name(f"{meta['case']}_noise3_s{meta['data_seed']}.npz")
            paired,_=read_input(paired_path)
            if not np.array_equal(observation_at_noise(arrays,.03,meta["data_seed"],True),paired["u"]):
                raise ValueError("Shared noise3 reconstruction differs")
        results=[];matrices={}
        for arm in job["oracle_arms"]:
            features=arrays["clean_u"] if arm["features"]=="clean" else observed
            targets=arrays["clean_u"] if arm["targets"]=="clean" else observed
            for weak_variant in sorted({c["weak_variant"] for c in job["configurations"]}):
                terms=compress_rows(design_rows(features,targets,arrays["x"],arrays["t"],
                                               case.value_range,core,weak_variant))
                key=arm["name"]+"__"+weak_variant
                matrices[key]={k:dict(augmented_qr=v.augmented_qr.tolist(),rows=v.rows,
                                     normalization_count=v.normalization_count) for k,v in terms.items()}
                for conf in job["configurations"]:
                    if conf["weak_variant"]!=weak_variant:continue
                    for constraint in ("unconstrained","positive_diffusion"):
                        record=dict(case=meta["case"],data_seed=meta["data_seed"],
                            source_input_id=meta["id"],noise_level=job["noise_level"],
                            clipping=job["clipping"],oracle_arm=arm["name"],oracle=arm["oracle"],
                            constraint=constraint,**conf)
                        try:
                            if constraint=="unconstrained":
                                theta,solver,_=least_squares(terms,conf["weights"])
                                solver["minimum_diffusion"]=diffusion_minimum(theta,case.value_range)
                            else:
                                theta,solver=positive_diffusion(terms,conf["weights"],case.value_range)
                            record.update(status=solver["status"],solver=solver,
                                          evaluation=evaluation(theta,case,observed))
                        except Exception as exc:
                            record.update(status="failed_exception",exception_type=type(exc).__name__,
                                          exception=str(exc),traceback=traceback.format_exc())
                        results.append(record)
        expected=len(job["oracle_arms"])*len(job["configurations"])*2
        if len(results)!=expected:raise AssertionError((len(results),expected))
        counts=dict(Counter(r["status"] for r in results))
        summary=dict(id=job["id"],path=str(path),fits=len(results),statuses=counts,
                     elapsed_seconds=time.time()-start)
        write_json(path,dict(identity=identity,job=job,source_metadata=meta,
                            observed_array_hash=array_hash(observed),
                            clean_array_hash=array_hash(arrays["clean_u"]),
                            matrices=matrices,results=results,summary=summary))
        return summary
    except Exception as exc:
        write_json(path.with_suffix(".FAILED.json"),dict(identity=identity,job=job,
                    exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("shared","oracle"),required=True)
    parser.add_argument("--input-root",type=Path,default=SHARED_OUTPUT)
    parser.add_argument("--weight-manifest",type=Path,
                        default=REPO/"research/reviewer-sensitivity-2026-10-15/MANIFEST.json")
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--workers",type=int,choices=(1,2),default=1)
    parser.add_argument("--cases",default="case_a,case_exp")
    parser.add_argument("--seeds",default="0,1,2,3,4")
    parser.add_argument("--noise-levels",default="0,0.003,0.01,0.02,0.03,0.05")
    parser.add_argument("--variants",default="",help="Optional predeclared subset for staged execution")
    args=parser.parse_args()
    out=args.output_root.resolve();input_root=args.input_root.resolve()
    protocol=json.loads((STUDY/"PROTOCOL.json").read_text())
    inputs=json.loads((input_root/"INPUT_MANIFEST.json").read_text())
    if inputs["count"]!=20:raise ValueError("Expected complete twenty-input manifest")
    all_configs=configurations(args.weight_manifest)
    selected=(protocol["oracle_configurations"] if args.mode=="oracle" else list(all_configs))
    if args.variants:
        requested=args.variants.split(",")
        if not set(requested)<=set(selected):raise ValueError("Undeclared variant selection")
        selected=requested
    confs=[all_configs[name] for name in selected]
    cases=args.cases.split(",");seeds=[int(s) for s in args.seeds.split(",")]
    levels=[float(n) for n in args.noise_levels.split(",")]
    if not set(cases)<=set(protocol["noise_curve"]["cases"]):raise ValueError("Undeclared case")
    if not set(seeds)<=set(protocol["noise_curve"]["seeds"]):raise ValueError("Undeclared seed")
    if not set(levels)<=set(protocol["noise_curve"]["levels"]):raise ValueError("Undeclared noise")
    sources={str(p.relative_to(REPO)):file_hash(p) for p in sorted(STUDY.glob("*.py"))}
    sources[str((STUDY/"PROTOCOL.json").relative_to(REPO))]=file_hash(STUDY/"PROTOCOL.json")
    for p in sorted((REPO/"src").rglob("*.py"))+sorted((REPO/"revision").glob("*.py")):
        sources[str(p.relative_to(REPO))]=file_hash(p)
    implementation=canonical_hash(sources)
    jobs=[]
    for rec in inputs["inputs"]:
        if rec["case"] not in cases or rec["data_seed"] not in seeds:continue
        if args.mode=="oracle" and rec["setting"]!="clean":continue
        path=input_root/rec["file"]
        if file_hash(path)!=rec["file_sha256"]:raise ValueError(f"Input changed: {path}")
        options=([(0. if rec["setting"]=="clean" else .03,"original_clipped")]
                 if args.mode=="shared" else
                 [(level,clip) for level in levels for clip in protocol["noise_curve"]["clipping_arms"]])
        for level,clip in options:
            ident=(rec["id"] if args.mode=="shared" else
                   f"{rec['case']}_s{rec['data_seed']}_noise{level:g}_{clip}")
            arms=protocol["oracle_arms"] if args.mode=="oracle" else protocol["oracle_arms"][:1]
            jobs.append(dict(id=ident,mode=args.mode,input_path=str(path),output_root=str(out),
                input_sha256=rec["file_sha256"],noise_level=level,clipping=clip,
                oracle_arms=arms,configurations=confs,implementation=implementation))
    planned=sum(len(j["oracle_arms"])*len(j["configurations"])*2 for j in jobs)
    run=dict(mode=args.mode,jobs=jobs,expected_fits=planned,implementation=implementation,
        protocol_sha256=file_hash(STUDY/"PROTOCOL.json"),
        input_manifest_sha256=file_hash(input_root/"INPUT_MANIFEST.json"),
        weight_manifest_sha256=file_hash(args.weight_manifest),
        workers=args.workers,command=list(sys.argv),sources=sources,
        selection_rule=protocol["noise_curve"]["configuration_rule"],
        dependent_repeats="Shared fields, weights, clipping and zero-noise oracle arms are paired diagnostics, not independent samples")
    run_path=out/args.mode/"RUN_MANIFEST.json"
    if run_path.exists():
        old=json.loads(run_path.read_text())
        if old["jobs"]!=jobs:raise ValueError("Existing run differs; use a separate output directory")
    else:write_json(run_path,run)
    environment=dict(python=sys.version,platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in ("numpy","scipy","torch")},
        git_status=subprocess.run(["git","status","--short"],cwd=REPO,capture_output=True,text=True,check=True).stdout,
        started_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/args.mode/"ENVIRONMENT.json",environment)
    start=time.time();summaries=[]
    if args.workers==1:
        for job in jobs:
            s=execute_bundle(job);summaries.append(s)
            print(json.dumps(s),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            futures={pool.submit(execute_bundle,j):j["id"] for j in jobs}
            for future in as_completed(futures):
                s=future.result();summaries.append(s)
                print(json.dumps(s),flush=True)
    counts=Counter()
    for s in summaries:counts.update(s["statuses"])
    actual=sum(s["fits"] for s in summaries)
    if actual!=planned:raise AssertionError((actual,planned))
    passed=actual==counts.get("passed",0)
    completion=dict(status="passed" if passed else "finished_with_failures",mode=args.mode,
                    expected_fits=planned,actual_fits=actual,status_counts=dict(counts),
                    bundle_count=len(summaries),elapsed_seconds=time.time()-start,
                    summaries=sorted(summaries,key=lambda s:s["id"]))
    write_json(out/args.mode/"COMPLETION.json",completion)
    print(json.dumps({k:v for k,v in completion.items() if k!="summaries"},indent=2))
    if not passed:raise SystemExit(2)


if __name__=="__main__":main()
