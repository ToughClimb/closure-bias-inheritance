"""Compare all twenty default global fits with actual float32 loss arithmetic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
from objective_controls import (DEFAULT_OUTPUT,SHARED_OUTPUT,STUDY,DEFAULT_WEIGHTS,TERMS,
    compress_rows,design_rows,evaluation,file_hash,least_squares,load_core,
    positive_diffusion,read_input,write_json)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=SHARED_OUTPUT)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();core=load_core(args.output_root);torch=core.torch
    manifest=json.loads((args.input_root/"INPUT_MANIFEST.json").read_text())
    records=[];start=time.time()

    class Poly(torch.nn.Module):
        def __init__(self,theta,dtype):
            super().__init__();self.theta=torch.nn.Parameter(torch.tensor(theta,dtype=dtype))
        def diffusion(self,u):return sum(self.theta[j]*u**j for j in range(3))
        def reaction(self,u):return sum(self.theta[3+j]*u**j for j in range(4))
        def smoothness_penalty(self,u):
            return sum(j*self.theta[j]*u**(j-1) for j in range(1,3)).square().mean()+sum(
                j*self.theta[3+j]*u**(j-1) for j in range(1,4)).square().mean()

    for rec in manifest["inputs"]:
        arrays,meta=read_input(args.input_root/rec["file"])
        case=core.extra_case(meta["case"]);u=arrays["u"];x=arrays["x"];t=arrays["t"]
        terms=compress_rows(design_rows(u,u,x,t,case.value_range,core))
        # This second estimator rounds only observed fields to float32 before
        # double-precision fitting. It is distinct from full float32 arithmetic.
        quantized=u.astype(np.float32).astype(np.float64)
        qterms=compress_rows(design_rows(quantized,quantized,x,t,case.value_range,core))
        phi,dphi=core.make_resolved_test_functions_1d(x,4)
        for constraint in ("unconstrained","positive_diffusion"):
            if constraint=="unconstrained":
                theta,solver,_=least_squares(terms,DEFAULT_WEIGHTS)
                qtheta,qsolver,_=least_squares(qterms,DEFAULT_WEIGHTS)
            else:
                theta,solver=positive_diffusion(terms,DEFAULT_WEIGHTS,case.value_range)
                qtheta,qsolver=positive_diffusion(qterms,DEFAULT_WEIGHTS,case.value_range)
            values={}
            for name,dtype in (("torch64",torch.float64),("torch32",torch.float32)):
                tensor=lambda a:torch.tensor(a,dtype=dtype)
                model=Poly(theta,dtype)
                losses=core.objective_terms(model,tensor(u),float(x[1]-x[0]),float(t[1]-t[0]),
                    tensor(phi),tensor(dphi),torch.linspace(*case.value_range,256,dtype=dtype),
                    tensor([case.value_range[0]]))
                values[name]={k:float(v.detach()) for k,v in losses.items()}
                values[name]["total"]=sum(DEFAULT_WEIGHTS[k]*values[name][k] for k in TERMS)
            reference={k:terms[k].loss(theta) for k in TERMS}
            reference["total"]=sum(DEFAULT_WEIGHTS[k]*reference[k] for k in TERMS)
            differences={name:{k:dict(absolute=values[name][k]-reference[k],
                            relative=abs(values[name][k]-reference[k])/max(1e-12,abs(reference[k])))
                              for k in reference} for name in values}
            ev=evaluation(theta,case,u);qev=evaluation(qtheta,case,u)
            item=dict(input=meta["id"],constraint=constraint,status=solver["status"],
                quantized_fit_status=qsolver["status"],native_float64_objective=reference,
                direct_evaluations=values,differences=differences,
                coefficients=theta.tolist(),quantized_input_coefficients=qtheta.tolist(),
                quantized_input_coefficient_relative_difference=float(np.linalg.norm(qtheta-theta)/max(1e-12,np.linalg.norm(theta))),
                native_evaluation=ev,quantized_input_evaluation=qev,
                quantization_E_D_change=qev["E_D"]-ev["E_D"],quantization_E_R_change=qev["E_R"]-ev["E_R"])
            if max(d["relative"] for d in differences["torch64"].values())>1e-7:
                raise AssertionError(f"Float64 identity failed: {meta['id']}")
            if not all(np.isfinite(v) for v in values["torch32"].values()):
                raise FloatingPointError("Nonfinite float32 objective")
            records.append(item)
    summary=dict(status="passed" if all(r["status"]==r["quantized_fit_status"]=="passed" for r in records) else "completed_with_solver_failures",
        count=len(records),elapsed_seconds=time.time()-start,
        max_total_float32_relative_difference=max(r["differences"]["torch32"]["total"]["relative"] for r in records),
        max_total_float32_absolute_difference=max(abs(r["differences"]["torch32"]["total"]["absolute"]) for r in records),
        max_quantized_input_coefficient_relative_difference=max(r["quantized_input_coefficient_relative_difference"] for r in records),
        max_absolute_E_D_change=max(abs(r["quantization_E_D_change"]) for r in records),
        max_absolute_E_R_change=max(abs(r["quantization_E_R_change"]) for r in records))
    write_json(args.output_root/"pilot/PRECISION_AUDIT.json",dict(summary=summary,records=records,
        note="Float64 coefficient reference is formula-matched, not bitwise equivalent to float32 neural arithmetic. Float32 direct evaluation uses float32 coefficients, observations, tests and support, following revision_core. Quantized-input refitting only rounds observations before double-precision least squares and is a separate diagnostic.",
        sources={p.name:file_hash(p) for p in STUDY.glob("*.py")}))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
