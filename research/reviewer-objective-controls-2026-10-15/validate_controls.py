"""CPU-only numerical identities and constrained-solver pilot, with saved evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from objective_controls import (DEFAULT_OUTPUT, SHARED_OUTPUT, STUDY, TERMS, DEFAULT_WEIGHTS,
    Term, compress_rows, design_rows, diffusion_minimum, evaluation, file_hash,
    least_squares, load_core, observation_at_noise, positive_diffusion, read_input, write_json)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=SHARED_OUTPUT)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    out=args.output_root.resolve()/"pilot"
    core=load_core(args.output_root)
    torch=core.torch
    start=time.time()
    manifest=json.loads((args.input_root/"INPUT_MANIFEST.json").read_text())
    assert manifest["count"]==20 and len(manifest["inputs"])==20
    checked=[]
    for record in manifest["inputs"]:
        arrays,meta=read_input(args.input_root/record["file"])
        assert meta["id"]==record["id"] and meta["file_sha256"]==record["file_sha256"]
        level=0. if meta["setting"]=="clean" else .03
        reproduced=observation_at_noise(arrays,level,meta["data_seed"],True)
        assert np.array_equal(reproduced,arrays["u"]),meta["id"]
        checked.append(dict(id=meta["id"],file_sha256=meta["file_sha256"],noise_reproduction_max=0.))

    class Polynomial(torch.nn.Module):
        def __init__(self,theta):
            super().__init__();self.theta=torch.nn.Parameter(torch.tensor(theta,dtype=torch.float64))
        def diffusion(self,u):return sum(self.theta[j]*u**j for j in range(3))
        def reaction(self,u):return sum(self.theta[3+j]*u**j for j in range(4))
        def smoothness_penalty(self,u):
            return sum(j*self.theta[j]*u**(j-1) for j in range(1,3)).square().mean()+sum(
                j*self.theta[3+j]*u**(j-1) for j in range(1,4)).square().mean()

    diagnostics=[];pilot=[]
    rng=np.random.default_rng(20261015)
    for case_name in ("case_a","case_exp"):
        case=core.extra_case(case_name)
        for setting in ("clean","noise3"):
            arrays,meta=read_input(args.input_root/"inputs"/f"{case_name}_{setting}_s0.npz")
            u=arrays["u"];x=arrays["x"];t=arrays["t"]
            for variant in ("space_weak","space_time_weak"):
                rows=design_rows(u,u,x,t,case.value_range,core,variant)
                terms=compress_rows(rows)
                theta=rng.normal(size=7)*np.array([.02,.03,.01,.1,.3,.3,.1])
                model=Polynomial(theta)
                tensor=lambda a:torch.tensor(a,dtype=torch.float64)
                phi,dphi=core.make_resolved_test_functions_1d(x,4)
                temporal=None
                if variant=="space_time_weak":
                    psi,dpsi=core.make_time_test_functions_1d(t);w=np.ones(len(t));w[[0,-1]]=.5
                    temporal=tuple(tensor(a) for a in (psi,dpsi,w))
                direct=core.objective_terms(model,tensor(u),float(x[1]-x[0]),float(t[1]-t[0]),
                    tensor(phi),tensor(dphi),torch.linspace(*case.value_range,256,dtype=torch.float64),
                    tensor([case.value_range[0]]),variant,temporal)
                for key in TERMS:
                    a,b,n=rows[key]
                    matrix=float(np.mean((a@theta-b)**2)*len(b)/n)
                    qr=terms[key].loss(theta)
                    observed=float(direct[key].detach())
                    rel=abs(matrix-observed)/max(1.,abs(observed))
                    qr_rel=abs(matrix-qr)/max(1.,abs(matrix))
                    grad=torch.autograd.grad(direct[key],model.theta,retain_graph=True)[0].detach().numpy()
                    expected=2*a.T@(a@theta-b)/n
                    grad_rel=float(np.linalg.norm(grad-expected)/max(1.,np.linalg.norm(expected)))
                    step=1e-5
                    finite=[]
                    for j in range(7):
                        delta=np.zeros(7);delta[j]=step
                        finite.append((terms[key].loss(theta+delta)-terms[key].loss(theta-delta))/(2*step))
                    finite_rel=float(np.linalg.norm(np.asarray(finite)-expected)/max(1.,np.linalg.norm(expected)))
                    assert max(rel,qr_rel,grad_rel)<1e-10,(meta["id"],variant,key,rel,grad_rel)
                    assert finite_rel<1e-6,(meta["id"],variant,key,finite_rel)
                    diagnostics.append(dict(input=meta["id"],variant=variant,term=key,
                        loss_relative_difference=rel,qr_relative_difference=qr_rel,
                        autograd_relative_difference=grad_rel,finite_difference_relative_difference=finite_rel))
                weights=DEFAULT_WEIGHTS.copy()
                if variant=="space_time_weak":weights["mass"]=weights["strong"]=0.
                theta_ls,report,_=least_squares(terms,weights)
                full_a=np.vstack([np.sqrt(weights[k]/rows[k][2])*rows[k][0] for k in TERMS if weights[k]])
                full_b=np.concatenate([np.sqrt(weights[k]/rows[k][2])*rows[k][1] for k in TERMS if weights[k]])
                norms=np.linalg.norm(full_a,axis=0)
                direct_theta=np.linalg.lstsq(full_a/norms,full_b,rcond=None)[0]/norms
                theta_difference=float(np.linalg.norm(direct_theta-theta_ls)/max(1.,np.linalg.norm(theta_ls)))
                assert theta_difference<1e-8,theta_difference
                constrained,qp=positive_diffusion(terms,weights,case.value_range)
                assert report["rank"]==7 and report["stationarity_relative"]<1e-9
                assert qp["status"]=="passed",qp
                pilot.append(dict(input=meta["id"],variant=variant,weights=weights,
                                  raw_vs_qr_svd_coefficient_relative_difference=theta_difference,
                                  unconstrained=report,constrained=qp,
                                  unconstrained_evaluation=evaluation(theta_ls,case,u),
                                  constrained_evaluation=evaluation(constrained,case,u)))

    # A known convex optimum and a deliberately between-grid negative quadratic.
    unit_checks=[]
    for name,target in [("known_boundary_optimum",np.array([-1.,0.,0.,.2,-.3,.4,-.1])),
                        ("interior_cut",np.array([.515**2+1e-4-1e-5,-1.03,1.,0.,0.,0.,0.]))]:
        terms={k:Term(np.column_stack([np.eye(7),-target]),7,1) for k in TERMS}
        weights={k:float(k=="weak") for k in TERMS}
        theta,qp=positive_diffusion(terms,weights,(0.,1.))
        assert qp["status"]=="passed",qp
        if name=="known_boundary_optimum":
            expected=target.copy();expected[0]=1e-4+1e-10
            assert np.max(np.abs(theta-expected))<1e-7,(theta,expected)
        else:
            assert len(qp["attempts"])>=2,qp
        unit_checks.append(dict(name=name,result=qp))
    result=dict(status="passed",input_manifest_sha256=file_hash(args.input_root/"INPUT_MANIFEST.json"),
                input_identity_checks=checked,identities=diagnostics,pilot_fits=pilot,
                unit_checks=unit_checks,elapsed_seconds=time.time()-start,command=sys.argv,
                sources={p.name:file_hash(p) for p in STUDY.glob("*.py")},
                maximum_identity_error=max(max(d["loss_relative_difference"],d["qr_relative_difference"],
                    d["autograd_relative_difference"]) for d in diagnostics))
    write_json(out/"VALIDATION.json",result)
    print(json.dumps({k:result[k] for k in ("status","elapsed_seconds","maximum_identity_error")},indent=2))


if __name__=="__main__":main()
