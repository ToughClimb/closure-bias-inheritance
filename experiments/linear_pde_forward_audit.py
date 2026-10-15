"""Declared independent forward audit of saved linear-PDE neural operators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import shutil
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

# Support both direct script execution and python -m from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.linear_pde_extension import (
    LocalOperator, SYSTEMS, fields, put_csv, put_json, rollout_metrics, sha,
    spectral_features,
)
from experiments.linear_pde_report import read_cases, flexible_csv


def numpy_operator_with_gradient(model,xs,ys):
    s={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
    W=s['linear.weight'][0];W1,b1=s['hidden1.weight'],s['hidden1.bias']
    W2,b2=s['hidden2.weight'],s['hidden2.bias'];W3=s['output.weight'][0]
    origin=np.tanh(np.tanh(b1)@W2.T+b2)@W3
    def evaluate(X):
        Z=np.asarray(X)/xs;h1=np.tanh(Z@W1.T+b1);h2=np.tanh(h1@W2.T+b2)
        value=ys*(Z@W+h2@W3-origin)
        grad=ys*(W+(((1-h2*h2)*W3)@W2*(1-h1*h1))@W1)/xs
        return value,grad
    return evaluate


def fourier_feature_matrices(n,unknowns):
    identity=np.eye(n);k=2*np.pi*np.fft.fftfreq(n,d=1/n)
    basis=np.fft.fft(identity,axis=0)
    matrices=dict(D=np.fft.ifft(-k[:,None]**2*basis,axis=0).real,
                  c=-np.fft.ifft(1j*k[:,None]*basis,axis=0).real,a=identity)
    return [matrices[name] for name in unknowns]


def select_representatives(cases):
    selected=[]
    for system in SYSTEMS:
        for noise in [0.,.01]:
            failed=[];unresolved=[]
            for case in cases:
                if case['system']!=system or case['noise']!=noise:continue
                for state in case['states']:
                    for ic,check in state['rollouts']['teacher']['grid_checks'].items():
                        if check['grid_agreement_pass']:continue
                        record=dict(system=system,noise=noise,case=case['case'],run_id=state['run_id'],initial_condition=ic,
                            old_coarse_fine_difference=check['coarse_fine_max_difference'],checkpoint_sha256=state['checkpoint_sha256'])
                        (failed if check['coarse_fine_max_difference'] is None else unresolved).append(record)
            if failed:
                choice=sorted(failed,key=lambda r:(r['run_id'],r['initial_condition']))[0]
                choice['selection']='Lexicographically first runID/initial-condition with an original failed integration'
            elif unresolved:
                choice=sorted(unresolved,key=lambda r:(-r['old_coarse_fine_difference'],r['run_id'],r['initial_condition']))[0]
                choice['selection']='Largest original finite grid discrepancy; lexical tie break'
            else:continue
            selected.append(choice)
    return selected


def declare(output):
    target=output/'forward_audit';target.mkdir(exist_ok=True)
    cases=read_cases(output)
    record=dict(source_sha256=sha(__file__),parent_protocol_sha256=sha(output/'PROTOCOL.json'),
        rationale='Declared after original grid checks found unresolved neural rollouts; audit existing fixed teachers, not retraining or selection of favourable results',
        principal_derivative=dict(coverage='Every diffusiveteacher and every declared initial condition: 90 states ×3 ICs=270; initial nx64 and every available saved trajectory at nx32/nx64',
            quantity='dN/d(uxx) at actual initial/saved-state Fourier jets',negative_threshold=-1e-10,
            interpretation='A negative sample detects a local violation of forward-parabolic monotonicity. Positive samples are not a global or continuous-trajectory certificate',
            independent_check='Analytic network input Jacobian agrees with Torch autograd on every initial field'),
        representatives=select_representatives(cases),
        forward_solvers=[dict(method='RK45',grid=128),dict(method='RK45',grid=256),dict(method='Radau',grid=128)],
        tolerances=dict(rtol=1e-9,atol=1e-11),rhs_budget=30000,wall_seconds_per_solve=45.,absolute_intermediate_state_limit=10.,
        independent_method='Radau with dense analytic Jacobian of the same Fourier semidiscretization; RK45/Radau agreement separates temporal solver error from spatial-grid sensitivity',
        saved_times=[0.,.05,.1,.2,.4,.8],additional_accepted_step_extrema=True,
        diagnostics=dict(temporal_agreement_max_abs=1e-5,spatial_agreement_max_abs=1e-4,
            interpretation='Thresholds are predeclared for this audit. A failed solver or unresolved grid is not automatically a proof of continuum blow-up'),
    )
    path=target/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=record:raise AssertionError('Forward audit source/protocol changed')
    else:put_json(path,record);shutil.copy2(__file__,target/'source.py')
    return target,record


def load_operator(output,case,state):
    path=output/'models'/f"{state['run_id']}.pt"
    if sha(path)!=state['checkpoint_sha256']:raise AssertionError('Teacher changed')
    record=torch.load(path,map_location='cpu',weights_only=False)
    model=LocalOperator(len(SYSTEMS[case['system']]['unknowns']));model.load_state_dict(record['state_dict'])
    return model,record['x_scale'],float(record['y_scale'])


def principal_checks(output,target,cases,protocol):
    initial_specs=json.loads((output/'PROTOCOL.json').read_text())['protocol']['evaluation']['initial_conditions']
    rows=[];max_jacobian_difference=0.
    for case in cases:
        unknowns=SYSTEMS[case['system']]['unknowns']
        if 'D' not in unknowns:continue
        col=unknowns.index('D')
        for state in case['states']:
            model,xs,ys=load_operator(output,case,state);evaluate=numpy_operator_with_gradient(model,xs,ys)
            path=output/'rollouts'/f"{state['run_id']}_teacher.npz"
            saved=np.load(path) if path.exists() else None
            for ic,spec in initial_specs.items():
                initial=fields(np.arange(64)/64,np.array([0.]),case['system'],**spec)['u'][0]
                X=spectral_features(initial,unknowns);_,grad=evaluate(X)
                Z=torch.tensor(X/xs,dtype=torch.float64,requires_grad=True)
                ad=torch.autograd.grad(model(Z).sum(),Z)[0].detach().numpy()*ys/xs
                discrepancy=float(np.max(np.abs(ad-grad)));max_jacobian_difference=max(max_jacobian_difference,discrepancy)
                if discrepancy>1e-11:raise AssertionError('Independent input Jacobian differs')
                common=dict(case=case['case'],run_id=state['run_id'],system=case['system'],training_grid=case['grid'],noise=case['noise'],seed=case['seed'],stage=state['stage'],initial_condition=ic)
                min_initial=float(grad[:,col].min())
                rows.append(dict(**common,source='initial',evaluation_grid=64,samples=len(initial),minimum_principal_derivative=min_initial,
                    maximum_principal_derivative=float(grad[:,col].max()),negative_samples=int(np.sum(grad[:,col]<-1e-10)),
                    negative_fraction=float(np.mean(grad[:,col]<-1e-10)),old_spatially_resolved=state['rollouts']['teacher']['grid_checks'][ic]['grid_agreement_pass']))
                if saved is not None:
                    for nx in [32,64]:
                        key=f'{ic}_n{nx}'
                        if key not in saved:continue
                        curves=saved[key];jet=np.concatenate([spectral_features(u,unknowns) for u in curves],axis=0)
                        _,g=evaluate(jet);principal=g[:,col]
                        rows.append(dict(**common,source='saved_times',evaluation_grid=nx,samples=len(principal),minimum_principal_derivative=float(principal.min()),
                            maximum_principal_derivative=float(principal.max()),negative_samples=int(np.sum(principal<-1e-10)),
                            negative_fraction=float(np.mean(principal<-1e-10)),old_spatially_resolved=state['rollouts']['teacher']['grid_checks'][ic]['grid_agreement_pass']))
            if saved is not None:saved.close()
    if sum(r['source']=='initial' for r in rows)!=270:raise AssertionError('Initial derivative coverage incomplete')
    put_csv(target/'principal_derivatives.csv',rows)
    summaries=[]
    for system in ['heat','advection_diffusion','reaction_diffusion']:
        for noise in [0.,.01]:
            subset=[r for r in rows if r['system']==system and r['noise']==noise and r['source']=='initial']
            save=[r for r in rows if r['system']==system and r['noise']==noise and r['source']=='saved_times']
            detected_saved={(r['run_id'],r['initial_condition']) for r in save if r['negative_samples']>0}
            summaries.append(dict(system=system,noise=noise,teacher_initial_condition_pairs=len(subset),
                negative_at_initial=sum(r['negative_samples']>0 for r in subset),negative_at_any_available_saved_time=len(detected_saved),
                initially_negative_and_old_unresolved=sum(r['negative_samples']>0 and not r['old_spatially_resolved'] for r in subset),
                old_unresolved=sum(not r['old_spatially_resolved'] for r in subset)))
    put_csv(target/'principal_summary.csv',summaries)
    put_json(target/'PRINCIPAL_CHECKS.json',dict(status='passed',initial_cases=270,all_sample_groups=len(rows),
        maximum_analytic_autograd_jacobian_difference=max_jacobian_difference,
        initially_negative_cases=sum(r['source']=='initial' and r['negative_samples']>0 for r in rows)))
    return summaries


class AuditStop(RuntimeError):pass


def integrate(evaluate,unknowns,initial,times,method,protocol):
    n=len(initial);matrices=fourier_feature_matrices(n,unknowns)
    calls=0;jac_calls=0;started=time.monotonic();last_t=0.
    def check(t,u):
        nonlocal last_t
        last_t=float(t)
        if calls>protocol['rhs_budget']:raise AuditStop('RHS budget exceeded')
        if time.monotonic()-started>protocol['wall_seconds_per_solve']:raise AuditStop('Declared per-solve wall budget exceeded')
        if not np.all(np.isfinite(u)):raise AuditStop('Nonfinite intermediate state')
        if np.max(np.abs(u))>protocol['absolute_intermediate_state_limit']:raise AuditStop('Intermediate state exceeded declared absolute limit')
    def rhs(t,u):
        nonlocal calls
        calls+=1;check(t,u)
        X=spectral_features(u,unknowns);value,_=evaluate(X)
        return value
    def jac(t,u):
        nonlocal jac_calls
        jac_calls+=1;check(t,u)
        X=spectral_features(u,unknowns);_,derivative=evaluate(X)
        return sum(derivative[:,j,None]*matrix for j,matrix in enumerate(matrices))
    kwargs=dict(method=method,rtol=protocol['tolerances']['rtol'],atol=protocol['tolerances']['atol'],dense_output=True,t_eval=times)
    if method=='Radau':kwargs['jac']=jac
    try:solution=solve_ivp(rhs,(times[0],times[-1]),initial,**kwargs)
    except AuditStop as error:
        return dict(status='failed',reason=str(error),rhs_evaluations=calls,jacobian_evaluations=jac_calls,
            last_attempted_time=last_t,elapsed_seconds=time.monotonic()-started),None,None
    meta=dict(status='completed' if solution.success else 'failed',reason=solution.message,rhs_evaluations=calls,
        jacobian_evaluations=jac_calls,last_attempted_time=last_t,elapsed_seconds=time.monotonic()-started)
    if not solution.success:return meta,None,None
    accepted_times=solution.sol.ts;accepted=solution.sol(accepted_times).T
    meta.update(accepted_steps=len(accepted_times)-1,accepted_state_minimum=float(accepted.min()),accepted_state_maximum=float(accepted.max()))
    return meta,solution.y.T,dict(times=accepted_times,states=accepted)


def forward_checks(output,target,cases,protocol):
    original=json.loads((output/'PROTOCOL.json').read_text())['protocol'];times=np.array(protocol['saved_times'])
    lookup={state['run_id']:(case,state) for case in cases for state in case['states']}
    rows=[];summary=[]
    for selected in protocol['representatives']:
        case,state=lookup[selected['run_id']];model,xs,ys=load_operator(output,case,state)
        evaluate=numpy_operator_with_gradient(model,xs,ys);unknowns=SYSTEMS[case['system']]['unknowns']
        spec=original['evaluation']['initial_conditions'][selected['initial_condition']];curves={};meta_records={}
        for solver in protocol['forward_solvers']:
            method,nx=solver['method'],solver['grid'];x=np.arange(nx)/nx
            ref=fields(x,times,case['system'],**spec)['u'];key=f'{selected["run_id"]}_{selected["initial_condition"]}_{method}_n{nx}'
            path=target/'solves'/f'{key}.json'
            if path.exists():
                record=json.loads(path.read_text())
                if record['status']=='completed':
                    with np.load(target/'solves'/f'{key}.npz') as stored:curves[(method,nx)]=stored['u']
            else:
                meta,u,accepted=integrate(evaluate,unknowns,ref[0],times,method,protocol)
                record=dict(**selected,method=method,evaluation_grid=nx,**meta)
                if u is not None:
                    record.update(rollout_metrics(u,ref,times,case['system'],selected['initial_condition']))
                    curves[(method,nx)]=u
                    (target/'solves').mkdir(exist_ok=True)
                    np.savez_compressed(target/'solves'/f'{key}.npz',times=times,u=u,accepted_times=accepted['times'],accepted_states=accepted['states'])
                put_json(path,record)
                print(json.dumps(record),flush=True)
            rows.append(record);meta_records[(method,nx)]=record
        spatial=float(np.max(np.abs(curves[('RK45',128)]-curves[('RK45',256)][:,::2]))) if ('RK45',128) in curves and ('RK45',256) in curves else None
        temporal=float(np.max(np.abs(curves[('RK45',128)]-curves[('Radau',128)]))) if ('RK45',128) in curves and ('Radau',128) in curves else None
        summary.append(dict(**selected,completed_solvers=len(curves),
            rk128_rk256_spatial_difference=spatial,rk128_radau128_temporal_difference=temporal,
            temporal_agreement=bool(temporal is not None and temporal<=protocol['diagnostics']['temporal_agreement_max_abs']),
            spatial_agreement=bool(spatial is not None and spatial<=protocol['diagnostics']['spatial_agreement_max_abs'])))
    flexible_csv(target/'solver_attempts.csv',rows);flexible_csv(target/'representative_summary.csv',summary)
    return summary


def report(target,principal,forward):
    lines=['# Independent forward audit of the fixed linear-PDE teachers','',
        'This follow-up uses the existing checkpoints and was declared after the original rollout-resolution checks. No model was retrained and no compression arm replaced.','',
        '## Principal derivative coverage','',
        'For all 270 diffusiveteacher/initial-condition pairs, evaluate the principal derivative dN/d(uxx) at the exact initial field. Every available saved nx32/nx64 teacher trajectory is also checked. A negative sample detects a local violation of forward-parabolic monotonicity at that sampled jet. Positive samples do not certify positivity of this derivative globally or over unsaved trajectory times. The analytic network derivative is independently compared with PyTorch autograd.','',
        '| System | Noise | Initial cases | Negative initially | Negative on available saved states | Old unresolved |',
        '|---|---:|---:|---:|---:|---:|']
    for r in principal:lines.append(f"| {r['system']} | {r['noise']:g} | {r['teacher_initial_condition_pairs']} | {r['negative_at_initial']} | {r['negative_at_any_available_saved_time']} | {r['old_unresolved']} |")
    lines+=['','## Deterministic representative refinement','',
        'Choose one case in each system/noise stratum containing an unresolved original teacher rollout. If any integration failed, choose the lexicographically first runID/initial-condition pair; otherwise choose the greatest original finite spatial discrepancy, with lexical tie breaking. Exact selected IDs and old diagnostics were frozen before new evaluations. Each selected case receives RK45 at nx128 and nx256 plus Radau at nx128. Both methods use rtol=1e-9 and atol=1e-11; Radau has an analytic dense Fourier Jacobian. Failed or budget-limited solves remain failures.','',
        '| Teacher / initial condition | Completed / 3 | RK128 versus Radau128 | RK128 versus RK256 |',
        '|---|---:|---:|---:|']
    for r in forward:
        fmt=lambda value:'unavailable' if value is None else f'{value:.6g}'
        lines.append(f"| {r['run_id']} / {r['initial_condition']} | {r['completed_solvers']} | {fmt(r['rk128_radau128_temporal_difference'])} | {fmt(r['rk128_rk256_spatial_difference'])} |")
    lines+=['',
        'Temporal agreement at a fixed spatial grid supports the diagnosis that spatial sensitivity is not simply an RK45 time-integration artifact. It does not establish convergence to a continuum solution. Neither an unresolved pseudospectral grid nor an integration failure alone proves continuum blow-up. Detected negative principal derivatives provide a separate structural diagnostic. The remaining unresolved cases retain their original status; representative cases are not extrapolated to all teachers.','',
        'Accepted-step extrema are additionally saved for completed refined solves. These cover every accepted time of that solve, still not a global continuous-time certificate. Derivative samples on an unresolved numerical trajectory describe the saved finite-dimensional trajectory; the exact initial-state derivative test is independent of its later integration accuracy.','',
        'Artifacts: `PROTOCOL.json`, frozen `source.py`, all `principal_derivatives.csv`, `principal_summary.csv`, `PRINCIPAL_CHECKS.json`, all `solves/*.json` and successful states, `solver_attempts.csv`, `representative_summary.csv`.','']
    (target/'README.md').write_text('\n'.join(lines))
    put_json(target/'COMPLETE.json',dict(status='audit_finished_all_diagnostic_statuses_retained',principal_initial_cases=270,
        representatives=len(forward),refined_solver_attempts=3*len(forward),
        completed_refined_solvers=sum(r['completed_solvers'] for r in forward),
        temporal_agreement_cases=sum(r['temporal_agreement'] for r in forward),
        spatial_agreement_cases=sum(r['spatial_agreement'] for r in forward),source_sha256=sha(__file__)))


def run(output):
    torch.set_num_threads(1);target,protocol=declare(output);cases=read_cases(output)
    principal=principal_checks(output,target,cases,protocol)
    forward=forward_checks(output,target,cases,protocol)
    report(target,principal,forward)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    if args.declare_only:declare(args.output)
    else:run(args.output)
