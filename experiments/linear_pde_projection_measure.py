"""Declared follow-up: project every saved teacher on observed training jets.

This follows the original box-versus-solution discrepancy; it is not presented
as an arm specified before the original experiment. Both arms remain reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import shutil

import numpy as np
import torch

# Support both direct script execution and python -m from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.linear_pde_extension import (
    LocalOperator, SYSTEMS, fields, feature_matrix, numpy_predictor,
    on_solution_errors, put_json, put_csv, rollout_metrics, scaled_ols, sha,
)
from experiments.linear_pde_report import read_cases


def declare(output):
    target=output/'projection_measure_followup';target.mkdir(exist_ok=True)
    records=[]
    for case in read_cases(output):
        for state in case['states']:
            records.append(dict(run_id=state['run_id'],checkpoint_sha256=state['checkpoint_sha256'],input_sha256=case['input_sha256']))
    if len(records)!=120:raise AssertionError('Follow-up must include all120 saved teachers')
    record=dict(source_sha256=sha(__file__),parent_protocol_sha256=sha(output/'PROTOCOL.json'),
        rationale='Declared after observing a discrepancy between uniform normalized-box projection and fixed attainable solution jets, especially RD; follow-up, not original preregistration',
        intervention='Replace only the frozen-projection measure by equal weight on every original observed training-residual feature row; no true derivatives or held-out solutions used in fitting',
        fixed_elements='Every one of the120 saved teacher states, original library and unknowns, no retraining, no per-seed arm selection; original box arm remains the primary saved arm',
        interpretation='Measure sensitivity changes both support and density; it cannot isolate support alone',
        evaluation='Same fixed attainable exact-solution jets and exact Fourier symbolic rollouts as original; original nonlinear teacher evaluations retained',
        physical_diagnostic='Sign of constant D rigorously diagnoses backward parabolicity if D<0 even when a finite smooth Fourier mixture has a formal finite-time solution',
        checkpoints=records)
    path=target/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=record:raise AssertionError('Follow-up protocol changed')
    else:put_json(path,record);shutil.copy2(__file__,target/'source.py')
    return target


def run(output):
    torch.set_num_threads(1);target=declare(output)
    original_protocol=json.loads((output/'PROTOCOL.json').read_text())['protocol']
    cases=read_cases(output);rows=[];coefficient_rows=[];rollouts=[];fits=[]
    for case in cases:
        unknowns=SYSTEMS[case['system']]['unknowns']
        with np.load(output/'inputs'/f"{case['case']}.npz") as data:
            X=data['X'];xs=data['x_scale'];ys=float(data['y_scale']);y=data['y']
        for state in case['states']:
            run_id=state['run_id'];path=output/'models'/f'{run_id}.pt'
            if sha(path)!=state['checkpoint_sha256']:raise AssertionError('Checkpoint changed')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            model=LocalOperator(len(unknowns));model.load_state_dict(saved['state_dict'])
            values=numpy_predictor(model,xs,ys)(X)
            fit=scaled_ols(X,values);coef=np.asarray(fit['coefficients'])
            independent=np.linalg.lstsq(X,values,rcond=None)[0]
            if np.max(np.abs(coef-independent))>1e-10:raise AssertionError('Independent projection differs')
            truecoef=np.array([SYSTEMS[case['system']][k] for k in unknowns])
            truth=X@truecoef;compressed=X@coef
            identity=float(abs(np.mean((values-truth)**2)-np.mean((compressed-truth)**2)-np.mean((values-compressed)**2)))
            if identity>1e-10*max(1.,float(np.mean(truth*truth))):raise AssertionError('Projection identity fails')
            common={key:case[key] for key in ['case','system','grid','noise','seed']};common.update(stage=state['stage'],run_id=run_id)
            fits.append(dict(**common,projection_identity_residual=identity,
                empirical_symbolic_residual_mse=float(np.mean((compressed-y)**2)),
                direct_residual_mse=case['direct']['residual_mse'],teacher_residual_mse=state['residual_mse'],
                teacher_projection_residual_mse=fit['residual_mse']))
            for name,value in zip(unknowns,coef,strict=True):
                coefficient_rows.append(dict(**common,coefficient=name,value=float(value),truth=SYSTEMS[case['system']][name]))
            for entry in on_solution_errors(model,xs,ys,case['system'],coef):
                if entry['method']=='frozen':rows.append(dict(**common,method='observed_jet_frozen',**{k:v for k,v in entry.items() if k!='method'}))
            times=np.asarray(original_protocol['evaluation']['rollout_times']);x=np.arange(128)/128
            candidatecoef={k:float(v) for k,v in zip(unknowns,coef,strict=True)}
            curves={}
            for initial_name,spec in original_protocol['evaluation']['initial_conditions'].items():
                reference=fields(x,times,case['system'],**spec)['u']
                candidate=fields(x,times,case['system'],coefficients=candidatecoef,**spec)['u']
                curves[initial_name]=candidate
                rollouts.append(dict(**common,method='observed_jet_frozen',initial_condition=initial_name,
                    negative_diffusion=bool(candidatecoef.get('D',0.)<0),
                    **rollout_metrics(candidate,reference,times,case['system'],initial_name)))
            (target/'rollouts').mkdir(exist_ok=True)
            np.savez_compressed(target/'rollouts'/f'{run_id}.npz',times=times,**curves)
            put_json(target/'fits'/f'{run_id}.json',dict(**common,coefficients=candidatecoef,
                fit=fit,input_sha256=case['input_sha256'],checkpoint_sha256=state['checkpoint_sha256']))
    for name,records in [('on_solution_jets',rows),('coefficients',coefficient_rows),('rollouts',rollouts),('projection_checks',fits)]:
        put_csv(target/f'{name}.csv',records)
    summary=[]
    for system in SYSTEMS:
        for noise in [0.,.01]:
            for stage in ['base','continued']:
                subset=[r for r in rows if r['system']==system and r['noise']==noise and r['stage']==stage and r['grid']==64 and r['initial_condition']=='heldout_mode3']
                vals=[r['relative_operator_error'] for r in subset]
                summary.append(dict(system=system,noise=noise,stage=stage,mean=float(np.mean(vals)),std=float(np.std(vals,ddof=1)),n=len(vals)))
    put_csv(target/'summary.csv',summary)
    put_json(target/'COMPLETE.json',dict(status='passed',teacher_states=len(fits),symbolic_rollouts=len(rollouts),
        max_projection_identity_residual=max(r['projection_identity_residual'] for r in fits),
        negative_diffusion_coefficients=sum(r['coefficient']=='D' and r['value']<0 for r in coefficient_rows),source_sha256=sha(__file__)))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    if args.declare_only:declare(args.output)
    else:run(args.output)
