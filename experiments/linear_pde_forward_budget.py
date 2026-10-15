"""Bounded budget extension for every RHS-budget-limited forward audit solve."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import shutil

import numpy as np
import torch

# Support both direct script execution and python -m from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.linear_pde_extension import SYSTEMS,fields,put_json,sha,rollout_metrics
from experiments.linear_pde_report import read_cases,flexible_csv
from experiments.linear_pde_forward_audit import load_operator,numpy_operator_with_gradient,integrate


def run(output,declare_only=False):
    torch.set_num_threads(1);audit=output/'forward_audit';target=audit/'budget_extension';target.mkdir(exist_ok=True)
    with (audit/'solver_attempts.csv').open() as stream:original=list(csv.DictReader(stream))
    selected=[r for r in original if r['status']=='failed' and r['reason']=='RHS budget exceeded']
    record=dict(source_sha256=sha(__file__),rationale='Predeclared additional budget for every, and only, original RHS-budget-limited audit solve; original failures retained',
        original_protocol_sha256=sha(audit/'PROTOCOL.json'),rhs_budget_multiplier=8,rhs_budget=240000,
        unchanged='Models, initial conditions, methods, grids, tolerances, wall/state limits; no retry of state-limit failures',
        selected=[dict(run_id=r['run_id'],initial_condition=r['initial_condition'],method=r['method'],grid=int(r['evaluation_grid'])) for r in selected])
    path=target/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=record:raise AssertionError('Budget protocol changed')
    else:put_json(path,record);shutil.copy2(__file__,target/'source.py')
    if declare_only:return
    protocol=json.loads((audit/'PROTOCOL.json').read_text());protocol['rhs_budget']=record['rhs_budget']
    original_protocol=json.loads((output/'PROTOCOL.json').read_text())['protocol']
    cases=read_cases(output);lookup={s['run_id']:(c,s) for c in cases for s in c['states']}
    times=np.asarray(protocol['saved_times']);rows=[]
    for item in record['selected']:
        case,state=lookup[item['run_id']];model,xs,ys=load_operator(output,case,state);evaluate=numpy_operator_with_gradient(model,xs,ys)
        spec=original_protocol['evaluation']['initial_conditions'][item['initial_condition']]
        nx=item['grid'];method=item['method'];ref=fields(np.arange(nx)/nx,times,case['system'],**spec)['u']
        key=f"{item['run_id']}_{item['initial_condition']}_{method}_n{nx}"
        meta,u,accepted=integrate(evaluate,SYSTEMS[case['system']]['unknowns'],ref[0],times,method,protocol)
        result=dict(**item,**meta)
        if u is not None:
            result.update(rollout_metrics(u,ref,times,case['system'],item['initial_condition']))
            np.savez_compressed(target/f'{key}.npz',times=times,u=u,accepted_times=accepted['times'],accepted_states=accepted['states'])
        put_json(target/f'{key}.json',result);rows.append(result);print(json.dumps(result),flush=True)
    flexible_csv(target/'solver_attempts.csv',rows)
    comparisons=[]
    for representative in protocol['representatives']:
        curves={}
        for method,nx in [('RK45',128),('RK45',256),('Radau',128)]:
            key=f"{representative['run_id']}_{representative['initial_condition']}_{method}_n{nx}"
            for directory in [target,audit/'solves']:
                npz=directory/f'{key}.npz'
                if npz.exists():
                    with np.load(npz) as data:curves[(method,nx)]=data['u']
                    break
        spatial=float(np.max(np.abs(curves[('RK45',128)]-curves[('RK45',256)][:,::2]))) if ('RK45',128) in curves and ('RK45',256) in curves else None
        temporal=float(np.max(np.abs(curves[('RK45',128)]-curves[('Radau',128)]))) if ('RK45',128) in curves and ('Radau',128) in curves else None
        comparisons.append(dict(**representative,completed_solvers_after_extension=len(curves),
            rk128_rk256_spatial_difference=spatial,rk128_radau128_temporal_difference=temporal,
            temporal_agreement=bool(temporal is not None and temporal<=protocol['diagnostics']['temporal_agreement_max_abs']),
            spatial_agreement=bool(spatial is not None and spatial<=protocol['diagnostics']['spatial_agreement_max_abs'])))
    flexible_csv(target/'representative_summary.csv',comparisons)
    put_json(target/'COMPLETE.json',dict(status='extension_finished_statuses_retained',selected_solves=len(selected),completed_solves=sum(r['status']=='completed' for r in rows),source_sha256=sha(__file__)))
    lines=['# Bounded extension of RHS-budget-limited forward solves','',
        'Every original RHS-budget failure was rerun at eight times the RHS budget. Source and exact IDs were frozen before these additional evaluations. The original failed records remain unchanged; no state-limit failure was rerun. Models, grids, tolerances, state limit, and 45-second wall budget remain identical.','',
        '| Run / initial condition | Completed methods /3 | RK128 versus Radau128 | RK128 versus RK256 |',
        '|---|---:|---:|---:|']
    for r in comparisons:
        fmt=lambda x:'unavailable' if x is None else f'{x:.7g}'
        lines.append(f"| {r['run_id']} / {r['initial_condition']} | {r['completed_solvers_after_extension']} | {fmt(r['rk128_radau128_temporal_difference'])} | {fmt(r['rk128_rk256_spatial_difference'])} |")
    lines+=['','Agreement thresholds remain 1e-5 for temporal solvers at the same grid, and 1e-4 for spatial refinement. This is a numerical diagnostic of the selected finite-dimensional systems, not a proof that a nonlinear continuum PDE is well posed.','']
    (target/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args();run(args.output,declare_only=args.declare_only)
