"""Preserve fixed-rule failures and independently refine forward evaluation.

This changes evaluation only. It does not replace teachers, symbolic
coefficients, observations, stopping rules, or original numerical artifacts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
import torch

from slit_flow_core import (FluidityNetwork, network_values, exact_fluidity, exact_flow_rate,
                            polynomial, geometry)
from run_slit_flow_controls import SOURCE, digest, write_json, write_csv, load_protocol


def composite_rule(panels, order):
    nodes, weights = leggauss(order)
    points = (np.arange(panels)[:,None] + (nodes[None,:]+1)/2)/panels
    weights = np.broadcast_to(weights[None,:]/(2*panels),points.shape).copy()
    return points,weights


def integral_values(function, pressure, y, points, weights):
    values = function((pressure*points).ravel()).reshape(points.shape)
    integral = (values*weights).sum(axis=1)
    cumulative = np.concatenate([np.cumsum(integral[::-1])[::-1],[0.]])
    indices = np.rint(y*len(points)).astype(int)
    if not np.allclose(indices/len(points),y,atol=1e-14):
        raise ValueError('The composite panel grid must include every observation node')
    return cumulative[indices],float(2*np.sum(points*values*weights)),values


def run(output, round_id):
    protocol=load_protocol(output)
    inputs=[]
    for stage,folder in [('adam1500','fits'),('lbfgs','continuation')]:
        for path in sorted((output/folder).glob('*.json')):
            record=json.loads(path.read_text())
            if record['status']!='complete':raise ValueError('Incomplete teacher')
            dataset=record['identity']['dataset'] if stage=='adam1500' else record['dataset']
            objective=record['identity']['objective'] if stage=='adam1500' else record['objective']
            inputs.append((stage,path,record,dataset,objective))
    if len(inputs)!=80:raise ValueError('Need all 60 teachers and 20 continuations')
    suffix='' if round_id==1 else '_V2'
    declared=dict(source_sha256=digest(__file__),original_protocol_sha256=digest(output/'PROTOCOL.json'),
        continuation_protocol_sha256=digest(output/'LBFGS_PROTOCOL.json'),
        coarse=dict(panels=1024,order=8) if round_id==1 else dict(panels=4096,order=16),
        fine=dict(panels=4096,order=16) if round_id==1 else dict(panels=8192,order=16),absolute_agreement_tolerance=1e-8,
        reason='Fixed forward quadrature fails for sharp noisy differential-objective continuations',
        action='Refine evaluation for every model, preserve original values, never refit or select models',
        input_sha256={str(path.relative_to(output)):digest(path) for _,path,_,_,_ in inputs})
    declaration=output/('QUADRATURE_AUDIT'+suffix+'_PROTOCOL.json')
    if declaration.exists() and json.loads(declaration.read_text())!=declared:
        raise ValueError('Independent audit protocol differs')
    write_json(declaration,declared)
    coarse=composite_rule(**declared['coarse'])
    fine=composite_rule(**declared['fine'])
    alpha=protocol['plan']['alpha']
    results={}
    checks=[]
    torch.set_num_threads(1)
    for stage,path,record,dataset,objective in inputs:
        if digest(output/record['model_file'])!=record['model_sha256']:raise ValueError('Model changed')
        if digest(output/dataset['file'])!=dataset['file_sha256']:raise ValueError('Observations changed')
        model=FluidityNetwork(protocol['plan']['width']).double()
        model.load_state_dict(torch.load(output/record['model_file'],weights_only=True)['state_dict'])
        function=lambda x:network_values(model,x)
        with np.load(output/dataset['file']) as data:y,observed=data['y'],data['velocity']
        metrics=json.loads(json.dumps(record['metrics']))
        flux={}
        uniform_integral=None
        continuous_relative=None
        for pressure in protocol['plan']['pressure_test']:
            uc,qc,vc=integral_values(function,pressure,y,*coarse)
            uf,qf,vf=integral_values(function,pressure,y,*fine)
            flux[str(pressure)]=abs(qf-exact_flow_rate(pressure,alpha))/exact_flow_rate(pressure,alpha)
            if pressure==1.:
                uniform_integral=uf
                true_fine=exact_fluidity(fine[0],alpha)
                true_coarse=exact_fluidity(coarse[0],alpha)
                fine_error=float(np.sum(fine[1]*(vf-true_fine)**2))
                coarse_error=float(np.sum(coarse[1]*(vc-true_coarse)**2))
                continuous_relative=float(np.sqrt(fine_error/np.sum(fine[1]*true_fine**2)))
                for family in record['symbols']:
                    coeff=record['symbols'][family]['frozen_coefficients']
                    symbolic=polynomial(fine[0].ravel(),coeff).reshape(fine[0].shape)
                    metrics[family+'_frozen']['refined_common_geometry']=geometry(true_fine,vf,symbolic,fine[1])
                squared_error_difference=abs(fine_error-coarse_error)
            else:squared_error_difference=0.
            # Adaptive integral with fixed breakpoints is independent of the
            # composite Gauss rule and prevents a narrow feature being skipped.
            adaptive=2*quad(lambda s:s*float(function(np.array([pressure*s]))[0]),0,1,
                points=np.linspace(0,1,65)[1:-1],epsabs=1e-10,epsrel=1e-10,limit=2048)[0]
            checks.append(dict(stage=stage,dataset=dataset['id'],objective=objective,pressure=pressure,
                coarse_flow=qc,refined_flow=qf,adaptive_flow=adaptive,coarse_vs_fine_flow=abs(qc-qf),
                adaptive_vs_fine_flow=abs(adaptive-qf),coarse_vs_fine_velocity=float(np.max(np.abs(uc-uf))),
                coarse_vs_fine_squared_closure_error=squared_error_difference,
                original_vs_refined_flow=abs(float(record['metrics']['neural']['flow_relative_errors'][str(pressure)])-flux[str(pressure)])))
        metrics['neural']['flow_relative_errors']=flux
        metrics['neural']['observation_mse']=float(np.mean((uniform_integral-observed)**2))
        metrics['neural']['refined_continuous_relative_closure_error']=continuous_relative
        # The headline closure metric remains the original declared 1025-point
        # norm; a separate refined continuous norm exposes under-resolution.
        key=f'{stage}/{dataset["id"]}/{objective}'
        results[key]=dict(input_record_sha256=digest(path),metrics=metrics)
    write_csv(output/('reports/refined_quadrature_checks'+suffix+'.csv'),checks)
    fields=['coarse_vs_fine_flow','adaptive_vs_fine_flow','coarse_vs_fine_velocity','coarse_vs_fine_squared_closure_error']
    maxima={field:max(row[field] for row in checks) for field in fields}
    status='passed' if max(maxima.values())<=declared['absolute_agreement_tolerance'] else 'failed'
    summary=dict(status=status,protocol_sha256=digest(declaration),maxima=maxima,
        all_models_evaluated=80,pressure_checks=len(checks),original_metric_difference_max=max(r['original_vs_refined_flow'] for r in checks),
        original_fits_and_symbols_preserved=True,results=results)
    write_json(output/('QUADRATURE_AUDIT'+suffix+'.json'),summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),flush=True)
    if status!='passed':raise ValueError('Refined/adaptive quadrature disagreement retained; do not report an unverified engineering metric')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--round',type=int,choices=[1,2],default=2)
    args=parser.parse_args()
    run(args.output,args.round)


if __name__=='__main__':main()
