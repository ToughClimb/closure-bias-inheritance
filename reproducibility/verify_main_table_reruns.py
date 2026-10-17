"""Verify the saved Table 2/Table 4 reruns without training or Julia searches."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import sympy as sp
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from closure_discovery.models.mlp_closure import ReactionDiffusionClosure


def check_teacher(path):
    record=json.loads((path/'record.json').read_text())
    assert record['status']=='complete'
    protocol=record['protocol']
    assert protocol['training']['epochs']==150 and protocol['seed'] in range(3)
    assert protocol['training']['hidden_width']==64 and protocol['training']['hidden_depth']==2
    model=ReactionDiffusionClosure(hidden_width=64,hidden_depth=2,
            input_range=record['neural_model']['input_range'],backbone='mlp',
            diffusion_degree=2,reaction_degree=3,residual_scale=.1)
    with np.load(path/'neural_state.npz',allow_pickle=False) as archive:
        state={key:torch.from_numpy(archive[key].copy()) for key in archive.files}
    model.load_state_dict(state,strict=True)
    with np.load(path/'curves.npz',allow_pickle=False) as archive:
        curves={key:archive[key] for key in archive.files}
    grid=curves['grid'];assert len(grid)==256 and grid.dtype==np.float32
    with torch.no_grad():
        for branch in ['diffusion','reaction']:
            pred=getattr(model,branch)(torch.from_numpy(grid.copy())).numpy()
            assert np.array_equal(pred,curves[branch+'_pred']),path
    with np.load(path/'observations.npz',allow_pickle=False) as archive:
        observed=archive['u'];nx=len(archive['x'])
    assert nx==protocol['simulation']['nx']
    assert observed.shape[0]==protocol['trajectories']
    lower,upper=float(observed.min()),float(observed.max())
    assert record['support']['observed_min']==lower and record['support']['observed_max']==upper
    visited=(grid>=lower)&(grid<=upper)
    assert int(visited.sum())==record['support']['visited_points']
    for branch in ['diffusion','reaction']:
        truth=curves[branch+'_true'].astype(float);pred=curves[branch+'_pred'].astype(float)
        for domain,mask in [('full',np.ones(len(grid),bool)),('visited',visited),('complement',~visited)]:
            actual=record['domain_errors'][branch+'_'+domain]
            if not mask.any():assert actual is None
            else:
                expected=float(np.linalg.norm(truth[mask]-pred[mask])/(np.linalg.norm(truth[mask])+1e-12))
                assert abs(actual-expected)<1e-12
    with np.load(path/'neural_validation.npz',allow_pickle=False) as archive:
        reference,prediction=archive['reference'],archive['prediction']
        error=float(np.linalg.norm(reference-prediction)/(np.linalg.norm(reference)+1e-12))
        assert abs(error-record['metrics']['unseen_rollout_relative_l2'])<1e-12
    return record,curves


def verify(records):
    root=records/'main_table_rerun'
    torch.set_num_threads(1)
    table2=[];table4=[]
    for path in sorted((root/'table2').iterdir()):
        record,_=check_teacher(path)
        assert record['protocol']['simulation']['nx']==48
        assert record['protocol']['trajectories']==6
        table2.append(record)
    assert len(table2)==18
    assert {(row['protocol']['case'],row['protocol']['setting'],row['protocol']['seed']) for row in table2}=={
        (case,setting,seed) for case in ['case_a','case_b','case_exp'] for setting in ['low','high'] for seed in range(3)}
    for row in table2:
        if row['protocol']['setting']=='high':assert row['support']['complement_points']==0
    with (root/'table2_domain_summary.csv').open(newline='') as stream:
        summaries=list(csv.DictReader(stream))
    assert len(summaries)==6
    for summary in summaries:
        group=[row for row in table2 if row['protocol']['case']==summary['case'] and row['protocol']['setting']==summary['regime']]
        assert len(group)==3
        for key in group[0]['domain_errors']:
            values=[row['domain_errors'][key] for row in group]
            if values[0] is None:
                assert all(value is None for value in values)
                assert summary[key+'_mean']==summary[key+'_sd']==''
            else:
                assert abs(statistics.mean(values)-float(summary[key+'_mean']))<1e-12
                assert abs(statistics.pstdev(values)-float(summary[key+'_sd']))<1e-12
    stops={};searches=[]
    for path in sorted((root/'table4').iterdir()):
        record,curves=check_teacher(path);table4.append(record)
        assert record['protocol']['simulation']['nx']==64 and record['protocol']['trajectories']==8
        assert record['protocol']['case']=='case_exp'
        seed=record['protocol']['seed']
        for branch in ['diffusion','reaction']:
            folder=path/'search'/branch
            row=json.loads((folder/'record.json').read_text());assert row['status']=='complete'
            config=row['config'];assert config['binary_operators']==['+','-','*'] and config['unary_operators']==['exp']
            parameters=json.loads((folder/'parameters.json').read_text())
            assert parameters['ncycles_per_iteration']==380
            assert parameters['niterations']==40 and parameters['populations']==8
            assert parameters['binary_operators']==['+','-','*'] and parameters['unary_operators']==['exp']
            assert config['niterations']==40 and config['populations']==8 and config['population_size']==48
            assert config['timeout_in_seconds']==180 and config['maxsize']==30 and config['precision']==32
            assert config['random_state']==seed+(101 if branch=='reaction' else 0)
            assert config['early_stop_condition'] is config['max_evals'] is None
            assert config['input_stream']=='devnull' and config['parallelism']=='serial' and config['deterministic']
            with (folder/'cycles.csv').open(newline='') as stream:cycles=list(csv.DictReader(stream))
            assert len(cycles)==row['cycle_callbacks'] and len(cycles)>0
            last=cycles[-1]
            assert int(last['remaining_population_cycles'])==row['remaining_population_cycles']
            assert int(last['completed_population_cycles'])==row['completed_population_cycles']
            assert row['completed_population_cycles']+row['remaining_population_cycles']==320
            assert abs(float(last['evaluations'])-row['evaluations'])<1e-6
            assert row['stopping_reason']==('iteration_budget' if row['remaining_population_cycles']==0 else 'timeout')
            if row['stopping_reason']=='timeout':assert row['seconds']>=180
            expression=sp.sympify(row['expression_sympy_srepr'])
            function=sp.lambdify(sp.Symbol('x0'),expression,modules=['numpy'])
            with np.load(folder/'prediction.npz',allow_pickle=False) as archive:
                predicted=archive['prediction'];grid=archive['grid']
            restored=np.broadcast_to(function(grid),grid.shape)
            assert np.max(abs(restored-predicted))<1e-10
            for kind,reference in [('surrogate_error',curves[branch+'_pred'].astype(float)),('true_error',curves[branch+'_true'].astype(float))]:
                expected=float(np.linalg.norm(reference-predicted)/(np.linalg.norm(reference)+1e-12))
                assert abs(expected-row['metrics'][kind])<1e-12
            stops[row['stopping_reason']]=stops.get(row['stopping_reason'],0)+1
            searches.append(row)
        with np.load(path/'pysr_validation.npz',allow_pickle=False) as archive:
            expected=float(np.linalg.norm(archive['reference']-archive['prediction'])/(np.linalg.norm(archive['reference'])+1e-12))
        validation=json.loads((path/'pysr_validation.json').read_text())
        assert abs(expected-validation['unseen_relative_l2'])<1e-12
    assert len(table4)==6 and len(searches)==12
    return dict(status='passed',table2_saved_models=18,table4_saved_teachers=6,
        neural_state_roundtrips_verified=24,pysr_expressions_and_stops_verified=12,stopping_counts=stops,
        neural_training_runs=0,pysr_searches_run=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,default=Path(__file__).resolve().parent/'paper-records-2026-10-16')
    args=parser.parse_args()
    print(json.dumps(verify(args.records.resolve()),indent=2))
