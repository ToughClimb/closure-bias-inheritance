"""Re-run the original Table 2/Table 4 cohorts with complete saved records.

Table 2 retains the original six-trajectory 48-cell/150-epoch protocol.
Table 4 retains the original Exp clean/5%-noise, three-seed 64-cell protocol.
All inputs, neural state dictionaries, curves and validation trajectories are
saved before symbolic searches. Existing complete fits are reused only after
checking their exact configuration and core-source fingerprint.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D
from closure_discovery.evaluation.metrics import relative_l2_error
from closure_discovery.evaluation.rollout import compare_cases_on_shared_initial_conditions
from closure_discovery.models.mlp_closure import ReactionDiffusionClosure
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig, make_paper_training_config, run_closure_identification,
)
from closure_discovery.symbolic.restricted_fit import (
    SymbolicClosurePair, fit_diffusion_candidates, fit_reaction_candidates,
    select_symbolic_expression,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def core_signature():
    files = {str(path.relative_to(ROOT)):sha(path) for path in sorted((ROOT/'src').rglob('*.py'))}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def environment():
    versions = {}
    for name in ['numpy','scipy','torch','pysr','juliacall','juliapkg']:
        try:versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=versions,
                torch_threads=torch.get_num_threads(), device='cpu')


def training_protocol(suite, case_name, setting, seed):
    case = CASE_BUILDERS[case_name]()
    interval = tuple(case.value_range)
    if suite == 'excitation' and setting == 'low':
        midpoint = sum(interval)/2
        half = .05*(interval[1]-interval[0])
        amplitude = (midpoint-half, midpoint+half)
    else:amplitude = interval
    return dict(suite=suite,case=case_name,setting=setting,seed=seed,
                simulation=dict(nx=48 if suite=='excitation' else 64,dt=1e-4,
                    t_final=.05 if suite=='excitation' else .1,
                    save_every=5 if suite=='excitation' else 10,boundary='periodic'),
                trajectories=6 if suite=='excitation' else 8,
                amplitude_range=amplitude,initial_clip_range=amplitude,
                observation=asdict(ObservationConfig(noise_level=.05 if setting=='noise5' else 0.)),
                training=asdict(make_paper_training_config(epochs=150)),
                modes=4,closure_grid_points=256,unseen_trajectories=4,unseen_seed_offset=1000,
                replication='Three data seeds; population SD matches the original tables')


def train(output, protocol):
    ident = '_'.join(str(protocol[key]) for key in ['case','setting','seed'])
    path = output/protocol['suite']/ident
    record = path/'record.json'
    signature = core_signature()
    if record.exists():
        value = json.loads(record.read_text())
        assert value['status']=='complete',record
        assert value['core_signature']==signature,record
        assert value['protocol']==json.loads(json.dumps(protocol)),record
        for artifact in value['artifacts']:
            assert sha(path/artifact['path'])==artifact['sha256']
        return value,path
    path.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    value=dict(id=ident,status='running',protocol=protocol,core_signature=signature,
               runner_sha256=sha(__file__),environment=environment(),
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(record,value)
    try:
        case=CASE_BUILDERS[protocol['case']]()
        config=SimulationConfig1D(**protocol['simulation'])
        result=run_closure_identification(case=case,simulation_config=config,
            num_trajectories=protocol['trajectories'],amplitude_range=tuple(protocol['amplitude_range']),
            num_initial_modes=4,observation_config=ObservationConfig(**protocol['observation']),
            training_config=make_paper_training_config(epochs=150),
            initial_clip_range=tuple(protocol['initial_clip_range']),seed=protocol['seed'],device='cpu')
        arrays={key:np.asarray(data) for key,data in result['dataset'].items()}
        npz(path/'observations.npz',**arrays)
        state={key:tensor.detach().cpu().numpy() for key,tensor in result['model'].state_dict().items()}
        npz(path/'neural_state.npz',**state)
        grid=np.asarray(result['evaluation_grid'])
        reconstructed=ReactionDiffusionClosure(hidden_width=64,hidden_depth=2,input_range=case.value_range,
                backbone='mlp',diffusion_degree=2,reaction_degree=3,residual_scale=.1)
        reconstructed.load_state_dict({key:torch.from_numpy(array.copy()) for key,array in state.items()},strict=True)
        with torch.no_grad():
            for branch in ['diffusion','reaction']:
                predicted=getattr(reconstructed,branch)(torch.from_numpy(grid.copy())).numpy()
                assert np.array_equal(predicted,result[branch+'_pred']),branch
        curves=dict(grid=grid,**{key:np.asarray(result[key]) for key in
                    ['diffusion_true','reaction_true','diffusion_pred','reaction_pred']})
        npz(path/'curves.npz',**curves)
        rollout=result['unseen_rollout']
        npz(path/'neural_validation.npz',times=rollout.true_dataset['t'],x=rollout.true_dataset['x'],
                initial=rollout.true_dataset['u0'],reference=rollout.true_dataset['u'],prediction=rollout.predicted_dataset['u'])
        save(path/'history.json',result['history'])
        observed=result['dataset']['u']
        lower,upper=float(np.min(observed)),float(np.max(observed))
        visited=(grid>=lower)&(grid<=upper)
        support=dict(observed_min=lower,observed_max=upper,grid_points=len(grid),
                     visited_points=int(visited.sum()),complement_points=int((~visited).sum()),
                     definition='Uniform preset grid intersected with the min-to-max observed-state enclosure; complement is the remaining preset-grid points')
        errors={}
        for branch in ['diffusion','reaction']:
            truth=curves[branch+'_true'].astype(float);pred=curves[branch+'_pred'].astype(float)
            for domain,mask in [('full',np.ones(len(grid),bool)),('visited',visited),('complement',~visited)]:
                errors[branch+'_'+domain]=None if not mask.any() else relative_l2_error(truth[mask],pred[mask])
            if visited.all():assert errors[branch+'_full']==errors[branch+'_visited']
        value.update(status='complete',metrics=result['metrics'],excitation=asdict(result['excitation']),
                     support=support,domain_errors=errors,
                     neural_model=dict(class_name='ReactionDiffusionClosure',state_file='neural_state.npz',
                         architecture=protocol['training'],input_range=list(case.value_range),
                         roundtrip_predictions_byte_identical=True),
                     artifacts=[dict(path=p.name,sha256=sha(p),size=p.stat().st_size)
                         for p in sorted(path.iterdir()) if p.name!='record.json' and p.is_file()])
    except Exception as error:
        value.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        save(record,value)
        raise
    value['seconds']=time.perf_counter()-start
    save(record,value)
    print(json.dumps(dict(id=ident,suite=protocol['suite'],status='complete',seconds=value['seconds'],
                         metrics=value['metrics'],support=value['support'])),flush=True)
    return value,path


def prepare_symbolic(value,path):
    destination=path/'restricted.json'
    if destination.exists():return
    protocol=value['protocol'];case=CASE_BUILDERS[protocol['case']]()
    with np.load(path/'curves.npz',allow_pickle=False) as archive:
        curves={key:archive[key].astype(float) for key in archive.files}
    grid=curves['grid'];expressions={}
    for branch,fitter in [('diffusion',fit_diffusion_candidates),('reaction',fit_reaction_candidates)]:
        selected=select_symbolic_expression(fitter(grid,curves[branch+'_pred']),relative_tolerance=.05,absolute_tolerance=1e-8)
        expressions[branch]=selected
    pair=SymbolicClosurePair(diffusion_expression=expressions['diffusion'],reaction_expression=expressions['reaction'],
                            value_range=case.value_range,name=value['id']+'_restricted')
    rollout=compare_cases_on_shared_initial_conditions(true_case=case,predicted_case=pair.to_case(),
        config=SimulationConfig1D(**protocol['simulation']),num_trajectories=4,seed=protocol['seed']+1000,
        amplitude_range=tuple(protocol['amplitude_range']),initial_clip_range=tuple(protocol['initial_clip_range']),num_modes=4)
    metrics={};prediction={}
    for branch,expr in expressions.items():
        predicted=expr.evaluate(grid);prediction[branch]=predicted
        true_error=relative_l2_error(curves[branch+'_true'],predicted)
        neural_error=value['metrics']['relative_error_'+('D' if branch=='diffusion' else 'R')]
        metrics[branch]=dict(surrogate_error=relative_l2_error(curves[branch+'_pred'],predicted),
                             true_error=true_error,BIR=true_error/(neural_error+1e-12))
    npz(path/'restricted_validation.npz',times=rollout.true_dataset['t'],reference=rollout.true_dataset['u'],prediction=rollout.predicted_dataset['u'])
    save(destination,dict(status='complete',teacher_record_sha256=sha(path/'record.json'),
         fitting_states=256,relative_selection_tolerance=.05,absolute_selection_tolerance=1e-8,
         expressions={branch:asdict(expr) for branch,expr in expressions.items()},metrics=metrics,
         unseen_relative_l2=rollout.relative_l2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['excitation','pysr-teachers'],required=True)
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/table2-table4-rerun-2026-10-17')
    parser.add_argument('--case',choices=['case_a','case_b','case_exp'])
    parser.add_argument('--seed',type=int,choices=range(3))
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    args.output_root.mkdir(parents=True,exist_ok=True)
    save(args.output_root/('environment_'+args.suite+'.json'),environment())
    cases=([args.case] if args.case else ['case_a','case_b','case_exp']) if args.suite=='excitation' else ['case_exp']
    settings=['low','high'] if args.suite=='excitation' else ['clean','noise5']
    seeds=[args.seed] if args.seed is not None else range(3)
    values=[]
    for case in cases:
        for setting in settings:
            for seed in seeds:
                value,path=train(args.output_root,training_protocol('excitation' if args.suite=='excitation' else 'pysr',case,setting,seed))
                if args.suite=='pysr-teachers':prepare_symbolic(value,path)
                values.append(dict(id=value['id'],status=value['status']))
    save(args.output_root/(args.suite+'_COMPLETE.json'),dict(status='complete',fits=values,expected=len(cases)*len(settings)*len(seeds)))


if __name__=='__main__':main()
