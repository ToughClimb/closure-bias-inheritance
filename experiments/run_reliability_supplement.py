"""Same-observation Exp controls and predeclared long-budget training.

No manuscript edits, observation regeneration, or truth-based selection.
Run prepare first, then baselines and train. Saved inputs and optimizer states
support resuming this new campaign; historical incomplete states are not reused.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'revision'))

import numpy as np
import torch
from revision_core import (ROOT, SOURCE_ROOT, DEFAULT_WEIGHTS, extra_case, train, fit_symbols,
    curve_metrics, observation_signature, source_signature, process_pool, report_completed,
    json_write, npz_write, torch_write, resolve_device, fit_weak_form_polynomial_closure,
    fit_strong_form_polynomial_closure, spacetime_polynomial)
from reliability_core import NumpyMLPClosure, objective_state, objective_diagnostic, tail_diagnostic, lbfgs_crosscheck
from closure_discovery.models.mlp_closure import ReactionDiffusionClosure


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_manifest():
    manifest = json.loads((ROOT / 'evidence/PROTOCOL.json').read_text())
    if manifest['source_signature'] != source_signature() or manifest['runner_sha256'] != digest(__file__):
        raise ValueError('Campaign source changed; preserve existing inputs and use a new audited work root')
    return manifest


def observations(dataset):
    path = ROOT / dataset['observation_file']
    if digest(path) != dataset['observation_file_sha256']:
        raise ValueError('Saved observation file changed')
    with np.load(path, allow_pickle=False) as values:
        result = {key: values[key].copy() for key in values.files}
    if hashlib.sha256(result['u'].tobytes()).hexdigest() != dataset['observation_u_sha256']:
        raise ValueError('Saved observation field changed')
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('Nonfinite saved observations')
    return result


def prepare(args):
    if args.observation_root is None or args.refit_root is None:
        raise ValueError('prepare requires --observation-root and --refit-root')
    plan = dict(case='case_exp', settings=['clean', 'noise3'], seeds=list(range(5)),
        input_policy='Use saved observations exactly; no regeneration', training_dtype='float32', device=str(resolve_device()),
        architecture='Original polynomial-residual width64 depth2 MLP', weights=DEFAULT_WEIGHTS,
        init_seed_rule='Data seed', adam_lr=.001, adam_budget=args.adam_budget,
        adam_extension_budget=args.adam_extension_budget, tail_window=1000,
        extension_rule='Extend only if trailing-window mean objective decline exceeds threshold',
        extension_relative_threshold=.01, lbfgs_outer_steps=args.lbfgs_outer_steps, lbfgs_inner_iterations=20,
        lbfgs_dtype='float64', lbfgs_line_search='strong_wolfe', lbfgs_gradient_tolerance=1e-7,
        lbfgs_parameter_tolerance=1e-10, lbfgs_objective_tolerance=1e-12,
        optimizer_crosscheck='Separate protocol; do not choose between methods using true errors',
        compression=dict(support_points=512, relative_tolerance=.05, absolute_tolerance=1e-8,
                         source='Fresh teacher checkpoint, never archived teacher relabeling'),
        baseline=dict(diffusion_degree=2, reaction_degree=3, ridge=1e-8, weak_fourier_modes=4),
        coefficient_refit=dict(teacher='Fresh 150-epoch teacher', phase1_nfev=40, additional_nfev=80,
            continuation_rule='Only phase-one optimizer budget exhaustion',
            positivity_floor=1e-4, reaction_anchor='None; disclosed difference from neural soft anchor'),
        evaluation=dict(grid_points=256, interval=[0.,1.8], clipping=False,
            neural='Native trained weights evaluated in float64; no table endpoint extension',
            report_numerical_completion_and_domain_validity_separately=True),
        classification='predeclared_reviewer_supplement', manuscript_policy='No manuscript changes')
    if args.adam_budget<1 or args.adam_extension_budget<args.adam_budget or args.lbfgs_outer_steps<1:
        raise ValueError('Invalid fixed training budgets')
    datasets=[]
    for setting in plan['settings']:
        for seed in plan['seeds']:
            old_id=f'main_case_exp_{setting}_{seed}_default'
            record_path=args.refit_root/'evidence/pysr_observation_refit'/f'{old_id}.json'
            archived=json.loads(record_path.read_text())
            source=args.observation_root/f'{old_id}.npz'
            with np.load(source,allow_pickle=False) as values:
                if values['u'].shape!=(8,101,64) or not np.isfinite(values['u']).all():
                    raise ValueError(f'Unexpected saved observations: {old_id}')
                actual=hashlib.sha256(values['u'].tobytes()).hexdigest()
                if actual!=archived['data_hash']:
                    raise ValueError(f'Observation bytes do not match final refit: {old_id}')
            relative=Path('evidence/observations')/source.name
            destination=ROOT/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if digest(destination)!=digest(source):raise ValueError('Existing campaign observation differs')
            else:
                temporary=destination.with_suffix('.partial');shutil.copyfile(source,temporary)
                if digest(temporary)!=digest(source):raise ValueError('Observation copy checksum failed')
                temporary.replace(destination)
            datasets.append(dict(id=f'sameobs_case_exp_{setting}_{seed}', case='case_exp',setting=setting,seed=seed,
                archived_id=old_id,observation_file=str(relative),observation_file_sha256=digest(destination),
                observation_u_sha256=actual,archived_u_sha256=archived['archived_data_hash'],
                matches_archived_u=archived['bitwise_observation_match'],
                original_observation_file=str(source.resolve()), archived_refit_file=str(record_path.resolve()),
                archived_refit_file_sha256=digest(record_path)))
    payload=dict(plan=plan,datasets=datasets,source_signature=source_signature(),runner_sha256=digest(__file__))
    protocol=ROOT/'evidence/PROTOCOL.json'
    if protocol.exists():
        if json.loads(protocol.read_text())!=payload:
            raise ValueError('Existing protocol differs; do not overwrite a declared campaign')
        print('existing_protocol_verified',flush=True);return
    json_write(protocol,payload)
    files=[]
    for directory in ('src','revision','experiments','tests'):
        for source in sorted((SOURCE_ROOT/directory).rglob('*.py')):
            if '__pycache__' in source.parts:continue
            target=ROOT/'provenance/source'/source.relative_to(SOURCE_ROOT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            files.append(dict(path=str(source.relative_to(SOURCE_ROOT)),sha256=digest(source)))
    distributions=sorted((distribution.metadata['Name'],distribution.version)
                         for distribution in importlib.metadata.distributions())
    json_write(ROOT/'provenance/environment.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv,python=sys.version,torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        packages=distributions,source_files=files,
        git_head=subprocess.check_output(['git','-C',str(SOURCE_ROOT),'rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','-C',str(SOURCE_ROOT),'status','--short'],text=True)))
    (ROOT/'provenance/working_tree.patch').write_bytes(subprocess.check_output(['git','-C',str(SOURCE_ROOT),'diff','--binary']))
    print(json.dumps(dict(event='prepared',datasets=len(datasets),work_root=str(ROOT),plan=plan)),flush=True)


def archive_failure(path,retry):
    if not path.exists():return None
    prior=json.loads(path.read_text())
    if prior['status']=='complete':return prior
    if not retry:raise ValueError(f'Incomplete cached task: {path}; inspect record and explicitly retry')
    destination=ROOT/'evidence/failures'/f'{path.parent.name}_{path.stem}_{time.time_ns()}.json'
    destination.parent.mkdir(parents=True,exist_ok=True);path.replace(destination)
    return None


def baseline_task(dataset,retry=False):
    manifest=load_manifest();identity=dict(dataset=dataset,plan=manifest['plan'],source_signature=source_signature())
    destination=ROOT/'evidence/baselines'/f"{dataset['id']}.json"
    prior=archive_failure(destination,retry)
    if prior is not None:
        if prior['identity']!=identity:raise ValueError('Baseline cache mismatch')
        return dataset['id'],'cached'
    start=time.perf_counter();result=dict(identity=identity,id=dataset['id'],status='running',models={})
    try:
        observed=observations(dataset);case=extra_case('case_exp');grid=np.linspace(*case.value_range,256)
        for label,fit in [('weak_poly',fit_weak_form_polynomial_closure),
                          ('strong_poly',fit_strong_form_polynomial_closure),('spacetime_poly',spacetime_polynomial)]:
            model,mse=fit(observed,value_range=case.value_range)
            result['models'][label]=dict(kind='polynomial',parameters_D=model.diffusion_coefficients.tolist(),
                parameters_R=model.reaction_coefficients.tolist(),fit_mse=float(mse),
                observation_file_sha256=dataset['observation_file_sha256'],
                metrics=curve_metrics(case,grid,model.diffusion,model.reaction,observed))
        result['status']='complete'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
    result['seconds']=time.perf_counter()-start;json_write(destination,result)
    print(json.dumps(dict(event='baselines',id=dataset['id'],status=result['status'])),flush=True)
    return dataset['id'],result['status']


def train_task(dataset,retry=False):
    manifest=load_manifest();plan=manifest['plan']
    identity=dict(dataset=dataset,plan=plan,source_signature=source_signature(),runner_sha256=manifest['runner_sha256'])
    destination=ROOT/'evidence/training'/f"{dataset['id']}.json"
    prior=archive_failure(destination,retry)
    if prior is not None:
        if prior['identity']!=identity:raise ValueError('Training cache mismatch')
        return dataset['id'],'cached'
    out=ROOT/'outputs/runs'/dataset['id'];out.mkdir(parents=True,exist_ok=True)
    result=dict(identity=identity,id=dataset['id'],status='running',stages={})
    json_write(destination,result);start=time.perf_counter()
    print(json.dumps(dict(event='training_started',id=dataset['id'],adam_budget=plan['adam_budget'])),flush=True)
    try:
        observed=observations(dataset);case=extra_case('case_exp')
        milestones=[epoch for epoch in (150,500,1500,3000,6000) if epoch<=plan['adam_budget']]
        model,history,snapshots=train(observed,case.value_range,epochs=plan['adam_budget'],init_seed=dataset['seed'],
            checkpoints=milestones,out_dir=out/'adam')
        tail=tail_diagnostic(history,window=min(plan['tail_window'],len(history)))
        extend=plan['adam_extension_budget']>plan['adam_budget'] and tail['relative_decline']>plan['extension_relative_threshold']
        result['initial_budget_tail']=tail;result['extended']=extend
        print(json.dumps(dict(event='adam_initial_budget',id=dataset['id'],tail=tail,extended=extend)),flush=True)
        final_epoch=plan['adam_extension_budget'] if extend else plan['adam_budget']
        if extend:
            milestones=sorted(set(milestones+[plan['adam_budget'],final_epoch]))
            model,history,snapshots=train(observed,case.value_range,epochs=final_epoch,init_seed=dataset['seed'],
                checkpoints=milestones,out_dir=out/'adam')
        result['final_adam_epoch']=final_epoch
        result['final_adam_tail']=tail_diagnostic(history,window=min(plan['tail_window'],len(history)))
        result['final_adam_objective']=objective_diagnostic(model,objective_state(model,observed,case.value_range))
        result['adam_budget_state']=('declining_at_maximum_budget' if result['final_adam_tail']['relative_decline']>.01
                                    else 'trailing_window_decline_below_one_percent')
        grid=np.linspace(*case.value_range,256);support=np.linspace(*case.value_range,512)
        for epoch in sorted(snapshots):
            saved=torch.load(out/'adam'/f'checkpoint_{epoch}.pt',map_location='cpu',weights_only=False)
            current=ReactionDiffusionClosure(input_range=case.value_range)
            current.load_state_dict(saved['model'])
            label=f'neural_adam_{epoch}'
            result['stages'][label]=export_stage(current,label,out,grid,support,case,observed,dataset)
        print(json.dumps(dict(event='adam_finished',id=dataset['id'],final_epoch=final_epoch,
                              tail=result['final_adam_tail'])),flush=True)
        refined,details=lbfgs_crosscheck(model,observed,case.value_range,out/'lbfgs',
            dict(id=dataset['id'],campaign_identity=identity,initial_adam_epoch=final_epoch),
            outer_steps=plan['lbfgs_outer_steps'],inner_iterations=plan['lbfgs_inner_iterations'])
        result['lbfgs']=details
        result['stages']['neural_lbfgs_float64']=export_stage(refined,'neural_lbfgs_float64',out,grid,support,case,observed,dataset)
        result['status']='complete'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
    result['seconds']=time.perf_counter()-start;json_write(destination,result)
    print(json.dumps(dict(event='training_finished',id=dataset['id'],status=result['status'],seconds=result['seconds'])),flush=True)
    return dataset['id'],result['status']


def export_stage(model,label,out,grid,support,case,observed,dataset):
    native=NumpyMLPClosure.from_model(model);verification=native.verify(model,case.value_range)
    native_path=out/f'{label}.npz';native.save(native_path)
    symbols=fit_symbols(support,native.diffusion(support),native.reaction(support))
    frozen=(symbols[0].evaluate,symbols[1].evaluate)
    d=native.diffusion(grid);r=native.reaction(grid)
    npz_write(out/f'{label}_curves.npz',grid=grid,true_D=case.diffusion(grid),true_R=case.reaction(grid),
        neural_D=d,neural_R=r,frozen_D=frozen[0](grid),frozen_R=frozen[1](grid))
    compression={}
    for branch,truth,teacher,symbol in [('D',case.diffusion(grid),d,frozen[0](grid)),('R',case.reaction(grid),r,frozen[1](grid))]:
        en=np.linalg.norm(teacher-truth);ec=np.linalg.norm(symbol-teacher);es=np.linalg.norm(symbol-truth)
        compression[branch]=dict(neural_absolute_error=float(en),compression_absolute_error=float(ec),
            symbolic_absolute_error=float(es),rho=float(ec/en) if en else None,BIR=float(es/en) if en else None)
    return dict(kind='native_mlp',native_file=str(native_path.relative_to(ROOT)),native_file_sha256=digest(native_path),
        export_verification=verification,metrics=curve_metrics(case,grid,native.diffusion,native.reaction,observed),
        frozen_metrics=curve_metrics(case,grid,*frozen,observed),symbols=[dataclasses.asdict(symbol) for symbol in symbols],
        compression=compression,observation_file_sha256=dataset['observation_file_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','baselines','train'],required=True)
    parser.add_argument('--observation-root',type=Path);parser.add_argument('--refit-root',type=Path)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--seed',type=int)
    parser.add_argument('--setting',choices=['clean','noise3']);parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--adam-budget',type=int,default=6000);parser.add_argument('--adam-extension-budget',type=int,default=20000)
    parser.add_argument('--lbfgs-outer-steps',type=int,default=15)
    args=parser.parse_args()
    if args.workers<1:parser.error('Workers must be positive')
    if args.mode=='prepare':prepare(args);return
    manifest=load_manifest()
    datasets=[dataset for dataset in manifest['datasets'] if (args.seed is None or dataset['seed']==args.seed)
              and (args.setting is None or dataset['setting']==args.setting)]
    worker=baseline_task if args.mode=='baselines' else train_task
    if args.workers==1:
        report_completed(worker(dataset,args.retry_failed) for dataset in datasets)
    else:
        with process_pool(args.workers) as pool:
            futures=[pool.submit(worker,dataset,args.retry_failed) for dataset in datasets]
            report_completed(future.result() for future in futures)


if __name__=='__main__':
    main()
