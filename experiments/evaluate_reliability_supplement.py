"""Audited forward validation and fresh-teacher observation refits.

Archived PySR checks retain archived teacher identities. Matched checks use the
fresh teacher, new baseline fits and precisely the saved campaign observations.
Violating trajectories and failed refits are retained, without state clipping.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'revision'))

import numpy as np
from artifact_io import json_write,npz_write,source_signature
from revision_core import (ROOT,extra_case,strict_rollout,curve_metrics,process_pool,
    report_completed)
from closure_discovery.data_generation.rd_solver_1d import random_fourier_initial_condition
from reliability_core import NumpyMLPClosure
from run_reliability_supplement import load_manifest,observations,digest
from reliability_templates import FixedFamilyTemplate,symbolic_pair,denominator_roots
from run_pysr_observation_refit import ExpressionTemplate,fit_to_observations


PATTERNS=dict(full_support=dict(kind='random',horizon=.5),
    fixed_midpoint=dict(kind='random_shifted',horizon=.5),
    uniform_zero=dict(kind='uniform',fraction=0.,horizon=.1),
    uniform_low=dict(kind='uniform',fraction=.02,horizon=.1),
    uniform_middle=dict(kind='uniform',fraction=.5,horizon=.1),
    uniform_upper=dict(kind='uniform',fraction=1.,horizon=.1),
    fourier_m1=dict(kind='fourier',mode=1,horizon=.1),
    fourier_m4=dict(kind='fourier',mode=4,horizon=.1))


def evaluation_identity():
    return dict(source_signature=source_signature(),evaluator_sha256=digest(__file__),
        templates_sha256=digest(Path(__file__).with_name('reliability_templates.py')),
        coefficient_fitter_sha256=digest(Path(__file__).with_name('run_pysr_observation_refit.py')),
        patterns=PATTERNS,save_dt=.01,range_tolerance=1e-10,
        rule='Track every accepted state and every RK trial stage; retain numerical and physical failures; no clipping')


def declare_evaluation():
    value=evaluation_identity();path=ROOT/'evidence/EVALUATION_PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Evaluation protocol/source changed')
    else:
        json_write(path,value)
        for source in (Path(__file__),Path(__file__).with_name('reliability_templates.py')):
            destination=ROOT/'provenance/source/experiments'/source.name
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    return value


def validation_initials(case,seed):
    x=np.arange(64)/64;rng=np.random.default_rng(seed+1000)
    shape=np.stack([random_fourier_initial_condition(x,rng,(0.,1.),4) for _ in range(4)])
    span=case.value_range[1]-case.value_range[0]
    midpoint=1.1 if case.name=='case_exp' else sum(case.value_range)/2
    half_range=.25 if case.name=='case_exp' else .2
    result={}
    for name,protocol in PATTERNS.items():
        if protocol['kind']=='random':initial=case.value_range[0]+shape*span
        elif protocol['kind']=='random_shifted':initial=midpoint+half_range*(2*shape-1)
        elif protocol['kind']=='uniform':initial=np.full((1,64),case.value_range[0]+protocol['fraction']*span)
        else:initial=(midpoint+.01*span*np.cos(2*np.pi*protocol['mode']*x))[None,:]
        result[name]=initial
    return result


def functions(descriptor):
    kind=descriptor['kind']
    if kind=='native_mlp':
        path=ROOT/descriptor['native_file']
        if digest(path)!=descriptor['native_file_sha256']:raise ValueError('Native teacher export changed')
        model=NumpyMLPClosure.load(path)
        return model.diffusion,model.reaction
    if kind=='polynomial':
        return tuple(lambda u,p=np.array(descriptor['parameters_'+branch]):np.polynomial.polynomial.polyval(u,p)
                     for branch in ('D','R'))
    if kind=='restricted':
        pair=symbolic_pair(descriptor['symbols']);return pair[0].evaluate,pair[1].evaluate
    if kind=='pysr':
        pair=[ExpressionTemplate.compile(descriptor['original_'+branch]) for branch in ('D','R')]
        return tuple(lambda u,t=template,p=np.array(descriptor['parameters_'+branch]):t.evaluate(p,u)
                     for branch,template in zip(('D','R'),pair,strict=True))
    raise ValueError(f'Unsupported closure descriptor: {kind}')


def endpoints(descriptor,case):
    D,R=functions(descriptor);grid=np.linspace(*case.value_range,4097)
    with np.errstate(all='ignore'):
        d=np.broadcast_to(np.asarray(D(grid),dtype=float),grid.shape)
        r=np.broadcast_to(np.asarray(R(grid),dtype=float),grid.shape)
    out=dict(sample_points=len(grid),evaluation_interval=case.value_range,
        R_lower=float(r[0]) if np.isfinite(r[0]) else None,
        R_upper=float(r[-1]) if np.isfinite(r[-1]) else None,
        all_sampled_values_finite=bool(np.isfinite(d).all() and np.isfinite(r).all()),
        sampled_min_D=float(d.min()) if np.isfinite(d).all() else None,
        sampled_max_D=float(d.max()) if np.isfinite(d).all() else None,
        sampled_positive_diffusion=bool(np.isfinite(d).all() and np.all(d>0)),
        lower_inward_reaction=bool(np.isfinite(r[0]) and r[0]>=-1e-8),
        upper_inward_reaction=bool(np.isfinite(r[-1]) and r[-1]<=1e-8),
        zero_equilibrium_residual=abs(float(r[0])) if np.isfinite(r[0]) else None,
        interpretation='Sampling is a diagnostic, not a global positivity or pole-free certificate')
    if descriptor['kind']=='restricted':
        out['detected_denominator_roots']={branch:denominator_roots(expression,case.value_range)
            for branch,expression in zip(('D','R'),symbolic_pair(descriptor['symbols']),strict=True)}
    return out


def validate_models(case,seed,models):
    result={};initials=validation_initials(case,seed)
    references={}
    for pattern,initial in initials.items():
        horizon=PATTERNS[pattern]['horizon']
        times,values,info=strict_rollout(case.diffusion,case.reaction,initial,t_final=horizon,state_range=case.value_range)
        if not info['numerical_completion'] or not info['accepted_state_range_valid'] or not info['rk_stage_state_range_valid']:
            raise FloatingPointError(f'Invalid reference trajectory: {pattern}')
        references[pattern]=(times,values,initial)
    for label,descriptor in models.items():
        D,R=functions(descriptor)
        entries={}
        for pattern,(true_times,truth,initial) in references.items():
            times,pred,info=strict_rollout(D,R,initial,t_final=PATTERNS[pattern]['horizon'],state_range=case.value_range)
            n=min(len(times),len(true_times))
            absolute=[float(np.sqrt(np.mean((pred[i]-truth[i])**2))) for i in range(n)]
            relative=[float(np.linalg.norm(pred[i]-truth[i])/np.linalg.norm(truth[i]))
                      if np.linalg.norm(truth[i])>0 else None for i in range(n)]
            complete=bool(info['numerical_completion'] and len(times)==len(true_times))
            entries[pattern]=dict(info=info,times=times,absolute_rms=absolute,relative_l2=relative,
                final_absolute_rms=absolute[-1] if complete else None,
                final_relative_l2=relative[-1] if complete else None,
                nonnegative_accepted_states=bool(info['accepted_state_min'] is not None and info['accepted_state_min']>=-1e-10),
                initial_min=float(initial.min()),initial_max=float(initial.max()),
                initial_condition_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                initial_trajectories=len(initial),error_rule='Absolute RMS for zero true solution; relative error undefined there')
            if PATTERNS[pattern]['kind']=='fourier':
                mode=PATTERNS[pattern]['mode'];projection=2*np.cos(2*np.pi*mode*np.arange(64)/64)/64
                entries[pattern].update(mode=mode,wavenumber=float(2*np.pi*mode),
                    predicted_modal_amplitude=np.einsum('tbs,s->tb',pred[:n],projection),
                    true_modal_amplitude=np.einsum('tbs,s->tb',truth[:n],projection),
                    predicted_spatial_mean=pred[:n].mean(-1),true_spatial_mean=truth[:n].mean(-1))
        result[label]=dict(descriptor=descriptor,endpoints=endpoints(descriptor,case),validation=entries)
    return result


def completed_cache(path,identity):
    if not path.exists():return None
    prior=json.loads(path.read_text())
    if prior['identity']!=identity:raise ValueError(f'Evaluation cache mismatch: {path}')
    if prior['status']!='complete':raise ValueError(f'Incomplete evaluation: {path}; inspect persisted failure')
    return prior


def archive_task(path):
    row=json.loads(path.read_text());case=extra_case(row['config']['case']);ident=row['id']
    identity=dict(**evaluation_identity(),archived_record_sha256=digest(path))
    destination=ROOT/'evidence/archive_validation'/f'{ident}.json'
    if completed_cache(destination,identity) is not None:return ident,'cached'
    start=time.perf_counter();result=dict(identity=identity,id=ident,config=row['config'],status='running')
    try:
        if row['status']!='complete':raise ValueError('Archive input is not complete')
        templates={branch:row['templates'][branch]['original_expression'] for branch in ('D','R')}
        originals=[ExpressionTemplate.compile(templates[branch]) for branch in ('D','R')]
        models={}
        for name in ('pysr_frozen','pysr_refit'):
            models[name]=dict(kind='pysr',original_D=templates['D'],original_R=templates['R'],
                parameters_D=originals[0].parameters.tolist() if name=='pysr_frozen' else row['fit']['parameters_D'],
                parameters_R=originals[1].parameters.tolist() if name=='pysr_frozen' else row['fit']['parameters_R'],
                teacher_identity='Archived neural teacher; no relabeling as fresh same-observation teacher',
                archived_observation_u_sha256=row['archived_data_hash'],refit_observation_u_sha256=row['data_hash'])
        result['models']=validate_models(case,row['config']['seed'],models)
        result['status']='complete'
    except Exception as error:result.update(status='failed',error=str(error),traceback=traceback.format_exc())
    result['seconds']=time.perf_counter()-start;json_write(destination,result)
    print(json.dumps(dict(event='archive_audit',id=ident,status=result['status'],seconds=result['seconds'])),flush=True)
    return ident,result['status']


def fresh_refit(dataset,stage):
    identity=dict(**evaluation_identity(),dataset=dataset,teacher=stage,
        protocol=load_manifest()['plan']['coefficient_refit'])
    destination=ROOT/'evidence/fresh_refits'/f"{dataset['id']}.json"
    if destination.exists():
        prior=json.loads(destination.read_text())
        if prior['identity']!=identity:raise ValueError('Fresh refit cache mismatch')
        return prior
    start=time.perf_counter();result=dict(identity=identity,id=dataset['id'],status='running',phases=[])
    try:
        observed=observations(dataset);case=extra_case('case_exp');pair=symbolic_pair(stage['symbols'])
        templates=[FixedFamilyTemplate(expression) for expression in pair]
        parameters,fit=fit_to_observations(observed,case.value_range,templates,max_nfev=40)
        result['phases'].append(fit);continued=not fit['success'] and fit['status']==0
        split=len(pair[0].parameters)
        if continued:
            warmed=[FixedFamilyTemplate(pair[0],parameters[:split]),FixedFamilyTemplate(pair[1],parameters[split:])]
            parameters,fit=fit_to_observations(observed,case.value_range,warmed,max_nfev=80)
            result['phases'].append(fit)
        teacher_D,teacher_R=functions(stage);support=np.linspace(*case.value_range,512)
        final_pair=[]
        for original,template,p,teacher in zip(pair,templates,(parameters[:split],parameters[split:]),(teacher_D,teacher_R),strict=True):
            mse=float(np.mean((template.evaluate(p,support)-teacher(support))**2))
            final_pair.append(dataclasses.replace(original,parameters=tuple(p),expression=template.expression(p),fit_mse=mse))
        result.update(status='complete',continued=continued,model=dict(kind='restricted',
            symbols=[dataclasses.asdict(expression) for expression in final_pair],
            teacher_native_sha256=stage['native_file_sha256'],observation_file_sha256=dataset['observation_file_sha256'],
            parameter_policy='All coefficient slots and rates in the selected family are independent; family and degrees fixed; no reaction anchor'),
            totals={key:sum(phase[key] for phase in result['phases'])
                    for key in ('nfev','residual_evaluations','solver_rhs_evaluations','failed_trial_evaluations')},
            optimizer_success=fit['success'],optimizer_status=fit['status'])
    except Exception as error:result.update(status='failed',error=str(error),traceback=traceback.format_exc())
    result['seconds']=time.perf_counter()-start;json_write(destination,result)
    print(json.dumps(dict(event='fresh_refit',id=dataset['id'],status=result['status'],seconds=result['seconds'])),flush=True)
    return result


def matched_task(dataset):
    training_path=ROOT/'evidence/training'/f"{dataset['id']}.json"
    baseline_path=ROOT/'evidence/baselines'/f"{dataset['id']}.json"
    training=json.loads(training_path.read_text());baseline=json.loads(baseline_path.read_text())
    identity=dict(**evaluation_identity(),dataset=dataset,training_sha256=digest(training_path),baseline_sha256=digest(baseline_path))
    destination=ROOT/'evidence/matched_validation'/f"{dataset['id']}.json"
    if completed_cache(destination,identity) is not None:return dataset['id'],'cached'
    result=dict(identity=identity,id=dataset['id'],config=dict(case='case_exp',setting=dataset['setting'],seed=dataset['seed']),status='running')
    start=time.perf_counter()
    try:
        if training['status']!='complete' or baseline['status']!='complete':raise ValueError('Matched inputs incomplete')
        observed=observations(dataset);case=extra_case('case_exp');grid=np.linspace(*case.value_range,256)
        models={label:{**descriptor} for label,descriptor in baseline['models'].items()}
        labels=['neural_adam_150',f"neural_adam_{training['final_adam_epoch']}",'neural_lbfgs_float64']
        roles={labels[0]:'neural_150',labels[1]:'neural_adam_final',labels[2]:'neural_lbfgs_float64'}
        for label in labels:
            stage=training['stages'][label];role=roles[label]
            models[role]=stage
            frozen_role=role.replace('neural','frozen',1)
            models[frozen_role]=dict(kind='restricted',symbols=stage['symbols'],
                teacher_native_sha256=stage['native_file_sha256'],
                observation_file_sha256=dataset['observation_file_sha256'])
        refit=fresh_refit(dataset,training['stages']['neural_adam_150'])
        result['fresh_refit_status']=refit['status'];result['fresh_refit_record_sha256']=digest(ROOT/'evidence/fresh_refits'/f"{dataset['id']}.json")
        if refit['status']=='complete':models['refit_150']=refit['model']
        else:result['fresh_refit_error']=refit['error']
        result['closure_metrics']={label:curve_metrics(case,grid,*functions(descriptor),observed)
                                   for label,descriptor in models.items()}
        result['models']=validate_models(case,dataset['seed'],models)
        result['observation_file_sha256']=dataset['observation_file_sha256']
        result['status']='complete'
    except Exception as error:result.update(status='failed',error=str(error),traceback=traceback.format_exc())
    result['seconds']=time.perf_counter()-start;json_write(destination,result)
    print(json.dumps(dict(event='matched_audit',id=dataset['id'],status=result['status'],seconds=result['seconds'])),flush=True)
    return dataset['id'],result['status']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['archive','matched'],required=True)
    parser.add_argument('--refit-root',type=Path);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int);parser.add_argument('--setting',choices=['clean','noise3'])
    parser.add_argument('--limit',type=int)
    args=parser.parse_args();load_manifest();declare_evaluation()
    if args.workers<1:parser.error('Workers must be positive')
    if args.mode=='archive':
        if args.refit_root is None:parser.error('Archive audit requires --refit-root')
        tasks=sorted((args.refit_root/'evidence/pysr_observation_refit').glob('*.json'))
        if len(tasks)!=30:raise ValueError('Archive audit expects all 30 final PySR inputs')
        worker=archive_task
    else:
        tasks=[dataset for dataset in load_manifest()['datasets']
            if (args.seed is None or dataset['seed']==args.seed) and (args.setting is None or dataset['setting']==args.setting)]
        worker=matched_task
    if args.limit:tasks=tasks[:args.limit]
    if args.workers==1:report_completed(worker(task) for task in tasks)
    else:
        with process_pool(args.workers) as pool:
            futures=[pool.submit(worker,task) for task in tasks]
            report_completed(future.result() for future in futures)


if __name__=='__main__':main()
