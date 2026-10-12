"""Expanded frozen-surrogate PySR comparison; observation refits are separate."""
import argparse
import resource
import json,time
import numpy as np
import traceback
import os,sys
import sysconfig
from pathlib import Path
from project_paths import ROOT, INPUT_ROOT, input_glob
from artifact_io import json_write, load_curves, npz_write, model_signature, source_signature, report_completed
TASK_ROOT=ROOT
SEARCH_PROTOCOL=dict(binary_operators=['+','-','*','/'],unary_operators=['exp'],
    elementwise_loss='loss(prediction, target) = (prediction-target)^2',model_selection='best',parsimony=1e-4,
    deterministic=True,parallelism='serial',niterations=40,populations=8,population_size=48,maxsize=30,
    timeout_in_seconds=180,stopping='first of iteration count or timeout; no truth/error stopping',support_points=512)
os.environ.setdefault('JULIA_DEPOT_PATH',str(TASK_ROOT/'outputs/julia'))
os.environ.setdefault('PYTHON_JULIAPKG_PROJECT',str(TASK_ROOT/'outputs/julia-project'))
os.environ.setdefault('JULIA_NUM_THREADS','1')
os.environ.setdefault('JULIA_NUM_PRECOMPILE_TASKS','1')
libname=sysconfig.get_config_var('LDLIBRARY')
if libname:
    candidates=[Path(sysconfig.get_config_var('LIBDIR') or sys.base_prefix)/libname,
                Path(sys.executable).resolve().parents[1]/'lib'/libname,
                Path(sys.base_prefix)/'lib'/libname]
    for python_lib in candidates:
        if python_lib.exists():
            os.environ.setdefault('JULIA_PYTHONCALL_LIB',str(python_lib));break


def run_one(path,args):
    row=json.loads(path.read_text());ident=row['id'];dest=ROOT/'evidence/pysr'/f'{ident}.json'
    if row['status']!='complete':raise ValueError(f'Incomplete neural input: {ident}')
    if dest.exists():
        prior=json.loads(dest.read_text())
        if prior['config']!=row['config']:
            raise ValueError(f'PySR cached configuration mismatch: {ident}; use a distinct work root')
        if prior['status']!='complete':
            if not getattr(args,'retry_failed',False):return 'cached_'+prior['status']
            archive=ROOT/'evidence/failures/pysr'/f'{ident}_{time.time_ns()}.json'
            archive.parent.mkdir(parents=True,exist_ok=True);dest.replace(archive)
        else:
            if prior.get('source_signature')!=source_signature() or prior.get('model_signature')!=model_signature(row):
                raise ValueError(f'PySR cached source/model signature mismatch: {ident}; use a distinct work root')
            settings=prior.get('settings',{});seed=row['config']['seed']
            expected=dict(SEARCH_PROTOCOL,seed_D=seed,seed_R=seed+101)
            if not settings.get('version') or {k:v for k,v in settings.items() if k!='version'}!=expected:
                raise ValueError(f'PySR cached search budget/protocol mismatch: {ident}')
            if not {'D','R'}.issubset(prior.get('metrics',{})) or not {'D','R'}.issubset(prior.get('expressions',{})):
                raise ValueError(f'Incomplete PySR cache record: {ident}')
            for branch in ('D','R'):
                metrics=prior['metrics'][branch];expression=prior['expressions'][branch]
                if not all(key in metrics and np.isfinite(metrics[key]) for key in ('true_error','compression','BIR','common_norm_bound')):
                    raise ValueError(f'Invalid PySR cache metrics: {ident}/{branch}')
                if not expression.get('expression') or not np.isfinite(expression.get('fit_loss',np.nan)):
                    raise ValueError(f'Invalid PySR cache expression: {ident}/{branch}')
            return 'cached'
    start=time.perf_counter();cpu=time.process_time();result=dict(id=ident,config=row['config'],classification='reviewer_experiment',status='running',source_signature=source_signature())
    try:
        result['model_signature']=model_signature(row)
        # Only new fits initialize Julia, before importing the Torch runtime.
        import pysr
        from pysr import PySRRegressor
        import revision_core as core
        globals().update({name:getattr(core,name) for name in ['extra_case','relative_l2_error','strict_rollout']})
        import sympy as sp
        c=extra_case(row['config']['case']);seed=row['config']['seed'];out=ROOT/'outputs/runs'/ident/'pysr';out.mkdir(parents=True,exist_ok=True)
        curves=load_curves(ident,epoch=row['config']['epochs']);grid=curves['grid'].astype(float);evaluation=np.linspace(*c.value_range,256)
        result['settings']=dict(SEARCH_PROTOCOL,version=pysr.__version__,seed_D=seed,seed_R=seed+101)
        result['expressions']={};result['metrics']={};preds={}
        for branch in ['D','R']:
            reg=PySRRegressor(model_selection='best',binary_operators=['+','-','*','/'],unary_operators=['exp'],
                elementwise_loss='loss(prediction, target) = (prediction-target)^2',niterations=40,populations=8,population_size=48,
                maxsize=30,timeout_in_seconds=180,parsimony=1e-4,random_state=seed+(101 if branch=='R' else 0),
                deterministic=True,parallelism='serial',progress=False,verbosity=0,output_directory=str(out/branch))
            fit_start=time.perf_counter();reg.fit(grid[:,None],curves[branch].astype(float))
            best=reg.get_best();expr=reg.sympy();prediction=reg.predict(evaluation[:,None]);neural=np.interp(evaluation,grid,curves[branch])
            if not np.all(np.isfinite(prediction)):raise FloatingPointError('Nonfinite PySR predictions')
            truth=c.diffusion(evaluation) if branch=='D' else c.reaction(evaluation)
            en=np.linalg.norm(truth-neural);ec=np.linalg.norm(prediction-neural);es=np.linalg.norm(truth-prediction)
            result['expressions'][branch]=dict(expression=str(expr),complexity=int(best['complexity']),fit_loss=float(best['loss']),seconds=time.perf_counter()-fit_start)
            result['metrics'][branch]=dict(true_error=relative_l2_error(truth,prediction),compression=relative_l2_error(neural,prediction),
                BIR=float(es/en),common_norm_bound=float(ec/en))
            preds[branch]=prediction
            fn=sp.lambdify(sp.Symbol('x0'),expr,'numpy')
            preds[branch+'_callable']=lambda z,f=fn:np.broadcast_to(f(z),np.shape(z)).astype(float)
        from closure_discovery.data_generation.rd_solver_1d import random_fourier_initial_condition
        rng=np.random.default_rng(seed+1000);x=np.arange(64)/64
        initial=np.stack([random_fourier_initial_condition(x,rng,c.value_range,4) for _ in range(4)])
        times,truth,ti=strict_rollout(c.diffusion,c.reaction,initial,t_final=.5)
        if ti['status']!='stable':raise FloatingPointError('Truth rollout did not complete')
        tp,pred,info=strict_rollout(preds['D_callable'],preds['R_callable'],initial,t_final=.5)
        result['rollout']=dict(times=tp,errors=[relative_l2_error(truth[i],pred[i]) for i in range(min(len(times),len(tp)))],info=info)
        preds.pop('D_callable');preds.pop('R_callable')
        npz_write(out/'curves.npz',grid=evaluation,**preds)
        result['status']='complete'
    except Exception as e:result['status']='failed';result['error']=str(e);result['traceback']=traceback.format_exc()
    result.update(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    json_write(dest,result);return result['status']


def main():
    p=argparse.ArgumentParser(epilog='Common paths: --work-root PATH (output/cache root), --input-root PATH (explicit artifact input root); defaults use the project directory.');p.add_argument('--limit',type=int);p.add_argument('--case',default='all');p.add_argument('--watch',action='store_true');p.add_argument('--expected',type=int);p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    if a.limit is not None and a.limit<1:p.error('--limit must be positive')
    if a.expected is not None and a.expected<1:p.error('--expected must be positive')
    if a.expected is None:a.expected=30 if a.case=='all' else 10
    if a.watch:
        completed=set()
        while len(completed)<a.expected:
            for path in input_glob('evidence/raw', 'main_*.json'):
                row=json.loads(path.read_text());c=row['config']
                if row['status']=='complete' and c['setting'] in ['clean','noise3'] and (a.case=='all' or c['case']==a.case) and path.stem not in completed:
                    status=run_one(path,a);report_completed([(path.stem,status)]);completed.add(path.stem)
            json_write(ROOT/'evidence/manifest_pysr.json',dict(ids=sorted(completed),expected=a.expected,niterations=40,
                timeout_per_branch_seconds=180,scope='Frozen surrogates; +,-,*,/,exp; no truth stopping'))
            if len(completed)<a.expected:time.sleep(5)
        return
    paths=[]
    for path in input_glob('evidence/raw', 'main_*.json'):
        row=json.loads(path.read_text());c=row['config']
        if row['status']=='complete' and c['setting'] in ['clean','noise3'] and (a.case=='all' or c['case']==a.case):paths.append(path)
    if a.limit:paths=paths[:a.limit]
    json_write(ROOT/'evidence/manifest_pysr.json',dict(ids=[p.stem for p in paths],niterations=40,timeout_per_branch_seconds=180,scope='frozen surrogate, no bias correction'))
    if not paths:raise ValueError('No completed matching neural inputs were found')
    report_completed((path.stem,run_one(path,a)) for path in paths)


if __name__=='__main__':main()
