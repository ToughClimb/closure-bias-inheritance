"""Resumable, process-parallel predeclared reviewer experiments.

Run from the project directory. Each task saves its configuration before fitting,
checkpoints/histories to outputs, and a compact raw JSON on success or failure.
"""
import argparse
import concurrent.futures as cf
import copy
import hashlib
import resource
import traceback
from revision_core import *


SETTINGS=dict(clean=(0.,1,1),noise1=(.01,1,1),noise2=(.02,1,1),noise3=(.03,1,1),noise5=(.05,1,1),
              sparse=(0.,2,2),coarse16=(0.,4,2),coarse8=(0.,8,2))
RAW_CACHE={}


def task_configs(group):
    tasks=[]
    if group in ('main','all'):
        for case in ['case_a','case_b','case_exp']:
            for setting in ['clean','noise1','noise3','noise5','sparse']:
                for seed in range(5):tasks.append(dict(group='main',case=case,setting=setting,seed=seed,epochs=150))
    if group in ('noise_knee','all'):
        # Reviewer-requested resolution of the 1–3% knee; no parameter search.
        for seed in range(5):tasks.append(dict(group='noise_knee',case='case_exp',setting='noise2',seed=seed,epochs=150))
    if group in ('ablation','all'):
        # Individual removal, cumulative addition, and one-at-a-time x0.1/x10.
        variants={}
        for k in DEFAULT_WEIGHTS:
            w=DEFAULT_WEIGHTS.copy();w[k]=0.;variants['drop_'+k]=w
        w={k:0. for k in DEFAULT_WEIGHTS};w['weak']=1.;variants['weak_only']=w.copy()
        for k in ['rollout','mass','strong','anchor','reg']:
            w[k]=DEFAULT_WEIGHTS[k];variants['add_through_'+k]=w.copy()
        for k in DEFAULT_WEIGHTS:
            for scale in [.1,10.]:
                w=DEFAULT_WEIGHTS.copy();w[k]*=scale;variants[f'{k}_x{scale:g}']=w
        # Full hybrid add_through_reg exactly equals main: reuse main, skip duplicate.
        variants.pop('add_through_reg')
        for seed in range(5):
            for variant,w in variants.items():
                for setting in ['clean','noise3']:
                    tasks.append(dict(group='ablation',case='case_exp',setting=setting,seed=seed,epochs=150,variant=variant,weights=w))
    if group in ('convergence','all'):
        for seed in range(5):
            for restart in range(3):
                tasks.append(dict(group='convergence',case='case_exp',setting='clean',seed=seed,epochs=1500,
                    init_seed=seed+100*restart,checkpoints=[150,500,1500],variant=f'restart{restart}'))
    if group in ('families','all'):
        for case in ['case_c','case_trig','case_tanh']:
            for setting in ['clean','noise3']:
                for seed in range(5):tasks.append(dict(group='families',case=case,setting=setting,seed=seed,epochs=150))
    if group in ('coarse','all'):
        # Retire the invalid four-mode nx8 arm. Resolved nx8 remains in
        # coarse_corrected; historical invalid outputs remain read-only inputs.
        for setting in ['coarse16']:
            for seed in range(5):tasks.append(dict(group='coarse',case='case_exp',setting=setting,seed=seed,epochs=150))
    if group in ('coarse_corrected','all'):
        for seed in range(5):tasks.append(dict(group='coarse_corrected',case='case_exp',setting='coarse8',seed=seed,epochs=150,num_test_modes=3))
    if group in ('weak','all'):
        for setting in ['clean','noise1','noise3','noise5']:
            for seed in range(5):
                for variant in ['space_weak_reduced','space_time_weak']:
                    w=DEFAULT_WEIGHTS.copy();w['strong']=0.;w['mass']=0.
                    tasks.append(dict(group='weak',case='case_exp',setting=setting,seed=seed,epochs=150,variant=variant,
                        weights=w,weak_variant='space_time_weak' if variant=='space_time_weak' else 'space_weak'))
    if group in ('support','all'):
        for setting in ['clean','noise3']:
            for seed in range(5):tasks.append(dict(group='support',case='case_exp',setting=setting,seed=seed,epochs=150,
                amplitude_range=[.2,.8]))
    if group in ('paired','all'):
        for setting in ['clean','noise3','noise5']:
            for seed in range(5):
                for variant in ['space_time_rectangle','space_time_trapezoid']:
                    w=DEFAULT_WEIGHTS.copy();w['strong']=0.;w['mass']=0.
                    tasks.append(dict(group='paired',case='case_exp',setting=setting,seed=seed,epochs=150,
                        variant=variant,weak_variant=variant,weights=w))
    if group=='secondary':
        from itertools import zip_longest
        streams=[task_configs(g) for g in ['ablation','families','weak','support','coarse']]
        for batch in zip_longest(*streams):tasks.extend(t for t in batch if t is not None)
        tasks.extend(task_configs('convergence'))
    return tasks


def task_id(config):
    return '_'.join(str(config.get(k,'default')) for k in ['group','case','setting','seed','variant'])


def work(config,retry_failed=False):
    allowed={'group','case','setting','seed','epochs','init_seed','checkpoints','variant','weights','weak_variant','amplitude_range','num_test_modes','classification'}
    unknown=set(config)-allowed
    if unknown:raise ValueError(f'Unsupported task configuration fields: {sorted(unknown)}')
    if config['epochs']<1:raise ValueError('epochs must be positive')
    ident=task_id(config);out=ROOT/'outputs/runs'/ident;raw_file=ROOT/'evidence/raw'/f'{ident}.json'
    if raw_file.exists():
        prior=json.loads(raw_file.read_text())
        if prior['config'] != config or prior.get('device','cpu')!=str(resolve_device()):
            raise ValueError(f'Cached configuration mismatch for {ident} (including device); choose a distinct work root or task variant')
        if prior.get('source_signature')!=source_signature():
            raise ValueError(f'Cached source signature mismatch for {ident}; retain historical results and use a distinct work root')
        if prior['status']=='complete':
            required=set(config.get('checkpoints',[]))|{config['epochs']}
            if not required.issubset(map(int,prior.get('epochs',{}))):
                raise ValueError(f'Cached task {ident} is missing required final epochs')
            if not (out/'history.json').exists() or any(not (out/f'curves_{e}.npz').exists() for e in required):
                raise ValueError(f'Cached task {ident} is missing required training exports')
            validate_history(json.loads((out/'history.json').read_text()),config['epochs'])
            for epoch in required:
                with np.load(out/f'curves_{epoch}.npz') as values:
                    if not all(key in values and np.isfinite(values[key]).all() for key in ('grid','D','R')):
                        raise ValueError(f'Cached task {ident} has invalid curve exports')
            if prior.get('controls') and not (out/'all_curves.npz').exists():
                raise ValueError(f'Cached task {ident} is missing control curves')
        if prior['status']=='failed' and retry_failed:
            archive=ROOT/'evidence/failures'/f'{ident}_{time.time_ns()}.json'
            archive.parent.mkdir(parents=True,exist_ok=True);raw_file.replace(archive)
        else:
            return ident,'cached' if prior['status']=='complete' else 'cached_'+prior['status']
    start=time.perf_counter();cpu_start=time.process_time();out.mkdir(parents=True,exist_ok=True);json_write(out/'config.json',config)
    result=dict(id=ident,config=config,classification=config.get('classification','reviewer_experiment'),status='running',source_signature=source_signature())
    try:
        case=extra_case(config['case']);seed=config['seed'];noise,ss,ts=SETTINGS[config['setting']]
        cfg=SimulationConfig1D(nx=64,t_final=.1,save_every=10)
        amplitude=tuple(config.get('amplitude_range',case.value_range))
        cache_key=(case.name,seed,amplitude)
        if cache_key not in RAW_CACHE:RAW_CACHE[cache_key]=generate_dataset(case,cfg,8,seed=seed,amplitude_range=amplitude,initial_clip_range=amplitude)
        raw=RAW_CACHE[cache_key]
        observed=apply_observation_model(raw,noise_level=noise,space_stride=ss,time_stride=ts,seed=seed,clip_range=case.value_range)
        result['data_hash']=hashlib.sha256(observed['u'].tobytes()).hexdigest()
        result['observation']=observation_diagnostics(raw,observed,case,seed,noise)
        effective_modes=min(config.get('num_test_modes',4),(len(observed['x'])-1)//2)
        result['test_functions']=dict(requested_modes=config.get('num_test_modes',4),effective_modes=effective_modes,policy='resolved_fourier')
        model,history,snaps=train(observed,case.value_range,epochs=config['epochs'],init_seed=config.get('init_seed',seed),
            weights=config.get('weights'),variant=config.get('weak_variant','space_weak'),checkpoints=config.get('checkpoints',[]),out_dir=out,num_test_modes=effective_modes)
        if not (set(config.get('checkpoints',[]))|{config['epochs']}).issubset(snaps):
            raise ValueError('Training returned incomplete final snapshots')
        result['device']=str(next(model.parameters()).device)
        result['torch_version']=torch.__version__
        if result['device'].startswith('cuda'):result['gpu_peak_allocated_mb']=torch.cuda.max_memory_allocated()/2**20
        result['epochs']={}
        for epoch,snap in snaps.items():
            g=np.linspace(*case.value_range,256);D=lambda z:np.interp(z,snap['grid'],snap['D']);R=lambda z:np.interp(z,snap['grid'],snap['R'])
            symbols=fit_symbols(snap['grid'],snap['D'],snap['R'])
            SD=symbols[0].evaluate;SR=symbols[1].evaluate
            metrics=curve_metrics(case,g,D,R,observed);sm=curve_metrics(case,g,SD,SR,observed)
            compression={}
            for key,true,n,s in [('D',case.diffusion(g),D(g),SD(g)),('R',case.reaction(g),R(g),SR(g))]:
                en=np.linalg.norm(n-true);ec=np.linalg.norm(s-n);es=np.linalg.norm(s-true)
                compression[key]=dict(BIR=float(es/en),bound=float(ec/en),compression_relative_neural=relative_l2_error(n,s),
                    observed_deviation=float(abs(es/en-1)),norm_conversion=float(np.linalg.norm(n)/np.linalg.norm(true)))
            result['epochs'][str(epoch)]=dict(neural=metrics,frozen=sm,symbols=[dataclasses.asdict(e) for e in symbols],
                compression=compression,training_seconds=snap['seconds'],last_pre_step_loss=history[epoch-1])
        # Controls and direct selected-family equation refits on main/family/coarse only.
        if config['group'] in ['main','noise_knee','families','coarse','coarse_corrected','support']:
            controls={};curves=dict(grid=g,true_D=case.diffusion(g),true_R=case.reaction(g),neural_D=D(g),neural_R=R(g),frozen_D=SD(g),frozen_R=SR(g))
            for name,fn in [('weak_poly',fit_weak_form_polynomial_closure),('strong_poly',fit_strong_form_polynomial_closure),('spacetime_poly',spacetime_polynomial)]:
                kwargs=dict(num_test_modes=effective_modes) if name!='strong_poly' else {}
                closure,mse=fn(observed,value_range=case.value_range,**kwargs)
                controls[name]=dict(metrics=curve_metrics(case,g,closure.diffusion,closure.reaction,observed),fit_mse=float(mse),
                    parameters_D=closure.diffusion_coefficients.tolist(),parameters_R=closure.reaction_coefficients.tolist())
                curves[name+'_D']=closure.diffusion(g);curves[name+'_R']=closure.reaction(g)
            for mode in ['weak','data']:
                d,r,info=refit_symbols(observed,case.value_range,symbols,mode=mode,num_test_modes=effective_modes)
                controls['refit_'+mode]=dict(metrics=curve_metrics(case,g,d,r,observed),fit=info)
                curves['refit_'+mode+'_D']=d(g);curves['refit_'+mode+'_R']=r(g)
            result['controls']=controls
            hist,bins=np.histogram(observed['u'],bins=32,range=case.value_range,density=True);curves.update(state_hist=hist,state_bins=bins)
            npz_write(out/'all_curves.npz',**curves)
            # Threshold sensitivity changes compression only, does not retrain/tune to truth.
            thresholds=[]
            dc=fit_diffusion_candidates(snap['grid'],snap['D']);rc=fit_reaction_candidates(snap['grid'],snap['R'])
            for rt in [0.,.01,.05,.2,1.]:
                for at in [0.,1e-10,1e-8,1e-6,1e-4]:
                    ed=select_symbolic_expression(dc,relative_tolerance=rt,absolute_tolerance=at)
                    er=select_symbolic_expression(rc,relative_tolerance=rt,absolute_tolerance=at)
                    sm=curve_metrics(case,g,ed.evaluate,er.evaluate,observed)
                    thresholds.append(dict(relative=rt,absolute=at,family_D=ed.family,family_R=er.family,
                        complexity_D=ed.complexity,complexity_R=er.complexity,BIR_D=sm['D']/metrics['D'],BIR_R=sm['R']/metrics['R']))
            result['thresholds']=thresholds
        result['status']='complete'
    except Exception as exc:
        result['status']='failed';result['error']=str(exc);result['traceback']=traceback.format_exc()
    result['seconds']=time.perf_counter()-start
    result['cpu_seconds']=time.process_time()-cpu_start
    result['peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    json_write(raw_file,result)
    return ident,result['status']


def main():
    p=argparse.ArgumentParser(epilog='Common paths: --work-root PATH (output/cache root), --input-root PATH (explicit artifact input root); defaults use the project directory.');p.add_argument('--group',choices=['main','noise_knee','ablation','convergence','families','coarse','coarse_corrected','weak','support','paired','secondary','all'],default='main');p.add_argument('--workers',type=int,default=3);p.add_argument('--limit',type=int);p.add_argument('--offset',type=int,default=0);p.add_argument('--manifest-only',action='store_true');p.add_argument('--config',type=Path,help='JSON tasks list or object with tasks; paths resolved from caller CWD');p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    if a.config:
        declared=json.loads(a.config.read_text());tasks=declared['tasks'] if isinstance(declared,dict) else declared
    else:tasks=task_configs(a.group)
    if a.offset:tasks=tasks[a.offset:]
    if a.limit:tasks=tasks[:a.limit]
    ids=[task_id(t) for t in tasks]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate task IDs would share checkpoints; choose distinct variants')
    json_write(ROOT/f'evidence/manifest_{a.group}.json',dict(tasks=tasks,workers=a.workers,seeds=list(range(5)),
        rule='Fixed observations; never choose settings, checkpoints, refits or seeds using true closure errors'))
    if a.manifest_only:return
    with process_pool(a.workers) as pool:
        futures=[pool.submit(work,t,a.retry_failed) for t in tasks]
        report_completed(f.result() for f in cf.as_completed(futures))


if __name__=='__main__':main()
