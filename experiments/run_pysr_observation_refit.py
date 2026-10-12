"""Refit floating coefficients of archived PySR expressions to observations.

The expression structure, integer powers and exact repeated-constant sharing are
fixed. Truth laws are used only for data generation and subsequent evaluation.
No Julia search, truth-error stopping, best-seed selection or model-family search
is performed here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'revision'))
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from revision_core import (ROOT, INPUT_ROOT, input_path, input_glob, source_signature,
    extra_case, generate_dataset, SimulationConfig1D, apply_observation_model,
    curve_metrics, strict_rollout, relative_l2_error, json_write, npz_write,
    process_pool, report_completed, model_signature, observation_diagnostics)
from run_suite import SETTINGS
from closure_discovery.data_generation.rd_solver_1d import random_fourier_initial_condition
from closure_discovery.data_generation.validation import uniform_spacing


@dataclass
class ExpressionTemplate:
    original: str
    symbolic: object
    constants: tuple
    coefficient_symbols: tuple
    function: object

    @classmethod
    def compile(cls,expression):
        u=sp.Symbol('x0');parsed=sp.sympify(expression)
        if parsed.free_symbols-{u}:raise ValueError('Expressions must use x0 as their only state variable')
        constants=tuple(sorted(parsed.atoms(sp.Float),key=sp.srepr))
        symbols=tuple(sp.Symbol(f'coefficient_{i}') for i in range(len(constants)))
        symbolic=parsed.xreplace(dict(zip(constants,symbols,strict=True)))
        function=sp.lambdify((u,*symbols),symbolic,'numpy',cse=True)
        return cls(expression,symbolic,constants,symbols,function)

    @property
    def parameters(self):return np.array([float(c) for c in self.constants])

    def evaluate(self,parameters,u):
        if len(parameters)!=len(self.constants):raise ValueError('Coefficient count does not match the fixed expression')
        with np.errstate(all='ignore'):
            result=self.function(np.asarray(u,dtype=float),*parameters)
        return np.broadcast_to(np.asarray(result,dtype=float),np.shape(u))

    def expression(self,parameters):
        return str(self.symbolic.xreplace({s:sp.Float(float(v),17) for s,v in zip(self.coefficient_symbols,parameters,strict=True)}))

    def metadata(self):
        return dict(original_expression=self.original,parameter_count=len(self.constants),
            initial_parameters=self.parameters.tolist(),constants=[str(c) for c in self.constants],
            policy='Refit distinct floating literals; exact repeated literals share a coefficient; integer constants and powers remain fixed')


def fit_to_observations(observed,value_range,templates,*,max_nfev=40,max_rhs_evaluations=20000):
    """Fixed-budget forward RK45 coefficient fit without any truth-law inputs."""
    u=np.asarray(observed['u'],dtype=float);times=np.asarray(observed['t'],dtype=float)
    if u.ndim!=3 or not np.isfinite(u).all():raise ValueError('Expected finite [batch,time,space] observations')
    dx=uniform_spacing(observed['x'],'space');uniform_spacing(times,'time',minimum=3)
    if u.shape[1]!=len(times) or u.shape[-1]!=len(observed['x']):raise ValueError('Observation dimensions do not match coordinates')
    if max_nfev<1 or max_rhs_evaluations<1:raise ValueError('Fit budgets must be positive')
    indices=np.arange(0,len(times),max(1,(len(times)-1)//10))
    if indices[-1]!=len(times)-1:indices=np.r_[indices,len(times)-1]
    saved_times=times[indices]-times[0];target=u[:,indices];shape=u[:,0].shape
    support=np.linspace(*value_range,512);split=len(templates[0].constants)
    initial=np.r_[templates[0].parameters,templates[1].parameters]
    counters=dict(residual_evaluations=0,failed_trial_evaluations=0,solver_rhs_evaluations=0)

    def closures(parameters,z):
        return templates[0].evaluate(parameters[:split],z),templates[1].evaluate(parameters[split:],z)

    def prediction(parameters):
        d,r=closures(parameters,support)
        if not np.isfinite(d).all() or not np.isfinite(r).all() or d.min()<=0:
            raise FloatingPointError('Nonfinite or nonpositive trial law on support')
        calls=0
        def rhs(time,state):
            nonlocal calls
            calls+=1;counters['solver_rhs_evaluations']+=1
            if calls>max_rhs_evaluations:raise FloatingPointError('Trial solver exceeded its RHS evaluation budget')
            z=state.reshape(shape);d,r=closures(parameters,z)
            if not np.isfinite(d).all() or not np.isfinite(r).all() or d.min()<=0:
                raise FloatingPointError('Invalid trial law along the observed trajectory')
            flux=.5*(d+np.roll(d,-1,-1))*(np.roll(z,-1,-1)-z)/dx
            result=((flux-np.roll(flux,1,-1))/dx+r).ravel()
            if not np.isfinite(result).all():raise FloatingPointError('Nonfinite trial RHS')
            return result
        sol=solve_ivp(rhs,(0,float(saved_times[-1])),u[:,0].ravel(),t_eval=saved_times,
            method='RK45',rtol=2e-6,atol=1e-8,max_step=min(5e-4,.35*dx*dx/float(d.max())))
        if not sol.success or sol.y.shape[1]!=len(saved_times):raise FloatingPointError('Incomplete trial solve')
        result=sol.y.T.reshape(len(saved_times),*shape).transpose(1,0,2)
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite trial prediction')
        return result

    def residual(parameters):
        counters['residual_evaluations']+=1
        d,_=closures(parameters,support)
        penalty=np.nan_to_num(np.minimum(d-1e-4,0)*100,nan=1e3,posinf=1e3,neginf=-1e3)/np.sqrt(len(support))
        try:data=(prediction(parameters)-target).ravel()/np.sqrt(target.size)
        except (FloatingPointError,ValueError,OverflowError):
            counters['failed_trial_evaluations']+=1;data=np.full(target.size,1e3/np.sqrt(target.size))
        return np.r_[data,penalty]

    start=time.perf_counter();initial_residual=residual(initial)
    if len(initial):
        fit=least_squares(residual,initial,x_scale=np.maximum(np.abs(initial),.01),max_nfev=max_nfev,
            ftol=1e-8,gtol=1e-8,xtol=1e-8)
        parameters=fit.x
        solver_info=dict(nfev=int(fit.nfev),status=int(fit.status),success=bool(fit.success),optimality=float(fit.optimality),message=fit.message)
    else:
        parameters=initial;solver_info=dict(nfev=0,status=0,success=True,optimality=None,message='No free floating coefficients in the selected expressions')
    final=prediction(parameters);final_residual=residual(parameters)
    info=dict(**solver_info,**counters,max_nfev=max_nfev,max_rhs_evaluations_per_trial=max_rhs_evaluations,
        parameter_count=len(parameters),parameters_D=parameters[:split].tolist(),parameters_R=parameters[split:].tolist(),
        expression_D=templates[0].expression(parameters[:split]),expression_R=templates[1].expression(parameters[split:]),
        initial_cost=float(.5*np.sum(initial_residual**2)),cost=float(.5*np.sum(final_residual**2)),
        observation_mse=float(np.mean((final-target)**2)),seconds=time.perf_counter()-start,
        fit_snapshot_indices=indices.tolist(),fit_trajectories=u.shape[0],nx=u.shape[-1],
        initial_state='First observed snapshot, including observation noise',
        solver='RK45 rtol2e-6 atol1e-8, max_step min(5e-4,.35 dx^2/maxD)',
        truth_used_for_fitting=False)
    return parameters,info


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_identity(raw,observed,case,seed,noise,neural,allow_reconstruction=False):
    actual=hashlib.sha256(observed['u'].tobytes()).hexdigest();expected=neural['data_hash']
    diagnostics=observation_diagnostics(raw,observed,case,seed,noise)
    differences={}
    def compare(reference,new,prefix=''):
        for key,value in reference.items():
            name=prefix+key
            if isinstance(value,dict):compare(value,new[key],name+'.')
            else:
                a=np.asarray(value,dtype=float);b=np.asarray(new[key],dtype=float)
                if a.shape!=b.shape or not np.allclose(a,b,rtol=1e-12,atol=1e-14):
                    raise ValueError(f'Reconstructed observation diagnostic mismatch: {name}')
                differences[name]=float(np.max(np.abs(a-b)))
    compare(neural['observation'],diagnostics)
    if actual!=expected and not allow_reconstruction:
        raise ValueError('Observation bytes differ from the archive; use --reconstruct-observations only for a separately recorded reconstruction protocol')
    return dict(data_hash=actual,archived_data_hash=expected,bitwise_observation_match=actual==expected,
        observation_diagnostics=diagnostics,diagnostic_absolute_differences=differences,
        observation_protocol='Archived bytes match' if actual==expected else
            'Local deterministic reconstruction with matching declared configuration and diagnostics; original observation arrays are unavailable; bitwise equivalence is not claimed')


def run_one(path,max_nfev,retry_failed=False,allow_reconstruction=False):
    frozen=json.loads(path.read_text());ident=frozen['id'];cfg=frozen['config']
    neural_path=input_path(Path('evidence/raw')/f'{ident}.json');neural=json.loads(neural_path.read_text())
    if frozen['status']!='complete' or neural['status']!='complete' or cfg!=neural['config']:
        raise ValueError(f'Incomplete or mismatched frozen inputs: {ident}')
    identity=dict(source_signature=source_signature(),experiment_source_sha256=digest(__file__),
        neural_record_sha256=digest(neural_path),pysr_record_sha256=digest(path),
        frozen_neural_signature=model_signature(neural),max_nfev=max_nfev,
        coefficient_policy='Distinct floating literals; shared equal literals; fixed integer structure',
        allow_reconstructed_observations=allow_reconstruction,
        validation_protocol='nx64, unclipped CFL RK4, T0.5, save_dt0.01, four unseen trajectories, two fixed IC regimes')
    dest=ROOT/'evidence/pysr_observation_refit'/f'{ident}.json'
    if dest.exists():
        prior=json.loads(dest.read_text())
        if any(prior.get(k)!=v for k,v in identity.items()):raise ValueError(f'Refit cache identity mismatch: {ident}')
        if prior['status']=='complete':return ident,'cached'
        if not retry_failed:return ident,'cached_'+prior['status']
        archive=ROOT/'evidence/failures/pysr_observation_refit'/f'{ident}_{time.time_ns()}.json'
        archive.parent.mkdir(parents=True,exist_ok=True);dest.replace(archive)
    start=time.perf_counter();cpu=time.process_time()
    result=dict(identity,id=ident,config=cfg,status='running',classification='fixed_expression_observation_refit',
        frozen_search_settings=frozen.get('settings'),input_link='Archived shared neural export and configuration; hashes retained')
    try:
        case=extra_case(cfg['case']);seed=cfg['seed']
        raw=generate_dataset(case,SimulationConfig1D(nx=64,t_final=.1,save_every=10),8,seed=seed,
            amplitude_range=tuple(cfg.get('amplitude_range',case.value_range)),initial_clip_range=tuple(cfg.get('amplitude_range',case.value_range)))
        noise,ss,ts=SETTINGS[cfg['setting']]
        observed=apply_observation_model(raw,noise_level=noise,space_stride=ss,time_stride=ts,seed=seed,clip_range=case.value_range)
        result.update(observation_identity(raw,observed,case,seed,noise,neural,allow_reconstruction))
        npz_write(ROOT/'evidence/observations'/f'{ident}.npz',**observed)
        templates=[ExpressionTemplate.compile(frozen['expressions'][b]['expression']) for b in ('D','R')]
        result['templates']={b:t.metadata() for b,t in zip(('D','R'),templates,strict=True)}
        initial=[t.parameters for t in templates];grid=np.linspace(*case.value_range,256)
        result['float64_frozen_metrics']=curve_metrics(case,grid,
            lambda z:templates[0].evaluate(initial[0],z),lambda z:templates[1].evaluate(initial[1],z),observed)
        result['archived_frozen_metrics']=frozen['metrics']
        result['neural_metrics']=neural['epochs'][str(cfg['epochs'])]['neural']
        result['polynomial_metrics']={name:control['metrics'] for name,control in neural['controls'].items()}
        parameters,fit=fit_to_observations(observed,case.value_range,templates,max_nfev=max_nfev)
        split=len(templates[0].constants);D=lambda z:templates[0].evaluate(parameters[:split],z);R=lambda z:templates[1].evaluate(parameters[split:],z)
        result['fit']=fit;result['metrics']=curve_metrics(case,grid,D,R,observed)
        result['paired_difference']={b:result['metrics'][b]-result['float64_frozen_metrics'][b] for b in ('D','R')}
        result['validation']={};x=np.arange(64)/64;rng=np.random.default_rng(seed+1000)
        shapes=np.stack([random_fourier_initial_condition(x,rng,(0.,1.),4) for _ in range(4)])
        midpoint=1.1 if case.name=='case_exp' else sum(case.value_range)/2
        half_range=.25 if case.name=='case_exp' else .2
        patterns={'full_support':case.value_range[0]+shapes*(case.value_range[1]-case.value_range[0]),
                  'fixed_midpoint_half_range':midpoint+half_range*(2*shapes-1)}
        for label,u0 in patterns.items():
            times,truth,info=strict_rollout(case.diffusion,case.reaction,u0,t_final=.5)
            if info['status']!='stable':raise FloatingPointError('Reference validation did not complete')
            entries={}
            for name,functions in [('frozen',(lambda z:templates[0].evaluate(initial[0],z),lambda z:templates[1].evaluate(initial[1],z))),('refit',(D,R))]:
                tp,pred,validation=strict_rollout(*functions,u0,t_final=.5);n=min(len(times),len(tp))
                errors=[relative_l2_error(truth[i],pred[i]) for i in range(n)]
                entries[name]=dict(info=validation,times=tp,errors=errors,
                    final_error=errors[-1] if validation['status']=='stable' and len(tp)==len(times) else None)
            result['validation'][label]=dict(midpoint=midpoint if label!='full_support' else sum(case.value_range)/2,
                half_range=half_range if label!='full_support' else (case.value_range[1]-case.value_range[0])/2,
                actual_initial_means=u0.mean(-1),methods=entries)
        npz_write(ROOT/'outputs/runs'/ident/'pysr_observation_refit.npz',grid=grid,
            true_D=case.diffusion(grid),true_R=case.reaction(grid),frozen_D=templates[0].evaluate(initial[0],grid),
            frozen_R=templates[1].evaluate(initial[1],grid),refit_D=D(grid),refit_R=R(grid))
        result['status']='complete'
    except Exception as error:
        result['status']='failed';result['error']=str(error);result['traceback']=traceback.format_exc()
    result.update(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    json_write(dest,result);return ident,result['status']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--max-nfev',type=int,default=40)
    parser.add_argument('--case',choices=['all','case_a','case_b','case_exp'],default='all')
    parser.add_argument('--setting',choices=['all','clean','noise3'],default='all')
    parser.add_argument('--seed',type=int);parser.add_argument('--limit',type=int)
    parser.add_argument('--retry-failed',action='store_true');parser.add_argument('--manifest-only',action='store_true')
    parser.add_argument('--reconstruct-observations',action='store_true',help='Explicitly use locally reconstructed observations when archive byte hashes differ; matching archived diagnostics are required and every array is saved')
    args=parser.parse_args()
    if args.workers<1 or args.max_nfev<1 or (args.limit is not None and args.limit<1):parser.error('Workers and budgets must be positive')
    paths=[]
    for path in input_glob('evidence/pysr'):
        row=json.loads(path.read_text());cfg=row['config']
        if cfg['group']!='main' or cfg['case'] not in ('case_a','case_b','case_exp') or cfg['setting'] not in ('clean','noise3'):continue
        if args.case!='all' and cfg['case']!=args.case:continue
        if args.setting!='all' and cfg['setting']!=args.setting:continue
        if args.seed is not None and cfg['seed']!=args.seed:continue
        paths.append(path)
    if args.limit is not None:paths=paths[:args.limit]
    if not paths:raise ValueError('No matching archived PySR inputs')
    json_write(ROOT/'evidence/manifest_pysr_observation_refit.json',dict(ids=[p.stem for p in paths],max_nfev=args.max_nfev,
        experiment_source_sha256=digest(__file__),input_root=str(INPUT_ROOT),workers=args.workers,
        reconstruct_observations=args.reconstruct_observations,
        rule='Fixed floating coefficients and integer structure; observations only; every selected input is evaluated and retained'))
    if args.manifest_only:return
    import concurrent.futures as cf
    with process_pool(args.workers) as pool:
        report_completed(f.result() for f in cf.as_completed([pool.submit(run_one,p,args.max_nfev,args.retry_failed,args.reconstruct_observations) for p in paths]))


if __name__=='__main__':main()
