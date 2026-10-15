"""Reproducible controlled Burgers campaign. Run train, evaluate, then report."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
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
import numpy as np
import torch
from burgers_core import (FluxNet,NU,shock,observation_operator,direct_fit,tensor_operator,
    loss,gradient_norm,frozen_fit,polynomial,projection_geometry,rollout,cole_hopf_periodic)
ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as file:
        out=csv.DictWriter(file,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)


def protocol():
    return dict(pde='u_t + d_x F(u) = 0.02 u_xx, F_true=u^2/2',
        known_viscosity=NU,unknown='Only scalar flux F(u); F(0)=0 gauge fixed exactly',
        training=dict(domain=[-1.,1.],boundary='Exact real-line shock observed on finite interval; no periodic assumption',
            time=[0.,.3],time_samples=31,spatial_samples=[65,129],mean=.25,jump_half=.5,
            noise_fraction_of_far_field_jump=[0.,.01],noise_absolute_sigma=[0.,.01],seeds=list(range(5)),
            no_clipping=True,clean_seeds='Five neural initialization restarts; identical clean observations',
            noisy_seeds='Five independent saved noise draws; each draw shared by all objectives and comparators',
            noise_rng_seed='830000 + 1000*spatial_samples + seed',
            objectives=['strong centered finite differences','weak compact spacetime tests, no state differentiation']),
        weak_tests=dict(space_centers=np.linspace(-.35,.45,9).tolist(),space_widths=[.25,.4],
            time_centers=[.08,.15,.22],time_width=.079,bump='(1-z^2)^4 supported |z|<1',quadrature='tensor trapezoid',
            equation='int F(u)*phi_x = -int u*(phi_t + nu*phi_xx)',normalization='Each test divided by integral abs(phi_x)'),
        neural=dict(form='F(u)=u*(offset+sum_j v_j*tanh(w_j*u+b_j)), hidden=16',dtype='float64',device='CPU',
            convexity_constrained=False,adam_steps=1500,adam_lr=.005,lbfgs_max_iter=150,lbfgs_max_eval=225,
            lbfgs_gradient_tolerance=1e-9,lbfgs_change_tolerance=1e-14,
            stages='Save Adam checkpoint then fixed-budget LBFGS continuation; final iterate, no truth selection'),
        symbolic=dict(degrees=[2,3],basis='u,u^2[,u^3]; no constant',
            freeze_measure='uniform trapezoidal grid of 1025 points on [-.25,.75]',
            direct='Scaled global linear least squares on identical saved observations/operator',
            refit='Fixed-family observation refitting equals direct least squares exactly; counted once',free_structure_search=False),
        evaluation=dict(closure_grid=[-.25,.75,1025],extrapolation_grid=[-.4,.9,1025],
            shock_states=[{'left':.75,'right':-.25},{'left':.6,'right':-.4}],
            shock_metric='Rankine-Hugoniot secant speed; a single-shock interpretation needs convexity on interval',
            periodic_domain=[-1.,1.],initial_conditions={'interpolation':'.25+.35*sin(pi*x)+.08*sin(2*pi*x)',
                'extrapolation':'.25+.65*sin(pi*x)'},horizons=[.5,1.5],
            forward='Centered conservative method of lines, DOP853; known positive viscosity',
            grids=[64,128],refine_to_256_if_max_endpoint_rms_grid_difference_exceeds=.01,
            refined_max_endpoint_rms_tolerance=.005,rtol=2e-7,atol=2e-9,
            state_bound_tolerance=.005,forward_diagnostics='All accepted adaptive endpoints; finite sampling is not a global certificate',
            truth='Independent analytic periodic Cole-Hopf heat Fourier solution, 8192 modes',
            time_check='Repeat worst numerical refinement case at tenfold tighter tolerances',
            physical_interpretation='Positive viscosity remains fixed. Nonconvex flux violates Burgers structure but does not imply parabolic ill-posedness.'),
        stopping='No selection or tuning by true flux, forward error or seed. All fits and failures retained.')


def declare(output):
    output.mkdir(parents=True,exist_ok=True)
    path=output/'PROTOCOL.json';p=protocol()
    if path.exists():
        if json.loads(path.read_text())!=p:raise ValueError('Protocol mismatch')
        return p
    save(path,p)
    source=output/'provenance/source';source.mkdir(parents=True)
    for name in ['burgers_core.py','burgers_campaign.py']:
        shutil.copy2(ROOT/'experiments'/name,source/name)
    save(output/'ENVIRONMENT.json',dict(command=sys.argv,python=sys.version,
        packages={n:importlib.metadata.version(n) for n in ['numpy','torch','scipy','matplotlib','pytest']},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        source_sha256={name:digest(ROOT/'experiments'/name) for name in ['burgers_core.py','burgers_campaign.py']}))
    return p


def input_file(output,n,noise,seed):
    path=output/'inputs'/f'n{n}_noise{noise:g}_seed{seed}.npz'
    if path.exists():return path
    path.parent.mkdir(parents=True,exist_ok=True)
    x=np.linspace(-1,1,n);t=np.linspace(0,.3,31);clean=shock(x,t)
    rng=np.random.default_rng(830000+1000*n+seed)
    noisy=clean+noise*rng.normal(size=clean.shape)
    np.savez_compressed(path,x=x,t=t,clean=clean,observed=noisy,noise=noisy-clean)
    return path


def model_metrics(evaluate):
    result={}
    for name,lo,hi in [('train',-.25,.75),('extrap',-.4,.9)]:
        u=np.linspace(lo,hi,1025)
        flux=evaluate(u,0);speed=evaluate(u,1);curvature=evaluate(u,2)
        result[name+'_flux_relative_l2']=float(np.linalg.norm(flux-.5*u*u)/np.linalg.norm(.5*u*u))
        result[name+'_speed_relative_l2']=float(np.linalg.norm(speed-u)/np.linalg.norm(u))
        result[name+'_min_curvature']=float(np.min(curvature))
        result[name+'_convexity_violated_on_grid']=bool(np.min(curvature)<0)
    for i,(left,right) in enumerate([(.75,-.25),(.6,-.4)]):
        secant=float((evaluate(np.array([left]))[0]-evaluate(np.array([right]))[0])/(left-right))
        result[f'shock{i}_secant_speed']=secant
        result[f'shock{i}_speed_error']=secant-(left+right)/2
    return result


def train(output,p):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for n in p['training']['spatial_samples']:
      for noise in p['training']['noise_absolute_sigma']:
       for seed in p['training']['seeds']:
        ip=input_file(output,n,noise,seed)
        data=np.load(ip)
        for arm in ['strong','weak']:
            run=f'n{n}_noise{noise:g}_seed{seed}_{arm}'
            dest=output/'fits'/f'{run}.json'
            if dest.exists():continue
            started=time.time()
            try:
                operator=observation_operator(data['x'],data['t'],data['observed'],arm)
                tensor=tensor_operator(operator);model=FluxNet(seed)
                opt=torch.optim.Adam(model.parameters(),lr=.005)
                for step in range(1500):
                    opt.zero_grad(set_to_none=True);objective=loss(model,tensor)
                    if not torch.isfinite(objective):raise FloatingPointError('Nonfinite Adam loss')
                    objective.backward();opt.step()
                adam_loss,adam_grad=gradient_norm(model,tensor)
                models=output/'models';models.mkdir(parents=True,exist_ok=True)
                torch.save(model.state_dict(),models/f'{run}_adam.pt')
                adam_metrics=model_metrics(model.numpy)
                calls=[0]
                opt=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=150,max_eval=225,
                    tolerance_grad=1e-9,tolerance_change=1e-14,line_search_fn='strong_wolfe')
                def closure():
                    calls[0]+=1;opt.zero_grad(set_to_none=True);value=loss(model,tensor)
                    if not torch.isfinite(value):raise FloatingPointError('Nonfinite LBFGS loss')
                    value.backward();return value
                opt.step(closure)
                final_loss,final_grad=gradient_norm(model,tensor)
                torch.save(model.state_dict(),models/f'{run}_final.pt')
                grid=np.linspace(-.25,.75,1025);teacher=model.numpy(grid)
                rows=[]
                for degree in [2,3]:
                    frozen=frozen_fit(grid,teacher,degree)
                    direct,info=direct_fit(operator,degree)
                    rows.append(dict(degree=degree,frozen_coefficients=frozen.tolist(),direct_coefficients=direct.tolist(),
                        direct_diagnostics=info,geometry=projection_geometry(grid,teacher,frozen),
                        frozen_metrics=model_metrics(lambda u,d=0:polynomial(u,frozen,d)),
                        direct_metrics=model_metrics(lambda u,d=0:polynomial(u,direct,d))))
                save(dest,dict(run=run,n=n,noise=noise,seed=seed,arm=arm,status='completed',
                    input_path=str(ip),input_sha256=digest(ip),seconds=time.time()-started,
                    adam=dict(loss=adam_loss,gradient_l2=adam_grad,metrics=adam_metrics),
                    final=dict(loss=final_loss,gradient_l2=final_grad,gradient_l2_below_1e9=final_grad<=1e-9,
                        closure_calls=calls[0],metrics=model_metrics(model.numpy)),symbolic=rows,
                    models={stage:dict(path=f'models/{run}_{stage}.pt',sha256=digest(models/f'{run}_{stage}.pt')) for stage in ['adam','final']}))
                print(run,'loss',final_loss,'grad',final_grad,'seconds',round(time.time()-started,1),flush=True)
            except Exception as error:
                save(output/'failures'/f'{run}.json',dict(run=run,error=repr(error),traceback=traceback.format_exc()))
                raise


def evaluations(fit,output):
    model=FluxNet(fit['seed']);model.load_state_dict(torch.load(output/fit['models']['final']['path'],weights_only=True))
    yield 'neural',model.numpy_evaluator()
    for row in fit['symbolic']:
        for name in ['frozen','direct']:
            coefficients=np.array(row[name+'_coefficients'])
            yield f'{name}{row["degree"]}',lambda u,d=0,c=coefficients:polynomial(u,c,d)


def evaluate(output,p):
    torch.set_num_threads(1)
    fit_paths=sorted((output/'fits').glob('*.json'))
    if len(fit_paths)!=40:raise ValueError(f'Expected 40 fits, found {len(fit_paths)}')
    protocol_path=output/'EVALUATION_PROTOCOL.json'
    if not protocol_path.exists():save(protocol_path,dict(plan=p['evaluation'],source_sha256=digest(__file__),core_sha256=digest(ROOT/'experiments/burgers_core.py')))
    cache={}
    for fit_path in fit_paths:
        fit=json.loads(fit_path.read_text())
        for name,method in evaluations(fit,output):
            flux_signature=hashlib.sha256(method(np.linspace(-.4,.9,2049)).tobytes()).hexdigest()
            for kind in ['interpolation','extrapolation']:
                key=f'{fit["run"]}_{name}_{kind}'
                dest=output/'forward'/f'{key}.json'
                if dest.exists():continue
                start=time.time()
                try:
                    cachekey=(flux_signature,kind)
                    if cachekey in cache:
                        previous=json.loads((output/'forward'/f'{cache[cachekey]}.json').read_text())
                        previous.update(key=key,run=fit['run'],method=name,duplicate_of=cache[cachekey])
                        save(dest,previous);continue
                    coarse=rollout(method,kind,n=64)
                    fine=rollout(method,kind,n=128)
                    discrepancy=np.sqrt(np.mean((coarse['u']-fine['u'][:,::2])**2,axis=1))
                    refined=False;grid=128
                    if max(discrepancy)>.01 or fine['status']!='completed':
                        finest=rollout(method,kind,n=256)
                        initial_discrepancy=discrepancy.copy()
                        discrepancy=np.sqrt(np.mean((fine['u']-finest['u'][:,::2])**2,axis=1))
                        coarse=fine;fine=finest;refined=True;grid=256
                    reference=np.array([cole_hopf_periodic(fine['x'],float(t),kind) for t in fine['t']])
                    error=np.sqrt(np.mean((fine['u']-reference)**2,axis=1))
                    denom=np.sqrt(np.mean((reference-np.mean(reference,axis=1)[:,None])**2,axis=1))
                    threshold=.005 if refined else .01
                    numerical_ok=max(discrepancy)<=threshold and fine['status']=='completed'
                    datafile=output/'forward_arrays'/f'{key}.npz';datafile.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(datafile,x=fine['x'],t=fine['t'],u=fine['u'],reference=reference,
                        coarse_u=coarse['u'],coarse_x=coarse['x'])
                    save(dest,dict(key=key,run=fit['run'],method=name,kind=kind,
                        n=grid,t=fine['t'].tolist(),grid_rms_difference=discrepancy.tolist(),
                        endpoint_rms_error=error.tolist(),endpoint_relative_fluctuation_error=(error/denom).tolist(),
                        refinement_triggered=refined,status='verified' if numerical_ok else 'numerical_resolution_failed',
                        mass_drift=fine['mass_drift'],maximum_principle_violation=fine['maximum_principle_violation'],
                        energy_increase=fine['energy_increase'],accepted_steps=fine['accepted_steps'],nfev=fine['nfev'],
                        diagnostic_scope=fine['diagnostic_scope'],array_path=str(datafile),array_sha256=digest(datafile),
                        seconds=time.time()-start,duplicate_of=None))
                    cache[cachekey]=key
                    print(key,'grid',grid,'delta',float(max(discrepancy)), 'status',numerical_ok,'sec',round(time.time()-start,1),flush=True)
                except Exception as error:
                    save(output/'failures'/f'{key}.json',dict(key=key,error=repr(error),traceback=traceback.format_exc()))
                    raise
    # This check is selected solely by largest numerical refinement discrepancy.
    records=[json.loads(path.read_text()) for path in (output/'forward').glob('*.json')]
    worst=max(records,key=lambda r:max(r['grid_rms_difference']))
    fit=json.loads((output/'fits'/f'{worst["run"]}.json').read_text())
    method=dict(evaluations(fit,output))[worst['method']]
    tight=rollout(method,worst['kind'],n=worst['n'],rtol=2e-8,atol=2e-10)
    original=np.load(worst['array_path'])['u']
    delta=float(np.max(np.sqrt(np.mean((tight['u']-original)**2,axis=1))))
    true_solver=[]
    for kind in ['interpolation','extrapolation']:
        for n in [64,128,256]:
            result=rollout(lambda u,d=0:polynomial(u,[0.,.5],d),kind,n=n)
            reference=np.array([cole_hopf_periodic(result['x'],t,kind) for t in result['t']])
            true_solver.append(dict(kind=kind,n=n,max_rms_error=float(np.max(np.sqrt(np.mean((result['u']-reference)**2,axis=1)))),
                mass_drift=result['mass_drift'],state_bound_violation=result['maximum_principle_violation']))
    save(output/'FORWARD_VALIDATION.json',dict(worst_case=worst['key'],tighter_time_tolerance_rms_difference=delta,
        time_tolerance_check_passed=delta<1e-5,independent_truth_checks=true_solver,
        verified=sum(r['status']=='verified' for r in records),total=len(records),
        maximum_mass_drift=max(r['mass_drift'] for r in records)))
    if delta>=1e-5:raise ValueError('Time tolerance convergence check failed')


def report(output,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fits=[json.loads(path.read_text()) for path in sorted((output/'fits').glob('*.json'))]
    if len(fits)!=40:raise ValueError('Incomplete campaign')
    rows=[];geometry=[];optimization=[]
    for fit in fits:
        common={k:fit[k] for k in ['run','n','noise','seed','arm']}
        rows.append(dict(**common,method='neural',**fit['final']['metrics']))
        optimization.append(dict(**common,adam_loss=fit['adam']['loss'],final_loss=fit['final']['loss'],
            adam_gradient_l2=fit['adam']['gradient_l2'],final_gradient_l2=fit['final']['gradient_l2'],
            adam_flux_error=fit['adam']['metrics']['train_flux_relative_l2'],final_flux_error=fit['final']['metrics']['train_flux_relative_l2'],
            strict_l2_gradient_converged=fit['final']['gradient_l2_below_1e9']))
        for item in fit['symbolic']:
            geometry.append(dict(**common,degree=item['degree'],**item['geometry']))
            for name in ['frozen','direct']:
                rows.append(dict(**common,method=name+str(item['degree']),**item[name+'_metrics']))
    csv_write(output/'reports/closure_by_seed.csv',rows)
    csv_write(output/'reports/projection_geometry.csv',geometry)
    csv_write(output/'reports/optimization.csv',optimization)
    groups=defaultdict(list)
    for row in rows:groups[(row['n'],row['noise'],row['arm'],row['method'])].append(row)
    summary=[]
    for (n,noise,arm,method),group in groups.items():
        summary.append(dict(n=n,noise=noise,arm=arm,method=method,count=len(group),
            flux_error_mean=float(np.mean([r['train_flux_relative_l2'] for r in group])),
            flux_error_sd=float(np.std([r['train_flux_relative_l2'] for r in group],ddof=1)),
            speed_error_mean=float(np.mean([r['train_speed_relative_l2'] for r in group])),
            extrap_flux_error_mean=float(np.mean([r['extrap_flux_relative_l2'] for r in group])),
            convexity_failures=sum(r['train_convexity_violated_on_grid'] for r in group),
            extrap_convexity_failures=sum(r['extrap_convexity_violated_on_grid'] for r in group),
            heldout_secant_speed_abs_error_mean=float(np.mean([abs(r['shock1_speed_error']) for r in group]))))
    csv_write(output/'reports/closure_summary.csv',summary)
    forwards=[]
    for path in sorted((output/'forward').glob('*.json')):
        row=json.loads(path.read_text());fit=next(f for f in fits if f['run']==row['run'])
        for j,t in enumerate(row['t'][1:],1):
            forwards.append(dict(run=row['run'],n_train=fit['n'],noise=fit['noise'],seed=fit['seed'],arm=fit['arm'],
                method=row['method'],kind=row['kind'],t=t,n_forward=row['n'],status=row['status'],
                rms_error=row['endpoint_rms_error'][j],relative_fluctuation_error=row['endpoint_relative_fluctuation_error'][j],
                grid_rms_difference=row['grid_rms_difference'][j],mass_drift=row['mass_drift'],
                maximum_principle_violation=row['maximum_principle_violation'],energy_increase=row['energy_increase']))
    if forwards:csv_write(output/'reports/forward_by_seed.csv',forwards)
    figs,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for noise,ax in zip([0.,.01],axes[0]):
        for arm,style in [('strong','-'),('weak','--')]:
            for method,color in [('neural','C0'),('frozen2','C1'),('direct2','C2')]:
                chosen=[r for r in summary if r['noise']==noise and r['arm']==arm and r['method']==method]
                chosen.sort(key=lambda r:r['n'])
                ax.plot([r['n'] for r in chosen],[r['flux_error_mean'] for r in chosen],style,color=color,marker='o',label=arm+' '+method)
        ax.set(title=f'Flux error: noise {100*noise:g}%',xlabel='Training spatial samples',ylabel='Relative L2 error',yscale='log')
        ax.legend(fontsize=6)
    ax=axes[1,0]
    for arm,color in [('strong','C0'),('weak','C1')]:
        r=[r for r in optimization if r['noise']==.01 and r['n']==129 and r['arm']==arm]
        for row in r:ax.plot([0,1],[row['adam_flux_error'],row['final_flux_error']],color=color,alpha=.5)
        ax.plot([],[],color=color,label=arm)
    ax.set(xticks=[0,1],xticklabels=['Adam','+ L-BFGS'],ylabel='Noisy fine-grid flux error',yscale='log');ax.legend()
    ax=axes[1,1]
    for arm,color in [('strong','C0'),('weak','C1')]:
        for row in [r for r in geometry if r['noise']==.01 and r['n']==129 and r['arm']==arm and r['degree']==2]:
            ax.scatter(np.sqrt(row['teacher_squared_error']),np.sqrt(row['symbolic_squared_error']),color=color)
        ax.scatter([],[],color=color,label=arm)
    bounds=ax.get_xlim();ax.plot(bounds,bounds,'k:',label='Equal error')
    ax.set(xlabel='Teacher absolute flux L2',ylabel='Frozen quadratic absolute flux L2');ax.legend()
    (output/'reports').mkdir(exist_ok=True)
    figs.savefig(output/'reports/mechanisms.png',dpi=180);plt.close(figs)
    checks=dict(fits=len(fits),all_inputs_hash_match=all(digest(f['input_path'])==f['input_sha256'] for f in fits),
        all_models_hash_match=all(digest(output/r['path'])==r['sha256'] for f in fits for r in f['models'].values()),
        max_projection_identity_residual=max(r['decomposition_residual'] for r in geometry),
        final_optimization_strict_l2_converged=sum(r['strict_l2_gradient_converged'] for r in optimization),
        final_objective_nonincrease=all(r['final_loss']<=r['adam_loss']+1e-12 for r in optimization),
        weak_truth_excluded='operators accept only saved state, x, t, fixed viscosity and test functions',
        train_gauge_max=0.)
    if forwards:checks['forward_rows']=len(forwards)
    save(output/'INTEGRITY_CHECKS.json',checks)
    lines=['# Controlled viscous Burgers neural-symbolic flux campaign','',
        'Known viscosity ν=0.02; learned scalar flux F(u), fixed F(0)=0 gauge. Exact real-line shock observations, 40 neural fits: 2 grids × clean/1% state noise × 5 seeds × strong/weak observation objectives. Each fit has a fixed Adam stage and a fixed L-BFGS continuation; two saved stages are paired, not independent runs. Clean seeds are initialization restarts.',
        '', 'Weak tests are compact in both space and time; no noisy state differentiation. Strong targets and features use centered differences of the same saved noisy fields. All symbolic models have the same zero gauge. Degree-2/3 families are fixed in advance. Direct fitting and observation refitting solve the same global linear problem and count as one comparator.',
        '', '## Fine training grid: relative flux L2 error', '', '| Noise | Objective | Model | Mean | SD | Convexity violations |','|---|---|---|---:|---:|---:|']
    for r in summary:
        if r['n']==129:
            lines.append(f'| {100*r["noise"]:g}% | {r["arm"]} | {r["method"]} | {r["flux_error_mean"]:.6g} | {r["flux_error_sd"]:.6g} | {r["convexity_failures"]}/5 |')
    lines += ['',f'Strict final L2 gradient ≤ 1e-9: {checks["final_optimization_strict_l2_converged"]}/40. Training objective and truth evaluation are distinct; a completed optimizer call does not certify convergence.',
        '', 'Burgers convexity is a constitutive-structure diagnostic. Nonconvex learned flux does not by itself imply ill-posedness with fixed positive viscosity. Secant speeds are Rankine–Hugoniot diagnostics; when convexity fails a single shock is not assumed. Forward smooth-wave cases separately test solution propagation.',
        '', 'Periodic forward fields use common held-out initial states and fixed viscosity. Independent Cole–Hopf heat-semigroup truth is used only for evaluation. All accepted adaptive endpoints are checked for conservation, state bounds and energy change. Mesh discrepancy is retained per case; `numerical_resolution_failed` cases cannot be treated as validated fields.',
        '', 'See `reports/closure_by_seed.csv`, `reports/optimization.csv`, `reports/projection_geometry.csv`, and `reports/forward_by_seed.csv`; frozen-fit geometry uses one common uniform 1025-point trapezoid measure. This campaign is a fixed-family reliability analysis, not free symbolic structure discovery.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate','report']);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=declare(args.output)
    {'train':train,'evaluate':evaluate,'report':report}[args.stage](args.output,p)
if __name__=='__main__':main()
