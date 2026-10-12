"""Direct forward-data refits and longer, varied-initial-condition validation."""
import argparse
import concurrent.futures as cf
import resource
import traceback
from revision_core import *
from run_suite import SETTINGS, task_id
from closure_discovery.data_generation.rd_solver_1d import random_fourier_initial_condition


def reconstruct(row):
    config=row['config'];case=extra_case(config['case']);snap=row['epochs'][str(max(map(int,row['epochs'])))];out=ROOT/'outputs/runs'/row['id']
    curves=load_curves(row['id']);g=curves['grid'];methods={}
    for name in ['neural','frozen','weak_poly','strong_poly','spacetime_poly','refit_weak','refit_data']:
        if name=='frozen':
            ed,er=[SymbolicExpression(**{**e,'parameters':tuple(e['parameters'])}) for e in snap['symbols']];methods[name]=(ed.evaluate,er.evaluate)
        elif name in ['neural']:
            # Preserve the source's tabulated export with explicit constant extrapolation.
            q=load_curves(row['id'],epoch=config['epochs']);grid=q['grid'];d=q['D'];r=q['R']
            methods[name]=(lambda z,g=grid,v=d:np.interp(z,g,v),lambda z,g=grid,v=r:np.interp(z,g,v))
        elif name.startswith('refit'):
            fit=row['controls'][name]['fit'];methods[name]=tuple(
                lambda z,f=fit,b=branch:eval_expr(f['family_'+b],f['parameters_'+b],z,f.get('numerator_degree_'+b),f.get('denominator_degree_'+b))
                for branch in ('D','R'))
        else:
            control=row['controls'][name];d=control['parameters_D'];r=control['parameters_R']
            methods[name]=(lambda z,p=d:np.polynomial.polynomial.polyval(z,p),lambda z,p=r:np.polynomial.polynomial.polyval(z,p))
    return case,snap,methods


def forward_refit(observed,value_range,symbols,max_nfev=40):
    """Fit selected symbolic form to saved observations by forward RK45.

    All observed trajectories used; their first noisy snapshot is the fixed initial
    condition. No derivative or true closure targets, no clipping. Record fit limit.
    """
    u=observed['u'];nx=u.shape[-1];dx=1/nx;t=observed['t'];indices=np.arange(0,len(t),max(1,(len(t)-1)//10))
    times=t[indices];target=u[:,indices];shape=u[:,0].shape;support=np.linspace(*value_range,512)
    split=len(symbols[0].parameters);p0=np.array(symbols[0].parameters+symbols[1].parameters)
    count=0;failed=0
    def eval_p(p,z):
        return (dataclasses.replace(symbols[0],parameters=tuple(p[:split])).evaluate(z),
                dataclasses.replace(symbols[1],parameters=tuple(p[split:])).evaluate(z))
    def residual(p):
        nonlocal count,failed
        count+=1;d,_=eval_p(p,support);pen=np.minimum(d-1e-4,0)*100
        try:
            if not np.all(np.isfinite(d)) or np.min(d)<=0:raise FloatingPointError('diffusion')
            def fun(time,state):
                z=state.reshape(shape);D,R=eval_p(p,z)
                if np.min(D)<=0 or not np.all(np.isfinite(D)) or not np.all(np.isfinite(R)):raise FloatingPointError('trial closure')
                flux=.5*(D+np.roll(D,-1,-1))*(np.roll(z,-1,-1)-z)/dx
                return ((flux-np.roll(flux,1,-1))/dx+R).ravel()
            sol=solve_ivp(fun,(0,float(times[-1])),u[:,0].ravel(),t_eval=times,method='RK45',
                          rtol=2e-6,atol=1e-8,max_step=min(5e-4,.35*dx*dx/float(np.max(d))))
            if not sol.success or sol.y.shape[1]!=len(times):raise FloatingPointError('solver')
            prediction=sol.y.T.reshape(len(times),*shape).transpose(1,0,2)
            data_res=(prediction-target).ravel()/np.sqrt(target.size)
            if not np.all(np.isfinite(data_res)):raise FloatingPointError('residual')
        except (FloatingPointError,OverflowError,ValueError):
            failed+=1;data_res=np.full(target.size,1e3/np.sqrt(target.size))
        return np.r_[data_res,pen/np.sqrt(pen.size)]
    start=time.perf_counter();initial=residual(p0)
    fit=least_squares(residual,p0,x_scale=np.maximum(np.abs(p0),.01),max_nfev=max_nfev,ftol=1e-8,gtol=1e-8,xtol=1e-8)
    info=dict(mode='forward_observation_RK45',initial_cost=float(.5*np.sum(initial**2)),cost=float(fit.cost),
        nfev=int(fit.nfev),actual_residual_evaluations=count,failed_trial_evaluations=failed,max_nfev=max_nfev,
        status=int(fit.status),success=bool(fit.success),optimality=float(fit.optimality),seconds=time.perf_counter()-start,
        family_D=symbols[0].family,family_R=symbols[1].family,parameters_D=fit.x[:split].tolist(),parameters_R=fit.x[split:].tolist(),
        numerator_degree_D=symbols[0].numerator_degree,denominator_degree_D=symbols[0].denominator_degree,
        numerator_degree_R=symbols[1].numerator_degree,denominator_degree_R=symbols[1].denominator_degree,
        conditional_initial_state='first observed snapshot, including noise',fit_trajectories=u.shape[0],fit_snapshot_indices=indices.tolist(),
        solver='RK45, rtol2e-6, atol1e-8, max_step min(5e-4,.35 dx^2/maxD)',nx=nx)
    return lambda z:eval_p(fit.x,z)[0],lambda z:eval_p(fit.x,z)[1],info


def rollout_task(path,do_refit=False):
    row=json.loads(Path(path).read_text());ident=row['id'];dest=ROOT/'evidence/rollout'/f'{ident}.json'
    identity=validation_identity(row,dict(kind='long_rollout_v2',forward_refit=do_refit))
    cached=cached_validation(dest,identity)
    if cached is not None:return ident,cached
    start=time.perf_counter();cpu=time.process_time();case,snap,methods=reconstruct(row);cfg=row['config'];seed=cfg['seed']
    result=dict(identity,id=ident,config=cfg,classification=row.get('classification','reviewer_experiment'),status='running',methods={})
    try:
        raw=generate_dataset(case,SimulationConfig1D(nx=64,t_final=.1,save_every=10),8,seed=seed,
            amplitude_range=tuple(cfg.get('amplitude_range',case.value_range)),initial_clip_range=tuple(cfg.get('amplitude_range',case.value_range)))
        noise,ss,ts=SETTINGS[cfg['setting']]
        observed=apply_observation_model(raw,noise_level=noise,space_stride=ss,time_stride=ts,seed=seed,clip_range=case.value_range)
        if do_refit:
            symbols=[SymbolicExpression(**{**e,'parameters':tuple(e['parameters'])}) for e in snap['symbols']]
            d,r,info=forward_refit(observed,case.value_range,symbols)
            methods['refit_pde']=(d,r);result['forward_refit']=dict(fit=info,metrics=curve_metrics(case,np.linspace(*case.value_range,256),d,r,observed))
        xs=np.arange(64)/64;rng=np.random.default_rng(seed+1000)
        # Original full range plus mean/amplitude combinations specified in advance.
        shape=np.stack([random_fourier_initial_condition(xs,rng,(0.,1.),4) for _ in range(4)])
        initial=shape*(case.value_range[1]-case.value_range[0])+case.value_range[0]
        patterns=[('full_support',initial)]
        if case.name=='case_exp':
            for mean,amp in [(.25,.1),(.6,.2),(1.1,.25),(1.5,.1)]:patterns.append((f'mean{mean:g}_amplitude{amp:g}',mean+amp*(2*shape-1)))
        elif cfg['group']=='families':patterns.append(('mean0.6_amplitude0.2',.6+.2*(2*shape-1)))
        horizon=.5 if case.name=='case_exp' else (.3 if cfg['group']=='families' else .1)
        arrays={}
        for label,u0 in patterns:
            tt,truth,ti=strict_rollout(case.diffusion,case.reaction,u0,t_final=horizon,save_dt=.01)
            if ti['status']!='stable':raise FloatingPointError('truth rollout: '+str(ti))
            for name,(D,R) in methods.items():
                times,pred,info=strict_rollout(D,R,u0,t_final=horizon,save_dt=.01)
                n=min(len(times),len(tt));series=[relative_l2_error(truth[i],pred[i]) for i in range(n)]
                entry=dict(pattern=label,mean=float(u0.mean()),amplitude=float((u0.max()-u0.min())/2),info=info,
                    times=times.tolist(),errors=series,truth_state_min=float(truth.min()),truth_state_max=float(truth.max()),
                    outside_closure_support_fraction=float(np.mean((truth<case.value_range[0])|(truth>case.value_range[1]))))
                for t in [.1,.25,.5]:
                    idx=np.flatnonzero(np.isclose(times,t));entry[f'error_T{t:g}']=float(series[idx[0]]) if len(idx) else None
                result['methods'].setdefault(name,[]).append(entry)
                if seed==0 and label=='full_support':arrays[name]=pred
            if seed==0 and label=='full_support':arrays['true']=truth;arrays['times']=tt
        if arrays:npz_write(ROOT/'outputs/runs'/ident/'long_rollouts.npz',**arrays)
        result['status']='complete'
    except Exception as e:result['status']='failed';result['error']=str(e);result['traceback']=traceback.format_exc()
    result.update(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                  peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    json_write(dest,result);return ident,result['status']


def main():
    p=argparse.ArgumentParser(epilog='Common paths: --work-root PATH (output/cache root), --input-root PATH (explicit artifact input root); defaults use the project directory.');p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int);p.add_argument('--case',default='case_exp');p.add_argument('--groups',nargs='+',default=['main']);p.add_argument('--refit-pde',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--expected',type=int);a=p.parse_args()
    if a.watch:
        submitted=set();futures={}
        with process_pool(a.workers) as pool:
            while True:
                for f in input_glob('evidence/raw', '*.json'):
                    row=json.loads(f.read_text());c=row['config']
                    if row['status']=='complete' and c['group'] in a.groups and (a.case=='all' or c['case']==a.case) and f.stem not in submitted:
                        submitted.add(f.stem);futures[pool.submit(rollout_task,f,a.refit_pde)]=f.stem
                for fut in list(futures):
                    if fut.done():report_completed([fut.result()]);futures.pop(fut)
                if a.expected is not None and len(submitted)>=a.expected and not futures:break
                time.sleep(5)
        return
    files=[]
    for f in input_glob('evidence/raw', '*.json'):
        row=json.loads(f.read_text());c=row['config']
        if row['status']=='complete' and c['group'] in a.groups and (a.case=='all' or c['case']==a.case):files.append(f)
    if a.limit:files=files[:a.limit]
    with process_pool(a.workers) as pool:
        report_completed(f.result() for f in cf.as_completed([pool.submit(rollout_task,path,a.refit_pde) for path in files]))


if __name__=='__main__':main()
