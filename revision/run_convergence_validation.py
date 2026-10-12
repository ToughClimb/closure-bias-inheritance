"""Validate every fixed-budget restart/checkpoint, retaining correlated structure."""
import argparse
import concurrent.futures as cf
import resource
from revision_core import *
from closure_discovery.data_generation.rd_solver_1d import random_fourier_initial_condition


def work(path):
    row=json.loads(path.read_text());ident=row['id'];dest=ROOT/'evidence/convergence_validation'/f'{ident}.json'
    identity=validation_identity(row,'convergence_all_checkpoints_v2')
    cached=cached_validation(dest,identity)
    if cached is not None:return ident,cached
    start=time.perf_counter();cpu=time.process_time();c=extra_case(row['config']['case']);seed=row['config']['seed']
    rng=np.random.default_rng(seed+1000);x=np.arange(64)/64
    shapes=np.stack([random_fourier_initial_condition(x,rng,amplitude_range=(0.,1.),num_modes=4) for _ in range(4)])
    patterns={'full_support':shapes*c.value_range[1],'midpoint1.1_amplitude0.25':1.1+.25*(2*shapes-1)}
    entries=[]
    for label,u0 in patterns.items():
        times,truth,info=strict_rollout(c.diffusion,c.reaction,u0,t_final=.5,save_dt=.01)
        if info['status']!='stable':raise FloatingPointError('Truth convergence validation failed')
        for epoch,snap in row['epochs'].items():
            q=load_curves(ident,epoch=int(epoch))
            sd,sr=[SymbolicExpression(**{**e,'parameters':tuple(e['parameters'])}) for e in snap['symbols']]
            for method,D,R in [('neural',lambda z:np.interp(z,q['grid'],q['D']),lambda z:np.interp(z,q['grid'],q['R'])),('frozen',sd.evaluate,sr.evaluate)]:
                tp,pred,validation=strict_rollout(D,R,u0,t_final=.5,save_dt=.01);n=min(len(tp),len(times))
                errors=[relative_l2_error(truth[i],pred[i]) for i in range(n)]
                entries.append(dict(epoch=int(epoch),method=method,pattern=label,times=tp,errors=errors,info=validation,
                    final_error=errors[-1] if validation['status']=='stable' else None))
    json_write(dest,dict(identity,id=ident,config=row['config'],classification='fixed_budget_convergence_validation',status='complete',
        entries=entries,seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        rule='Every restart/checkpoint; no true-error model selection; uncertainty aggregates restarts within each independent data seed'))
    return ident,'complete'


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int);p.add_argument('--watch',action='store_true');a=p.parse_args()
    completed=set()
    with process_pool(a.workers) as pool:
        while True:
            paths=[f for f in input_glob('evidence/raw','convergence_*.json') if f.stem not in completed]
            if a.limit:paths=paths[:a.limit]
            report_completed(f.result() for f in cf.as_completed([pool.submit(work,f) for f in paths]))
            completed.update(f.stem for f in paths)
            if not a.watch or len(completed)>=15 or a.limit:break
            time.sleep(15)


if __name__=='__main__':main()
