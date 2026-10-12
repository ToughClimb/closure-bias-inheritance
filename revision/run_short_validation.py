"""Source-convention short validation, distinct from final-time long errors.

All 101 saved times over T=.1 enter the trajectory norm, as in the original
main table. Fix observations, model and four unseen initial conditions. Compare
the source's clipped fixed-step policy with strict unclipped CFL validation.
Never select a model using either validation result.
"""
import argparse
import concurrent.futures as cf
import resource
from revision_core import *
from run_refit_rollout import reconstruct
from closure_discovery.data_generation.rd_solver_1d import random_fourier_initial_condition


def work(path):
    row=json.loads(Path(path).read_text());ident=row['id'];dest=ROOT/'evidence/short_validation'/f'{ident}.json'
    identity=validation_identity(row,'short_legacy_and_strict_v2')
    cached=cached_validation(dest,identity)
    if cached is not None:return ident,cached
    start=time.perf_counter();cpu=time.process_time();case,snap,methods=reconstruct(row);c=row['config']
    rng=np.random.default_rng(c['seed']+1000);x=np.arange(64)/64
    initial=np.stack([random_fourier_initial_condition(x,rng,case.value_range,4) for _ in range(4)])
    times,truth,ti=strict_rollout(case.diffusion,case.reaction,initial,t_final=.1,save_dt=.001,
                                 adaptive=False,state_clip=(-.25,2.),allow_nonpositive=True)
    if ti['status']!='stable':raise FloatingPointError('Invalid reference rollout')
    result=dict(identity,id=ident,config=c,classification='source_convention_validation',status='complete',
        norm='||truth-pred||_2 / (||truth||_2+1e-12), all 4x101x64 entries',
        protocol=dict(nx=64,dt=1e-4,T=.1,save_dt=.001,initial_seed=c['seed']+1000,
                      initial_state_range=case.value_range),truth_info=ti,methods={})
    for name,(D,R) in methods.items():
        entry={}
        for policy,kwargs in [('legacy',dict(adaptive=False,state_clip=(-.25,2.),allow_nonpositive=True)),
                              ('strict',dict(adaptive=True))]:
            tp,pred,info=strict_rollout(D,R,initial,t_final=.1,save_dt=.001,**kwargs)
            n=min(len(tp),len(times));complete=info['status']=='stable' and n==len(times)
            entry[policy]=dict(info=info,trajectory_error=relative_l2_error(truth,pred) if complete else None,
                final_error=relative_l2_error(truth[-1],pred[-1]) if complete else None,
                partial_trajectory_error=relative_l2_error(truth[:n],pred[:n]) if n else None,
                completed_saved_times=n,total_saved_times=len(times))
        result['methods'][name]=entry
    result.update(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                  peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    json_write(dest,result);return ident,'complete'


def main():
    p=argparse.ArgumentParser(epilog='Common paths: --work-root PATH (output/cache root), --input-root PATH (explicit artifact input root); defaults use the project directory.');p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int);a=p.parse_args()
    files=input_glob('evidence/raw', 'main_*.json')
    files=[f for f in files if json.loads(f.read_text())['status']=='complete']
    if a.limit:files=files[:a.limit]
    json_write(ROOT/'evidence/manifest_short_validation.json',dict(ids=[f.stem for f in files],
        expected=75,workers=a.workers,reason='Retain original all-time trajectory metric separately from final-time long validation'))
    with process_pool(a.workers) as pool:
        report_completed(f.result() for f in cf.as_completed([pool.submit(work,path) for path in files]))


if __name__=='__main__':main()
