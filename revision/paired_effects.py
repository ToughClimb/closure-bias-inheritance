"""Controlled numerical-issue effects; keep ranking, error and BIR separate."""
import argparse
import concurrent.futures as cf
from revision_core import *
from run_refit_rollout import reconstruct
from closure_discovery.data_generation.rd_solver_1d import random_fourier_initial_condition


def one(path):
    row=json.loads(Path(path).read_text());ident=row['id'];dest=ROOT/'evidence/paired'/f'{ident}.json'
    identity=validation_identity(row,'paired_clipping_v2')
    cached=cached_validation(dest,identity)
    if cached is not None:return ident,cached
    start=time.perf_counter();c,snap,methods=reconstruct(row);seed=row['config']['seed']
    rng=np.random.default_rng(seed+1000);x=np.arange(64)/64
    u0=np.stack([random_fourier_initial_condition(x,rng,c.value_range,4) for _ in range(4)])
    tt,truth,ti=strict_rollout(c.diffusion,c.reaction,u0,t_final=.5,adaptive=False)
    if ti['status']!='stable':raise FloatingPointError('Truth paired validation failed')
    result=dict(identity,id=ident,config=row['config'],classification='paired_correction_effect',data_seed=seed,init_seed=seed,
        conventions='same u0, nx64, RK4 dt1e-4, T.5, no model refit; change clipping only',methods={},compression_grid_effect=[])
    for name,(D,R) in methods.items():
        legacy_t,legacy,legacy_info=strict_rollout(D,R,u0,t_final=.5,adaptive=False,state_clip=(-.25,2.),allow_nonpositive=True)
        fixed_t,fixed,fixed_info=strict_rollout(D,R,u0,t_final=.5,adaptive=False,allow_nonpositive=True)
        entry=dict(legacy=legacy_info,unclipped=fixed_info,closure_errors_unchanged=True,BIR_unchanged=True)
        for horizon in [.1,.5]:
            a=np.flatnonzero(np.isclose(legacy_t,horizon));b=np.flatnonzero(np.isclose(fixed_t,horizon));target=np.flatnonzero(np.isclose(tt,horizon))[0]
            entry[f'legacy_error_T{horizon:g}']=relative_l2_error(truth[target],legacy[a[0]]) if len(a) else None
            entry[f'unclipped_error_T{horizon:g}']=relative_l2_error(truth[target],fixed[b[0]]) if len(b) else None
            if len(a) and len(b):entry[f'paired_error_difference_T{horizon:g}']=entry[f'unclipped_error_T{horizon:g}']-entry[f'legacy_error_T{horizon:g}']
        result['methods'][name]=entry
    # Fit-support density is a protocol choice, not a defect correction.
    q=load_curves(ident,epoch=row['config']['epochs']);evaluation=np.linspace(*c.value_range,256)
    nd=np.interp(evaluation,q['grid'],q['D']);nr=np.interp(evaluation,q['grid'],q['R'])
    for n in [256,512]:
        grid=np.linspace(*c.value_range,n);d=np.interp(grid,q['grid'],q['D']);r=np.interp(grid,q['grid'],q['R']);ed,er=fit_symbols(grid,d,r)
        result['compression_grid_effect'].append(dict(points=n,family_D=ed.family,family_R=er.family,
            error_D=relative_l2_error(c.diffusion(evaluation),ed.evaluate(evaluation)),error_R=relative_l2_error(c.reaction(evaluation),er.evaluate(evaluation)),
            BIR_D=float(np.linalg.norm(c.diffusion(evaluation)-ed.evaluate(evaluation))/np.linalg.norm(c.diffusion(evaluation)-nd)),
            BIR_R=float(np.linalg.norm(c.reaction(evaluation)-er.evaluate(evaluation))/np.linalg.norm(c.reaction(evaluation)-nr))))
    result['seconds']=time.perf_counter()-start;result['status']='complete';json_write(dest,result);return ident,'complete'


def main():
    p=argparse.ArgumentParser(epilog='Common paths: --work-root PATH (output/cache root), --input-root PATH (explicit artifact input root); defaults use the project directory.');p.add_argument('--workers',type=int,default=2);p.add_argument('--case',default='case_exp');p.add_argument('--limit',type=int);p.add_argument('--watch',action='store_true');p.add_argument('--expected',type=int,default=15);a=p.parse_args()
    if a.watch:
        submitted=set();futures={}
        with process_pool(a.workers) as pool:
            while True:
                for path in input_glob('evidence/raw', 'main_*.json'):
                    row=json.loads(path.read_text());cfg=row['config']
                    if row['status']=='complete' and cfg['case']==a.case and cfg['setting'] in ['clean','noise3','noise5'] and path.stem not in submitted:
                        submitted.add(path.stem);futures[pool.submit(one,path)]=path.stem
                for fut in list(futures):
                    if fut.done():report_completed([fut.result()]);futures.pop(fut)
                if len(submitted)>=a.expected and not futures:break
                time.sleep(5)
        return
    files=[]
    for f in input_glob('evidence/raw', 'main_*.json'):
        r=json.loads(f.read_text());cfg=r['config']
        if r['status']=='complete' and cfg['case']==a.case and cfg['setting'] in ['clean','noise3','noise5']:files.append(f)
    if a.limit:files=files[:a.limit]
    with process_pool(a.workers) as pool:
        report_completed(f.result() for f in cf.as_completed([pool.submit(one,path) for path in files]))


if __name__=='__main__':main()
