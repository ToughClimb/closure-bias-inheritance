"""Paired boundary-prior experiment using the immutable same-observation inputs."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse,dataclasses,hashlib,json,shutil,sys,time,traceback,subprocess
from pathlib import Path
import concurrent.futures as cf
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'revision'),str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
from scipy.integrate import solve_ivp
from revision_core import ROOT,SOURCE_ROOT,source_signature,json_write,npz_write,extra_case,curve_metrics
from reliability_templates import symbolic_pair,denominator_roots
from anchor_control_core import ARMS,fit_anchor_control,setup
from evaluate_reliability_supplement import validate_models
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def protocol():
    p=json.loads((ROOT/'PROTOCOL.json').read_text())
    for relative,digest in p['numerical_source_hashes'].items():
        if sha(SOURCE_ROOT/relative)!=digest: raise ValueError('Numerical source changed: '+relative)
    if source_signature()!=p['core_source_signature']: raise ValueError('Core source changed')
    return p


def declare(source):
    old=json.loads((source/'evidence/PROTOCOL.json').read_text())
    config=dict(settings=['clean','noise3'],seeds=list(range(5)),arms=list(ARMS),first_budget=40,
        continuation_budget=80,soft_weight=1.,max_rhs=20000,initial='Set reaction a0=0 for every arm; retain all other teacher coefficients',
        scales='max(abs(common projected initial parameters),0.01), fixed across phases; hard arm selects free subset',
        hard_constraint='Fix reaction numerator/constant a0=0 inside selected family, one fewer free coefficient; no structure search',
        soft_constraint='Add R(0)^2 to the refit objective, fixed coefficient1; not equivalent to neural hybrid anchor strength',
        selection='No true-error stopping, tuning or best-restart selection',validation='Original eight pattern batches plus independent scalar u0=0 and u0=.01 checks',
        interpretation='Measure closure/observation error cost and small-state growth signs, not merely structural zero-state invariance')
    datasets=[]
    for d in old['datasets']:
        ident=d['id'];obs=source/d['observation_file'];train=source/'evidence/training'/f'{ident}.json'
        if sha(obs)!=d['observation_file_sha256']: raise ValueError('Source observations changed')
        training=json.loads(train.read_text()); stage=training['stages']['neural_adam_150']
        setup(symbolic_pair(stage['symbols']))
        dest=ROOT/'inputs'/obs.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(obs,dest)
        datasets.append(dict(id=ident,setting=d['setting'],seed=d['seed'],observation_file=str(dest.relative_to(ROOT)),
            observation_file_sha256=sha(dest),observation_u_sha256=d['observation_u_sha256'],
            teacher_record_sha256=sha(train),teacher_native_sha256=stage['native_file_sha256'],symbols=stage['symbols'],
            archived_unanchored_file=str(source/'evidence/fresh_refits'/f'{ident}.json'),
            archived_unanchored_sha256=sha(source/'evidence/fresh_refits'/f'{ident}.json')))
    sources=['experiments/anchor_control_core.py','experiments/run_anchor_controls.py','experiments/reliability_templates.py',
             'experiments/evaluate_reliability_supplement.py','experiments/run_pysr_observation_refit.py']
    p=dict(plan=config,datasets=datasets,source_campaign=str(source),core_source_signature=source_signature(),
           numerical_source_hashes={s:sha(SOURCE_ROOT/s) for s in sources})
    path=ROOT/'PROTOCOL.json'
    if path.exists() and json.loads(path.read_text())!=p: raise ValueError('Protocol already declared differently')
    json_write(path,p)
    for s in sources:
        dest=ROOT/'provenance/source'/s;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE_ROOT/s,dest)
    json_write(ROOT/'provenance/environment.json',dict(python=sys.version,numpy=np.__version__,
        command=sys.argv,git_status=subprocess.check_output(['git','status','--short'],cwd=SOURCE_ROOT,text=True)))
    print('Declared ten datasets and thirty paired fits',flush=True)


def observed(d):
    path=ROOT/d['observation_file']
    if sha(path)!=d['observation_file_sha256']: raise ValueError('Observation file changed')
    with np.load(path,allow_pickle=False) as a:result={k:a[k].copy() for k in a.files}
    if hashlib.sha256(result['u'].tobytes()).hexdigest()!=d['observation_u_sha256']:raise ValueError('Observation field changed')
    return result


def scalar_check(R,initial):
    # Integration failure is a scientific outcome and is retained.
    def rhs(t,v):
        val=np.asarray(R(v),dtype=float)
        if not np.isfinite(val).all():raise FloatingPointError('Nonfinite scalar reaction')
        return val
    def zero(t,v):return v[0]
    zero.direction=-1;zero.terminal=True
    started=time.perf_counter()
    try:
        sol=solve_ivp(rhs,(0,.1),[initial],method='DOP853',rtol=1e-11,atol=1e-13,max_step=1e-4,
                      events=zero if initial>0 else None)
        return dict(initial_state=initial,success=bool(sol.success),status=int(sol.status),message=str(sol.message),
            reached_time=float(sol.t[-1]),final_state=float(sol.y[0,-1]),minimum_state=float(sol.y.min()),
            zero_time=float(sol.t_events[0][0]) if initial>0 and len(sol.t_events[0]) else None,
            nfev=int(sol.nfev),seconds=time.perf_counter()-started)
    except (FloatingPointError,ValueError,OverflowError) as e:
        return dict(initial_state=initial,success=False,status='failed',error=str(e),seconds=time.perf_counter()-started)


def task(arg):
    d,arm=arg;protocol();path=ROOT/'fits'/f"{d['id']}_{arm}.json"
    identity=dict(dataset=d,arm=arm,core=source_signature(),script_sha256=sha(__file__))
    if path.exists():
        o=json.loads(path.read_text())
        if o['identity']!=identity:raise ValueError('Cache identity mismatch')
        if o['status']!='complete':raise ValueError('Existing failed result; retain and inspect before retry')
        return d['id'],arm,'cached'
    start=time.perf_counter();result=dict(identity=identity,status='running')
    try:
        data=observed(d);case=extra_case('case_exp');pair=symbolic_pair(d['symbols']);plan=protocol()['plan']
        coefficients,fit=fit_anchor_control(data,case.value_range,pair,arm,first_budget=plan['first_budget'],
            continuation_budget=plan['continuation_budget'],soft_weight=plan['soft_weight'],max_rhs=plan['max_rhs'])
        split=len(pair[0].parameters);templates=setup(pair)[0]
        symbols=[dataclasses.replace(e,parameters=tuple(p),expression=t.expression(p))
                 for e,t,p in zip(pair,templates,(coefficients[:split],coefficients[split:]),strict=True)]
        model=dict(kind='restricted',symbols=[dataclasses.asdict(s) for s in symbols])
        D=lambda v:templates[0].evaluate(coefficients[:split],v)
        R=lambda v:templates[1].evaluate(coefficients[split:],v)
        rp=coefficients[split:];derivative=float(rp[1] if pair[1].family=='polynomial' else rp[1]-rp[0]*rp[3])
        grid=np.linspace(*case.value_range,256)
        result.update(fit=fit,model=model,metrics=curve_metrics(case,grid,D,R,data),R_prime_zero=derivative,
            R_at_001=float(R(np.array([.01]))[0]),scalar={str(v):scalar_check(R,v) for v in (0.,.01)},
            poles_in_U=denominator_roots(symbols[1],case.value_range))
        if pair[1].family=='polynomial':roots=np.polynomial.polynomial.polyroots(rp)
        else:roots=np.polynomial.polynomial.polyroots(rp[:3])
        result['positive_reaction_roots']=[float(x.real) for x in roots if abs(x.imag)<1e-10 and 0<x.real<=1.8]
        npz_write(ROOT/'curves'/f"{d['id']}_{arm}.npz",grid=grid,D=D(grid),R=R(grid),true_D=case.diffusion(grid),true_R=case.reaction(grid))
        result['status']='complete'
    except Exception as e:result.update(status='failed',error=str(e),traceback=traceback.format_exc())
    result['seconds']=time.perf_counter()-start;json_write(path,result)
    print(json.dumps(dict(id=d['id'],arm=arm,status=result['status'],seconds=result['seconds'],
        fit_success=result.get('fit',{}).get('optimizer_success'))),flush=True)
    return d['id'],arm,result['status']


def validate(arg):
    d,arm=arg;protocol();fit_path=ROOT/'fits'/f"{d['id']}_{arm}.json";out=ROOT/'validation'/fit_path.name
    row=json.loads(fit_path.read_text())
    if row['status']!='complete':raise ValueError('Cannot validate an incomplete fit')
    ident=dict(fit_sha256=sha(fit_path),script_sha256=sha(__file__),core=source_signature())
    if out.exists():
        prior=json.loads(out.read_text())
        if prior['identity']!=ident or prior['status']!='complete':raise ValueError('Validation cache mismatch')
        return d['id'],arm,'cached'
    start=time.perf_counter();result=dict(identity=ident,status='running')
    try:
        result['models']=validate_models(extra_case('case_exp'),d['seed'],{arm:row['model']})
        result['status']='complete'
    except Exception as e:result.update(status='failed',error=str(e),traceback=traceback.format_exc())
    result['seconds']=time.perf_counter()-start;json_write(out,result)
    print(json.dumps(dict(validation=d['id'],arm=arm,status=result['status'],seconds=result['seconds'])),flush=True)
    return d['id'],arm,result['status']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['declare','fit','validate'],required=True)
    ap.add_argument('--source-campaign',type=Path);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--limit',type=int);args=ap.parse_args()
    if args.phase=='declare':
        if args.source_campaign is None:ap.error('Need --source-campaign')
        declare(args.source_campaign.resolve());return
    p=protocol();tasks=[(d,a) for d in p['datasets'] for a in ARMS]
    if args.limit:tasks=tasks[:args.limit]
    worker=task if args.phase=='fit' else validate
    if args.workers==1:results=[worker(t) for t in tasks]
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(worker,tasks))
    if any(r[-1] not in ('complete','cached') for r in results):raise RuntimeError('Some tasks failed; all outcomes are retained')
    print('All '+str(len(results))+' tasks complete',flush=True)

if __name__=='__main__':main()
