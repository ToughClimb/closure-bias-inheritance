"""Continue only coefficient fits that exhausted the initial optimizer budget.

Every input receives an allocation of 40 initial plus at most 80 additional
function evaluations. Continuation depends only on optimizer termination status,
never on true closure error. The original fixed expression and saved observation
arrays are reused, and the initial 40-evaluation records remain unchanged.
"""
from pathlib import Path
import argparse
import copy
import sys
import time
import json
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from run_pysr_observation_refit import (ExpressionTemplate,fit_to_observations,digest,ROOT,
    source_signature,extra_case,curve_metrics,strict_rollout,relative_l2_error,json_write,npz_write,
    process_pool,report_completed,random_fourier_initial_condition)


class StartingTemplate:
    def __init__(self,template,parameters):
        self.template=template;self.start=np.asarray(parameters,dtype=float)
        if len(self.start)!=len(template.constants):raise ValueError('Starting coefficients do not match the original template')
    @property
    def constants(self):return self.template.constants
    @property
    def parameters(self):return self.start.copy()
    def evaluate(self,parameters,u):return self.template.evaluate(parameters,u)
    def expression(self,parameters):return self.template.expression(parameters)


def work(path,base_root,additional_nfev):
    base=json.loads(path.read_text());ident=base['id'];dest=ROOT/'evidence/pysr_observation_refit'/path.name
    if base['status']!='complete':raise ValueError(f'Incomplete initial fit: {ident}')
    observed_path=base_root/'evidence/observations'/f'{ident}.npz'
    identity=dict(base_record_sha256=digest(path),observation_file_sha256=digest(observed_path),
        experiment_source_sha256=digest(__file__),coefficient_fit_source_sha256=digest(Path(__file__).with_name('run_pysr_observation_refit.py')),
        source_signature=source_signature(),additional_nfev=additional_nfev)
    if dest.exists():
        prior=json.loads(dest.read_text())
        if any(prior.get(key)!=value for key,value in identity.items()):raise ValueError('Continuation cache identity mismatch')
        return ident,'cached' if prior['status']=='complete' else 'cached_'+prior['status']
    result=copy.deepcopy(base);result.update(identity,status='running',classification='optimizer_budget_continuation')
    result['fit_40']=copy.deepcopy(base['fit']);result['metrics_40']=copy.deepcopy(base['metrics'])
    result['validation_40']=copy.deepcopy(base['validation']);start=time.perf_counter()
    try:
        with np.load(observed_path) as values:observed={key:values[key].copy() for key in values.files}
        import hashlib
        if hashlib.sha256(observed['u'].tobytes()).hexdigest()!=base['data_hash']:
            raise ValueError('Saved observation bytes changed after the initial fit')
        templates=[ExpressionTemplate.compile(base['templates'][b]['original_expression']) for b in ('D','R')]
        start_parameters=[base['fit']['parameters_'+b] for b in ('D','R')]
        continued=not base['fit']['success'] and base['fit']['status']==0
        result['continued']=continued
        if continued:
            warmed=[StartingTemplate(template,parameters) for template,parameters in zip(templates,start_parameters,strict=True)]
            parameters,fit=fit_to_observations(observed,extra_case(base['config']['case']).value_range,warmed,max_nfev=additional_nfev)
            result['fit']=fit;result['total_nfev']=base['fit']['nfev']+fit['nfev']
        else:
            parameters=np.r_[*start_parameters];result['total_nfev']=base['fit']['nfev']
        result['allocated_total_nfev']=base['max_nfev']+additional_nfev
        case=extra_case(base['config']['case']);split=len(templates[0].constants)
        D=lambda z:templates[0].evaluate(parameters[:split],z);R=lambda z:templates[1].evaluate(parameters[split:],z)
        grid=np.linspace(*case.value_range,256);result['metrics']=curve_metrics(case,grid,D,R,observed)
        result['paired_difference']={b:result['metrics'][b]-base['float64_frozen_metrics'][b] for b in ('D','R')}
        result['validation']={};x=np.arange(64)/64;rng=np.random.default_rng(base['config']['seed']+1000)
        shapes=np.stack([random_fourier_initial_condition(x,rng,(0.,1.),4) for _ in range(4)])
        initial=[template.parameters for template in templates]
        for label,metadata in base['validation'].items():
            u0=(case.value_range[0]+shapes*(case.value_range[1]-case.value_range[0]) if label=='full_support' else
                metadata['midpoint']+metadata['half_range']*(2*shapes-1))
            if not np.allclose(u0.mean(-1),metadata['actual_initial_means'],rtol=0,atol=1e-14):raise ValueError('Validation initial conditions changed')
            times,truth,info=strict_rollout(case.diffusion,case.reaction,u0,t_final=.5)
            if info['status']!='stable':raise FloatingPointError('Reference validation failed')
            entries={}
            for name,functions in [('frozen',(lambda z:templates[0].evaluate(initial[0],z),lambda z:templates[1].evaluate(initial[1],z))),('refit',(D,R))]:
                tp,pred,validation=strict_rollout(*functions,u0,t_final=.5);n=min(len(times),len(tp))
                errors=[relative_l2_error(truth[i],pred[i]) for i in range(n)]
                entries[name]=dict(info=validation,times=tp,errors=errors,
                    final_error=errors[-1] if validation['status']=='stable' and len(tp)==len(times) else None)
            result['validation'][label]=dict(midpoint=metadata['midpoint'],half_range=metadata['half_range'],
                actual_initial_means=u0.mean(-1),methods=entries)
        npz_write(ROOT/'outputs/runs'/ident/'pysr_observation_refit.npz',grid=grid,true_D=case.diffusion(grid),true_R=case.reaction(grid),
            frozen_D=templates[0].evaluate(initial[0],grid),frozen_R=templates[1].evaluate(initial[1],grid),refit_D=D(grid),refit_R=R(grid))
        result['status']='complete'
    except Exception as error:result['status']='failed';result['error']=str(error);result['traceback']=traceback.format_exc()
    result['continuation_seconds']=time.perf_counter()-start;json_write(dest,result)
    return ident,result['status']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--additional-nfev',type=int,default=80)
    args=parser.parse_args();base=args.base_root.resolve()
    if args.workers<1 or args.additional_nfev<1:parser.error('Workers and budgets must be positive')
    paths=sorted((base/'evidence/pysr_observation_refit').glob('*.json'))
    if len(paths)!=30:raise ValueError('Continuation expects all 30 initial experiment records')
    json_write(ROOT/'evidence/manifest_pysr_observation_refit_continuation.json',dict(ids=[p.stem for p in paths],
        base_root=str(base),additional_nfev=args.additional_nfev,workers=args.workers,
        rule='Continue only optimizer-budget exhaustion; keep converged fits; no true-error selection'))
    import concurrent.futures as cf
    with process_pool(args.workers) as pool:
        report_completed(f.result() for f in cf.as_completed([pool.submit(work,path,base,args.additional_nfev) for path in paths]))


if __name__=='__main__':main()
