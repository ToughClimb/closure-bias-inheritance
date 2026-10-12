"""Validate previously fitted PySR pilot without refitting or replacing evidence."""
from revision_core import *
import sympy as sp
from closure_discovery.data_generation.rd_solver_1d import random_fourier_initial_condition


def main():
    for path in input_glob('evidence/pysr', '*.json'):
        r=json.loads(path.read_text())
        if r['status']!='complete' or 'rollout' in r:continue
        dest=ROOT/'evidence/pysr_validation'/path.name
        identity=validation_identity(r,'pysr_followup_rollout_v2')
        cached=cached_validation(dest,identity)
        if cached is not None:
            report_completed([(r['id'],cached)]);continue
        start=time.perf_counter();cpu=time.process_time();c=extra_case(r['config']['case']);seed=r['config']['seed'];fns={}
        for branch,e in r['expressions'].items():
            fn=sp.lambdify(sp.Symbol('x0'),sp.sympify(e['expression']),'numpy')
            fns[branch]=lambda u,f=fn:np.broadcast_to(f(u),np.shape(u)).astype(float)
        rng=np.random.default_rng(seed+1000);x=np.arange(64)/64
        initial=np.stack([random_fourier_initial_condition(x,rng,c.value_range,4) for _ in range(4)])
        times,truth,ti=strict_rollout(c.diffusion,c.reaction,initial,t_final=.5)
        if ti['status']!='stable':raise FloatingPointError('Truth PySR followup validation failed')
        tp,pred,info=strict_rollout(fns['D'],fns['R'],initial,t_final=.5)
        result=dict(identity,id=r['id'],config=r['config'],status='complete',classification='followup_validation_without_refit',
            rollout=dict(times=tp,errors=[relative_l2_error(truth[i],pred[i]) for i in range(min(len(times),len(tp)))],info=info),
            seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)
        json_write(dest,result);print(r['id'],info['status'])


if __name__=='__main__':main()
