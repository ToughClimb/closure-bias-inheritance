"""Independent implicit integrator and targeted grid audit of Burgers rollouts.

Initial centered-scheme records are preserved. Refinement targets are selected
by numerical diagnostics only, not by physical accuracy or favorable results.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from burgers_core import conservative_rhs,periodic_initial,NU,cole_hopf_periodic
from burgers_campaign import save,evaluations


def implicit_rollout(evaluate,kind,n,rtol=2e-9,atol=2e-11):
    x=np.linspace(-1,1,n,endpoint=False);dx=2/n;u0=periodic_initial(x,kind)
    index=np.arange(n);left=(index-1)%n;right=(index+1)%n
    def jac(t,u):
        speed=evaluate(u,1)
        return coo_matrix((np.r_[np.full(n,-2*NU/dx**2),speed[left]/(2*dx)+NU/dx**2,-speed[right]/(2*dx)+NU/dx**2],
                           (np.r_[index,index,index],np.r_[index,left,right])),shape=(n,n)).tocsc()
    result=solve_ivp(lambda t,u:conservative_rhs(u,dx,evaluate),(0,1.5),u0,method='Radau',jac=jac,
                    rtol=rtol,atol=atol,dense_output=True)
    if not result.success or not np.all(np.isfinite(result.y)):raise RuntimeError(result.message)
    accepted=result.y.T;times=np.array([0.,.5,1.5]);states=result.sol(times).T
    return dict(x=x,t=times,u=states,maximum_principle_violation=float(max(0.,accepted.max()-u0.max(),u0.min()-accepted.min())),
        mass_drift=float(np.max(abs(np.mean(accepted,axis=1)-np.mean(u0)))),
        energy_increase=float(np.max(np.diff(np.mean(accepted**2,axis=1)))),accepted_steps=len(result.t)-1,nfev=result.nfev)


def main(output):
    records=[json.loads(p.read_text()) for p in (output/'forward').glob('*.json')]
    target=max(records,key=lambda r:max(r['grid_rms_difference']))
    state_cases=[r for r in records if r['maximum_principle_violation']>1e-4]
    plan=dict(reason='Time-tolerance test produced identical fixed-max-step paths; independently change to implicit Radau. Two centered-scheme state overshoots exceed 1e-4.',
        integrator='Radau with independently assembled analytic sparse Jacobian',rtol=2e-9,atol=2e-11,
        time_comparison_case=target['key'],spatial_cases=[r['key'] for r in state_cases],
        spatial_grids=[512,1024],stop_if_state_overshoot_below=1e-6,spatial_rms_difference_tolerance=.002,
        preserve_initial_records=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    protocol=output/'INDEPENDENT_FORWARD_AUDIT_PROTOCOL.json'
    if protocol.exists() and json.loads(protocol.read_text())!=plan:raise ValueError('Audit protocol mismatch')
    save(protocol,plan)
    fit=json.loads((output/'fits'/f'{target["run"]}.json').read_text())
    evaluate=dict(evaluations(fit,output))[target['method']]
    independent=implicit_rollout(evaluate,target['kind'],target['n'])
    earlier=np.load(target['array_path'])['u']
    time_difference=float(np.max(np.sqrt(np.mean((earlier-independent['u'])**2,axis=1))))
    audited=[]
    for row in state_cases:
        fit=json.loads((output/'fits'/f'{row["run"]}.json').read_text())
        evaluate=dict(evaluations(fit,output))[row['method']]
        previous=np.load(row['array_path'])['u'];previous_grid=row['n']
        stages=[]
        for n in plan['spatial_grids']:
            start=time.time();result=implicit_rollout(evaluate,row['kind'],n)
            discrepancy=float(np.max(np.sqrt(np.mean((previous-result['u'][:,::(n//previous_grid)])**2,axis=1))))
            reference=np.array([cole_hopf_periodic(result['x'],t,row['kind']) for t in result['t']])
            errors=np.sqrt(np.mean((result['u']-reference)**2,axis=1))
            denominator=np.sqrt(np.mean((reference-np.mean(reference,axis=1)[:,None])**2,axis=1))
            targetfile=output/'forward_refinement_arrays'/f'{row["key"]}_n{n}_radau.npz';targetfile.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(targetfile,x=result['x'],t=result['t'],u=result['u'],reference=reference)
            record=dict(key=row['key'],n=n,maximum_principle_violation=result['maximum_principle_violation'],
                grid_rms_difference=discrepancy,endpoint_rms_error=errors.tolist(),endpoint_relative_fluctuation_error=(errors/denominator).tolist(),
                mass_drift=result['mass_drift'],energy_increase=result['energy_increase'],nfev=result['nfev'],
                accepted_steps=result['accepted_steps'],array_path=str(targetfile),
                array_sha256=hashlib.sha256(targetfile.read_bytes()).hexdigest(),seconds=time.time()-start)
            stages.append(record)
            print(row['key'],n,record['maximum_principle_violation'],discrepancy,round(time.time()-start,1),flush=True)
            previous=result['u'];previous_grid=n
            if result['maximum_principle_violation']<=1e-6 and discrepancy<=.002:break
        audited.append(dict(key=row['key'],original_state_overshoot=row['maximum_principle_violation'],stages=stages,
            passed=stages[-1]['maximum_principle_violation']<=1e-6 and stages[-1]['grid_rms_difference']<=.002))
    result=dict(time_integrator_rms_difference=time_difference,time_integrator_passed=time_difference<1e-6,
                state_cases=audited,all_passed=time_difference<1e-6 and all(r['passed'] for r in audited))
    save(output/'INDEPENDENT_FORWARD_AUDIT.json',result)
    if not result['all_passed']:raise ValueError('Independent forward audit requires further numerical investigation')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
