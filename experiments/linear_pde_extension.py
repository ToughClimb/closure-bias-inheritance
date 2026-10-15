"""Four linear-PDE operator-learning reliability experiments.

The teacher learns the local time-derivative map from a specified differential
library. This is an operator-learning calibration, not a neural constitutive
closure and not unconstrained symbolic structure discovery.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

# Support both direct script execution and python -m from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.basic_pde_controls import linear_fields, stencil_fields, linear_fit

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = {
    'heat': dict(D=.03, c=0., a=0., unknowns=['D']),
    'advection': dict(D=0., c=.4, a=0., unknowns=['c']),
    'advection_diffusion': dict(D=.03, c=.4, a=0., unknowns=['D', 'c']),
    'reaction_diffusion': dict(D=.03, c=0., a=.8, unknowns=['D', 'a']),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def put_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def protocol():
    return dict(
        formulation='Local neural operator ut=N(uxx,-ux,u), retaining only the known nonzero PDE terms; not a state surrogate or nonlinear constitutive discovery',
        systems=SYSTEMS, training_grids=[32, 64], noise_relative_rms=[0., .01], seeds=list(range(5)),
        training=dict(modes=[0, 1, 2], amplitudes=[.3, .5, .2], time_end=.2, time_samples=21,
                      derivatives='Centered second-order finite differences; periodic space; interior times',
                      noise='Independent Gaussian noise on u, sigma=noise_relative_rms*rms(clean_u), no clipping; same observed u for every arm'),
        teacher=dict(hidden_width=16, hidden_layers=2, activation='tanh', architecture='linear skip plus g(X)-g(0), ensuring N(0)=0',
                     normalization='Per-feature RMS and target RMS from the noisy training residuals only',
                     dtype='float64', device='cpu', threads=1, adam_steps=300, adam_lr=.01,
                     lbfgs_max_iter=100, lbfgs_max_eval=125, lbfgs_tolerance_grad=1e-9,
                     lbfgs_tolerance_change=1e-12, history_size=30, selection='Final budgeted state, no truth-based selection'),
        continuation=dict(grids=[64], lbfgs_max_iter=200, lbfgs_max_eval=250,
                          interpretation='Paired continuation of every nx64 initial state with a fresh optimizer; not additional independent datasets'),
        symbolic=dict(library='Same specified linear differential terms, no constant, no structure search',
                      frozen='OLS projection of frozen teacher on a uniform tensor grid in normalized feature coordinates [-2,2], 41 points/axis',
                      direct='Global scaled OLS on the exact same noisy residual pairs; refitting the fixed symbolic family to these residuals is identical and counted once',
                      physical_constraints='None; negative diffusion and invariant violations retained'),
        evaluation=dict(calibration_grid_points_per_axis=41, calibration_grid_halfwidth=2.,
                        rollout_grids=[32, 64], rollout_times=[0., .05, .1, .2, .4, .8],
                        initial_conditions={'training_modes':dict(modes=[0,1,2], amplitudes=[.3,.5,.2]),
                                            'heldout_mode3':dict(modes=[0,3], amplitudes=[.3,.2]),
                                            'positive_low_modes':dict(modes=[0,1,2], amplitudes=[.3,.1,.05])},
                        teacher_solver='Fourier pseudospectral method of lines, scipy RK45, rtol=1e-7, atol=1e-9',
                        teacher_rhs_budget=12000, state_absolute_limit=10.,
                        truth='Exact continuum Fourier solution; symbolic rollout also exact, independently checked through Fourier method of lines',
                        comparison='Teacher coarse/fine differences reported; unresolved rollouts do not become validated successes',
                        physical_checks='Mass balance when a=0; exponential mean law when a!=0; L2/amplitude/phase; minimum on positive initial conditions; derivative sign on calibration grid'),
        identifiability='Exact derivative single-mode RD rank 1 versus two-mode and constant-plus-mode rank 2; finite-difference noisy full rank is not evidence of population identifiability',
        counts=dict(independent_observation_tables=80, initial_teachers=80, paired_continued_states=40),
    )


def declare(output):
    output.mkdir(parents=True, exist_ok=True)
    record = dict(protocol=protocol(), source_sha256=sha(__file__))
    path = output / 'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError('Protocol/source mismatch; preserve this run and choose a new output directory')
    else:
        put_json(path, record)
        shutil.copy2(__file__, output / 'source.py')
        put_json(output / 'ENVIRONMENT.json', dict(command=sys.argv, python=sys.version,
            packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','torch','matplotlib','pytest']},
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)))
    return record


def fields(x, t, system, modes=(0,1,2), amplitudes=(.3,.5,.2), coefficients=None):
    p = SYSTEMS[system].copy()
    if coefficients is not None: p.update(coefficients)
    return linear_fields(x,t,diffusion=p['D'],speed=p['c'],reaction=p['a'],
                         modes=modes,amplitudes=list(amplitudes))


def feature_matrix(f, unknowns):
    return np.column_stack([{'D':f['uxx'], 'c':-f['ux'], 'a':f['u']}[k].ravel() for k in unknowns])


def scaled_ols(X, y):
    scales = np.sqrt(np.mean(X*X,axis=0))
    if np.any(scales == 0):
        return dict(status='unidentifiable', rank=int(np.linalg.matrix_rank(X)), coefficients=None, singular_values=[])
    normalized = X/scales
    singular = np.linalg.svd(normalized,compute_uv=False)
    rank = int(np.linalg.matrix_rank(normalized))
    if rank < X.shape[1]:
        return dict(status='unidentifiable',rank=rank,coefficients=None,singular_values=singular.tolist())
    coefficients = np.linalg.lstsq(normalized,y,rcond=None)[0]/scales
    return dict(status='identified',rank=rank,coefficients=coefficients.tolist(),singular_values=singular.tolist(),
                residual_mse=float(np.mean((X@coefficients-y)**2)))


class LocalOperator(torch.nn.Module):
    def __init__(self, features, width=16):
        super().__init__()
        self.linear = torch.nn.Linear(features,1,bias=False,dtype=torch.float64)
        self.hidden1 = torch.nn.Linear(features,width,dtype=torch.float64)
        self.hidden2 = torch.nn.Linear(width,width,dtype=torch.float64)
        self.output = torch.nn.Linear(width,1,bias=False,dtype=torch.float64)

    def nonlinear(self, x):
        return self.output(torch.tanh(self.hidden2(torch.tanh(self.hidden1(x)))))

    def forward(self,x):
        return self.linear(x)+self.nonlinear(x)-self.nonlinear(torch.zeros((1,x.shape[1]),dtype=x.dtype,device=x.device))


def train(model, X, y, continuation=False):
    p=protocol()['teacher']; cp=protocol()['continuation']
    Xt=torch.as_tensor(X,dtype=torch.float64); yt=torch.as_tensor(y[:,None],dtype=torch.float64)
    history=[]; started=time.monotonic()
    def loss():return torch.mean((model(Xt)-yt)**2)
    if not continuation:
        optimizer=torch.optim.Adam(model.parameters(),lr=p['adam_lr'])
        for iteration in range(p['adam_steps']):
            optimizer.zero_grad(); objective=loss(); objective.backward(); optimizer.step()
            if iteration%50==0 or iteration==p['adam_steps']-1:
                history.append(dict(stage='adam',iteration=iteration,objective=float(objective.detach())))
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,
        max_iter=cp['lbfgs_max_iter'] if continuation else p['lbfgs_max_iter'],
        max_eval=cp['lbfgs_max_eval'] if continuation else p['lbfgs_max_eval'],
        tolerance_grad=p['lbfgs_tolerance_grad'],tolerance_change=p['lbfgs_tolerance_change'],
        history_size=p['history_size'],line_search_fn='strong_wolfe')
    closures=0
    def closure():
        nonlocal closures
        optimizer.zero_grad(); objective=loss()
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite training objective')
        objective.backward(); closures+=1
        return objective
    optimizer.step(closure)
    final=closure()
    gradient=float(torch.sqrt(sum(torch.sum(p.grad*p.grad) for p in model.parameters())))
    state=optimizer.state[next(iter(model.parameters()))]
    history.append(dict(stage='lbfgs',iteration=int(state.get('n_iter',0)),objective=float(final.detach())))
    return dict(history=history,gradient_l2=gradient,gradient_l2_below_1e9=gradient<=1e-9,
                closure_evaluations=closures,elapsed_seconds=time.monotonic()-started,
                objective=float(final.detach()),status='finite_budget_completed')


def numpy_predictor(model, xs, ys):
    state={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
    W=state['linear.weight'][0]
    W1,b1=state['hidden1.weight'],state['hidden1.bias']
    W2,b2=state['hidden2.weight'],state['hidden2.bias']
    W3=state['output.weight'][0]
    origin=np.tanh(np.tanh(b1)@W2.T+b2)@W3
    def predict(X):
        Z=np.asarray(X)/xs
        return ys*(Z@W+np.tanh(np.tanh(Z@W1.T+b1)@W2.T+b2)@W3-origin)
    return predict


def calibration(model, xs, ys, system):
    p=protocol()['evaluation']; unknowns=SYSTEMS[system]['unknowns']
    axis=np.linspace(-p['calibration_grid_halfwidth'],p['calibration_grid_halfwidth'],p['calibration_grid_points_per_axis'])
    Z=np.stack(np.meshgrid(*([axis]*len(unknowns)),indexing='ij'),axis=-1).reshape(-1,len(unknowns))
    X=Z*xs
    prediction=numpy_predictor(model,xs,ys)(X)
    true_coefficients=np.array([SYSTEMS[system][k] for k in unknowns])
    truth=X@true_coefficients
    fit=scaled_ols(X,prediction)
    compressed=X@np.array(fit['coefficients'])
    denominator=np.mean(truth**2)
    eN=np.mean((prediction-truth)**2); eS=np.mean((compressed-truth)**2); eC=np.mean((compressed-prediction)**2)
    Zt=torch.tensor(Z,dtype=torch.float64,requires_grad=True)
    evaluated=model(Zt)
    jac=torch.autograd.grad(evaluated.sum(),Zt)[0].detach().numpy()*ys/xs
    derivatives={name:dict(minimum=float(jac[:,i].min()),maximum=float(jac[:,i].max())) for i,name in enumerate(unknowns)}
    return dict(coefficients=fit['coefficients'],teacher_relative_error=float(np.sqrt(eN/denominator)),
                symbolic_relative_error=float(np.sqrt(eS/denominator)),compression_relative_error=float(np.sqrt(eC/denominator)),
                teacher_squared_error=float(eN),symbolic_squared_error=float(eS),compression_squared_error=float(eC),
                projection_identity_residual=float(abs(eN-eS-eC)),
                teacher_fraction_retained=float(eS/eN) if eN>0 else 0.,sampled_operator_derivatives=derivatives),dict(X=X,truth=truth,teacher=prediction,symbolic=compressed)


def spectral_features(u,unknowns):
    n=len(u); k=2*np.pi*np.fft.fftfreq(n,d=1/n)
    fft=np.fft.fft(u)
    values=dict(u=u,ux=np.fft.ifft(1j*k*fft).real,uxx=np.fft.ifft(-k*k*fft).real)
    return feature_matrix(values,unknowns)


class RolloutLimit(RuntimeError):pass


def integrate_operator(predict, unknowns, initial, times, *, rtol=1e-7, atol=1e-9, rhs_budget=12000, state_limit=10.):
    calls=0
    def rhs(t,u):
        nonlocal calls
        calls+=1
        if calls>rhs_budget:raise RolloutLimit('RHS evaluation budget exceeded')
        if not np.all(np.isfinite(u)):raise RolloutLimit('Nonfinite intermediate state')
        if np.max(np.abs(u))>state_limit:raise RolloutLimit('Intermediate absolute state exceeded declared limit')
        derivative=predict(spectral_features(u,unknowns))
        if not np.all(np.isfinite(derivative)):raise RolloutLimit('Nonfinite learned derivative')
        return derivative
    try:
        solution=solve_ivp(rhs,(times[0],times[-1]),initial,t_eval=times,rtol=rtol,atol=atol,method='RK45')
    except RolloutLimit as error:
        return dict(status='failed',reason=str(error),rhs_evaluations=calls),None
    if not solution.success:
        return dict(status='failed',reason=solution.message,rhs_evaluations=calls),None
    return dict(status='completed',reason=solution.message,rhs_evaluations=calls),solution.y.T


def rollout_metrics(candidate,reference,times,system,initial_name):
    norm=np.linalg.norm(reference,axis=1)
    relative=np.linalg.norm(candidate-reference,axis=1)/norm
    expected_mean=reference.mean(axis=1)
    return dict(relative_error_t02=float(relative[list(times).index(.2)]),relative_error_t08=float(relative[-1]),
                absolute_rms_error_t08=float(np.sqrt(np.mean((candidate[-1]-reference[-1])**2))),
                true_absolute_rms_t08=float(np.sqrt(np.mean(reference[-1]**2))),
                predicted_absolute_rms_t08=float(np.sqrt(np.mean(candidate[-1]**2))),
                max_mean_balance_error=float(np.max(np.abs(candidate.mean(axis=1)-expected_mean))),
                minimum_state=float(candidate.min()),maximum_state=float(candidate.max()),
                positive_initial_violation=bool(initial_name=='positive_low_modes' and candidate.min() < -1e-8),
                l2_ratio_t08=float(np.linalg.norm(candidate[-1])/np.linalg.norm(candidate[0])),
                truth_l2_ratio_t08=float(np.linalg.norm(reference[-1])/np.linalg.norm(reference[0])))


def run_rollouts(output,run_id,model,xs,ys,system,coefficients,method):
    p=protocol()['evaluation']; unknowns=SYSTEMS[system]['unknowns']
    times=np.array(p['rollout_times']); rows=[]; arrays={}; summaries={}
    predict=numpy_predictor(model,xs,ys) if method=='teacher' else lambda X:X@np.asarray(coefficients)
    for initial_name,initial_spec in p['initial_conditions'].items():
        curves={}
        for nx in p['rollout_grids']:
            x=np.arange(nx)/nx
            reference=fields(x,times,system,**initial_spec)['u']
            meta,candidate=integrate_operator(predict,unknowns,reference[0],times,
                rhs_budget=p['teacher_rhs_budget'],state_limit=p['state_absolute_limit'])
            row=dict(run_id=run_id,method=method,initial_condition=initial_name,grid=nx,**meta)
            if method!='teacher':
                symbolic={k:float(v) for k,v in zip(unknowns,coefficients,strict=True)}
                exact=fields(x,times,system,coefficients=symbolic,**initial_spec)['u']
                row.update({'exact_'+k:v for k,v in rollout_metrics(exact,reference,times,system,initial_name).items()})
                arrays[f'{initial_name}_n{nx}_exact']=exact
            if candidate is not None:
                row.update(rollout_metrics(candidate,reference,times,system,initial_name))
                curves[nx]=candidate
                arrays[f'{initial_name}_n{nx}']=candidate
                if method!='teacher':
                    row['symbolic_solver_exact_max_difference']=float(np.max(np.abs(candidate-exact)))
            rows.append(row)
        convergence=None
        if len(curves)==len(p['rollout_grids']):
            coarse,fine=(curves[n] for n in p['rollout_grids'])
            convergence=float(np.max(np.abs(coarse-fine[:,::2])))
        summaries[initial_name]=dict(coarse_fine_max_difference=convergence,
            grid_agreement_pass=bool(convergence is not None and convergence<=1e-4))
    if arrays:
        (output/'rollouts').mkdir(exist_ok=True)
        np.savez_compressed(output/'rollouts'/f'{run_id}_{method}.npz',times=times,**arrays)
    return dict(rows=rows,grid_checks=summaries)


def on_solution_errors(model,xs,ys,system,coefficients):
    """Independent physically attainable jets, distinct from the box projection norm."""
    x=np.arange(128)/128;t=np.linspace(0,.8,33)
    unknowns=SYSTEMS[system]['unknowns']; predict=numpy_predictor(model,xs,ys)
    rows=[]
    for name,spec in protocol()['evaluation']['initial_conditions'].items():
        f=fields(x,t,system,**spec);X=feature_matrix(f,unknowns); truth=f['ut'].ravel()
        for method,values in [('teacher',predict(X)),('frozen',X@np.asarray(coefficients))]:
            rows.append(dict(initial_condition=name,method=method,
                relative_operator_error=float(np.linalg.norm(values-truth)/np.linalg.norm(truth)),
                absolute_operator_rms_error=float(np.sqrt(np.mean((values-truth)**2))),
                truth_operator_rms=float(np.sqrt(np.mean(truth**2)))))
    return rows


def deterministic_checks(output):
    x=np.arange(64)/64;t=np.linspace(0,.2,21); rows=[]
    for modes in [(1,),(1,2),(0,1)]:
        exact=fields(x,t,'reaction_diffusion',modes=modes,amplitudes=[1/len(modes)]*len(modes))
        result=linear_fit(exact,('D','a'))
        rows.append(dict(modes=list(modes),**result))
    if [r['rank'] for r in rows]!=[1,2,2]:raise AssertionError('Incorrect identifiability ranks')
    put_json(output/'IDENTIFIABILITY.json',rows)
    independent=[]
    for system,p in SYSTEMS.items():
        f=fields(x,t,system)
        residual=f['ut']-p['D']*f['uxx']+p['c']*f['ux']-p['a']*f['u']
        independent.append(dict(system=system,analytical_residual_max=float(np.max(np.abs(residual)))))
    if max(r['analytical_residual_max'] for r in independent)>1e-12:raise AssertionError('Manufactured solution failed')
    put_json(output/'ANALYTICAL_CHECKS.json',independent)


def execute_case(output,system,nx,noise,seed):
    case=f'{system}_n{nx}_noise{noise:g}_seed{seed}'
    complete=output/'cases'/f'{case}.json'
    if complete.exists():return json.loads(complete.read_text())
    p=protocol(); q=p['training']; unknowns=SYSTEMS[system]['unknowns']
    x=np.arange(nx)/nx;t=np.linspace(0,q['time_end'],q['time_samples'])
    clean=fields(x,t,system,modes=q['modes'],amplitudes=q['amplitudes'])['u']
    sigma=noise*float(np.sqrt(np.mean(clean*clean)))
    epsilon=np.random.default_rng(seed).normal(size=clean.shape)
    observed=clean+sigma*epsilon
    deriv=stencil_fields(observed,1/nx,t[1]-t[0])
    X=feature_matrix(deriv,unknowns);y=deriv['ut'].ravel()
    xs=np.sqrt(np.mean(X*X,axis=0));ys=float(np.sqrt(np.mean(y*y)))
    for directory in ['inputs','models','calibration','cases','failures']:(output/directory).mkdir(exist_ok=True)
    input_path=output/'inputs'/f'{case}.npz'
    np.savez_compressed(input_path,x=x,t=t,clean=clean,observed=observed,standard_noise=epsilon,X=X,y=y,x_scale=xs,y_scale=ys)
    direct=scaled_ols(X,y)
    if direct['status']!='identified':raise AssertionError('Declared multiexcitation training should be identifiable')
    torch.manual_seed(seed)
    model=LocalOperator(len(unknowns),p['teacher']['hidden_width'])
    initial={k:v.clone() for k,v in model.state_dict().items()}
    torch.save(initial,output/'models'/f'{case}_initial.pt')
    result=dict(case=case,system=system,grid=nx,noise=noise,noise_sigma=sigma,seed=seed,
                input_sha256=sha(input_path),direct=direct,states=[])
    stages=['base','continued'] if nx in p['continuation']['grids'] else ['base']
    for stage in stages:
        run_id=f'{case}_{stage}'
        optimization=train(model,X/xs,y/ys,continuation=(stage=='continued'))
        checkpoint=output/'models'/f'{run_id}.pt'
        torch.save(dict(state_dict=model.state_dict(),x_scale=xs,y_scale=ys,system=system,unknowns=unknowns),checkpoint)
        metrics,cal_arrays=calibration(model,xs,ys,system)
        np.savez_compressed(output/'calibration'/f'{run_id}.npz',**cal_arrays)
        # Validate the fast NumPy evaluator against the actual trained PyTorch teacher.
        with torch.no_grad():check=model(torch.tensor(X/xs,dtype=torch.float64)).numpy().ravel()*ys
        numpy_error=float(np.max(np.abs(check-numpy_predictor(model,xs,ys)(X))))
        if numpy_error>1e-10:raise AssertionError('Numpy checkpoint predictor differs from PyTorch')
        state=dict(stage=stage,run_id=run_id,optimization=optimization,calibration=metrics,
                   checkpoint_sha256=sha(checkpoint),numpy_torch_max_difference=numpy_error,
                   residual_mse=float(np.mean((check-y)**2)),rollouts={})
        state['on_solution_errors']=on_solution_errors(model,xs,ys,system,metrics['coefficients'])
        for method,coef in [('teacher',None),('frozen',metrics['coefficients'])]:
            state['rollouts'][method]=run_rollouts(output,run_id,model,xs,ys,system,coef,method)
        # Direct same-observation fit is the same for base and continuation.
        if stage=='base':
            result['direct_rollouts']=run_rollouts(output,case,model,xs,ys,system,direct['coefficients'],'direct')
        result['states'].append(state)
        put_json(output/'cases'/f'{run_id}.partial.json',state)
        print(json.dumps(dict(case=case,stage=stage,loss=state['residual_mse'],
            operator_teacher_error=metrics['teacher_relative_error'],operator_symbolic_error=metrics['symbolic_relative_error'])),flush=True)
    result['status']='completed_with_rollout_statuses_recorded'
    put_json(complete,result)
    return result


def run(output):
    torch.set_num_threads(1)
    declare(output);deterministic_checks(output)
    results=[]
    for system in SYSTEMS:
        for nx in protocol()['training_grids']:
            for noise in protocol()['noise_relative_rms']:
                for seed in protocol()['seeds']:
                    try:results.append(execute_case(output,system,nx,noise,seed))
                    except Exception as error:
                        failure=dict(system=system,grid=nx,noise=noise,seed=seed,type=type(error).__name__,message=str(error))
                        put_json(output/'failures'/f'{system}_n{nx}_noise{noise:g}_seed{seed}.json',failure)
                        raise
    put_json(output/'RUN_COMPLETE.json',dict(completed_cases=len(results),teacher_states=sum(len(r['states']) for r in results),
        source_sha256=sha(__file__),status='all_fits_finished_rollout_failures_retained'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    if args.declare_only:declare(args.output)
    else:run(args.output)
