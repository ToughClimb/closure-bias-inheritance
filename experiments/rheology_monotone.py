"""Post-result, fixed-margin intervention on real rheometry polynomial models.

Bernstein coefficient constraints provide a sufficient global certificate for
positive incremental stress inside each training normalization rectangle.
The observations are real measurements; held-out values are not exact truth.
"""
from __future__ import annotations
import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import minimize,nnls
from scipy.special import comb
from rheology_core import features,powers,design,polynomial_stress_margin
from rheology_evaluate import Predictor,metrics,processing_reference,evaluate_processing

ROOT=Path(__file__).resolve().parents[1]
DEGREE=6
MARGIN=.05
GUARD=1e-8


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temp.replace(path)


def power_to_bernstein(power,n=5):
    """Exact rational coefficients of (2s-1)^power in degree-n Bernstein basis."""
    if power>n or power<0:raise ValueError('Degree elevation only')
    return np.array([float(sum((Fraction(math.comb(power,a)*2**a*(-1)**(power-a)*math.comb(i,a),math.comb(n,a))
                         for a in range(min(i,power)+1)),Fraction())) for i in range(n+1)])


def derivative_bernstein_matrix():
    """36x28 map: monomial coefficients of P6 to Bernstein coefficients of dP/dz."""
    result=np.zeros((36,28))
    for k,(i,j) in enumerate(powers(DEGREE)):
        if j:result[:,k]=(j*np.outer(power_to_bernstein(i),power_to_bernstein(j-1))).ravel()
    return result


def margin_matrix(rate_bounds):
    return derivative_bernstein_matrix()*(2/np.log(rate_bounds[1]/rate_bounds[0]))


def bernstein_evaluate(unit_points,coefficients,n=5):
    unit_points=np.asarray(unit_points,float);c=np.asarray(coefficients).reshape(n+1,n+1)
    k=np.arange(n+1)
    bx=comb(n,k)*unit_points[:,0,None]**k*(1-unit_points[:,0,None])**(n-k)
    bz=comb(n,k)*unit_points[:,1,None]**k*(1-unit_points[:,1,None])**(n-k)
    return np.einsum('ni,ij,nj->n',bx,c,bz)


def constrained_fit(a,y,rate_bounds,margin=MARGIN):
    """Solve the exact log-LS QP in whitened coordinates; no held-out inputs."""
    a=np.asarray(a,float);y=np.asarray(y,float)
    q,r=np.linalg.qr(a,mode='reduced')
    if np.linalg.matrix_rank(r)<28:raise ValueError('Unidentifiable degree-six fit')
    transform=solve_triangular(r,np.eye(28),lower=False)*np.sqrt(len(y))
    target=q.T@y/np.sqrt(len(y))
    g=margin_matrix(rate_bounds)
    raw_h=g@transform
    row_scale=np.linalg.norm(raw_h,axis=1)
    h=raw_h/row_scale[:,None]
    rhs=np.full(36,margin+GUARD-1)/row_scale
    constant=np.zeros(28);constant[0]=float(y.mean())
    initial=r@constant/np.sqrt(len(y))
    if min(h@initial-rhs)<=0:raise AssertionError('Constant start must be strictly feasible')
    objective=lambda v:.5*np.sum((v-target)**2)
    gradient=lambda v:v-target
    start=time.time()
    result=minimize(objective,initial,jac=gradient,method='SLSQP',
        constraints={'type':'ineq','fun':lambda v:h@v-rhs,'jac':lambda v:h},
        options={'ftol':1e-13,'maxiter':1000,'disp':False})
    v=result.x;coef=transform@v
    slack=h@v-rhs
    active=slack<1e-6
    multipliers=np.zeros(36)
    if active.any():multipliers[active]=nnls(h[active].T,gradient(v),maxiter=10000)[0]
    stationarity=float(np.linalg.norm(gradient(v)-h.T@multipliers))
    complementarity=float(np.max(np.abs(multipliers*slack)))
    bernstein=1+g@coef
    # The whitened primal/dual objectives omit the same constant orthogonal residual.
    # For any nonnegative lambda, the quadratic dual is exact and provides a bound.
    dual=float(-.5*np.sum((h.T@multipliers)**2)-multipliers@(h@target-rhs))
    primal=float(objective(v))
    gap=primal-dual
    diagnostic=dict(success=bool(result.success),status=int(result.status),message=str(result.message),
        iterations=int(result.nit),function_evaluations=int(result.nfev),seconds=time.time()-start,
        training_log_mse=float(np.mean((a@coef-y)**2)),
        minimum_bernstein_margin=float(bernstein.min()),maximum_bernstein_margin=float(bernstein.max()),
        primal_normalized_constraint_violation=float(max(0.,-slack.min())),
        kkt_stationarity_l2=stationarity,kkt_complementarity_max=complementarity,
        dual_multiplier_min=float(multipliers.min()),whitened_primal_dual_gap=float(gap),
        active_constraints=int(active.sum()),design_rank=int(np.linalg.matrix_rank(r)),design_condition_number=float(np.linalg.cond(r)),
        normalized_primal_slack=slack.tolist(),constraint_row_norms=row_scale.tolist(),
        constraint_row_normalization='Euclidean norms in QR-whitened coefficient coordinates',
        passed=bool(result.success and bernstein.min()>=margin and stationarity<1e-7 and complementarity<1e-8 and gap>=-1e-9 and gap<1e-8))
    return coef,bernstein,multipliers,diagnostic


def protocol(source):
    return dict(status='post-result intervention, predeclared before these fits',
        trigger='Observed sampled incremental-stress violations in degree-six unconstrained direct (16/50) and frozen (39/100) models',
        hypothesis='A global sufficient monotonic-stress constraint removes this structural defect; held-out error and engineering decisions may improve or worsen',
        source_directory=str(source.resolve()),degree=6,polynomial_parameters=28,margin=MARGIN,numerical_guard=GUARD,
        inequality='eta + gamma*deta/dgamma >= 0.05 eta; M=1+(2/log(rate_max/rate_min))*dP/dz',
        domain='Full normalized training rectangle x,z in [-1,1]; no certificate outside it',
        certificate='36 tensor Bernstein degree(5,5) coefficients of M must exceed 0.05+1e-8',
        conservativeness='Sufficient Bernstein coefficient inequalities, not necessary conditions for positivity',
        direct_models=50,frozen_models=100,neural_retraining=0,
        data='Exactly the saved primary_high_shear training/heldout row IDs and training-only normalization bounds',
        direct_objective='Mean squared log-viscosity residual on all original primary training observations',
        frozen_objective='Mean squared log-viscosity discrepancy from each final NN teacher on original uniform 61x61 feature grid',
        optimizer=dict(method='SLSQP with analytic objective and constraint gradients; QR-whitened quadratic objective',
            start='Strictly feasible constant polynomial equal to target mean',ftol=1e-13,maxiter=1000,
            checks='Unscaled Bernstein margin, primal feasibility, nonnegative dual multipliers, KKT and primal-dual gap'),
        engineering='Unchanged held-out gamma=20/s eta<=0.4 Pa s window metric; 1001 temperature samples and same measurement interpolation',
        heldout_selection='None: all 50 material/replicate folds and both frozen teacher seeds included',
        baseline='Saved unconstrained degree6 complexity records, matched by material/fold/method/seed',
        uncertainty='Average NN seeds within each fold first; descriptive across-fold SD, shared training folds are not independent trials',
        source_sha256={name:digest(ROOT/'experiments'/name) for name in ['rheology_monotone.py','rheology_core.py','rheology_evaluate.py']},
        input_sha256={name:digest(source/name) for name in ['observations.csv','training_specs.json','models/checkpoint_6000.npz',
            'complexity/model_definitions.json','complexity/by_model.csv']})


def predict_metrics(spec,coef,data,teacher):
    predictor=Predictor(spec,coefficients=coef,degree=6)
    test=data.iloc[spec['heldout_row_ids']];high=test[test.rate>=10]
    pred=predictor(high.temperature.to_numpy(),high.rate.to_numpy())
    if not np.isfinite(pred).all() or not np.isfinite(np.exp(pred)).all() or np.any(np.exp(pred)<=0):
        raise FloatingPointError('Nonfinite/nonpositive heldout predicted viscosity')
    row={'high_shear_'+k:v for k,v in metrics(high.viscosity.to_numpy(),pred).items()}
    processing,cross,_=evaluate_processing(predictor,processing_reference(test));row.update(processing)
    xx=predictor.x(high.temperature.to_numpy(),high.rate.to_numpy())
    row['heldout_outside_certified_rectangle']=int(np.any(abs(xx)>1+1e-12,axis=1).sum())
    row['minimum_heldout_stress_margin']=float(predictor.margin(xx).min())
    grid=np.stack(np.meshgrid(np.linspace(-1,1,61),np.linspace(-1,1,201),indexing='ij'),axis=-1).reshape(-1,2)
    margin=predictor.margin(grid);row['minimum_sampled_stress_margin']=float(margin.min())
    row['sampled_nonmonotone_points']=int((margin<=0).sum())
    row['heldout_log_compression_rmse']=None if teacher is None else float(np.sqrt(np.mean((pred-teacher(high.temperature.to_numpy(),high.rate.to_numpy()))**2)))
    return row,cross,pred


def run(source,output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    plan=protocol(source)
    if (output/'PROTOCOL.json').exists():
        if json.loads((output/'PROTOCOL.json').read_text())!=plan:raise ValueError('Protocol/source mismatch')
    else:
        save(output/'PROTOCOL.json',plan)
        p=output/'provenance/source';p.mkdir(parents=True)
        for name in plan['source_sha256']:shutil.copy2(ROOT/'experiments'/name,p/name)
        save(output/'ENVIRONMENT.json',dict(command=sys.argv,python=sys.version,
            packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','pytest']}))
    data=pd.read_csv(source/'observations.csv');specs=json.loads((source/'training_specs.json').read_text())
    checkpoint=dict(np.load(source/'models/checkpoint_6000.npz'))
    baseline_models=json.loads((source/'complexity/model_definitions.json').read_text())
    baseline_metrics=pd.read_csv(source/'complexity/by_model.csv').set_index('model_id')
    grid=np.stack(np.meshgrid(np.linspace(-1,1,61),np.linspace(-1,1,61),indexing='ij'),axis=-1).reshape(-1,2)
    grid_a=design(grid,6)
    seen=set();rows=[];models=[];failures=[];crossings=[]
    for spec in specs:
        if spec['arm']!='primary_high_shear':continue
        i=spec['index'];weights={name:checkpoint[name][i] for name in ('w0','b0','w2','b2','w4','b4')}
        teacher=Predictor(spec,weights=weights);train=data.iloc[spec['training_row_ids']]
        teacher_grid=teacher.predict_x(grid)
        candidates=[('frozen_d6',grid_a,teacher_grid,teacher)]
        key=(spec['material'],spec['fold'])
        if key not in seen:
            seen.add(key)
            x=teacher.x(train.temperature.to_numpy(),train.rate.to_numpy())
            candidates.append(('direct_d6',design(x,6),np.log(train.viscosity.to_numpy()),None))
        for method,a,y,teacher_for_metrics in candidates:
            direct=method.startswith('direct');seed=-1 if direct else spec['seed']
            key=f'{spec["material"]}_fold{spec["fold"]}_seed{seed}_{method}'
            path=output/'fits'/f'{key}.json'
            base=next(m for m in baseline_models if m['method']==method and m['material']==spec['material'] and m['fold']==spec['fold'] and m['seed']==seed)
            try:
                if path.exists():
                    fit=json.loads(path.read_text());coef=np.array(fit['coefficients']);diagnostic=fit['diagnostic'];bernstein=np.array(fit['bernstein_margin'])
                else:
                    coef,bernstein,multipliers,diagnostic=constrained_fit(a,y,spec['rate_bounds'])
                    fit=dict(key=key,method=method,material=spec['material'],fold=spec['fold'],seed=seed,spec_index=i,
                        coefficients=coef.tolist(),bernstein_margin=bernstein.tolist(),dual_multipliers=multipliers.tolist(),diagnostic=diagnostic,
                        baseline_model_id=base['model_id'],training_row_ids=spec['training_row_ids'],heldout_row_ids=spec['heldout_row_ids'],
                        temperature_bounds=spec['temperature_bounds'],rate_bounds=spec['rate_bounds'])
                    save(path,fit)
                if not diagnostic['passed']:
                    failures.append(dict(key=key,diagnostic=diagnostic));continue
                for intervention,c in [('unconstrained',np.array(base['coefficients'])),('bernstein_constrained',coef)]:
                    metric,cross,pred=predict_metrics(spec,c,data,teacher_for_metrics)
                    model_id=len(models)
                    models.append(dict(model_id=model_id,key=key,method=method,intervention=intervention,
                        material=spec['material'],fold=spec['fold'],seed=seed,spec_index=i,degree=6,coefficients=c.tolist(),
                        evaluation_rectangle=dict(temperature_bounds=spec['temperature_bounds'],rate_bounds=spec['rate_bounds']),
                        certificate_applies=(intervention=='bernstein_constrained')))
                    row=dict(model_id=model_id,key=key,material=spec['material'],fold=spec['fold'],seed=seed,method=method,
                        intervention=intervention,**metric,objective_log_mse=float(np.mean((a@c-y)**2)),
                        minimum_bernstein_margin=float((1+margin_matrix(spec['rate_bounds'])@c).min()))
                    if intervention=='bernstein_constrained':
                        row.update(fit_seconds=diagnostic['seconds'],kkt_stationarity_l2=diagnostic['kkt_stationarity_l2'],
                            kkt_complementarity_max=diagnostic['kkt_complementarity_max'],whitened_primal_dual_gap=diagnostic['whitened_primal_dual_gap'])
                    else:
                        original=baseline_metrics.loc[base['model_id']]
                        row['baseline_relative_l2_recheck_difference']=abs(float(original['high_shear_relative_l2'])-metric['high_shear_relative_l2'])
                        row['baseline_window_recheck_difference']=abs(float(original['processing_window_symmetric_difference_celsius'])-metric['processing_window_symmetric_difference_celsius'])
                    rows.append(row);crossings.append(dict(model_id=model_id,**cross))
                print(key,diagnostic['minimum_bernstein_margin'],diagnostic['kkt_stationarity_l2'],flush=True)
            except Exception as error:
                failure=dict(key=key,error=repr(error),traceback=traceback.format_exc());save(output/'failures'/f'{key}.json',failure);failures.append(failure)
    pd.DataFrame(rows).to_csv(output/'by_model.csv',index=False)
    save(output/'model_definitions.json',models);save(output/'crossing_sets.json',crossings)
    save(output/'FAILURES.json',failures)
    if failures:raise AssertionError(f'{len(failures)} failures retained; no COMPLETE marker')
    if len(models)!=300:raise AssertionError(f'Expected 300 paired model records; found {len(models)}')
    report(output,plan,rows)


def report(output,plan,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.DataFrame(rows)
    cols=['high_shear_relative_l2','high_shear_log_rmse','processing_window_symmetric_difference_celsius',
        'processing_window_falseeligible_celsius','objective_log_mse','minimum_sampled_stress_margin','minimum_bernstein_margin']
    paired=table.groupby(['method','intervention','material','fold'],as_index=False)[cols].mean()
    paired.to_csv(output/'by_replicate_seedmean.csv',index=False)
    summary=paired.groupby(['method','intervention'])[cols].agg(['mean','std'])
    summary.columns=['_'.join(v) for v in summary.columns];summary.to_csv(output/'summary.csv')
    constrained=table[table.intervention=='bernstein_constrained'];base=table[table.intervention=='unconstrained']
    completion=dict(status='passed',quadratic_programs=150,direct_fits=50,frozen_fits=100,new_neural_training=0,
        heldout_replicate_files=50,degree=6,coefficients_per_model=28,
        minimum_global_bernstein_margin=float(constrained.minimum_bernstein_margin.min()),
        sampled_stress_violations_constrained=int((constrained.sampled_nonmonotone_points>0).sum()),
        sampled_stress_violations_baseline={method:int((d.sampled_nonmonotone_points>0).sum()) for method,d in base.groupby('method')},
        max_kkt_stationarity_l2=float(constrained.kkt_stationarity_l2.max()),
        max_kkt_complementarity=float(constrained.kkt_complementarity_max.max()),
        max_primal_dual_gap=float(constrained.whitened_primal_dual_gap.max()),
        max_baseline_relative_l2_recheck_difference=float(base.baseline_relative_l2_recheck_difference.max()),
        max_baseline_window_recheck_difference=float(base.baseline_window_recheck_difference.max()),
        constraint_positive_margin=MARGIN,certificate_scope='Only each training rectangle, sufficient Bernstein inequalities verified in floating point',
        min_heldout_outside_rectangle=int(constrained.heldout_outside_certified_rectangle.min()),
        max_heldout_outside_rectangle=int(constrained.heldout_outside_certified_rectangle.max()),
        total_fit_seconds=float(constrained.fit_seconds.sum()),mean_fit_seconds=float(constrained.fit_seconds.mean()),
        protocol_sha256=digest(output/'PROTOCOL.json'))
    if completion['max_baseline_relative_l2_recheck_difference']>1e-12 or completion['max_baseline_window_recheck_difference']>1e-10:
        raise AssertionError('Baseline metric recheck failed')
    save(output/'COMPLETE.json',completion)
    with plt.rc_context({'axes.titlesize':10,'axes.labelsize':9,'font.size':9}):
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for method,color in [('direct_d6','C0'),('frozen_d6','C1')]:
            s=paired[paired.method==method]
            u=s[s.intervention=='unconstrained'].set_index(['material','fold'])
            c=s[s.intervention=='bernstein_constrained'].set_index(['material','fold']).loc[u.index]
            for ax,metric,title in zip(axes,['high_shear_relative_l2','processing_window_symmetric_difference_celsius','objective_log_mse'],
                                      ['Held-out viscosity relative L2','Processing-window error (C)','Training/projection log MSE']):
                ax.scatter(u[metric],c[metric],s=15,alpha=.6,color=color,label=method)
                ax.set_xlabel('Unconstrained');ax.set_ylabel('Constrained');ax.set_title(title)
        for ax in axes:
            lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],'k:',lw=1);ax.legend(fontsize=7)
        fig.savefig(output/'intervention.png',dpi=180);plt.close(fig)
    lines=['# Global monotonic-stress intervention on measured rheometry','',
        'This follow-up was declared after the unconstrained degree-six models exhibited incremental-stress violations. It evaluates one fixed intervention on every primary fold: 50 direct fits and 100 frozen projections of the existing final teachers. There is no neural retraining or held-out tuning.',
        '', 'The viscosity is eta=exp(P6(x,z)), hence positive. The 36 tensor Bernstein coefficients of M=1+(2/log(gamma_max/gamma_min))*partial_z P6 are constrained to exceed 0.05+1e-8. Bernstein basis functions are nonnegative and sum to one, giving a sufficient global M>=0.05 certificate throughout each full training rectangle. This restriction can be conservative; a failed unconstrained Bernstein bound does not prove that an unconstrained function is nonmonotone.',
        '',f'All 150 convex quadratic programs passed primal/KKT checks. Minimum certified coefficient margin: {completion["minimum_global_bernstein_margin"]:.10g}; largest stationarity residual: {completion["max_kkt_stationarity_l2"]:.3g}; largest primal-dual gap: {completion["max_primal_dual_gap"]:.3g}.',
        '',f'Original sampled nonmonotone models: {completion["sampled_stress_violations_baseline"]}; constrained: 0/150, with the stronger within-rectangle Bernstein certificate.',
        '', '| Fit | Intervention | Held-out relative L2 | Held-out log RMSE | Window disagreement (C) | False-eligible window (C) |',
        '|---|---|---:|---:|---:|---:|']
    for (method,intervention),r in summary.iterrows():
        lines.append(f'| {method} | {intervention} | {r["high_shear_relative_l2_mean"]:.6g} | {r["high_shear_log_rmse_mean"]:.6g} | {r["processing_window_symmetric_difference_celsius_mean"]:.6g} | {r["processing_window_falseeligible_celsius_mean"]:.6g} |')
    lines += ['', 'NN seeds are averaged within each material/held-out-replicate fold before aggregating these 50 repeated empirical endpoints. Shared training folds make the displayed SD descriptive, not independent-trial uncertainty.',
        '', 'This post-result intervention reuses the already examined held-out folds, so it is an exploratory repair experiment, not a new independent confirmatory test.',
        '', 'The engineering endpoint is unchanged: eta<=0.4 Pa s at gamma=20/s, compared against held-out measured temperature curves using the original interpolation and 1001-point window integration. A global incremental-stress certificate does not guarantee viscosity-prediction accuracy, temperature-window accuracy, or accuracy outside the training rectangle.',
        '',f'Individual held-out high-shear sets have {completion["min_heldout_outside_rectangle"]}–{completion["max_heldout_outside_rectangle"]} observations outside their training rectangles; these predictions remain in evaluation but do not receive the global certificate. Mean SLSQP solve-only runtime (excluding QR/setup): {completion["mean_fit_seconds"]:.4g} s; inference still uses 28 polynomial coefficients.',
        '', 'Raw input, teacher checkpoints, original unconstrained results and manuscript files were not changed. This is an intervention experiment for constitutive reliability, not a claim of a new polynomial optimization method.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
