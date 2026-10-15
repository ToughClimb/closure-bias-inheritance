"""Independent artifact checks and reports for the frozen linear PDE campaign."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Support both direct script execution and python -m from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.linear_pde_extension import (
    SYSTEMS, fields, feature_matrix, put_json, put_csv, sha,
)
from experiments.basic_pde_controls import stencil_fields


def flexible_csv(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    put_csv(path,[{key:row.get(key,'') for key in keys} for row in rows])


def read_cases(output):
    return [json.loads(p.read_text()) for p in sorted((output/'cases').glob('*.json')) if not p.name.endswith('.partial.json')]


def clarify(output):
    record=dict(
        reason='Independent audit identified imprecise record counting and evaluation wording; numerical protocol, budgets, saved data, models and selections are unchanged',
        observed_records=80,unique_clean_observation_arrays=8,clean_teacher_initializations=40,
        noisy_observation_arrays=40,initial_teachers=80,paired_continued_states=40,
        dependence='The same seed generates corresponding normalized noise across systems/grids where shapes agree; do not pool these as independent noise experiments',
        original_protocol_correction='counts.independent_observation_tables=80 denotes task records, not 80 independent datasets; clean records repeat identical observations across five network seeds',
        projection_norm='Uniform tensor box in each teacher own training-normalized coordinates. Physical box differs across condition; compare clean/noisy/grid effects on the separately fixed attainable exact-solution jets and rollouts',
        positive_state_check='Only the six saved rollout times are checked. Passing is not a continuous-time positivity certificate',
        derivative_check='Teacher derivatives on the finite calibration box grid are sampled diagnostics, not global parabolicity/transport certificates',
        rollout_failures='RHS budget/state limits and spatial-grid disagreement are numerical diagnostics; an unresolved rollout is not automatically proof that the continuum learned PDE has no solution',
        symbolic_exact_diagnostic='Additional reporting check: maximum numerical/exact discrepancy >1e-6 (10 times declared relative tolerance at unit state scale) is flagged separately from solver completion; constant D<0 independently certifies backward parabolicity',
        phase='Report Fourier amplitude and wrapped phase on declared saved fields, masking phase when either coefficient is smaller than 1e-8 of the initial true modal magnitude',
        proof_scope='Linear projection identity is standard and applies in its specified box norm. Unknown physical jets are evaluated independently. The neural architecture does not enforce flux conservation or positive diffusivity',
    )
    put_json(output/'PROTOCOL_CLARIFICATION.json',record)


def common_jet_errors(system,coefficients,initial_spec):
    f=fields(np.arange(128)/128,np.linspace(0,.8,33),system,**initial_spec)
    X=feature_matrix(f,SYSTEMS[system]['unknowns']);truth=f['ut'].ravel();prediction=X@np.asarray(coefficients)
    return float(np.linalg.norm(prediction-truth)/np.linalg.norm(truth))


def centered_feature_noise_covariance(unknowns,sigma,h):
    """Covariance of centered (uxx, -ux, u) noise at the same spacetime point."""
    covariance=np.zeros((len(unknowns),len(unknowns)))
    variance={'D':6*sigma*sigma/h**4,'c':sigma*sigma/(2*h*h),'a':sigma*sigma}
    for i,name in enumerate(unknowns):covariance[i,i]=variance[name]
    if 'D' in unknowns and 'a' in unknowns:
        i,j=unknowns.index('D'),unknowns.index('a');covariance[i,j]=covariance[j,i]=-2*sigma*sigma/h**2
    return covariance


def phase_rows(output,run_id,method,system,initial_specs,checks):
    path=output/'rollouts'/f'{run_id}_{method}.npz'
    if not path.exists():return []
    rows=[]
    with np.load(path) as data:
        times=data['times']
        for initial_name,spec in initial_specs.items():
            key=f'{initial_name}_n64'
            if key not in data:continue
            candidate=data[key];truth=fields(np.arange(64)/64,times,system,**spec)['u']
            predhat=np.fft.rfft(candidate,axis=1)/64;truehat=np.fft.rfft(truth,axis=1)/64
            for mode in spec['modes']:
                if mode==0:continue
                threshold=1e-8*abs(truehat[0,mode])
                for i,t in enumerate(times):
                    reference=truehat[i,mode];prediction=predhat[i,mode]
                    defined=abs(reference)>threshold and abs(prediction)>threshold
                    rows.append(dict(run_id=run_id,method=method,system=system,initial_condition=initial_name,mode=mode,time=float(t),
                        truth_mode_amplitude=float(2*abs(reference)),prediction_mode_amplitude=float(2*abs(prediction)),
                        absolute_amplitude_error=float(2*abs(abs(prediction)-abs(reference))),
                        amplitude_ratio=float(abs(prediction)/abs(reference)) if abs(reference)>threshold else '',
                        phase_defined=defined,wrapped_phase_error=float(np.angle(prediction/reference)) if defined else '',
                        spatial_grid_agreement=checks[initial_name]['grid_agreement_pass']))
    return rows


def verify(output,cases):
    p=json.loads((output/'PROTOCOL.json').read_text())
    if sha(output/'source.py')!=p['source_sha256']:raise AssertionError('Frozen numerical source hash differs')
    complete=json.loads((output/'RUN_COMPLETE.json').read_text())
    if len(cases)!=80 or complete['teacher_states']!=120:raise AssertionError('Incomplete formal campaign')
    initial_model_hashes=defaultdict(list);projection=[];objective=[];ols=[];npchecks=[];hashes=[]
    actual_unique_clean=set();actual_unique_noisy=set();states=0
    for case in cases:
        path=output/'inputs'/f"{case['case']}.npz"
        if sha(path)!=case['input_sha256']:raise AssertionError('Input hash differs')
        with np.load(path) as data:
            X=data['X'];y=data['y'];observed=data['observed']
            # Independent unscaled least-squares cross-check, no use of training optimizer.
            expected=np.linalg.lstsq(X,y,rcond=None)[0]
            ols.append(float(np.max(np.abs(expected-np.array(case['direct']['coefficients'])))))
            identity=(case['system'],case['grid'],observed.tobytes())
            (actual_unique_clean if case['noise']==0 else actual_unique_noisy).add(identity)
        for state in case['states']:
            states+=1;checkpoint=output/'models'/f"{state['run_id']}.pt"
            if sha(checkpoint)!=state['checkpoint_sha256']:raise AssertionError('Checkpoint hash differs')
            npchecks.append(state['numpy_torch_max_difference'])
            with np.load(output/'calibration'/f"{state['run_id']}.npz") as data:
                teacher=data['teacher'];truth=data['truth'];symbolic=data['symbolic'];design=data['X']
                eN=np.mean((teacher-truth)**2);eS=np.mean((symbolic-truth)**2);eC=np.mean((symbolic-teacher)**2)
                projection.append(float(abs(eN-eS-eC)))
                target_scale=max(1.,float(np.mean(truth*truth)))
                if abs(eN-eS-eC)>1e-11*target_scale:raise AssertionError('Projection identity fails')
                direct_projection=design@np.linalg.lstsq(design,teacher,rcond=None)[0]
                if np.max(np.abs(direct_projection-symbolic))>1e-10:raise AssertionError('Frozen projection differs')
        hashes.append(dict(path=str(path.relative_to(output)),sha256=sha(path)))
    if (len(actual_unique_clean),len(actual_unique_noisy))!=(8,40):raise AssertionError('Observed-array counts differ')
    result=dict(status='passed',completed_cases=len(cases),teacher_states=states,unique_clean_arrays=8,unique_noisy_arrays=40,
        maximum_independent_ols_coefficient_difference=max(ols),maximum_projection_identity_residual=max(projection),
        maximum_numpy_torch_prediction_difference=max(npchecks),numerical_source_sha256=p['source_sha256'],
        interpretation='Artifact/data/projection checks passed; rollout completion and grid resolution assessed separately, not assumed')
    if max(ols)>1e-10 or max(npchecks)>1e-10:raise AssertionError('Independent numerical checks differ')
    put_json(output/'INTEGRITY_CHECKS.json',result)
    return result


def report(output):
    clarify(output)
    cases=read_cases(output);verification=verify(output,cases)
    protocol=json.loads((output/'PROTOCOL.json').read_text())['protocol'];initial_specs=protocol['evaluation']['initial_conditions']
    states=[];rollouts=[];phases=[];coefficients=[];on_jets=[];noise_population=[]
    for case in cases:
        common={key:case[key] for key in ['case','system','grid','noise','seed']}
        direct=np.asarray(case['direct']['coefficients'])
        with np.load(output/'inputs'/f"{case['case']}.npz") as data:
            clean=data['clean'];dt=float(data['t'][1]-data['t'][0]);h=1/case['grid']
        clean_fields=stencil_fields(clean,h,dt)
        names=SYSTEMS[case['system']]['unknowns'];Xclean=feature_matrix(clean_fields,names);yclean=clean_fields['ut'].ravel()
        sigma=case['noise_sigma'];covariance=centered_feature_noise_covariance(names,sigma,h)
        G=Xclean.T@Xclean/len(Xclean);b=Xclean.T@yclean/len(Xclean)
        population=np.linalg.solve(G+covariance,b)
        for name,value,observed in zip(names,population,direct,strict=True):
            noise_population.append(dict(**common,coefficient=name,population_expected_objective_minimizer=float(value),observed_finite_sample_ols=float(observed),
                truth=SYSTEMS[case['system']][name],note='Expected-objective optimizer, not expectation of finite-sample coefficient ratio'))
        for name,value in zip(SYSTEMS[case['system']]['unknowns'],direct,strict=True):
            coefficients.append(dict(**common,stage='shared',method='direct',coefficient=name,value=float(value),truth=SYSTEMS[case['system']][name]))
        for initial_name,spec in initial_specs.items():
            on_jets.append(dict(**common,stage='shared',method='direct',initial_condition=initial_name,
                relative_operator_error=common_jet_errors(case['system'],direct,spec)))
        for r in case['direct_rollouts']['rows']:
            rollouts.append(dict(**common,stage='shared',**{k:v for k,v in r.items() if k!='grid'},evaluation_grid=r['grid'],
                spatial_grid_agreement=case['direct_rollouts']['grid_checks'][r['initial_condition']]['grid_agreement_pass'],
                coarse_fine_max_difference=case['direct_rollouts']['grid_checks'][r['initial_condition']]['coarse_fine_max_difference']))
        phases.extend(phase_rows(output,case['case'],'direct',case['system'],initial_specs,case['direct_rollouts']['grid_checks']))
        for state in case['states']:
            cal=state['calibration'];op=state['optimization']
            states.append(dict(**common,stage=state['stage'],run_id=state['run_id'],residual_mse=state['residual_mse'],
                direct_residual_mse=case['direct']['residual_mse'],gradient_l2=op['gradient_l2'],gradient_l2_below_1e9=op['gradient_l2_below_1e9'],
                projection_teacher_relative_error=cal['teacher_relative_error'],projection_symbolic_relative_error=cal['symbolic_relative_error'],
                projection_fraction_retained=cal['teacher_fraction_retained'],projection_identity_residual=cal['projection_identity_residual'],
                sampled_min_diffusion_derivative=cal['sampled_operator_derivatives'].get('D',{}).get('minimum',''),
                training_elapsed_seconds=op['elapsed_seconds']))
            for name,value in zip(SYSTEMS[case['system']]['unknowns'],cal['coefficients'],strict=True):
                coefficients.append(dict(**common,stage=state['stage'],method='frozen',coefficient=name,value=value,truth=SYSTEMS[case['system']][name]))
            on_jets.extend(dict(**common,stage=state['stage'],**r) for r in state['on_solution_errors'])
            for method,record in state['rollouts'].items():
                for r in record['rows']:
                    rollouts.append(dict(**common,stage=state['stage'],**{k:v for k,v in r.items() if k!='grid'},evaluation_grid=r['grid'],
                        spatial_grid_agreement=record['grid_checks'][r['initial_condition']]['grid_agreement_pass'],
                        coarse_fine_max_difference=record['grid_checks'][r['initial_condition']]['coarse_fine_max_difference']))
                phases.extend(phase_rows(output,state['run_id'],method,case['system'],initial_specs,record['grid_checks']))
    for row in rollouts:
        if row['method']!='teacher':
            row['symbolic_numerical_exact_check_pass']=bool(row['status']=='completed' and row.get('symbolic_solver_exact_max_difference',1.)<=1e-6)
    for name,rows in [('states',states),('coefficients',coefficients),('on_solution_jets',on_jets),('rollouts',rollouts),('fourier_amplitude_phase',phases),('noise_population_predictions',noise_population)]:
        flexible_csv(output/'reports'/f'{name}.csv',rows)

    grouped=defaultdict(list)
    for row in on_jets:
        if row['initial_condition']=='heldout_mode3':
            grouped[tuple(row[key] for key in ['system','grid','noise','stage','method'])].append(row['relative_operator_error'])
    summaries=[]
    for (system,grid,noise,stage,method),values in sorted(grouped.items()):
        summaries.append(dict(system=system,grid=grid,noise=noise,stage=stage,method=method,n=len(values),
            mean=float(np.mean(values)),std=float(np.std(values,ddof=1)),minimum=float(min(values)),maximum=float(max(values))))
    put_csv(output/'reports/heldout_operator_summary.csv',summaries)
    counts=[]
    for system in SYSTEMS:
        for method in ['teacher','frozen','direct']:
            for noise in [0.,.01]:
                relevant=[r for r in rollouts if r['system']==system and r['method']==method and r['noise']==noise and r['grid']==64 and r['evaluation_grid']==64 and r['stage'] in ('base','shared')]
                # Each row is one initial condition at evaluation grid64 for a teacher trained at grid64.
                counts.append(dict(system=system,method=method,noise=noise,rollouts=len(relevant),
                    completed=sum(r['status']=='completed' for r in relevant),
                    spatially_resolved=sum(r['spatial_grid_agreement'] for r in relevant),
                    saved_positive_initial_violations_on_resolved=sum(bool(r.get('positive_initial_violation',False)) and r['spatial_grid_agreement'] for r in relevant),
                    maximum_saved_mean_error=max((r.get('max_mean_balance_error',0.) for r in relevant if r['spatial_grid_agreement']),default=None)))
    flexible_csv(output/'reports/rollout_status_summary.csv',counts)

    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    positions=[(32,0.,'base'),(64,0.,'base'),(64,0.,'continued'),(32,.01,'base'),(64,.01,'base'),(64,.01,'continued')]
    colors={'teacher':'#3669b3','frozen':'#e78b2f','direct':'#4a9a66'}
    for ax,(system,p) in zip(axes.flat,SYSTEMS.items(),strict=True):
        for method in colors:
            means=[]
            for i,(grid,noise,stage) in enumerate(positions):
                vals=grouped[(system,grid,noise,'shared' if method=='direct' else stage,method)]
                means.append(np.mean(vals))
                ax.scatter([i]*len(vals),vals,color=colors[method],alpha=.35,s=14)
            ax.plot(range(6),means,'o-',color=colors[method],label=method)
        ax.set_yscale('log');ax.set_title(system.replace('_',' '));ax.set_xticks(range(6),['32 C','64 C','64 C+','32 N','64 N','64 N+'])
        ax.set_ylabel('held-out physical-jet operator error');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(output/'reports/heldout_operator_errors.png',dpi=180);plt.close(fig)

    table=[]
    for system in SYSTEMS:
        for noise in [0.,.01]:
            vals=[]
            for stage,method in [('base','teacher'),('base','frozen'),('continued','teacher'),('continued','frozen'),('shared','direct')]:
                vals.append(np.mean(grouped[(system,64,noise,stage,method)]))
            table.append('| '+system+' | '+('clean' if noise==0 else '1%')+' | '+' | '.join(f'{v:.5g}' for v in vals)+' |')
    total_failed=sum(r['status']!='completed' for r in rollouts)
    teacher_rows=[r for r in rollouts if r['method']=='teacher' and r['evaluation_grid']==64]
    teacher_resolved=sum(r['spatial_grid_agreement'] for r in teacher_rows)
    numeric_errors=[r['symbolic_solver_exact_max_difference'] for r in rollouts if 'symbolic_solver_exact_max_difference' in r]
    numerical_exact_flags=[r for r in rollouts if r['method']!='teacher' and r['status']=='completed' and not r['symbolic_numerical_exact_check_pass']]
    negative_diffusions=[r for r in coefficients if r['coefficient']=='D' and r['value']<0]
    summary=dict(states=len(states),initial_states=sum(r['stage']=='base' for r in states),continued_states=sum(r['stage']=='continued' for r in states),
        all_rollout_attempts=len(rollouts),failed_rollout_attempts=total_failed,
        teacher_grid64_saved_trajectories=len(teacher_rows),teacher_grid64_spatially_agreed=teacher_resolved,
        maximum_symbolic_semidiscrete_exact_difference=max(numeric_errors),strict_gradient_criterion_states=sum(r['gradient_l2_below_1e9'] for r in states),
        completed_symbolic_rollouts_failing_exact_check=len(numerical_exact_flags),negative_frozen_diffusion_states=len(negative_diffusions))
    put_json(output/'SUMMARY.json',summary)
    text='''# Four linear PDE neural-operator reliability controls

Completed a specified-library operator-learning campaign for heat, advection, advection-diffusion, and linear reaction-diffusion. The neural teacher approximates the local map from the known relevant differential features to ut. This is a genuinely trained neural operator followed by frozen linear symbolic compression. It is distinct from learning state-dependent nonlinear constitutive closures in the main paper. No manuscript was edited.

80 initial teachers and 40 paired continuations finished. The clean portion has 8 unique observation arrays and 40 network initializations; the noisy portion has 40 observation arrays. Corresponding normalized random draws are shared across applicable conditions, so these are paired conditions, not 80 independent datasets. Five-seed averages are accompanied by individual rows.

The exact continuum truth has modes 0,1,2. Training uses centered finite differences of observed states, with clean or Gaussian noise of 1% of the clean-state RMS. nx32 and nx64 have the same temporal spacing. Thus spatial refinement at fixed noise can amplify differentiation errors. All models receive the same residual pairs within a case. Direct OLS solves the specified symbolic family globally; fixed-family residual refitting is the identical estimator and counted once.

The frozen teacher is projected onto that linear family on a uniform [-2,2] normalized-feature box. This box is specific to the teacher's training-derived scales and need not coincide with observed solution jets. Its projection identity is checked within each record. Cross-condition physical comparisons instead use the identical exact-solution jets on x=128 points, t=0..0.8, including a held-out mode 3.

## Held-out mode 3: relative local-operator error on fixed attainable jets

The following numbers are five-seed means at training nx64. “Continued” is the same checkpoint trained for an additional fixed L-BFGS budget with a fresh optimizer, not a new dataset.

| System | Noise | Teacher | Frozen | Continued teacher | Continued frozen | Direct same-observation |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

## Rollout and physical checks

All three methods are integrated by the same Fourier method of lines and RK45 on nx32 and nx64, with saved times 0, .05, .1, .2, .4, .8. The three initial conditions are the training modal mixture, a held-out mode 3, and a positive low-mode mixture. Exact continuum Fourier solutions provide references; frozen/direct symbolic solutions also have their own exact evaluation. Absolute RMS errors, amplitude, wrapped phase, mean-balance errors, and sampled positivity accompany relative errors.

'''+f'There were {len(rollouts)} numerical rollout attempts, including both spatial grids and all methods/stages; {total_failed} failed the declared solver/RHS/state criteria and remain in the tables. At evaluation nx64, {teacher_resolved}/{len(teacher_rows)} teacher trajectories agreed with nx32 within the declared max-norm tolerance 1e-4. Spatially unresolved trajectories are not labelled validated physical failures or successes. The largest completed symbolic numerical-versus-exact difference was {max(numeric_errors):.4g}; {len(numerical_exact_flags)} completed symbolic integrations failed the separate 1e-6 exact-solution diagnostic. Use the stored exact symbolic metrics for these cases.\n\n'+f'The original box projection produced {len(negative_diffusions)} stored states with negative constant diffusion, all in noisy reaction-diffusion. D<0 makes the continuum operator backward parabolic: arbitrarily high modes grow at rates proportional to k². A finite smooth Fourier-mixture formula can still exist, but it does not restore well-posedness. The large numerical-versus-exact discrepancy is therefore retained as high-frequency roundoff sensitivity, not reported as an accurate trajectory.\n\n'+'''
Saved-time nonnegativity is a diagnostic at six times, not a continuous-time certificate. The neural local-operator architecture enforces N(0)=0 but does not enforce flux conservation, positive diffusivity, or pure transport. Therefore apparent conservation or diffusion-sign failures diagnose this architecture and its fit; they do not establish that every neural constitutive formulation has these failures. Fourier pseudospectral teacher rollouts can become spatially unresolved, which is explicitly retained rather than hidden.

## Identifiability and interpretation

Exact single-mode reaction-diffusion data identify only a-Dk²: rank 1, with no uniquely reported D/a estimate. Two nonzero modes or one constant plus one nonzero mode give rank 2. The formal campaign uses modes 0,1,2. Noise-induced numerical rank is not evidence that a genuinely unexcited physical parameter became identifiable.

For the specified box projection and a true operator within the linear library, squared teacher error equals squared projected bias plus squared removed nonlinear component. This is a standard orthogonal-projection identity. It establishes precisely what frozen compression retains and removes in this norm; measured rollout reliability must still be checked in solution space. Five initializations and their continuations expose optimization sensitivity without claiming global convergence.

'''+f"Only {summary['strict_gradient_criterion_states']}/{len(states)} stored states met the reported strict L2 gradient threshold 1e-9; all are described as finite-budget fits. The final completion marker establishes completed work, not convergence.\n\n"+'''
## Artifacts

- `PROTOCOL.json` and `source.py`: frozen numerical design and implementation, saved before formal runs.
- `PROTOCOL_CLARIFICATION.json`: audit correction of record counting and diagnostic scope; no changed numerical protocol or selection.
- `inputs/`, `models/`, `cases/`, `calibration/`, `rollouts/`: every observation table, initialization, final checkpoint, optimizer status, projection array, and saved rollout.
- `reports/states.csv`, `coefficients.csv`, `on_solution_jets.csv`, `rollouts.csv`, `fourier_amplitude_phase.csv`: full unaggregated results, with failures and grid status.
- `reports/heldout_operator_summary.csv` and `heldout_operator_errors.png`: common-domain physical operator errors. C=clean, N=1% noise, +=paired continuation.
- `reports/noise_population_predictions.csv`: explicit expected-objective errors-in-variables prediction, retaining the covariance between noisy uxx and u. This is not an assertion about the expected finite-sample OLS ratio.
- `IDENTIFIABILITY.json`, `ANALYTICAL_CHECKS.json`, `INTEGRITY_CHECKS.json`, `SUMMARY.json`: explicit control and integrity outcomes.

Reproduce from the repository with the root environment:

```sh
python -m experiments.linear_pde_extension --output outputs/linear-pde-extension-2026-10-13
python -m experiments.linear_pde_report --output outputs/linear-pde-extension-2026-10-13
```

The runner resumes completed cases only if the numerical source and protocol match exactly. Use a different output directory for a changed protocol. Original runs and failed rollouts must remain available.
'''
    followup=output/'projection_measure_followup'
    if (followup/'COMPLETE.json').exists():
        followup_complete=json.loads((followup/'COMPLETE.json').read_text())
        followup_protocol=json.loads((followup/'PROTOCOL.json').read_text())
        if sha(followup/'source.py')!=followup_protocol['source_sha256']:raise AssertionError('Follow-up source hash differs')
        if followup_complete['teacher_states']!=120:raise AssertionError('Incomplete projection-measure follow-up')
        with (followup/'summary.csv').open() as stream:followup_summary=list(csv.DictReader(stream))
        with (followup/'on_solution_jets.csv').open() as stream:followup_jets=list(csv.DictReader(stream))
        fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
        for ax,system in zip(axes.flat,SYSTEMS,strict=True):
            for method,color in [('box','#e78b2f'),('observed jets','#9263b6'),('direct','#4a9a66')]:
                means=[]
                for i,(noise,stage) in enumerate([(0.,'base'),(0.,'continued'),(.01,'base'),(.01,'continued')]):
                    if method=='observed jets':
                        vals=[float(r['relative_operator_error']) for r in followup_jets if r['system']==system and int(r['grid'])==64 and float(r['noise'])==noise and r['stage']==stage and r['initial_condition']=='heldout_mode3']
                    else:vals=grouped[(system,64,noise,'shared' if method=='direct' else stage,'direct' if method=='direct' else 'frozen')]
                    means.append(np.mean(vals));ax.scatter([i]*len(vals),vals,color=color,alpha=.3,s=12)
                ax.plot(range(4),means,'o-',color=color,label=method)
            ax.set_yscale('log');ax.set_title(system.replace('_',' '));ax.set_xticks(range(4),['clean','clean +','noisy','noisy +'])
            ax.set_ylabel('held-out physical-jet error');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(output/'reports/projection_measure_comparison.png',dpi=180);plt.close(fig)
        measure_table=[]
        for system in SYSTEMS:
            for noise in [0.,.01]:
                for stage in ['base','continued']:
                    new=next(float(r['mean']) for r in followup_summary if r['system']==system and float(r['noise'])==noise and r['stage']==stage)
                    old=float(np.mean(grouped[(system,64,noise,stage,'frozen')]))
                    baseline=float(np.mean(grouped[(system,64,noise,'shared','direct')]))
                    measure_table.append(f'| {system} | {noise:g} | {stage} | {old:.5g} | {new:.5g} | {baseline:.5g} |')
        text+='''
## Declared follow-up: sensitivity to the projection measure

After observing discrepancies between the box norm and attainable reaction-diffusion jets, a separate protocol was saved before projecting **all 120 already trained checkpoints** on the original observed training jets. The symbolic family and teachers are identical; each observed feature row receives equal weight, and only the frozen neural outputs serve as compression targets. Held-out truth does not select support, weights, coefficients, or the better arm. Both arms remain reported. This intervention changes support and density together, so it isolates projection-measure sensitivity, not support alone.

| System | Noise | State | Box frozen | Observed-jet frozen | Direct |
|---|---:|---|---:|---:|---:|
'''+ '\n'.join(measure_table)+f'''

Numbers are relative local-operator errors on the same fixed attainable held-out mode-3 jets, averaged over five seeds at training nx64. For clean reaction-diffusion, the initial teachers' box-compressed error is about 0.099, versus about 0.0024 under the observed-jet projection. This demonstrates a change in the frozen expression caused by the compression measure while holding the trained teacher fixed. It does not establish that every empirical projection will generalize.

The observed-jet projection produced {followup_complete['negative_diffusion_coefficients']} negative diffusion coefficients among the 90 states whose PDE contains diffusion, compared with {len(negative_diffusions)} in the original box arm (the remaining 30 advection states have no diffusion coefficient). The measured positive result is retained alongside the original failures. Five-seed noisy heat and advection-diffusion errors remain large under both measures, consistent with a derivative-noise error component already present in the fitted residuals. The direct estimator's population prediction is available separately. The paired measure comparison is plotted in `reports/projection_measure_comparison.png`.

`projection_measure_followup/` contains its pre-execution protocol, source snapshot, all 120 fits, 360 exact symbolic rollouts, complete coefficient/jet/rollout tables, and independent projection checks. These are additional deterministic evaluations of existing teachers, not 120 additional trained models or independent datasets.

Reproduce this declared follow-up with:

```sh
python -m experiments.linear_pde_projection_measure --output outputs/linear-pde-extension-2026-10-13
python -m experiments.linear_pde_report --output outputs/linear-pde-extension-2026-10-13
```
'''
    audit=output/'forward_audit'
    if (audit/'COMPLETE.json').exists():
        audit_protocol=json.loads((audit/'PROTOCOL.json').read_text())
        if sha(audit/'source.py')!=audit_protocol['source_sha256']:raise AssertionError('Forward audit source changed')
        principal=json.loads((audit/'PRINCIPAL_CHECKS.json').read_text())
        budget=audit/'budget_extension'
        summary_path=budget/'representative_summary.csv' if (budget/'COMPLETE.json').exists() else audit/'representative_summary.csv'
        with summary_path.open() as stream:representatives=list(csv.DictReader(stream))
        with (audit/'principal_derivatives.csv').open() as stream:principal_rows=list(csv.DictReader(stream))
        unresolved={(r['run_id'],r['initial_condition']) for r in principal_rows if r['source']=='initial' and r['old_spatially_resolved']=='False'}
        negatives={(r['run_id'],r['initial_condition']) for r in principal_rows if int(r['negative_samples'])>0}
        timewise=[]
        for representative in representatives:
            curves={}
            for method,nx in [('RK45',128),('RK45',256),('Radau',128)]:
                key=f"{representative['run_id']}_{representative['initial_condition']}_{method}_n{nx}"
                for directory in [budget,audit/'solves']:
                    path=directory/f'{key}.npz'
                    if path.exists():
                        with np.load(path) as stored:curves[(method,nx)]=stored['u']
                        break
            for i,t in enumerate(audit_protocol['saved_times']):
                row=dict(run_id=representative['run_id'],initial_condition=representative['initial_condition'],time=t)
                if ('RK45',128) in curves and ('Radau',128) in curves:
                    row['rk128_radau128_max_difference']=float(np.max(np.abs(curves[('RK45',128)][i]-curves[('Radau',128)][i])))
                if ('RK45',128) in curves and ('RK45',256) in curves:
                    row['rk128_rk256_max_difference']=float(np.max(np.abs(curves[('RK45',128)][i]-curves[('RK45',256)][i,::2])))
                timewise.append(row)
        flexible_csv(audit/'timewise_method_grid_differences.csv',timewise)
        summary['forward_audit']=dict(initial_cases=principal['initial_cases'],initially_negative_cases=principal['initially_negative_cases'],
            initially_unresolved_diffusive_pairs=len(unresolved),unresolved_pairs_with_negative_initial_or_saved_derivative=len(unresolved&negatives),
            maximum_analytic_autograd_difference=principal['maximum_analytic_autograd_jacobian_difference'],representatives=len(representatives),
            interpretation='Original rollout counts remain unchanged; deterministic follow-up diagnostics are separate')
        put_json(output/'SUMMARY.json',summary)
        text+='''
## Independent forward audit: distinguishing unresolved numerical trajectories

Every one of the 270 diffusiveteacher/initial-condition pairs received a principal derivative check at its exact initial field, plus checks on every available saved nx32/nx64 trajectory. Analytic network input Jacobians agree with automatic differentiation within 8.04e-14. There are no negative initial principal derivatives among 135 clean pairs, versus 119/135 noisy pairs. All 119 belong to the 125 originally unresolved noisy diffusive pairs. Including available saved states detects a negative derivative in 122/125. Initial-state checks are independent of later integration accuracy; checks on unresolved saved states describe only that finite-dimensional trajectory.

A negative dN/d(uxx) diagnoses a local violation of forward-parabolic monotonicity. It is not elevated into a theorem that every nonlinear continuum initial-value problem is ill posed. Positive finite samples do not give a global certificate.

One deterministic representative was selected in each system/noise stratum containing an unresolved original rollout: lexically first failed integration, or greatest finite grid discrepancy if none failed. The exact IDs were saved before evaluation. Tighter RK45 at nx128/256 and independent Radau at nx128 use rtol=1e-9 and atol=1e-11; Radau has an independently checked dense analytic Fourier Jacobian. Every RHS-budget failure received a separately declared uniform eightfold budget extension; original failures remain.

- The clean-advection representative is resolved after refinement: RK128/RK256 differ by 2.15e-5, and RK128/Radau128 by 7.96e-9. The original unresolved flag was not evidence of a physical model failure.
- The noisy-heat representative has RK128/Radau128 agreement of 8.75e-10, while RK256 remains budget-limited. Spatial validation remains incomplete.
- The noisy-advection representative completes after the budget extension. At t=0.2 the temporal-method difference is 7.83e-8 but the spatial difference is 0.04739, identifying spatial sensitivity despite agreement of the two time integrators. At t=0.8 the differences reach 0.1903 and 0.5426 respectively, so its late trajectory remains unvalidated. Time-resolved discrepancies are saved separately.
- Noisy advection-diffusion and reaction-diffusion representatives reach the declared absolute-state limit with both RK45 and Radau at closely matching nx128 times (about 0.02114 and 0.01291); finer explicit grids reach the limit earlier. Together with negative initial principal derivatives, this is a structural diagnostic independent of an RK45-only failure. A state-limit event alone is not a proof of continuum blow-up.

The original 185/360 neural paired-grid pass count is preserved. Representative follow-ups do not retrospectively convert the whole population into validated trajectories. Full protocols and records are in `forward_audit/` and `forward_audit/budget_extension/`. Additional accepted-step extrema accompany completed refined solves. `timewise_method_grid_differences.csv` distinguishes early-time and late-time agreement. The two new Jacobian checks bring the numerical test count to 13.
'''
    (output/'README.md').write_text(text)
    put_json(output/'REPORT_PROVENANCE.json',dict(report_source_sha256=sha(__file__),numerical_source_sha256=sha(output/'source.py'),
        integrity=verification,summary=summary))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.output)
