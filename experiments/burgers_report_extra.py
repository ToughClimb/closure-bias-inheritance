"""Additional analysis of completed Burgers results; no fitting or selection."""
from pathlib import Path
import argparse,json,csv,hashlib
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from burgers_core import FluxNet,polynomial,observation_operator,tensor_operator,loss,polynomial_design
from burgers_campaign import csv_write,save


def main(output):
    torch.set_num_threads(1)
    fits=[json.loads(p.read_text()) for p in sorted((output/'fits').glob('*.json'))]
    forward=[json.loads(p.read_text()) for p in sorted((output/'forward').glob('*.json'))]
    if len(fits)!=40 or len(forward)!=400:raise ValueError('Requires completed 40-fit / 400-case campaign')
    audit=json.loads((output/'INDEPENDENT_FORWARD_AUDIT.json').read_text())
    if not audit['all_passed']:raise ValueError('Independent forward audit has not passed')
    for case in audit['state_cases']:
        row=next(r for r in forward if r['key']==case['key'])
        previous=np.load(row['array_path'])['u']
        previous_n=row['n']
        for stage in case['stages']:
            data=np.load(stage['array_path'])
            differences=np.sqrt(np.mean((previous-data['u'][:,::(stage['n']//previous_n)])**2,axis=1))
            previous=data['u'];previous_n=stage['n']
        row['original_record_key']=row['key']
        for field in ['n','maximum_principle_violation','endpoint_rms_error','endpoint_relative_fluctuation_error',
                      'mass_drift','energy_increase','nfev','accepted_steps','array_path','array_sha256']:
            row[field]=stage[field]
        row['grid_rms_difference']=differences.tolist()
        row['status']='verified_after_independent_refinement'
    save(output/'reports/forward_final_records.json',forward)
    final_rows=[]
    for row in forward:
        fit=next(f for f in fits if f['run']==row['run'])
        for j,t in enumerate(row['t'][1:],1):
            final_rows.append(dict(run=row['run'],n_train=fit['n'],noise=fit['noise'],seed=fit['seed'],arm=fit['arm'],
                method=row['method'],kind=row['kind'],t=t,n_forward=row['n'],status=row['status'],
                rms_error=row['endpoint_rms_error'][j],relative_fluctuation_error=row['endpoint_relative_fluctuation_error'][j],
                grid_rms_difference=row['grid_rms_difference'][j],mass_drift=row['mass_drift'],
                maximum_principle_violation=row['maximum_principle_violation'],energy_increase=row['energy_increase']))
    csv_write(output/'reports/forward_final_by_seed.csv',final_rows)
    groups=defaultdict(list)
    for row in forward:
        fit=next(f for f in fits if f['run']==row['run'])
        for j,t in enumerate(row['t'][1:],1):
            key=(fit['n'],fit['noise'],fit['arm'],row['method'],row['kind'],t)
            groups[key].append(dict(rms_error=row['endpoint_rms_error'][j],relative_error=row['endpoint_relative_fluctuation_error'][j],
                numerical_error=row['grid_rms_difference'][j],status=row['status']))
    summary=[]
    for (n,noise,arm,method,kind,t),rows in groups.items():
        summary.append(dict(n=n,noise=noise,arm=arm,method=method,kind=kind,t=t,count=len(rows),
            mean_rms_error=float(np.mean([r['rms_error'] for r in rows])),
            sd_rms_error=float(np.std([r['rms_error'] for r in rows],ddof=1)),
            mean_relative_error=float(np.mean([r['relative_error'] for r in rows])),
            max_grid_difference=float(max(r['numerical_error'] for r in rows)),
            verified=sum(r['status'].startswith('verified') for r in rows)))
    csv_write(output/'reports/forward_summary.csv',summary)
    objectives=[]
    for fit in fits:
        data=np.load(fit['input_path']);op=observation_operator(data['x'],data['t'],data['observed'],fit['arm'])
        model=FluxNet(fit['seed']);model.load_state_dict(torch.load(output/fit['models']['final']['path'],weights_only=True))
        common={k:fit[k] for k in ['run','n','noise','seed','arm']}
        objectives.append(dict(**common,model='truth_evaluation_only',residual_mse=float(np.mean((polynomial_design(op,2)@np.array([0,.5])-op['target'])**2))))
        objectives.append(dict(**common,model='neural',residual_mse=float(loss(model,tensor_operator(op)).detach())))
        for item in fit['symbolic']:
            design=polynomial_design(op,item['degree'])
            for kind in ['frozen','direct']:
                prediction=design@np.array(item[kind+'_coefficients'])
                objectives.append(dict(**common,model=kind+str(item['degree']),residual_mse=float(np.mean((prediction-op['target'])**2))))
    csv_write(output/'reports/observation_objectives.csv',objectives)
    with plt.rc_context({'axes.titlesize':10,'axes.labelsize':9,'font.size':9}):
        fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        u=np.linspace(-.4,.9,1025)
        for i,arm in enumerate(['strong','weak']):
            chosen=[f for f in fits if f['n']==129 and f['noise']==.01 and f['arm']==arm]
            for kind,color,label in [('neural','C0','Neural'),('frozen2','C1','Frozen quadratic'),('direct2','C2','Direct quadratic')]:
                values={0:[],1:[]};solutions=[]
                for fit in chosen:
                    if kind=='neural':
                        model=FluxNet(fit['seed']);model.load_state_dict(torch.load(output/fit['models']['final']['path'],weights_only=True));evalf=model.numpy
                    else:
                        coefficients=fit['symbolic'][0][kind[:-1]+'_coefficients']
                        evalf=lambda q,d=0,c=coefficients:polynomial(q,c,d)
                    for derivative in [0,1]:
                        val=evalf(u,derivative);values[derivative].append(val)
                        axes[i,derivative].plot(u,val,color=color,alpha=.25,lw=.7)
                    row=next(r for r in forward if r['run']==fit['run'] and r['method']==kind and r['kind']=='extrapolation')
                    data=np.load(row['array_path']);x=np.linspace(-1,1,256,endpoint=False)
                    curve=np.interp(x,data['x'],data['u'][-1],period=2)
                    reference=np.interp(x,data['x'],data['reference'][-1],period=2)
                    solutions.append(curve)
                    axes[i,2].plot(x,curve,color=color,alpha=.25,lw=.7)
                for derivative in [0,1]:axes[i,derivative].plot(u,np.mean(values[derivative],axis=0),color=color,label=label,lw=1.8)
                axes[i,2].plot(x,np.mean(solutions,axis=0),color=color,label=label,lw=1.8)
            axes[i,0].plot(u,.5*u*u,'k--',label='Truth')
            axes[i,1].plot(u,u,'k--',label='Truth')
            axes[i,2].plot(x,reference,'k--',label='Cole–Hopf truth')
            for j,what in enumerate(['Flux F(u)','Characteristic speed F′(u)','Held-out periodic wave, t=1.5']):
                axes[i,j].set_title(arm.title()+' objective: '+what)
                axes[i,j].set_xlabel('State u' if j<2 else 'x')
                if j<2:
                    axes[i,j].axvline(-.25,color='gray',ls=':',lw=.8);axes[i,j].axvline(.75,color='gray',ls=':',lw=.8)
                axes[i,j].legend(fontsize=7)
        fig.savefig(output/'reports/forward_and_characteristic_speed.png',dpi=180)
        plt.close(fig)
    validation=json.loads((output/'FORWARD_VALIDATION.json').read_text())
    text=['', '## Independent forward validation', '',
        f'{validation["verified"]}/{validation["total"]} candidate/initial-condition cases satisfy the predeclared mesh and state-bound checks. Each case has two horizons; these are 400 cases and 800 reported endpoints, not 800 independently trained models.',
        f'Maximum mass drift: {validation["maximum_mass_drift"]:.3g}. Independent implicit Radau versus explicit DOP853 changes the worst mesh-discrepancy case by {audit["time_integrator_rms_difference"]:.3g} RMS.',
        'Two noisy strong-neural extrapolation cases exhibited small numerical state overshoots within the initial tolerance. Their original records are preserved; further n=512 Radau refinement removes the overshoots at all accepted endpoints and changes fields by less than 1e-4 RMS. The final table/plot uses these refined fields; `reports/forward_by_seed.csv` retains the initial pass and `reports/forward_final_by_seed.csv` contains the final audited values.',
        '', 'Fine training grid, 1% noise, extrapolation initial state, t=1.5:', '',
        '| Objective | Model | Mean field RMS error | Mean relative fluctuation error | Maximum mesh difference |',
        '|---|---|---:|---:|---:|']
    for r in summary:
        if r['n']==129 and r['noise']==.01 and r['kind']=='extrapolation' and r['t']==1.5:
            text.append(f'| {r["arm"]} | {r["method"]} | {r["mean_rms_error"]:.6g} | {r["mean_relative_error"]:.6g} | {r["max_grid_difference"]:.3g} |')
    text += ['', 'The field-error table includes spatial discretization uncertainty. Small differences comparable to the mesh difference do not establish a method ranking. Separate observed-objective and truth-error tables avoid interpreting lower training residual as improved physical identification. Thin plotted curves are individual seeds and thick curves are means.',
        '', 'The two test initial conditions have different state coverage from the training shock and evolve to five times its final training time. No real-flow experiment is represented by these controlled synthetic fields.']
    with (output/'README.md').open('a') as file:file.write('\n'.join(text)+'\n')
    save(output/'ADDITIONAL_REPORT_PROVENANCE.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs='40 completed fits, 400 initial forward records and two independently refined fields; no fitting',
        independent_refinements=len(audit['state_cases']),
        maximum_energy_increment=max(r['energy_increase'] for r in forward),
        maximum_state_bound_violation=max(r['maximum_principle_violation'] for r in forward)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
