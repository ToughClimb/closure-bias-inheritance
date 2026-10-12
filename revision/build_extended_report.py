"""Additional validation summaries, correction effects and portable raw curves."""
from collections import defaultdict
from build_report import stats,write_csv
from revision_core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def records(directory):
    for f in input_glob(Path('evidence')/directory, '*.json'):
        r=json.loads(f.read_text())
        if r.get('status')=='complete' and r.get('classification')!='runtime_pilot':yield r


def summarize(rows,keys,metrics):
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,rs in sorted(grouped.items()):
        for metric in metrics:
            result.append({**dict(zip(keys,key)),('statistic' if 'metric' in keys else 'metric'):metric,
                           **stats([r.get(metric) for r in rs]),'total_runs':len(rs)})
    return result


def convergence_rollout_report():
    rows=[];groups=defaultdict(list)
    for r in records('convergence_validation'):
        for e in r['entries']:
            c=r['config'];key=(c['seed'],e['epoch'],e['method'],e['pattern'])
            groups[key].append(e)
            rows.append(dict(id=r['id'],data_seed=c['seed'],init_seed=c['init_seed'],epoch=e['epoch'],
                method=e['method'],pattern=e['pattern'],status=e['info']['status'],final_error=e['final_error']))
    datasets=[];curves=defaultdict(list)
    for key,es in sorted(groups.items()):
        stable=[e for e in es if e['info']['status']=='stable']
        complete=len(es)==3 and len(stable)==3
        datasets.append(dict(zip(['data_seed','epoch','method','pattern'],key),restarts=len(es),stable=len(stable),
            restart_mean_final_error=float(np.mean([e['final_error'] for e in stable])) if complete else None))
        if complete:
            curves[key[1:]].append((stable[0]['times'],np.mean([e['errors'] for e in stable],axis=0)))
    write_csv(ROOT/'evidence/convergence_rollout_raw.csv',rows)
    write_csv(ROOT/'evidence/convergence_rollout_dataset.csv',datasets)
    write_csv(ROOT/'evidence/convergence_rollout_summary.csv',summarize(datasets,['epoch','method','pattern'],['restart_mean_final_error']))
    if curves:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,pattern in zip(axes,['full_support','midpoint1.1_amplitude0.25']):
            for epoch,color in [(150,'#1f77b4'),(500,'#ff7f0e'),(1500,'#2ca02c')]:
                for method in ['neural','frozen']:
                    values=curves.get((epoch,method,pattern),[])
                    if not values:continue
                    t=values[0][0];v=np.stack([a for _,a in values]);avg=v.mean(0);sd=v.std(0,ddof=1) if len(v)>1 else np.zeros_like(avg)
                    ax.plot(t,avg,label=f'{method}, {epoch} epochs',color=color,ls='--' if method=='frozen' else '-')
                    if method=='neural':ax.fill_between(t,np.maximum(0,avg-sd),avg+sd,color=color,alpha=.1)
            baselines=[]
            for r in records('rollout'):
                c=r['config']
                if c['group']=='main' and c['case']=='case_exp' and c['setting']=='clean':
                    for e in r['methods']['weak_poly']:
                        if e['pattern'].replace('mean','midpoint')==pattern and e['info']['status']=='stable':baselines.append(e)
            if baselines:ax.plot(baselines[0]['times'],np.mean([e['errors'] for e in baselines],axis=0),color='gray',ls=':',label='weak polynomial')
            ax.set(xlabel='Time',ylabel='Relative state error',title=pattern);ax.legend(fontsize=7)
        fig.suptitle('Fixed budgets: average all three restarts per dataset, then dataset mean ± sample SD; counts/failures in CSV')
        fig.tight_layout();fig.savefig(ROOT/'figures/convergence_rollouts.png');fig.savefig(ROOT/'figures/convergence_rollouts.pdf');plt.close(fig)


def extended():
    (ROOT/'figures').mkdir(exist_ok=True)
    convergence_rollout_report()
    two=[];sr=[];roll=[];short=[];paired=[];magnitudes=[];threshold_ranges=[];expressions=[]
    for r in records('2d'):
        c=r['config'];setting='clean' if c['noise']==0 else 'noise3'
        for name,m in r['metrics'].items():two.append(dict(id=r['id'],case=c['case'],setting=setting,seed=c['seed'],method=name,**m))
        for e in r['rollout']:
            roll.append(dict(id=r['id'],group='2d',case=c['case'],setting=setting,seed=c['seed'],method=e['method'],
                pattern=e['pattern'],status=e['info']['status'],last_time=e['times'][-1],final_error=e['errors'][-1] if e['info']['status']=='stable' else None,
                **{f'T{t:g}':e['errors'][int(np.flatnonzero(np.isclose(e['times'],t))[0])] if np.any(np.isclose(e['times'],t)) else None for t in [.1,.25,.3,.5]}))
        out=ROOT/'outputs/runs'/r['id'];q=load_curves(r['id'])
        write_csv(ROOT/'evidence/curves'/f"{r['id']}.csv",[dict(u=float(z),**{k:float(q[k][i]) for k in q.files if k not in ['grid','state_hist','state_bins']}) for i,z in enumerate(q['grid'])])
        write_csv(ROOT/'evidence/histories'/f"{r['id']}.csv",load_history(r['id']))
    for r in records('pysr'):
        c=r['config']
        for branch,m in r['metrics'].items():sr.append(dict(id=r['id'],case=c['case'],setting=c['setting'],seed=c['seed'],branch=branch,**m))
        for branch,e in r['expressions'].items():expressions.append(dict(id=r['id'],stage='pysr',case=c['case'],setting=c['setting'],seed=c['seed'],branch=branch,**e))
        q=load_curves(r['id'],pysr=True)
        write_csv(ROOT/'evidence/curves'/f"pysr_{r['id']}.csv",[dict(u=float(z),D=float(q['D'][i]),R=float(q['R'][i])) for i,z in enumerate(q['grid'])])
        validation=r.get('rollout')
        companion=input_path(Path('evidence/pysr_validation')/f"{r['id']}.json")
        if validation is None and companion.exists():validation=json.loads(companion.read_text())['rollout']
        if validation is not None:
            roll.append(dict(id=r['id'],group='pysr',case=c['case'],setting=c['setting'],seed=c['seed'],method='pysr',pattern='full_support',
                status=validation['info']['status'],last_time=validation['times'][-1],final_error=validation['errors'][-1] if validation['info']['status']=='stable' else None,
                **{f'T{str(t).replace(".","_")}':validation['errors'][int(np.flatnonzero(np.isclose(validation['times'],t))[0])] if np.any(np.isclose(validation['times'],t)) else None for t in [.1,.25,.5]}))
    for r in records('rollout'):
        c=r['config']
        for name,entries in r['methods'].items():
            for e in entries:
                roll.append(dict(id=r['id'],group=c['group'],case=c['case'],setting=c['setting'],seed=c['seed'],method=name,
                    pattern=e['pattern'],status=e['info']['status'],last_time=e['times'][-1],final_error=e['errors'][-1] if e['info']['status']=='stable' else None,
                    T0_1=e.get('error_T0.1'),T0_25=e.get('error_T0.25'),T0_5=e.get('error_T0.5'),
                    outside_closure_support_fraction=e['outside_closure_support_fraction']))
    for r in records('short_validation'):
        c=r['config']
        for name,policies in r['methods'].items():
            for policy,m in policies.items():short.append(dict(id=r['id'],case=c['case'],setting=c['setting'],seed=c['seed'],method=name,policy=policy,
                status=m['info']['status'],trajectory_error=m['trajectory_error'],final_error=m['final_error'],clip_steps=m['info']['clip_steps']))
    for r in records('paired'):
        c=r['config']
        for name,m in r['methods'].items():
            for t in [.1,.5]:paired.append(dict(id=r['id'],kind='clipping',setting=c['setting'],seed=c['seed'],method=name,metric=f'rollout_T{t:g}',
                original=m.get(f'legacy_error_T{t:g}'),corrected=m.get(f'unclipped_error_T{t:g}'),delta=m.get(f'paired_error_difference_T{t:g}'),
                original_status=m['legacy']['status'],corrected_status=m['unclipped']['status'],clip_steps=m['legacy']['clip_steps']))
        a,b=r['compression_grid_effect']
        for branch in ['D','R']:
            for metric in ['error','BIR']:
                key=metric+'_'+branch;paired.append(dict(id=r['id'],kind='support_density_choice',setting=c['setting'],seed=c['seed'],method='frozen',metric=key,
                    original=a[key],corrected=b[key],delta=b[key]-a[key]))
    raw=list(records('raw'));lookup={r['id']:r for r in raw}
    for r in raw:
        c=r['config'];out=ROOT/'outputs/runs'/r['id'];epochs=sorted(map(int,r['epochs']))
        for epoch in epochs:
            snap=r['epochs'][str(epoch)];weights=c.get('weights',DEFAULT_WEIGHTS)
            for term in DEFAULT_WEIGHTS:magnitudes.append(dict(id=r['id'],group=c['group'],case=c['case'],setting=c['setting'],variant=c.get('variant','default'),seed=c['seed'],epoch=epoch,
                term=term,weight=weights[term],raw=snap['last_pre_step_loss'][term],weighted=weights[term]*snap['last_pre_step_loss'][term]))
            for branch,e in zip(['D','R'],snap['symbols']):expressions.append(dict(id=r['id'],stage='restricted',case=c['case'],setting=c['setting'],seed=c['seed'],epoch=epoch,branch=branch,**e))
            for branch,m in snap['compression'].items():paired.append(dict(id=r['id'],kind='bound_normalization',setting=c['setting'],seed=c['seed'],method=c['group']+'/'+c.get('variant','default')+'/'+str(epoch),metric=branch,
                original=m['bound']/m['norm_conversion'],corrected=m['bound'],delta=m['bound']-m['bound']/m['norm_conversion'],norm_conversion=m['norm_conversion']))
        if 'thresholds' in r:
            for branch in ['D','R']:
                vals=[t['BIR_'+branch] for t in r['thresholds']]
                threshold_ranges.append(dict(id=r['id'],group=c['group'],case=c['case'],setting=c['setting'],seed=c['seed'],branch=branch,
                    BIR_min=min(vals),BIR_max=max(vals),BIR_span=max(vals)-min(vals),distinct_families=len(set(t['family_'+branch] for t in r['thresholds']))))
        if c['group'] in ['paired','coarse_corrected']:
            other=r['id'].replace('space_time_trapezoid','space_time_rectangle') if c['group']=='paired' else r['id'].replace('coarse_corrected_','coarse_',1)
            if other==r['id'] or other not in lookup:continue
            before=lookup[other]
            if before['data_hash']!=r['data_hash']:raise ValueError('Unpaired observation data')
            methods=['neural','frozen']+list(r.get('controls',{}))
            for name in methods:
                a=before['epochs']['150'][name] if name in ['neural','frozen'] else before['controls'][name]['metrics']
                b=r['epochs']['150'][name] if name in ['neural','frozen'] else r['controls'][name]['metrics']
                for branch in ['D','R']:paired.append(dict(id=r['id'],kind='time_quadrature' if c['group']=='paired' else 'nyquist_test',setting=c['setting'],seed=c['seed'],method=name,metric=branch,
                    original=a[branch],corrected=b[branch],delta=b[branch]-a[branch]))
            for branch in ['D','R']:
                a=before['epochs']['150']['compression'][branch]['BIR'];b=r['epochs']['150']['compression'][branch]['BIR']
                paired.append(dict(id=r['id'],kind='time_quadrature' if c['group']=='paired' else 'nyquist_test',setting=c['setting'],seed=c['seed'],method='frozen',metric='BIR_'+branch,original=a,corrected=b,delta=b-a))
        # Portable raw curves and exact diagnostic histories, without checkpoints.
        if (out/'all_curves.npz').exists():
            q=load_curves(r['id']);g=q['grid'];curve_rows=[dict(u=float(z),**{k:float(q[k][i]) for k in q.files if k not in ['grid','state_hist','state_bins']}) for i,z in enumerate(g)]
            write_csv(ROOT/'evidence/curves'/f"{r['id']}.csv",curve_rows)
            if 'state_hist' in q:write_csv(ROOT/'evidence/state_histograms'/f"{r['id']}.csv",[dict(lower=a,upper=b,density=v) for a,b,v in zip(q['state_bins'][:-1],q['state_bins'][1:],q['state_hist'])])
        if (out/'history.json').exists():write_csv(ROOT/'evidence/histories'/f"{r['id']}.csv",load_history(r['id']))
    for name,rows in [('2d_metrics_raw',two),('pysr_metrics_raw',sr),('rollout_raw',roll),('short_validation_raw',short),('correction_effects_raw',paired),('loss_magnitudes_raw',magnitudes),('threshold_ranges_raw',threshold_ranges),('symbolic_expressions_raw',expressions)]:write_csv(ROOT/'evidence'/f'{name}.csv',rows)
    ts=summarize(two,['case','setting','method'],['D','R']);ps=summarize(sr,['case','setting','branch'],['true_error','compression','BIR','common_norm_bound'])
    rr=summarize(roll,['group','case','setting','method','pattern'],['final_error','T0_1','T0_25','T0_5','T0.1','T0.25','T0.3','T0.5'])
    ss=summarize(short,['case','setting','method','policy'],['trajectory_error','final_error'])
    cs=summarize(paired,['kind','setting','method','metric'],['original','corrected','delta'])
    for name,rows in [('2d_metrics_summary',ts),('pysr_metrics_summary',ps),('rollout_summary',rr),('short_validation_summary',ss),('correction_effects_summary',cs)]:write_csv(ROOT/'evidence'/f'{name}.csv',rows)
    # Failure rates are a result; uncertainty excludes failed horizons, with n and total.
    grouped=defaultdict(list)
    for r in roll:grouped[tuple(r[k] for k in ['group','case','setting','method','pattern'])].append(r)
    write_csv(ROOT/'evidence/rollout_stability.csv',[dict(zip(['group','case','setting','method','pattern'],k),attempts=len(v),stable=sum(r['status']=='stable' for r in v)) for k,v in sorted(grouped.items())])
    convergence=defaultdict(list)
    for r in raw:
        if r['config']['group']=='convergence':convergence[r['config']['seed']].append(r)
    ds=[]
    for seed,rs in sorted(convergence.items()):
        for epoch in [150,500,1500]:
            for branch in ['D','R']:
                for method in ['neural','frozen']:
                    values=[r['epochs'][str(epoch)][method][branch] for r in rs]
                    ds.append(dict(data_seed=seed,epoch=epoch,branch=branch,method=method,restarts=len(rs),
                        restart_mean=float(np.mean(values)),restart_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
                        restart_min=float(np.min(values)),restart_max=float(np.max(values))))
    write_csv(ROOT/'evidence/convergence_dataset_raw.csv',ds)
    write_csv(ROOT/'evidence/convergence_dataset_summary.csv',summarize([r for r in ds if r['restarts']==3],['epoch','branch','method'],['restart_mean']))
    diagnostics=[]
    for rs in convergence.values():
        for r in rs:
            h=load_history(r['id']);final=h[-1];earlier=h[-151]
            diagnostics.append(dict(id=r['id'],data_seed=r['config']['seed'],init_seed=r['config']['init_seed'],
                loss150=h[149]['loss'],loss500=h[499]['loss'],loss1350=earlier['loss'],loss1500=final['loss'],
                relative_objective_decline_last150=(earlier['loss']-final['loss'])/earlier['loss'],final_gradient_norm=final['grad_norm'],
                interpretation='Fixed 1500-epoch budget; objective decline/gradient are diagnostics, not proof of global convergence'))
    write_csv(ROOT/'evidence/convergence_diagnostics.csv',diagnostics)
    sensitivity=defaultdict(list)
    for r in raw:
        if 'thresholds' not in r:continue
        c=r['config']
        for t in r['thresholds']:
            for b in ['D','R']:sensitivity[(c['group'],c['case'],c['setting'],b,t['relative'],t['absolute'])].append(t['BIR_'+b])
    write_csv(ROOT/'evidence/thresholds_summary.csv',[dict(zip(['group','case','setting','branch','relative','absolute'],k),**stats(v)) for k,v in sorted(sensitivity.items())])
    if sensitivity:
        cases=['case_a','case_b','case_exp','case_c','case_trig','case_tanh'];rels=[0.,.01,.05,.2,1.];abss=[0.,1e-10,1e-8,1e-6,1e-4]
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        for ax,case in zip(axes.ravel(),cases):
            group='main' if case in cases[:3] else 'families'
            a=np.array([[np.mean(sensitivity[(group,case,'clean','D',rt,at)]) if (group,case,'clean','D',rt,at) in sensitivity else np.nan for rt in rels] for at in abss])
            im=ax.imshow(np.log10(a),aspect='auto',cmap='coolwarm',norm=TwoSlopeNorm(vmin=-.5,vcenter=0.,vmax=1.))
            ax.set_xticks(range(5),['0','.01','.05','.2','1']);ax.set_yticks(range(5),['0','1e-10','1e-8','1e-6','1e-4']);ax.set(title=case,xlabel='Relative tolerance',ylabel='Absolute MSE tolerance');fig.colorbar(im,ax=ax,label='log10(mean diffusion BIR)')
        fig.suptitle('Frozen selection sensitivity, clean five-seed means; same numerical models');fig.tight_layout();fig.savefig(ROOT/'figures/threshold_sensitivity.png');plt.close(fig)
    plots(ts,ps,roll,raw)
    print('2D metrics',len(two),'PySR branches',len(sr),'rollout cases',len(roll),'paired effects',len(paired))


def plots(two,sr,roll,raw):
    dest=ROOT/'figures';dest.mkdir(exist_ok=True);plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':9,'savefig.dpi':170})
    if two:
        cases=sorted(set(r['case'] for r in two));fig,axes=plt.subplots(1,len(cases),figsize=(4.3*len(cases),4),squeeze=False)
        for ax,case in zip(axes[0],cases):
            rows=[r for r in two if r['case']==case and r['metric']=='R'];labels=[r['setting']+' '+r['method'] for r in rows]
            ax.bar(range(len(rows)),[r['mean'] for r in rows],yerr=[r['sd'] or 0 for r in rows],color='#598bab',capsize=2)
            ax.set_xticks(range(len(rows)),labels,rotation=60,ha='right');ax.set(title=case,ylabel='Relative R error',yscale='log')
        fig.suptitle('Larger 2D training; mean ± sample SD, seed counts in CSV');fig.tight_layout();fig.savefig(dest/'2d_controls.png');plt.close(fig)
    if sr:
        rows=[r for r in sr if r['metric']=='BIR'];fig,ax=plt.subplots(figsize=(11,4))
        ax.errorbar(range(len(rows)),[r['mean'] for r in rows],yerr=[r['sd'] or 0 for r in rows],fmt='o',capsize=3);ax.axhline(1,ls='--',color='gray')
        ax.set_xticks(range(len(rows)),[r['case']+' '+r['setting']+' '+r['branch'] for r in rows],rotation=60,ha='right');ax.set(ylabel='BIR',title='PySR fits frozen surrogates; mean ± sample SD')
        fig.tight_layout();fig.savefig(dest/'pysr_bir.png');plt.close(fig)
    support=[r for r in raw if r['config']['group']=='support' and r['config']['seed']==0]
    if support:
        fig,axes=plt.subplots(len(support),3,figsize=(13,3.4*len(support)),squeeze=False)
        for i,r in enumerate(support):
            q=load_curves(r['id']);g=q['grid'];m=r['epochs']['150']['neural']
            for j,branch in enumerate(['D','R']):
                ax=axes[i,j]
                for name in ['true','weak_poly','strong_poly','neural','frozen','refit_weak','refit_data']:
                    ax.plot(g,q[name+'_'+branch],label=name,ls='--' if name in ['frozen','refit_data'] else '-')
                ax.axvspan(m['visited_min'],m['visited_max'],alpha=.1,color='gray');ax.set(xlabel='u',ylabel=branch+'(u)',title='Narrow support '+r['config']['setting']);ax.legend(fontsize=6)
            axes[i,2].bar((q['state_bins'][:-1]+q['state_bins'][1:])/2,q['state_hist'],width=np.diff(q['state_bins']),color='#9eb9cb')
            axes[i,2].set(xlabel='u',ylabel='Observed density',title='Shaded closure interval is visited min/max')
        fig.tight_layout();fig.savefig(dest/'narrow_support_closures.png');fig.savefig(dest/'narrow_support_closures.pdf');plt.close(fig)
    rs=[r for r in records('rollout') if r['config']['case']=='case_exp' and r['config']['group']=='main']
    if rs:
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for j,setting in enumerate(['clean','noise3','noise5']):
            for i,pattern in enumerate(['full_support','mean1.1_amplitude0.25']):
                ax=axes[i,j]
                for name in ['neural','frozen','weak_poly','refit_weak','refit_data','refit_pde','spacetime_poly']:
                    arrays=[];times=None
                    for r in rs:
                        if r['config']['setting']!=setting:continue
                        for e in r['methods'].get(name,[]):
                            if e['pattern']==pattern and e['info']['status']=='stable':arrays.append(e['errors']);times=e['times']
                    if arrays:
                        v=np.asarray(arrays);avg=v.mean(0);sd=v.std(0,ddof=1) if len(v)>1 else np.zeros(v.shape[1]);line=ax.plot(times,avg,label=name)[0]
                        ax.fill_between(times,np.maximum(0,avg-sd),avg+sd,alpha=.12,color=line.get_color())
                ax.set(title=setting+' '+pattern.replace('mean','midpoint'),xlabel='Time',ylabel='Relative state error')
                if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6)
        fig.suptitle('Longer Exp validation; stable runs only, mean ± sample SD; failures retained in CSV');fig.tight_layout();fig.savefig(dest/'long_rollout_growth.png');plt.close(fig)
    rs=[r for r in raw if r['config']['group']=='ablation']
    if rs:
        variants=list(dict.fromkeys(r['config']['variant'] for r in rs));fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,setting in zip(axes,['clean','noise3']):
            a=np.full((len(variants),3),np.nan)
            for i,variant in enumerate(variants):
                subset=[r for r in rs if r['config']['variant']==variant and r['config']['setting']==setting]
                if subset:a[i]=[np.mean([r['epochs']['150']['neural'][b] for r in subset]) for b in ['D','R']]+[np.mean([abs(r['epochs']['150']['compression']['R']['BIR']-1) for r in subset])]
            im=ax.imshow(np.log10(np.maximum(a,1e-8)),aspect='auto',cmap='viridis');ax.set_yticks(range(len(variants)),variants);ax.set_xticks(range(3),['D error','R error','|BIR_R−1|']);ax.set_title(setting);fig.colorbar(im,ax=ax,label='log10(mean value)')
        fig.suptitle('Loss ablations; fixed 150-epoch budget, complete seed counts in CSV');fig.tight_layout();fig.savefig(dest/'loss_ablation.png');plt.close(fig)


if __name__=='__main__':extended()
