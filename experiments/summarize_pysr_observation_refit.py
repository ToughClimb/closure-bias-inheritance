"""Produce per-seed results, paired uncertainty and plots for observation refits."""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'revision'))
from project_paths import ROOT,INPUT_ROOT
from artifact_io import json_write,load_curves
from build_report import write_csv,stats

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refit-root',type=Path,required=True)
    args=parser.parse_args();refit=args.refit_root.resolve()
    records=[];failures=[]
    for path in sorted((refit/'evidence/pysr_observation_refit').glob('*.json')):
        row=json.loads(path.read_text())
        (records if row['status']=='complete' else failures).append(row)
    if not records:raise ValueError('No complete observation-refit records')
    metric_rows=[];paired=[];validation=[];resources=[]
    for row in records:
        cfg=row['config'];base={k:cfg[k] for k in ('case','setting','seed')};ident=row['id']
        raw=json.loads((INPUT_ROOT/'evidence/raw'/f'{ident}.json').read_text())
        epoch=raw['epochs'][str(cfg['epochs'])]
        methods={'neural':epoch['neural'],'restricted_frozen':epoch['frozen'],
                 'pysr_frozen':row['float64_frozen_metrics'],'pysr_refit':row['metrics']}
        methods.update({key:control['metrics'] for key,control in raw['controls'].items()})
        original_forward=INPUT_ROOT/'evidence/rollout'/f'{ident}.json'
        if original_forward.exists():
            forward=json.loads(original_forward.read_text())
            if 'forward_refit' in forward:methods['restricted_forward_refit']=forward['forward_refit']['metrics']
        for method,values in methods.items():
            for metric in ('D','R','D_visited','R_visited','D_extrapolation','R_extrapolation','min_D'):
                metric_rows.append(dict(base,id=ident,method=method,metric=metric,value=values.get(metric)))
        for branch in ('D','R'):
            frozen=row['float64_frozen_metrics'][branch];value=row['metrics'][branch]
            paired.append(dict(base,id=ident,branch=branch,frozen=frozen,refit=value,difference=value-frozen,
                               error_ratio=value/frozen if frozen else None,improved=value<frozen))
        for pattern,entry in row['validation'].items():
            for method,values in entry['methods'].items():
                validation.append(dict(base,id=ident,pattern=pattern,method=method,status=values['info']['status'],
                    final_error=values['final_error'],completed_saved_times=len(values['times']),total_saved_times=51))
        resources.append(dict(base,id=ident,fit_optimizer_converged=row['fit']['success'],
            fit_status=row['fit']['status'],actual_nfev=row.get('total_nfev',row['fit']['nfev']),
            allocated_nfev=row.get('allocated_total_nfev',row['max_nfev']),
            parameter_count=row['fit']['parameter_count'],cost=row['fit']['cost'],
            first_phase_cost=row.get('fit_40',row['fit'])['cost'],initial_cost=row.get('fit_40',row['fit'])['initial_cost'],
            continuation=row.get('continued',False),bitwise_observation_match=row['bitwise_observation_match'],
            failed_trial_evaluations=row['fit']['failed_trial_evaluations']))
        arrays=refit/'outputs/runs'/ident/'pysr_observation_refit.npz'
        with np.load(arrays) as values:
            keys=list(values.files);grid=values['grid']
            write_csv(ROOT/'curves'/f'{ident}.csv',[dict(u=float(u),**{key:float(values[key][i]) for key in keys if key!='grid'}) for i,u in enumerate(grid)])
    groups=defaultdict(list)
    for row in metric_rows:groups[tuple(row[k] for k in ('case','setting','method','metric'))].append(row)
    summary=[]
    for key,rows in sorted(groups.items()):
        if len({r['seed'] for r in rows})!=len(rows):raise ValueError('Duplicate seeds in comparison')
        summary.append(dict(zip(('case','setting','method','metric'),key),total_runs=len(rows),**stats([r['value'] for r in rows])))
    groups=defaultdict(list)
    for row in paired:groups[(row['case'],row['setting'],row['branch'])].append(row)
    paired_summary=[dict(zip(('case','setting','branch'),key),improved_seeds=sum(r['improved'] for r in rows),
                         total_runs=len(rows),**stats([r['difference'] for r in rows])) for key,rows in sorted(groups.items())]
    groups=defaultdict(list)
    for row in validation:groups[(row['case'],row['setting'],row['pattern'],row['method'])].append(row)
    stability=[dict(zip(('case','setting','pattern','method'),key),stable_runs=sum(r['status']=='stable' for r in rows),
                    total_runs=len(rows),**stats([r['final_error'] for r in rows])) for key,rows in sorted(groups.items())]
    for name,rows in [('metrics_raw',metric_rows),('metrics_summary',summary),('paired_raw',paired),
        ('paired_summary',paired_summary),('validation_raw',validation),('validation_summary',stability),('optimizer_status',resources)]:
        write_csv(ROOT/f'{name}.csv',rows)
    lines=['# Fixed expression observation refit results','',
        'Means and sample SD across five data seeds. Structural integer constants and powers are fixed; distinct floating coefficients are fitted to observations. The original Julia searches are reused.',
        '', 'Observations are explicitly reconstructed locally with the archived configuration, seeds and diagnostic checks. Saved arrays provide exact inputs for the new paired comparison. Original archive byte hashes do not match; bitwise identity with missing historical observation arrays is not claimed.',
        '', '| Case | Setting | Neural R | Frozen PySR R | Refit PySR R | Weak polynomial R | Improved seeds |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for case,setting in sorted({(r['config']['case'],r['config']['setting']) for r in records}):
        def cell(method):
            r=next(r for r in summary if r['case']==case and r['setting']==setting and r['method']==method and r['metric']=='R')
            return f"{r['mean']:.3g} ± {r['sd']:.2g}" if r['sd'] is not None else f"{r['mean']:.3g}"
        improvement=next(r for r in paired_summary if r['case']==case and r['setting']==setting and r['branch']=='R')
        lines.append(f"| {case} | {setting} | {cell('neural')} | {cell('pysr_frozen')} | {cell('pysr_refit')} | {cell('weak_poly')} | {improvement['improved_seeds']}/{improvement['total_runs']} |")
    converged=sum(r['fit_optimizer_converged'] for r in resources)
    lines += ['',f'Optimizer convergence: {converged}/{len(resources)}. Budget exhaustion and unsuccessful validation remain in the raw records.',
        '','Archive neural, polynomial and restricted-refit controls are retained as historical references. The new frozen versus refitted PySR comparison uses the same compiled expression and exact saved local inputs.',
        '','No conclusion about a global optimum or general bias-correction capability follows from optimizer termination alone.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160,'font.size':9})
    fig,axes=plt.subplots(3,2,figsize=(13,10),squeeze=False)
    plot_methods=['neural','pysr_frozen','pysr_refit','weak_poly']
    for i,case in enumerate(('case_a','case_b','case_exp')):
        for j,setting in enumerate(('clean','noise3')):
            ax=axes[i,j]
            for index,method in enumerate(plot_methods):
                rs=[r for r in metric_rows if r['case']==case and r['setting']==setting and r['method']==method and r['metric']=='R']
                values=[r['value'] for r in rs]
                if not values:continue
                ax.scatter(np.full(len(values),index),values,s=18,alpha=.65)
                ax.errorbar(index,np.mean(values),yerr=np.std(values,ddof=1) if len(values)>1 else 0,fmt='ks',capsize=3)
            ax.set(yscale='log',title=case+' '+setting,ylabel='Relative reaction error')
            ax.set_xticks(range(len(plot_methods)),['Neural','Frozen PySR','Refit PySR','Weak poly'],rotation=15)
    fig.suptitle('Fixed-expression observation refits; points are data seeds, black markers mean ± sample SD')
    fig.tight_layout();fig.savefig(ROOT/'reaction_errors.png');fig.savefig(ROOT/'reaction_errors.pdf');plt.close(fig)
    observation_root=refit
    if not (refit/'evidence/observations').exists():
        manifest=json.loads((refit/'evidence/manifest_pysr_observation_refit_continuation.json').read_text())
        observation_root=Path(manifest['base_root'])
    for case in ('case_a','case_b','case_exp'):
        fig,axes=plt.subplots(2,3,figsize=(15,8),squeeze=False)
        for i,setting in enumerate(('clean','noise3')):
            selected=next((r for r in records if r['config']['case']==case and r['config']['setting']==setting and r['config']['seed']==0),None)
            if selected is None:continue
            ident=selected['id'];reference=load_curves(ident)
            with np.load(refit/'outputs/runs'/ident/'pysr_observation_refit.npz') as values:
                curves={key:values[key].copy() for key in values.files}
            grid=curves['grid']
            for j,branch in enumerate(('D','R')):
                ax=axes[i,j]
                for label,values,color,style in [('Truth',curves['true_'+branch],'#222222','-'),
                    ('Neural',np.interp(grid,reference['grid'],reference['neural_'+branch]),'#b64646','-'),
                    ('Weak polynomial',np.interp(grid,reference['grid'],reference['weak_poly_'+branch]),'#2c8065','--'),
                    ('Frozen PySR',curves['frozen_'+branch],'#497bac','--'),('Refit PySR',curves['refit_'+branch],'#dc8a31','-')]:
                    ax.plot(grid,values,label=label,color=color,ls=style,lw=1.6)
                ax.set(xlabel='u',ylabel=branch+'(u)',title=case+' '+setting+' '+branch);ax.legend(fontsize=8)
            with np.load(observation_root/'evidence/observations'/f'{ident}.npz') as values:
                axes[i,2].hist(values['u'].ravel(),bins=32,range=(grid[0],grid[-1]),density=True,color='#9eb9cb')
            axes[i,2].set(xlabel='u',ylabel='Observed state density',title='Saved local observations, seed 0')
        fig.suptitle('Fixed seed-0 closure curves and observed-state support');fig.tight_layout()
        fig.savefig(ROOT/f'{case}_closures.png');fig.savefig(ROOT/f'{case}_closures.pdf');plt.close(fig)
    files=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(ROOT.glob('*.csv'))]
    json_write(ROOT/'report_manifest.json',dict(status='complete' if len(records)==30 and not failures else 'partial',
        complete_records=len(records),failed_records=len(failures),optimizer_converged=converged,refit_root=str(refit),
        historical_input_root=str(INPUT_ROOT),report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),outputs=files))
    print('reported',len(records),'inputs','optimizer converged',converged,'report directory',ROOT)


if __name__=='__main__':main()
