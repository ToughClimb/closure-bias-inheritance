"""Build compact raw/uncertainty summaries and publication-style figures."""
import csv
from collections import defaultdict
from revision_core import *
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def stats(values):
    v=np.asarray([x for x in values if x is not None and np.isfinite(x)],dtype=float)
    if not len(v):return dict(n=0,mean=None,sd=None,ci95_low=None,ci95_high=None)
    avg=float(v.mean());sd=float(v.std(ddof=1)) if len(v)>1 else None
    half=float(student_t.ppf(.975,len(v)-1)*sd/np.sqrt(len(v))) if len(v)>1 else None
    return dict(n=len(v),mean=avg,sd=sd,ci95_low=avg-half if half is not None else None,ci95_high=avg+half if half is not None else None)


def write_csv(path,rows):
    path = Path(path)
    if not rows:
        path.unlink(missing_ok=True)
        return
    keys=list(dict.fromkeys(k for r in rows for k in r));path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def collect():
    raw=[];flat=[];bir=[];thresholds=[];resource=[];status=defaultdict(lambda:defaultdict(int));classification=defaultdict(lambda:defaultdict(int))
    for directory in ['raw','2d','rollout','pysr','paired','short_validation','pysr_validation','convergence_validation','failures']:
        for path in input_glob(Path('evidence')/directory, '*.json'):
            r=json.loads(path.read_text());group=directory+'/'+r.get('config',{}).get('group',directory)
            status[group][r['status']]+=1
            classification[group][r.get('classification','unclassified')+'/'+r['status']]+=1
            resource.append(dict(id=r['id'],group=group,status=r['status'],seconds=r.get('seconds'),cpu_seconds=r.get('cpu_seconds'),peak_rss_mb=r.get('peak_rss_mb'),classification=r.get('classification')))
            if r['status']!='complete':continue
            if directory!='raw':continue
            raw.append(r);c=r['config'];base=dict(id=r['id'],group=c['group'],case=c['case'],setting=c['setting'],variant=c.get('variant','default'),seed=c['seed'])
            for epoch,e in r['epochs'].items():
                for name in ['neural','frozen']:flat.append(dict(**base,epoch=int(epoch),method=name,**e[name]))
                for key,m in e['compression'].items():bir.append(dict(**base,epoch=int(epoch),branch=key,**m))
            for name,control in r.get('controls',{}).items():flat.append(dict(**base,epoch=c['epochs'],method=name,**control['metrics']))
            for threshold in r.get('thresholds',[]):thresholds.append(dict(**base,**threshold))
    write_csv(ROOT/'evidence/metrics_raw.csv',flat);write_csv(ROOT/'evidence/bir_raw.csv',bir)
    write_csv(ROOT/'evidence/thresholds_raw.csv',thresholds);write_csv(ROOT/'evidence/resources_raw.csv',resource)
    json_write(ROOT/'evidence/status.json',dict(counts={k:dict(v) for k,v in status.items()},classifications={k:dict(v) for k,v in classification.items()},
        wall_seconds_task_sum=sum(r['seconds'] or 0 for r in resource),cpu_seconds_known_sum=sum(r['cpu_seconds'] or 0 for r in resource),
        note='Task wall times overlap. RSS is worker high-water mark; do not add across tasks. CPU accounting excludes Julia subprocesses unless independently recorded.'))
    groups=defaultdict(list)
    for row in flat:groups[tuple(row[k] for k in ['group','case','setting','variant','epoch','method'])].append(row)
    summaries=[]
    for key,rs in sorted(groups.items()):
        for metric in ['D','R','D_visited','R_visited','D_extrapolation','R_extrapolation','min_D']:
            summaries.append(dict(zip(['group','case','setting','variant','epoch','method'],key),metric=metric,**stats([r[metric] for r in rs])))
    write_csv(ROOT/'evidence/metrics_summary.csv',summaries)
    bgroups=defaultdict(list)
    for row in bir:bgroups[tuple(row[k] for k in ['group','case','setting','variant','epoch','branch'])].append(row)
    bs=[]
    for key,rs in sorted(bgroups.items()):
        for metric in ['BIR','bound','compression_relative_neural']:
            bs.append(dict(zip(['group','case','setting','variant','epoch','branch'],key),metric=metric,**stats([r[metric] for r in rs])))
    write_csv(ROOT/'evidence/bir_summary.csv',bs)
    return raw,flat,summaries,bs


def plots(raw,summaries,bs):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    dest=ROOT/'figures';dest.mkdir(exist_ok=True)
    for case in ['case_a','case_b','case_exp','case_trig','case_tanh','case_c']:
        rs=[r for r in raw if r['config']['case']==case and r['config']['seed']==0 and r['config']['group'] in ['main','families'] and r['config']['setting'] in ['clean','noise3']]
        if not rs:continue
        fig,axes=plt.subplots(len(rs),3,figsize=(13,3.6*len(rs)),squeeze=False)
        for i,r in enumerate(rs):
            q=load_curves(r['id']);g=q['grid']
            names=['true','weak_poly','strong_poly','neural','frozen','refit_weak','refit_data']
            for j,branch in enumerate(['D','R']):
                ax=axes[i,j]
                for name in names:
                    if name+'_'+branch in q:ax.plot(g,q[name+'_'+branch],label=name,ls='--' if name in ['frozen','refit_data'] else '-',lw=2 if name=='true' else 1.2)
                ax.set(xlabel='u',ylabel=branch+'(u)',title=case+' '+r['config']['setting']);ax.legend(fontsize=7)
            axes[i,2].bar((q['state_bins'][:-1]+q['state_bins'][1:])/2,q['state_hist'],width=np.diff(q['state_bins']),color='#9eb9cb')
            axes[i,2].set(xlabel='u',ylabel='Observed density',title='Visited-state distribution, seed 0')
        fig.tight_layout();fig.savefig(dest/f'{case}_closures.png');fig.savefig(dest/f'{case}_closures.pdf');plt.close(fig)
    # Main clean/noise comparison with explicitly counted seeds.
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,case in zip(axes,['case_a','case_b','case_exp']):
        for method in ['neural','weak_poly','strong_poly','refit_weak','refit_data','spacetime_poly']:
            levels={'clean':0,'noise1':1,'noise2':2,'noise3':3,'noise5':5}
            rows=[r for r in summaries if r['group'] in ['main','noise_knee'] and r['case']==case and r['method']==method and r['metric']=='R' and r['setting'] in levels]
            rows.sort(key=lambda r:levels[r['setting']])
            if rows:ax.errorbar([levels[r['setting']] for r in rows],[r['mean'] for r in rows],yerr=[r['sd'] or 0 for r in rows],marker='o',label=method,capsize=3)
        ax.set(xlabel='Noise (% of std(u))',ylabel='Relative reaction error',title=case,yscale='log');ax.legend(fontsize=7)
    fig.suptitle('Means ± sample SD; full seed counts in metrics_summary.csv');fig.tight_layout();fig.savefig(dest/'noise_controls.png');plt.close(fig)
    # BIR is shown with common-normalization bound, not treated as independent proof.
    rows=[r for r in bs if r['group'] in ['main','families'] and r['setting']=='clean' and r['metric']=='BIR']
    if rows:
        fig,ax=plt.subplots(figsize=(10,4))
        for i,r in enumerate(rows):ax.errorbar(i,r['mean'],yerr=r['sd'] or 0,fmt='o',capsize=3,label=None)
        ax.axhline(1,color='gray',ls='--');ax.set_xticks(range(len(rows)),[r['case']+' '+r['branch'] for r in rows],rotation=45,ha='right');ax.set(ylabel='BIR, mean ± sample SD',title='Frozen compression across matched and mismatch families')
        fig.tight_layout();fig.savefig(dest/'bir_families.png');plt.close(fig)
    # Fixed-seed convergence, all restart traces, no selected best true-error run.
    rs=[r for r in raw if r['config']['group']=='convergence']
    if rs:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for r in rs:
            h=load_history(r['id'])
            axes[0].semilogy([x['epoch'] for x in h],[x['loss'] for x in h],alpha=.5)
            for ax,branch in zip(axes[1:],['D','R']):ax.plot([int(e) for e in r['epochs']],[v['neural'][branch] for v in r['epochs'].values()],marker='o',alpha=.5)
        for ax,title in zip(axes,['Training objective','Diffusion error','Reaction error']):ax.set(xlabel='Epoch',title=title)
        fig.tight_layout();fig.savefig(dest/'convergence_restarts.png');plt.close(fig)


def main():
    raw,flat,summaries,bs=collect();plots(raw,summaries,bs)
    print('raw experiment records',len(raw),'metric rows',len(flat),'summaries',len(summaries))


if __name__=='__main__':main()
