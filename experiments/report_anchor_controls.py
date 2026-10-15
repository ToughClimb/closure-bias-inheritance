"""Summarize the paired boundary-prior experiment from immutable fit records."""
from pathlib import Path
import argparse,csv,json,sys,hashlib,copy
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def dump(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def model_diagnostics(root):
    records=[]
    for path in sorted((root/'fits').glob('*.json')):
        row=json.loads(path.read_text());assert row['status']=='complete',path
        symbol=row['model']['symbols'][1];coeff=np.array(symbol['parameters'])
        if symbol['family']=='rational':numerator=coeff[:3];denominator=np.r_[1,coeff[3:]]
        elif symbol['family']=='polynomial':numerator=coeff;denominator=np.array([1.])
        else:raise ValueError(symbol['family'])
        roots=np.polynomial.polynomial.polyroots(numerator)
        positive=[float(x.real) for x in roots if abs(x.imag)<1e-10 and 0<x.real<=1.8]
        assert positive==row['positive_reaction_roots'],path
        def derivative(x):
            n=np.polynomial.polynomial.polyval(x,numerator);d=np.polynomial.polynomial.polyval(x,denominator)
            np_=np.polynomial.polynomial.polyval(x,np.polynomial.polynomial.polyder(numerator))
            dp=np.polynomial.polynomial.polyval(x,np.polynomial.polynomial.polyder(denominator))
            return float((np_*d-n*dp)/d**2)
        dataset=row['identity']['dataset'];fit=row['fit']
        model=copy.deepcopy(row['model'])
        for item in model['symbols']:
            item['source_teacher_fit_mse']=item.pop('fit_mse')
        records.append(dict(id=dataset['id'],setting=dataset['setting'],seed=dataset['seed'],arm=fit['arm'],
            fit_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            observation_file_sha256=dataset['observation_file_sha256'],model=model,metrics=row['metrics'],
            R0=fit['R0'],R_prime_zero=row['R_prime_zero'],R_at_001=row['R_at_001'],
            positive_reaction_roots=positive,root_derivatives=[derivative(x) for x in positive],
            reaction_poles_all_real=[float(x.real) for x in np.polynomial.polynomial.polyroots(denominator) if abs(x.imag)<1e-10],
            scalar=row['scalar'],fit_components=fit['components'],optimizer_success=fit['optimizer_success']))
    assert len(records)==30,len(records)
    return records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work-root',type=Path,required=True);args=ap.parse_args()
    root=args.work_root.resolve();rows=[]
    for p in sorted((root/'fits').glob('*.json')):
        r=json.loads(p.read_text());assert r['status']=='complete',p
        d=r['identity']['dataset'];f=r['fit']
        rows.append(dict(id=d['id'],setting=d['setting'],seed=d['seed'],arm=f['arm'],D=r['metrics']['D'],R=r['metrics']['R'],
            observation_mse=f['components']['observation_mse'],diffusion_penalty=f['components']['diffusion_penalty'],anchor_penalty=f['components']['anchor_penalty'],
            R0=f['R0'],R_prime_zero=r['R_prime_zero'],R_at_001=r['R_at_001'],
            positive_initial_zero_time=r['scalar']['0.01']['zero_time'],scalar_zero_success=r['scalar']['0.0']['success'],
            scalar_zero_final=r['scalar']['0.0'].get('final_state'),positive_initial_success=r['scalar']['0.01']['success'],
            positive_initial_final=r['scalar']['0.01'].get('final_state'),optimizer_success=f['optimizer_success'],nfev=f['nfev'],parameter_count=f['parameter_count']))
    assert len(rows)==30,len(rows)
    out=root/'reports';out.mkdir(exist_ok=True)
    dump(out/'by_seed.csv',rows)
    summaries=[];paired=[]
    for setting in ['clean','noise3']:
        for arm in ['unanchored','soft_anchor','hard_anchor']:
            rs=sorted([r for r in rows if r['setting']==setting and r['arm']==arm],key=lambda r:r['seed']);assert len(rs)==5
            result=dict(setting=setting,arm=arm,n=5)
            for key in ['D','R','observation_mse','R0','R_prime_zero','R_at_001']:
                v=np.array([r[key] for r in rs]);result[key+'_mean']=float(v.mean());result[key+'_sd']=float(v.std(ddof=1))
            result.update(positive_growth_at_001=sum(r['R_at_001']>0 for r in rs),positive_slope_zero=sum(r['R_prime_zero']>0 for r in rs),
                negative_R0=sum(r['R0']<0 for r in rs),positive_initial_reaches_zero=sum(r['positive_initial_zero_time'] is not None for r in rs),
                zero_scalar_failure=sum(not r['scalar_zero_success'] for r in rs),optimizer_success_count=sum(r['optimizer_success'] for r in rs))
            summaries.append(result)
        base=sorted([r for r in rows if r['setting']==setting and r['arm']=='unanchored'],key=lambda r:r['seed'])
        for arm in ['soft_anchor','hard_anchor']:
            other=sorted([r for r in rows if r['setting']==setting and r['arm']==arm],key=lambda r:r['seed'])
            for metric in ['D','R','observation_mse']:
                v=np.array([a[metric]-b[metric] for a,b in zip(other,base,strict=True)]);sd=float(v.std(ddof=1));half=float(t.ppf(.975,4)*sd/np.sqrt(5))
                paired.append(dict(setting=setting,arm=arm,reference='unanchored_common_initial',metric=metric,n=5,mean_difference=float(v.mean()),sample_sd=sd,
                    ci95_low=float(v.mean()-half),ci95_high=float(v.mean()+half),by_seed=json.dumps(v.tolist()),rule='Exploratory paired t mean interval; no multiple-comparison correction'))
    dump(out/'summary.csv',summaries);dump(out/'paired.csv',paired)
    validations=[]
    for p in sorted((root/'validation').glob('*.json')):
        r=json.loads(p.read_text());assert r['status']=='complete',p
        fit=json.loads((root/'fits'/p.name).read_text());setting=fit['identity']['dataset']['setting'];seed=fit['identity']['dataset']['seed'];arm=fit['fit']['arm']
        for pattern,v in r['models'][arm]['validation'].items():
            info=v['info'];validations.append(dict(setting=setting,seed=seed,arm=arm,pattern=pattern,
                complete=info['numerical_completion'],accepted_range_valid=info['accepted_state_range_valid'],rk_range_valid=info['rk_stage_state_range_valid'],
                nonnegative=v['nonnegative_accepted_states'],final_relative_error=v['final_relative_l2'],initial_trajectories=v['initial_trajectories']))
    assert len(validations)==240,len(validations)
    dump(out/'forward_by_seed.csv',validations)
    (out/'model_diagnostics.json').write_text(json.dumps(model_diagnostics(root),indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    for ax,key,title in zip(axes,['R','observation_mse','R_at_001'],['Relative reaction error','Observation mean squared error','Reaction at u = 0.01']):
        for i,arm in enumerate(['unanchored','soft_anchor','hard_anchor']):
            rs=[r for r in rows if r['setting']=='noise3' and r['arm']==arm]
            ax.scatter([i]*5,[r[key] for r in rs],s=25,alpha=.75)
        ax.set(title=title,xticks=range(3),xticklabels=['No anchor','Soft','Hard'])
        if key=='R_at_001':ax.axhline(.01*(1.1*np.exp(-1.4*.01)-.22),color='black',ls='--',label='Truth');ax.axhline(0,color='gray',lw=.7);ax.legend()
        if key=='observation_mse':ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.savefig(out/'anchor_controls.png',dpi=200);plt.close(fig)
    summary=dict(fits=30,validation_model_pattern_batches=len(validations),summary=summaries,all_optimizer_local_success=all(r['optimizer_success'] for r in rows))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Reported 30 fits and',len(validations),'validated batches')
if __name__=='__main__':main()
