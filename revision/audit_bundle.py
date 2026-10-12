"""Check portable inputs against their recorded source and declared campaign."""
import argparse
import csv
from revision_core import *
from run_suite import task_configs, task_id


def full_campaign():
    tasks=task_configs('all')
    required={'raw':{task_id(t):t for t in tasks},'2d':{},'rollout':{},
              'short_validation':{},'paired':{},'pysr':{},'convergence_validation':{}}
    for task in tasks:
        ident=task_id(task);group=task['group']
        if group in ('main','families','support','coarse'):required['rollout'][ident]=task
        if group=='main':
            required['short_validation'][ident]=task
            if task['case']=='case_exp' and task['setting'] in ('clean','noise3','noise5'):required['paired'][ident]=task
            if task['setting'] in ('clean','noise3'):required['pysr'][ident]=task
        if group=='convergence':required['convergence_validation'][ident]=task
    for case,noise in [('case_exp',0.),('case_exp',.03),('case_trig',0.),('case_trig',.03),('case_a',0.)]:
        for seed in range(5):
            config=dict(case=case,noise=noise,seed=seed,epochs=150,nx=40,ny=40,t_final=.1,
                        save_every=20,num_trajectories=6,num_initial_modes=4,num_test_modes=2)
            required['2d'][f'2d_{case}_{noise:g}_{seed}_e150']=config
    for directory,expected in required.items():
        actual={}
        for path in input_glob(Path('evidence')/directory):
            row=json.loads(path.read_text());actual[row['id']]=row
        for ident,config in expected.items():
            row=actual.get(ident)
            if row is None or row.get('status')!='complete' or row.get('config')!=config:
                raise ValueError(f'Incomplete campaign record: {directory}/{ident}')
    return {key:len(value) for key,value in required.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--require-full',action='store_true');a=p.parse_args()
    missing=[];checked=0;max_difference=0.;diagnostics=[]
    identity_path=input_path('evidence/source_identity.json')
    identity=json.loads(identity_path.read_text()) if identity_path.exists() else None
    if identity is not None:
        recorded_root=identity_path.parents[1]
        for item in identity['source_files']:
            path=recorded_root/item['path']
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError(f'Recorded input source changed or missing: {item["path"]}')
    for directory in ['raw','2d','pysr']:
        for f in input_glob(Path('evidence')/directory):
            r=json.loads(f.read_text())
            if r['status']!='complete':continue
            ident=r['id'];required=[]
            if directory=='raw':
                epochs=set(map(int,r.get('epochs',{})));expected=set(r['config'].get('checkpoints',[]))|{r['config']['epochs']}
                if not expected.issubset(epochs):raise ValueError(f'Incomplete final epoch coverage: {ident}')
                required += [Path('evidence/surrogate_curves')/f'{ident}_e{e}.csv' for e in r['epochs']]
                required += [Path('evidence/histories')/f'{ident}.csv']
                if r.get('controls'):required += [Path('evidence/curves')/f'{ident}.csv',Path('evidence/state_histograms')/f'{ident}.csv']
            elif directory=='2d':
                required += [Path('evidence/curves')/f'{ident}.csv',Path('evidence/histories')/f'{ident}.csv',Path('evidence/state_histograms')/f'{ident}.csv']
            else:required += [Path('evidence/curves')/f'pysr_{ident}.csv']
            for rel in required:
                if not input_path(rel).exists():missing.append(str(rel))
                else:checked+=1
            invalid_nyquist=(directory=='raw' and r['config']['group']=='coarse' and
                r['config']['setting']=='coarse8' and r.get('test_functions',{}).get('effective_modes',4)>3)
            if invalid_nyquist:diagnostics.append(ident)
            elif directory in ('raw','2d') and input_path(Path('evidence/histories')/f'{ident}.csv').exists():
                validate_history(load_history(ident),r['config']['epochs'])
            if directory=='raw':
                case=extra_case(r['config']['case']);g=np.linspace(*case.value_range,256)
                for epoch,record in r['epochs'].items():
                    path=input_path(Path('evidence/surrogate_curves')/f'{ident}_e{epoch}.csv')
                    if not path.exists():continue
                    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
                    x=np.array([float(v['u']) for v in rows])
                    if len(x)<2 or not np.isfinite(x).all() or not (np.diff(x)>0).all():
                        raise ValueError(f'Invalid interpolation grid: {ident}/{epoch}')
                    for branch,truth in [('D',case.diffusion(g)),('R',case.reaction(g))]:
                        y=np.interp(g,x,np.array([float(v[branch]) for v in rows]))
                        actual=relative_l2_error(truth,y);expected=record['neural'][branch]
                        max_difference=max(max_difference,abs(actual-expected))
                        if not np.isclose(actual,expected,rtol=1e-12,atol=1e-14):
                            raise ValueError(f'Portable CSV metric mismatch: {ident} epoch {epoch} {branch}')
    coverage=full_campaign() if a.require_full else None
    result=dict(status='passed' if not missing else 'failed',classification='delivery_integrity_audit',
        checked_inputs=checked,missing=missing,max_curve_metric_difference=max_difference,
        input_source_commit=identity.get('source_commit') if identity else None,current_source_signature=source_signature(),
        historical_invalid_nyquist_ids=diagnostics,full_campaign_required=a.require_full,campaign_coverage=coverage)
    json_write(ROOT/'evidence/bundle_audit.json',result)
    print(result['status'],'checked_inputs',checked,'missing',len(missing),'historical_diagnostics',len(diagnostics),'max_curve_metric_difference',max_difference)
    if missing:raise SystemExit(1)


if __name__=='__main__':main()
