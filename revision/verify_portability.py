"""Compare a clean representative rerun with delivered reviewer-suite records."""
import argparse
import platform
from revision_core import *


def verify(args):
    ident='main_case_exp_clean_0_default'
    a=json.loads((args.reference_root/'evidence/raw'/f'{ident}.json').read_text())
    b=json.loads((args.rerun_root/'evidence/raw'/f'{ident}.json').read_text())
    if a['status']!='complete' or b['status']!='complete':raise ValueError('Both representative runs must be complete')
    if a['config']!=b['config']:raise ValueError('Representative configurations do not match')
    if set(a['controls'])!=set(b['controls']):raise ValueError('Representative control methods do not match')
    checks=[]
    for method in ['neural','frozen']+list(a['controls']):
        x=a['epochs']['150'][method] if method in ['neural','frozen'] else a['controls'][method]['metrics']
        y=b['epochs']['150'][method] if method in ['neural','frozen'] else b['controls'][method]['metrics']
        for key in ['D','R','D_visited','R_visited','min_D']:
            checks.append(dict(method=method,metric=key,reference=x[key],rerun=y[key],
                               absolute_difference=abs(x[key]-y[key]),passed=bool(np.isclose(x[key],y[key],rtol=1e-5,atol=1e-8))))
    for branch in ['D','R']:
        x=a['epochs']['150']['compression'][branch]['BIR'];y=b['epochs']['150']['compression'][branch]['BIR']
        checks.append(dict(method='frozen',metric='BIR_'+branch,reference=x,rerun=y,absolute_difference=abs(x-y),passed=bool(np.isclose(x,y,rtol=1e-5,atol=1e-8))))
    validation_checks=[]
    for directory in ['rollout','short_validation']:
        ref=args.reference_root/'evidence'/directory/f'{ident}.json';new=args.rerun_root/'evidence'/directory/f'{ident}.json'
        if not ref.exists() or not new.exists():
            raise FileNotFoundError(f'Representative {directory} input is missing; execute the documented validation command first')
        x=json.loads(ref.read_text());y=json.loads(new.read_text())
        if x['status']!='complete' or y['status']!='complete':raise ValueError(f'{directory} validation is incomplete')
        if set(x['methods'])!=set(y['methods']):raise ValueError(f'{directory} method coverage mismatch')
        if directory=='rollout':
            for method,entries in x['methods'].items():
                if not entries or len(entries)!=len(y['methods'][method]):
                    raise ValueError(f'Rollout entry coverage mismatch for {method}')
                for old,new in zip(entries,y['methods'][method],strict=True):
                    same=old['pattern']==new['pattern'] and old['info']['status']==new['info']['status']
                    errors=np.asarray(old['errors']);rerun=np.asarray(new['errors'])
                    times=np.asarray(old['times']);rerun_times=np.asarray(new['times'])
                    if errors.shape!=rerun.shape or times.shape!=rerun_times.shape or times.shape!=errors.shape:
                        raise ValueError(f'Rollout series coverage mismatch for {method}/{old["pattern"]}')
                    same=same and np.allclose(times,rerun_times,rtol=0,atol=1e-12)
                    validation_checks.append(dict(stage=directory,method=method,pattern=old['pattern'],
                        passed=bool(same and np.allclose(errors,rerun,rtol=1e-5,atol=1e-8)),
                        max_absolute_difference=float(np.max(np.abs(errors-rerun)))))
            for branch in ['D','R']:
                old=x['forward_refit']['metrics'][branch];new=y['forward_refit']['metrics'][branch]
                validation_checks.append(dict(stage='forward_refit',method=branch,passed=bool(np.isclose(old,new,rtol=1e-5,atol=1e-8)),max_absolute_difference=abs(old-new)))
        else:
            for method,policies in x['methods'].items():
                if set(policies)!=set(y['methods'][method]):raise ValueError(f'Short-validation policy coverage mismatch for {method}')
                for policy,old in policies.items():
                    new=y['methods'][method][policy]
                    validation_checks.append(dict(stage=directory,method=method,policy=policy,
                        passed=old['info']['status']==new['info']['status'] and bool(np.isclose(old['trajectory_error'],new['trajectory_error'],rtol=1e-5,atol=1e-8)),
                        max_absolute_difference=abs(old['trajectory_error']-new['trajectory_error'])))
    if len(checks)!=37 or len(validation_checks)!=56:
        raise ValueError(f'Incomplete representative coverage: {len(checks)} metric checks and {len(validation_checks)} validation checks; expected 37 and 56')
    result=dict(status='passed' if all(c['passed'] for c in checks+validation_checks) and a['data_hash']==b['data_hash'] else 'failed',
        classification='representative_clean_install_reproduction_only',system=platform.system(),machine=platform.machine(),
        python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device=b.get('device','cpu'),
        identical_observation_hash=a['data_hash']==b['data_hash'],observation_hash=b['data_hash'],
        tolerance=dict(relative=1e-5,absolute=1e-8),max_absolute_difference=max(c['absolute_difference'] for c in checks),checks=checks,
        validation_checks=validation_checks,max_validation_absolute_difference=max(c['max_absolute_difference'] for c in validation_checks),
        scope='One full-budget clean 1D Case Exp seed 0 with neural/symbolic/polynomial/equation-refit stages; not the complete reviewer sweep; no Windows/macOS/GPU numerical claim')
    result['metric_check_count']=len(checks);result['validation_check_count']=len(validation_checks)
    json_write(ROOT/'evidence/portability_reproduction.json',result)
    print(result['status'],'metric_checks',len(checks),'validation_checks',len(validation_checks),'max_absolute_difference',result['max_absolute_difference'],'data_hash_match',result['identical_observation_hash'])
    if result['status']!='passed':raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rerun-root',type=Path,required=True)
    parser.add_argument('--reference-root',type=Path,default=INPUT_ROOT)
    args=parser.parse_args()
    try:verify(args)
    except Exception as error:
        json_write(ROOT/'evidence/portability_reproduction.json',dict(status='failed',error=str(error),classification='representative_reproduction'))
        raise


if __name__=='__main__':main()
