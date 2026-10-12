"""Map actual experiment records to the explicitly requested reviewer additions."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'revision'))
from project_paths import ROOT,INPUT_ROOT
from artifact_io import json_write
from build_report import write_csv


def load(directory):
    result=[]
    for path in (INPUT_ROOT/'evidence'/directory).glob('*.json'):
        row=json.loads(path.read_text())
        if row['status']=='complete':result.append(row)
    return result


def main():
    raw=load('raw');groups={}
    invalid=[]
    for row in raw:
        c=row['config']
        if c['group']=='coarse' and c['setting']=='coarse8':invalid.append(row['id']);continue
        groups.setdefault(c['group'],[]).append(row)
    rows=[]
    def item(key,reviewers,description,records,expected,files):
        count=len(records)
        rows.append(dict(experiment=key,review_items=reviewers,description=description,completed=count,expected=expected,
            status='available' if count==expected else 'incomplete',evidence=files,
            independent_data_seeds=5 if count else 0))
    item('main_five_seeds','R2.4 R4.5 R5.M2 A.2 A.7','A/B/Exp; clean/1%/3%/5%/sparse; polynomial and selected-family equation-refit controls',groups.get('main',[]),75,'evidence/raw/main_*; evidence/metrics_summary.csv')
    item('loss_ablation','R2.3 R3.3 R3.10 R4.2 A.3','23 removal/cumulative/weight variants; clean and 3% noise; five data seeds',groups.get('ablation',[]),230,'evidence/raw/ablation_*')
    item('convergence_restarts','R5.M2','Five datasets times three initializations; all 150/500/1500-epoch checkpoints',groups.get('convergence',[]),15,'evidence/raw/convergence_*')
    item('convergence_validation','R5.M2 R5.M4','Validate every checkpoint/restart; uncertainty averages within each data seed',load('convergence_validation'),15,'evidence/convergence_validation')
    item('additional_mismatch_families','R2.2 A.4','Rational/saturation, trigonometric and tanh cases; clean and 3% noise',groups.get('families',[]),30,'evidence/raw/families_*')
    two=[r for r in load('2d') if r['classification']!='runtime_pilot']
    item('stronger_2d','R2.2 R3.6 R4.3 A.10','40x40 training with six trajectories to T=.1; 32x32 validation to T=.3',two,25,'evidence/2d')
    item('coarse_nx16','R3.11','nx16 coarse observation grid',groups.get('coarse',[]),5,'evidence/raw/coarse_*coarse16*')
    item('coarse_resolved_nx8','R3.11','nx8 with three resolved Fourier modes; original invalid nx8 excluded',groups.get('coarse_corrected',[]),5,'evidence/raw/coarse_corrected_*')
    item('time_weak_alternative','R3.1 R5.M3','Reduced spatial versus full time-weak loss; clean/1%/3%/5% noise',groups.get('weak',[]),40,'evidence/raw/weak_*')
    item('quadrature_comparison','R3.1 R5.M3','Recorded rectangle/trapezoidal comparisons; use corrected trapezoidal results for the alternative',groups.get('paired',[]),30,'evidence/raw/paired_*')
    item('noise_knee_2percent','R3.7 R5.m4','Five-seed 2% noise extension between the 1% and 3% settings',groups.get('noise_knee',[]),5,'evidence/raw/noise_knee_*')
    item('narrow_support','R5.M4','Narrow initial-state range; visited and extrapolation closure errors',groups.get('support',[]),10,'evidence/raw/support_*')
    item('pysr_cases_noise','R3.9 A.6','A/B/Exp times clean/3% noise times five seeds; original fitted expressions retained',load('pysr'),30,'evidence/pysr')
    item('selected_family_forward_refit','R5.M1','Observation-only forward PDE refit for all five main Exp conditions',
        [r for r in load('rollout') if r['config']['group']=='main' and 'forward_refit' in r],25,'evidence/rollout/main_case_exp_*')
    item('long_varied_ic_validation','R5.M4','Longer horizons and preset initial-state regimes; invalid coarse experiments excluded',
        [r for r in load('rollout') if r['id'] not in invalid],120,'evidence/rollout')
    item('short_validation','R4.5 R5.M4','Original all-time short-trajectory norm; clipped and strict validation policies',load('short_validation'),75,'evidence/short_validation')
    item('clipping_pairs','R5.M3','15 paired seed/setting clipping comparisons',load('paired'),15,'evidence/paired')
    additional=[];failures=[]
    for path in (ROOT/'pysr-observation-refit/evidence/pysr_observation_refit').glob('*.json'):
        row=json.loads(path.read_text())
        (additional if row['status']=='complete' else failures).append(row)
    item('pysr_observation_refit','R5.M1','Additional fixed-expression floating-coefficient refits; local reconstructed observations retained',additional,30,'pysr-observation-refit/evidence/pysr_observation_refit')
    continued=[]
    for path in (ROOT/'pysr-observation-refit-120/evidence/pysr_observation_refit').glob('*.json'):
        record=json.loads(path.read_text())
        if record['status']=='complete':continued.append(record)
        else:failures.append(record)
    item('pysr_refit_budget_extension','R5.M1 R5.M2','Same 30 inputs; continue optimizer-budget exhaustion with up to 80 further evaluations',continued,30,'pysr-observation-refit-120/evidence/pysr_observation_refit')
    threshold_count=sum(len(r.get('thresholds',[])) for rs in groups.values() for r in rs)
    write_csv(ROOT/'experiment_inventory.csv',rows)
    result=dict(status='complete' if all(r['status']=='available' for r in rows) else 'partial',experiments=rows,
        threshold_sensitivity_records=threshold_count,historical_invalid_nyquist_ids=invalid,
        new_refit_failures=[dict(id=r['id'],error=r.get('error')) for r in failures],
        new_refit_optimizer_converged=sum(r['fit']['success'] for r in additional),
        new_refit_budget_exhausted=sum(r['fit']['status']==0 and r['fit']['parameter_count']>0 for r in additional),
        final_refit_optimizer_converged=sum(r['fit']['success'] for r in continued),
        final_refit_budget_exhausted=sum(r['fit']['status']==0 and r['fit']['parameter_count']>0 for r in continued),
        archive_root=str(INPUT_ROOT),work_root=str(ROOT),
        note='Available experiment evidence is not a completed manuscript revision or theoretical argument. Representative CPU/GPU verification is not counted as additional independent seeds.')
    json_write(ROOT/'experiment_inventory.json',result)
    lines=['# Reviewer experiment status','',result['note'],'',
        '| Experiment | Review items | Completed | Status |','| --- | --- | --- | --- |']
    lines += [f"| {r['experiment']} | {r['review_items']} | {r['completed']}/{r['expected']} | {r['status']} |" for r in rows]
    lines += ['',f'Initial 40-evaluation PySR fits: optimizer converged in {result["new_refit_optimizer_converged"]}; budget exhausted in {result["new_refit_budget_exhausted"]}.',
        f'After the continuation allocation: optimizer converged in {result["final_refit_optimizer_converged"]}; budget exhausted in {result["final_refit_budget_exhausted"]}. The same 30 inputs are reused; no independent seeds are added.',
        f'Tolerance-sensitivity records: {threshold_count}. Five historical invalid nx8 runs remain diagnostic inputs.',
        '','New refits save their exact locally reconstructed observations. Archive observation byte hashes are recorded separately; bitwise reproduction of missing original arrays is not claimed.']
    (ROOT/'EXPERIMENT_STATUS.md').write_text('\n'.join(lines)+'\n')
    print(result['status'],'archived evidence groups',len(rows)-2,'new PySR refits',len(additional),'/30','extended records',len(continued),'/30','new failures',len(failures))


if __name__=='__main__':main()
