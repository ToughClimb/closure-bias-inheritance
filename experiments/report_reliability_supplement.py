"""Rebuild the reviewer supplement from completed, authenticated local records.

No training, coefficient fitting, observation regeneration or manuscript edits.
Reports preserve numerical failures and physical violations as separate outcomes.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'revision'))

import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from artifact_io import ROOT, json_write
from revision_core import extra_case, curve_metrics
from run_reliability_supplement import load_manifest, observations, digest
from evaluate_reliability_supplement import evaluation_identity, functions


LABELS = {
    'weak_poly': 'Space-weak polynomial',
    'strong_poly': 'Strong-residual polynomial',
    'spacetime_poly': 'Space-time-weak polynomial',
    'neural_150': 'Neural, Adam 150',
    'frozen_150': 'Frozen family, Adam 150 teacher',
    'refit_150': 'Observation refit, Adam 150 family',
    'neural_adam_final': 'Neural, final Adam budget',
    'frozen_adam_final': 'Frozen family, final Adam teacher',
    'neural_lbfgs_float64': 'Neural, float64 L-BFGS crosscheck',
    'frozen_lbfgs_float64': 'Frozen family, L-BFGS teacher',
}
COLORS = dict(neural_150='#e69f00', frozen_150='#d55e00', refit_150='#009e73',
              weak_poly='#56b4e9', neural_adam_final='#0072b2')


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(name, rows):
    if not rows:
        raise ValueError(f'Empty report table: {name}')
    keys = list(dict.fromkeys(key for row in rows for key in row))
    path = ROOT / 'reports' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def moments(values):
    values = np.array([v for v in values if v is not None], dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite reported statistic')
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                sample_sd=float(values.std(ddof=1)) if len(values) > 1 else None)


def paired(values):
    a = np.asarray(values, dtype=float)
    if a.shape != (5,) or not np.isfinite(a).all():
        raise ValueError('Paired statistics require all five predeclared seeds')
    mean = float(a.mean()); sd = float(a.std(ddof=1)); se = sd / np.sqrt(len(a))
    radius = float(student_t.ppf(.975, len(a)-1)) * se
    statistic = mean/se if se else None
    return dict(n=len(a), mean_difference=mean, sample_sd_difference=sd,
                ci95_lower=mean-radius, ci95_upper=mean+radius,
                t_statistic=statistic,
                two_sided_p=float(2*student_t.sf(abs(statistic), len(a)-1)) if statistic is not None else None,
                difference_by_seed=a.tolist(),
                interpretation='Exploratory paired t interval; n=5, no multiple-comparison correction')


def complete_records(folder, expected):
    paths = sorted((ROOT/'evidence'/folder).glob('*.json'))
    if len(paths) != expected:
        raise ValueError(f'{folder}: expected {expected}, found {len(paths)}')
    rows = [read(path) for path in paths]
    if any(row['status'] != 'complete' for row in rows):
        raise ValueError(f'{folder}: incomplete task records; inspect raw failures')
    return {row['id']: row for row in rows}


def authenticate(manifest, training, baselines, matched, archive, refits):
    identity = evaluation_identity()
    if read(ROOT/'evidence/EVALUATION_PROTOCOL.json') != identity:
        raise ValueError('Evaluation implementation changed after declaration')
    checks = []
    for dataset in manifest['datasets']:
        ident = dataset['id']; observed = observations(dataset)
        train, base, audit, refit = (group[ident] for group in (training, baselines, matched, refits))
        if train['identity']['dataset'] != dataset or base['identity']['dataset'] != dataset:
            raise ValueError('Matched training/baseline dataset lineage differs')
        for key, value in identity.items():
            if audit['identity'][key] != value or refit['identity'][key] != value:
                raise ValueError('Matched evaluation protocol differs')
        for folder, key in [('training', 'training_sha256'), ('baselines', 'baseline_sha256')]:
            if audit['identity'][key] != digest(ROOT/'evidence'/folder/f'{ident}.json'):
                raise ValueError('Matched source record changed')
        if audit['fresh_refit_record_sha256'] != digest(ROOT/'evidence/fresh_refits'/f'{ident}.json'):
            raise ValueError('Matched refit record changed')
        if set(audit['models']) != set(LABELS):
            raise ValueError('Matched comparison is missing a predeclared method')
        teacher = train['stages']['neural_adam_150']
        if refit['identity']['teacher'] != teacher:
            raise ValueError('Refit does not use the fresh 150-epoch teacher')
        if refit['model']['teacher_native_sha256'] != teacher['native_file_sha256']:
            raise ValueError('Fresh teacher weights differ')
        case = extra_case('case_exp'); grid = np.linspace(*case.value_range, 256)
        initial_hashes = {}
        for label, model in audit['models'].items():
            descriptor = model['descriptor']
            if descriptor['observation_file_sha256'] != dataset['observation_file_sha256']:
                raise ValueError('A method uses different saved observations')
            actual = curve_metrics(case, grid, *functions(descriptor), observed)
            for field, value in actual.items():
                expected = audit['closure_metrics'][label][field]
                if value is None:
                    if expected is not None: raise ValueError('Metric null mismatch')
                elif not np.isclose(value, expected, rtol=2e-12, atol=1e-14):
                    raise ValueError('Stored closure metric differs from native weights/coefficients')
            for pattern, result in model['validation'].items():
                signature = result['initial_condition_sha256']
                if pattern in initial_hashes and initial_hashes[pattern] != signature:
                    raise ValueError('Forward comparison initial conditions differ')
                initial_hashes[pattern] = signature
                if result['info']['state_clip'] is not None or result['info']['clip_steps']:
                    raise ValueError('A validation trajectory was clipped')
                if pattern == 'uniform_zero' and any(v is not None for v in result['relative_l2']):
                    raise ValueError('Relative error was assigned to a zero reference')
                if not result['info']['numerical_completion'] and result['final_absolute_rms'] is not None:
                    raise ValueError('A failed trajectory has a completed-horizon error')
        bounds = 0
        for stage in train['stages'].values():
            for branch in ('D', 'R'):
                geometry = stage['compression'][branch]; rho, bir = geometry['rho'], geometry['BIR']
                if not max(0., 1-rho)-1e-10 <= bir <= 1+rho+1e-10:
                    raise ValueError('Common-norm triangle bound failed')
                bounds += 1
        checks.append(dict(id=ident, observation_file_sha256=dataset['observation_file_sha256'],
            observation_u_sha256=dataset['observation_u_sha256'], methods=len(audit['models']),
            identical_initial_conditions=True, native_metrics_recomputed=True,
            geometry_bounds_checked=bounds))
    archived_directory = Path(manifest['datasets'][0]['archived_refit_file']).parent
    for row in archive.values():
        archived_path = ROOT/'evidence/archive_inputs'/f"{row['id']}.json"
        if not archived_path.exists(): archived_path = archived_directory/f"{row['id']}.json"
        if digest(archived_path) != row['identity']['archived_record_sha256']:
            raise ValueError('Archived PySR input record changed')
        if len(row['models']) != 2:
            raise ValueError('Archived audit omitted a model')
        for key, value in identity.items():
            if row['identity'][key] != value:
                raise ValueError('Archived audit protocol differs')
        hashes = {}
        for model in row['models'].values():
            for pattern, result in model['validation'].items():
                if result['info']['state_clip'] is not None:
                    raise ValueError('Archived validation was clipped')
                signature = result['initial_condition_sha256']
                if pattern in hashes and hashes[pattern] != signature:
                    raise ValueError('Archived forward initial conditions differ')
                hashes[pattern] = signature
    return checks


def uniform_annotations(uniform):
    annotations = {}
    for name, status in [('checks', 'independent_agreement'),
                         ('failed_forward_patterns_retained', 'not_checked_rk4_guard_failed'),
                         ('independent_model_failures', 'independent_solver_failure_or_disagreement')]:
        for row in uniform[name]:
            key = tuple(row[k] for k in ('cohort', 'id', 'method', 'pattern'))
            if key in annotations: raise ValueError('Duplicate uniform audit outcome')
            annotations[key] = status
    if len(annotations) != 640: raise ValueError('Uniform audit omitted a model-pattern outcome')
    return annotations


def tabulate(training, matched, archive, refits, annotations):
    metrics, budgets, geometry, forwards, endpoint_rows, costs = [], [], [], [], [], []
    for ident, row in matched.items():
        setting, seed = row['config']['setting'], row['config']['seed']
        for label, values in row['closure_metrics'].items():
            metrics.append(dict(id=ident, setting=setting, seed=seed, method=label, **values))
    for ident, row in training.items():
        setting = row['identity']['dataset']['setting']; seed = row['identity']['dataset']['seed']
        budgets.append(dict(id=ident, setting=setting, seed=seed, extended=row['extended'],
            final_adam_epoch=row['final_adam_epoch'], adam_budget_state=row['adam_budget_state'],
            initial_tail_decline=row['initial_budget_tail']['relative_decline'],
            final_tail_decline=row['final_adam_tail']['relative_decline'],
            adam_post_step_loss=row['final_adam_objective']['loss'],
            adam_post_step_gradient_norm=row['final_adam_objective']['gradient_norm'],
            adam_post_step_gradient_max=row['final_adam_objective']['gradient_max'],
            lbfgs_stopping=row['lbfgs']['stopping'], lbfgs_outer_steps=row['lbfgs']['outer_steps'],
            lbfgs_closure_calls=row['lbfgs']['closure_calls'],
            lbfgs_initial_loss=row['lbfgs']['initial']['loss'], lbfgs_final_loss=row['lbfgs']['final']['loss'],
            lbfgs_final_gradient_max=row['lbfgs']['final']['gradient_max'],
            lbfgs_relative_parameter_change=row['lbfgs']['relative_parameter_change'],
            total_task_seconds=row['seconds']))
        for stage_label, stage in row['stages'].items():
            for branch in ('D', 'R'):
                expression = stage['symbols'][0 if branch == 'D' else 1]
                geometry.append(dict(id=ident, setting=setting, seed=seed, stage=stage_label, branch=branch,
                    neural_relative_error=stage['metrics'][branch], frozen_relative_error=stage['frozen_metrics'][branch],
                    family=expression['family'], complexity=expression['complexity'], **stage['compression'][branch]))
    for cohort, records in [('fresh_same_observation', matched), ('archived_pysr', archive)]:
        for ident, row in records.items():
            for label, model in row['models'].items():
                endpoint_rows.append(dict(cohort=cohort, id=ident, **row['config'], method=label, **model['endpoints']))
                for pattern, result in model['validation'].items():
                    info = result['info']
                    key = ('fresh' if cohort == 'fresh_same_observation' else 'archived', ident, label, pattern)
                    independent = annotations[key] if pattern.startswith('uniform_') else 'not_applicable'
                    forwards.append(dict(cohort=cohort, id=ident, case=row['config']['case'],
                        setting=row['config']['setting'], seed=row['config']['seed'], method=label, pattern=pattern,
                        numerical_completion=info['numerical_completion'], numerical_reason=info['reason'],
                        accepted_state_range_valid=info['accepted_state_range_valid'],
                        rk_stage_state_range_valid=info['rk_stage_state_range_valid'],
                        nonnegative_accepted_states=result['nonnegative_accepted_states'],
                        accepted_state_min=info['accepted_state_min'], accepted_state_max=info['accepted_state_max'],
                        rk_stage_state_min=info['rk_stage_state_min'], rk_stage_state_max=info['rk_stage_state_max'],
                        first_accepted_range_violation=info['first_accepted_range_violation'],
                        first_rk_stage_range_violation=info['first_rk_stage_range_violation'],
                        final_absolute_rms=result['final_absolute_rms'], final_relative_l2=result['final_relative_l2'],
                        independent_uniform_status=independent,
                        independently_checked_final_absolute_rms=result['final_absolute_rms'] if independent == 'independent_agreement' else None,
                        steps=info['steps'], initial_condition_sha256=result['initial_condition_sha256']))
    for ident, row in refits.items():
        dataset = row['identity']['dataset']
        costs.append(dict(id=ident, setting=dataset['setting'], seed=dataset['seed'], continued=row['continued'],
            phases=len(row['phases']), optimizer_success=row['optimizer_success'], optimizer_status=row['optimizer_status'],
            parameter_count=row['phases'][-1]['parameter_count'], **row['totals'],
            final_observation_mse=row['phases'][-1]['observation_mse'], seconds=row['seconds']))
    return metrics, budgets, geometry, forwards, endpoint_rows, costs


def summaries(metrics, forwards):
    closure = []
    for setting in ('clean', 'noise3'):
        for method in LABELS:
            selected = [r for r in metrics if r['setting'] == setting and r['method'] == method]
            if len(selected) != 5: raise ValueError('Incomplete five-seed method group')
            closure.append(dict(setting=setting, method=method,
                **{branch+'_'+key: value for branch in ('D', 'R')
                   for key, value in moments([r[branch] for r in selected]).items()}))
    forward = []
    keys = sorted({(r['cohort'], r['case'], r['setting'], r['method'], r['pattern']) for r in forwards})
    for key in keys:
        rows = [r for r in forwards if tuple(r[k] for k in ('cohort', 'case', 'setting', 'method', 'pattern')) == key]
        if len(rows) != 5: raise ValueError('Incomplete five-seed forward group')
        count = lambda field: sum(bool(r[field]) for r in rows)
        forward.append(dict(zip(('cohort', 'case', 'setting', 'method', 'pattern'), key, strict=True), n=5,
            numerical_completion_count=count('numerical_completion'),
            accepted_range_valid_count=count('accepted_state_range_valid'),
            rk_stage_range_valid_count=count('rk_stage_state_range_valid'),
            nonnegative_count=count('nonnegative_accepted_states'),
            accepted_negative_count=sum(r['accepted_state_min'] is not None and r['accepted_state_min'] < -1e-10 for r in rows),
            accepted_nonfinite_extrema_count=sum(r['accepted_state_min'] is None for r in rows),
            completed_and_accepted_valid_count=sum(r['numerical_completion'] and r['accepted_state_range_valid'] for r in rows),
            completed_and_rk_valid_count=sum(r['numerical_completion'] and r['rk_stage_state_range_valid'] for r in rows),
            completed_and_nonnegative_count=sum(r['numerical_completion'] and r['nonnegative_accepted_states'] for r in rows),
            independent_uniform_agreement_count=sum(r['independent_uniform_status']=='independent_agreement' for r in rows) if key[-1].startswith('uniform_') else None,
            independent_uniform_failure_count=sum(r['independent_uniform_status']=='independent_solver_failure_or_disagreement' for r in rows) if key[-1].startswith('uniform_') else None,
            independent_uniform_failure_seeds=[r['seed'] for r in rows if r['independent_uniform_status']=='independent_solver_failure_or_disagreement'],
            numerical_failure_seeds=[r['seed'] for r in rows if not r['numerical_completion']],
            accepted_range_violation_seeds=[r['seed'] for r in rows if not r['accepted_state_range_valid']],
            **{'absolute_rms_'+k: v for k, v in moments([r['final_absolute_rms'] for r in rows]).items()},
            **{'checked_absolute_rms_'+k: v for k, v in moments([r['independently_checked_final_absolute_rms'] for r in rows]).items()},
            **{'relative_l2_'+k: v for k, v in moments([r['final_relative_l2'] for r in rows]).items()}))
    return closure, forward


def pairwise(matched):
    comparisons = [('refit_150', 'frozen_150'), ('refit_150', 'weak_poly'),
                   ('neural_adam_final', 'neural_150'), ('neural_lbfgs_float64', 'neural_adam_final')]
    return [dict(setting=setting, branch=branch, method_A=a, method_B=b,
                 difference_definition='error(A) - error(B); negative favors A',
                 **paired([row['closure_metrics'][a][branch]-row['closure_metrics'][b][branch]
                           for row in sorted(matched.values(), key=lambda r: r['config']['seed'])
                           if row['config']['setting'] == setting]))
            for setting in ('clean', 'noise3') for branch in ('D', 'R') for a, b in comparisons]


def save_figure(figure, name):
    figure.tight_layout()
    path = ROOT/'reports/figures'/name; path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches='tight'); plt.close(figure)


def plot_closures(matched):
    case = extra_case('case_exp'); grid = np.linspace(*case.value_range, 256)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for column, setting in enumerate(('clean', 'noise3')):
        rows = [r for r in matched.values() if r['config']['setting'] == setting]
        for index, branch in enumerate(('D', 'R')):
            ax = axes[index, column]; truth = case.diffusion(grid) if branch == 'D' else case.reaction(grid)
            ax.plot(grid, truth, color='black', linewidth=2, label='True law (evaluation only)')
            for label, color in COLORS.items():
                values = np.stack([functions(r['models'][label]['descriptor'])[index](grid) for r in rows])
                mean, sd = values.mean(0), values.std(0, ddof=1)
                ax.plot(grid, mean, color=color, label=LABELS[label], linewidth=1.4,
                        linestyle='--' if label == 'frozen_150' else '-')
                ax.fill_between(grid, mean-sd, mean+sd, color=color, alpha=.09)
            ax.set(xlabel='State u', ylabel=branch+'(u)', title=f'{setting}: five-seed mean ± sample SD')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    save_figure(fig, 'same_observation_closures.png')


def plot_budgets(training, geometry):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    epochs = (150, 500, 1500, 3000, 6000, 20000)
    for column, setting in enumerate(('clean', 'noise3')):
        for row in training.values():
            if row['identity']['dataset']['setting'] != setting: continue
            history = read(ROOT/'outputs/runs'/row['id']/'adam/history.json')
            axes[0, column].plot([r['epoch'] for r in history], [r['loss'] for r in history],
                                 alpha=.7, linewidth=.8, label='seed '+str(row['identity']['dataset']['seed']))
        axes[0, column].set(xscale='log', yscale='log', xlabel='Adam epoch', ylabel='Training objective (pre-step)', title=setting)
        for branch, color in [('D', '#0072b2'), ('R', '#d55e00')]:
            for field, style, label in [('neural_relative_error', '-', 'neural'), ('frozen_relative_error', '--', 'frozen')]:
                means, sds, xs = [], [], []
                for epoch in epochs:
                    selected = [r[field] for r in geometry if r['setting'] == setting and r['branch'] == branch
                                and r['stage'] == f'neural_adam_{epoch}']
                    if len(selected) == 5:
                        xs.append(epoch); means.append(np.mean(selected)); sds.append(np.std(selected, ddof=1))
                axes[1, column].errorbar(xs, means, yerr=sds, fmt='o'+style, color=color, capsize=2,
                                        label=branch+' '+label)
        axes[1, column].set(xscale='log', yscale='log', xlabel='Fixed Adam checkpoint', ylabel='Relative closure error')
        shown_epochs = [e for e in epochs if e <= max(r['final_adam_epoch'] for r in training.values()
                                                    if r['identity']['dataset']['setting'] == setting)]
        axes[1, column].set_xticks(shown_epochs, [str(e) if e<10000 else str(e//1000)+'k' for e in shown_epochs])
        axes[1, column].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[1, column].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    save_figure(fig, 'training_budget.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    stages = list(dict.fromkeys(r['stage'] for r in geometry))
    for ax, branch in zip(axes, ('D', 'R'), strict=True):
        selected = [r for r in geometry if r['branch'] == branch]
        hi = max(r['rho'] for r in selected)
        x = np.geomspace(max(1e-6, min(r['rho'] for r in selected)*.8), hi*1.2, 300)
        ax.fill_between(x, np.maximum(0., 1-x), 1+x, color='#777777', alpha=.12, label='Common-norm triangle bounds')
        ax.axhline(1., color='black', linewidth=.6)
        for i, stage in enumerate(stages):
            for setting, marker in [('clean', 'o'), ('noise3', 'x')]:
                rows = [r for r in selected if r['stage'] == stage and r['setting'] == setting]
                ax.scatter([r['rho'] for r in rows], [r['BIR'] for r in rows], marker=marker,
                           color=plt.cm.tab10(i%10), s=24, label=stage.replace('neural_', '')+' '+setting)
        ax.set(xscale='log', xlabel='rho = compression error / neural error', ylabel='BIR = frozen error / neural error', title=branch)
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=6, ncol=2, loc='upper left')
    save_figure(fig, 'compression_geometry.png')


def plot_dynamics(matched, annotations):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    patterns = ('uniform_zero', 'fourier_m1', 'fourier_m4')
    for i, setting in enumerate(('clean', 'noise3')):
        rows = [r for r in matched.values() if r['config']['setting'] == setting]
        for j, pattern in enumerate(patterns):
            ax = axes[i, j]
            for label in ('neural_150', 'refit_150', 'weak_poly', 'neural_adam_final'):
                completed = [r['models'][label]['validation'][pattern] for r in rows
                             if r['models'][label]['validation'][pattern]['info']['numerical_completion'] and
                             (pattern != 'uniform_zero' or annotations[('fresh',r['id'],label,pattern)] == 'independent_agreement')]
                condition = 'checked' if pattern == 'uniform_zero' else 'complete'
                if not completed:
                    ax.plot([], [], color=COLORS[label], label=LABELS[label]+f' (0/5 {condition})'); continue
                times = completed[0]['times']
                if pattern == 'uniform_zero':
                    values = np.array([r['absolute_rms'] for r in completed]); ylabel='Absolute RMS from zero reference'
                else:
                    values = np.array([np.asarray(r['predicted_modal_amplitude'])[:, 0] / r['predicted_modal_amplitude'][0][0]
                                       for r in completed]); ylabel='Mode amplitude / initial amplitude'
                mean = values.mean(0); sd = values.std(0, ddof=1) if len(values)>1 else np.zeros_like(mean)
                ax.plot(times, mean, color=COLORS[label], label=LABELS[label]+f' ({len(completed)}/5 {condition})')
                lower = np.maximum(0., mean-sd) if pattern == 'uniform_zero' else mean-sd
                ax.fill_between(times, lower, mean+sd, color=COLORS[label], alpha=.08)
            reference = rows[0]['models']['neural_150']['validation'][pattern]
            if pattern != 'uniform_zero':
                values = np.asarray(reference['true_modal_amplitude'])[:, 0]
                ax.plot(reference['times'], values/values[0], color='black', linestyle='--', label='True mode')
            ax.set(xlabel='Time', ylabel=ylabel, title=setting+' '+pattern); ax.grid(alpha=.2)
            if pattern == 'uniform_zero': ax.set_ylim(bottom=0.)
    axes[0, 0].legend(fontsize=6)
    axes[1, 0].legend(fontsize=6)
    save_figure(fig, 'uniform_fourier_dynamics.png')


def fmt(value):
    return '—' if value is None else f'{value:.5g}'


def mean_sd(row, branch):
    return fmt(row[branch+'_mean'])+' ± '+fmt(row[branch+'_sample_sd'])


def markdown_report(manifest, closure, budget, forward, costs, paired_rows, verification, uniform):
    lookup = {(row['setting'], row['method']): row for row in closure}
    lines = [
        '# EAAI-26-10676：同观测、训练预算与状态可靠性补实验', '',
        '本报告由已完成的本地原始记录重建。论文正文、Overleaf 提交稿及已有理论草稿均未改动；没有推送到 GitHub。', '',
        '## 优先阅读的结果', '',
        f"- 干净 Exp：150 轮神经模型的平均反应误差为 {fmt(lookup['clean','neural_150']['R_mean'])}，20,000 轮降至 {fmt(lookup['clean','neural_adam_final']['R_mean'])}；固定 150 轮所选家族的观测重拟合为 {fmt(lookup['clean','refit_150']['R_mean'])}。原固定预算误差不能全部解释成表示能力极限。",
        f"- 3% 噪声：原损失下神经平均反应误差从 {fmt(lookup['noise3','neural_150']['R_mean'])} 升至 6000 轮的 {fmt(lookup['noise3','neural_adam_final']['R_mean'])}。观测重拟合为 {fmt(lookup['noise3','refit_150']['R_mean'])}，空间弱式与时空弱式多项式参照分别为 {fmt(lookup['noise3','weak_poly']['R_mean'])} 和 {fmt(lookup['noise3','spacetime_poly']['R_mean'])}；不能声称重拟合普遍胜过直接多项式参照。",
        '- 新含噪观测重拟合的 5/5 模型在均匀零初值下进入负状态，其中 seed 1、3、4 共三例在区间外遇到有理函数极点。原 RK4 守卫仍报完成，独立 ODE 与到极点时间积分识别出这些失败；不将其伪终点纳入经核验的动态误差均值。', '',
        '## 范围与来源', '',
        '关键 Exp 基准采用干净与 3% 噪声各五个预先指定种子，共 10 份保存的观测 NPZ。新神经训练、三种多项式参照、新教师的冻结家族及观测系数重拟合均使用同一份文件，按完整 NPZ 和观测场字节核验。观测没有重新生成。', '',
        '新主对照包括空间弱式、强残差、时空弱式多项式；150 轮神经教师、其冻结符号家族和观测重拟合；长预算神经教师及其冻结家族；Float64 L-BFGS 交叉检查及其冻结家族。神经导出直接计算保存的权重，在支持区间外也使用实际网络，不再用历史查表端点延拓。评价网格统一为 [0,1.8] 的 256 点。', '',
        '历史 PySR 的 30 个输入、60 个冻结/重拟合模型另外完成状态审计，保留历史教师身份。它们没有被当作新教师的端到端同输入符号搜索；此次也没有新跑 Julia/PySR 结构搜索。', '',
        '## 同观测闭合误差', '',
        '下表是全部五种子的均值 ± 样本标准差。D、R 都是相同网格上的相对 L2 误差；真律仅用于事后评价。', '',
    ]
    for setting in ('clean', 'noise3'):
        lines += [f'### {setting}', '', '| 方法 | D 误差 | R 误差 |', '| --- | ---: | ---: |']
        for row in closure:
            if row['setting'] == setting:
                lines.append(f"| {LABELS[row['method']]} | {mean_sd(row,'D')} | {mean_sd(row,'R')} |")
        lines.append('')
    lines += [
        '![同观测闭合曲线](figures/same_observation_closures.png)', '',
        '系数重拟合改变了目标：它固定新 150 轮教师选择的家族与阶数，独立优化所有系数槽位及速率，目标是保存的轨迹观测而非教师曲线。它使用 RK45 前向拟合和采样正扩散惩罚，没有神经训练中的软反应端点锚定。三种多项式参照的扩散/反应次数固定为 2/3；所选符号家族的类型、次数和复杂度另行记录，并不保证与这些参照相同。因此这是一项明确不同的观测纠偏协议，不能声称所有目标、模型空间、先验及成本都相同，也不能将它等同于直接从数据进行结构搜索。', '',
        '配对差值采用 A−B，负值表示 A 的误差较小；全部十六组差值和五种子明细见 [paired_closure_differences.csv](paired_closure_differences.csv)。n=5 的 t 区间为探索性描述，未作多重比较校正。', '',
        '## 训练预算与优化状态', '',
        '原默认网络、初始化种子、Adam 学习率 0.001 和损失权重保持固定，每步使用全部 8×101×64 观测。先跑 6000 轮；仅当最后 1000 轮前 100 与后 100 轮的平均训练目标下降超过 1% 时，从同一模型及 Adam 优化器状态续到 20,000 轮。停止/续跑规则不使用真实闭合误差，没有重启选优或最好种子选择。', '',
        '| 输入 | Adam 终点 | 尾窗下降 | 终点梯度最大值 | L-BFGS 外步 | L-BFGS 停止状态 |',
        '| --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in budget:
        lines.append(f"| {row['setting']} seed {row['seed']} | {row['final_adam_epoch']} | {row['final_tail_decline']:.2%} | {fmt(row['adam_post_step_gradient_max'])} | {row['lbfgs_outer_steps']} | {row['lbfgs_stopping']} |")
    lines += ['',
        '尾窗下降小于阈值只说明该固定窗口内的变化较小；负下降表示末段均值上升。Float64 L-BFGS 为单独的优化器与精度交叉检查，最多 15 个外步、每步最多 20 次内迭代，采用强 Wolfe 线搜索。梯度阈值是 1e-7；另有最大参数变化 1e-10 且目标变化 1e-12 的停止条件。达到这些局部条件或预算上限均不证明全局恢复或全局收敛；参数/目标停滞时梯度仍可能高于阈值。', '',
        'Adam 历史记录是更新前目标，表格的终点诊断重新计算更新后目标和梯度；两者不要求逐字相等。L-BFGS 的初始 Float64 目标与 Adam Float32 目标也分别记录。完整数值、尾段状态、L-BFGS 调用数及运行时间见 [training_budget.csv](training_budget.csv)。', '',
        '![训练预算](figures/training_budget.png)', '',
        '### 压缩距离与误差比例', '',
        '所有检查点都从其自身教师重新压缩；家族选择保留原 5% 教师曲线相对容差及 1e-8 绝对容差。rho=eC/eN，BIR=eS/eN，使用完全相同的 256 点网格与分支范数。每个点都核验 max(0,1−rho)≤BIR≤1+rho。该界是标准范数结果；小 rho 下近一不是新的恢复定理。', '',
        '教师误差变小时，即使教师幅值归一化的压缩容差不变，rho 也可能上升；冻结压缩可能降低或增加闭合误差。因此应把保留误差的说法限定到冻结教师及压缩误差相对教师误差小的条件，不能推广成预算无关的近一或符号方法不能纠偏。', '',
        '![rho 与 BIR](figures/compression_geometry.png)', '',
        '## 前向可靠性：完成、非负与支持区间分开', '',
        '每个模型验证八种初值：四条全支持随机场、四条固定中段随机场（均至 T=0.5）；零、低、中、上端四种均匀状态与 m=1、m=4 的小幅 Fourier 场（至 T=0.1）。单位周期区间内实际角波数为 2πm，网格 64 点。RK4 使用最大步长 1e-4 及扩散 CFL 限制，没有状态裁剪；评价分别追踪初始值、每个接受步及每个 RK 试探阶段。', '',
        '下面只列新主对照的均匀零初值。真解为零，相对误差分母不存在，使用绝对 RMS。区间通过数指接受状态始终在 [0,1.8] 内（容差 1e-10）；数值完成并不要求区间通过。误差均值仅使用与独立标量 ODE 一致的终点，并给出人数；这仍不是物理状态通过的证书。', '',
        '| 数据 | 方法 | RK4 守卫完成 /5 | 非负 /5 | 接受状态区间通过 /5 | 独立失败 /5 | 经独立核对的绝对 RMS 均值 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in forward:
        if row['cohort'] == 'fresh_same_observation' and row['pattern'] == 'uniform_zero':
            lines.append(f"| {row['setting']} | {LABELS[row['method']]} | {row['numerical_completion_count']} | {row['nonnegative_count']} | {row['accepted_range_valid_count']} | {row['independent_uniform_failure_count']} | {fmt(row['checked_absolute_rms_mean'])} (n={row['checked_absolute_rms_n']}) |")
    lines += ['',
        '平均误差仅对完成目标时长的模型计算，明确保留 n；失败模型没有以零填充或从状态计数中删除。非负通过数包括已跟踪的接受状态，需要与数值完成列一起看；非有限状态导致极值不可用时不记为通过，也不自动归为已经测得负状态。汇总另列实测负状态数、极值不可用数与完成且通过数。全部 1280 个模型×初值组合及首次越界、极值、数值失败原因见 [forward_details.csv](forward_details.csv)；分组计数和完成条件下误差见 [forward_summary.csv](forward_summary.csv)。接受步和 RK 阶段均有各自状态列。', '',
        '对于本非负基准，负状态直接违反允许状态域；越出评价上端首先意味着支持区间外推，应另看应用约束，不能一概称为非物理。R(0) 非零即不保持零平衡，即便 R(0)>0 且轨迹保持非负也是如此。', '',
        '均匀状态不含扩散项，因此可独立暴露反应平衡与边界行为。另用独立标量 DOP853 求解器复核全部已通过 RK4 守卫的均匀状态终点与绝对误差；没有完成和独立核验失败的组合也保留在 [uniform_ode_checks.json](../evidence/uniform_ode_checks.json) 中。审计执行 complete 不表示所有模型通过；独立求解器失败或终点不一致者，不纳入上表经核对的均值。原 RK4 守卫完成条件下的数值仍在 CSV 中单独保留。', '',
        '新增的具体反例是新主对照 noise3 seed 1 的观测重拟合反应函数：R(0)=−0.5546342091，使均匀状态立即越出非负域，随后接近区间外实极点 u=−0.0176935543891。沿均匀 ODE 的 1/R 积分给出到达极点的时间约 0.0138674407182，与 DOP853 的终止时间一致。固定步 RK4 跨过了该奇点，守卫仍报完成，T=0.1 的数值为约 0.05010；该数值不能当作原连续动力学的可信预测。支持区间内未检出分母根，不能保证越界后的动力学仍有效。', '',
        '首次标量核验在这个反例处中止的原始失败记录与当时脚本均保留在 evidence/failures。之后仅修改独立审计的记录逻辑，使它继续检查全部模型并明确记录失败；没有修改训练、拟合、原 RK4 轨迹或原前向协议，也没有放宽 2e-9 一致性阈值。', '',
        'Fourier 模态与空间均值分开保存，可检查反应背景与模态衰减；图中使用实际非线性数值轨迹，没有把线性化公式当成精确解。均匀零图只含独立核对一致者，Fourier 图仅以原数值守卫完成为条件；图例给出人数，全部失败仍在表中。阴影为样本标准差，均匀图的 RMS 阴影下端仅在绘图时截到零；不代表状态范围或置信区间。', '',
        '![均匀与 Fourier 动态](figures/uniform_fourier_dynamics.png)', '',
    ]
    lines += ['### 独立核验发现的极点反例', '',
              '| 输入 | 极点状态 | 到达时间（1/R 积分） | 原 RK4 的 T=0.1 伪终点 |',
              '| --- | ---: | ---: | ---: |']
    for failure in uniform['independent_model_failures']:
        pole = failure.get('rational_pole')
        if pole is not None:
            lines.append(f"| {failure['id']} / {failure['method']} / {failure['pattern']} | {fmt(pole['pole_state'])} | {fmt(pole['travel_time'])} | {fmt(failure['rk4_signed_endpoint'])} |")
    lines += ['', '### 历史 PySR 状态审计', '',
              '| 基准 / 数据 | 模型 | 零初值数值完成 /5 | 零初值非负 /5 | 接受状态区间通过 /5 |',
              '| --- | --- | ---: | ---: | ---: |']
    for row in forward:
        if row['cohort'] == 'archived_pysr' and row['pattern'] == 'uniform_zero':
            lines.append(f"| {row['case']} / {row['setting']} | {row['method']} | {row['numerical_completion_count']} | {row['nonnegative_count']} | {row['accepted_range_valid_count']} |")
    lines += ['',
        '端点反应值、4097 点上的有限性及正扩散诊断、受限有理/饱和家族在支持区间内检测到的实分母根见 [endpoint_diagnostics.csv](endpoint_diagnostics.csv)。采样正扩散及数值无越界不构成全区间解析证书；此次没有添加物理约束修补后重新拟合的协议，也没有覆盖全部历史神经、二维及其他闭合模型。', '',
        '## 系数优化成本与完成状态', '',
        '| 输入 | 优化器局部停止成功 | 分配阶段 | nfev | 实际残差调用 | RHS 调用 | 失败试探 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in costs:
        lines.append(f"| {row['setting']} seed {row['seed']} | {row['optimizer_success']} | {row['phases']} | {row['nfev']} | {row['residual_evaluations']} | {row['solver_rhs_evaluations']} | {row['failed_trial_evaluations']} |")
    lines += ['',
        '第一阶段 max_nfev=40，仅在预算耗尽时以系数热启动第二阶段 max_nfev=80；第二阶段重新启动 least_squares，没有声称恢复其内部信赖域。nfev 不含全部数值 Jacobian 额外残差调用；RHS 计数包含失败试探与拟合后检查。执行 complete 与优化器 success 是不同字段，预算耗尽者仍保留。', '',
        '## 可供后续修稿采用的边界', '',
        '- 将 150 轮结果表述为固定预算参照，结合全种子长训练说明误差对优化预算的敏感性；没有证据把原误差全部归结为网络表示能力。',
        '- 用新同观测受限家族对照回答冻结压缩与观测纠偏的区别；历史 PySR 作为另一套有教师来源限制的结构搜索证据。',
        '- 误差改善、数值完成、非负状态与支持区间有效性分别报告。未施加的端点约束不能事后假定已经满足。',
        '- 此次不支持完整因果噪声归因、全局可辨识性、未知真实闭合的恢复证书或所有神经符号算法的普遍结论。没有四臂噪声位置训练，也没有新真实数据、三维或结构搜索。', '',
        '## 文件与复现', '',
        '- [summary.json](summary.json)：统计汇总与输入身份核验。',
        '- [closure_by_seed.csv](closure_by_seed.csv)、[closure_summary.csv](closure_summary.csv)：全种子闭合指标。',
        '- [compression_geometry.csv](compression_geometry.csv)：每个教师检查点的压缩距离、BIR、rho 与家族复杂度。',
        '- [coefficient_refit_costs.csv](coefficient_refit_costs.csv)：局部停止标志与真实调用成本。',
        '- [verification.json](verification.json)：10 个输入及全部主对照的文件、教师、初值、指标和范数界核验。',
        '- [训练协议](../evidence/PROTOCOL.json)、[前向协议](../evidence/EVALUATION_PROTOCOL.json)、[环境](../provenance/environment.json)、[原始测试结果](../evidence/tests.xml)及[CUDA 续跑检查](../evidence/cuda_tests.xml)。', '',
        '常规测试 96 项通过，2 项 opt-in CUDA 测试在该次默认运行中跳过；随后在本地 RTX 5070 Ti / WSL 环境单独启用这两项，1D、2D 的连续训练与优化器断点恢复均逐位一致，2 项通过。大补实验本身使用 1D GPU 训练；这不等于已经完整运行全部 2D GPU 矩阵或验证 Windows 原生 Python。', '',
        '在项目根目录使用保留的根虚拟环境运行；历史 PySR 只需现有最终 JSON，不需重跑 Julia搜索。工作根及输入根应明确指定：', '',
        '```bash',
        'python experiments/report_reliability_supplement.py \\',
        '  --work-root outputs/reviewer-supplement-2026-10-12 \\',
        '  --input-root outputs/revision',
        '```', '',
        '报告重建不重新训练、拟合或积分。原始记录和本地优化器断点仍保留；任何数值源代码变化都会使本实验的身份核验拒绝旧缓存，应另建有说明的实验根目录。', '',
    ]
    return '\n'.join(lines)


def main():
    manifest = load_manifest()
    training, baselines, matched, archive, refits = (
        complete_records(folder, count) for folder, count in
        [('training', 10), ('baselines', 10), ('matched_validation', 10), ('archive_validation', 30), ('fresh_refits', 10)])
    verification = authenticate(manifest, training, baselines, matched, archive, refits)
    uniform = read(ROOT/'evidence/uniform_ode_checks.json')
    if (uniform['status'] != 'complete' or uniform['total'] != 640 or
        uniform['protocol'] != evaluation_identity() or
        uniform['verifier_sha256'] != digest(Path(__file__).with_name('verify_uniform_supplement.py'))):
        raise ValueError('Independent uniform-state check is incomplete or unauthenticated')
    for check in uniform['checks'] + uniform['failed_forward_patterns_retained'] + uniform['independent_model_failures']:
        folder = 'matched_validation' if check['cohort'] == 'fresh' else 'archive_validation'
        if check['source_record_sha256'] != digest(ROOT/'evidence'/folder/f"{check['id']}.json"):
            raise ValueError('Uniform verification input changed')
    suites = ET.parse(ROOT/'evidence/tests.xml').getroot().iter('testsuite')
    totals = {key: 0 for key in ('tests', 'failures', 'errors', 'skipped')}
    for suite in suites:
        for key in totals: totals[key] += int(suite.attrib[key])
    if totals['failures'] or totals['errors']:
        raise ValueError('Required test suite failed')
    cuda_tests = {key: 0 for key in ('tests', 'failures', 'errors', 'skipped')}
    for suite in ET.parse(ROOT/'evidence/cuda_tests.xml').getroot().iter('testsuite'):
        for key in cuda_tests: cuda_tests[key] += int(suite.attrib[key])
    if cuda_tests['tests'] != 2 or any(cuda_tests[key] for key in ('failures', 'errors', 'skipped')):
        raise ValueError('Both opt-in CUDA checkpoint-resume checks must pass')
    annotations = uniform_annotations(uniform)
    metrics, budget, geometry, forwards, endpoints, costs = tabulate(training, matched, archive, refits, annotations)
    closure, forward = summaries(metrics, forwards); comparisons = pairwise(matched)
    for name, rows in [('closure_by_seed.csv', metrics), ('closure_summary.csv', closure),
                       ('training_budget.csv', budget), ('compression_geometry.csv', geometry),
                       ('forward_details.csv', forwards), ('forward_summary.csv', forward),
                       ('endpoint_diagnostics.csv', endpoints), ('coefficient_refit_costs.csv', costs),
                       ('paired_closure_differences.csv', comparisons)]:
        write_csv(name, rows)
    numerical_failures = [row for row in forwards if not row['numerical_completion']]
    violations = [row for row in forwards if not row['accepted_state_range_valid']]
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        experiment_scope='Fresh Exp clean/noise3, five seeds each; archived PySR reliability audited separately',
        datasets=10, fresh_models=100, archived_models=60, model_pattern_validations=len(forwards),
        numerical_failure_count=len(numerical_failures), accepted_range_violation_count=len(violations),
        closure_summary=closure, forward_summary=forward, paired_closure_differences=comparisons,
        training_budget=budget, coefficient_costs=costs,
        totals={key:sum(row[key] for row in costs) for key in
                ('nfev', 'residual_evaluations', 'solver_rhs_evaluations', 'failed_trial_evaluations')},
        verification=verification, all_failed_and_violating_model_patterns_retained=True,
        reporting_policy='No true-error stopping or seed selection; conditional forward errors include completion counts')
    summary['tests'] = dict(**totals, passed=totals['tests']-totals['skipped'])
    summary['tests']['separate_cuda_checkpoint_resume'] = dict(**cuda_tests, passed=cuda_tests['tests'])
    summary['uniform_ode_verification'] = {key: uniform[key] for key in
        ('status', 'total', 'compared', 'numerical_failures', 'independent_failure_count',
         'all_completed_rk4_uniform_endpoints_independently_verified', 'maximum_endpoint_difference', 'maximum_absolute_rms_difference')}
    json_write(ROOT/'reports/summary.json', summary)
    json_write(ROOT/'reports/verification.json', dict(status='passed', datasets=verification,
        scope='Input/source/teacher/initial-condition provenance checks, not certification of every model',
        uniform_ode_audit=summary['uniform_ode_verification'], independent_uniform_model_failures=uniform['independent_model_failures'],
        protocols_authenticated=True, tests=summary['tests'], archived_audits=30, numerical_failures_retained=numerical_failures,
        accepted_range_violations_retained=violations))
    plot_closures(matched); plot_budgets(training, geometry); plot_dynamics(matched, annotations)
    (ROOT/'reports/README.md').write_text(markdown_report(manifest, closure, budget, forward, costs, comparisons, verification, uniform))
    destination = ROOT/'provenance/source/experiments'/Path(__file__).name
    shutil.copyfile(__file__, destination)
    json_write(ROOT/'reports/ARTIFACTS.json', dict(report_builder_sha256=digest(__file__),
        artifacts=[dict(path=str(path.relative_to(ROOT)), sha256=digest(path))
                   for path in sorted((ROOT/'reports').rglob('*')) if path.is_file() and path.name != 'ARTIFACTS.json']))
    print(json.dumps(dict(event='reports_rebuilt', datasets=len(verification), validations=len(forwards),
        numerical_failures=len(numerical_failures), accepted_range_violations=len(violations),
        report=str(ROOT/'reports/README.md'))), flush=True)


if __name__ == '__main__':
    main()
