"""Authenticate all new artifacts, evaluate linear noise gains, and report."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from slit_flow_core import (FluidityNetwork, network_values, exact_fluidity, exact_velocity,
                            exact_flow_rate, basis, velocity_basis, polynomial, oscillatory_counterexample,
                            oscillatory_derivative, velocity_prediction, flow_prediction)
from run_slit_flow_controls import SOURCE, digest, write_json, write_csv, load_protocol


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def authenticate(output):
    protocol = load_protocol(output)
    continuation = json.loads((output / 'LBFGS_PROTOCOL.json').read_text())
    if digest(SOURCE / 'experiments/continue_slit_flow_controls.py') != continuation['continuation_source_sha256']:
        raise ValueError('Continuation source changed')
    environment = json.loads((output / 'provenance/environment.json').read_text())
    for relative, sha in environment['existing_core_sha256'].items():
        if digest(SOURCE / relative) != sha:
            raise ValueError('Previous numerical core changed: ' + relative)
    records = []
    maximum_curve_error = 0.0
    maximum_analytic_flow_error = 0.0
    audit_protocol=json.loads((output/'QUADRATURE_AUDIT_V2_PROTOCOL.json').read_text())
    audit=json.loads((output/'QUADRATURE_AUDIT_V2.json').read_text())
    if audit['status']!='passed' or audit['protocol_sha256']!=digest(output/'QUADRATURE_AUDIT_V2_PROTOCOL.json'):
        raise ValueError('Independent refined forward audit has not passed')
    if audit_protocol['source_sha256']!=digest(SOURCE/'experiments/audit_slit_flow_quadrature.py'):
        raise ValueError('Independent quadrature audit source changed')
    quadrature_checks=[]
    refined_checks=read_csv(output/'reports/refined_quadrature_checks_V2.csv')
    refined_lookup={(r['stage'],r['dataset'],r['objective'],float(r['pressure'])):r for r in refined_checks}
    by_initial = {}
    for dataset in protocol['datasets']:
        input_path = output / dataset['file']
        if digest(input_path) != dataset['file_sha256']:
            raise ValueError('Observation file changed')
        with np.load(input_path) as data:
            if hashlib.sha256(data['velocity'].tobytes()).hexdigest() != dataset['observation_sha256']:
                raise ValueError('Observation array changed')
        for objective in protocol['plan']['objectives']:
            path = output / 'fits' / (dataset['id'] + '_' + objective + '.json')
            record = json.loads(path.read_text())
            if record['status'] != 'complete' or record['identity']['dataset'] != dataset:
                raise ValueError('Incomplete or mismatched original result')
            records.append(('adam1500', record, dataset, objective))
            by_initial[path.name] = record
    for item in continuation['inputs']:
        if digest(output / item['file']) != item['sha256']:
            raise ValueError('Original input to continuation changed')
        path = output / 'continuation' / Path(item['file']).name
        record = json.loads(path.read_text())
        expected = dict(input=item, continuation_protocol_sha256=digest(output / 'LBFGS_PROTOCOL.json'))
        if record['status'] != 'complete' or record['identity'] != expected:
            raise ValueError('Incomplete or mismatched continuation')
        records.append(('lbfgs', record, record['dataset'], record['objective']))
    if len(records) != 80 or len(list((output / 'fits').glob('*.json'))) != 60 or len(list((output / 'continuation').glob('*.json'))) != 20:
        raise ValueError('Unexpected or missing results')
    torch.set_num_threads(1)
    for stage, record, dataset, objective in records:
        for name in ('model', 'curves'):
            if digest(output / record[name + '_file']) != record[name + '_sha256']:
                raise ValueError('Model or curve artifact changed')
        saved = torch.load(output / record['model_file'], weights_only=True)
        model = FluidityNetwork(protocol['plan']['width']).double()
        model.load_state_dict(saved['state_dict'])
        function=lambda stress:network_values(model,stress)
        for pressure in (1.,1.25):
            independent=refined_lookup[(stage,dataset['id'],objective,pressure)]
            fixed=flow_prediction(function,pressure,order=128)
            ycheck=np.array([0.,.5,.9])
            u32=velocity_prediction(function,ycheck,pressure,order=32)
            u64=velocity_prediction(function,ycheck,pressure,order=64)
            u128=velocity_prediction(function,ycheck,pressure,order=128)
            quadrature_checks.append(dict(stage=stage,dataset=dataset['id'],objective=objective,pressure=pressure,
                fixed_order128_flow=fixed,adaptive_flow=float(independent['adaptive_flow']),
                flow_absolute_difference=abs(fixed-float(independent['adaptive_flow'])),
                velocity_order32_vs128=float(np.max(np.abs(u32-u128))),velocity_order64_vs128=float(np.max(np.abs(u64-u128)))))
        with np.load(output / record['curves_file']) as curves:
            difference = float(np.max(np.abs(network_values(model, curves['stress']) - curves['neural'])))
            maximum_curve_error = max(maximum_curve_error, difference)
            np.testing.assert_allclose(curves['truth'], exact_fluidity(curves['stress'], protocol['plan']['alpha']), atol=0, rtol=0)
        for family in record['symbols']:
            for kind, method in [('frozen',family + '_frozen'), ('observation',family + '_observation_fit')]:
                coefficients = record['symbols'][family][kind + '_coefficients']
                if not np.isfinite(coefficients).all():
                    raise ValueError('Nonfinite coefficients')
                for pressure in protocol['plan']['pressure_test']:
                    analytic = sum(2 * c * pressure**(2*i+1) / (2*i+3) for i,c in enumerate(coefficients))
                    independent = 2 * quad(lambda y: y * float(polynomial(np.array([pressure*y]),coefficients)[0]),0,1)[0]
                    maximum_analytic_flow_error = max(maximum_analytic_flow_error, abs(analytic-independent))
                    error = abs(analytic - exact_flow_rate(pressure, protocol['plan']['alpha'])) / exact_flow_rate(pressure, protocol['plan']['alpha'])
                    np.testing.assert_allclose(record['metrics'][method]['flow_relative_errors'][str(pressure)], error, atol=2e-14, rtol=1e-9)
        if stage == 'lbfgs':
            before = by_initial[Path(record['identity']['input']['file']).name]
            for family in record['symbols']:
                np.testing.assert_array_equal(record['symbols'][family]['observation_coefficients'], before['symbols'][family]['observation_coefficients'])
    if maximum_curve_error > 1e-13 or maximum_analytic_flow_error > 1e-12:
        raise ValueError('Saved model/independent integral discrepancy')
    maximum_adaptive_flow_difference=max(float(row['flow_absolute_difference']) for row in quadrature_checks)
    write_csv(output/'reports/quadrature_checks.csv',quadrature_checks)
    # The failed fixed-rule check remains a scientific/numerical diagnostic.
    # Only the separately validated refined audit supplies headline forward metrics.
    revised=[]
    for stage,record,dataset,objective in records:
        key=f'{stage}/{dataset["id"]}/{objective}'
        source_path=output/('fits' if stage=='adam1500' else 'continuation')/(dataset['id']+'_'+objective+'.json')
        if digest(source_path)!=audit['results'][key]['input_record_sha256']:
            raise ValueError('Refined evaluation refers to a different teacher')
        revised_record=dict(record,metrics=audit['results'][key]['metrics'])
        revised.append((stage,revised_record,dataset,objective))
    records=revised
    fd = read_csv(output / 'controls/finite_difference.csv')
    maximum_noise_relative_error = max(abs(float(r['empirical_noise_mean_square']) / float(r['predicted_noise_variance']) - 1) for r in fd)
    if maximum_noise_relative_error > 0.03:
        raise ValueError('Monte Carlo derivative variance differs by more than 3%')
    projection = read_csv(output / 'controls/projection.csv')
    maximum_projection_identity_error = max(float(row['projection_energy_identity_residual']) for row in projection)
    if maximum_projection_identity_error > 1e-12:
        raise ValueError('Projection energy identity failed')
    counterexamples = read_csv(output / 'controls/monotonicity.csv')
    for row in counterexamples:
        cycles = int(row['cycles'])
        witness = np.array([1/(2*cycles)])
        np.testing.assert_allclose(oscillatory_derivative(witness,cycles),[-0.9],atol=1e-14)
        independent = 2*quad(lambda t:t*float(oscillatory_counterexample(t,cycles)),0,1,
                             points=np.linspace(0,1,2*cycles+1)[1:-1],epsabs=1e-12,epsrel=1e-12,limit=4096)[0]
        np.testing.assert_allclose(independent,float(row['flow_compressed']),atol=1e-12)
    suites=ET.parse(output/'provenance/unit_checks.xml').getroot().findall('testsuite')
    test_count=sum(int(suite.attrib['tests']) for suite in suites)
    failed_tests=sum(int(suite.attrib['failures'])+int(suite.attrib['errors']) for suite in suites)
    if test_count!=7 or failed_tests:raise ValueError('Required scientific unit checks did not pass')
    integrity = dict(status='passed', original_teachers=60, paired_continuations=20, saved_input_records=30,
        source_and_input_hashes_verified=True, original_numerical_core_unchanged=True,
        maximum_saved_model_curve_difference=maximum_curve_error, maximum_independent_polynomial_flow_difference=maximum_analytic_flow_error,
        original_fixed_rule_flow_difference_from_adaptive_integral=maximum_adaptive_flow_difference,
        original_neural_velocity_order32_vs128=max(float(row['velocity_order32_vs128']) for row in quadrature_checks),
        original_neural_velocity_order64_vs128=max(float(row['velocity_order64_vs128']) for row in quadrature_checks),
        refined_quadrature_audit_passed=True,refined_quadrature_maxima=audit['maxima'],
        original_fixed_rule_failure_preserved=True,
        maximum_monte_carlo_noise_variance_relative_difference=maximum_noise_relative_error,
        maximum_projection_energy_identity_error=maximum_projection_identity_error,
        all_analytic_monotonicity_witnesses_verified=True,
        unit_checks={'tests':test_count,'failures':failed_tests,'junit_file':'provenance/unit_checks.xml'},
        interpretation='80 neural states include 20 continuations; clean seeds are restarts, not independent observations; no truth-based model selection')
    write_json(output / 'INTEGRITY_CHECKS.json', integrity)
    return protocol, records, integrity


def linear_noise_gains(output, protocol):
    rows = []
    alpha = protocol['plan']['alpha']
    for n in protocol['plan']['intervals']:
        y = np.linspace(0,1,n+1)
        a = velocity_basis(y)
        b = basis(y[1:-1])
        difference = np.zeros((n-1,n+1))
        index = np.arange(n-1)
        difference[index,index] = n/2
        difference[index,index+2] = -n/2
        sigma = 0.01*float(exact_velocity(y,alpha=alpha).max())
        observation_gain = np.linalg.pinv(a)
        differential_gain = np.linalg.pinv(b) @ difference
        for name,gain,bias in [('observation',observation_gain,np.zeros(2)),('differential',differential_gain,np.array([alpha/n**2,0.]))]:
            covariance = sigma**2 * gain @ gain.T
            rows.append(dict(intervals=n,estimator=name,coefficient1_bias=float(bias[0]),coefficient3_bias=float(bias[1]),
                coefficient1_noise_sd=float(np.sqrt(covariance[0,0])),coefficient3_noise_sd=float(np.sqrt(covariance[1,1])),
                noise_covariance13=float(covariance[0,1]),
                note='Analytical coefficient covariance of the fixed cubic linear estimator, retaining differentiated-noise correlations'))
    write_csv(output / 'reports/linear_noise_gains.csv',rows)


def paired_error_decomposition(output,protocol,records):
    lookup={(stage,d['intervals'],d['noise'],d['seed'],o):r for stage,r,d,o in records}
    rows=[]
    for stage,record,dataset,objective in records:
        if dataset['noise']!=.01:continue
        clean=lookup[(stage,dataset['intervals'],0.,dataset['seed'],objective)]
        with np.load(output/record['curves_file']) as curves:
            mask=curves['stress']<=1
            stress=curves['stress'][mask]
            noisy=curves['neural'][mask]
            truth=curves['truth'][mask]
        with np.load(output/clean['curves_file']) as curves:
            clean_teacher=curves['neural'][mask]
        bias=protocol['plan']['alpha']*stress/dataset['intervals']**2 if objective=='differential' else np.zeros_like(stress)
        for family in record['symbols']:
            symbolic=polynomial(stress,record['symbols'][family]['frozen_coefficients'])
            components=[bias,clean_teacher-truth-bias,noisy-clean_teacher,symbolic-noisy]
            total=symbolic-truth
            residual=float(np.max(np.abs(total-sum(components))))
            if residual>1e-13:raise ValueError('Paired vector decomposition fails')
            squared=[float(np.mean(component**2)) for component in components]
            total_squared=float(np.mean(total**2))
            row=dict(stage=stage,intervals=dataset['intervals'],seed=dataset['seed'],objective=objective,family=family,
                numerical_target_bias_norm=float(np.sqrt(squared[0])),clean_learning_residual_norm=float(np.sqrt(squared[1])),
                paired_noise_response_norm=float(np.sqrt(squared[2])),compression_change_norm=float(np.sqrt(squared[3])),
                symbolic_true_error_norm=float(np.sqrt(total_squared)),cross_terms_total=total_squared-sum(squared),
                exact_vector_identity_residual=residual,common_norm='Unweighted RMS on the original1025-point stress grid[0,1]')
            rows.append(row)
    write_csv(output/'reports/paired_error_decomposition.csv',rows)


def tabulate(output, records):
    rows = []
    for stage,record,dataset,objective in records:
        for method,value in record['metrics'].items():
            row = dict(stage=stage,intervals=dataset['intervals'],noise=dataset['noise'],seed=dataset['seed'],objective=objective,method=method,
                relative_closure_error=value['relative_closure_error'],observation_mse=value['observation_mse'],
                minimum_derivative=value['sampled_min_derivative'], minimum_derivative_extended=value['sampled_min_derivative_extended'],
                monotonicity_violation=value['sampled_min_derivative']<=0, zero_response=value['zero_response'],
                positive_response=value['sampled_min_positive_response']>0,
                derivative_check='sampled1025points' if method=='neural' else 'endpoint minimum is exact for this odd cubic or linear family',
                flow_error05=value['flow_relative_errors']['0.5'],flow_error1=value['flow_relative_errors']['1.0'],flow_error125=value['flow_relative_errors']['1.25'])
            if method=='neural':
                row['refined_continuous_relative_closure_error']=value['refined_continuous_relative_closure_error']
            else:
                family=method.split('_observation')[0] if method.endswith('observation_fit') else method.removesuffix('_frozen')
                slot='observation_coefficients' if method.endswith('observation_fit') else 'frozen_coefficients'
                coefficients=record['symbols'][family][slot]
                a=coefficients[0]-1
                b=(coefficients[1] if len(coefficients)==2 else 0.)-0.6
                row['refined_continuous_relative_closure_error']=float(np.sqrt(max(0.,a*a/3+2*a*b/5+b*b/7)/(1/3+1.2/5+.36/7)))
            rows.append(row)
    write_csv(output/'reports/by_seed.csv',rows)
    unique_references = [r for r in rows if r['stage']=='adam1500' and r['objective']=='differential' and r['method'].endswith('observation_fit')]
    write_csv(output/'reports/unique_observation_fits.csv',unique_references)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['stage'],row['intervals'],row['noise'],row['objective'],row['method'])].append(row)
    summaries = []
    for key,values in sorted(groups.items()):
        summary = dict(zip(('stage','intervals','noise','objective','method'),key))
        summary['n'] = len(values)
        for field in ('relative_closure_error','refined_continuous_relative_closure_error','observation_mse','flow_error05','flow_error1','flow_error125'):
            data = np.array([v[field] for v in values])
            summary[field+'_mean'] = float(data.mean())
            summary[field+'_sd'] = float(data.std(ddof=1))
        summary['monotonicity_violations'] = sum(v['monotonicity_violation'] for v in values)
        summary['minimum_derivative_over_seeds'] = min(v['minimum_derivative'] for v in values)
        summaries.append(summary)
    write_csv(output/'reports/summary.csv',summaries)
    return rows,summaries


def figures(output, protocol, summaries):
    folder = output/'reports'
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    projections=read_csv(output/'controls/projection.csv')
    for degree,name in [(3,'Cubic, truth in family'),(1,'Linear, omitted cubic term')]:
        subset=[r for r in projections if int(r['degree'])==degree]
        axes[0,0].plot([float(r['angle_radians']) for r in subset],[float(r['symbolic_error']) for r in subset],'o-',label=name)
    axes[0,0].axhline(.03,color='gray',ls=':',label='Fixed teacher error')
    axes[0,0].set(xlabel='Error-direction angle (rad)',ylabel='Common-norm closure error',title='Controlled projection and family remainder')
    axes[0,0].legend(fontsize=8)
    variance=read_csv(output/'controls/finite_difference.csv')
    n=[int(r['intervals']) for r in variance]
    axes[0,1].loglog(n,[float(r['predicted_noise_variance']) for r in variance],'-',label='Analytical variance')
    axes[0,1].loglog(n,[float(r['empirical_noise_mean_square']) for r in variance],'o',label='20,000 noise draws per grid')
    axes[0,1].set(xlabel='Grid intervals',ylabel='Differentiated noise variance',title='Refinement at fixed observation noise')
    axes[0,1].legend(fontsize=8)
    for objective in ('differential','integral'):
        for method,style in [('neural','-'),('odd_cubic_frozen','--')]:
            subset=sorted([r for r in summaries if r['stage']=='adam1500' and r['noise']==.01 and r['objective']==objective and r['method']==method],key=lambda r:r['intervals'])
            axes[1,0].plot([r['intervals'] for r in subset],[r['relative_closure_error_mean'] for r in subset],style,marker='o',label=objective+' '+method)
    axes[1,0].set(xlabel='Grid intervals',ylabel='Mean relative closure error',title='Paired trained teachers, 1% velocity noise')
    axes[1,0].legend(fontsize=7)
    counter=read_csv(output/'controls/monotonicity.csv')
    axes[1,1].loglog([float(r['compression_error']) for r in counter],[float(r['bir'])-1 for r in counter],'o-',label='Constructed counterexample')
    axes[1,1].set(xlabel='Compression L2 error',ylabel='BIR minus one',title='Derivative minimum = -0.9 in every control')
    axes[1,1].legend(fontsize=8)
    fig.savefig(folder/'mechanisms.png',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,(stage,objective) in zip(axes.ravel(),[(s,o) for s in ('adam1500','lbfgs') for o in ('differential','integral')],strict=True):
        base=output/('curves' if stage=='adam1500' else 'continuation_curves')
        arrays=[]
        for seed in range(5):
            with np.load(base/f'n64_noise0.01_seed{seed}_{objective}.npz') as values:
                arrays.append({k:values[k].copy() for k in values.files})
        grid=arrays[0]['stress']
        for method,color,label in [('neural','tab:blue','Neural mean'),('odd_cubic_frozen','tab:orange','Frozen cubic mean'),('odd_cubic_observation','tab:green','Observation-fit mean')]:
            values=np.array([a[method] for a in arrays])
            for individual in values:
                ax.plot(grid,individual,color=color,lw=.6,alpha=.22)
            ax.plot(grid,values.mean(axis=0),color=color,label=label)
        ax.plot(grid,arrays[0]['truth'],'k--',label='Truth')
        ax.axvline(1,color='gray',ls=':')
        ax.set(xlabel='Shear stress',ylabel='Shear rate',title=stage+' / '+objective+' / all five seeds')
        ax.legend(fontsize=8)
    fig.suptitle('1% velocity noise: individual curves and means; dotted line = training limit',fontsize=10)
    fig.savefig(folder/'trained_closures.png',dpi=220)
    plt.close(fig)


def readme(output, protocol, records, integrity, summaries):
    text=['# Controlled slit-flow theory and synthetic experiment','',
        '本轮先建立可识别的合成对照。真实工程数据未下载；投稿原稿及当前独立新稿正文均未修改。', '',
        '## 系统与证据范围','',
        '新系统是无滑移、对称、稳态受压缝隙流。未知闭合为剪切应力到剪切速率的逆流变函数 Phi；它是本构黏度关系的逆表示。真函数为 Phi(tau)=tau+0.6 tau^3，速度与流量都有解析式。数据由解析式生成，训练只接收保存的速度、压力和位置，真闭合仅供诊断和评价。', '',
        '该实验提供不同物理结构的一维稳态机制证据，不增加三维、真实材料、壁面滑移或历史依赖的覆盖。固定线性/三次符号家族用于隔离机制，没有运行新的自由结构搜索。', '',
        '原始协议声明三个网格、干净/1%噪声、五种子、求导/积分两种神经目标，共60个教师；零噪声的五种子是同一观测上的网络重启。追加协议覆盖全部20个nx64教师的同观测L-BFGS延续，沿用原目标，未按真误差选种子或停止。80个网络状态中有20个配对延续，不视为80个独立数据集。', '',
        '同一固定符号家族的观测重拟合与直接符号最小二乘在此线性前向模型中是同一个全局估计器。两个教师路线重复展示这项参照，独立记录见 unique_observation_fits.csv，不能双重计算样本量。', '',
        '## 可核验的机制','',
        '- 离散求导：干净速度的中心差分产生精确的0.6 h^2 tau误差，落在三次符号家族内，理想冻结压缩完整保留它；解析观测拟合去掉这项误差。',
        '- 测量噪声：求导后的点方差为sigma^2/(2h^2)。三个网格各20000次噪声抽样核验，最大相对差为'+f"{100*integrity['maximum_monte_carlo_noise_variance_relative_difference']:.3f}%"+'。目标噪声的增长不能直接当作最终闭合误差的增长；linear_noise_gains.csv另算保留空间相关性的系数协方差。',
        '- 符号近似：固定同等幅度误差，改变家族内/家族外方向；真值在三次家族中时移除正交分量。线性家族另显示模型近似余项。',
        '- 工程量：同等L2闭合误差可以明显影响流量，也可以完全不影响流量，因为流量只测到误差对特定权函数的投影。',
        '- 配对归因：同网格、同初始化、同预算的干净/含噪教师，将最终符号误差精确拆成已知数值目标偏差、干净学习残差、算法对噪声的响应和符号改动。四个向量不假设独立，保留交叉项；干净学习残差不被伪装成纯优化误差。见paired_error_decomposition.csv。',
        '- 物理性质：构造的紧凑振荡表达式可保持正响应、压缩误差趋零、BIR趋一，却始终违反本轮规定的增量单调性。它是解析对照，不能说成符号搜索自然生成的失败。流量仍有限，诊断不是求解器爆炸。', '',
        '## 实际训练结果','',
        '以下均为同一nx64观测条件中的五种子平均相对闭合误差。延续与初始教师逐一配对；不把算法完成解释成所有模型达到梯度收敛。', '',
        '| 观测 | 教师目标 | 训练阶段 | 神经误差 | 冻结三次误差 | 同家族观测拟合误差 | 神经单调性违规 | 三次违规 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    lookup={(r['stage'],r['noise'],r['objective'],r['method']):r for r in summaries if r['intervals']==64}
    for noise in (0.,.01):
        for objective in ('differential','integral'):
            for stage in ('adam1500','lbfgs'):
                n=lookup[(stage,noise,objective,'neural')]
                s=lookup[(stage,noise,objective,'odd_cubic_frozen')]
                r=lookup[(stage,noise,objective,'odd_cubic_observation_fit')]
                text.append(f"| {noise:g} | {objective} | {stage} | {n['relative_closure_error_mean']:.6g} | {s['relative_closure_error_mean']:.6g} | {r['relative_closure_error_mean']:.6g} | {n['monotonicity_violations']}/5 | {s['monotonicity_violations']}/5 |")
    continuations=[r for stage,r,_,_ in records if stage=='lbfgs']
    gradient_pass=sum(r['optimizer']['gradient_tolerance_satisfied'] for r in continuations)
    text += ['',f'追加20次优化中，{gradient_pass}/20满足记录的严格L2梯度范数阈值1e-9；完整梯度、迭代、目标与停止预算都保留。PyTorch优化器自身的梯度判断使用最大分量，此处L2指标更严格。', '',
        '## 独立积分核验发现与修正','',
        '第一轮报告核验未通过：尖锐含噪教师的128点流量求积与自适应积分最大差0.0645851。失败检查和全部原始教师、符号系数、曲线及指标保留，未重训练或覆盖。新增审计对全部80个模型状态、三个压力共240项，用分片高阶求积、两级网格加密和带断点的独立自适应积分核验。报告中的流量及神经观测MSE采用通过审计的加密结果；原始1025点闭合误差保留，同时另报加密连续范数，避免混淆评价范数。', '',
        '原始求导/积分训练目标不变。含噪求导延续中，目标拟合改善可以伴随尖锐响应、单调性丢失和明显外推误差；验证积分也需具备相应分辨率。该层数值误差与闭合本身误差分开记录。', '',
        '神经导数违规由1025点网格检出，不能据此声称其余网络在整个连续区间内获证。线性和奇三次表达式的导数最小值在区间端点取得，本例的符号单调性检查可直接代数核验。压力1.25访问训练区间外的应力；该外推结果另列，不能混作插值。', '',
        '## 文件','',
        '- PROTOCOL.json、LBFGS_PROTOCOL.json：固定设计、来源和输入哈希。',
        '- controls/：解析误差方向、离散求导、噪声和物理反例。',
        '- fits/、models/、curves/：60个初始神经教师、两类冻结符号拟合及共享观测参照。',
        '- continuation/、continuation_models/、continuation_curves/：20个配对延续。',
        '- reports/by_seed.csv、summary.csv：全部结果、均值和样本SD；没有删除违规或选最佳重启。',
        '- reports/linear_noise_gains.csv：解析线性估计器的噪声增益。',
        '- reports/mechanisms.png、trained_closures.png：独立图像，无PDF。',
        '- INTEGRITY_CHECKS.json：输入与源代码、模型曲线、独立积分和理论恒等式核验。', '',
        '理论推导单独保存在公开仓库 research/controlled-slit-flow-2026-10-13/THEORY.tex，尚未并入论文。', '',
        '背景资料：[解析Poiseuille流研究](https://arxiv.org/abs/2406.13832)、[近期本构识别研究](https://arxiv.org/html/2510.24673v4)。上述具体多项式偏差、投影、协方差和反例均为此控制模型的直接推导，不将标准数学工具包装成新定理。', '',
        '## 重建报告','',
        '使用根目录保留的虚拟环境，在公开仓库目录运行：', '',
        '    python experiments/report_slit_flow_controls.py --output outputs/controlled-slit-flow-2026-10-13', '']
    (output/'README.md').write_text('\n'.join(text))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol,records,integrity=authenticate(args.output)
    rows,summaries=tabulate(args.output,records)
    linear_noise_gains(args.output,protocol)
    paired_error_decomposition(args.output,protocol,records)
    figures(args.output,protocol,summaries)
    readme(args.output,protocol,records,integrity,summaries)
    write_json(args.output/'provenance/report_source.json',dict(report_sha256=digest(__file__),command=sys.argv))
    print(json.dumps(integrity,indent=2))


if __name__=='__main__':
    main()
