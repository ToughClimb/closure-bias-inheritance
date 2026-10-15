"""Rebuild prose, summary tables and figures from completed saved evaluations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rheology_evaluate import Predictor,processing_reference


def run(output):
    output=Path(output)
    initial=json.loads((output/'EVALUATION_COMPLETE.json').read_text())
    followup=json.loads((output/'complexity/COMPLETE.json').read_text())
    assert initial['status']==followup['status']=='passed'
    old=pd.read_csv(output/'reports/by_model.csv');new=pd.read_csv(output/'complexity/by_model.csv')
    primary=old[(old.arm=='primary_high_shear')&old.stage.isin([0,6000])]
    allprimary=pd.concat([primary,new],ignore_index=True)
    # Equal material/fold weighting; collapse network initialization first.
    fields=['high_shear_relative_l2','high_shear_log_rmse','processing_window_symmetric_difference_celsius','processing_window_falseeligible_celsius']
    perrep=allprimary.groupby(['material','fold','method'])[fields].mean().reset_index()
    perrep.to_csv(output/'reports/final_primary_by_replicate.csv',index=False)
    repeat_summary=perrep.groupby(['material','method'])[fields].agg(['mean','std'])
    repeat_summary.columns=['_'.join(c) for c in repeat_summary.columns]
    repeat_summary.to_csv(output/'reports/material_repeat_summary.csv')
    seed_summary=allprimary.groupby(['material','fold','method'])[fields].std()
    seed_summary.to_csv(output/'reports/initialization_sensitivity.csv')
    overview=perrep.groupby('method')[fields].mean()
    counts=allprimary.groupby('method').agg(model_instances=('model_id','size'),nonmonotone_models=('minimum_sampled_stress_margin',lambda a:int((a<=0).sum())))
    overview=overview.join(counts);overview.to_csv(output/'reports/final_primary_summary.csv')
    summary_lines=['|模型|留出黏度相对 L2|加工参考窗口差 / °C|检测到应力非单调|','|---|---:|---:|---:|']
    selected=['neural','frozen_d2','frozen_d3','frozen_d4','frozen_d6','frozen_d8','direct_d6','direct_d8']
    for method in selected:
        r=overview.loc[method]
        summary_lines.append(f'|{method}|{100*r.high_shear_relative_l2:.2f}%|{r.processing_window_symmetric_difference_celsius:.2f}|{int(r.nonmonotone_models)}/{int(r.model_instances)}|')
    empirical=pd.read_csv(output/'complexity/empirical_training_baseline.csv')
    common=pd.read_csv(output/'complexity/common_support_model_comparison.csv')
    commonmeans=common.groupby(['material','fold','method'])[['relative_l2','common_window_symmetric_difference_celsius']].mean().groupby('method').mean()
    em=empirical[['relative_l2','common_window_symmetric_difference_celsius']].mean()
    refstatus=old[(old.arm=='primary_high_shear')&(old.method=='direct_d2')].reference_crossing_status.value_counts().to_dict()
    f1=pd.read_csv(output/'reports/flow_bands.csv');f2=pd.read_csv(output/'complexity/flow_bands.csv')
    raw=pd.read_csv(output/'observations.csv')
    noise=old[(old.method=='neural')&(old.stage==6000)].groupby('arm')[fields].mean()
    lines=[
'# 真实流变数据：闭合误差与加工参考窗口',
'',
'本结果使用 Beckwith 与 Gupta 的 DLP 树脂/空心玻璃微球复合材料稳态剪切流变数据（Zenodo DOI `10.5281/zenodo.19699588`，CC BY 4.0）。50 个完整测试文件、10 个配方、每个配方 5 次测试；共 12,500 条测量。原始 UTF-16 文件、逐行数值与单位转换全部保留，cP 转 Pa·s 的系数是 0.001。数据归原作者；本目录提供独立建模与诊断。',
'',
f'实际温度为 {raw.temperature.min():.2f}–{raw.temperature.max():.2f} °C，剪切率 {raw.rate.min():.3f}–{raw.rate.max():.1f} s⁻¹。原始记录含 238 个非正黏度值，全部位于剪切率 10 s⁻¹ 以下；源论文指出该区域接近仪器扭矩下限。主分析固定采用实际剪切率 ≥10 s⁻¹，共 {int((raw.rate>=10).sum())} 条记录，没有非正值。全范围正值训练作为附加分支，非正原始记录保留并进入原空间误差。125 °C 的明显黏度上升也保留。',
'',
'训练/验证按整个物理测试文件划分：每个配方用 4 次测试训练，1 次测试留出，轮换 5 次；同一文件所有温度和剪切率始终在同一侧。只用训练数据确定归一化。200 个分别训练的网络来自 10 配方 ×5 折 ×2 初始种子 ×2 数据域；两个预算检查点不是新增独立实验。批量实现具有独立参数，梯度与单独训练的等价性已测试。',
'',
'下面是最终 6000 步、高剪切域的结果。先平均两个网络初始种子，再平均 50 个留出文件。窗口误差是 η≤0.40 Pa·s、γ=20 s⁻¹ 参考窗口与留出测量插值窗口的对称差长度；该黏度阈值来自源论文，是加工比较指标，不能单独保证可打印性。',
'',*summary_lines,'',
'd 表示归一化逆温度与对数剪切率的总多项式次数，黏度为该多项式的指数。最初固定报告 d=1/2/3；观察到较大的压缩逼近误差后，另行冻结并完成 d=4/6/8 敏感性实验。全部次数均报告，没有根据留出误差挑选表达式。直接拟合和固定表达式系数重拟合是同一个凸最小二乘问题，只计一次。',
'',
'这些结果区分了两种问题：低次数表达式的逼近能力不足，不能称作忠实压缩下的偏差继承；提高次数后误差和窗口预测改善，但应力导数的可靠性问题增多。指数参数化确保黏度为正，却不保证 d(γη)/dγ>0。这里的失败来自采样点上的明确负导数；通过有限网格检查不等于连续域证书。误差用已留出的真实测量作为参照，不计算未知真实材料定律的 BIR。',
'',
f'只用四次训练测量的插值参照也已完成。在共同支持的 {int(empirical.n_eligible.sum())}/{int(empirical.n_total.sum())} 个留出高剪切点上，其平均相对 L2 为 {100*em.relative_l2:.2f}%，神经网络在完全相同的点上为 {100*commonmeans.loc["neural","relative_l2"]:.2f}%。共同温度支持上的窗口差分别为 {em.common_window_symmetric_difference_celsius:.2f} °C 和 {commonmeans.loc["neural","common_window_symmetric_difference_celsius"]:.2f} °C。这是同一设计网格上的新物理样本预测，不能当作未见工况外推或新发现材料机理的证明。',
'',
f'包含低剪切正值的训练分支，最终高剪切相对 L2 为 {100*noise.loc["full_positive","high_shear_relative_l2"]:.2f}%，主分支为 {100*noise.loc["primary_high_shear","high_shear_relative_l2"]:.2f}%；两者窗口误差分别为 {noise.loc["full_positive","processing_window_symmetric_difference_celsius"]:.2f} °C 和 {noise.loc["primary_high_shear","processing_window_symmetric_difference_celsius"]:.2f} °C。此比较反映在已声明目标和预算下使用仪器受限测量区域的影响。',
'',
f'50 条留出 γ=20 s⁻¹ 温度曲线中，交叉状态为 {refstatus}。因此主要窗口指标覆盖全部文件；单一交叉温度误差只作补充。γ=20 不是原始采样点，采用每次测试自身相邻剪切率的 log–log 插值。交叉集合指 10 个测量温度节点上的分段对数插值；连续模型的窗口差另在 1001 点网格计算，约 0.09 °C 分辨率。',
'',
'## 条件化的狭缝流动后果',
'',
'另在名义 35/75/125 °C、半间隙 0.5 mm 下计算固定压力梯度的狭缝流动后果。实际查询温度和壁面应力只由四次训练测量决定，剪切率支持取四次训练曲线的交集。对观测不到的低剪切中心区域，只在单调逆本构延拓假设下给出上下界，不外推一个假定中心黏度。这个区间不是统计置信区间。',
'',
f'最初的 NN、冻结 d2 与直接 d2 共 750 个条件：{initial["flow_status_counts"]}。固定 d8 随后又检查 450 个条件：{followup["flow_status_counts"]}。不支持指定壁面应力或应力非单调的案例完整保留；没有通过排序应力或裁剪来恢复反演。源数据没有流量实测，所以这些是依赖稳态、广义牛顿与无滑移假设的 PDE 预测敏感性，而实测验证的终点是黏度及其加工参考窗口。',
'',
f'最初固定 128 点 Gauss 在 3 个 NN 条件上未达到独立积分检查的要求，失败表保存在 `reports_attempt1/`。只提高积分分辨率后，最终相对差最大 {initial["max_flow_relative_quadrature_difference"]:.3g}；d8 最大 {followup["max_flow_relative_quadrature_difference"]:.3g}。没有重新训练或重新选择模型。',
'',
'## 复核与文件',
'',
'- `PROTOCOL.json`：训练前协议；`COMPLEXITY_FOLLOWUP_PROTOCOL.json`：观察首轮结果后冻结的扩充协议。',
'- `observations.csv`：保留原文件、原始行号、测试 ID、实际温度与状态字段的全部数据。',
'- `reports/by_model.csv`、`complexity/by_model.csv`：全部次数、文件、种子、预算和失败指标。',
'- `reports/material_repeat_summary.csv`：每个配方内五次留出测试的均值与描述性标准差；`initialization_sensitivity.csv`：单独记录两个网络初始种子的差异。五折训练集合互相重叠，没有用温度点数或种子数增加独立样本量。',
'- `reports/heldout_predictions.npz`：首轮全部保存预测及逐行留出标识；系数和模型权重均保存。',
'- `complexity/empirical_training_baseline.csv`、`common_support_model_comparison.csv`：插值参照、支持覆盖率及公平的共同点比较。',
'- `reports/flow_bands.csv`、`complexity/flow_bands.csv`：每个条件的支持区间、状态、上下界和独立积分差。',
'- `models/`、`training_specs.json`、`training_arrays.npz`：复现拟合与整个文件分割。',
'- `provenance/`：环境、源代码、日志、测试和保留的失败。',
'',
'结果图：`reports/fidelity_and_reliability.png`、`reports/processing_examples.png`。本补充未修改论文正文，也未推送仓库。',
'',
'只重建报告与图，不重训：',
'```bash',
'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/rheology_report.py --output outputs/rheology-engineering-2026-10-13',
'```',
'',
'来源：[原始数据](https://zenodo.org/records/19699588)，[数据说明论文](https://www.nature.com/articles/s41597-026-07391-0)，[数据许可元数据](https://api.datacite.org/dois/10.5281/zenodo.19699588)。',
]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    degrees=np.array([1,2,3,4,6,8]);fig,axes=plt.subplots(1,3,figsize=(12.5,3.6))
    for prefix,label,color in [('frozen','Frozen compression','tab:blue'),('direct','Direct / coefficient refit','tab:orange')]:
        names=[f'{prefix}_d{d}' for d in degrees]
        axes[0].plot(degrees,[overview.loc[n,'high_shear_log_rmse'] for n in names],'o-',label=label,color=color)
        axes[1].plot(degrees,[overview.loc[n,'processing_window_symmetric_difference_celsius'] for n in names],'o-',color=color)
        axes[2].plot(degrees,[100*overview.loc[n,'nonmonotone_models']/overview.loc[n,'model_instances'] for n in names],'o-',color=color)
    for ax,key in zip(axes,['high_shear_log_rmse','processing_window_symmetric_difference_celsius',None]):
        y=overview.loc['neural',key] if key else 100*overview.loc['neural','nonmonotone_models']/overview.loc['neural','model_instances']
        ax.axhline(y,color='black',ls='--',lw=1,label='Neural teacher' if ax is axes[0] else None);ax.set_xlabel('Total polynomial degree');ax.set_xticks(degrees);ax.grid(alpha=.2)
    axes[0].set_ylabel('Held-out log-viscosity RMSE');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Window difference (degrees C)');axes[2].set_ylabel('Sampled stress violations (%)')
    fig.tight_layout();fig.savefig(output/'reports/fidelity_and_reliability.png',dpi=180);plt.close(fig)
    specs=json.loads((output/'training_specs.json').read_text());ck=dict(np.load(output/'models/checkpoint_6000.npz'))
    modeldefs=json.loads((output/'reports/model_definitions.json').read_text())+json.loads((output/'complexity/model_definitions.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,7));chosen=['Neat','0_13gcc_10%','0_23gcc_20%','0_31gcc_40%']
    for ax,material in zip(axes.ravel(),chosen):
        reference=processing_reference(raw[(raw.material==material)&(raw.replicate==0)])
        tt=np.linspace(reference[0,1],reference[-1,1],1001)
        ax.plot(reference[:,1],reference[:,2],'ko',ms=3,label='Held-out run (rate interpolation)')
        for method,color in [('neural','black'),('frozen_d2','tab:blue'),('frozen_d8','tab:purple'),('direct_d8','tab:orange')]:
            matching=[m for m in modeldefs if m['arm']=='primary_high_shear' and m['material']==material and m['fold']==0 and m['method']==method and m['stage'] in (0,6000) and m['seed'] in (-1,0)]
            assert len(matching)==1
            m=matching[0];spec=specs[m['spec_index']];w=None if m['coefficients'] is not None else {n:ck[n][spec['index']] for n in ('w0','b0','w2','b2','w4','b4')}
            predictor=Predictor(spec,weights=w,coefficients=m['coefficients'],degree=m['degree'])
            ax.plot(tt,np.exp(predictor(tt,np.full_like(tt,20.))),color=color,label=method)
        ax.axhline(.4,color='gray',ls=':');ax.set_yscale('log');ax.set_title(material);ax.set_xlabel('Temperature (degrees C)');ax.set_ylabel('Viscosity (Pa s)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('Fixed fold 0, seed 0; shear rate 20/s');fig.tight_layout();fig.savefig(output/'reports/processing_examples.png',dpi=180);plt.close(fig)
    tracked=[output/'reports/by_model.csv',output/'complexity/by_model.csv',output/'reports/flow_bands.csv',output/'complexity/flow_bands.csv',output/'complexity/empirical_training_baseline.csv']
    (output/'REPORT_NUMERICAL_SOURCES.json').write_text(json.dumps({str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
