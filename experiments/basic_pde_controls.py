"""Analytical and linear-inverse calibration controls for five basic PDEs.

These are mechanism controls, not five new neural-symbolic campaigns. They
keep exact continuum dynamics, finite-difference inverse bias, observation
noise, rank deficiency, and flux gauge freedom separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,record):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def linear_fields(x,t,*,diffusion=0.,speed=0.,reaction=0.,modes=(1,),amplitudes=None):
    """Exact real Fourier solution and independent analytical derivatives."""
    x=np.asarray(x,dtype=float)
    t=np.asarray(t,dtype=float)
    amplitudes=amplitudes or [1/len(modes)]*len(modes)
    if len(amplitudes)!=len(modes):raise ValueError('One amplitude per mode required')
    result={key:np.zeros((len(t),len(x))) for key in ('u','ut','ux','uxx')}
    for mode,amplitude in zip(modes,amplitudes,strict=True):
        k=2*np.pi*mode
        rate=reaction-diffusion*k*k
        argument=k*x[None,:]-speed*k*t[:,None]
        scale=amplitude*np.exp(rate*t[:,None])
        u=scale*np.cos(argument)
        ux=-k*scale*np.sin(argument)
        result['u']+=u
        result['ux']+=ux
        result['uxx']-=k*k*u
        result['ut']+=rate*u-speed*ux
    return result


def stencil_fields(u,dx,dt):
    middle=u[1:-1]
    return dict(ut=(u[2:]-u[:-2])/(2*dt),
                ux=(np.roll(middle,-1,axis=-1)-np.roll(middle,1,axis=-1))/(2*dx),
                uxx=(np.roll(middle,-1,axis=-1)-2*middle+np.roll(middle,1,axis=-1))/dx**2,
                u=middle)


def linear_fit(fields,unknowns):
    design=np.column_stack([{'D':fields['uxx'],'c':-fields['ux'],'a':fields['u']}[name].ravel() for name in unknowns])
    target=fields['ut'].ravel()
    scales=np.linalg.norm(design,axis=0)
    if np.any(scales==0):raise ValueError('An unexcited coefficient must be recorded explicitly')
    normalized=design/scales
    singular=np.linalg.svd(normalized,compute_uv=False)
    rank=int(np.linalg.matrix_rank(normalized))
    if rank<len(unknowns):
        return dict(status='unidentifiable',rank=rank,unknowns=list(unknowns),singular_values=singular.tolist(),coefficients=None)
    coefficients,_,_,_=np.linalg.lstsq(normalized,target,rcond=None)
    coefficients=coefficients/scales
    return dict(status='identified',rank=rank,unknowns=list(unknowns),singular_values=singular.tolist(),
                coefficients={name:float(value) for name,value in zip(unknowns,coefficients,strict=True)},
                residual_rms=float(np.sqrt(np.mean((design@coefficients-target)**2))))


def clean_discrete_coefficients(nx,dt,*,diffusion=0.,speed=0.,reaction=0.,mode=1):
    k=2*np.pi*mode
    h=1/nx
    q2=4*np.sin(k*h/2)**2/h**2
    p=np.sin(k*h)/h
    temporal=np.sinh((reaction-diffusion*k*k-1j*speed*k)*dt)/dt
    if reaction!=0:raise ValueError('D and reaction are confounded on a single mode')
    return dict(D=float(-temporal.real/q2),c=float(-temporal.imag/p))


def advection_phase_error(delta_speed,time,mode=1):
    return 2*np.abs(np.sin(delta_speed*(2*np.pi*mode)*np.asarray(time)/2))


def burgers_fields(x,t,*,viscosity=.02,mean=.25,jump_half=.5,flux_linear=0.):
    """Exact viscous shock on the real line, common initial and far-field states.

    Flux is F(u)=u^2/2+flux_linear*u+C. The arbitrary constant C drops
    out of the PDE entirely and is deliberately not an input to this solution.
    """
    x=np.asarray(x,dtype=float)
    t=np.asarray(t,dtype=float)
    speed=mean+flux_linear
    z=jump_half*(x[None,:]-speed*t[:,None])/(2*viscosity)
    tangent=np.tanh(z)
    sech_squared=1-tangent*tangent
    u=mean-jump_half*tangent
    ux=-jump_half**2*sech_squared/(2*viscosity)
    uxx=jump_half**3*sech_squared*tangent/(2*viscosity**2)
    return dict(u=u,ux=ux,uxx=uxx,ut=-speed*ux,shock_center=speed*t)


def plan():
    return dict(systems=['heat','advection','advection_diffusion','linear_reaction_diffusion','viscous_burgers'],
        interpretation='Analytical/linear-inverse calibration and counterexamples; no new neural fitting or free symbolic search',
        periodic_length=1.,mesh_intervals=[16,32,64],time_samples=21,time_end=.1,
        heat_diffusion=.03,transport_speed=.4,reaction_rate=.8,
        noise_sigma=.01,noise_seeds=list(range(5)),draws_per_seed=400,
        noise_controls=['target_only','state_only','joint'],noise_clipping=False,
        coefficient_fit='Scaled ordinary least squares; rank deficient designs have no reported unique coefficient estimate',
        reaction_modes=[[1],[1,2],[0,1]],advection_speed_errors=[.001,.01,.05],advection_test_horizon=50.,
        burgers=dict(viscosity=.02,mean=.25,jump_half=.5,flux_linear_errors=[0.,.002,.01,.05],
                     flux_constant_errors=[0.,.1,1.],domain='Real-line exact shock, observed on[-2,2], common initial/far-field states'),
        no_truth_selection='Parameters and modes fixed before executing controls; truth used only by construction and evaluation')


def declare(output):
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(plan=plan(),source_sha256=digest(__file__))
    path=output/'PROTOCOL.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Existing protocol/source differs')
    if not path.exists():
        write_json(path,protocol)
        shutil.copy2(__file__,output/'source.py')
        write_json(output/'ENVIRONMENT.json',dict(command=sys.argv,python=sys.version,
            packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','matplotlib','pytest']},
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],cwd=SOURCE,text=True),
            original_core_sha256={str(p.relative_to(SOURCE)):digest(p) for p in sorted((SOURCE/'src').rglob('*.py'))+sorted((SOURCE/'revision').glob('*.py'))}))
    return protocol


def deterministic_controls(output,p):
    rows=[]
    maximum_residual=0.
    for nx in p['mesh_intervals']:
        x=np.arange(nx)/nx
        t=np.linspace(0,p['time_end'],p['time_samples'])
        dt=t[1]-t[0]
        for system,D,c,unknowns in [('heat',.03,0.,('D',)),('advection',0.,.4,('c',)),('advection_diffusion',.03,.4,('D','c'))]:
            exact=linear_fields(x,t,diffusion=D,speed=c)
            continuous={k:v[1:-1] for k,v in exact.items()}
            oracle=linear_fit(continuous,unknowns)
            sampled=linear_fit(stencil_fields(exact['u'],1/nx,dt),unknowns)
            predicted=clean_discrete_coefficients(nx,dt,diffusion=D,speed=c)
            residual=np.max(np.abs(exact['ut']-D*exact['uxx']+c*exact['ux']))
            maximum_residual=max(maximum_residual,float(residual))
            for name in unknowns:
                reference={'D':D,'c':c}[name]
                rows.append(dict(system=system,intervals=nx,coefficient=name,truth=reference,
                    exact_derivative_fit=oracle['coefficients'][name],finite_difference_fit=sampled['coefficients'][name],
                    predicted_discrete_fit=predicted[name],relative_discretization_error=abs(sampled['coefficients'][name]-reference)/abs(reference)))
            np.savez_compressed(output/f'{system}_n{nx}.npz',x=x,t=t,u=exact['u'])
    write_csv(output/'reports/discrete_coefficients.csv',rows)
    coefficient_errors=[abs(r['finite_difference_fit']-r['predicted_discrete_fit']) for r in rows]
    if max(coefficient_errors)>1e-11:raise ValueError('Finite-difference coefficient predictions fail')

    identifiability=[]
    nx=64;x=np.arange(nx)/nx;t=np.linspace(0,.5,51)
    for modes in p['reaction_modes']:
        exact=linear_fields(x,t,diffusion=.03,reaction=.8,modes=tuple(modes))
        fit=linear_fit(exact,('D','a'))
        identifiability.append(dict(modes=modes,**fit))
    if identifiability[0]['rank']!=1 or identifiability[1]['rank']!=2 or identifiability[2]['rank']!=2:
        raise ValueError('Excitation-rank controls fail')
    write_json(output/'reports/identifiability.json',identifiability)
    compensation=[]
    k=2*np.pi
    for delta in (.01,-.04):
        D=.03+delta;a=.8+delta*k*k
        training_true=linear_fields(x,t,diffusion=.03,reaction=.8,modes=(1,))['u']
        training_other=linear_fields(x,t,diffusion=D,reaction=a,modes=(1,))['u']
        test_true=linear_fields(x,t,diffusion=.03,reaction=.8,modes=(2,))['u']
        test_other=linear_fields(x,t,diffusion=D,reaction=a,modes=(2,))['u']
        compensation.append(dict(diffusion=D,reaction=a,training_mode1_max_error=float(np.max(np.abs(training_true-training_other))),
            mode2_relative_endpoint_error=float(np.linalg.norm(test_other[-1]-test_true[-1])/np.linalg.norm(test_true[-1])),
            diffusion_positive=D>0,mode2_true_growth_rate=.8-.03*(2*k)**2,mode2_candidate_growth_rate=a-D*(2*k)**2))
    write_csv(output/'reports/parameter_compensation.csv',compensation)

    times=np.linspace(0,50,1001)
    phase=[]
    for delta in p['advection_speed_errors']:
        errors=advection_phase_error(delta,times)
        phase.append(dict(speed_error=delta,maximum_relative_field_error=float(errors.max()),
                          first_time_for_relative_error_two=1/(2*delta),both_solutions_conserve_mass=True,
                          both_solutions_conserve_L2=True))
    write_csv(output/'reports/advection_phase.csv',phase)
    np.savez_compressed(output/'advection_phase.npz',t=times,**{'delta'+str(delta):advection_phase_error(delta,times) for delta in p['advection_speed_errors']})

    x=np.linspace(-2,2,4097);t=np.array([0.,1.])
    true=burgers_fields(x,t)
    burgers=[]
    for delta in p['burgers']['flux_linear_errors']:
        fields=burgers_fields(x,t,flux_linear=delta)
        residual=fields['ut']+(fields['u']+delta)*fields['ux']-.02*fields['uxx']
        maximum_residual=max(maximum_residual,float(np.max(np.abs(residual))))
        burgers.append(dict(flux_linear_error=delta,shock_speed=.25+delta,shock_position_error_at_t1=delta,
            initial_condition_max_difference=float(np.max(np.abs(fields['u'][0]-true['u'][0]))),
            relative_endpoint_field_error=float(np.linalg.norm(fields['u'][-1]-true['u'][-1])/np.linalg.norm(true['u'][-1])),
            analytical_pde_residual_max=float(np.max(np.abs(residual)))))
    write_csv(output/'reports/burgers_shock.csv',burgers)
    write_csv(output/'reports/flux_gauge.csv',[dict(flux_constant=constant,absolute_flux_function_error=abs(constant),
        pde_operator_difference=0.,shock_position_difference=0.,gauge_fix='Set F(0)=0 before flux-error comparison')
        for constant in p['burgers']['flux_constant_errors']])
    if maximum_residual>1e-11:raise ValueError('Analytical continuum PDE residual fails')
    return dict(maximum_analytical_pde_residual=maximum_residual,
                maximum_discrete_coefficient_prediction_error=max(coefficient_errors),single_mode_rank=1,two_mode_rank=2)


def noise_controls(output,p):
    rows=[]
    sigma=p['noise_sigma']
    for nx in p['mesh_intervals']:
        x=np.arange(nx)/nx;t=np.linspace(0,.1,21);dt=t[1]-t[0]
        exact=linear_fields(x,t,diffusion=.03)
        clean=stencil_fields(exact['u'],1/nx,dt)
        X,Y=clean['uxx'],clean['ut']
        A=float(np.mean(X*X));M=float(np.mean(X*Y))
        scalar_clean=M/A
        input_noise_variance=6*sigma*sigma*nx**4
        target_noise_variance=sigma*sigma/(2*dt*dt)
        populations={'target_only':scalar_clean,'state_only':M/(A+input_noise_variance),'joint':M/(A+input_noise_variance)}
        draws={name:[] for name in populations}
        moment={name:[] for name in populations}
        for seed in p['noise_seeds']:
            noise=np.random.default_rng(seed).normal(scale=sigma,size=(p['draws_per_seed'],len(t),nx))
            target_noise=(noise[:,2:]-noise[:,:-2])/(2*dt)
            middle=noise[:,1:-1]
            input_noise=(np.roll(middle,-1,axis=-1)-2*middle+np.roll(middle,1,axis=-1))*nx**2
            for name in populations:
                observedX=X[None]+(0 if name=='target_only' else input_noise)
                observedY=Y[None]+(0 if name=='state_only' else target_noise)
                denominator=np.mean(observedX**2,axis=(-2,-1))
                numerator=np.mean(observedX*observedY,axis=(-2,-1))
                estimates=numerator/denominator
                # In target-only the fixed design denominator is scalar.
                draws[name].extend(np.ravel(estimates).tolist())
                moment[name].append((float(np.mean(numerator)),float(np.mean(denominator))))
        for name,estimates in draws.items():
            estimates=np.asarray(estimates)
            expected_design=A+(0 if name=='target_only' else input_noise_variance)
            measured_cross=np.mean([r[0] for r in moment[name]])
            measured_design=np.mean([r[1] for r in moment[name]])
            moment_minimizer=measured_cross/measured_design
            rows.append(dict(intervals=nx,noise_location=name,draws=len(estimates),sigma=sigma,clean_discrete_coefficient=scalar_clean,
                expected_objective_minimizer=populations[name],monte_carlo_objective_minimizer=moment_minimizer,
                finite_sample_estimator_mean=float(estimates.mean()),finite_sample_estimator_sd=float(estimates.std(ddof=1)),
                negative_coefficient_count=int(np.sum(estimates<=0)),theoretical_input_noise_variance=input_noise_variance,
                theoretical_target_noise_variance=target_noise_variance,design_second_moment_theory=expected_design,
                design_second_moment_empirical=measured_design,
                relative_population_check_error=abs(moment_minimizer/populations[name]-1),
                interpretation='Population minimizer of expected objective differs from expectation of finite-sample OLS ratio'))
    write_csv(output/'reports/noise_location.csv',rows)
    maximum=max(r['relative_population_check_error'] for r in rows)
    if maximum>.03:raise ValueError('Noise-location population prediction differs by more than3%; raw results retained')
    return dict(maximum_noise_population_relative_difference=maximum,noise_draws_per_grid=p['draws_per_seed']*len(p['noise_seeds']))


def figure(output):
    with (output/'reports/noise_location.csv').open() as stream:noise=list(csv.DictReader(stream))
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for name in ('target_only','state_only','joint'):
        rows=[r for r in noise if r['noise_location']==name]
        axes[0,0].plot([int(r['intervals']) for r in rows],[float(r['finite_sample_estimator_mean']) for r in rows],'o-',label=name.replace('_',' '))
    axes[0,0].axhline(.03,color='black',ls=':',label='True diffusion')
    axes[0,0].set(xlabel='Grid intervals',ylabel='Mean fitted diffusion',title='Separate target and constitutive-input noise')
    axes[0,0].legend(fontsize=8)
    with np.load(output/'advection_phase.npz') as phase:
        for name in phase.files:
            if name!='t':axes[0,1].plot(phase['t'],phase[name],label=name.replace('delta','Speed error '))
    axes[0,1].set(xlabel='Time',ylabel='Relative field error',title='Phase error despite exact mass and L2 conservation')
    axes[0,1].legend(fontsize=8)
    x=np.linspace(-2,2,2049);t=np.array([1.])
    for delta in (0.,.01,.05):axes[1,0].plot(x,burgers_fields(x,t,flux_linear=delta)['u'][0],label='Flux slope error '+str(delta))
    axes[1,0].set(xlim=(.1,.45),xlabel='Position',ylabel='Velocity',title='Same viscous profile, shifted shock position')
    axes[1,0].legend(fontsize=8)
    time=np.linspace(0,.5,101);k=2*np.pi
    for name,D,a in [('Truth',.03,.8),('Wrong positive D',.04,.8+.01*k*k),('Negative D',-.01,.8-.04*k*k)]:
        axes[1,1].plot(time,np.exp((a-D*(2*k)**2)*time),label=name)
    axes[1,1].set(xlabel='Time',ylabel='Mode2 amplitude',title='Identical mode1 training trajectory, different test mode')
    axes[1,1].legend(fontsize=8)
    fig.savefig(output/'reports/mechanisms.png',dpi=220)
    plt.close(fig)


def report(output,p,checks):
    lines=['# 基础PDE的可控机制分析','',
        '这一轮是解析与线性反演校准。五类系统没有被包装成五套完整神经符号训练；真实工程数据未下载，论文正文未改。','',
        '| 系统 | 独立检查的机制 | 结果文件 |','| --- | --- | --- |',
        '| 热方程 | 衰减率；空间/时间差分偏差；输入噪声改变最小二乘目标 | discrete_coefficients.csv、noise_location.csv |',
        '| 线性输运 | 速度偏差累积成相位误差；守恒量可全部通过 | advection_phase.csv |',
        '| 对流—扩散 | 实部控制衰减、虚部控制相位；两种数值系数偏差分离 | discrete_coefficients.csv |',
        '| 线性反应—扩散 | 单模态只识别a-Dk²；二模态/均匀模态破除补偿 | identifiability.json、parameter_compensation.csv |',
        '| 黏性Burgers | 通量加常数是不可辨识规范；线性通量误差改变激波速度 | burgers_shock.csv、flux_gauge.csv |','',
        '## 关键解释','',
        '单模态反应—扩散控制使用平衡态附近可正可负的扰动变量，不把它当成非负浓度。一个保持正扩散但参数错误的模型，以及一个负扩散模型，都能逐点匹配训练模态。第二空间模态或零波数模态使归一化设计矩阵从秩1变为秩2。这是当前激励下的明确不可辨识性，不是用状态覆盖率代替一般逆唯一性证明。','',
        '热方程的噪声位置控制只改变指定的一项：目标、状态设计矩阵或两者。每个网格使用五个固定噪声种子，各400次抽样，无裁剪。输入噪声带来的扩散收缩可以直接由期望目标推导；目标噪声在此常系数参照中增加方差。期望目标的最优值与有限样本估计器平均值分别报告，不把两个数学量混称。所有非正系数估计保留。','',
        '输运的平均值和L2范数可以完全守恒，速度误差仍会产生明显的相位误差。Burgers通量的常数偏移不会改变任何解，因此比较通量函数误差前需要约定F(0)=0。线性通量偏差b u使本轮解析黏性激波的速度增加b；共同初值与无穷远状态不变。这里观察的是实线上精确解的有限窗口，不声称做了有限区间边界求解。','',
        '## 核验','',
        f"解析PDE残差最大 {checks['maximum_analytical_pde_residual']:.3g}；有限差分系数预测误差最大 {checks['maximum_discrete_coefficient_prediction_error']:.3g}。",'',
        f"噪声期望目标的理论/Monte Carlo最优系数最大相对差 {100*checks['maximum_noise_population_relative_difference']:.3g}%。",'',
        '全部条件事先固定，源文件、环境、数据哈希与既有代码哈希保留。这里只执行最小校准控制；主贡献仍由非线性闭合研究及独立可靠性证据承担。','',
        '推导：公开仓库research/basic-pde-controls-2026-10-13/THEORY.tex。图像：reports/mechanisms.png。无PDF。','',
        '背景：[Clawpack线性输运教材](https://www.clawpack.org/riemann_book/html/Advection.html)、[Burgers教材](https://www.clawpack.org/riemann_book/html/Burgers.html)。本包的误差、矩阵和解析波核验由源码直接计算。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol=declare(args.output)
    if (args.output/'INTEGRITY_CHECKS.json').exists():raise ValueError('Completed controls already exist; keep results and use a new output directory')
    (args.output/'reports').mkdir(exist_ok=True)
    checks=deterministic_controls(args.output,protocol['plan'])
    checks.update(noise_controls(args.output,protocol['plan']))
    environment=json.loads((args.output/'ENVIRONMENT.json').read_text())
    for relative,sha in environment['original_core_sha256'].items():
        if digest(SOURCE/relative)!=sha:raise ValueError('Original core changed')
    figure(args.output)
    report(args.output,protocol['plan'],checks)
    checks.update(status='passed',original_core_unchanged=True,
                  source_sha256=digest(__file__),output_sha256={str(path.relative_to(args.output)):digest(path)
                  for path in sorted(args.output.glob('*.npz'))+sorted((args.output/'reports').glob('*.csv'))+sorted((args.output/'reports').glob('*.json'))})
    write_json(args.output/'INTEGRITY_CHECKS.json',checks)
    print(json.dumps({k:v for k,v in checks.items() if k!='output_sha256'},indent=2))


if __name__=='__main__':main()
