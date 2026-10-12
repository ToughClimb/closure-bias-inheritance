"""Regression tests for the reviewed numerical and experiment-integrity defects."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import csv
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import revision_core as core
from closure_discovery.data_generation.observations import apply_observation_model
from closure_discovery.data_generation.validation import uniform_spacing
from closure_discovery.models.mlp_closure import KANScalarNet
from closure_discovery.pipelines.train_1d_closure import UnseenRolloutConfig
from closure_discovery.pipelines.train_2d_closure import apply_observation_model_2d, ObservationConfig2D
from closure_discovery.pipelines.training_state import assert_finite_state
from closure_discovery.symbolic.restricted_fit import SymbolicExpression, fit_rational_family
from closure_discovery.weak_form.weak_residual import weak_residual_space_time_1d

REPO=Path(__file__).resolve().parents[1]


def experiment(name):
    spec=importlib.util.spec_from_file_location(name,REPO/'experiments'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


@pytest.fixture
def small_observations():
    case=core.extra_case('case_exp')
    raw=core.generate_dataset(case,core.SimulationConfig1D(nx=16,t_final=.005,save_every=5),2,
        seed=2,amplitude_range=case.value_range,initial_clip_range=case.value_range)
    return raw,case.value_range


@pytest.mark.parametrize('degrees',[(2,1),(1,2),(0,1),(2,0),(0,0),(2,2)])
def test_rational_layout_matches_fitted_loss_and_serialization(degrees):
    numerator,denominator=degrees;x=np.linspace(0,1.8,101)
    top=np.arange(1,numerator+2)*.2;bottom=np.arange(1,denominator+1)*.3
    y=np.polynomial.polynomial.polyval(x,top)/np.polynomial.polynomial.polyval(x,np.r_[1.,bottom])
    fitted=fit_rational_family(x,y,numerator_degree=numerator,denominator_degree=denominator,ridge=0)[0]
    restored=SymbolicExpression(**json.loads(json.dumps(asdict(fitted))))
    actual=float(np.mean((restored.evaluate(x)-y)**2))
    assert np.isclose(actual,fitted.fit_mse,rtol=1e-8,atol=1e-30)
    assert actual<1e-24


def test_legacy_rational_default_is_unambiguous():
    x=np.array([0.,.3,1.]);parameters=(.1,.2,.3,.4,.5)
    legacy=SymbolicExpression('rational',parameters,'',5,0)
    explicit=SymbolicExpression('rational',parameters,'',5,0,2,2)
    assert np.array_equal(legacy.evaluate(x),explicit.evaluate(x))
    with pytest.raises(ValueError,match='degree metadata'):
        SymbolicExpression('rational',parameters[:4],'',4,0).evaluate(x)


@pytest.mark.parametrize('shape',[(),(2,1),(1,),(2,3,1),(2,3)])
def test_kan_preserves_axes_and_independent_derivatives(shape):
    net=KANScalarNet((0.,1.));net.base_weight.data.fill_(2.)
    u=torch.full(shape,.4,requires_grad=True);result=net(u)
    assert result.shape==u.shape
    gradient=torch.autograd.grad(result.flatten()[0],u)[0]
    assert gradient.flatten()[0]==4.
    assert torch.count_nonzero(gradient)==1


def grid_data(two_dimensions=False):
    shape=(2,9,64,32) if two_dimensions else (2,9,64)
    u=np.ones(shape);data=dict(x=np.arange(64)/64,t=np.linspace(0,.08,9),u=u,u0=u[:,0].copy())
    if two_dimensions:data['y']=np.arange(32)/32
    return data


@pytest.mark.parametrize('two_dimensions',[False,True])
@pytest.mark.parametrize('stride',[3,5,64,0,1.5,True])
def test_invalid_periodic_subsampling_is_rejected(two_dimensions,stride):
    data=grid_data(two_dimensions)
    with pytest.raises(ValueError):
        if two_dimensions:apply_observation_model_2d(data,ObservationConfig2D(space_stride=stride),seed=0,clip_range=(0.,1.8))
        else:apply_observation_model(data,space_stride=stride)


@pytest.mark.parametrize('two_dimensions',[False,True])
def test_valid_subsampling_remains_uniform(two_dimensions):
    data=grid_data(two_dimensions)
    observed=(apply_observation_model_2d(data,ObservationConfig2D(space_stride=4,time_stride=2),seed=0,clip_range=(0.,1.8))
              if two_dimensions else apply_observation_model(data,space_stride=4,time_stride=2))
    assert observed['u'].shape[1]==5
    assert uniform_spacing(observed['x'],'x')==4/64
    if two_dimensions:assert uniform_spacing(observed['y'],'y')==4/32
    with pytest.raises(ValueError,match='three saved times'):
        apply_observation_model(data,time_stride=5)


@pytest.mark.parametrize('noise',[float('nan'),float('inf'),-.1])
def test_nonfinite_or_negative_noise_is_rejected(noise):
    with pytest.raises(ValueError):apply_observation_model(grid_data(),noise_level=noise)


def test_adam_overflow_is_detected_even_with_finite_parameters():
    net=torch.nn.Linear(1,1);opt=torch.optim.Adam(net.parameters())
    net(torch.ones(1,1)).sum().backward();opt.step()
    assert_finite_state(net,opt)
    next(iter(opt.state.values()))['exp_avg_sq'].fill_(float('inf'))
    with pytest.raises(FloatingPointError,match='optimizer state'):assert_finite_state(net,opt)


def test_bad_gradient_fails_before_optimizer_step():
    net=torch.nn.Linear(1,1);net.weight.grad=torch.full_like(net.weight,float('inf'))
    with pytest.raises(FloatingPointError,match='gradient'):assert_finite_state(net,gradients=True)


def test_training_never_commits_overflowed_adam_state(small_observations,tmp_path,monkeypatch):
    original=torch.optim.Adam.step
    def overflow(opt,*args,**kwargs):
        value=original(opt,*args,**kwargs)
        next(iter(opt.state.values()))['exp_avg_sq'].fill_(float('inf'))
        return value
    monkeypatch.setattr(torch.optim.Adam,'step',overflow)
    raw,value_range=small_observations
    with pytest.raises(FloatingPointError,match='optimizer state'):
        core.train(raw,value_range,epochs=2,out_dir=tmp_path)
    assert not list(tmp_path.glob('checkpoint_*.pt'))


@pytest.mark.parametrize('export',['history','curves'])
def test_checkpoint_recovers_interrupted_exports(small_observations,tmp_path,monkeypatch,export):
    raw,value_range=small_observations
    full,history,_=core.train(raw,value_range,epochs=5,init_seed=2,checkpoints=[2])
    name='json_write' if export=='history' else 'npz_write';original=getattr(core,name)
    def interrupted(path,*args,**kwargs):
        raise OSError('Injected export interruption after checkpoint commit')
    with monkeypatch.context() as scoped:
        scoped.setattr(core,name,interrupted)
        with pytest.raises(OSError,match='Injected'):
            core.train(raw,value_range,epochs=2,init_seed=2,out_dir=tmp_path)
    assert (tmp_path/'checkpoint_2.pt').exists()
    (tmp_path/'history.json').write_text('{broken')
    (tmp_path/'curves_2.npz').unlink(missing_ok=True)
    resumed,resumed_history,snapshots=core.train(raw,value_range,epochs=5,init_seed=2,checkpoints=[2],out_dir=tmp_path)
    assert resumed_history==history
    assert set(snapshots)=={2,5}
    assert len(json.loads((tmp_path/'history.json').read_text()))==5
    for a,b in zip(full.parameters(),resumed.parameters(),strict=True):assert torch.equal(a,b)


def test_final_checkpoint_rebuilds_missing_final_curve(small_observations,tmp_path):
    raw,value_range=small_observations
    full,_,snapshots=core.train(raw,value_range,epochs=2,out_dir=tmp_path)
    (tmp_path/'curves_2.npz').unlink();(tmp_path/'history.json').unlink()
    resumed,history,recovered=core.train(raw,value_range,epochs=2,out_dir=tmp_path)
    assert len(history)==2 and set(recovered)=={2}
    assert np.array_equal(recovered[2]['R'],snapshots[2]['R'])
    for a,b in zip(full.parameters(),resumed.parameters(),strict=True):assert torch.equal(a,b)


def test_corrupt_checkpoint_history_is_rejected(small_observations,tmp_path):
    raw,value_range=small_observations;core.train(raw,value_range,epochs=2,out_dir=tmp_path)
    path=tmp_path/'checkpoint_2.pt';saved=torch.load(path,weights_only=False)
    saved['history']=saved['history'][1:];torch.save(saved,path)
    with pytest.raises(ValueError,match='every epoch'):core.train(raw,value_range,epochs=3,out_dir=tmp_path)


def test_worker_does_not_mark_empty_snapshots_complete(tmp_path,monkeypatch):
    import run_suite
    monkeypatch.setattr(run_suite,'ROOT',tmp_path)
    monkeypatch.setattr(run_suite,'train',lambda *args,**kwargs:(None,[],{}))
    config=dict(group='ablation',case='case_exp',setting='clean',seed=0,epochs=2)
    ident,status=run_suite.work(config)
    assert status=='failed'
    assert json.loads((tmp_path/'evidence/raw'/f'{ident}.json').read_text())['status']=='failed'
    assert run_suite.work(config)[1]=='cached_failed'
    assert run_suite.work(config,retry_failed=True)[1]=='failed'
    assert len(list((tmp_path/'evidence/failures').glob('*.json')))==1
    with pytest.raises(RuntimeError):core.report_completed([(ident,status)])


def test_completed_cache_requires_complete_epoch_history(tmp_path,monkeypatch):
    import run_suite
    monkeypatch.setattr(run_suite,'ROOT',tmp_path)
    config=dict(group='ablation',case='case_exp',setting='clean',seed=0,epochs=2)
    ident,status=run_suite.work(config);assert status=='complete'
    (tmp_path/'outputs/runs'/ident/'history.json').write_text('[]')
    with pytest.raises(ValueError,match='every epoch'):run_suite.work(config)


def test_declared_campaign_has_no_invalid_nyquist_arm():
    import run_suite
    configs=run_suite.task_configs('all');ids=[run_suite.task_id(c) for c in configs]
    assert len(configs)==445 and len(ids)==len(set(ids))
    assert not any(c['group']=='coarse' and c['setting']=='coarse8' for c in configs)
    assert all(c['num_test_modes']==3 for c in configs if c['group']=='coarse_corrected')


@pytest.mark.parametrize('horizon',[.025,.003,1e-13,0.])
def test_rollout_uses_actual_grid_and_reaches_exact_horizon(horizon):
    nx=16;x=np.arange(nx)/nx;u0=(.5+.1*np.cos(2*np.pi*x))[None,:]
    times,u,info=core.strict_rollout(lambda u:.03,lambda u:0.,u0,t_final=horizon,save_dt=.01)
    rate=-4*.03*np.sin(np.pi/nx)**2*nx**2
    exact=.5+.1*np.exp(rate*horizon)*np.cos(2*np.pi*x)
    assert info['status']=='stable' and info['nx']==nx
    assert times[-1]==horizon and info['reached_final_time']
    assert np.max(np.abs(u[-1,0]-exact))<1e-12
    with pytest.raises(ValueError,match='does not match'):
        core.strict_rollout(lambda u:.03,lambda u:0.,u0,nx=64,t_final=horizon)


def test_zero_horizon_does_not_hide_invalid_constitutive_law(tmp_path):
    _,_,info=core.strict_rollout(lambda u:np.nan,lambda u:0.,np.ones((1,16)),t_final=0)
    assert info['status']=='failed'
    core.json_write(tmp_path/'failed.json',info)
    assert json.loads((tmp_path/'failed.json').read_text())['status']=='failed'


def test_2d_heat_mode_and_final_partial_save():
    from run_2d import rollout_2d
    nx,ny=8,12;x=np.arange(nx)/nx;y=np.arange(ny)/ny
    u0=(.5+.1*np.cos(2*np.pi*x)[:,None]*np.cos(2*np.pi*y)[None,:])[None,:,:]
    times,u,info=rollout_2d(lambda u:.03,lambda u:0.,u0,t_final=.025,save_dt=.01)
    rate=-4*.03*(np.sin(np.pi/nx)**2*nx**2+np.sin(np.pi/ny)**2*ny**2)
    exact=.5+(u0-.5)*np.exp(rate*.025)
    assert info['status']=='stable' and times[-1]==.025
    assert np.max(np.abs(u[-1]-exact))<1e-12


def test_time_weak_manufactured_solution_converges():
    x=np.arange(16)/16;phi,gphi=core.make_test_functions_1d(x,num_modes=2)
    class Law:
        def diffusion(self,u):return torch.zeros_like(u)
        def reaction(self,u):return 2*torch.sqrt(u)
    errors=[]
    for n in (21,41,81):
        t=np.linspace(0,1,n);psi,dpsi=core.make_time_test_functions_1d(t)
        u=torch.from_numpy(np.broadcast_to(t[None,:,None]**2,(1,n,16)).copy())
        residual=weak_residual_space_time_1d(u,1/(n-1),1/16,*[torch.from_numpy(v) for v in (phi,gphi,psi,dpsi)],Law())
        errors.append(float(residual.abs().max()))
    assert errors[1]<errors[0]/3 and errors[2]<errors[1]/3


def test_disabled_polynomial_unseen_validation_is_respected(small_observations,monkeypatch):
    import closure_discovery.baselines.polynomial as polynomial
    raw,value_range=small_observations
    def forbidden(*args,**kwargs):raise AssertionError('Unseen validation was disabled')
    monkeypatch.setattr(polynomial,'compare_cases_on_shared_initial_conditions',forbidden)
    result=polynomial.run_polynomial_baseline(method='weak',case=core.extra_case('case_exp'),
        simulation_config=core.SimulationConfig1D(nx=16,t_final=.005,save_every=5),num_trajectories=2,
        amplitude_range=value_range,raw_dataset=raw,unseen_rollout_config=UnseenRolloutConfig(enabled=False),num_test_modes=2)
    assert result['rollout'] is None
    assert 'unseen_rollout_relative_l2' not in result['metrics']


def test_2d_report_includes_nondefault_case(tmp_path):
    benchmark=experiment('benchmark_2d_bias_inheritance')
    row=dict(case='case_b',**{k:1. for k in ['neural_true_ErrD','neural_true_ErrR','symbolic_true_ErrD',
        'symbolic_true_ErrR','neural_unseen','symbolic_unseen','bir_D','bir_R','weak_loss','state_bin_coverage']})
    path=tmp_path/'summary.md';benchmark.write_markdown(path,[row])
    assert '| case_b |' in path.read_text()


def test_empty_report_removes_stale_csv(tmp_path):
    from build_report import write_csv
    path=tmp_path/'results.csv';write_csv(path,[dict(error=.5)])
    assert path.exists();write_csv(path,[]);assert not path.exists()


def pysr_summary_row(**overrides):
    return dict(case='case_exp',setting='clean',seed='0',operator_mode='exp',protocol_id='protocol',search_id='search',
        surrogate_signature='frozen',**{k:'1' for k in ['pysr_surrogate_ErrD','pysr_surrogate_ErrR','pysr_true_ErrD',
        'pysr_true_ErrR','pysr_unseen','bir_D','bir_R','pysr_complexity_D','pysr_complexity_R']}) | overrides


def test_pysr_summary_separates_cases_operators_and_protocols():
    summary=experiment('summarize_pysr_comparison')
    rows=[pysr_summary_row(),pysr_summary_row(case='case_b',pysr_true_ErrR='9'),
          pysr_summary_row(operator_mode='exp_div'),pysr_summary_row(protocol_id='long_budget'),
          pysr_summary_row(search_id='different_search')]
    result=summary.aggregate_comparison(rows)
    assert len(result)==5 and all(r['num_seeds']==1 for r in result)
    assert [r['true_ErrR_mean'] for r in result if r['case']=='case_b']==[9.]
    with pytest.raises(ValueError,match='Duplicate seeds'):summary.aggregate_comparison([rows[0],rows[0]])


def test_pysr_reference_requires_identical_frozen_target():
    summary=experiment('summarize_pysr_comparison')
    ref=dict(case='case_exp',setting='clean',seed='0',protocol_id='protocol',surrogate_signature='other')
    with pytest.raises(ValueError,match='frozen target mismatch'):
        summary.aggregate_comparison([pysr_summary_row()],[ref])


def test_default_pysr_paths_do_not_overwrite_other_protocols():
    runner=experiment('run_pysr_symbolic_compression')
    paths=[runner.default_output_path('case_exp','clean',0,operator,protocol,search)
        for operator in ('exp','exp_div') for protocol in ('short','long') for search in ('low','high')]
    assert len(set(paths))==8


def test_public_pysr_command_records_protocol_and_keeps_distinct_runs(tmp_path,monkeypatch):
    import sympy as sp
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    runner=experiment('run_pysr_symbolic_compression')
    class FakeRegressor:
        def __init__(self,**kwargs):self.value=.03 if Path(kwargs['tempdir']).name=='diffusion' else 0.
        def fit(self,x,y):self.loss=float(np.mean((y-self.value)**2))
        def get_best(self):return dict(sympy_format=sp.Float(self.value),complexity=1,loss=self.loss)
    monkeypatch.setitem(sys.modules,'pysr',SimpleNamespace(PySRRegressor=FakeRegressor,__version__='test-only'))
    monkeypatch.chdir(tmp_path)
    common=['pysr','--nx','16','--t-final','.002','--save-every','5','--num-trajectories','2',
        '--num-modes','2','--hidden-width','8','--hidden-depth','1']
    for epochs,operators in [(1,'exp'),(2,'exp'),(1,'exp_div')]:
        monkeypatch.setattr(sys,'argv',common+['--epochs',str(epochs),'--operator-mode',operators])
        runner.main()
    paths=list((tmp_path/'results').glob('pysr_symbolic_*.csv'))
    assert len(paths)==3
    rows=[next(csv.DictReader(path.open())) for path in paths]
    assert len({row['protocol_id'] for row in rows})==2
    assert len({row['search_id'] for row in rows})==2
    assert all(row['surrogate_signature'] and row['device']=='cpu' for row in rows)


@pytest.mark.parametrize('change',['source','weights'])
def test_resume_rejects_incompatible_training_identity(small_observations,tmp_path,change):
    raw,value_range=small_observations;core.train(raw,value_range,epochs=2,out_dir=tmp_path)
    kwargs={}
    if change=='source':
        path=tmp_path/'checkpoint_2.pt';saved=torch.load(path,weights_only=False)
        saved['source_signature']='older implementation';torch.save(saved,path)
    else:kwargs['weights']={**core.DEFAULT_WEIGHTS,'mass':.5}
    with pytest.raises(ValueError,match='configuration mismatch'):
        core.train(raw,value_range,epochs=3,out_dir=tmp_path,**kwargs)


def test_legacy_checkpoint_requires_opt_in_and_complete_history(small_observations,tmp_path,monkeypatch):
    raw,value_range=small_observations
    full,history,_=core.train(raw,value_range,epochs=4)
    core.train(raw,value_range,epochs=2,out_dir=tmp_path)
    path=tmp_path/'checkpoint_2.pt';saved=torch.load(path,weights_only=False)
    for key in ('checkpoint_schema','history','snapshots','source_signature'):saved.pop(key)
    torch.save(saved,path)
    monkeypatch.delenv('EAAI_ALLOW_LEGACY_CHECKPOINT',raising=False)
    with pytest.raises(ValueError,match='Legacy checkpoint requires'):core.train(raw,value_range,epochs=4,out_dir=tmp_path)
    monkeypatch.setenv('EAAI_ALLOW_LEGACY_CHECKPOINT','1')
    with pytest.warns(RuntimeWarning,match='legacy checkpoint'):
        resumed,resumed_history,_=core.train(raw,value_range,epochs=4,out_dir=tmp_path)
    assert resumed_history==history
    for a,b in zip(full.parameters(),resumed.parameters(),strict=True):assert torch.equal(a,b)
