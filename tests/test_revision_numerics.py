"""Numerical regression checks, independent invariants and manufactured solutions."""
import numpy as np
import torch
from revision_core import *
from closure_discovery.weak_form.weak_residual import weak_residual_space_time_1d


def test_rk4_heat_mode_and_mass():
    nx=64; x=np.arange(nx)/nx;u0=(.5+.1*np.cos(2*np.pi*x))[None,:];D=lambda u: np.full_like(u,.03);R=lambda u:np.zeros_like(u)
    t,u,info=strict_rollout(D,R,u0,t_final=.1)
    discrete_rate=-4*.03*np.sin(np.pi/nx)**2/(1/nx)**2
    exact=.5+.1*np.exp(discrete_rate*.1)*np.cos(2*np.pi*x)
    assert info['status']=='stable'
    assert np.max(np.abs(u[-1,0]-exact))<1e-10
    assert np.max(np.abs(u.mean(-1)-.5))<1e-12


def test_rollout_rejects_negative_diffusion():
    _,_,info=strict_rollout(lambda u: -np.ones_like(u),lambda u:np.zeros_like(u),np.full((1,64),.5),t_final=.1)
    assert info['status']=='failed'


def test_time_weak_constant_state():
    # Exact stationary state: every temporal mode must have a zero residual.
    x=np.arange(64)/64;t=np.linspace(0,.1,101);phi,gphi=make_test_functions_1d(x);psi,dpsi=make_time_test_functions_1d(t)
    class Zero:
        def diffusion(self,u):return torch.zeros_like(u)
        def reaction(self,u):return torch.zeros_like(u)
    u=torch.ones((1,101,64),dtype=torch.float64)
    old=weak_residual_space_time_1d(u,.001,1/64,*[torch.from_numpy(z) for z in (phi,gphi,psi,dpsi)],Zero())
    w=np.ones(101);w[[0,-1]]=.5
    corrected=-np.einsum('mt,kx,btx->bmk',dpsi*w,phi,u.numpy())*.001/64
    assert float(old.abs().max())<1e-12
    assert np.max(np.abs(corrected))<1e-12


def test_bir_common_normalization_bound():
    g=np.linspace(0,1.8,256);truth=.01+.035*(1-np.exp(-2.5*g));neural=truth+.005*g;symbolic=neural+.0001*np.sin(g)
    en=np.linalg.norm(neural-truth);ec=np.linalg.norm(symbolic-neural);es=np.linalg.norm(symbolic-truth)
    assert abs(es/en-1)<=ec/en+1e-14
    # Published surrogate error uses the neural norm; eC/eN needs a norm conversion.
    c_rel=relative_l2_error(neural,symbolic);n_rel=relative_l2_error(truth,neural)
    assert np.isclose(ec/en,c_rel/n_rel*np.linalg.norm(neural)/np.linalg.norm(truth))


def test_optimizer_resume_preserves_training(tmp_path):
    c=extra_case('case_exp');raw=generate_dataset(c,SimulationConfig1D(nx=16,t_final=.005,save_every=5),2,
        seed=2,amplitude_range=c.value_range,initial_clip_range=c.value_range)
    full,_,_=train(raw,c.value_range,epochs=5,init_seed=2)
    train(raw,c.value_range,epochs=2,init_seed=2,out_dir=tmp_path)
    resumed,history,_=train(raw,c.value_range,epochs=5,init_seed=2,out_dir=tmp_path)
    assert len(history)==5
    for a,b in zip(full.parameters(),resumed.parameters()):assert torch.equal(a,b)


def test_resolved_tests_exclude_nyquist_without_changing_nominal_grid():
    for nx in [8, 16, 64]:
        x=np.arange(nx)/nx
        phi,grad=make_resolved_test_functions_1d(x,4)
        assert np.allclose(np.sum(phi**2,axis=1)/nx,1.)
        assert np.max(np.abs(grad))<100
        expected=make_test_functions_1d(x,num_modes=min(4,(nx-1)//2))
        assert np.array_equal(phi,expected[0])
        assert np.array_equal(grad,expected[1])
    import pytest
    with pytest.raises(ValueError,match='Nyquist'):
        make_test_functions_1d(np.arange(8)/8,num_modes=4)


def test_resume_rejects_different_observations(tmp_path):
    import pytest
    c=extra_case('case_exp');raw=generate_dataset(c,SimulationConfig1D(nx=16,t_final=.005,save_every=5),2,
        seed=2,amplitude_range=c.value_range,initial_clip_range=c.value_range)
    train(raw,c.value_range,epochs=2,init_seed=2,out_dir=tmp_path)
    changed={**raw,'u':raw['u'].copy()};changed['u'][0,0,0]+=.001
    with pytest.raises(ValueError,match='observation signature'):
        train(changed,c.value_range,epochs=5,init_seed=2,out_dir=tmp_path)


def test_cache_rejects_different_budget(tmp_path,monkeypatch):
    import pytest
    import run_suite
    monkeypatch.setattr(run_suite,'ROOT',tmp_path)
    config=dict(group='ablation',case='case_exp',setting='clean',seed=0,epochs=2)
    ident=run_suite.task_id(config)
    assert run_suite.work(config)==(ident,'complete')
    assert run_suite.work(config)==(ident,'cached')
    with pytest.raises(ValueError,match='Cached configuration mismatch'):
        run_suite.work({**config,'epochs':5})
    with pytest.raises(ValueError,match='Unsupported task configuration'):
        run_suite.work({**config,'nx':16})


def test_2d_optimizer_resume_preserves_training(tmp_path):
    from run_2d import train_2d
    from closure_discovery.data_generation.rd_solver_2d import SimulationConfig2D,generate_dataset as generate_2d
    c=extra_case('case_exp');raw=generate_2d(c,SimulationConfig2D(nx=8,ny=8,t_final=.002,save_every=5),2,
        seed=2,amplitude_range=c.value_range,num_modes=2,initial_clip_range=c.value_range)
    full,_,_=train_2d(raw,c.value_range,epochs=4,seed=2)
    train_2d(raw,c.value_range,epochs=2,seed=2,out_dir=tmp_path)
    resumed,h,_=train_2d(raw,c.value_range,epochs=4,seed=2,out_dir=tmp_path)
    assert len(h)==4
    for a,b in zip(full.parameters(),resumed.parameters()):assert torch.equal(a,b)


def test_optimizer_checkpoint_survives_directory_move(tmp_path):
    c=extra_case('case_exp');raw=generate_dataset(c,SimulationConfig1D(nx=16,t_final=.005,save_every=5),2,
        seed=2,amplitude_range=c.value_range,initial_clip_range=c.value_range)
    full,h,_=train(raw,c.value_range,epochs=5,init_seed=2)
    initial=tmp_path/'original';moved=tmp_path/'relocated'
    train(raw,c.value_range,epochs=2,init_seed=2,out_dir=initial)
    initial.rename(moved)
    resumed,rh,_=train(raw,c.value_range,epochs=5,init_seed=2,out_dir=moved)
    assert h==rh
    for a,b in zip(full.parameters(),resumed.parameters()):assert torch.equal(a,b)


def test_cuda_request_fails_when_unavailable(monkeypatch):
    import revision_core
    import pytest
    monkeypatch.setattr(revision_core,'DEVICE_OPTION','cuda')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA requested but unavailable'):revision_core.resolve_device()


def test_auto_device_falls_back_to_cpu(monkeypatch):
    import revision_core
    monkeypatch.setattr(revision_core,'DEVICE_OPTION','auto')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert revision_core.resolve_device()==torch.device('cpu')
