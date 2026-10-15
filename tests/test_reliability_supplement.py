"""Physical and provenance checks for the reviewer reliability supplement."""
import numpy as np
import pytest
import torch

from closure_discovery.models.mlp_closure import ReactionDiffusionClosure
from revision_core import strict_rollout, extra_case, generate_dataset, SimulationConfig1D
from reliability_core import NumpyMLPClosure, lbfgs_crosscheck


def test_completion_does_not_imply_nonnegative_state():
    times, values, info = strict_rollout(lambda u:.03, lambda u:-.25,
        np.zeros((1,16)), t_final=.01, state_range=(0.,1.8))
    assert info['numerical_completion']
    assert not info['accepted_state_range_valid']
    assert not info['rk_stage_state_range_valid']
    assert info['first_accepted_range_violation']['time']>0
    assert np.max(abs(values[-1]+.0025))<1e-12
    assert info['accepted_state_min']==pytest.approx(-.0025)


def test_initial_extrema_are_retained_after_reaction_decay():
    _, values, info = strict_rollout(lambda u:.03, lambda u:-u,
        np.ones((1,16)), t_final=.1, state_range=(0.,1.1))
    assert info['state_max']<1
    assert info['accepted_state_max']==1
    assert info['rk_stage_state_max']==1
    assert info['accepted_state_range_valid']


def test_stage_violation_is_separate_from_accepted_step():
    # Deliberately large explicit step: RK4 stages leave [0,1], final state stays.
    _, values, info = strict_rollout(lambda u:.001, lambda u:-2*u,
        np.ones((1,16)), t_final=1., save_dt=1., max_dt=1., adaptive=False,state_range=(0.,1.))
    assert values[-1].min()>=0 and values[-1].max()<=1
    assert info['accepted_state_range_valid']
    assert not info['rk_stage_state_range_valid']
    assert info['first_rk_stage_range_violation']['stage']=='k4'
    assert info['rk_stage_state_min']==pytest.approx(-1.)


def test_range_tracking_preserves_existing_integrator_values():
    x=np.arange(16)/16; initial=(.5+.1*np.cos(2*np.pi*x))[None,:]
    args=(lambda u:.03, lambda u:u*(1-u), initial)
    t0,u0,_=strict_rollout(*args,t_final=.03)
    t1,u1,info=strict_rollout(*args,t_final=.03,state_range=(0.,1.))
    assert np.array_equal(t0,t1) and np.array_equal(u0,u1)
    assert info['accepted_state_range_valid']


def test_native_network_export_matches_weights_and_extrapolation(tmp_path):
    torch.manual_seed(3)
    model=ReactionDiffusionClosure(hidden_width=5,hidden_depth=1,input_range=(0.,1.8)).double()
    with torch.no_grad():
        model.reaction_net.coefficients.copy_(torch.tensor([.1,.2,-.1,.3],dtype=torch.float64))
        model.reaction_net.residual_net.network[-1].weight.fill_(.03)
        model.diffusion_net.residual_net.network[-1].weight.fill_(.02)
    exported=NumpyMLPClosure.from_model(model)
    assert exported.verify(model,(0.,1.8))['outside_support_probe']
    exported.save(tmp_path/'native.npz');loaded=NumpyMLPClosure.load(tmp_path/'native.npz')
    probe=np.array([-.2,0.,1.,1.8,2.])
    assert np.array_equal(exported.reaction(probe),loaded.reaction(probe))
    assert loaded.reaction(np.array([2.]))[0]!=loaded.reaction(np.array([1.8]))[0]


@pytest.mark.parametrize('bounds',[(0.,0.),(1.,0.),(0.,np.inf)])
def test_invalid_state_range_is_rejected(bounds):
    with pytest.raises(ValueError,match='state_range'):
        strict_rollout(lambda u:.03,lambda u:0.,np.ones((1,16)),state_range=bounds)


def test_lbfgs_safe_checkpoint_resume_matches_uninterrupted(tmp_path,monkeypatch):
    import reliability_core
    torch.set_num_threads(1);torch.manual_seed(2)
    case=extra_case('case_exp')
    observed=generate_dataset(case,SimulationConfig1D(nx=16,t_final=.003,save_every=5),2,
        seed=2,amplitude_range=case.value_range,initial_clip_range=case.value_range)
    model=ReactionDiffusionClosure(hidden_width=5,hidden_depth=1,input_range=case.value_range)
    identity=dict(id='test',observation='fixed',protocol='two_safe_outer_steps')
    full,details=lbfgs_crosscheck(model,observed,case.value_range,tmp_path/'full',identity,
        outer_steps=2,inner_iterations=2)
    write=reliability_core.torch_write
    def interrupted(path,value):
        write(path,value)
        raise OSError('Injected interruption after safe optimizer checkpoint')
    monkeypatch.setattr(reliability_core,'torch_write',interrupted)
    with pytest.raises(OSError,match='safe optimizer checkpoint'):
        lbfgs_crosscheck(model,observed,case.value_range,tmp_path/'resumed',identity,
            outer_steps=2,inner_iterations=2)
    monkeypatch.setattr(reliability_core,'torch_write',write)
    resumed,again=lbfgs_crosscheck(model,observed,case.value_range,tmp_path/'resumed',identity,
        outer_steps=2,inner_iterations=2)
    for first,second in zip(full.parameters(),resumed.parameters(),strict=True):
        assert torch.equal(first,second)
    assert details['closure_calls']==again['closure_calls']
