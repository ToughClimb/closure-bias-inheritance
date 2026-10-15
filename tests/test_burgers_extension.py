import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from burgers_core import (FluxNet,shock,observation_operator,direct_fit,polynomial,
                          conservative_rhs,cole_hopf_periodic,periodic_initial,
                          rollout,frozen_fit,projection_geometry,NU)


def test_flux_derivatives_match_autodifferentiation_and_numpy():
    model=FluxNet(4)
    u=torch.linspace(-.4,.9,91,dtype=torch.float64,requires_grad=True)
    a=torch.autograd.grad(model(u).sum(),u,create_graph=True)[0]
    b=torch.autograd.grad(a.sum(),u)[0]
    np.testing.assert_allclose(model(u,1).detach(),a.detach(),atol=2e-16)
    np.testing.assert_allclose(model(u,2).detach(),b.detach(),atol=2e-16)
    evaluate=model.numpy_evaluator()
    for d in range(3):
        np.testing.assert_allclose(evaluate(u.detach().numpy(),d),model(u,d).detach(),atol=2e-16)
    assert model.numpy(np.array([0.]))[0]==0


def test_weak_and_strong_flux_identification_refines_without_analytic_derivatives():
    errors={arm:[] for arm in ['weak','strong']}
    for n in [65,129,257]:
        x=np.linspace(-1,1,n);t=np.linspace(0,.3,61)
        data=shock(x,t)
        for arm in errors:
            coef,info=direct_fit(observation_operator(x,t,data,arm),2)
            assert info['rank']==2
            errors[arm].append(np.linalg.norm(coef-[0,.5]))
    for arm in errors:
        assert errors[arm][-1]<errors[arm][0]/5
        assert errors[arm][-1]<.003


def test_cole_hopf_recovers_initial_condition_and_pde():
    x=np.linspace(-1,1,257,endpoint=False)
    for kind in ['interpolation','extrapolation']:
        np.testing.assert_allclose(cole_hopf_periodic(x,0,kind),periodic_initial(x,kind),atol=1e-9)
        h=1e-4;t=.3
        u=cole_hopf_periodic(x,t,kind)
        ut=(cole_hopf_periodic(x,t+h,kind)-cole_hopf_periodic(x,t-h,kind))/(2*h)
        ux=(cole_hopf_periodic(x+h,t,kind)-cole_hopf_periodic(x-h,t,kind))/(2*h)
        uxx=(cole_hopf_periodic(x+h,t,kind)-2*u+cole_hopf_periodic(x-h,t,kind))/h**2
        assert max(abs(ut+u*ux-NU*uxx))<2e-5


def test_forward_operator_is_conservative_and_grid_refines_to_independent_truth():
    evaluate=lambda u,derivative=0:polynomial(u,[0,.5],derivative)
    errors=[]
    for n in [64,128]:
        result=rollout(evaluate,'interpolation',n=n,horizon=.5)
        reference=cole_hopf_periodic(result['x'],.5,'interpolation')
        errors.append(np.sqrt(np.mean((result['u'][-1]-reference)**2)))
        assert result['mass_drift']<1e-12
        assert result['status']=='completed'
        assert abs(np.sum(conservative_rhs(result['u'][-1],2/n,evaluate)))<1e-10
    assert errors[1]<errors[0]/3


def test_projection_geometry_and_gauge_fix():
    u=np.linspace(-.25,.75,1025)
    teacher=.5*u*u+.01*u+.015*np.sin(11*u)*u
    coef=frozen_fit(u,teacher,2)
    geometry=projection_geometry(u,teacher,coef)
    assert geometry['decomposition_residual']<1e-16
    assert geometry['symbolic_squared_error']<=geometry['teacher_squared_error']+1e-16
    assert polynomial(np.array([0.]),coef)[0]==0


def test_implicit_sparse_jacobian_integrator_matches_explicit_temporal_reference():
    from burgers_forward_audit import implicit_rollout
    evaluate=lambda u,derivative=0:polynomial(u,[0,.5],derivative)
    explicit=rollout(evaluate,'interpolation',n=64)
    implicit=implicit_rollout(evaluate,'interpolation',n=64)
    np.testing.assert_allclose(implicit['u'],explicit['u'],atol=1e-8)
    assert implicit['mass_drift']<1e-12
