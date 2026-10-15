import numpy as np
import pytest
import torch

from experiments.linear_pde_extension import (
    LocalOperator, calibration, feature_matrix, fields, integrate_operator,
    numpy_predictor, scaled_ols, spectral_features, SYSTEMS,
)
from experiments.linear_pde_report import centered_feature_noise_covariance


@pytest.mark.parametrize('system',list(SYSTEMS))
def test_manufactured_continuum_and_symbolic_solver(system):
    x=np.arange(32)/32;t=np.array([0.,.05,.1,.2,.4,.8])
    f=fields(x,t,system);p=SYSTEMS[system]
    coefficients=np.array([p[k] for k in p['unknowns']])
    X=feature_matrix(f,p['unknowns'])
    assert np.max(np.abs(X@coefficients-f['ut'].ravel()))<2e-14
    exact=scaled_ols(X,f['ut'].ravel())
    np.testing.assert_allclose(exact['coefficients'],coefficients,rtol=1e-12,atol=1e-12)
    meta,u=integrate_operator(lambda X:X@coefficients,p['unknowns'],f['u'][0],t)
    assert meta['status']=='completed'
    assert np.max(np.abs(u-f['u']))<3e-7


def test_rank_deficiency_is_explicit():
    f=fields(np.arange(32)/32,np.linspace(0,.2,10),'reaction_diffusion',modes=(1,),amplitudes=(1,))
    fit=scaled_ols(feature_matrix(f,['D','a']),f['ut'].ravel())
    assert fit['rank']==1
    assert fit['status']=='unidentifiable'
    assert fit['coefficients'] is None


def test_numpy_and_torch_predictor_zero_and_scaling():
    torch.manual_seed(3);model=LocalOperator(2)
    xs=np.array([13.,.8]);ys=2.3;X=np.random.default_rng(4).normal(size=(100,2))
    predict=numpy_predictor(model,xs,ys)
    reference=model(torch.tensor(X/xs,dtype=torch.float64)).detach().numpy().ravel()*ys
    np.testing.assert_allclose(predict(X),reference,rtol=1e-13,atol=1e-13)
    assert abs(predict(np.zeros((1,2)))[0])<1e-14


def test_projection_removes_only_orthogonal_component():
    torch.manual_seed(9);model=LocalOperator(2)
    summary,arrays=calibration(model,np.array([5.,3.]),2.,'advection_diffusion')
    assert summary['projection_identity_residual']<2e-14
    assert summary['symbolic_squared_error']<=summary['teacher_squared_error']+1e-14
    assert np.max(np.abs(arrays['X'].T@(arrays['teacher']-arrays['symbolic'])))<1e-9


def test_fourier_differentiation_is_independent_of_continuum_generator():
    n=64;x=np.arange(n)/n;u=.3+.2*np.cos(2*np.pi*3*x)
    X=spectral_features(u,['D','c','a'])
    np.testing.assert_allclose(X[:,0],-.2*(6*np.pi)**2*np.cos(6*np.pi*x),atol=2e-11)
    np.testing.assert_allclose(X[:,1],.2*6*np.pi*np.sin(6*np.pi*x),atol=2e-13)
    np.testing.assert_array_equal(X[:,2],u)


def test_rollout_failure_preserves_reason_and_count():
    meta,result=integrate_operator(lambda X:X[:,0],['a'],np.ones(16),np.array([0.,1.]),rhs_budget=2)
    assert result is None
    assert meta['status']=='failed'
    assert meta['rhs_evaluations']==3
    assert 'budget' in meta['reason']


def test_centered_differentiation_noise_covariance_independent_draws():
    rng=np.random.default_rng(7103);sigma=.01;h=1/32;dt=.01
    left,center,right,past,future=rng.normal(scale=sigma,size=(5,200000))
    features=np.column_stack([(left-2*center+right)/h**2,(left-right)/(2*h),center])
    target=(future-past)/(2*dt)
    covariance=np.cov(features,rowvar=False,bias=True)
    exact=centered_feature_noise_covariance(['D','c','a'],sigma,h)
    scale=np.sqrt(np.diag(exact))
    assert np.max(np.abs((covariance-exact)/scale[:,None]/scale[None,:]))<.008
    cross=np.mean(features*target[:,None],axis=0)
    assert np.max(np.abs(cross)/scale/np.std(target))<.008


def test_population_noise_optimizer_keeps_u_laplacian_cross_covariance():
    x=np.arange(64)/64;t=np.linspace(0,.2,21)
    f=fields(x,t,'reaction_diffusion');X=feature_matrix(f,['D','a']);y=f['ut'].ravel()
    G=X.T@X/len(X);b=X.T@y/len(X)
    covariance=centered_feature_noise_covariance(['D','a'],.004,1/64)
    beta=np.linalg.solve(G+covariance,b)
    # The gradient of the expected noisy objective is exactly zero there.
    gradient=2*((G+covariance)@beta-b)
    assert np.max(np.abs(gradient))<1e-13
    independent_noise_approx=np.linalg.solve(G+np.diag(np.diag(covariance)),b)
    assert np.linalg.norm(beta-independent_noise_approx)>1e-3
