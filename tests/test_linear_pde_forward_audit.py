import numpy as np
import torch

from experiments.linear_pde_extension import LocalOperator,spectral_features
from experiments.linear_pde_forward_audit import numpy_operator_with_gradient,fourier_feature_matrices


def test_analytic_operator_jacobian_matches_autograd_and_finite_difference():
    torch.manual_seed(37);model=LocalOperator(2)
    xs=np.array([11.,.7]);ys=2.1;X=np.random.default_rng(7).normal(size=(40,2))*xs
    evaluate=numpy_operator_with_gradient(model,xs,ys);value,jac=evaluate(X)
    Z=torch.tensor(X/xs,dtype=torch.float64,requires_grad=True);pred=model(Z)
    ad=torch.autograd.grad(pred.sum(),Z)[0].detach().numpy()*ys/xs
    np.testing.assert_allclose(jac,ad,atol=1e-13)
    for j in range(2):
        perturb=np.zeros_like(X);perturb[:,j]=1e-5*xs[j]
        fd=(evaluate(X+perturb)[0]-evaluate(X-perturb)[0])/(2e-5*xs[j])
        np.testing.assert_allclose(fd,jac[:,j],atol=1e-9,rtol=1e-8)


def test_dense_fourier_chain_rule_jacobian_matches_directional_derivative():
    torch.manual_seed(38);model=LocalOperator(2);unknowns=['D','a'];n=32
    x=np.arange(n)/n;u=.3+.2*np.cos(2*np.pi*x);direction=np.sin(4*np.pi*x)+.1
    evaluate=numpy_operator_with_gradient(model,np.array([10.,.3]),.8)
    matrices=fourier_feature_matrices(n,unknowns);X=spectral_features(u,unknowns)
    _,grad=evaluate(X);J=sum(grad[:,j,None]*matrix for j,matrix in enumerate(matrices))
    step=1e-6
    fd=(evaluate(spectral_features(u+step*direction,unknowns))[0]-evaluate(spectral_features(u-step*direction,unknowns))[0])/(2*step)
    np.testing.assert_allclose(J@direction,fd,atol=1e-6,rtol=1e-6)
