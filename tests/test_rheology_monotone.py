import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from rheology_monotone import (derivative_bernstein_matrix,margin_matrix,bernstein_evaluate,
                               constrained_fit,power_to_bernstein)
from rheology_core import design,polynomial_stress_margin,powers


def test_bernstein_conversion_all_basis_functions_and_corners():
    axis=np.linspace(0,1,9)
    unit=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
    matrix=derivative_bernstein_matrix()
    x=2*unit-1
    for col,(i,j) in enumerate(powers(6)):
        exact=np.zeros(len(x)) if j==0 else j*x[:,0]**i*x[:,1]**(j-1)
        np.testing.assert_allclose(bernstein_evaluate(unit,matrix[:,col]),exact,atol=6e-14)
    rng=np.random.default_rng(65);coef=rng.normal(size=28)
    np.testing.assert_allclose(bernstein_evaluate(unit,1+margin_matrix((11.5,50.1))@coef),
        polynomial_stress_margin(x,coef,6,(11.5,50.1)),atol=1e-13)


def test_feasible_linear_solution_is_unchanged_and_qp_kkt_holds():
    axis=np.linspace(-1,1,11);x=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
    a=design(x,6);truth=np.zeros(28);truth[:3]=[.4,-.2,.5]
    coef,bern,dual,info=constrained_fit(a,a@truth,(10,50))
    assert info['passed']
    np.testing.assert_allclose(a@coef,a@truth,atol=2e-8)
    assert bern.min()>.05


def test_nonmonotone_target_is_globally_constrained_with_kkt_certificate():
    axis=np.linspace(-1,1,13);x=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
    a=design(x,6);target=-2*x[:,1]
    coef,bern,dual,info=constrained_fit(a,target,(10,50))
    assert info['passed']
    assert bern.min()>=.05
    assert np.min(polynomial_stress_margin(x,coef,6,(10,50)))>=.05
    assert info['training_log_mse']>0
    assert np.all(dual>=0)
