import sys
import csv
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from basic_pde_controls import (linear_fields,linear_fit,stencil_fields,clean_discrete_coefficients,
                                advection_phase_error,burgers_fields,deterministic_controls,plan)


@pytest.mark.parametrize('D,c,a',[(.03,0.,0.),(0.,.4,0.),(.03,.4,0.),(.03,0.,.8)])
def test_fourier_solution_satisfies_balance(D,c,a):
    fields=linear_fields(np.arange(64)/64,np.linspace(0,.1,21),diffusion=D,speed=c,reaction=a,modes=(1,2))
    np.testing.assert_allclose(fields['ut']-D*fields['uxx']+c*fields['ux']-a*fields['u'],0.,atol=2e-15)


@pytest.mark.parametrize('D,c,unknowns',[(.03,0.,('D',)),(0.,.4,('c',)),(.03,.4,('D','c'))])
def test_observation_stencils_have_predicted_parameter_bias(D,c,unknowns):
    x=np.arange(32)/32;t=np.linspace(0,.1,21)
    fields=linear_fields(x,t,diffusion=D,speed=c)
    fit=linear_fit(stencil_fields(fields['u'],1/32,.005),unknowns)
    predicted=clean_discrete_coefficients(32,.005,diffusion=D,speed=c)
    for key in unknowns:assert abs(fit['coefficients'][key]-predicted[key])<1e-13


def test_second_mode_resolves_reaction_diffusion_compensation():
    x=np.arange(64)/64;t=np.linspace(0,.5,51)
    one=linear_fields(x,t,diffusion=.03,reaction=.8,modes=(1,))
    assert linear_fit(one,('D','a'))['status']=='unidentifiable'
    other=linear_fields(x,t,diffusion=.04,reaction=.8+.01*(2*np.pi)**2,modes=(1,))
    np.testing.assert_allclose(one['u'],other['u'],atol=2e-15)
    two=linear_fields(x,t,diffusion=.03,reaction=.8,modes=(1,2))
    fit=linear_fit(two,('D','a'))
    assert fit['rank']==2
    np.testing.assert_allclose([fit['coefficients']['D'],fit['coefficients']['a']],[.03,.8],atol=1e-14)


def test_phase_error_is_not_detected_by_mass_or_energy():
    x=np.arange(1024)/1024;t=np.array([0.,5.,25.])
    true=linear_fields(x,t,speed=.4)['u']
    wrong=linear_fields(x,t,speed=.42)['u']
    actual=np.linalg.norm(wrong-true,axis=1)/np.linalg.norm(true,axis=1)
    np.testing.assert_allclose(actual,advection_phase_error(.02,t),atol=2e-14)
    np.testing.assert_allclose(wrong.mean(axis=1),true.mean(axis=1),atol=5e-16)
    np.testing.assert_allclose(np.mean(wrong**2,axis=1),np.mean(true**2,axis=1),atol=5e-16)


def test_viscous_shock_preserves_initial_state_and_exposes_flux_slope_error():
    x=np.linspace(-2,2,4097);t=np.array([0.,1.])
    true=burgers_fields(x,t)
    modified=burgers_fields(x,t,flux_linear=.05)
    np.testing.assert_array_equal(true['u'][0],modified['u'][0])
    np.testing.assert_allclose(modified['shock_center']-true['shock_center'],.05*t,atol=1e-16)
    np.testing.assert_allclose(modified['ut']+(modified['u']+.05)*modified['ux']-.02*modified['uxx'],0.,atol=2e-14)


def test_reported_coefficient_check_matches_saved_measurements(tmp_path):
    (tmp_path/'reports').mkdir()
    check=deterministic_controls(tmp_path,plan())
    with (tmp_path/'reports/discrete_coefficients.csv').open() as stream:rows=list(csv.DictReader(stream))
    measured=max(abs(float(row['finite_difference_fit'])-float(row['predicted_discrete_fit'])) for row in rows)
    assert check['maximum_discrete_coefficient_prediction_error']==measured
    assert measured<1e-11
