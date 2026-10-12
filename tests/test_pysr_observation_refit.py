"""Independent manufactured-solution checks of fixed-expression data refitting."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import sympy as sp
import pytest

spec=importlib.util.spec_from_file_location('pysr_observation_refit',Path(__file__).resolve().parents[1]/'experiments/run_pysr_observation_refit.py')
refit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=refit;spec.loader.exec_module(refit)


def test_expression_template_preserves_values_and_integer_structure():
    text='0.03 + 0.03*x0 + 0.2*x0**2*exp(-0.4*x0)'
    template=refit.ExpressionTemplate.compile(text);x=np.linspace(0,1.8,31)
    expected=sp.lambdify(sp.Symbol('x0'),sp.sympify(text),'numpy')(x)
    assert np.allclose(template.evaluate(template.parameters,x),expected,rtol=1e-14,atol=1e-15)
    assert len(template.constants)==3
    assert sp.Integer(2) in template.symbolic.atoms(sp.Integer)
    changed=template.parameters.copy();changed[0]+=.1
    assert not np.allclose(template.evaluate(changed,x),expected)
    restored=refit.ExpressionTemplate.compile(template.expression(template.parameters))
    assert np.allclose(restored.evaluate(restored.parameters,x),expected)


def test_refit_recovers_manufactured_heat_and_reaction_without_truth_targets():
    # Exact solution of the discrete Laplacian plus a linear reaction law.
    nx=16;x=np.arange(nx)/nx;t=np.linspace(0,.05,11)
    diffusion=.03;reaction=.2;rate=-4*diffusion*np.sin(np.pi/nx)**2*nx**2
    initials=[(.5,.1),(.8,.15)]
    u=np.stack([mean*np.exp(reaction*t[:,None])+amp*np.exp((rate+reaction)*t[:,None])*np.cos(2*np.pi*x)
                for mean,amp in initials])
    observed=dict(x=x,t=t,u=u,u0=u[:,0])
    templates=[refit.ExpressionTemplate.compile('0.015'),refit.ExpressionTemplate.compile('0.1*x0')]
    parameters,info=refit.fit_to_observations(observed,(0.,1.2),templates,max_nfev=30)
    assert info['success'] and not info['truth_used_for_fitting']
    assert abs(parameters[0]-diffusion)<1e-6
    assert abs(parameters[1]-reaction)<1e-6
    assert info['cost']<info['initial_cost']*1e-6
    assert info['failed_trial_evaluations']>=0


def test_reconstruction_is_explicit_and_diagnostics_must_match():
    case=refit.extra_case('case_exp');raw=refit.generate_dataset(case,
        refit.SimulationConfig1D(nx=16,t_final=.005,save_every=5),2,seed=0,
        amplitude_range=case.value_range,initial_clip_range=case.value_range)
    observed=refit.apply_observation_model(raw,seed=0,clip_range=case.value_range)
    neural=dict(data_hash='different_archive_bytes',observation=refit.observation_diagnostics(raw,observed,case,0,0.))
    with pytest.raises(ValueError,match='Observation bytes differ'):
        refit.observation_identity(raw,observed,case,0,0.,neural)
    identity=refit.observation_identity(raw,observed,case,0,0.,neural,True)
    assert not identity['bitwise_observation_match']
    assert all(value==0 for value in identity['diagnostic_absolute_differences'].values())
    neural['observation']['excitation']['gradient_rms']*=2
    with pytest.raises(ValueError,match='diagnostic mismatch'):
        refit.observation_identity(raw,observed,case,0,0.,neural,True)


def test_continuation_preserves_original_coefficient_indexing():
    import importlib
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
    continuation=importlib.import_module('continue_pysr_observation_refit')
    original=refit.ExpressionTemplate.compile('0.3 + 0.1*x0 + exp(0.2*x0)')
    starting=original.parameters.copy();starting[0]*=-1
    warmed=continuation.StartingTemplate(original,starting)
    x=np.linspace(0,1,11)
    assert np.array_equal(warmed.parameters,starting)
    assert warmed.constants==original.constants
    assert np.array_equal(warmed.evaluate(warmed.parameters,x),original.evaluate(starting,x))
