"""Numerical checks of the paired intervention, not paper-output snapshots."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
from anchor_control_core import fit_anchor_control,setup
from closure_discovery.symbolic.restricted_fit import SymbolicExpression


def pair():
    return (SymbolicExpression(family='polynomial',parameters=(.02,),expression='.02',complexity=1,fit_mse=0.),
            SymbolicExpression(family='polynomial',parameters=(-.1,.2),expression='-.1+.2*u',complexity=2,fit_mse=0.))


def observation(duration=.01):
    t=np.linspace(0,duration,5);x=np.arange(4)/4
    u=np.full((1,len(t),len(x)),.5)
    return dict(u=u,t=t,x=x)


def test_common_initial_and_hard_constraint_do_not_change_diffusion():
    _,split,p,scale=setup(pair())
    assert np.array_equal(p,np.array([.02,0.,.2]))
    assert split==1 and np.array_equal(scale,np.array([.02,.01,.2]))
    values={}
    for arm in ('unanchored','soft_anchor','hard_anchor'):
        coef,info=fit_anchor_control(observation(),(0.,1.),pair(),arm,first_budget=2,continuation_budget=2)
        assert info['initial_full_parameters']==p.tolist()
        assert info['fixed_initial_scales']==scale.tolist()
        assert not info['truth_used_for_fitting']
        values[arm]=(coef,info)
    hard,meta=values['hard_anchor']
    assert hard[1]==0. and meta['R0']==0. and meta['parameter_count']==2
    assert values['unanchored'][1]['parameter_count']==3


def test_fit_objective_is_decomposed_and_soft_anchor_scale_is_explicit():
    for arm in ('unanchored','soft_anchor','hard_anchor'):
        coef,meta=fit_anchor_control(observation(),(0.,1.),pair(),arm,first_budget=3,continuation_budget=3,soft_weight=2.)
        assert abs(meta['cost']-.5*sum(meta['components'].values()))<1e-15
        assert meta['components']['diffusion_penalty']==0.
        expected=2.*meta['R0']**2 if arm=='soft_anchor' else 0.
        assert abs(meta['components']['anchor_penalty']-expected)<1e-15


def test_known_constant_solution_is_recovered_without_true_law_targets():
    # The initial unconstrained law -.1+.2u has equilibrium .5. The paired
    # projected start deliberately loses that equilibrium; the data-only fit
    # must restore a zero derivative at the observed state.
    coef,meta=fit_anchor_control(observation(.1),(0.,1.),pair(),'unanchored',first_budget=40,continuation_budget=80)
    assert meta['optimizer_success'] and meta['components']['observation_mse']<1e-12
    assert abs(coef[1]+.5*coef[2])<2e-5
