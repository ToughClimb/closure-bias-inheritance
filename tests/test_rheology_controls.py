import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from rheology_core import (features,design,fit_log_polynomial,polynomial_stress_margin,
                          whole_replicate_split,threshold_crossing,interpolate_log_rate,
                          slit_flow_band,ViscosityNetwork)
from rheology_train import BatchedNetworks
from rheology_evaluate import Predictor
from rheology_complexity import empirical_predict


def test_temperature_and_rate_coordinates_use_kelvin_and_log():
    x=features([35,125],[1,50])
    np.testing.assert_allclose(x,[[1,-1],[-1,1]],atol=1e-15)
    with pytest.raises(ValueError):features([35],[0])


def test_exact_power_law_and_stress_margin():
    t,g=np.meshgrid(np.linspace(35,125,5),np.geomspace(1,50,11));x=features(t.ravel(),g.ravel())
    y=np.log(.7)+.3*np.log(g.ravel())
    c,_=fit_log_polynomial(x,y,1)
    np.testing.assert_allclose(design(x,1)@c,y,atol=2e-15)
    np.testing.assert_allclose(polynomial_stress_margin(x,c,1),1.3,atol=2e-15)


def test_whole_replicate_split_contains_all_temperatures():
    ids=np.repeat(np.arange(5),250);train,test=whole_replicate_split(ids,2)
    assert test.sum()==250 and not np.intersect1d(ids[train],ids[test]).size


def test_no_crossing_extrapolation_or_arbitrary_branch_selection():
    assert threshold_crossing([35,75,125],[.8,.2,.8])['status']=='multiple'
    out=threshold_crossing([35,75],[.8,.2]);assert out['status']=='unique'
    assert out['crossings_celsius']==[55.]
    assert threshold_crossing([35,75],[.8,.6])['status']=='unbracketed'


def test_rate_interpolation_exact_power_law_and_no_extrapolation():
    r=np.array([18.8,22.2]);np.testing.assert_allclose(interpolate_log_rate(r,.5*r**.3),.5*20**.3)
    with pytest.raises(ValueError):interpolate_log_rate(r,np.ones(2),50)


@pytest.mark.parametrize('exponent',[0.,.3,-.5])
def test_conditional_band_contains_known_power_law_flow(exponent):
    k=.4;minimum=10.;maximum=50.;wall=k*40**(1+exponent);h=.0005
    result=slit_flow_band(lambda g:np.log(k)+exponent*np.log(g),min_rate=minimum,max_rate=maximum,wall_stress=wall,half_gap=h)
    q=2*h*h*(wall/k)**(1/(1+exponent))/(2+1/(1+exponent))
    assert result['status']=='valid_conditional_band'
    assert result['Q_lower_m2_per_s'] <= q <= result['Q_upper_m2_per_s']
    assert result['relative_quadrature_difference']<1e-10


def test_nonmonotone_stress_is_not_sorted_or_repaired():
    result=slit_flow_band(lambda g:-2*np.log(g),min_rate=1,max_rate=50,wall_stress=.5)
    assert result['status']=='nonmonotone_stress'


def test_batched_parameters_and_masked_gradients_equal_independent_networks():
    seeds=[0,1];batch=BatchedNetworks(seeds);torch.manual_seed(9)
    x=torch.randn(2,7,2,dtype=torch.float64);y=torch.randn(2,7,dtype=torch.float64)
    mask=torch.ones(2,7,dtype=torch.float64);mask[1,4:]=0
    loss=(((batch(x)-y)**2)*mask).sum(1)/mask.sum(1);loss.sum().backward()
    for i,seed in enumerate(seeds):
        torch.manual_seed(seed);single=ViscosityNetwork().double()
        n=7 if i==0 else 4
        ((single(x[i,:n])-y[i,:n])**2).mean().backward()
        torch.testing.assert_close(batch(x)[i,:n],single(x[i,:n]),atol=1e-15,rtol=1e-14)
        for layer in (0,2,4):
            torch.testing.assert_close(getattr(batch,f'w{layer}').grad[i].T,single.net[layer].weight.grad,atol=1e-15,rtol=1e-13)
            torch.testing.assert_close(getattr(batch,f'b{layer}').grad[i,0],single.net[layer].bias.grad,atol=1e-15,rtol=1e-13)


def test_neural_stress_margin_matches_independent_finite_difference():
    batch=BatchedNetworks([0]);weights={n:p.detach().numpy()[0] for n,p in batch.named_parameters()}
    spec={'temperature_bounds':[35,125],'rate_bounds':[11.5,50.],'offset':-.3}
    model=Predictor(spec,weights=weights);rate=np.geomspace(12,49,31);t=np.full_like(rate,75.)
    delta=1e-5
    derivative=(model(t,rate*np.exp(delta))-model(t,rate*np.exp(-delta)))/(2*delta)
    np.testing.assert_allclose(model.margin(model.x(t,rate)),1+derivative,atol=2e-11)


def test_empirical_reference_interpolates_whole_training_repeats_without_extrapolation():
    import pandas as pd
    records=[]
    for rep in range(4):
        for t in (35.,75.,125.):
            for g in (10.,20.,50.):
                records.append(dict(replicate=rep,nominal_temperature=t,temperature=t,rate=g,
                                    viscosity=np.exp(.01*t)*g**.3))
    frame=pd.DataFrame(records)
    result=empirical_predict(frame,[70.,30.,70.],[15.,15.,60.])
    np.testing.assert_allclose(result[0],.7+.3*np.log(15),atol=1e-14)
    assert np.isnan(result[1:]).all()
