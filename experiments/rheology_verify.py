"""Independent artifact reconstruction checks for the completed campaign."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rheology_core import features
from rheology_evaluate import Predictor


def run(output,raw):
    output,raw=Path(output),Path(raw)
    archive=(raw/'Rheological_Data_For_Syntactic_Foam.zip').read_bytes()
    assert hashlib.md5(archive).hexdigest()=='ef9df53250cb9984fa1697b87d781db1'
    audit=json.loads((output/'DATA_AUDIT.json').read_text())
    for relative,expected in audit['original_file_sha256'].items():
        assert hashlib.sha256((raw/relative).read_bytes()).hexdigest()==expected
    data=pd.read_csv(output/'observations.csv');specs=json.loads((output/'training_specs.json').read_text())
    arrays=np.load(output/'training_arrays.npz')
    assert len(specs)==200 and data.test_id.nunique()==50
    coordinate_error=0.;target_error=0.
    for s in specs:
        train=data.iloc[s['training_row_ids']];test=data.iloc[s['heldout_row_ids']]
        assert not set(s['training_row_ids'])&set(s['heldout_row_ids'])
        assert set(train.replicate)==set(range(5))-{s['fold']}
        assert set(test.replicate)=={s['fold']} and len(test)==250
        assert set(train.material)==set(test.material)=={s['material']}
        if s['arm']=='primary_high_shear':assert (train.rate>=10).all()
        assert (train.viscosity>0).all()
        n=len(train);i=s['index']
        expected=features(train.temperature.to_numpy(),train.rate.to_numpy(),s['temperature_bounds'],s['rate_bounds'])
        coordinate_error=max(coordinate_error,float(np.max(abs(expected-arrays['x'][i,:n]))))
        target_error=max(target_error,float(np.max(abs(np.log(train.viscosity.to_numpy())-s['offset']-arrays['y'][i,:n]))))
        assert arrays['mask'][i].sum()==n and not arrays['mask'][i,n:].any()
    assert coordinate_error<1e-12 and target_error<1e-12
    old=pd.read_csv(output/'reports/by_model.csv');new=pd.read_csv(output/'complexity/by_model.csv')
    assert len(old)==1900 and len(new)==450
    rawpred=np.load(output/'reports/heldout_predictions.npz');max_metric_error=0.
    for row in old.itertuples():
        ids=rawpred['observation_ids'][row.model_id];obs=data.iloc[ids]
        assert (obs.replicate==row.fold).all() and (obs.material==row.material).all()
        high=obs.rate.to_numpy()>=10
        eta=obs.viscosity.to_numpy()[high];pred=np.exp(rawpred['log_viscosity'][row.model_id,high])
        metric=float(np.linalg.norm(pred-eta)/np.linalg.norm(eta))
        max_metric_error=max(max_metric_error,abs(metric-row.high_shear_relative_l2))
    defs=json.loads((output/'complexity/model_definitions.json').read_text());new=new.set_index('model_id')
    for m in defs:
        s=specs[m['spec_index']];test=data.iloc[s['heldout_row_ids']];test=test[test.rate>=10]
        predictor=Predictor(s,coefficients=m['coefficients'],degree=m['degree'])
        eta=test.viscosity.to_numpy();pred=np.exp(predictor(test.temperature.to_numpy(),test.rate.to_numpy()))
        metric=float(np.linalg.norm(pred-eta)/np.linalg.norm(eta))
        max_metric_error=max(max_metric_error,abs(metric-new.loc[m['model_id'],'high_shear_relative_l2']))
    assert max_metric_error<1e-10
    baseline=pd.read_csv(output/'complexity/empirical_training_baseline.csv')
    assert len(baseline)==50 and baseline.n_total.sum()==(data.rate>=10).sum()==5001
    assert baseline.n_eligible.sum()==4477
    expected=set((a,m,f,s) for a in ('primary_high_shear','full_positive') for m in data.material.unique() for f in range(5) for s in (0,1))
    assert {(s['arm'],s['material'],s['fold'],s['seed']) for s in specs}==expected
    preserved=None
    if (output/'reports_attempt1').exists():
        previous=pd.read_csv(output/'reports_attempt1/by_model.csv')
        pd.testing.assert_frame_equal(previous,old)
        preserved='All1900 measured-data metric rows identical before/after flow quadrature refinement'
    numerical=[output/'models/checkpoint_2000.npz',output/'models/checkpoint_6000.npz',output/'training_specs.json',
               output/'reports/by_model.csv',output/'complexity/by_model.csv',output/'reports/flow_bands.csv',output/'complexity/flow_bands.csv']
    result=dict(status='passed',archive_md5_verified=True,all_raw_member_hashes_verified=True,whole_file_splits_checked=200,
                max_training_coordinate_reconstruction_error=coordinate_error,max_training_target_reconstruction_error=target_error,
                measured_prediction_model_instances_checked=2350,max_relative_l2_reconstruction_error=max_metric_error,
                primary_observations=5001,empirical_reference_common_support_observations=4477,
                original_metrics_preserved=preserved,
                numerical_sha256={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in numerical})
    (output/'VERIFICATION_FINAL.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--raw',type=Path,required=True)
    a=p.parse_args();run(a.output,a.raw)
