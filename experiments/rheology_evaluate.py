"""Held-out measured-data endpoints and conditional engineering predictions."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rheology_core import (features,design,fit_log_polynomial,polynomial_stress_margin,
                           threshold_crossing,interpolate_log_rate,slit_flow_band)


class Predictor:
    def __init__(self,spec,weights=None,coefficients=None,degree=None):
        self.spec,self.weights,self.coefficients,self.degree=spec,weights,coefficients,degree

    def x(self,t,g):
        return features(t,g,self.spec['temperature_bounds'],self.spec['rate_bounds'])

    def predict_x(self,x):
        if self.weights is None:return design(x,self.degree)@self.coefficients
        for layer in (0,2,4):
            x=x@self.weights[f'w{layer}']+self.weights[f'b{layer}'][0]
            if layer!=4:x=np.tanh(x)
        return x[...,0]+self.spec['offset']

    def __call__(self,t,g):return self.predict_x(self.x(t,g))

    def margin(self,x):
        if self.weights is None:
            return polynomial_stress_margin(x,self.coefficients,self.degree,self.spec['rate_bounds'])
        h1=np.tanh(x@self.weights['w0']+self.weights['b0'][0])
        d1=(1-h1*h1)*self.weights['w0'][1]
        h2=np.tanh(h1@self.weights['w2']+self.weights['b2'][0])
        d2=(1-h2*h2)*(d1@self.weights['w2'])
        derivative=(d2@self.weights['w4'])[...,0]
        return 1+derivative*2/np.log(self.spec['rate_bounds'][1]/self.spec['rate_bounds'][0])


def interpolate_condition(frame,rate):
    a=frame.sort_values('rate')
    eta=interpolate_log_rate(a.rate.to_numpy(),a.viscosity.to_numpy(),rate)
    temp=float(np.interp(np.log(rate),np.log(a.rate),a.temperature))
    return temp,eta


def processing_reference(frame):
    out=[]
    for nominal,f in frame.groupby('nominal_temperature',sort=True):
        # Only the two brackets around 20/s enter; unrelated negative low-rate
        # observations remain in the raw dataset and do not block this endpoint.
        low=f[f.rate<=20].iloc[-1];high=f[f.rate>=20].iloc[0]
        t,eta=interpolate_condition(pd.DataFrame([low,high]).drop_duplicates('rate'),20.)
        out.append((float(nominal),t,eta))
    return np.asarray(out)


def evaluate_processing(predictor,ref):
    t,eta=ref[:,1],ref[:,2]
    pred=np.exp(predictor(t,np.full_like(t,20.)))
    truth_cross=threshold_crossing(t,eta);pred_cross=threshold_crossing(t,pred)
    # Scalar crossing errors are supplementary; all-replicate decision errors
    # remain available even if the temperature window has two boundaries.
    mae=None
    if truth_cross['status']=='unique' and pred_cross['status']=='unique':
        mae=abs(truth_cross['crossings_celsius'][0]-pred_cross['crossings_celsius'][0])
    tt=np.linspace(t[0],t[-1],1001)
    reference_eligible=np.interp(tt,t,np.log(eta))<=np.log(.4)
    predicted_eligible=predictor(tt,np.full_like(tt,20.))<=np.log(.4)
    mismatch=reference_eligible!=predicted_eligible
    falseeligible=predicted_eligible&~reference_eligible
    length=lambda b:float(np.trapezoid(b.astype(float),tt))
    return dict(processing_measured_grid_disagreement=float(np.mean((pred<=.4)!=(eta<=.4))),
                processing_measured_grid_falseeligible=float(np.mean((pred<=.4)&(eta>.4))),
                processing_window_symmetric_difference_celsius=length(mismatch),
                processing_window_falseeligible_celsius=length(falseeligible),
                processing_reference_window_length_celsius=length(reference_eligible),
                reference_crossing_status=truth_cross['status'],predicted_crossing_status=pred_cross['status'],
                unique_crossing_absolute_error_celsius=mae),dict(reference=truth_cross,prediction=pred_cross),pred


def metrics(observed,logprediction):
    observed=np.asarray(observed);prediction=np.exp(logprediction);positive=observed>0
    return dict(n=len(observed),n_log=int(positive.sum()),
                relative_l2=float(np.linalg.norm(prediction-observed)/np.linalg.norm(observed)),
                rmse_pa_s=float(np.sqrt(np.mean((prediction-observed)**2))),
                log_rmse=float(np.sqrt(np.mean((logprediction[positive]-np.log(observed[positive]))**2))))


def run(output):
    output=Path(output);reports=output/'reports';reports.mkdir(exist_ok=True)
    if (output/'EVALUATION_COMPLETE.json').exists():raise FileExistsError('Completed evaluation exists; use report rebuild separately')
    assert (output/'TRAINING_COMPLETE.json').exists()
    data=pd.read_csv(output/'observations.csv');specs=json.loads((output/'training_specs.json').read_text())
    grid=np.stack(np.meshgrid(np.linspace(-1,1,61),np.linspace(-1,1,61),indexing='ij'),axis=-1).reshape(-1,2)
    physical_grid=np.stack(np.meshgrid(np.linspace(-1,1,61),np.linspace(-1,1,201),indexing='ij'),axis=-1).reshape(-1,2)
    pinv={d:np.linalg.pinv(design(grid,d)) for d in (1,2,3)}
    checkpoints={step:dict(np.load(output/'models'/f'checkpoint_{step}.npz')) for step in (2000,6000)}
    rows=[];models=[];crossings=[];flows=[];predictions=[];observation_ids=[];process_pred=[];process_ref=[]
    teacher_log_predictions={};training_recheck=[];direct_seen=set();start=time.time()
    for stage in (2000,6000):
        checkpoint=checkpoints[stage]
        for spec in specs:
            i=spec['index'];weights={name:checkpoint[name][i] for name in ('w0','b0','w2','b2','w4','b4')}
            teacher=Predictor(spec,weights=weights)
            train=data.iloc[spec['training_row_ids']];test=data.iloc[spec['heldout_row_ids']]
            xtrain=teacher.x(train.temperature.to_numpy(),train.rate.to_numpy())
            recalculated=float(np.mean((teacher.predict_x(xtrain)-np.log(train.viscosity.to_numpy()))**2))
            training_recheck.append(abs(recalculated-float(checkpoint['loss'][i])))
            reference=processing_reference(test)
            candidates=[('neural',teacher,None)]
            teachergrid=teacher.predict_x(grid)
            for degree in (1,2,3):
                coeff=pinv[degree]@teachergrid
                candidates.append((f'frozen_d{degree}',Predictor(spec,coefficients=coeff,degree=degree),coeff))
            key=(spec['arm'],spec['material'],spec['fold'])
            if key not in direct_seen:
                direct_seen.add(key)
                for degree in (1,2,3):
                    coeff,_=fit_log_polynomial(xtrain,np.log(train.viscosity.to_numpy()),degree)
                    candidates.append((f'direct_d{degree}',Predictor(spec,coefficients=coeff,degree=degree),coeff))
            teacher_pred=teacher(test.temperature.to_numpy(),test.rate.to_numpy())
            for method,predictor,coeff in candidates:
                direct=method.startswith('direct')
                model_id=len(models)
                meta=dict(model_id=model_id,arm=spec['arm'],material=spec['material'],fold=spec['fold'],
                          seed=-1 if direct else spec['seed'],stage=0 if direct else stage,method=method,
                          spec_index=i,degree=predictor.degree,
                          coefficients=None if coeff is None else coeff.tolist())
                models.append(meta)
                prediction=predictor(test.temperature.to_numpy(),test.rate.to_numpy())
                if not np.isfinite(prediction).all() or not np.isfinite(np.exp(prediction)).all():
                    raise FloatingPointError(f'Nonfinite prediction for {meta}')
                row={k:v for k,v in meta.items() if k not in ('coefficients',)}
                primary=test.rate.to_numpy()>=10
                for scope,mask in [('all',np.ones(len(test),bool)),('high_shear',primary),('low_shear',~primary)]:
                    row.update({scope+'_'+k:v for k,v in metrics(test.viscosity.to_numpy()[mask],prediction[mask]).items()})
                row['heldout_log_compression_rmse']=None if direct else float(np.sqrt(np.mean((prediction-teacher_pred)**2)))
                margin=predictor.margin(physical_grid)
                row['minimum_sampled_stress_margin']=float(np.min(margin))
                row['sampled_nonmonotone_points']=int((margin<=0).sum())
                processing,cross,pp=evaluate_processing(predictor,reference);row.update(processing)
                crossings.append(dict(model_id=model_id,**cross))
                predictions.append(prediction);observation_ids.append(test.index.to_numpy());process_pred.append(pp);process_ref.append(reference)
                rows.append(row)
                if spec['arm']=='primary_high_shear' and ((stage==6000 and method in ('neural','frozen_d2')) or method=='direct_d2'):
                    fulltrain=data[(data.material==spec['material'])&(data.replicate!=spec['fold'])]
                    for nominal in (35,75,125):
                        context=[]
                        for _,f in fulltrain[fulltrain.nominal_temperature==nominal].groupby('replicate'):
                            f=f[f.rate>=10]
                            t40,eta40=interpolate_condition(f,40.)
                            context.append((t40,eta40,float(f.rate.min()),float(f.rate.max())))
                        context=np.asarray(context);t_eval=float(context[:,0].mean());wall=40*float(np.exp(np.log(context[:,1]).mean()))
                        lower=float(context[:,2].max());upper=float(context[:,3].min())
                        flow=slit_flow_band(lambda g:predictor(t_eval,g),min_rate=lower,max_rate=upper,wall_stress=wall)
                        flows.append(dict(model_id=model_id,nominal_temperature=nominal,actual_training_mean_temperature=t_eval,
                                          min_supported_rate=lower,max_supported_rate=upper,**flow))
            if i%25==0:print(json.dumps(dict(stage=stage,teacher_index=i,models=len(models),seconds=time.time()-start)),flush=True)
    pd.DataFrame(rows).to_csv(reports/'by_model.csv',index=False)
    pd.DataFrame(flows).to_csv(reports/'flow_bands.csv',index=False)
    (reports/'model_definitions.json').write_text(json.dumps(models,indent=2))
    (reports/'crossing_sets.json').write_text(json.dumps(crossings,indent=2))
    np.savez_compressed(reports/'heldout_predictions.npz',log_viscosity=np.asarray(predictions),observation_ids=np.asarray(observation_ids),
                        processing_predictions=np.asarray(process_pred),processing_reference=np.asarray(process_ref))
    table=pd.DataFrame(rows)
    metrics_cols=[x for x in table.columns if x.startswith(('high_shear_','all_','processing_')) and pd.api.types.is_numeric_dtype(table[x])]
    # Collapse NN initialization first, then treat the five held-out files as
    # repeated empirical endpoints. SD is descriptive, with shared training folds.
    paired=table.groupby(['arm','material','fold','stage','method'],as_index=False)[metrics_cols].mean()
    paired.to_csv(reports/'by_replicate_seedmean.csv',index=False)
    summary=paired.groupby(['arm','stage','method'])[metrics_cols].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(reports/'summary.csv')
    quadrature=[f['relative_quadrature_difference'] for f in flows if f['status']=='valid_conditional_band']
    integrity=dict(status='passed',teachers=200,checkpoints=2,evaluated_model_instances=len(models),
                   unique_direct_fits=3*len(direct_seen),raw_experimental_test_runs=50,
                   max_saved_training_loss_recheck_difference=max(training_recheck),
                   max_flow_relative_quadrature_difference=max(quadrature,default=0),
                   valid_flow_panels=len(quadrature),all_flow_panels=len(flows),
                   flow_status_counts=pd.Series([f['status'] for f in flows]).value_counts().to_dict(),
                   heldout_row_count_per_model=250,nonfinite_saved_predictions=0,
                   protocol_sha256=hashlib.sha256((output/'PROTOCOL.json').read_bytes()).hexdigest(),seconds=time.time()-start)
    if max(training_recheck)>1e-10 or max(quadrature,default=0)>2e-7:
        integrity['status']='failed_numerical_check'
        (output/'INTEGRITY_CHECKS.json').write_text(json.dumps(integrity,indent=2))
        raise AssertionError(integrity)
    (output/'INTEGRITY_CHECKS.json').write_text(json.dumps(integrity,indent=2))
    (output/'EVALUATION_COMPLETE.json').write_text(json.dumps(integrity,indent=2))
    print(json.dumps(integrity,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
