"""Declared post-result fidelity/complexity and empirical-reference follow-up."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rheology_core import design,fit_log_polynomial,slit_flow_band
from rheology_evaluate import Predictor,metrics,processing_reference,evaluate_processing,interpolate_condition


def empirical_curve(train,rate):
    points=[]
    for nominal,f in train.groupby('nominal_temperature',sort=True):
        values=[]
        for _,run in f.groupby('replicate'):
            if rate<run.rate.min() or rate>run.rate.max():return None
            t,eta=interpolate_condition(run,rate);values.append((t,np.log(eta)))
        points.append(np.mean(values,axis=0))
    return np.asarray(points)


def empirical_predict(train,temperatures,rates):
    out=[]
    for t,g in zip(temperatures,rates):
        curve=empirical_curve(train,float(g))
        if curve is None or t<curve[0,0] or t>curve[-1,0]:out.append(np.nan)
        else:out.append(float(np.interp(t,curve[:,0],curve[:,1])))
    return np.asarray(out)


def flow_panels(data,spec,predictor):
    train=data[(data.material==spec['material'])&(data.replicate!=spec['fold'])&(data.rate>=10)]
    out=[]
    for nominal in (35,75,125):
        context=[]
        for _,f in train[train.nominal_temperature==nominal].groupby('replicate'):
            t,eta=interpolate_condition(f,40.);context.append((t,eta,f.rate.min(),f.rate.max()))
        c=np.asarray(context);t=float(c[:,0].mean());wall=40*float(np.exp(np.log(c[:,1]).mean()))
        lo,hi=float(c[:,2].max()),float(c[:,3].min())
        result=slit_flow_band(lambda g:predictor(t,g),min_rate=lo,max_rate=hi,wall_stress=wall)
        out.append(dict(nominal_temperature=nominal,actual_training_mean_temperature=t,min_supported_rate=lo,max_supported_rate=hi,**result))
    return out


def run(output):
    output=Path(output);out=output/'complexity';out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists():raise FileExistsError('Completed follow-up exists')
    protocol=json.loads((output/'COMPLEXITY_FOLLOWUP_PROTOCOL.json').read_text())
    assert protocol['degrees']==[4,6,8]
    data=pd.read_csv(output/'observations.csv');specs=json.loads((output/'training_specs.json').read_text())
    checkpoint=dict(np.load(output/'models/checkpoint_6000.npz'))
    grid=np.stack(np.meshgrid(np.linspace(-1,1,61),np.linspace(-1,1,61),indexing='ij'),axis=-1).reshape(-1,2)
    physical=np.stack(np.meshgrid(np.linspace(-1,1,61),np.linspace(-1,1,201),indexing='ij'),axis=-1).reshape(-1,2)
    pinv={d:np.linalg.pinv(design(grid,d)) for d in (4,6,8)}
    models=[];rows=[];flows=[];crossings=[];seen=set();baseline_rows=[];common_rows=[]
    for spec in specs:
        if spec['arm']!='primary_high_shear':continue
        i=spec['index'];weights={name:checkpoint[name][i] for name in ('w0','b0','w2','b2','w4','b4')}
        teacher=Predictor(spec,weights=weights);train=data.iloc[spec['training_row_ids']];test=data.iloc[spec['heldout_row_ids']]
        testhigh=test[test.rate>=10];xtrain=teacher.x(train.temperature.to_numpy(),train.rate.to_numpy())
        teachergrid=teacher.predict_x(grid);reference=processing_reference(test)
        candidates=[]
        for degree in (4,6,8):
            coefficients=pinv[degree]@teachergrid
            candidates.append((f'frozen_d{degree}',Predictor(spec,coefficients=coefficients,degree=degree),coefficients,None))
        key=(spec['material'],spec['fold'])
        if key not in seen:
            seen.add(key)
            for degree in (4,6,8):
                coefficients,singular=fit_log_polynomial(xtrain,np.log(train.viscosity.to_numpy()),degree)
                candidates.append((f'direct_d{degree}',Predictor(spec,coefficients=coefficients,degree=degree),coefficients,float(singular[0]/singular[-1])))
        for method,predictor,coefficients,condition in candidates:
            direct=method.startswith('direct')
            meta=dict(model_id=len(models),arm=spec['arm'],material=spec['material'],fold=spec['fold'],seed=-1 if direct else spec['seed'],
                      stage=0 if direct else 6000,method=method,spec_index=i,degree=predictor.degree,coefficients=coefficients.tolist())
            models.append(meta);row={k:v for k,v in meta.items() if k!='coefficients'}
            pred=predictor(testhigh.temperature.to_numpy(),testhigh.rate.to_numpy())
            row.update({'high_shear_'+k:v for k,v in metrics(testhigh.viscosity.to_numpy(),pred).items()})
            row['heldout_log_compression_rmse']=None if direct else float(np.sqrt(np.mean((pred-teacher(testhigh.temperature.to_numpy(),testhigh.rate.to_numpy()))**2)))
            row['projection_grid_log_compression_rmse']=None if direct else float(np.sqrt(np.mean((predictor.predict_x(grid)-teachergrid)**2)))
            row['design_condition_number']=condition
            margin=predictor.margin(physical);row['minimum_sampled_stress_margin']=float(margin.min());row['sampled_nonmonotone_points']=int((margin<=0).sum())
            processing,cross,_=evaluate_processing(predictor,reference);row.update(processing);crossings.append(dict(model_id=meta['model_id'],**cross));rows.append(row)
            if predictor.degree==8:
                flows.extend(dict(model_id=meta['model_id'],**f) for f in flow_panels(data,spec,predictor))
        if i%20==0:print(json.dumps(dict(followup_teacher=i,models=len(models))),flush=True)
    pd.DataFrame(rows).to_csv(out/'by_model.csv',index=False)
    pd.DataFrame(flows).to_csv(out/'flow_bands.csv',index=False)
    (out/'model_definitions.json').write_text(json.dumps(models,indent=2));(out/'crossing_sets.json').write_text(json.dumps(crossings,indent=2))
    originals=json.loads((output/'reports/model_definitions.json').read_text())
    originals=[dict(m,collection='initial') for m in originals if m['arm']=='primary_high_shear' and m['stage'] in (0,6000)]
    pool=originals+[dict(m,collection='complexity') for m in models]
    for (material,fold),frame in data.groupby(['material','replicate']):
        train=data[(data.material==material)&(data.replicate!=fold)&(data.rate>=10)]
        test=frame[frame.rate>=10];base=empirical_predict(train,test.temperature.to_numpy(),test.rate.to_numpy());eligible=np.isfinite(base)
        ref=processing_reference(frame);curve=empirical_curve(train,20.)
        lo=max(float(ref[0,1]),float(curve[0,0]));hi=min(float(ref[-1,1]),float(curve[-1,0]));tt=np.linspace(lo,hi,1001)
        truth=np.interp(tt,ref[:,1],np.log(ref[:,2]))<=np.log(.4)
        base_elig=np.interp(tt,curve[:,0],curve[:,1])<=np.log(.4)
        window=lambda pred:dict(common_window_symmetric_difference_celsius=float(np.trapezoid((pred!=truth).astype(float),tt)),
                                common_window_falseeligible_celsius=float(np.trapezoid((pred&~truth).astype(float),tt)))
        bm=dict(material=material,fold=int(fold),n_eligible=int(eligible.sum()),n_total=len(test),coverage=float(eligible.mean()),
                common_window_low=lo,common_window_high=hi,**metrics(test.viscosity.to_numpy()[eligible],base[eligible]),**window(base_elig))
        baseline_rows.append(bm)
        for m in pool:
            if m['material']!=material or m['fold']!=fold:continue
            spec=specs[m['spec_index']]
            weights=None if m['coefficients'] is not None else {name:checkpoint[name][spec['index']] for name in ('w0','b0','w2','b2','w4','b4')}
            predictor=Predictor(spec,weights=weights,coefficients=m['coefficients'],degree=m['degree'])
            pred=predictor(test.temperature.to_numpy()[eligible],test.rate.to_numpy()[eligible])
            common_rows.append(dict(collection=m['collection'],model_id=m['model_id'],material=material,fold=int(fold),seed=m['seed'],method=m['method'],
                                    n_eligible=int(eligible.sum()),**metrics(test.viscosity.to_numpy()[eligible],pred),
                                    **window(predictor(tt,np.full_like(tt,20.))<=np.log(.4))))
    pd.DataFrame(baseline_rows).to_csv(out/'empirical_training_baseline.csv',index=False)
    pd.DataFrame(common_rows).to_csv(out/'common_support_model_comparison.csv',index=False)
    q=[f['relative_quadrature_difference'] for f in flows if f['status']=='valid_conditional_band']
    result=dict(status='passed',new_neural_trainings=0,all_final_primary_teachers=100,frozen_fits=300,unique_direct_fits=150,
                flow_panels=len(flows),flow_status_counts=pd.Series([f['status'] for f in flows]).value_counts().to_dict(),
                max_flow_relative_quadrature_difference=max(q,default=0),baseline_whole_replicates=len(baseline_rows),
                baseline_eligible_observations=int(sum(x['n_eligible'] for x in baseline_rows)),
                baseline_total_observations=int(sum(x['n_total'] for x in baseline_rows)))
    if max(q,default=0)>2e-7:raise AssertionError(result)
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
