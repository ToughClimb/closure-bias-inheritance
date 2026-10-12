"""Command-level failure, cache and complete-comparison tests without Julia."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import artifact_io
import project_paths
import revision_core as core

REPO=Path(__file__).resolve().parents[1]
IDENT='main_case_exp_clean_0_default'
CONTROLS=['weak_poly','strong_poly','spacetime_poly','refit_weak','refit_data']


def run_cli(script,*args,work_root,input_root=None,extra_env=None):
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    if extra_env:env.update(extra_env)
    command=[sys.executable,str(REPO/'revision'/script),'--work-root',str(work_root)]
    if input_root:command+=['--input-root',str(input_root)]
    return subprocess.run(command+list(args),cwd=REPO,env=env,text=True,capture_output=True,timeout=60)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def neural_input(root):
    row=dict(id=IDENT,status='complete',config=dict(group='main',case='case_exp',setting='clean',seed=0,epochs=2),data_hash='observations')
    path=root/'evidence/raw'/f'{IDENT}.json';write_json(path,row)
    curves=root/'outputs/runs'/IDENT/'curves_2.npz';curves.parent.mkdir(parents=True,exist_ok=True)
    grid=np.linspace(0,1.8,512);np.savez(curves,grid=grid,D=np.full_like(grid,.03),R=np.zeros_like(grid))
    return path,row,curves


def test_pysr_cli_failure_cache_and_retry_remain_failures(tmp_path):
    source=tmp_path/'inputs';path,row,_=neural_input(source)
    fake=tmp_path/'backend';fake.mkdir()
    (fake/'pysr.py').write_text("import sys\nassert 'torch' not in sys.modules\nraise RuntimeError('Injected Julia startup failure')\n")
    work=tmp_path/'work';env={'PYTHONPATH':str(fake)}
    fresh=run_cli('run_pysr_suite.py','--limit','1',work_root=work,input_root=source,extra_env=env)
    assert fresh.returncode!=0
    dest=work/'evidence/pysr'/f'{IDENT}.json';failed=json.loads(dest.read_text())
    assert failed['status']=='failed' and 'Injected Julia startup failure' in failed['error']
    cached=run_cli('run_pysr_suite.py','--limit','1',work_root=work,input_root=source,extra_env=env)
    assert cached.returncode!=0 and 'cached_failed' in cached.stdout
    watched=run_cli('run_pysr_suite.py','--watch','--expected','1',work_root=work,input_root=source,extra_env=env)
    assert watched.returncode!=0 and 'cached_failed' in watched.stdout
    retried=run_cli('run_pysr_suite.py','--limit','1','--retry-failed',work_root=work,input_root=source,extra_env=env)
    assert retried.returncode!=0
    archives=list((work/'evidence/failures/pysr').glob('*.json'))
    assert len(archives)==1 and json.loads(archives[0].read_text())==failed


def test_pysr_help_never_initializes_julia(tmp_path):
    fake=tmp_path/'backend';fake.mkdir();(fake/'pysr.py').write_text("raise RuntimeError('Julia must not initialize for help')\n")
    result=run_cli('run_pysr_suite.py','--help',work_root=tmp_path/'work',extra_env={'PYTHONPATH':str(fake)})
    assert result.returncode==0 and '--retry-failed' in result.stdout


@pytest.mark.parametrize('change',['budget','model','source','configuration','metrics'])
def test_pysr_cache_checks_frozen_target_and_budget(tmp_path,monkeypatch,change):
    import run_pysr_suite as suite
    path,row,curve_path=neural_input(tmp_path)
    monkeypatch.setattr(suite,'ROOT',tmp_path);monkeypatch.setattr(project_paths,'ROOT',tmp_path)
    monkeypatch.setattr(project_paths,'INPUT_ROOT',tmp_path)
    prior=dict(id=IDENT,status='complete',config=row['config'],source_signature=artifact_io.source_signature(),
        model_signature=artifact_io.model_signature(row),settings=dict(suite.SEARCH_PROTOCOL,version='test',seed_D=0,seed_R=101),
        metrics={b:dict(true_error=.1,compression=.01,BIR=1.,common_norm_bound=.1) for b in ('D','R')},
        expressions={b:dict(expression='0.03' if b=='D' else '0',fit_loss=0.,complexity=1) for b in ('D','R')})
    dest=tmp_path/'evidence/pysr'/f'{IDENT}.json';write_json(dest,prior)
    assert suite.run_one(path,SimpleNamespace(retry_failed=False))=='cached'
    if change=='budget':prior['settings']['niterations']=1
    elif change=='source':prior['source_signature']='different'
    elif change=='configuration':prior['config']={**row['config'],'epochs':5}
    elif change=='metrics':prior['metrics']['D'].pop('compression')
    else:
        grid=np.linspace(0,1.8,512);np.savez(curve_path,grid=grid,D=np.full_like(grid,.06),R=np.zeros_like(grid))
    write_json(dest,prior)
    with pytest.raises(ValueError):suite.run_one(path,SimpleNamespace(retry_failed=False))


def representative_records(root):
    metrics={key:.1 for key in ('D','R','D_visited','R_visited','min_D')}
    config=dict(group='main',case='case_exp',setting='clean',seed=0,epochs=150)
    raw=dict(id=IDENT,status='complete',config=config,data_hash='same',controls={name:dict(metrics=metrics) for name in CONTROLS},
        epochs={'150':dict(neural=metrics,frozen=metrics,compression={b:dict(BIR=1.) for b in ('D','R')})})
    methods=['neural','frozen']+CONTROLS
    patterns=['full_support','mean0.25_amplitude0.1','mean0.6_amplitude0.2','mean1.1_amplitude0.25','mean1.5_amplitude0.1']
    entries=[dict(pattern=p,info=dict(status='stable'),times=[0.,.1,.5],errors=[0.,.1,.2]) for p in patterns]
    long=dict(id=IDENT,status='complete',config=config,methods={m:copy.deepcopy(entries) for m in methods+['refit_pde']},
        forward_refit=dict(metrics=dict(D=.02,R=.009)))
    short=dict(id=IDENT,status='complete',config=config,methods={m:{p:dict(info=dict(status='stable'),trajectory_error=.03)
        for p in ('legacy','strict')} for m in methods})
    for directory,row in [('raw',raw),('rollout',long),('short_validation',short)]:write_json(root/'evidence'/directory/f'{IDENT}.json',row)
    return raw,long,short


@pytest.mark.parametrize('damage',[None,'truncate_all','truncate_one','time','method','policy','both_incomplete'])
def test_portability_cannot_pass_with_missing_validation(tmp_path,damage):
    reference=tmp_path/'reference';rerun=tmp_path/'rerun';representative_records(reference)
    _,long,short=representative_records(rerun)
    if damage=='truncate_all':
        for entries in long['methods'].values():entries.clear()
    elif damage=='truncate_one':long['methods']['neural'].pop()
    elif damage=='time':long['methods']['neural'][0]['times'].pop()
    elif damage=='method':long['methods'].pop('refit_pde')
    elif damage=='policy':short['methods']['neural'].pop('strict')
    elif damage=='both_incomplete':
        long['methods']['neural'].pop()
        write_json(reference/'evidence/rollout'/f'{IDENT}.json',long)
    write_json(rerun/'evidence/rollout'/f'{IDENT}.json',long);write_json(rerun/'evidence/short_validation'/f'{IDENT}.json',short)
    work=tmp_path/'work'
    write_json(work/'evidence/portability_reproduction.json',dict(status='passed',old=True))
    result=run_cli('verify_portability.py','--reference-root',str(reference),'--rerun-root',str(rerun),work_root=work)
    record=json.loads((work/'evidence/portability_reproduction.json').read_text())
    if damage is None:
        assert result.returncode==0,result.stderr
        assert record['status']=='passed' and record['metric_check_count']==37 and record['validation_check_count']==56
    else:assert result.returncode!=0 and record['status']=='failed'


def test_worker_cli_completes_tiny_training_and_reuses_checked_cache(tmp_path):
    config=tmp_path/'tasks.json';write_json(config,[dict(group='ablation',case='case_exp',setting='clean',seed=0,epochs=2,classification='runtime_pilot')])
    first=run_cli('run_suite.py','--config',str(config),'--workers','1',work_root=tmp_path/'work')
    assert first.returncode==0,first.stderr
    second=run_cli('run_suite.py','--config',str(config),'--workers','1',work_root=tmp_path/'work')
    assert second.returncode==0 and 'cached' in second.stdout
    write_json(config,json.loads(config.read_text())*2)
    duplicate=run_cli('run_suite.py','--config',str(config),'--workers','1',work_root=tmp_path/'work')
    assert duplicate.returncode!=0 and 'Duplicate task IDs' in duplicate.stderr


@pytest.mark.cuda
@pytest.mark.skipif(os.environ.get('CLOSURE_TEST_CUDA')!='1' or not torch.cuda.is_available(),reason='Opt-in local CUDA check')
@pytest.mark.parametrize('dimension',[1,2])
def test_cuda_optimizer_resume_matches_continuous_training(tmp_path,monkeypatch,dimension):
    monkeypatch.setattr(core,'DEVICE_OPTION','cuda');case=core.extra_case('case_exp')
    if dimension==1:
        raw=core.generate_dataset(case,core.SimulationConfig1D(nx=16,t_final=.005,save_every=5),2,
            seed=2,amplitude_range=case.value_range,initial_clip_range=case.value_range)
        train=lambda **kwargs:core.train(raw,case.value_range,init_seed=2,**kwargs)
    else:
        from run_2d import train_2d
        from closure_discovery.data_generation.rd_solver_2d import SimulationConfig2D,generate_dataset
        raw=generate_dataset(case,SimulationConfig2D(nx=8,ny=8,t_final=.002,save_every=5),2,seed=2,
            amplitude_range=case.value_range,num_modes=2,initial_clip_range=case.value_range)
        train=lambda **kwargs:train_2d(raw,case.value_range,seed=2,**kwargs)
    full,history,_=train(epochs=4)
    train(epochs=2,out_dir=tmp_path)
    (tmp_path/'history.json').write_text('{broken')
    resumed,resumed_history,_=train(epochs=4,out_dir=tmp_path)
    assert resumed_history==history
    for a,b in zip(full.parameters(),resumed.parameters(),strict=True):assert torch.equal(a,b)
