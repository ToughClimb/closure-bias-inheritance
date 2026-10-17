"""Run the original Table 4 search protocol on six saved rerun teachers.

Each branch retains complete search parameters, cycle/evaluation/termination
records, its full hall of fame, native private checkpoint and an executable
SymPy export. A read-only Julia logger records actual remaining population
cycles rather than treating requested iteration limits as observed stopping.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def configure_runtime(output):
    runtime=output/'julia-runtime'
    os.environ.setdefault('JULIA_DEPOT_PATH',str((runtime/'depot').resolve()))
    os.environ.setdefault('PYTHON_JULIAPKG_PROJECT',str((runtime/'project').resolve()))
    os.environ.setdefault('JULIA_NUM_THREADS','1')
    os.environ.setdefault('JULIA_NUM_PRECOMPILE_TASKS','1')
    library=Path(sysconfig.get_config_var('LIBDIR'))/sysconfig.get_config_var('LDLIBRARY')
    assert library.is_file(),library
    os.environ.setdefault('JULIA_PYTHONCALL_LIB',str(library))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/table2-table4-rerun-2026-10-17')
    parser.add_argument('--seed',type=int,choices=range(3))
    parser.add_argument('--setting',choices=['clean','noise5'])
    args=parser.parse_args();output=args.output_root.resolve()
    configure_runtime(output)
    # Initialize Julia before loading Torch. The targets were saved by the
    # separate teacher-training command, so this process never trains networks.
    import pysr
    from pysr import PySRRegressor
    from pysr.logger_specs import AbstractLoggerSpec
    from pysr.julia_import import jl
    import numpy as np
    import sympy as sp
    jl.include(str(ROOT/'experiments/main_table_search_logger.jl'))

    @dataclass
    class CycleLoggerSpec(AbstractLoggerSpec):
        path:str

        def create_logger(self):return jl.MainTableSearchAudit.create(self.path,320)
        def write_hparams(self,logger,hparams):pass
        def close(self,logger):jl.MainTableSearchAudit.finish(logger)

    settings=dict(binary_operators=['+','-','*'],unary_operators=['exp'],
        elementwise_loss='PySR default L2DistLoss (squared error)',model_selection='best',parsimony=1e-4,
        deterministic=True,parallelism='serial',niterations=40,populations=8,population_size=48,
        maxsize=30,timeout_in_seconds=180.,precision=32,early_stop_condition=None,max_evals=None,
        input_stream='devnull',support_points=256)
    versions=dict(pysr=pysr.__version__,julia=str(jl.seval('string(VERSION)')),
                  symbolic_regression=str(jl.seval('string(Base.pkgversion(SymbolicRegression))')))
    save(output/'pysr_search_protocol.json',dict(settings=settings,versions=versions,
         logger_source_sha256=sha(ROOT/'experiments/main_table_search_logger.jl'),
         runner_sha256=sha(__file__),backend_source_sha256=sha(str(jl.seval('pathof(SymbolicRegression)'))),
         stopping_observation='Actual remaining population cycles logged after every completed backend cycle. A normal return with cycles left is timeout: loss/evaluation/user-input/external stopping is disabled.',
         command=['python','experiments/rerun_main_table_pysr.py']))
    print(json.dumps(dict(event='runtime_ready',versions=versions)),flush=True)
    all_results=[]
    for setting in ([args.setting] if args.setting else ['clean','noise5']):
        for seed in ([args.seed] if args.seed is not None else range(3)):
            teacher=output/'pysr'/f'case_exp_{setting}_{seed}'
            trow=json.loads((teacher/'record.json').read_text());assert trow['status']=='complete'
            with np.load(teacher/'curves.npz',allow_pickle=False) as archive:
                curves={key:archive[key].astype(float) for key in archive.files}
            for branch in ['diffusion','reaction']:
                path=teacher/'search'/branch;record=path/'record.json'
                search_seed=seed+(101 if branch=='reaction' else 0)
                config=dict(settings,random_state=search_seed)
                if record.exists():
                    prior=json.loads(record.read_text())
                    assert prior['status']=='complete',record
                    assert prior['config']==config and prior['teacher_record_sha256']==sha(teacher/'record.json'),record
                    all_results.append(prior);continue
                path.mkdir(parents=True,exist_ok=True)
                value=dict(status='running',id=trow['id'],branch=branch,config=config,versions=versions,
                           teacher_record_sha256=sha(teacher/'record.json'),
                           started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                save(record,value);start=time.perf_counter()
                try:
                    logger=CycleLoggerSpec(str(path/'cycles.csv'))
                    model=PySRRegressor(model_selection='best',binary_operators=['+','-','*'],unary_operators=['exp'],
                        niterations=40,populations=8,population_size=48,maxsize=30,timeout_in_seconds=180.,
                        parsimony=1e-4,random_state=search_seed,deterministic=True,parallelism='serial',
                        progress=False,verbosity=0,precision=32,input_stream='devnull',early_stop_condition=None,
                        max_evals=None,logger_spec=logger,output_directory=str(path/'native'),run_id='search')
                    model.fit(curves['grid'][:,None],curves[branch+'_pred'])
                    seconds=time.perf_counter()-start
                    # The observer is a process-local logger, not part of the
                    # fitted mathematical model. Retain the portable native
                    # PySR checkpoint after closing and removing that observer.
                    model.logger_spec=None
                    model._checkpoint()
                    assert (path/'native/search/checkpoint.pkl').is_file()
                    total,remaining,evaluations,callbacks=jl.MainTableSearchAudit.statistics(str(path/'cycles.csv'))
                    total,remaining,callbacks=int(total),int(remaining),int(callbacks)
                    assert total==320 and callbacks>0 and 0<=remaining<320
                    reason='iteration_budget' if remaining==0 else 'timeout'
                    if reason=='timeout':assert seconds>=180.,(seconds,remaining)
                    expression=model.sympy();best=model.get_best();pred=model.predict(curves['grid'][:,None])
                    assert np.isfinite(pred).all()
                    # NumPy 2.x feature names may appear in SymPy's repr as
                    # Symbol(np.str_('x0')). Export a plain-string symbol so
                    # the exact expression can be parsed independently.
                    expression_repr=sp.srepr(expression).replace("np.str_('x0')", "'x0'")
                    restored=sp.sympify(expression_repr)
                    fn=sp.lambdify(sp.Symbol('x0'),restored,modules=['numpy'])
                    restored_pred=np.broadcast_to(fn(curves['grid']),pred.shape)
                    restore_error=float(np.max(np.abs(restored_pred-pred)))
                    assert restore_error<1e-10
                    model.equations_.drop(columns=[key for key in ['lambda_format','sympy_format','jax_format','torch_format']
                                                  if key in model.equations_]).to_csv(path/'hall_of_fame.csv',index=False)
                    np.savez_compressed(path/'prediction.npz',grid=curves['grid'],prediction=pred)
                    truth=curves[branch+'_true'];target=curves[branch+'_pred']
                    error=lambda a,b:float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-12))
                    value.update(status='complete',seconds=seconds,stopping_reason=reason,
                        total_population_cycles=total,completed_population_cycles=total-remaining,
                        remaining_population_cycles=remaining,cycle_callbacks=callbacks,evaluations=float(evaluations),
                        expression=str(expression),expression_sympy_srepr=expression_repr,
                        complexity=int(best['complexity']),fit_loss=float(best['loss']),
                        expression_roundtrip_max_absolute_difference=restore_error,
                        metrics=dict(surrogate_error=error(target,pred),true_error=error(truth,pred),
                                     BIR=error(truth,pred)/(trow['metrics']['relative_error_'+('D' if branch=='diffusion' else 'R')]+1e-12)),
                        native_checkpoint='native/search/checkpoint.pkl')
                    save(record,value);all_results.append(value)
                    print(json.dumps(dict(event='branch_complete',id=trow['id'],branch=branch,seconds=seconds,
                                          stop=reason,remaining=remaining,expression=str(expression))),flush=True)
                except Exception as error:
                    value.update(status='failed',error=repr(error),traceback=traceback.format_exc(),seconds=time.perf_counter()-start)
                    save(record,value);raise
            validation_record=teacher/'pysr_validation.json'
            if not validation_record.exists():
                from closure_discovery.data_generation.cases import CASE_BUILDERS, ReactionDiffusionCase
                from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D
                from closure_discovery.evaluation.rollout import compare_cases_on_shared_initial_conditions
                branches={name:json.loads((teacher/'search'/name/'record.json').read_text())
                          for name in ['diffusion','reaction']}
                functions={name:sp.lambdify(sp.Symbol('x0'),sp.sympify(row['expression_sympy_srepr']),modules=['numpy'])
                           for name,row in branches.items()}
                def evaluate(name,values):
                    values=np.asarray(values)
                    return np.broadcast_to(functions[name](values),values.shape).astype(float)
                case=CASE_BUILDERS['case_exp']()
                symbolic=ReactionDiffusionCase(name=trow['id']+'_pysr',description='Original-protocol PySR rerun',
                    diffusion=lambda values:evaluate('diffusion',values),
                    reaction=lambda values:evaluate('reaction',values),value_range=case.value_range)
                protocol=trow['protocol']
                rollout=compare_cases_on_shared_initial_conditions(true_case=case,predicted_case=symbolic,
                    config=SimulationConfig1D(**protocol['simulation']),num_trajectories=4,seed=seed+1000,
                    amplitude_range=tuple(protocol['amplitude_range']),initial_clip_range=tuple(protocol['initial_clip_range']),num_modes=4)
                np.savez_compressed(teacher/'pysr_validation.npz',times=rollout.true_dataset['t'],
                    initial=rollout.true_dataset['u0'],reference=rollout.true_dataset['u'],prediction=rollout.predicted_dataset['u'])
                save(validation_record,dict(status='complete',unseen_relative_l2=rollout.relative_l2,
                    branch_records_sha256={name:sha(teacher/'search'/name/'record.json') for name in branches},
                    validation_sha256=sha(teacher/'pysr_validation.npz')))
    if args.seed is None and args.setting is None:
        assert len(all_results)==12 and all(row['status']=='complete' for row in all_results)
        save(output/'pysr_search_COMPLETE.json',dict(status='complete',branches=12,
             stopping_counts={reason:sum(row['stopping_reason']==reason for row in all_results)
                              for reason in ['iteration_budget','timeout']}))


if __name__=='__main__':main()
