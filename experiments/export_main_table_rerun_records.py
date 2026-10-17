"""Export completed original-protocol reruns as portable frozen evidence.

Run after rerun_main_table_records.py and rerun_main_table_pysr.py. The exporter
reads trusted native PySR checkpoints from this project's own completed runs
to capture all effective Python parameters. It performs no fitting or search.
Published models use numeric neural state arrays and exact symbolic exports.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from rerun_main_table_pysr import configure_runtime


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(path.read_text())
def csv_write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=ROOT/'outputs/table2-table4-rerun-2026-10-17')
    parser.add_argument('--destination',type=Path,default=ROOT/'reproducibility/paper-records-2026-10-16/main_table_rerun')
    args=parser.parse_args();source=args.input_root.resolve();destination=args.destination.resolve()
    assert not destination.exists(),destination
    assert read(source/'excitation_COMPLETE.json')['status']=='complete'
    assert read(source/'pysr-teachers_COMPLETE.json')['status']=='complete'
    assert read(source/'pysr_search_COMPLETE.json')['status']=='complete'
    configure_runtime(source)
    import pysr
    import numpy as np
    destination.mkdir(parents=True)
    origins={}

    def copy(path,target):
        target=destination/target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        origins[str(target.relative_to(destination))]=dict(source=str(path.relative_to(ROOT)),sha256=sha(path))

    def export_teacher(path,suite):
        record=read(path/'record.json');assert record['status']=='complete'
        target=Path(suite)/path.name
        for name in ['record.json','observations.npz','neural_state.npz','curves.npz','neural_validation.npz','history.json']:
            copy(path/name,target/name)
        return record,target

    table2=[];aggregate=[]
    for path in sorted((source/'excitation').iterdir()):
        record,_=export_teacher(path,'table2');table2.append(record)
    assert len(table2)==18
    for case in ['case_a','case_b','case_exp']:
        for setting in ['low','high']:
            group=[row for row in table2 if row['protocol']['case']==case and row['protocol']['setting']==setting]
            assert len(group)==3
            row=dict(case=case,regime=setting,n=3)
            for key in ['bin_coverage','weak_diffusion_energy','final_weak_loss','relative_error_D','relative_error_R','unseen_rollout_relative_l2']:
                values=[value['excitation']['state_bin_coverage' if key=='bin_coverage' else key]
                        if key in ['bin_coverage','weak_diffusion_energy'] else value['metrics'][key] for value in group]
                row[key+'_mean']=statistics.mean(values);row[key+'_sd']=statistics.pstdev(values)
            aggregate.append(row)
    csv_write(destination/'table2_aggregate.csv',aggregate)
    copy(source/'table2_domain_summary.csv','table2_domain_summary.csv')
    table4=[];branch_count=0
    for path in sorted((source/'pysr').iterdir()):
        record,target=export_teacher(path,'table4')
        for name in ['restricted.json','restricted_validation.npz','pysr_validation.json','pysr_validation.npz']:
            copy(path/name,target/name)
        branches={}
        for name in ['diffusion','reaction']:
            folder=path/'search'/name;row=read(folder/'record.json');assert row['status']=='complete'
            native=folder/row['native_checkpoint'];assert native.is_file()
            with native.open('rb') as stream:model=pickle.load(stream)
            parameters=model.get_params(deep=False)
            parameters['output_directory']='native'
            parameters['logger_spec']=None
            parameters['observed_cycle_logger']='experiments/main_table_search_logger.jl'
            # All mathematical search parameters are primitive JSON values.
            json.dumps(parameters,allow_nan=False)
            save(destination/target/'search'/name/'parameters.json',parameters)
            public=dict(row)
            public.pop('native_checkpoint')
            public['model_export']='expression_sympy_srepr; exact constructor representation verified against native PySR predictions'
            save(destination/target/'search'/name/'record.json',public)
            origins[str(target/'search'/name/'record.json')]=dict(source=str((folder/'record.json').relative_to(ROOT)),
                sha256=sha(folder/'record.json'),metadata_change='Private native-checkpoint locator replaced by the delivered exact expression export; numeric fields unchanged')
            for filename in ['cycles.csv','prediction.npz','hall_of_fame.csv']:
                copy(folder/filename,target/'search'/name/filename)
            branches[name]=row;branch_count+=1
        restricted=read(path/'restricted.json');validation=read(path/'pysr_validation.json')
        for method in ['restricted','PySR']:
            metrics=restricted['metrics'] if method=='restricted' else {name:row['metrics'] for name,row in branches.items()}
            values=dict(setting=record['protocol']['setting'],seed=record['protocol']['seed'],method=method,
                        unseen=restricted['unseen_relative_l2'] if method=='restricted' else validation['unseen_relative_l2'])
            for branch,label in [('diffusion','D'),('reaction','R')]:
                values['surrogate_'+label]=metrics[branch]['surrogate_error']
                values['true_'+label]=metrics[branch]['true_error']
                values['BIR_'+label]=metrics[branch]['BIR']
                values['complexity_'+label]=restricted['expressions'][branch]['complexity'] if method=='restricted' else branches[branch]['complexity']
                values['teacher_'+label]=record['metrics']['relative_error_'+label]
            table4.append(values)
    assert branch_count==12 and len(table4)==12
    csv_write(destination/'table4_by_seed.csv',table4)
    summary=[]
    for setting in ['clean','noise5']:
        for method in ['restricted','PySR']:
            group=[row for row in table4 if row['setting']==setting and row['method']==method]
            assert len(group)==3
            row=dict(setting=setting,method=method,n=3)
            for key in [key for key in group[0] if key not in ['setting','seed','method']]:
                values=[value[key] for value in group]
                row[key+'_mean']=statistics.mean(values);row[key+'_sd']=statistics.pstdev(values)
            summary.append(row)
    csv_write(destination/'table4_summary.csv',summary)
    copy(source/'pysr_search_protocol.json','pysr_search_protocol.json')
    copy(source/'pysr_search_COMPLETE.json','pysr_search_COMPLETE.json')
    for filename in ['Project.toml','Manifest.toml']:
        copy(source/'julia-runtime/project'/filename,Path('environment')/filename)
    save(destination/'environment/python_packages.json',dict(python=sys.version,
        packages=sorted([dict(name=dist.metadata['Name'],version=dist.version) for dist in importlib.metadata.distributions()],key=lambda row:row['name'].lower()),
        pysr_version=pysr.__version__,neural_device='cpu',neural_threads=1))
    save(destination/'PROVENANCE.json',dict(status='complete',table2_saved_models=18,table4_saved_teachers=6,
        pysr_completed_searches=12,source_core_signature=table2[0]['core_signature'],
        executed_source_sha256={name:sha(ROOT/'experiments'/name) for name in
            ['rerun_main_table_records.py','rerun_main_table_pysr.py','main_table_search_logger.jl']},
        commands=[['python','experiments/rerun_main_table_records.py','--suite','excitation'],
                  ['python','experiments/rerun_main_table_records.py','--suite','pysr-teachers'],
                  ['python','experiments/rerun_main_table_pysr.py'],
                  ['python','experiments/export_main_table_rerun_records.py']],
        copied_input_hashes=origins,
        note='Original-protocol reruns. Symbolic models are exact executable exports, neural models use native numeric state arrays. No fitted model in this cohort is represented solely by rounded printed coefficients.'))
    print(json.dumps(dict(status='complete',table2_models=18,table4_teachers=6,pysr_searches=12,
                         summary=summary),indent=2))


if __name__=='__main__':main()
