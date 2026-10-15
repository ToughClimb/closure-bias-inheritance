"""Read-only CUDA100 budget reports with a fixed common objective evaluator."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import zipfile

from make_manifest import implementation_signature
from run_study import context, read_manifest, terms
from study_io import DEFAULT_OUTPUT, REPO, STUDY, canonical_hash, file_hash, load_core, load_input


def main():
    cpu_script = REPO/'research/reviewer-sensitivity-2026-10-15-v2/plot_results.py'
    spec = importlib.util.spec_from_file_location('cpu_reporting_library', cpu_script)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    pd, np, plt = report.pd, report.np, report.plt
    manifest = read_manifest(STUDY/'MANIFEST.json')
    plan = json.loads((STUDY/'BUDGET_REPORTING_PLAN.json').read_text())
    assert len(manifest['tasks']) == plan['tasks'] == 100
    assert manifest['required_device'] == 'cuda' and implementation_signature() == manifest['implementation_signature']
    destination = DEFAULT_OUTPUT/'reports/budget_visualization'
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination/'REPORT_STATUS.json'
    state = dict(report_success=False, complete=False, cohort='fresh_cuda_long', source_device='cuda', evaluation_device='cpu',
                 generated_utc=datetime.now(timezone.utc).isoformat(), expected_tasks=100, expected_evaluation_pairs=300, expected_diagnostics=700,
                 implementation_signature=manifest['implementation_signature'], protocol_sha256=manifest['protocol_sha256'],
                 manifest_sha256=file_hash(STUDY/'MANIFEST.json'), reporting_source_sha256=file_hash(__file__),
                 reporting_plan_sha256=file_hash(STUDY/'BUDGET_REPORTING_PLAN.json'), cpu_plot_library_sha256=file_hash(cpu_script),
                 optimizer_steps=0, cpu_results_used_in_budget_lines=0, selection='Every declared CUDA task/seed/checkpoint; no true-error selection')
    report.json_write(status_path,state)
    archive_path=destination/'REPORT_INPUT_SNAPSHOT.zip'
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('MANIFEST.json','PROTOCOL.json','NUMERICAL_POLICY.json','GPU_COHORT_AMENDMENT.json','BUDGET_REPORTING_PLAN.json','report_budget.py'):
            archive.writestr(name,(STUDY/name).read_bytes())
        archive.writestr('cpu_plot_library.py',cpu_script.read_bytes())
        metrics, gradients, quality, inventory, sources = report.read_records(DEFAULT_OUTPUT,manifest,'full',archive)
    inventory.to_csv(destination/'task_inventory.csv',index=False)
    assert inventory.stage_ready.all() and len(metrics)==600 and len(quality)==700
    assert set(metrics.epoch)=={150,500,1500} and quality.fp64_pass.all()
    for frame in (metrics,gradients,quality):
        frame['source_cohort']='fresh_cuda_long'; frame['source_device']='cuda'
        frame['protocol_sha256']=manifest['protocol_sha256']; frame['implementation_signature']=manifest['implementation_signature']
    metrics.to_csv(destination/'metrics_by_seed.csv',index=False)
    quality.to_csv(destination/'diagnostic_quality_by_seed.csv',index=False)
    gradients.to_csv(destination/'gradient_by_seed.csv',index=False)
    report.summarize(metrics,['case','setting','variant','epoch','method'],['D','R','BIR_D','BIR_R','min_D','R0']).to_csv(destination/'metrics_summary.csv',index=False)
    paired = report.budget_contrasts(metrics)
    paired.to_csv(destination/'paired_budget_by_seed.csv',index=False)
    report.summarize(paired,['case','setting','variant','epoch','method','metric'],['difference']).to_csv(destination/'paired_budget_summary.csv',index=False)

    # Only model state is loaded. No optimizer object is created or restored.
    core=load_core(DEFAULT_OUTPUT,'cpu'); torch=core.torch; torch.set_num_threads(1)
    assert core.DEVICE_OPTION=='cpu'
    inputs={}
    objective_rows=[]; component_checks=[]; checkpoint_sources=[]
    for task in manifest['tasks']:
        ident=(task['case'],task['setting'],task['data_seed'])
        if ident not in inputs:
            inputs[ident]=load_input(core,DEFAULT_OUTPUT,*ident)
        observed,interval,meta=inputs[ident]
        result_path=DEFAULT_OUTPUT/'runs'/task['id']/'result.json'
        result=json.loads(result_path.read_text())
        assert result['status']=='complete' and result['completed_epoch']==1500 and result['identity']['device']=='cuda'
        for epoch in (150,500,1500):
            checkpoint=result_path.parent/f'checkpoint_{epoch}.pt'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert saved['epoch']==epoch and all(saved.get(k)==v for k,v in result['identity'].items())
            model=core.ReactionDiffusionClosure(input_range=interval)
            model.load_state_dict(saved['model'])
            assert all(p.device.type=='cpu' for p in model.parameters())
            ctx=context(core,model,observed,interval,manifest['protocol'],'space_weak','cpu')
            values=terms(core,ctx)
            values={k:float(v.detach()) for k,v in values.items()}
            assert all(np.isfinite(v) for v in values.values())
            native=result['gradient_diagnostics'][str(epoch)]
            if task['weak_variant']=='space_weak':
                for term,value in values.items():
                    expected=native['raw_losses'][term]
                    tolerance=plan['cross_device_component_verification']['atol']+plan['cross_device_component_verification']['rtol']*abs(expected)
                    error=abs(value-expected)
                    check=dict(id=task['id'],epoch=epoch,term=term,cpu_value=value,native_cuda_value=expected,absolute_error=error,tolerance=tolerance,passed=error<=tolerance)
                    component_checks.append(check)
                    if not check['passed']:
                        raise ArithmeticError(f'Fixed cross-device raw-component verification failed: {check}')
            common=sum(plan['uniform_reference_weights'][k]*v for k,v in values.items())
            row=dict(id=task['id'],case=task['case'],setting=task['setting'],seed=task['data_seed'],variant=task['variant'],epoch=epoch,
                     source_cohort='fresh_cuda_long',source_device='cuda',evaluation_device='cpu',
                     data_hash=result['data_hash'],observation_signature=result['observation_signature'],initial_model_sha256=result['initial_model_sha256'],
                     checkpoint_sha256=file_hash(checkpoint),protocol_sha256=manifest['protocol_sha256'],implementation_signature=manifest['implementation_signature'],
                     native_weak_variant=task['weak_variant'],native_postupdate_objective=native['total_loss'],common_space_weak_default_objective=common,
                     **{f'common_raw_{k}':v for k,v in values.items()})
            objective_rows.append(row)
            checkpoint_sources.append(dict(path=str(checkpoint.relative_to(DEFAULT_OUTPUT)),sha256=row['checkpoint_sha256']))
            del saved,model,ctx
        if len(objective_rows)%60==0:
            print(f'CPU read-only common objective evaluated for {len(objective_rows)}/300 saved CUDA states',flush=True)
    assert len(objective_rows)==300 and len(component_checks)==1440
    objective=pd.DataFrame(objective_rows)
    objective.to_csv(destination/'objective_by_seed.csv',index=False)
    pd.DataFrame(component_checks).to_csv(destination/'cross_device_component_verification.csv',index=False)
    report.summarize(objective,['case','setting','variant','epoch'],['native_postupdate_objective','common_space_weak_default_objective']).to_csv(destination/'objective_summary.csv',index=False)
    base=objective.loc[objective.epoch.eq(150),['case','setting','seed','variant','common_space_weak_default_objective']]
    changes=objective.loc[objective.epoch.gt(150)].merge(base,on=['case','setting','seed','variant'],suffixes=('','_150'),validate='many_to_one')
    changes['difference']=changes.common_space_weak_default_objective-changes.common_space_weak_default_objective_150
    changes.to_csv(destination/'common_objective_paired_budget_by_seed.csv',index=False)
    report.summarize(changes,['case','setting','variant','epoch'],['difference']).to_csv(destination/'common_objective_paired_budget_summary.csv',index=False)
    figures=report.Figures(destination/'figures',False,' [fresh CUDA100 cohort]')
    variants=manifest['protocol']['longer_variants']
    cells=[(c,s) for c in manifest['protocol']['cases'] for s in manifest['protocol']['settings']]
    for method in ('neural','frozen'):
        fig,axs=plt.subplots(4,2,figsize=(12,12),squeeze=False)
        for row,(case,setting) in enumerate(cells):
            for col,branch in enumerate(('D','R')):
                ax=axs[row,col]
                for index,variant in enumerate(variants):
                    subset=metrics.loc[metrics.case.eq(case)&metrics.setting.eq(setting)&metrics.variant.eq(variant)&metrics.method.eq(method)]
                    summary=subset.groupby('epoch')[branch].agg(['mean','std']).sort_index()
                    ax.errorbar(summary.index,summary['mean'],yerr=summary['std'],fmt='o-',ms=3,capsize=2,color=plt.get_cmap('tab10')(index),label=report.VARIANT_NAMES[variant])
                ax.set_title(f'{report.CASE_NAMES[case]}, {report.SETTING_NAMES[setting]}: {branch}')
                ax.set_xticks([150,500,1500]);ax.set_xlabel('Declared Adam epochs');ax.set_ylabel('Relative closure error');ax.grid(alpha=.18)
        report.legend_if_present(axs[0,0],fontsize=7,ncol=2)
        figures.save(fig,f'budget_errors_{method}',f'Budget dependence: {report.METHOD_NAMES[method]}',
                     'All 100 fresh CUDA fits, each trained from its own initial state through 1500 epochs. Mean ± sample SD, five paired seeds. No CPU endpoints or interrupted CPU continuation are used; fixed budgets do not certify convergence.')
    for field,name,label in (('common_space_weak_default_objective','budget_objective','Common space-weak default objective'),('native_postupdate_objective','budget_native_objective','Native post-update objective')):
        fig,axs=plt.subplots(2,2,figsize=(12,9),squeeze=False)
        for ax,(case,setting) in zip(axs.flat,cells):
            for index,variant in enumerate(variants):
                subset=objective.loc[objective.case.eq(case)&objective.setting.eq(setting)&objective.variant.eq(variant)]
                summary=subset.groupby('epoch')[field].agg(['mean','std']).sort_index()
                ax.errorbar(summary.index,summary['mean'],yerr=summary['std'],fmt='o-',ms=3,capsize=2,color=plt.get_cmap('tab10')(index),label=report.VARIANT_NAMES[variant])
            ax.set_title(f'{report.CASE_NAMES[case]}, {report.SETTING_NAMES[setting]}')
            ax.set_xticks([150,500,1500]);ax.set_xlabel('Declared Adam epochs');ax.set_ylabel(label);ax.set_yscale('symlog',linthresh=1e-7);ax.grid(alpha=.18)
        report.legend_if_present(axs[0,0],fontsize=7,ncol=2)
        caption=('Post hoc CPU evaluation of all 300 saved CUDA models using one fixed space-weak objective and all six original default weights. Mean ± sample SD, five seeds. This common score does not certify convergence or an optimizer gap.' if name=='budget_objective' else 'Native post-update weighted objective, mean ± sample SD, five seeds. Compare budgets within each arm; differently weighted native objectives are not comparable rankings across arms.')
        figures.save(fig,name,label,caption)
    with zipfile.ZipFile(archive_path,'a',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('checkpoint_sources.json',json.dumps(checkpoint_sources,indent=2))
        archive.writestr('objective_by_seed.csv',(destination/'objective_by_seed.csv').read_bytes())
        archive.writestr('cross_device_component_verification.csv',(destination/'cross_device_component_verification.csv').read_bytes())
    state.update(report_success=True,complete=True,stage_ready_tasks=100,actual_evaluation_pairs=300,actual_diagnostics=700,
                 source_records=sources,checkpoint_sources=checkpoint_sources,common_objective_states=300,cross_device_same_form_states=240,
                 cross_device_component_checks=1440,cross_device_all_passed=True,cross_device_max_error_over_tolerance=max(r['absolute_error']/r['tolerance'] for r in component_checks),
                 fp64_gates_passed=True,figures=figures.paths,report_input_snapshot_sha256=file_hash(archive_path))
    report.json_write(status_path,state)
    print(json.dumps({k:v for k,v in state.items() if k not in {'source_records','checkpoint_sources','figures'}},indent=2))


if __name__=='__main__':
    main()
