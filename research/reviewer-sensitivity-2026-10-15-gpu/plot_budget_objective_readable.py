"""CSV-only correction of budget objective axes; never loads models/checkpoints."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

STUDY=Path(__file__).resolve().parent
REPO=STUDY.parents[1]
ROOT=REPO/'outputs'/STUDY.name
SOURCE=ROOT/'reports/budget_visualization'
DEST=ROOT/'reports/budget_visualization-readable'
VARIANTS=['default','drop_strong','drop_anchor','reduced_space_weak','space_time_weak']
LABELS=['Default','No strong term','No anchor term','Reduced space weak','Space-time weak']
CELLS=[('case_a','clean'),('case_a','noise3'),('case_exp','clean'),('case_exp','noise3')]
CASE_NAMES={'case_a':'Case A','case_exp':'Case Exp'}
SETTING_NAMES={'clean':'Clean','noise3':'3% noise'}
FIELDS=[('common_space_weak_default_objective','budget_objective','Common space-weak default objective'),('native_postupdate_objective','budget_native_objective','Native post-update objective')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def summary_groups(objective,field):
    return {(case,setting,variant):objective.loc[objective.case.eq(case)&objective.setting.eq(setting)&objective.variant.eq(variant)].groupby('epoch')[field].agg(['mean','std']).sort_index()
            for case,setting in CELLS for variant in VARIANTS}


def artist_snapshot(ax,containers):
    return dict(yscale=ax.get_yscale(),ylim=list(map(float,ax.get_ylim())),data_lim_y=list(map(float,ax.dataLim.intervaly)),
                major_formatter=type(ax.yaxis.get_major_formatter()).__name__,major_locator=type(ax.yaxis.get_major_locator()).__name__,
                offset_text=ax.yaxis.get_offset_text().get_text(),yticks=list(map(float,ax.get_yticks())),tick_labels=[t.get_text() for t in ax.get_yticklabels()],
                artists=[dict(variant=variant,x=np.asarray(container.lines[0].get_xdata(),dtype=float).tolist(),
                              y=np.asarray(container.lines[0].get_ydata(),dtype=float).tolist(),
                              errorbar_segments=[segment.tolist() for collection in container.lines[2] for segment in collection.get_segments()]) for variant,container in containers])


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'figures').mkdir(exist_ok=True)
    source_csv=SOURCE/'objective_by_seed.csv'
    summary_csv=SOURCE/'objective_summary.csv'
    original_status=SOURCE/'REPORT_STATUS.json'
    receipts={str(p):sha(p) for p in (source_csv,summary_csv,original_status)}
    original=json.loads(original_status.read_text())
    assert original['report_success'] and original['complete'] and original['common_objective_states']==300
    objective=pd.read_csv(source_csv,float_precision='round_trip')
    expected=pd.read_csv(summary_csv,float_precision='round_trip')
    assert len(objective)==300 and not objective.duplicated(['case','setting','seed','variant','epoch']).any()
    assert objective.source_device.eq('cuda').all() and objective.source_cohort.eq('fresh_cuda_long').all()
    assert set(objective.epoch)=={150,500,1500}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170,'pdf.fonttype':42,'ps.fonttype':42,'axes.formatter.useoffset':False})
    diagnostic=[];stats=[];figures=[]
    # Recreate the original artists from CSV only, retaining pre-/post-draw data.
    # This makes the scale diagnosis reviewable without reevaluating any model.
    for field,name,label in FIELDS:
        groups=summary_groups(objective,field)
        fig,axs=plt.subplots(2,2,figsize=(12,9),squeeze=False)
        pending=[]
        for ax,(case,setting) in zip(axs.flat,CELLS):
            containers=[]
            for index,variant in enumerate(VARIANTS):
                values=groups[(case,setting,variant)]
                containers.append((variant,ax.errorbar(values.index,values['mean'],yerr=values['std'],fmt='o-',ms=3,capsize=2,color=plt.get_cmap('tab10')(index),label=LABELS[index])))
            before=artist_snapshot(ax,containers)
            ax.set_yscale('symlog',linthresh=1e-7)
            pending.append((ax,case,setting,containers,before))
        fig.canvas.draw()
        for ax,case,setting,containers,before in pending:
            diagnostic.append(dict(field=field,case=case,setting=setting,before_scale=before,after_symlog_draw=artist_snapshot(ax,containers)))
        plt.close(fig)

        fig,axs=plt.subplots(2,2,figsize=(12,9),squeeze=False)
        pending=[]
        for ax,(case,setting) in zip(axs.flat,CELLS):
            # Select the scale before adding artists; never reuse symlog axes.
            scale='log' if setting=='clean' or field=='native_postupdate_objective' else 'linear'
            ax.set_yscale(scale)
            containers=[];lowers=[];uppers=[]
            for index,variant in enumerate(VARIANTS):
                values=groups[(case,setting,variant)]
                means=values['mean'].to_numpy(copy=True);sds=values['std'].to_numpy(copy=True)
                assert np.isfinite(means).all() and np.isfinite(sds).all() and (means-sds>0).all()
                for epoch,mean,sd in zip(values.index,means,sds):
                    row=expected.loc[expected.case.eq(case)&expected.setting.eq(setting)&expected.variant.eq(variant)&expected.epoch.eq(epoch)&expected.metric.eq(field)]
                    assert len(row)==1
                    published_mean=float(row.iloc[0]['mean']);published_sd=float(row.iloc[0]['sample_sd'])
                    # The original plot uses pandas groupby.std while its CSV
                    # summary uses NumPy std; retain the plot's same reduction.
                    assert np.isclose(mean,published_mean,rtol=5e-14,atol=1e-15) and np.isclose(sd,published_sd,rtol=5e-14,atol=1e-15)
                    stats.append(dict(field=field,case=case,setting=setting,variant=variant,epoch=int(epoch),n=5,mean=float(mean),sample_sd=float(sd),mean_minus_sd=float(mean-sd),mean_plus_sd=float(mean+sd),matches_original_summary_to_float64_reduction_roundoff=True,mean_difference_from_published_summary=float(mean-published_mean),sd_difference_from_published_summary=float(sd-published_sd)))
                containers.append((variant,ax.errorbar(values.index.to_numpy(copy=True),means,yerr=sds,fmt='o-',ms=3,capsize=2,color=plt.get_cmap('tab10')(index),label=LABELS[index])))
                lowers.extend(means-sds);uppers.extend(means+sds)
            if scale=='linear':
                ax.set_ylim(0,max(uppers)*1.12)
                formatter=ScalarFormatter(useOffset=False);formatter.set_scientific(False)
                ax.yaxis.set_major_formatter(formatter)
            else:
                ax.set_ylim(min(lowers)/1.5,max(uppers)*1.5)
            ax.set_title(f'{CASE_NAMES[case]}, {SETTING_NAMES[setting]} ({scale} y-axis)')
            ax.set_xticks([150,500,1500]);ax.set_xlabel('Declared Adam epochs');ax.set_ylabel(label);ax.grid(alpha=.18)
            pending.append((ax,case,setting,containers))
        axs[0,0].legend(fontsize=7,ncol=2)
        title=label+' [fresh CUDA100 cohort]'
        fig.suptitle(textwrap.fill(title,width=120,break_long_words=False),fontsize=12)
        caption=('All 300 saved CUDA-model scores from the original CSV, mean ± sample SD across five paired seeds. Clean panels use positive log axes; noisy panels use linear axes starting at zero. The same fixed default space-weak score is used for every arm; it does not certify convergence or an optimizer gap.' if field=='common_space_weak_default_objective' else 'Original native post-update scores, mean ± sample SD across five paired seeds; all panels use positive log axes. Compare budgets within an arm; different weighted native objectives are not comparable rankings across arms.')
        fig.text(.5,.012,textwrap.fill(caption,width=180,break_long_words=False),ha='center',va='bottom',fontsize=7)
        fig.tight_layout(rect=(0,.075,1,.94))
        fig.canvas.draw()
        for ax,case,setting,containers in pending:
            snapshot=artist_snapshot(ax,containers)
            assert snapshot['ylim'][0]>=0
            diagnostic.append(dict(field=field,case=case,setting=setting,corrected_axes=snapshot))
        for suffix in ('pdf','png'):
            path=DEST/'figures'/f'{name}.{suffix}'
            fig.savefig(path)
            figures.append(dict(path=path.name,sha256=sha(path),source='csv_only_replot'))
        plt.close(fig)
    # Keep the chosen error figure paired with its untouched numerical rendering.
    for name in ('budget_errors_neural','budget_errors_frozen'):
        for suffix in ('pdf','png'):
            src=SOURCE/'figures'/f'{name}.{suffix}';dst=DEST/'figures'/src.name
            shutil.copyfile(src,dst)
            assert sha(src)==sha(dst)
            receipts[str(src)]=sha(src)
            figures.append(dict(path=dst.name,sha256=sha(dst),source='byte_identical_copy'))
    for name in ('objective_by_seed.csv','objective_summary.csv'):
        shutil.copyfile(SOURCE/name,DEST/name)
        assert sha(SOURCE/name)==sha(DEST/name)
    assert all(sha(path)==receipt for path,receipt in receipts.items())
    write(DEST/'AXIS_ARTIST_DIAGNOSIS.json',dict(classification='csv_only_plot_diagnosis',matplotlib=matplotlib.__version__,objective_records=300,records=diagnostic))
    write(DEST/'PLOTTED_MEAN_SD_VALUES.json',dict(source_csv_sha256=sha(source_csv),all120_group_statistics_match_original_summary_to_float64_reduction_roundoff=True,original_plot_reduction_preserved='pandas groupby mean/std, same as report_budget.py',groups=stats))
    state=dict(report_success=True,complete=True,classification='presentation_only_csv_replot',source_cohort='fresh_cuda_long',
               source_csv=str(source_csv),source_csv_sha256=sha(source_csv),source_summary_csv_sha256=sha(summary_csv),
               original_report=str(original_status),original_report_sha256=sha(original_status),original_artifacts_modified=False,
               generated_utc=datetime.now(timezone.utc).isoformat(),rendering_source=str(Path(__file__)),rendering_source_sha256=sha(__file__),
               implementation_signature=original['implementation_signature'],protocol_sha256=original['protocol_sha256'],
               objective_records=300,plotted_group_statistics=120,original_plot_mean_and_sample_sd_reduction_preserved=True,maximum_mean_difference_from_published_numpy_summary=max(abs(r['mean_difference_from_published_summary']) for r in stats),maximum_sd_difference_from_published_numpy_summary=max(abs(r['sd_difference_from_published_summary']) for r in stats),
               model_or_checkpoint_files_read=0,model_evaluations=0,optimizer_steps=0,scientific_csv_bytes_unchanged=True,
               figure_artifacts=figures,source_receipts=receipts,
               axis_policy='Common objective: clean positive log and noisy linear starting0; native objective: positive log. Scale chosen before artists, explicit positive bounds, no numeric offsets.')
    write(DEST/'REPORT_STATUS.json',state)
    print(json.dumps(state,indent=2))


if __name__=='__main__':
    main()
