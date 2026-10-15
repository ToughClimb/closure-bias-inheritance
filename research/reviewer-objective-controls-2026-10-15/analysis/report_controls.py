"""Read-only aggregation and mechanism plots from the complete coefficient fits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

STUDY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STUDY))
from objective_controls import DEFAULT_OUTPUT,DEFAULT_WEIGHTS,TERMS,file_hash,write_json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(root):
    rows=[];hashes={};solver_checks=[]
    for mode in ("shared","oracle"):
        complete=json.loads((root/mode/"COMPLETION.json").read_text())
        count=0
        for path in sorted((root/mode/"bundles").glob("*.json")):
            hashes[str(path.relative_to(root))]=file_hash(path)
            bundle=json.loads(path.read_text())
            for record in bundle["results"]:
                count+=1
                row={k:record[k] for k in ("case","data_seed","noise_level","clipping",
                    "oracle_arm","oracle","constraint","variant","weak_variant","phase","status")}
                row["mode"]=mode;row["source_input_id"]=record["source_input_id"]
                row.update({"weight_"+k:v for k,v in record["weights"].items()})
                if "solver" in record:
                    solver=record["solver"];ev=record["evaluation"]
                    row.update({k:ev[k] for k in ("E_D","E_R","R0","R001")})
                    row["minimum_diffusion"]=ev["minimum_diffusion"]["value"]
                    row["objective"]=solver.get("objective")
                    row["common_default_objective"]=sum(DEFAULT_WEIGHTS[k]*solver.get("losses",{}).get(k,np.nan) for k in TERMS)
                    row.update({"loss_"+k:v for k,v in solver.get("losses",{}).items()})
                    for j,v in enumerate(solver.get("coefficients",[])):row[f"coefficient_{j}"]=v
                    ls=solver.get("least_squares",solver)
                    for key in ("rank","condition_scaled","condition_unscaled","stationarity_relative"):
                        row["ls_"+key]=ls.get(key)
                    row["constraint_used_optimizer"]=solver.get("used_optimizer",False)
                    for key in ("kkt_stationarity_relative","kkt_complementarity_relative",
                                "primal_dual_gap_relative"):
                        row[key]=solver.get(key)
                rows.append(row)
        if count!=complete["actual_fits"]:raise AssertionError((mode,count,complete))
    return pd.DataFrame(rows),hashes


def save_figure(fig,path):
    fig.savefig(path.with_suffix(".png"),dpi=180,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


def plot_oracles(data,out):
    names=["default","drop_strong","space_time_weak"]
    titles=["Default hybrid","Without strong term","Space-time weak"]
    arms=["clean_features_clean_targets","clean_features_observed_targets",
          "observed_features_clean_targets","observed_features_observed_targets"]
    labels=["Clean features / clean targets","Clean features / noisy targets",
            "Noisy features / clean targets","Noisy features / noisy targets"]
    colors=["#333333","#009E73","#D55E00","#0072B2"]
    subset=data[(data["mode"]=="oracle")&(data.constraint=="unconstrained")&
                (data.clipping=="original_clipped")&(data.status=="passed")]
    for metric,symbol in [("E_D","diffusion"),("E_R","reaction")]:
        fig,axes=plt.subplots(2,3,figsize=(13.2,7.2),sharex=True,layout="constrained")
        for i,case in enumerate(("case_a","case_exp")):
            for j,(name,title) in enumerate(zip(names,titles)):
                ax=axes[i,j]
                for arm,label,color in zip(arms,labels,colors):
                    d=subset[(subset.case==case)&(subset.variant==name)&(subset.oracle_arm==arm)]
                    g=d.groupby("noise_level")[metric].agg(["mean","std","count"])
                    if (g["count"]!=5).any():raise AssertionError("Expected five paired seeds")
                    x=100*g.index.to_numpy();y=g["mean"].to_numpy();s=g["std"].to_numpy()
                    ax.plot(x,y,"o-",label=label,color=color,linewidth=1.6,markersize=3.5)
                    ax.fill_between(x,np.maximum(y-s,1e-12),y+s,color=color,alpha=.10)
                ax.set_yscale("log");ax.grid(alpha=.18)
                ax.set_title(("A: " if i==0 else "Exp: ")+title)
                if j==0:ax.set_ylabel("Relative "+symbol+" error")
                if i==1:ax.set_xlabel("Observation noise (% of clean-field SD)")
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
        save_figure(fig,out/("oracle_"+symbol+"_noise"))
    fig,axes=plt.subplots(2,2,figsize=(10.4,7.1),sharex=True,layout="constrained")
    for i,case in enumerate(("case_a","case_exp")):
        for j,metric in enumerate(("E_D","E_R")):
            ax=axes[i,j]
            for name,title,color in zip(names,titles,colors[1:]):
                for clip,style in [("original_clipped","-"),("unclipped_diagnostic","--")]:
                    d=data[(data["mode"]=="oracle")&(data.constraint=="unconstrained")&
                           (data.case==case)&(data.variant==name)&(data.clipping==clip)&
                           (data.oracle_arm=="observed_features_observed_targets")&(data.status=="passed")]
                    g=d.groupby("noise_level")[metric].mean()
                    ax.plot(100*g.index,g,marker="o",linestyle=style,color=color,markersize=3,
                            label=title+(" / clipped" if clip=="original_clipped" else " / unclipped"))
            ax.set_yscale("log");ax.grid(alpha=.18)
            ax.set_title(("A" if i==0 else "Exp")+": "+("diffusion" if j==0 else "reaction"))
            ax.set_ylabel("Relative closure error")
            if i==1:ax.set_xlabel("Observation noise (%)")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    save_figure(fig,out/"paired_clipping_effect")


def plot_weights(data,out):
    shared=data[(data["mode"]=="shared")&(data.status=="passed")]
    for term in ("strong","anchor"):
        selected=shared[shared.weak_variant=="space_weak"]
        for k,value in DEFAULT_WEIGHTS.items():
            if k!=term:selected=selected[np.isclose(selected["weight_"+k],value)]
        fig,axes=plt.subplots(2,3,figsize=(13.0,7.2),layout="constrained")
        for i,case in enumerate(("case_a","case_exp")):
            for j,(metric,title) in enumerate([( "E_D","Diffusion error"),("E_R","Reaction error"),
                                             ("common_default_objective","Default-objective score")]):
                ax=axes[i,j]
                for noise,color in [(0.,"#0072B2"),(.03,"#D55E00")]:
                    for constraint,style in [("unconstrained","-"),("positive_diffusion","--")]:
                        d=selected[(selected.case==case)&(selected.noise_level==noise)&(selected.constraint==constraint)]
                        g=d.groupby("weight_"+term)[metric].agg(["mean","std","count"])
                        if (g["count"]!=5).any():raise AssertionError("Unexpected weight duplicate/seed count")
                        label=("Clean" if noise==0 else "3% noise")+(" / positive D" if constraint=="positive_diffusion" else " / unconstrained")
                        ax.plot(g.index,g["mean"],marker="o",linestyle=style,color=color,label=label,markersize=3.5)
                ax.set_xscale("symlog",linthresh=.001 if term=="strong" else .03)
                ax.set_yscale("log");ax.grid(alpha=.18)
                ax.set_title(("A: " if i==0 else "Exp: ")+title)
                if i==1:ax.set_xlabel(term.capitalize()+" loss weight")
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
        save_figure(fig,out/(term+"_weight_global_optima"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();root=args.output_root.resolve();out=root/"reports";out.mkdir(exist_ok=True)
    data,hashes=collect(root)
    data.to_csv(out/"per_seed.csv",index=False)
    group=["mode","case","noise_level","clipping","oracle_arm","constraint","variant","weak_variant"]
    metrics=["E_D","E_R","objective","common_default_objective","minimum_diffusion","R0","R001"]
    passed=data[data.status=="passed"]
    summary=passed.groupby(group)[metrics].agg(["mean","std","count","min","max"])
    summary.columns=["_".join(v) for v in summary.columns]
    summary=summary.reset_index()
    attempted=data.groupby(group).size().reset_index(name="attempted")
    summary=attempted.merge(summary,on=group,how="left")
    summary.to_csv(out/"summary.csv",index=False)
    oracles=data[data["mode"]=="oracle"]
    contrasts=[]
    cols=["case","data_seed","noise_level","clipping","constraint","variant"]
    for key,frame in oracles.groupby(cols):
        frame=frame.set_index("oracle_arm")
        if len(frame)!=4:raise AssertionError("Four oracle arms required")
        row=dict(zip(cols,key));row["all_solvers_passed"]=bool(frame.status.eq("passed").all())
        for metric in ("E_D","E_R"):
            cc=frame.loc["clean_features_clean_targets",metric]
            co=frame.loc["clean_features_observed_targets",metric]
            oc=frame.loc["observed_features_clean_targets",metric]
            oo=frame.loc["observed_features_observed_targets",metric]
            row[metric+"_target_contamination"]=co-cc
            row[metric+"_feature_contamination"]=oc-cc
            row[metric+"_combined_contamination"]=oo-cc
            row[metric+"_interaction"]=oo-co-oc+cc
        contrasts.append(row)
    pd.DataFrame(contrasts).to_csv(out/"paired_oracle_error_contrasts.csv",index=False)
    clip_keys=["case","data_seed","noise_level","constraint","variant","oracle_arm"]
    clipped=oracles[oracles.clipping=="original_clipped"].set_index(clip_keys)
    unclipped=oracles[oracles.clipping=="unclipped_diagnostic"].set_index(clip_keys)
    effect=(clipped[["E_D","E_R"]]-unclipped[["E_D","E_R"]]).rename(columns=lambda x:x+"_clipped_minus_unclipped")
    effect.to_csv(out/"paired_clipping_error_contrasts.csv")
    plt.rcParams.update({"font.size":10,"axes.titlesize":11,"axes.spines.top":False,"axes.spines.right":False})
    plot_oracles(data,out);plot_weights(data,out)
    selected=summary[(summary.noise_level==.03)&(summary.clipping=="original_clipped")&
                     (summary.constraint=="unconstrained")&summary.variant.isin(["default","drop_strong","space_time_weak"])]
    selected.to_csv(out/"key_noise3_mechanisms.csv",index=False)
    checks=dict(total_fits=len(data),status_counts=data.status.value_counts().to_dict(),
        mode_counts=data["mode"].value_counts().to_dict(),
        shared_weight_objectives=int(data[data["mode"]=="shared"].variant.nunique()),
        oracle_weight_objectives=sorted(oracles.variant.unique().tolist()),
        maximum_ls_stationarity_relative=float(data.ls_stationarity_relative.max()),
        maximum_constrained_kkt_stationarity_relative=float(data.kkt_stationarity_relative.max()),
        maximum_constrained_kkt_complementarity_relative=float(data.kkt_complementarity_relative.max()),
        maximum_constrained_primal_dual_gap_relative=float(data.primal_dual_gap_relative.max()),
        minimum_constrained_diffusion=float(data[data.constraint=="positive_diffusion"].minimum_diffusion.min()),
        unconstrained_negative_diffusion_count=int(((data.constraint=="unconstrained")&(data.minimum_diffusion<=0)).sum()),
        constrained_active_optimization_count=int(data.constraint_used_optimizer.sum()),
        report_source_sha256=file_hash(__file__),source_bundle_hashes=hashes,
        interpretation=["Global optima refer to fixed seven-coefficient families, not the neural hypothesis class.",
            "Main weights compare original observed data; oracle arms use privileged clean synthetic fields explicitly.",
            "Oracle contrasts of norm errors are descriptive paired effects, not a variance decomposition or population bias estimator.",
            "Weight plots show a common default-objective score; own weighted minima for each changed objective are separately in CSV.",
            "Shaded bands are sample SD over five data seeds, not independent repeated weight samples.",
            "Clipping, oracle and zero-noise repetitions share data and do not increase sample size."])
    write_json(out/"REPORT_CHECKS.json",checks)
    print(json.dumps({k:v for k,v in checks.items() if k not in ("source_bundle_hashes","interpretation")},indent=2))


if __name__=="__main__":main()
