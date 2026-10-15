"""Generate the scientific figure and two evidence tables from completed records."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import response_bounds as rb


def main():
    output = rb.OUT / "assets"; output.mkdir(exist_ok=True)
    response = json.loads((rb.OUT / "formal_dt100/SUMMARY.json").read_text())
    observations = [json.loads((rb.OUT / f"observation_ellipsoid/seed{s}.json").read_text()) for s in range(5)]
    metrics = [json.loads((rb.OUT / f"metric_verified/seed{s}.json").read_text()) for s in range(5)]
    lines = [r"\begin{table}[t]", r"\centering\small",
        r"\caption{Case~A frozen-model response audit: raw sign agreement and signs determined by computable intervals. Each column comprises 60 comparisons repeating five seed units. The reference uncertainty is an absolute spatial RMS radius. No determined sign is incorrect.}",
        r"\label{tab:theory_response}", r"\begin{tabular}{lrrrr}", r"\toprule",
        r"State predictor & Raw agreement & $\varepsilon_y=0$ & $10^{-5}$ & $10^{-4}$\\", r"\midrule"]
    for key, label in (("first", "First"), ("second", "Second")):
        d = response["by_predictor"][key]
        lines.append(f"{label} & {d['sign_agreements']}/60 & {d['certified']}/60 & {d['at_reference_1e_5']}/60 & {d['at_reference_1e_4']}/60" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (output / "theory_response_checks.tex").write_text("\n".join(lines))
    lines = [r"\begin{table}[t]", r"\centering\small",
        r"\caption{Case~A PDE response metrics and constructed projection controls at $T=0.6$. The commutator is normalized by $\|M\|_2$. Each largest positive mode is tested by direct nonlinear solves at a small amplitude and half that amplitude. Both branch fitting errors decrease and the forward squared error increases in every control. Relative gain error compares the direct gain with $\epsilon^2\lambda_{\max}(PMP-M)$.}",
        r"\label{tab:theory_metric}", r"\begin{tabular}{rrrrr}", r"\toprule",
        r"Seed & $\|PM-MP\|_2/\|M\|_2$ & $\lambda_{\max}(PMP-M)$ & Gain error & Half-amplitude\\", r"\midrule"]
    for d in metrics:
        a = d["matrix_diagnostics"]["0.6"]; r0, r1 = d["constructed_witness_checks"]
        lines.append(f"{d['seed']} & {a['relative_commutator']:.3f} & {a['maximum_gain']:.3f} & {100*r0['relative_linearization_error']:.2f}\\% & {100*r1['relative_linearization_error']:.2f}\\%" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (output / "theory_metric_checks.tex").write_text("\n".join(lines))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.titlesize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.3), layout="constrained")
    colors = ("#245b83", "#be6a2e", "#286d58")
    ax = axes[0]
    ax.errorbar(range(5), [d["center_gain"] * 1e5 for d in observations],
                yerr=[d["interval_halfwidth"] * 1e5 for d in observations],
                fmt="o", color=colors[0], capsize=5, label="Consistency interval", zorder=3)
    ax.scatter(np.arange(5) + .12, [d["actual_constitutive_gain"] * 1e5 for d in observations],
               marker="x", color=colors[2], s=35, label="Actual truth gain", zorder=4)
    ax.axhline(0, color="0.5", linestyle="--", lw=.8)
    ax.set(xticks=range(5), xlabel="Data seed", ylabel=r"Closure error change ($10^{-5}$)",
           title="A  Observation-based\nranking interval")
    ax.legend(loc="lower right", fontsize=12, frameon=False)
    ax = axes[1]
    ax.bar(range(5), [d["matrix_diagnostics"]["0.6"]["maximum_gain"] for d in metrics],
           color=colors[0], width=.62)
    ax.axhline(0, color="0.5", lw=.8)
    ax.set(xticks=range(5), xlabel="Data seed", ylabel=r"$\lambda_{\max}(PMP-M)$",
           title="B  Complete PDE\nresponse metric", ylim=(0, .45))
    ax.text(.97, .97, "9 functions\n4 unseen fields",
            transform=ax.transAxes, ha="right", va="top", fontsize=12)
    ax = axes[2]
    critical = response["critical"]
    ax.errorbar([0, 1], [r["predicted_change"] * 1e7 for r in critical],
                yerr=[r["bound"] * 1e7 for r in critical], fmt="o", color=colors[0], capsize=6)
    ax.axhline(0, color="0.5", linestyle="--", lw=.8)
    ax.axhline(critical[0]["actual_change"] * 1e7, color=colors[2], lw=1.3, label="Direct nonlinear solve")
    ax.set(xticks=[0, 1], xticklabels=["First state", "Second state"],
           ylabel=r"Squared-error change ($10^{-7}$)",
           title="C  Cancellation\nboundary", xlim=(-.5, 1.5))
    ax.text(0, -9.2, "Undetermined", ha="center", fontsize=12, color=colors[1])
    ax.text(1, -9.2, "Deterioration", ha="center", fontsize=12, color=colors[2])
    ax.set_ylim(-10.5, 6)
    ax.legend(loc="upper right", fontsize=11, frameon=False)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output / f"theory_pipeline_diagnostics.{ext}", dpi=200)
    plt.close(fig)
    rb.save(output / "ASSETS.json", dict(status="complete", figure_panels=3,
        driver_sha256=rb.sha(__file__), files={p.name: rb.sha(p) for p in output.iterdir() if p.is_file() and p.name != "ASSETS.json"}))


if __name__ == "__main__":
    main()
