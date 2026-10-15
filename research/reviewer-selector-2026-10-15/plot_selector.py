"""Vector mechanism figures from the independently reproduced selector records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_selector import REPO, sha, write_csv, write_json

SEED_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
METRIC_COLORS = {"eN": "#525252", "eC": "#D55E00", "eS": "#0072B2"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.4, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 250,
                     "font.family": "DejaVu Sans", "axes.unicode_minus": True})


def read_csv(path):
    return list(csv.DictReader(Path(path).open()))


def choose(record, branch, relative, absolute):
    candidates = record["branches"][branch]["candidates"]
    best = min(c["fit_mse"] for c in candidates)
    eligible = [c for c in candidates if c["fit_mse"] <= (1 + relative) * best + absolute]
    return min(eligible, key=lambda c: (c["complexity"], c["fit_mse"]))


def select_records(records, cohort, case, setting):
    selected = [r for r in records if all(r["config"][k] == value for k, value in
                                         [("cohort", cohort), ("case", case), ("setting", setting)])]
    selected.sort(key=lambda r: r["config"]["seed"])
    assert [r["config"]["seed"] for r in selected] == list(range(5))
    return selected


def coordinates(events, ids, axis, fixed, base, lo, hi):
    values = list(base)
    for event in events:
        if (event["id"] in ids and event["branch"] == "D"
                and event["boundary_axis"] == axis and float(event["fixed_other_value"]) == fixed):
            values += [float(event[k]) for k in
                       ["analytic_boundary", "separated_before_value", "separated_after_value"]]
    return np.array(sorted({x for x in values if lo <= x <= hi}))


def matrix(records, x, axis, fixed, metric, figure, collected):
    values = []
    for record in records:
        line = []
        for value in x:
            relative, absolute = (fixed, float(value)) if axis == "absolute" else (float(value), fixed)
            candidate = choose(record, "D", relative, absolute)
            data = {**candidate["metrics"], "complexity": candidate["complexity"]}
            line.append(data[metric])
            if metric == "BIR":
                collected.append(dict(figure=figure, **{k: record["config"][k] for k in
                                                        ["id", "cohort", "case", "setting", "seed", "epoch"]},
                                      branch="D", relative=relative, absolute=absolute,
                                      family=candidate["family"], complexity=candidate["complexity"],
                                      candidate_index=candidate["index"], **candidate["metrics"]))
        values.append(line)
    result = np.asarray(values, dtype=float)
    assert result.shape == (5, len(x)) and np.isfinite(result).all()
    return result


def draw_mean(ax, x, values, color, label=None, log=False, trace=True, ls="-"):
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    if trace:
        for line in values:
            ax.step(x, line, where="post", color=color, lw=.6, alpha=.22, zorder=1)
    ax.step(x, mean, where="post", color=color, lw=1.6, label=label, ls=ls, zorder=3)
    lower = mean - sd
    if log:
        ax.fill_between(x, lower, mean + sd, where=lower > 0, step="post", color=color, alpha=.10, lw=0)
    else:
        ax.fill_between(x, lower, mean + sd, step="post", color=color, alpha=.10, lw=0)
    return mean


def save_figure(fig, destination, name, caption):
    output = []
    for extension in ["pdf", "png"]:
        path = destination / f"{name}.{extension}"
        fig.savefig(path, bbox_inches="tight", metadata={"Title": caption} if extension == "pdf" else None)
        output.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
    plt.close(fig)
    return dict(name=name, caption=caption, files=output)


def mark_default(ax, value, text=False):
    ax.axvline(value, color="#222222", lw=.9, ls=(0, (4, 3)), alpha=.85, zorder=0)
    if text:
        ax.text(value, .02, " default", transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=6.5,
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=.3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/reviewer-selector-2026-10-15")
    args = parser.parse_args()
    output = args.output.resolve()
    result = json.loads((output / "RESULTS.json").read_text())
    assert result["status"] == "complete" and result["failed_teachers"] == 0
    records = [json.loads(path.read_text()) for path in sorted((output / "candidate_records").glob("*.json"))]
    assert len(records) == 85
    events = read_csv(output / "selection_transitions.csv")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    collected = []
    artifacts = []
    clean = select_records(records, "sameobs_6000", "case_exp", "clean")
    noisy = select_records(records, "sameobs_6000", "case_exp", "noise3")
    clean_ids = {r["config"]["id"] for r in clean}
    first_switches = [r for r in events if r["id"] in clean_ids and r["branch"] == "D"
                      and r["boundary_axis"] == "absolute" and float(r["fixed_other_value"]) == .05
                      and r["before_family"] == "rational" and r["after_family"] == "exp_decay"]
    first_switches.sort(key=lambda r: int(r["seed"]))
    assert len(first_switches) == 5

    # Predeclared 6000-epoch clean/noisy cohort; fixed teachers on both rows.
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.8), constrained_layout=True)
    for row, subset in enumerate([clean, noisy]):
        ids = {r["config"]["id"] for r in subset}
        x = coordinates(events, ids, "absolute", .05, np.geomspace(1e-12, 1e-4, 161), 1e-12, 1e-4)
        for metric in ["eN", "eC", "eS"]:
            values = matrix(subset, x, "absolute", .05, metric, "sameobs6000_absolute", collected)
            assert metric != "eN" or np.all(np.ptp(values, axis=1) == 0)
            draw_mean(axes[row, 0], x, values, METRIC_COLORS[metric],
                      {"eN": r"$e_N$ (fixed teacher)", "eC": r"$e_C$ (compression)", "eS": r"$e_S$ (symbolic truth error)"}[metric],
                      log=True, ls="--" if metric == "eN" else "-")
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_ylabel(r"Common-grid $L^2$ distance")
        if row == 0:
            axes[row, 0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=.95)
        bir = matrix(subset, x, "absolute", .05, "BIR", "sameobs6000_absolute", collected)
        draw_mean(axes[row, 1], x, bir, "#0072B2")
        axes[row, 1].axhline(1, color="#666666", lw=.7, ls=":")
        if row == 0:
            axes[row, 1].set_yscale("log")
            for event in first_switches:
                axes[row, 1].axvline(float(event["analytic_boundary"]), ymin=0, ymax=.14,
                                    color=SEED_COLORS[int(event["seed"])], lw=.8, alpha=.55)
        else:
            formatter = ScalarFormatter(useOffset=False)
            axes[row, 1].yaxis.set_major_formatter(formatter)
        axes[row, 1].set_ylabel(r"BIR $=e_S/e_N$")
        minimum = matrix(subset, x, "absolute", .05, "minimum_D_support", "sameobs6000_absolute", collected)
        draw_mean(axes[row, 2], x, minimum, "#009E73")
        axes[row, 2].axhline(0, color="#333333", lw=.7, ls=":")
        axes[row, 2].set_ylabel(r"$\min_{u\in\mathcal{U}} D_S(u)$")
        for col, title in enumerate(["Error components", "Error propagation", "Diffusion margin"]):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.set_xlim(1e-12, 1e-4)
            ax.set_xticks([1e-12, 1e-10, 1e-8, 1e-6, 1e-4])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel(r"Absolute MSE tolerance $\tau_a$")
            ax.set_title(f"({chr(97 + row * 3 + col)}) {'Clean' if row == 0 else '3% noise'}: {title}")
            mark_default(ax, 1e-8, text=(col == 0))
    fig.suptitle(r"Case Exp, 6000 epochs: frozen selection at $\tau_r=0.05$", fontsize=10)
    artifacts.append(save_figure(fig, figures, "selector_sameobs6000_absolute",
        "Exact-observation 6000-epoch Exp teachers, five per setting. Thin lines are individual teachers; thick lines and bands show mean and sample SD. Teacher errors are fixed. Nonpositive mean-minus-SD limits are omitted on logarithmic axes. Default tau_a=1e-8 is dashed. Diffusion minima use endpoints and critical points of the selected functions, not a grid-only positivity certificate. No PDE rollout is inferred."))

    # Detail at the five actual rational->exponential switches.
    x = coordinates(events, clean_ids, "absolute", .05, np.linspace(.7e-9, 3.5e-9, 200), .7e-9, 3.5e-9)
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.65), constrained_layout=True)
    for ax, metric, label in zip(axes, ["BIR", "rho", "cosine_alignment"],
                                 [r"BIR $=e_S/e_N$", r"$\rho=e_C/e_N$", "Error/perturbation cosine"]):
        values = matrix(clean, x, "absolute", .05, metric, "sameobs6000_transition", collected)
        mean = values.mean(0)
        sd = values.std(0, ddof=1)
        ax.fill_between(x / 1e-9, mean - sd, mean + sd, color="#555555", alpha=.10, step="post", lw=0)
        ax.step(x / 1e-9, mean, where="post", color="#222222", lw=1.5, label="Mean")
        for seed, line in enumerate(values):
            ax.step(x / 1e-9, line, where="post", color=SEED_COLORS[seed], lw=1., label=f"Seed {seed}")
            boundary = float(first_switches[seed]["analytic_boundary"])
            ax.axvline(boundary / 1e-9, color=SEED_COLORS[seed], lw=.7, ls=":", alpha=.7)
        ax.set(xlim=(.7, 3.5), xlabel=r"$\tau_a\,/\,10^{-9}$", ylabel=label)
        ax.set_title(f"({chr(97 + list(axes).index(ax))}) {label}")
    axes[0].legend(ncols=2, frameon=False, loc="lower left", fontsize=5.9)
    fig.suptitle(r"Clean Exp, 6000 epochs: five actual family switches ($\tau_r=0.05$)", fontsize=9.4)
    artifacts.append(save_figure(fig, figures, "selector_sameobs6000_transition",
        "Five fixed teachers, tau_r=.05. Coloured vertical lines mark actual rational-to-exponential selection boundaries, validated on both sides. Grey bands are sample SD and the black line is the five-teacher mean. The default absolute tolerance is 1e-8, outside this zoom. Candidate choice uses teacher fit MSE and complexity, never truth error."))

    # Relative-tolerance exception retained per seed rather than hidden in a mean.
    sparse = select_records(records, "historical_main_150", "case_a", "sparse")
    sparse_ids = {r["config"]["id"] for r in sparse}
    x = coordinates(events, sparse_ids, "relative", 1e-6, np.linspace(0, 1, 201), 0, 1)
    transitions = [r for r in events if r["id"] in sparse_ids and r["branch"] == "D"
                   and r["boundary_axis"] == "relative" and float(r["fixed_other_value"]) == 1e-6]
    assert {int(r["seed"]) for r in transitions} == {0, 1}
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.65), constrained_layout=True)
    for metric in ["eN", "eC", "eS"]:
        values = matrix(sparse, x, "relative", 1e-6, metric, "A_sparse_relative", collected)
        draw_mean(axes[0], x, values, METRIC_COLORS[metric],
                  {"eN": r"$e_N$ (fixed teacher)", "eC": r"$e_C$", "eS": r"$e_S$"}[metric],
                  ls="--" if metric == "eN" else "-")
    axes[0].set_ylabel(r"Common-grid $L^2$ distance")
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.9, loc="lower right")
    bir = matrix(sparse, x, "relative", 1e-6, "BIR", "A_sparse_relative", collected)
    draw_mean(axes[1], x, bir, "#222222", trace=False)
    complexity = matrix(sparse, x, "relative", 1e-6, "complexity", "A_sparse_relative", collected)
    for seed in range(5):
        for ax, values in [(axes[1], bir), (axes[2], complexity)]:
            ax.step(x, values[seed], where="post", color=SEED_COLORS[seed], lw=1.,
                    ls=["-", "--", "-.", ":", (0, (5, 2, 1, 2))][seed], label=f"Seed {seed}")
    axes[1].axhline(1, color="#888888", lw=.7, ls=":")
    axes[1].set_ylabel(r"BIR $=e_S/e_N$")
    axes[2].set_ylabel("Selected complexity")
    axes[2].set_yticks([2, 3, 4, 5])
    axes[2].set_ylim(2.7, 4.5)
    fig.legend(handles=[Line2D([0], [0], color=SEED_COLORS[seed], lw=1., label=f"Seed {seed}")
                        for seed in range(5)], ncols=5, frameon=False,
               loc="lower center", bbox_to_anchor=(.67, -.055), fontsize=6.5)
    for event in transitions:
        value = float(event["analytic_boundary"])
        seed = int(event["seed"])
        for ax in axes[1:]:
            ax.axvline(value, color=SEED_COLORS[seed], lw=.8, ls=":")
        axes[2].text(value + .012, 4.33 - .15 * seed, f"Seed {seed}: {value:.3f}", color=SEED_COLORS[seed], fontsize=6.3)
    for i, ax in enumerate(axes):
        mark_default(ax, .05)
        ax.set(xlim=(0, 1), xlabel=r"Relative MSE tolerance $\tau_r$")
        ax.set_title(["(a) Error components", "(b) Individual error ratios", "(c) Individual complexity"][i])
    fig.suptitle(r"Historical Case A, sparse observations, 150 epochs: fixed $\tau_a=10^{-6}$", fontsize=9.4)
    artifacts.append(save_figure(fig, figures, "selector_A_sparse_relative",
        "Historical 150-epoch A sparse cohort, five fixed teachers. Seed 0 changes complexity at tau_r=.1984386614 and seed 1 at .3205800381; seeds 2-4 do not switch over [0,1] at tau_a=1e-6. Per-teacher curves are retained. The teacher error remains fixed. Default relative tolerance .05 is dashed. Bands denote sample SD."))

    write_csv(output / "figure_sweeps.csv", collected)
    write_csv(output / "figure_actual_switches.csv", first_switches + transitions)
    write_json(output / "FIGURES.json", dict(status="complete", script=str(Path(__file__).resolve()),
                script_sha256=sha(Path(__file__)), source_results_sha256=sha(output / "RESULTS.json"),
                input_records=85, figure_count=len(artifacts),
                data_files=[dict(path=str(output / name), sha256=sha(output / name))
                            for name in ["figure_sweeps.csv", "figure_actual_switches.csv"]],
                figures=artifacts, scope="75 historical teachers and ten 6000-epoch supplement teachers remain separate; no new neural-weight sweep teachers pooled."))
    print(json.dumps(dict(figures=len(artifacts), per_teacher_plot_rows=len(collected), output=str(figures))))


if __name__ == "__main__":
    main()
