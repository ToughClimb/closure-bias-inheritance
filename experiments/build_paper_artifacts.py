from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SYMBOLIC_METRICS = [
    "neural_true_ErrD",
    "neural_true_ErrR",
    "symbolic_surrogate_ErrD",
    "symbolic_surrogate_ErrR",
    "symbolic_true_ErrD",
    "symbolic_true_ErrR",
    "neural_unseen",
    "symbolic_unseen",
    "bir_D",
    "bir_R",
]

BASELINE_SETTINGS_MAIN = ["clean", "noise_5", "sparse_both"]
SETTING_ORDER = ["clean", "noise_1", "noise_5", "sparse_space", "sparse_time", "sparse_both"]
SETTING_LABELS = {
    "clean": "clean",
    "noise_1": "noise 1%",
    "noise_5": "noise 5%",
    "sparse_space": "sparse x",
    "sparse_time": "sparse t",
    "sparse_both": "sparse x+t",
}
CASE_ORDER = ["case_a", "case_b", "case_exp"]
CASE_LABELS = {"case_a": "Case A", "case_b": "Case B", "case_exp": "Case Exp"}
METHOD_ORDER = ["strong_poly", "weak_poly", "mlp", "neural+symbolic"]
METHOD_LABELS = {
    "strong_poly": "strong_poly",
    "weak_poly": "weak_poly",
    "mlp": "neural",
    "neural+symbolic": "neural+symbolic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper tables and figures from benchmark CSV files.")
    parser.add_argument(
        "--symbolic-csv",
        action="append",
        required=True,
        help="Path to a symbolic compression benchmark CSV. Can be passed multiple times.",
    )
    parser.add_argument(
        "--baseline-csv",
        action="append",
        default=[],
        help="Path to a polynomial baseline benchmark CSV. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_artifacts"))
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def aggregate_rows(rows: list[dict[str, str]], group_keys: tuple[str, ...], value_keys: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    aggregated: list[dict[str, object]] = []
    for group in sorted(grouped):
        group_rows = grouped[group]
        record: dict[str, object] = {key: value for key, value in zip(group_keys, group)}
        for value_key in value_keys:
            values = np.asarray([to_float(row, value_key) for row in group_rows], dtype=np.float64)
            record[f"{value_key}_mean"] = float(np.mean(values))
            record[f"{value_key}_std"] = float(np.std(values))
        aggregated.append(record)
    return aggregated


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.3e} +/- {std_value:.3e}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def setting_sort_key(setting: str) -> int:
    try:
        return SETTING_ORDER.index(setting)
    except ValueError:
        return len(SETTING_ORDER)


def case_sort_key(case_name: str) -> int:
    try:
        return CASE_ORDER.index(case_name)
    except ValueError:
        return len(CASE_ORDER)


def method_sort_key(method_name: str) -> int:
    try:
        return METHOD_ORDER.index(method_name)
    except ValueError:
        return len(METHOD_ORDER)


def build_symbolic_summary_markdown(aggregated_rows: list[dict[str, object]]) -> str:
    lines = [
        "# Symbolic Compression Summary",
        "",
        "Each cell reports mean +/- std across seeds.",
        "",
    ]
    for case_name in sorted({str(row["case"]) for row in aggregated_rows}, key=case_sort_key):
        lines.extend(
            [
                f"## {CASE_LABELS.get(case_name, case_name)}",
                "",
                "| Setting | Neural ErrD | Neural ErrR | Symbolic surrogate ErrD | Symbolic surrogate ErrR | Symbolic true ErrD | Symbolic true ErrR | Neural unseen | Symbolic unseen | BIR_D | BIR_R |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        case_rows = [row for row in aggregated_rows if str(row["case"]) == case_name]
        for row in sorted(case_rows, key=lambda item: setting_sort_key(str(item["setting"]))):
            lines.append(
                "| "
                + " | ".join(
                    [
                        SETTING_LABELS.get(str(row["setting"]), str(row["setting"])),
                        format_mean_std(float(row["neural_true_ErrD_mean"]), float(row["neural_true_ErrD_std"])),
                        format_mean_std(float(row["neural_true_ErrR_mean"]), float(row["neural_true_ErrR_std"])),
                        format_mean_std(float(row["symbolic_surrogate_ErrD_mean"]), float(row["symbolic_surrogate_ErrD_std"])),
                        format_mean_std(float(row["symbolic_surrogate_ErrR_mean"]), float(row["symbolic_surrogate_ErrR_std"])),
                        format_mean_std(float(row["symbolic_true_ErrD_mean"]), float(row["symbolic_true_ErrD_std"])),
                        format_mean_std(float(row["symbolic_true_ErrR_mean"]), float(row["symbolic_true_ErrR_std"])),
                        format_mean_std(float(row["neural_unseen_mean"]), float(row["neural_unseen_std"])),
                        format_mean_std(float(row["symbolic_unseen_mean"]), float(row["symbolic_unseen_std"])),
                        format_mean_std(float(row["bir_D_mean"]), float(row["bir_D_std"])),
                        format_mean_std(float(row["bir_R_mean"]), float(row["bir_R_std"])),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def build_baseline_summary_markdown(aggregated_rows: list[dict[str, object]]) -> str:
    lines = [
        "# Baseline Summary",
        "",
        "Representative settings for method comparison. Each cell reports mean +/- std across seeds.",
        "",
        "| Case | Setting | Method | ErrD | ErrR | Unseen |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    selected_rows = [
        row
        for row in aggregated_rows
        if str(row["setting"]) in BASELINE_SETTINGS_MAIN and str(row["case"]) in CASE_ORDER
    ]
    for row in sorted(
        selected_rows,
        key=lambda item: (
            case_sort_key(str(item["case"])),
            setting_sort_key(str(item["setting"])),
            method_sort_key(str(item["method"])),
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    CASE_LABELS.get(str(row["case"]), str(row["case"])),
                    SETTING_LABELS.get(str(row["setting"]), str(row["setting"])),
                    METHOD_LABELS.get(str(row["method"]), str(row["method"])),
                    format_mean_std(float(row["ErrD_mean"]), float(row["ErrD_std"])),
                    format_mean_std(float(row["ErrR_mean"]), float(row["ErrR_std"])),
                    format_mean_std(float(row["unseen_mean"]), float(row["unseen_std"])),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def plot_error_propagation(aggregated_rows: list[dict[str, object]], output_path: Path) -> None:
    n_cases = len(CASE_ORDER)
    fig, axes = plt.subplots(2, n_cases, figsize=(4.2 * n_cases, 7), sharex=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, n_cases)
    stage_labels = ["Stage 1 neural", "Stage 2 symbolic fit", "Stage 3 symbolic true"]
    metrics_by_component = {
        "D": ["neural_true_ErrD_mean", "symbolic_surrogate_ErrD_mean", "symbolic_true_ErrD_mean"],
        "R": ["neural_true_ErrR_mean", "symbolic_surrogate_ErrR_mean", "symbolic_true_ErrR_mean"],
    }
    errors_by_component = {
        "D": ["neural_true_ErrD_std", "symbolic_surrogate_ErrD_std", "symbolic_true_ErrD_std"],
        "R": ["neural_true_ErrR_std", "symbolic_surrogate_ErrR_std", "symbolic_true_ErrR_std"],
    }
    legend_handles: list[object] = []

    for col, case_name in enumerate(CASE_ORDER):
        case_rows = [row for row in aggregated_rows if str(row["case"]) == case_name]
        case_rows = sorted(case_rows, key=lambda item: setting_sort_key(str(item["setting"])))
        x = np.arange(len(case_rows))
        width = 0.25
        for row_index, component in enumerate(["D", "R"]):
            ax = axes[row_index, col]
            for offset, (metric_key, error_key, stage_label) in enumerate(
                zip(metrics_by_component[component], errors_by_component[component], stage_labels, strict=True)
            ):
                values = [float(row[metric_key]) for row in case_rows]
                errors = [float(row[error_key]) for row in case_rows]
                bars = ax.bar(x + (offset - 1) * width, values, width=width, yerr=errors, capsize=3, label=stage_label)
                if row_index == 0 and col == 0:
                    legend_handles.append(bars[0])
            ax.set_yscale("log")
            ax.set_title(f"{CASE_LABELS.get(case_name, case_name)} {'Diffusion' if component == 'D' else 'Reaction'}")
            ax.set_xticks(x)
            ax.set_xticklabels([SETTING_LABELS.get(str(row["setting"]), str(row["setting"])) for row in case_rows], rotation=20)
            ax.set_ylabel("relative error")
            ax.grid(axis="y", linestyle=":", alpha=0.4)
    if legend_handles:
        fig.legend(
            legend_handles,
            stage_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_and_bir(aggregated_rows: list[dict[str, object]], output_path: Path) -> None:
    n_cases = len(CASE_ORDER)
    fig, axes = plt.subplots(2, n_cases, figsize=(4.2 * n_cases, 7), sharex=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, n_cases)
    rollout_handles = None
    bir_handles = None

    for col, case_name in enumerate(CASE_ORDER):
        case_rows = [row for row in aggregated_rows if str(row["case"]) == case_name]
        case_rows = sorted(case_rows, key=lambda item: setting_sort_key(str(item["setting"])))
        x = np.arange(len(case_rows))
        labels = [SETTING_LABELS.get(str(row["setting"]), str(row["setting"])) for row in case_rows]

        rollout_ax = axes[0, col]
        neural_line = rollout_ax.errorbar(
            x,
            [float(row["neural_unseen_mean"]) for row in case_rows],
            yerr=[float(row["neural_unseen_std"]) for row in case_rows],
            marker="o",
            linewidth=2,
            label="neural",
        )
        symbolic_line = rollout_ax.errorbar(
            x,
            [float(row["symbolic_unseen_mean"]) for row in case_rows],
            yerr=[float(row["symbolic_unseen_std"]) for row in case_rows],
            marker="s",
            linewidth=2,
            label="symbolic",
        )
        if rollout_handles is None:
            rollout_handles = [neural_line.lines[0], symbolic_line.lines[0]]
        rollout_ax.set_yscale("log")
        rollout_ax.set_title(f"{CASE_LABELS.get(case_name, case_name)} rollout preservation")
        rollout_ax.set_xticks(x)
        rollout_ax.set_xticklabels(labels, rotation=20)
        rollout_ax.set_ylabel("unseen rollout relative L2")
        rollout_ax.grid(axis="y", linestyle=":", alpha=0.4)

        bir_ax = axes[1, col]
        width = 0.35
        bir_d = bir_ax.bar(
            x - 0.5 * width,
            [float(row["bir_D_mean"]) for row in case_rows],
            width=width,
            yerr=[float(row["bir_D_std"]) for row in case_rows],
            capsize=3,
            label="BIR_D",
        )
        bir_r = bir_ax.bar(
            x + 0.5 * width,
            [float(row["bir_R_mean"]) for row in case_rows],
            width=width,
            yerr=[float(row["bir_R_std"]) for row in case_rows],
            capsize=3,
            label="BIR_R",
        )
        if bir_handles is None:
            bir_handles = [bir_d[0], bir_r[0]]
        bir_ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        bir_ax.set_title(f"{CASE_LABELS.get(case_name, case_name)} bias inheritance")
        bir_ax.set_xticks(x)
        bir_ax.set_xticklabels(labels, rotation=20)
        bir_ax.set_ylabel("bias inheritance ratio")
        bir_ax.grid(axis="y", linestyle=":", alpha=0.4)

    if rollout_handles is not None and bir_handles is not None:
        fig.legend(
            rollout_handles + bir_handles,
            ["neural rollout", "symbolic rollout", r"$\mathrm{BIR}_D$", r"$\mathrm{BIR}_R$"],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=4,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_case_exp_observation_breakdown(aggregated_rows: list[dict[str, object]], output_path: Path) -> None:
    case_rows = {
        str(row["setting"]): row
        for row in aggregated_rows
        if str(row["case"]) == "case_exp"
    }
    panel_settings = [
        ("Noise progression", ["clean", "noise_1", "noise_5"]),
        ("Sparse observation breakdown", ["clean", "sparse_time", "sparse_space", "sparse_both"]),
    ]
    metric_specs = [
        ("neural_true_ErrD", "neural ErrD", "o"),
        ("neural_true_ErrR", "neural ErrR", "s"),
        ("symbolic_unseen", "symbolic unseen", "^"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes = np.atleast_1d(axes)
    legend_handles: list[object] = []

    for axis, (title, settings) in zip(axes, panel_settings, strict=True):
        x = np.arange(len(settings))
        for metric_key, label, marker in metric_specs:
            means = [float(case_rows[setting][f"{metric_key}_mean"]) for setting in settings]
            stds = [float(case_rows[setting][f"{metric_key}_std"]) for setting in settings]
            line = axis.errorbar(
                x,
                means,
                yerr=stds,
                marker=marker,
                linewidth=2,
                capsize=3,
                label=label,
            )
            if len(legend_handles) < len(metric_specs):
                legend_handles.append(line.lines[0])
        axis.set_title(title)
        axis.set_yscale("log")
        axis.set_xticks(x)
        axis.set_xticklabels([SETTING_LABELS[setting] for setting in settings], rotation=20)
        axis.set_ylabel("relative metric value")
        axis.grid(axis="y", linestyle=":", alpha=0.4)

    if legend_handles:
        fig.legend(
            legend_handles,
            [label for _, label, _ in metric_specs],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=3,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    symbolic_rows: list[dict[str, str]] = []
    for csv_path in args.symbolic_csv:
        symbolic_rows.extend(load_csv(Path(csv_path)))

    symbolic_aggregated = aggregate_rows(symbolic_rows, ("case", "setting"), SYMBOLIC_METRICS)
    symbolic_fieldnames = ["case", "setting"] + [f"{metric}_{suffix}" for metric in SYMBOLIC_METRICS for suffix in ("mean", "std")]
    write_csv(tables_dir / "symbolic_summary.csv", symbolic_aggregated, symbolic_fieldnames)
    write_text(tables_dir / "symbolic_summary.md", build_symbolic_summary_markdown(symbolic_aggregated))

    plot_error_propagation(symbolic_aggregated, figures_dir / "symbolic_error_propagation.png")
    plot_rollout_and_bir(symbolic_aggregated, figures_dir / "symbolic_rollout_and_bir.png")
    plot_case_exp_observation_breakdown(symbolic_aggregated, figures_dir / "case_exp_observation_breakdown.png")

    if args.baseline_csv:
        baseline_rows: list[dict[str, str]] = []
        for csv_path in args.baseline_csv:
            baseline_rows.extend(load_csv(Path(csv_path)))

        symbolic_method_rows = []
        for row in symbolic_rows:
            symbolic_method_rows.append(
                {
                    "case": row["case"],
                    "setting": row["setting"],
                    "seed": row["seed"],
                    "method": "neural+symbolic",
                    "ErrD": row["symbolic_true_ErrD"],
                    "ErrR": row["symbolic_true_ErrR"],
                    "unseen": row["symbolic_unseen"],
                }
            )

        combined_baseline_rows = baseline_rows + symbolic_method_rows
        baseline_aggregated = aggregate_rows(combined_baseline_rows, ("case", "setting", "method"), ["ErrD", "ErrR", "unseen"])
        baseline_fieldnames = ["case", "setting", "method", "ErrD_mean", "ErrD_std", "ErrR_mean", "ErrR_std", "unseen_mean", "unseen_std"]
        write_csv(tables_dir / "baseline_summary.csv", baseline_aggregated, baseline_fieldnames)
        write_text(tables_dir / "baseline_summary.md", build_baseline_summary_markdown(baseline_aggregated))

    print(f"paper_artifacts_saved={output_dir}")


if __name__ == "__main__":
    main()
