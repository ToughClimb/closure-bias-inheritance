from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D, generate_dataset
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig,
    make_paper_training_config,
    run_closure_identification,
    summarize_training_objective,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep observation noise levels for Stage-1 closure recovery.")
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_exp")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--t-final", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--backbone", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--kan-grid-size", type=int, default=16)
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="0,0.01,0.02,0.03,0.04,0.05",
        help="Comma-separated list of noise levels, interpreted as multiples of std(u).",
    )
    parser.add_argument("--amplitude-min", type=float, default=None)
    parser.add_argument("--amplitude-max", type=float, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to results/noise_sweep_<case>.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional Markdown summary path. Defaults to results/noise_sweep_<case>.md",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=None,
        help="Optional figure path. Defaults to results/paper_artifacts/figures/<case>_noise_sweep.png",
    )
    return parser.parse_args()


def parse_noise_levels(text: str) -> list[float]:
    levels = []
    for piece in text.split(","):
        value = float(piece.strip())
        if value < 0.0:
            raise ValueError("noise levels must be non-negative")
        levels.append(value)
    return sorted(dict.fromkeys(levels))


def nominal_snr_db(noise_level: float) -> float:
    if noise_level <= 0.0:
        return float("inf")
    return float(-20.0 * math.log10(noise_level))


def realized_snr_db(clean_u: np.ndarray, observed_u: np.ndarray) -> float:
    signal_std = float(np.std(clean_u))
    noise_std = float(np.std(observed_u - clean_u))
    if noise_std <= 1.0e-12:
        return float("inf")
    return float(20.0 * math.log10(signal_std / noise_std))


def fmt(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.3e} +/- {std_value:.3e}"


def summarize(values: list[float]) -> tuple[float, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("inf"), 0.0
    return float(mean(finite_values)), float(pstdev(finite_values) if len(finite_values) > 1 else 0.0)


def build_markdown(rows: list[dict[str, float | int | str]]) -> str:
    grouped: dict[float, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["noise_level"])].append(row)

    lines = [
        "# Noise Sweep Summary",
        "",
        "Each row reports mean +/- std across seeds.",
        "",
        "| Noise | Nominal SNR (dB) | Realized SNR (dB) | ErrD | ErrR | Unseen | Weak loss | Strong loss |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for noise_level in sorted(grouped):
        group = grouped[noise_level]
        nominal = nominal_snr_db(noise_level)
        realized_mean, realized_std = summarize([float(row["realized_snr_db"]) for row in group])
        errd_mean, errd_std = summarize([float(row["ErrD"]) for row in group])
        errr_mean, errr_std = summarize([float(row["ErrR"]) for row in group])
        unseen_mean, unseen_std = summarize([float(row["unseen"]) for row in group])
        weak_mean, weak_std = summarize([float(row["final_weak_loss"]) for row in group])
        strong_mean, strong_std = summarize([float(row["final_strong_loss"]) for row in group])
        nominal_text = "inf" if math.isinf(nominal) else f"{nominal:.2f}"
        realized_text = "inf" if math.isinf(realized_mean) else fmt(realized_mean, realized_std)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{100.0 * noise_level:.0f}%",
                    nominal_text,
                    realized_text,
                    fmt(errd_mean, errd_std),
                    fmt(errr_mean, errr_std),
                    fmt(unseen_mean, unseen_std),
                    fmt(weak_mean, weak_std),
                    fmt(strong_mean, strong_std),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def plot_noise_sweep(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    grouped: dict[float, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["noise_level"])].append(row)

    noise_levels = sorted(grouped)
    x = np.asarray([100.0 * level for level in noise_levels], dtype=np.float64)
    snr_values = [nominal_snr_db(level) for level in noise_levels]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    metric_specs = [
        ("ErrD", "Diffusion closure error", "#1f77b4"),
        ("ErrR", "Reaction closure error", "#d62728"),
    ]

    for axis, (metric_key, title, color) in zip(axes, metric_specs, strict=True):
        means = np.asarray([mean([float(row[metric_key]) for row in grouped[level]]) for level in noise_levels], dtype=np.float64)
        stds = np.asarray(
            [
                pstdev([float(row[metric_key]) for row in grouped[level]]) if len(grouped[level]) > 1 else 0.0
                for level in noise_levels
            ],
            dtype=np.float64,
        )
        axis.plot(x, means, marker="o", linewidth=2, color=color)
        axis.fill_between(x, means - stds, means + stds, color=color, alpha=0.2)
        axis.set_title(title)
        axis.set_xlabel("noise level (% of std(u))")
        axis.set_ylabel("relative error")
        axis.grid(axis="y", linestyle=":", alpha=0.4)

        top = axis.twiny()
        top.set_xlim(axis.get_xlim())
        top.set_xticks(x)
        top.set_xticklabels(["inf" if math.isinf(value) else f"{value:.0f}" for value in snr_values])
        top.set_xlabel("nominal SNR (dB)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    noise_levels = parse_noise_levels(args.noise_levels)
    case = CASE_BUILDERS[args.case]()
    amplitude_range = (
        case.value_range[0] if args.amplitude_min is None else args.amplitude_min,
        case.value_range[1] if args.amplitude_max is None else args.amplitude_max,
    )
    simulation_config = SimulationConfig1D(
        nx=args.nx,
        dt=args.dt,
        t_final=args.t_final,
        save_every=args.save_every,
        boundary="periodic",
    )
    training_config = make_paper_training_config(
        epochs=args.epochs,
        backbone=args.backbone,
        hidden_width=args.hidden_width,
        hidden_depth=args.hidden_depth,
        kan_grid_size=args.kan_grid_size,
        num_test_modes=args.num_modes,
    )
    output_csv = args.output_csv or Path("results") / f"noise_sweep_{args.case}.csv"
    output_md = args.output_md or Path("results") / f"noise_sweep_{args.case}.md"
    output_figure = args.output_figure or Path("results/paper_artifacts/figures") / f"{args.case}_noise_sweep.png"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("noise_sweep_benchmark")
    print(f"  case={args.case}")
    print(f"  noise_levels={noise_levels}")
    print(f"  amplitude_range={amplitude_range}")
    print(f"  num_seeds={args.num_seeds}")
    print(f"  training_objective={summarize_training_objective(training_config)}")

    rows: list[dict[str, float | int | str]] = []
    for offset in range(args.num_seeds):
        seed = args.base_seed + offset
        raw_dataset = generate_dataset(
            case=case,
            config=simulation_config,
            num_trajectories=args.num_trajectories,
            seed=seed,
            amplitude_range=amplitude_range,
            num_modes=args.num_modes,
            initial_clip_range=amplitude_range,
        )
        clean_u = np.asarray(raw_dataset["u"], dtype=np.float64)
        for noise_level in noise_levels:
            result = run_closure_identification(
                case=case,
                simulation_config=simulation_config,
                num_trajectories=args.num_trajectories,
                amplitude_range=amplitude_range,
                num_initial_modes=args.num_modes,
                observation_config=ObservationConfig(noise_level=noise_level, space_stride=1, time_stride=1),
                training_config=training_config,
                initial_clip_range=amplitude_range,
                seed=seed,
                raw_dataset=raw_dataset,
            )
            observed_u = np.asarray(result["dataset"]["u"], dtype=np.float64)
            rows.append(
                {
                    "case": args.case,
                    "seed": seed,
                    "noise_level": noise_level,
                    "noise_percent": 100.0 * noise_level,
                    "nominal_snr_db": nominal_snr_db(noise_level),
                    "realized_snr_db": realized_snr_db(clean_u, observed_u),
                    "ErrD": float(result["metrics"]["relative_error_D"]),
                    "ErrR": float(result["metrics"]["relative_error_R"]),
                    "unseen": float(result["metrics"]["unseen_rollout_relative_l2"]),
                    "final_weak_loss": float(result["metrics"]["final_weak_loss"]),
                    "final_strong_loss": float(result["metrics"]["final_strong_loss"]),
                }
            )
            print(
                f"  seed={seed} noise={noise_level:.2f} "
                f"ErrD={result['metrics']['relative_error_D']:.3e} "
                f"ErrR={result['metrics']['relative_error_R']:.3e} "
                f"unseen={result['metrics']['unseen_rollout_relative_l2']:.3e}"
            )

    fieldnames = [
        "case",
        "seed",
        "noise_level",
        "noise_percent",
        "nominal_snr_db",
        "realized_snr_db",
        "ErrD",
        "ErrR",
        "unseen",
        "final_weak_loss",
        "final_strong_loss",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_md.write_text(build_markdown(rows), encoding="utf-8")
    plot_noise_sweep(rows, output_figure)
    print(f"  csv_saved={output_csv}")
    print(f"  md_saved={output_md}")
    print(f"  figure_saved={output_figure}")


if __name__ == "__main__":
    main()
