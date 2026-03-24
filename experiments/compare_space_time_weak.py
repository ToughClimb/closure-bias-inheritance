from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D, generate_dataset
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig,
    TrainingConfig,
    UnseenRolloutConfig,
    make_paper_training_config,
    run_closure_identification,
    summarize_training_objective,
)


def parse_noise_levels(text: str) -> list[float]:
    levels = []
    for piece in text.split(","):
        value = float(piece.strip())
        if value < 0.0:
            raise ValueError("noise levels must be non-negative")
        levels.append(value)
    return sorted(dict.fromkeys(levels))


def summarize(values: list[float]) -> tuple[float, float]:
    return float(mean(values)), float(pstdev(values) if len(values) > 1 else 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare space-weak and space-time weak Stage-1 training.")
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_exp")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--t-final", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="0,0.01,0.03,0.05",
        help="Comma-separated multiples of std(u).",
    )
    parser.add_argument("--amplitude-min", type=float, default=None)
    parser.add_argument("--amplitude-max", type=float, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/space_time_weak_diagnostic.csv"),
    )
    return parser.parse_args()


def make_reduced_config(*, epochs: int, num_modes: int, weak_form_variant: str) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        lr=1.0e-3,
        rollout_weight=0.1,
        mass_weight=0.0,
        strong_weight=0.0,
        reaction_anchor_weight=1.0,
        reg_weight=1.0e-4,
        backbone="mlp",
        hidden_width=64,
        hidden_depth=2,
        diffusion_degree=2,
        reaction_degree=3,
        residual_scale=0.1,
        kan_grid_size=16,
        num_test_modes=num_modes,
        num_bump_functions=4,
        num_time_test_modes=num_modes,
        weak_form_variant=weak_form_variant,
        objective_name=f"{weak_form_variant}_reduced",
    )


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

    configs = {
        "paper_hybrid": make_paper_training_config(epochs=args.epochs, num_test_modes=args.num_modes),
        "space_weak_reduced": make_reduced_config(
            epochs=args.epochs,
            num_modes=args.num_modes,
            weak_form_variant="space_weak",
        ),
        "space_time_weak_reduced": make_reduced_config(
            epochs=args.epochs,
            num_modes=args.num_modes,
            weak_form_variant="space_time_weak",
        ),
    }
    unseen_rollout_config = UnseenRolloutConfig(
        enabled=True,
        num_trajectories=4,
        seed_offset=1000,
        amplitude_range=amplitude_range,
        initial_clip_range=amplitude_range,
        num_modes=args.num_modes,
    )

    rows: list[dict[str, float | int | str]] = []
    print("space_time_weak_diagnostic")
    print(f"  case={args.case}")
    print(f"  noise_levels={noise_levels}")
    for method_name, config in configs.items():
        print(f"  {method_name}: {summarize_training_objective(config)}")

    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        raw_dataset = generate_dataset(
            case=case,
            config=simulation_config,
            num_trajectories=args.num_trajectories,
            seed=seed,
            amplitude_range=amplitude_range,
            num_modes=args.num_modes,
            initial_clip_range=amplitude_range,
        )
        for noise_level in noise_levels:
            observation_config = ObservationConfig(noise_level=noise_level, space_stride=1, time_stride=1)
            for method_name, config in configs.items():
                result = run_closure_identification(
                    case=case,
                    simulation_config=simulation_config,
                    num_trajectories=args.num_trajectories,
                    amplitude_range=amplitude_range,
                    num_initial_modes=args.num_modes,
                    observation_config=observation_config,
                    training_config=config,
                    unseen_rollout_config=unseen_rollout_config,
                    initial_clip_range=amplitude_range,
                    seed=seed,
                    raw_dataset=raw_dataset,
                )
                row = {
                    "method": method_name,
                    "seed": seed,
                    "noise_level": noise_level,
                    "ErrD": float(result["metrics"]["relative_error_D"]),
                    "ErrR": float(result["metrics"]["relative_error_R"]),
                    "unseen": float(result["metrics"]["unseen_rollout_relative_l2"]),
                    "weak_loss": float(result["metrics"]["final_weak_loss"]),
                    "strong_loss": float(result["metrics"]["final_strong_loss"]),
                }
                rows.append(row)
                print(
                    f"seed={seed} noise={noise_level:.3f} method={method_name} "
                    f"ErrD={row['ErrD']:.3e} ErrR={row['ErrR']:.3e} unseen={row['unseen']:.3e} "
                    f"weak={row['weak_loss']:.3e} strong={row['strong_loss']:.3e}"
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "seed", "noise_level", "ErrD", "ErrR", "unseen", "weak_loss", "strong_loss"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nsummary")
    for method_name in configs:
        for noise_level in noise_levels:
            group = [row for row in rows if row["method"] == method_name and float(row["noise_level"]) == noise_level]
            errd_mean, errd_std = summarize([float(row["ErrD"]) for row in group])
            errr_mean, errr_std = summarize([float(row["ErrR"]) for row in group])
            unseen_mean, unseen_std = summarize([float(row["unseen"]) for row in group])
            print(
                f"method={method_name} noise={noise_level:.3f} "
                f"ErrD={errd_mean:.3e}+/-{errd_std:.3e} "
                f"ErrR={errr_mean:.3e}+/-{errr_std:.3e} "
                f"unseen={unseen_mean:.3e}+/-{unseen_std:.3e}"
            )

    print(f"\nwrote {args.output_csv}")


if __name__ == "__main__":
    main()
