from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.datasets import select_trajectories
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D, generate_dataset
from closure_discovery.evaluation.metrics import pairwise_relative_l2_dispersion
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig,
    TrainingConfig,
    run_closure_identification,
)


def _parse_int_list(raw: str) -> list[int]:
    items = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        items.append(int(piece))
    if not items:
        raise ValueError("Expected a non-empty comma-separated list of integers.")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capacity ablation for the Stage-1 neural closure recovery. "
            "Generates one fixed dataset, then sweeps MLP width/depth and repeats across seeds."
        )
    )
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_exp")
    parser.add_argument("--dataset-seed", type=int, default=0, help="Seed for generating the shared master dataset.")
    parser.add_argument("--base-seed", type=int, default=100, help="Base seed for model init/training stochasticity.")
    parser.add_argument("--num-seeds", type=int, default=3, help="Number of independent training seeds per config.")
    parser.add_argument("--master-trajectories", type=int, default=8, help="Number of trajectories in the master dataset.")
    parser.add_argument("--subset-size", type=int, default=None, help="Optional subset size from master trajectories.")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--t-final", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--space-stride", type=int, default=1)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--amplitude-min", type=float, default=None)
    parser.add_argument("--amplitude-max", type=float, default=None)
    parser.add_argument("--widths", type=_parse_int_list, default="64", help="Comma-separated MLP hidden widths.")
    parser.add_argument("--depths", type=_parse_int_list, default="2", help="Comma-separated MLP hidden depths.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/capacity_ablation.csv"),
        help="CSV path for aggregated ablation results.",
    )
    return parser.parse_args()


def summarize(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def main() -> None:
    args = parse_args()
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
    observation_config = ObservationConfig(
        noise_level=args.noise_level,
        space_stride=args.space_stride,
        time_stride=args.time_stride,
    )

    master_dataset = generate_dataset(
        case=case,
        config=simulation_config,
        num_trajectories=args.master_trajectories,
        seed=args.dataset_seed,
        amplitude_range=amplitude_range,
        num_modes=4,
    )

    if args.subset_size is not None:
        subset_size = min(int(args.subset_size), args.master_trajectories)
        indices = np.arange(subset_size, dtype=int)
        master_dataset = select_trajectories(master_dataset, indices)

    rows: list[dict[str, float | int | str]] = []
    print("capacity_ablation")
    print(f"  case={args.case}")
    print(f"  amplitude_range={amplitude_range}")
    print(f"  master_trajectories={args.master_trajectories}")
    if args.subset_size is not None:
        print(f"  subset_size={args.subset_size}")
    print(f"  observation=(noise={args.noise_level}, space_stride={args.space_stride}, time_stride={args.time_stride})")
    print(f"  simulation={asdict(simulation_config)}")
    print(f"  epochs={args.epochs}")
    print(f"  widths={args.widths}")
    print(f"  depths={args.depths}")

    for width in args.widths:
        for depth in args.depths:
            results = []
            for offset in range(args.num_seeds):
                seed = args.base_seed + offset
                result = run_closure_identification(
                    case=case,
                    simulation_config=simulation_config,
                    num_trajectories=args.master_trajectories,
                    amplitude_range=amplitude_range,
                    num_initial_modes=4,
                    observation_config=observation_config,
                    training_config=TrainingConfig(
                        epochs=args.epochs,
                        backbone="mlp",
                        hidden_width=width,
                        hidden_depth=depth,
                    ),
                    seed=seed,
                    raw_dataset=master_dataset,
                )
                results.append(result)

            diffusion_preds = np.stack([result["diffusion_pred"] for result in results], axis=0)
            reaction_preds = np.stack([result["reaction_pred"] for result in results], axis=0)

            err_d = [float(result["metrics"]["relative_error_D"]) for result in results]
            err_r = [float(result["metrics"]["relative_error_R"]) for result in results]
            weak = [float(result["metrics"]["final_weak_loss"]) for result in results]
            unseen = [float(result["metrics"]["unseen_rollout_relative_l2"]) for result in results]

            err_d_mean, err_d_std = summarize(err_d)
            err_r_mean, err_r_std = summarize(err_r)
            weak_mean, weak_std = summarize(weak)
            unseen_mean, unseen_std = summarize(unseen)

            row = {
                "case": args.case,
                "setting": "clean" if args.noise_level == 0.0 and args.space_stride == 1 and args.time_stride == 1 else "custom",
                "dataset_seed": args.dataset_seed,
                "base_seed": args.base_seed,
                "num_seeds": args.num_seeds,
                "master_trajectories": args.master_trajectories,
                "nx": args.nx,
                "dt": args.dt,
                "t_final": args.t_final,
                "save_every": args.save_every,
                "epochs": args.epochs,
                "noise_level": args.noise_level,
                "space_stride": args.space_stride,
                "time_stride": args.time_stride,
                "hidden_width": width,
                "hidden_depth": depth,
                "ErrD_mean": err_d_mean,
                "ErrD_std": err_d_std,
                "ErrR_mean": err_r_mean,
                "ErrR_std": err_r_std,
                "weak_loss_mean": weak_mean,
                "weak_loss_std": weak_std,
                "unseen_mean": unseen_mean,
                "unseen_std": unseen_std,
                "diffusion_dispersion": float(pairwise_relative_l2_dispersion(diffusion_preds)),
                "reaction_dispersion": float(pairwise_relative_l2_dispersion(reaction_preds)),
            }
            rows.append(row)

            print(f"  width={width:<4} depth={depth:<3} | ErrD={err_d_mean:.3e} ± {err_d_std:.3e} | ErrR={err_r_mean:.3e} ± {err_r_std:.3e} | unseen={unseen_mean:.3e} ± {unseen_std:.3e}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"csv_saved={args.output_csv}")


if __name__ == "__main__":
    main()

