from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D
from closure_discovery.evaluation.rollout import compare_cases_on_shared_initial_conditions
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig,
    make_paper_training_config,
    run_closure_identification,
    summarize_training_objective,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cross-resolution anti-inverse-crime settings.")
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_a")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-train-trajectories", type=int, default=8)
    parser.add_argument("--num-test-trajectories", type=int, default=4)
    parser.add_argument("--fine-nx", type=int, default=64)
    parser.add_argument("--fine-dt", type=float, default=1.0e-4)
    parser.add_argument("--fine-t-final", type=float, default=0.05)
    parser.add_argument("--fine-save-every", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--backbone", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--kan-grid-size", type=int, default=16)
    parser.add_argument("--val-nx", type=int, default=48)
    parser.add_argument("--val-dt", type=float, default=7.5e-5)
    parser.add_argument("--val-t-final", type=float, default=0.05)
    parser.add_argument("--val-save-every", type=int, default=8)
    parser.add_argument("--amplitude-min", type=float, default=None)
    parser.add_argument("--amplitude-max", type=float, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path for aggregated cross-resolution results.",
    )
    return parser.parse_args()


def summarize(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3e}"
    return f"{mean(values):.3e} ± {pstdev(values):.3e}"


def main() -> None:
    args = parse_args()
    case = CASE_BUILDERS[args.case]()
    amplitude_range = (
        case.value_range[0] if args.amplitude_min is None else args.amplitude_min,
        case.value_range[1] if args.amplitude_max is None else args.amplitude_max,
    )

    generation_config = SimulationConfig1D(
        nx=args.fine_nx,
        dt=args.fine_dt,
        t_final=args.fine_t_final,
        save_every=args.fine_save_every,
        boundary="periodic",
    )
    validation_config = SimulationConfig1D(
        nx=args.val_nx,
        dt=args.val_dt,
        t_final=args.val_t_final,
        save_every=args.val_save_every,
        boundary="periodic",
    )
    training_config = make_paper_training_config(
        epochs=args.epochs,
        backbone=args.backbone,
        kan_grid_size=args.kan_grid_size,
    )
    settings = [
        {"space_stride": 1, "time_stride": 1},
        {"space_stride": 2, "time_stride": 1},
        {"space_stride": 1, "time_stride": 2},
        {"space_stride": 2, "time_stride": 2},
    ]

    print("cross_resolution_benchmark")
    print(f"  case={args.case}")
    print(f"  amplitude_range={amplitude_range}")
    print(f"  backbone={args.backbone}")
    print(
        f"  generation_grid=nx:{generation_config.nx}, dx:{generation_config.dx:.3e}, "
        f"saved_dt:{generation_config.saved_dt:.3e}, saved_t_final:{generation_config.last_saved_time:.3e}"
    )
    print(
        f"  validation_grid=nx:{validation_config.nx}, dx:{validation_config.dx:.3e}, "
        f"saved_dt:{validation_config.saved_dt:.3e}, saved_t_final:{validation_config.last_saved_time:.3e}"
    )
    print(f"  training_objective={summarize_training_objective(training_config)}")
    rows: list[dict[str, float | str]] = []

    for setting in settings:
        aggregated: dict[str, list[float]] = {
            "relative_error_D": [],
            "relative_error_R": [],
            "final_weak_loss": [],
            "unseen_rollout_relative_l2": [],
            "cross_grid_rollout_relative_l2": [],
        }

        for offset in range(args.num_seeds):
            seed = args.base_seed + offset
            result = run_closure_identification(
                case=case,
                simulation_config=generation_config,
                num_trajectories=args.num_train_trajectories,
                amplitude_range=amplitude_range,
                num_initial_modes=4,
                observation_config=ObservationConfig(
                    noise_level=0.0,
                    space_stride=setting["space_stride"],
                    time_stride=setting["time_stride"],
                ),
                training_config=training_config,
                initial_clip_range=amplitude_range,
                seed=seed,
            )
            learned_case = result["tabulated_closure"].to_case(
                description="Learned closure exported from neural surrogate"
            )
            rollout = compare_cases_on_shared_initial_conditions(
                true_case=case,
                predicted_case=learned_case,
                config=validation_config,
                num_trajectories=args.num_test_trajectories,
                seed=seed + 1000,
                amplitude_range=amplitude_range,
                initial_clip_range=amplitude_range,
                num_modes=4,
            )

            aggregated["relative_error_D"].append(result["metrics"]["relative_error_D"])
            aggregated["relative_error_R"].append(result["metrics"]["relative_error_R"])
            aggregated["final_weak_loss"].append(result["metrics"]["final_weak_loss"])
            aggregated["unseen_rollout_relative_l2"].append(result["metrics"]["unseen_rollout_relative_l2"])
            aggregated["cross_grid_rollout_relative_l2"].append(rollout.relative_l2)

        print(
            f"  stride(space={setting['space_stride']}, time={setting['time_stride']}) "
            f"| ErrD={summarize(aggregated['relative_error_D'])} "
            f"| ErrR={summarize(aggregated['relative_error_R'])} "
            f"| weak={summarize(aggregated['final_weak_loss'])} "
            f"| unseen={summarize(aggregated['unseen_rollout_relative_l2'])} "
            f"| cross_grid={summarize(aggregated['cross_grid_rollout_relative_l2'])}"
        )

        rows.append(
            {
                "case": case.name,
                "backbone": args.backbone,
                "space_stride": float(setting["space_stride"]),
                "time_stride": float(setting["time_stride"]),
                "fine_nx": float(generation_config.nx),
                "fine_saved_dt": generation_config.saved_dt,
                "fine_saved_t_final": generation_config.last_saved_time,
                "val_nx": float(validation_config.nx),
                "val_saved_dt": validation_config.saved_dt,
                "val_saved_t_final": validation_config.last_saved_time,
                "num_seeds": float(args.num_seeds),
                "num_train_trajectories": float(args.num_train_trajectories),
                "num_test_trajectories": float(args.num_test_trajectories),
                "epochs": float(args.epochs),
                "relative_error_D_mean": mean(aggregated["relative_error_D"]),
                "relative_error_D_std": pstdev(aggregated["relative_error_D"]) if len(aggregated["relative_error_D"]) > 1 else 0.0,
                "relative_error_R_mean": mean(aggregated["relative_error_R"]),
                "relative_error_R_std": pstdev(aggregated["relative_error_R"]) if len(aggregated["relative_error_R"]) > 1 else 0.0,
                "final_weak_loss_mean": mean(aggregated["final_weak_loss"]),
                "final_weak_loss_std": pstdev(aggregated["final_weak_loss"]) if len(aggregated["final_weak_loss"]) > 1 else 0.0,
                "unseen_rollout_relative_l2_mean": mean(aggregated["unseen_rollout_relative_l2"]),
                "unseen_rollout_relative_l2_std": pstdev(aggregated["unseen_rollout_relative_l2"])
                if len(aggregated["unseen_rollout_relative_l2"]) > 1
                else 0.0,
                "cross_grid_rollout_relative_l2_mean": mean(aggregated["cross_grid_rollout_relative_l2"]),
                "cross_grid_rollout_relative_l2_std": pstdev(aggregated["cross_grid_rollout_relative_l2"])
                if len(aggregated["cross_grid_rollout_relative_l2"]) > 1
                else 0.0,
            }
        )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote_csv={args.output_csv}")


if __name__ == "__main__":
    main()
