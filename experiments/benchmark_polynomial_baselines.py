from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

from closure_discovery.baselines.polynomial import run_polynomial_baseline
from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D, generate_dataset
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig,
    make_paper_training_config,
    run_closure_identification,
    summarize_training_objective,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark neural and polynomial baselines across observation settings.")
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_a")
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
    parser.add_argument("--kan-grid-size", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    parser.add_argument("--amplitude-min", type=float, default=None)
    parser.add_argument("--amplitude-max", type=float, default=None)
    parser.add_argument("--diffusion-degree", type=int, default=2)
    parser.add_argument("--reaction-degree", type=int, default=3)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to results/polynomial_baselines_<case>.csv",
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
    simulation_config = SimulationConfig1D(
        nx=args.nx,
        dt=args.dt,
        t_final=args.t_final,
        save_every=args.save_every,
        boundary="periodic",
    )
    output_csv = args.output_csv or Path("results") / f"polynomial_baselines_{args.case}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    settings = [
        {"name": "clean", "noise_level": 0.0, "space_stride": 1, "time_stride": 1},
        {"name": "noise_1", "noise_level": 0.01, "space_stride": 1, "time_stride": 1},
        {"name": "noise_5", "noise_level": 0.05, "space_stride": 1, "time_stride": 1},
        {"name": "sparse_space", "noise_level": 0.0, "space_stride": 2, "time_stride": 1},
        {"name": "sparse_time", "noise_level": 0.0, "space_stride": 1, "time_stride": 2},
        {"name": "sparse_both", "noise_level": 0.0, "space_stride": 2, "time_stride": 2},
    ]
    rows: list[dict[str, float | int | str]] = []

    print("polynomial_baseline_benchmark")
    print(f"  case={args.case}")
    print(f"  amplitude_range={amplitude_range}")
    print(f"  backbone={args.backbone}")
    print(
        f"  polynomial_library=(diffusion_degree={args.diffusion_degree}, "
        f"reaction_degree={args.reaction_degree})"
    )
    training_config = make_paper_training_config(
        epochs=args.epochs,
        backbone=args.backbone,
        kan_grid_size=args.kan_grid_size,
        num_test_modes=args.num_modes,
    )
    print(f"  training_objective={summarize_training_objective(training_config)}")

    for setting in settings:
        metrics = {
            "neural": {"ErrD": [], "ErrR": [], "unseen": []},
            "strong": {"ErrD": [], "ErrR": [], "unseen": []},
            "weak": {"ErrD": [], "ErrR": [], "unseen": []},
        }

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
            observation_config = ObservationConfig(
                noise_level=setting["noise_level"],
                space_stride=setting["space_stride"],
                time_stride=setting["time_stride"],
            )

            neural_result = run_closure_identification(
                case=case,
                simulation_config=simulation_config,
                num_trajectories=args.num_trajectories,
                amplitude_range=amplitude_range,
                num_initial_modes=args.num_modes,
                observation_config=observation_config,
                training_config=training_config,
                initial_clip_range=amplitude_range,
                seed=seed,
                raw_dataset=raw_dataset,
            )
            strong_result = run_polynomial_baseline(
                method="strong",
                case=case,
                simulation_config=simulation_config,
                num_trajectories=args.num_trajectories,
                amplitude_range=amplitude_range,
                num_initial_modes=args.num_modes,
                observation_config=observation_config,
                diffusion_degree=args.diffusion_degree,
                reaction_degree=args.reaction_degree,
                ridge=args.ridge,
                seed=seed,
                raw_dataset=raw_dataset,
            )
            weak_result = run_polynomial_baseline(
                method="weak",
                case=case,
                simulation_config=simulation_config,
                num_trajectories=args.num_trajectories,
                amplitude_range=amplitude_range,
                num_initial_modes=args.num_modes,
                observation_config=observation_config,
                diffusion_degree=args.diffusion_degree,
                reaction_degree=args.reaction_degree,
                ridge=args.ridge,
                seed=seed,
                raw_dataset=raw_dataset,
                num_test_modes=args.num_modes,
                num_bump_functions=4,
            )

            for key, result in [("neural", neural_result), ("strong", strong_result), ("weak", weak_result)]:
                metrics[key]["ErrD"].append(result["metrics"]["relative_error_D"])
                metrics[key]["ErrR"].append(result["metrics"]["relative_error_R"])
                metrics[key]["unseen"].append(result["metrics"]["unseen_rollout_relative_l2"])
                method_name = {
                    "neural": args.backbone,
                    "strong": "strong_poly",
                    "weak": "weak_poly",
                }[key]
                rows.append(
                    {
                        "case": args.case,
                        "setting": setting["name"],
                        "seed": seed,
                        "noise_level": setting["noise_level"],
                        "space_stride": setting["space_stride"],
                        "time_stride": setting["time_stride"],
                        "method": method_name,
                        "ErrD": result["metrics"]["relative_error_D"],
                        "ErrR": result["metrics"]["relative_error_R"],
                        "unseen": result["metrics"]["unseen_rollout_relative_l2"],
                    }
                )

        print(
            f"  setting={setting['name']} "
            f"(noise={setting['noise_level']}, space={setting['space_stride']}, time={setting['time_stride']})"
        )
        display_names = {"neural": args.backbone, "strong": "strong", "weak": "weak"}
        for key in ["neural", "strong", "weak"]:
            print(
                f"    {display_names[key]:<6} | ErrD={summarize(metrics[key]['ErrD'])} "
                f"| ErrR={summarize(metrics[key]['ErrR'])} "
                f"| unseen={summarize(metrics[key]['unseen'])}"
            )

    fieldnames = [
        "case",
        "setting",
        "seed",
        "noise_level",
        "space_stride",
        "time_stride",
        "method",
        "ErrD",
        "ErrR",
        "unseen",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  csv_saved={output_csv}")


if __name__ == "__main__":
    main()
