from __future__ import annotations

import argparse

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
    parser = argparse.ArgumentParser(description="Compare neural closure discovery against polynomial baselines.")
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_a")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--t-final", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--backbone", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--kan-grid-size", type=int, default=16)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--space-stride", type=int, default=1)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--amplitude-min", type=float, default=None)
    parser.add_argument("--amplitude-max", type=float, default=None)
    parser.add_argument("--diffusion-degree", type=int, default=2)
    parser.add_argument("--reaction-degree", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    return parser.parse_args()


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
    raw_dataset = generate_dataset(
        case=case,
        config=simulation_config,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
        amplitude_range=amplitude_range,
        num_modes=args.num_modes,
        initial_clip_range=amplitude_range,
    )
    observation_config = ObservationConfig(
        noise_level=args.noise_level,
        space_stride=args.space_stride,
        time_stride=args.time_stride,
    )

    neural_result = run_closure_identification(
        case=case,
        simulation_config=simulation_config,
        num_trajectories=args.num_trajectories,
        amplitude_range=amplitude_range,
        num_initial_modes=args.num_modes,
        observation_config=observation_config,
        training_config=make_paper_training_config(
            epochs=args.epochs,
            backbone=args.backbone,
            kan_grid_size=args.kan_grid_size,
            num_test_modes=args.num_modes,
        ),
        initial_clip_range=amplitude_range,
        seed=args.seed,
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
        seed=args.seed,
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
        seed=args.seed,
        raw_dataset=raw_dataset,
        num_test_modes=args.num_modes,
        num_bump_functions=4,
    )

    print("baseline_comparison")
    print(f"  case={args.case}")
    print(f"  amplitude_range={amplitude_range}")
    print(f"  backbone={args.backbone}")
    print(
        f"  observation=(noise={args.noise_level}, space_stride={args.space_stride}, "
        f"time_stride={args.time_stride})"
    )
    print(
        f"  polynomial_library=(diffusion_degree={args.diffusion_degree}, "
        f"reaction_degree={args.reaction_degree})"
    )
    print(
        "  training_objective="
        + summarize_training_objective(
            make_paper_training_config(
                epochs=args.epochs,
                backbone=args.backbone,
                kan_grid_size=args.kan_grid_size,
                num_test_modes=args.num_modes,
            )
        )
    )
    print(
        f"  {args.backbone:<11} | ErrD={neural_result['metrics']['relative_error_D']:.6e} "
        f"| ErrR={neural_result['metrics']['relative_error_R']:.6e} "
        f"| unseen={neural_result['metrics']['unseen_rollout_relative_l2']:.6e}"
    )
    print(
        f"  strong_poly | ErrD={strong_result['metrics']['relative_error_D']:.6e} "
        f"| ErrR={strong_result['metrics']['relative_error_R']:.6e} "
        f"| unseen={strong_result['metrics']['unseen_rollout_relative_l2']:.6e} "
        f"| fit_mse={strong_result['metrics']['fit_mse']:.6e}"
    )
    print(
        f"  weak_poly   | ErrD={weak_result['metrics']['relative_error_D']:.6e} "
        f"| ErrR={weak_result['metrics']['relative_error_R']:.6e} "
        f"| unseen={weak_result['metrics']['unseen_rollout_relative_l2']:.6e} "
        f"| fit_mse={weak_result['metrics']['fit_mse']:.6e}"
    )


if __name__ == "__main__":
    main()
