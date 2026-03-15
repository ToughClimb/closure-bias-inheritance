from __future__ import annotations

import argparse

import numpy as np

from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D
from closure_discovery.evaluation.metrics import relative_l2_error
from closure_discovery.evaluation.rollout import compare_cases_on_shared_initial_conditions
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig,
    TrainingConfig,
    run_closure_identification,
)
from closure_discovery.symbolic.restricted_fit import (
    SymbolicClosurePair,
    fit_diffusion_candidates,
    fit_reaction_candidates,
    select_symbolic_expression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress a learned neural closure into a restricted symbolic family.")
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_exp")
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
    parser.add_argument("--selection-rel-tol", type=float, default=0.05)
    parser.add_argument("--selection-abs-tol", type=float, default=1.0e-8)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def _print_top_candidates(title: str, candidates, top_k: int) -> None:
    print(title)
    for rank, candidate in enumerate(sorted(candidates, key=lambda item: (item.fit_mse, item.complexity))[:top_k], start=1):
        print(
            f"  {rank}. family={candidate.family} "
            f"| complexity={candidate.complexity} "
            f"| fit_mse={candidate.fit_mse:.6e} "
            f"| expr={candidate.expression}"
        )


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

    neural_result = run_closure_identification(
        case=case,
        simulation_config=simulation_config,
        num_trajectories=args.num_trajectories,
        amplitude_range=amplitude_range,
        num_initial_modes=args.num_modes,
        observation_config=ObservationConfig(
            noise_level=args.noise_level,
            space_stride=args.space_stride,
            time_stride=args.time_stride,
        ),
        training_config=TrainingConfig(
            epochs=args.epochs,
            backbone=args.backbone,
            kan_grid_size=args.kan_grid_size,
            num_test_modes=args.num_modes,
        ),
        seed=args.seed,
    )

    u_grid = np.asarray(neural_result["evaluation_grid"], dtype=np.float64)
    diffusion_true = np.asarray(neural_result["diffusion_true"], dtype=np.float64)
    reaction_true = np.asarray(neural_result["reaction_true"], dtype=np.float64)
    diffusion_neural = np.asarray(neural_result["diffusion_pred"], dtype=np.float64)
    reaction_neural = np.asarray(neural_result["reaction_pred"], dtype=np.float64)

    diffusion_candidates = fit_diffusion_candidates(u_grid, diffusion_neural)
    reaction_candidates = fit_reaction_candidates(u_grid, reaction_neural)
    diffusion_symbolic = select_symbolic_expression(
        diffusion_candidates,
        relative_tolerance=args.selection_rel_tol,
        absolute_tolerance=args.selection_abs_tol,
    )
    reaction_symbolic = select_symbolic_expression(
        reaction_candidates,
        relative_tolerance=args.selection_rel_tol,
        absolute_tolerance=args.selection_abs_tol,
    )

    symbolic_pair = SymbolicClosurePair(
        diffusion_expression=diffusion_symbolic,
        reaction_expression=reaction_symbolic,
        value_range=case.value_range,
        name=f"{case.name}_symbolic_seed_{args.seed}",
    )
    symbolic_case = symbolic_pair.to_case(description="Restricted symbolic compression of neural surrogate")

    diffusion_symbolic_values = diffusion_symbolic.evaluate(u_grid)
    reaction_symbolic_values = reaction_symbolic.evaluate(u_grid)

    symbolic_metrics = {
        "surrogate_error_D": relative_l2_error(diffusion_neural, diffusion_symbolic_values),
        "surrogate_error_R": relative_l2_error(reaction_neural, reaction_symbolic_values),
        "true_error_D": relative_l2_error(diffusion_true, diffusion_symbolic_values),
        "true_error_R": relative_l2_error(reaction_true, reaction_symbolic_values),
    }

    symbolic_rollout = compare_cases_on_shared_initial_conditions(
        true_case=case,
        predicted_case=symbolic_case,
        config=simulation_config,
        num_trajectories=4,
        seed=args.seed + 1000,
        amplitude_range=amplitude_range,
        num_modes=args.num_modes,
    )

    print("symbolic_compression")
    print(f"  case={args.case}")
    print(f"  backbone={args.backbone}")
    print(f"  amplitude_range={amplitude_range}")
    print(
        f"  observation=(noise={args.noise_level}, space_stride={args.space_stride}, "
        f"time_stride={args.time_stride})"
    )
    print(
        f"  neural_metrics=(ErrD={neural_result['metrics']['relative_error_D']:.6e}, "
        f"ErrR={neural_result['metrics']['relative_error_R']:.6e}, "
        f"unseen={neural_result['metrics']['unseen_rollout_relative_l2']:.6e})"
    )

    _print_top_candidates("  diffusion_candidates", diffusion_candidates, top_k=args.top_k)
    _print_top_candidates("  reaction_candidates", reaction_candidates, top_k=args.top_k)

    print("  selected_symbolic")
    print(
        f"    diffusion | family={diffusion_symbolic.family} "
        f"| fit_mse={diffusion_symbolic.fit_mse:.6e} "
        f"| expr={diffusion_symbolic.expression}"
    )
    print(
        f"    reaction  | family={reaction_symbolic.family} "
        f"| fit_mse={reaction_symbolic.fit_mse:.6e} "
        f"| expr={reaction_symbolic.expression}"
    )
    print(
        f"  symbolic_metrics=(surrogate_ErrD={symbolic_metrics['surrogate_error_D']:.6e}, "
        f"surrogate_ErrR={symbolic_metrics['surrogate_error_R']:.6e}, "
        f"true_ErrD={symbolic_metrics['true_error_D']:.6e}, "
        f"true_ErrR={symbolic_metrics['true_error_R']:.6e}, "
        f"unseen={symbolic_rollout.relative_l2:.6e})"
    )


if __name__ == "__main__":
    main()
