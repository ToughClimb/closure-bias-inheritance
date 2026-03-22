from __future__ import annotations

import argparse

from closure_discovery.data_generation.cases import case_a
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig,
    TrainingConfig,
    run_closure_identification,
    summarize_training_objective,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 1D Case A closure discovery MVP.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--t-final", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--rollout-weight", type=float, default=0.1)
    parser.add_argument("--mass-weight", type=float, default=1.0)
    parser.add_argument("--strong-weight", type=float, default=1.0)
    parser.add_argument("--reaction-anchor-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=1.0e-4)
    parser.add_argument("--backbone", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--kan-grid-size", type=int, default=16)
    parser.add_argument("--num-bump-functions", type=int, default=4)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--space-stride", type=int, default=1)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--amplitude-min", type=float, default=0.2)
    parser.add_argument("--amplitude-max", type=float, default=0.8)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    case = case_a()
    config = SimulationConfig1D(
        nx=args.nx,
        dt=args.dt,
        t_final=args.t_final,
        save_every=args.save_every,
        boundary="periodic",
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        rollout_weight=args.rollout_weight,
        mass_weight=args.mass_weight,
        strong_weight=args.strong_weight,
        reaction_anchor_weight=args.reaction_anchor_weight,
        reg_weight=args.reg_weight,
        backbone=args.backbone,
        hidden_width=args.hidden_width,
        hidden_depth=args.hidden_depth,
        kan_grid_size=args.kan_grid_size,
        num_test_modes=args.num_modes,
        num_bump_functions=args.num_bump_functions,
        objective_name="custom_cli",
    )
    result = run_closure_identification(
        case=case,
        simulation_config=config,
        num_trajectories=args.num_trajectories,
        amplitude_range=(args.amplitude_min, args.amplitude_max),
        num_initial_modes=args.num_modes,
        observation_config=ObservationConfig(
            noise_level=args.noise_level,
            space_stride=args.space_stride,
            time_stride=args.time_stride,
        ),
        training_config=training_config,
        initial_clip_range=(args.amplitude_min, args.amplitude_max),
        seed=args.seed,
    )
    excitation = result["excitation"]

    print(
        "excitation_summary "
        f"state=[{excitation.state_min:.3f},{excitation.state_max:.3f}] "
        f"bin_coverage={excitation.state_bin_coverage:.3f} "
        f"grad_rms={excitation.gradient_rms:.3e} "
        f"curv_rms={excitation.curvature_rms:.3e} "
        f"weak_diffusion_energy={excitation.weak_diffusion_energy:.3e}"
    )
    print(f"backbone={args.backbone}")
    print(f"training_objective={summarize_training_objective(training_config)}")
    for record in result["history"]:
        epoch = int(record["epoch"])
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} "
                f"loss={record['loss']:.6e} "
                f"weak={record['weak_loss']:.6e} "
                f"rollout={record['rollout_loss']:.6e} "
                f"mass={record['mass_loss']:.6e} "
                f"strong={record['strong_loss']:.6e} "
                f"anchor={record['reaction_anchor_loss']:.6e} "
                f"reg={record['reg_loss']:.6e}"
            )

    print(f"relative_error_D={result['metrics']['relative_error_D']:.6e}")
    print(f"relative_error_R={result['metrics']['relative_error_R']:.6e}")
    print(f"unseen_rollout_mse={result['metrics']['unseen_rollout_mse']:.6e}")
    print(f"unseen_rollout_relative_l2={result['metrics']['unseen_rollout_relative_l2']:.6e}")
    print(f"final_mass_loss={result['metrics']['final_mass_loss']:.6e}")
    print(f"final_strong_loss={result['metrics']['final_strong_loss']:.6e}")
    print(f"final_reaction_anchor_loss={result['metrics']['final_reaction_anchor_loss']:.6e}")


if __name__ == "__main__":
    main()
