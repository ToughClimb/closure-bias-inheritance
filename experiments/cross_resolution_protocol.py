from __future__ import annotations

import argparse

from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D
from closure_discovery.evaluation.rollout import compare_cases_on_shared_initial_conditions
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig,
    TrainingConfig,
    run_closure_identification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cross-resolution anti-inverse-crime experiment.")
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_a")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train-trajectories", type=int, default=8)
    parser.add_argument("--num-test-trajectories", type=int, default=4)
    parser.add_argument("--fine-nx", type=int, default=64)
    parser.add_argument("--fine-dt", type=float, default=1.0e-4)
    parser.add_argument("--fine-t-final", type=float, default=0.05)
    parser.add_argument("--fine-save-every", type=int, default=5)
    parser.add_argument("--space-stride", type=int, default=2)
    parser.add_argument("--time-stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--backbone", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--kan-grid-size", type=int, default=16)
    parser.add_argument("--val-nx", type=int, default=48)
    parser.add_argument("--val-dt", type=float, default=7.5e-5)
    parser.add_argument("--val-t-final", type=float, default=0.05)
    parser.add_argument("--val-save-every", type=int, default=8)
    parser.add_argument("--amplitude-min", type=float, default=None)
    parser.add_argument("--amplitude-max", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = CASE_BUILDERS[args.case]()
    amplitude_range = (
        case.value_range[0] if args.amplitude_min is None else args.amplitude_min,
        case.value_range[1] if args.amplitude_max is None else args.amplitude_max,
    )

    fine_config = SimulationConfig1D(
        nx=args.fine_nx,
        dt=args.fine_dt,
        t_final=args.fine_t_final,
        save_every=args.fine_save_every,
        boundary="periodic",
    )
    observation_config = ObservationConfig(
        noise_level=0.0,
        space_stride=args.space_stride,
        time_stride=args.time_stride,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        backbone=args.backbone,
        kan_grid_size=args.kan_grid_size,
    )

    result = run_closure_identification(
        case=case,
        simulation_config=fine_config,
        num_trajectories=args.num_train_trajectories,
        amplitude_range=amplitude_range,
        num_initial_modes=4,
        observation_config=observation_config,
        training_config=training_config,
        seed=args.seed,
    )

    learned_case = result["tabulated_closure"].to_case(description="Learned closure exported from neural surrogate")
    validation_config = SimulationConfig1D(
        nx=args.val_nx,
        dt=args.val_dt,
        t_final=args.val_t_final,
        save_every=args.val_save_every,
        boundary="periodic",
    )
    rollout = compare_cases_on_shared_initial_conditions(
        true_case=case,
        predicted_case=learned_case,
        config=validation_config,
        num_trajectories=args.num_test_trajectories,
        seed=args.seed + 1000,
        amplitude_range=amplitude_range,
        num_modes=4,
    )

    observed = result["dataset"]
    observed_dx = float(observed["x"][1] - observed["x"][0])
    observed_dt = float(observed["t"][1] - observed["t"][0])

    print("cross_resolution_protocol")
    print(f"  backbone={args.backbone}")
    print(f"  amplitude_range={amplitude_range}")
    print(
        f"  generation_grid=nx:{fine_config.nx}, dx:{fine_config.dx:.3e}, "
        f"saved_dt:{fine_config.dt * fine_config.save_every:.3e}"
    )
    print(
        f"  identification_grid=nx:{observed['x'].shape[0]}, dx:{observed_dx:.3e}, "
        f"saved_dt:{observed_dt:.3e}"
    )
    print(
        f"  validation_grid=nx:{validation_config.nx}, dx:{validation_config.dx:.3e}, "
        f"saved_dt:{validation_config.dt * validation_config.save_every:.3e}"
    )
    print(f"  relative_error_D={result['metrics']['relative_error_D']:.6e}")
    print(f"  relative_error_R={result['metrics']['relative_error_R']:.6e}")
    print(f"  final_weak_loss={result['metrics']['final_weak_loss']:.6e}")
    print(f"  unseen_rollout_relative_l2_train_grid={result['metrics']['unseen_rollout_relative_l2']:.6e}")
    print(f"  forward_rollout_mse={rollout.mse:.6e}")
    print(f"  forward_rollout_relative_l2={rollout.relative_l2:.6e}")


if __name__ == "__main__":
    main()
