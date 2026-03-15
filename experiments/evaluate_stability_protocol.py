from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate seed and subset stability for closure recovery.")
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_a")
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--num-seed-runs", type=int, default=3)
    parser.add_argument("--num-subset-runs", type=int, default=3)
    parser.add_argument("--master-trajectories", type=int, default=8)
    parser.add_argument("--subset-size", type=int, default=4)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--t-final", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--backbone", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--kan-grid-size", type=int, default=16)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--amplitude-min", type=float, default=None)
    parser.add_argument("--amplitude-max", type=float, default=None)
    return parser.parse_args()


def summarize(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3e}"
    return f"{mean(values):.3e} ± {pstdev(values):.3e}"


def aggregate_results(results: list[dict]) -> dict[str, float]:
    diffusion_preds = np.stack([result["diffusion_pred"] for result in results], axis=0)
    reaction_preds = np.stack([result["reaction_pred"] for result in results], axis=0)

    return {
        "bin_coverage": mean([result["excitation"].state_bin_coverage for result in results]),
        "gradient_rms": mean([result["excitation"].gradient_rms for result in results]),
        "final_weak_loss_mean": mean([result["metrics"]["final_weak_loss"] for result in results]),
        "relative_error_D_mean": mean([result["metrics"]["relative_error_D"] for result in results]),
        "relative_error_R_mean": mean([result["metrics"]["relative_error_R"] for result in results]),
        "relative_error_D_std": pstdev([result["metrics"]["relative_error_D"] for result in results]) if len(results) > 1 else 0.0,
        "relative_error_R_std": pstdev([result["metrics"]["relative_error_R"] for result in results]) if len(results) > 1 else 0.0,
        "unseen_rollout_relative_l2_mean": mean([result["metrics"]["unseen_rollout_relative_l2"] for result in results]),
        "diffusion_dispersion": pairwise_relative_l2_dispersion(diffusion_preds),
        "reaction_dispersion": pairwise_relative_l2_dispersion(reaction_preds),
    }


def print_result_block(title: str, results: list[dict]) -> None:
    aggregate = aggregate_results(results)
    print(title)
    print(f"  runs={len(results)}")
    print(f"  bin_coverage={aggregate['bin_coverage']:.3f}")
    print(f"  gradient_rms={aggregate['gradient_rms']:.3e}")
    print(f"  final_weak_loss={summarize([result['metrics']['final_weak_loss'] for result in results])}")
    print(f"  relative_error_D={summarize([result['metrics']['relative_error_D'] for result in results])}")
    print(f"  relative_error_R={summarize([result['metrics']['relative_error_R'] for result in results])}")
    print(f"  unseen_rollout_relative_l2={summarize([result['metrics']['unseen_rollout_relative_l2'] for result in results])}")
    print(f"  diffusion_dispersion={aggregate['diffusion_dispersion']:.3e}")
    print(f"  reaction_dispersion={aggregate['reaction_dispersion']:.3e}")


def main() -> None:
    args = parse_args()
    case = CASE_BUILDERS[args.case]()
    simulation_config = SimulationConfig1D(
        nx=args.nx,
        dt=args.dt,
        t_final=args.t_final,
        save_every=args.save_every,
        boundary="periodic",
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        backbone=args.backbone,
        kan_grid_size=args.kan_grid_size,
    )
    observation_config = ObservationConfig(noise_level=args.noise_level)
    amplitude_range = (
        case.value_range[0] if args.amplitude_min is None else args.amplitude_min,
        case.value_range[1] if args.amplitude_max is None else args.amplitude_max,
    )

    master_dataset = generate_dataset(
        case=case,
        config=simulation_config,
        num_trajectories=args.master_trajectories,
        seed=args.dataset_seed,
        amplitude_range=amplitude_range,
        num_modes=4,
    )

    seed_results = []
    for offset in range(args.num_seed_runs):
        result = run_closure_identification(
            case=case,
            simulation_config=simulation_config,
            num_trajectories=args.master_trajectories,
            amplitude_range=amplitude_range,
            num_initial_modes=4,
            observation_config=observation_config,
            training_config=training_config,
            seed=args.base_seed + offset,
            raw_dataset=master_dataset,
        )
        seed_results.append(result)

    subset_size = min(args.subset_size, args.master_trajectories)
    subset_results = []
    for offset in range(args.num_subset_runs):
        rng = np.random.default_rng(args.base_seed + offset)
        indices = np.sort(rng.choice(args.master_trajectories, size=subset_size, replace=False))
        subset_dataset = select_trajectories(master_dataset, indices)
        result = run_closure_identification(
            case=case,
            simulation_config=simulation_config,
            num_trajectories=subset_size,
            amplitude_range=amplitude_range,
            num_initial_modes=4,
            observation_config=observation_config,
            training_config=training_config,
            seed=args.base_seed,
            raw_dataset=subset_dataset,
        )
        subset_results.append(result)

    print("stability_protocol")
    print(f"  backbone={args.backbone}")
    print(f"  master_trajectories={args.master_trajectories}")
    print(f"  subset_size={subset_size}")
    print(f"  amplitude_range={amplitude_range}")
    print_result_block("seed_stability", seed_results)
    print_result_block("subset_stability", subset_results)


if __name__ == "__main__":
    main()
