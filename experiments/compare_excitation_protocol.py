from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig,
    make_paper_training_config,
    run_closure_identification,
    summarize_training_objective,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare low and high excitation training regimes.")
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_a")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-trajectories", type=int, default=6)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--t-final", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--backbone", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--kan-grid-size", type=int, default=16)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path for aggregated excitation results.",
    )
    return parser.parse_args()


def summarize(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3e}"
    return f"{mean(values):.3e} ± {pstdev(values):.3e}"


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
    training_config = make_paper_training_config(
        epochs=args.epochs,
        backbone=args.backbone,
        kan_grid_size=args.kan_grid_size,
    )
    observation_config = ObservationConfig(noise_level=args.noise_level)

    lower, upper = case.value_range
    center = 0.5 * (lower + upper)
    half_width = 0.05 * (upper - lower)
    regimes = {
        "low_excitation": (center - half_width, center + half_width),
        "high_excitation": case.value_range,
    }

    print(f"backbone={args.backbone}")
    print(f"training_objective={summarize_training_objective(training_config)}")
    rows: list[dict[str, float | str]] = []

    for name, amplitude_range in regimes.items():
        metrics_by_name: dict[str, list[float]] = {
            "bin_coverage": [],
            "gradient_rms": [],
            "curvature_rms": [],
            "weak_diffusion_energy": [],
            "final_weak_loss": [],
            "relative_error_D": [],
            "relative_error_R": [],
            "unseen_rollout_relative_l2": [],
        }

        for offset in range(args.num_seeds):
            result = run_closure_identification(
                case=case,
                simulation_config=simulation_config,
                num_trajectories=args.num_trajectories,
                amplitude_range=amplitude_range,
                num_initial_modes=4,
                observation_config=observation_config,
                training_config=training_config,
                initial_clip_range=amplitude_range,
                seed=args.base_seed + offset,
            )
            excitation = result["excitation"]
            metrics = result["metrics"]
            metrics_by_name["bin_coverage"].append(excitation.state_bin_coverage)
            metrics_by_name["gradient_rms"].append(excitation.gradient_rms)
            metrics_by_name["curvature_rms"].append(excitation.curvature_rms)
            metrics_by_name["weak_diffusion_energy"].append(excitation.weak_diffusion_energy)
            metrics_by_name["final_weak_loss"].append(metrics["final_weak_loss"])
            metrics_by_name["relative_error_D"].append(metrics["relative_error_D"])
            metrics_by_name["relative_error_R"].append(metrics["relative_error_R"])
            metrics_by_name["unseen_rollout_relative_l2"].append(metrics["unseen_rollout_relative_l2"])

        print(name)
        print(f"  amplitude_range={amplitude_range}")
        print(f"  bin_coverage={summarize(metrics_by_name['bin_coverage'])}")
        print(f"  gradient_rms={summarize(metrics_by_name['gradient_rms'])}")
        print(f"  curvature_rms={summarize(metrics_by_name['curvature_rms'])}")
        print(f"  weak_diffusion_energy={summarize(metrics_by_name['weak_diffusion_energy'])}")
        print(f"  final_weak_loss={summarize(metrics_by_name['final_weak_loss'])}")
        print(f"  relative_error_D={summarize(metrics_by_name['relative_error_D'])}")
        print(f"  relative_error_R={summarize(metrics_by_name['relative_error_R'])}")
        print(f"  unseen_rollout_relative_l2={summarize(metrics_by_name['unseen_rollout_relative_l2'])}")

        rows.append(
            {
                "case": case.name,
                "backbone": args.backbone,
                "regime": name,
                "amplitude_min": float(amplitude_range[0]),
                "amplitude_max": float(amplitude_range[1]),
                "noise_level": float(args.noise_level),
                "num_seeds": float(args.num_seeds),
                "num_trajectories": float(args.num_trajectories),
                "epochs": float(args.epochs),
                "bin_coverage_mean": mean(metrics_by_name["bin_coverage"]),
                "bin_coverage_std": pstdev(metrics_by_name["bin_coverage"]) if len(metrics_by_name["bin_coverage"]) > 1 else 0.0,
                "gradient_rms_mean": mean(metrics_by_name["gradient_rms"]),
                "gradient_rms_std": pstdev(metrics_by_name["gradient_rms"]) if len(metrics_by_name["gradient_rms"]) > 1 else 0.0,
                "curvature_rms_mean": mean(metrics_by_name["curvature_rms"]),
                "curvature_rms_std": pstdev(metrics_by_name["curvature_rms"]) if len(metrics_by_name["curvature_rms"]) > 1 else 0.0,
                "weak_diffusion_energy_mean": mean(metrics_by_name["weak_diffusion_energy"]),
                "weak_diffusion_energy_std": pstdev(metrics_by_name["weak_diffusion_energy"]) if len(metrics_by_name["weak_diffusion_energy"]) > 1 else 0.0,
                "final_weak_loss_mean": mean(metrics_by_name["final_weak_loss"]),
                "final_weak_loss_std": pstdev(metrics_by_name["final_weak_loss"]) if len(metrics_by_name["final_weak_loss"]) > 1 else 0.0,
                "relative_error_D_mean": mean(metrics_by_name["relative_error_D"]),
                "relative_error_D_std": pstdev(metrics_by_name["relative_error_D"]) if len(metrics_by_name["relative_error_D"]) > 1 else 0.0,
                "relative_error_R_mean": mean(metrics_by_name["relative_error_R"]),
                "relative_error_R_std": pstdev(metrics_by_name["relative_error_R"]) if len(metrics_by_name["relative_error_R"]) > 1 else 0.0,
                "unseen_rollout_relative_l2_mean": mean(metrics_by_name["unseen_rollout_relative_l2"]),
                "unseen_rollout_relative_l2_std": pstdev(metrics_by_name["unseen_rollout_relative_l2"])
                if len(metrics_by_name["unseen_rollout_relative_l2"]) > 1
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
