from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D, generate_dataset
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


SETTING_LIBRARY = [
    {"name": "clean", "noise_level": 0.0, "space_stride": 1, "time_stride": 1},
    {"name": "noise_1", "noise_level": 0.01, "space_stride": 1, "time_stride": 1},
    {"name": "noise_5", "noise_level": 0.05, "space_stride": 1, "time_stride": 1},
    {"name": "sparse_space", "noise_level": 0.0, "space_stride": 2, "time_stride": 1},
    {"name": "sparse_time", "noise_level": 0.0, "space_stride": 1, "time_stride": 2},
    {"name": "sparse_both", "noise_level": 0.0, "space_stride": 2, "time_stride": 2},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark neural-to-symbolic closure compression across observation settings.")
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_exp")
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
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--kan-grid-size", type=int, default=16)
    parser.add_argument("--selection-rel-tol", type=float, default=0.05)
    parser.add_argument("--selection-abs-tol", type=float, default=1.0e-8)
    parser.add_argument("--amplitude-min", type=float, default=None)
    parser.add_argument("--amplitude-max", type=float, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to results/symbolic_compression_<case>.csv",
    )
    return parser.parse_args()


def summarize(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3e}"
    return f"{mean(values):.3e} ± {pstdev(values):.3e}"


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1.0e-12:
        return float("nan")
    return float(numerator / denominator)


def run_symbolic_compression_once(
    *,
    case,
    simulation_config: SimulationConfig1D,
    num_trajectories: int,
    amplitude_range: tuple[float, float],
    num_modes: int,
    epochs: int,
    backbone: str,
    hidden_width: int,
    hidden_depth: int,
    kan_grid_size: int,
    observation_config: ObservationConfig,
    seed: int,
    selection_rel_tol: float,
    selection_abs_tol: float,
    raw_dataset: dict[str, np.ndarray],
) -> dict[str, object]:
    neural_result = run_closure_identification(
        case=case,
        simulation_config=simulation_config,
        num_trajectories=num_trajectories,
        amplitude_range=amplitude_range,
        num_initial_modes=num_modes,
        observation_config=observation_config,
        training_config=TrainingConfig(
            epochs=epochs,
            backbone=backbone,
            hidden_width=hidden_width,
            hidden_depth=hidden_depth,
            kan_grid_size=kan_grid_size,
            num_test_modes=num_modes,
        ),
        seed=seed,
        raw_dataset=raw_dataset,
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
        relative_tolerance=selection_rel_tol,
        absolute_tolerance=selection_abs_tol,
    )
    reaction_symbolic = select_symbolic_expression(
        reaction_candidates,
        relative_tolerance=selection_rel_tol,
        absolute_tolerance=selection_abs_tol,
    )

    symbolic_pair = SymbolicClosurePair(
        diffusion_expression=diffusion_symbolic,
        reaction_expression=reaction_symbolic,
        value_range=case.value_range,
        name=f"{case.name}_symbolic_seed_{seed}",
    )
    symbolic_case = symbolic_pair.to_case(description="Restricted symbolic compression of neural surrogate")

    diffusion_symbolic_values = diffusion_symbolic.evaluate(u_grid)
    reaction_symbolic_values = reaction_symbolic.evaluate(u_grid)
    symbolic_rollout = compare_cases_on_shared_initial_conditions(
        true_case=case,
        predicted_case=symbolic_case,
        config=simulation_config,
        num_trajectories=4,
        seed=seed + 1000,
        amplitude_range=amplitude_range,
        num_modes=num_modes,
    )

    neural_true_err_d = float(neural_result["metrics"]["relative_error_D"])
    neural_true_err_r = float(neural_result["metrics"]["relative_error_R"])
    symbolic_true_err_d = relative_l2_error(diffusion_true, diffusion_symbolic_values)
    symbolic_true_err_r = relative_l2_error(reaction_true, reaction_symbolic_values)

    return {
        "neural_true_ErrD": neural_true_err_d,
        "neural_true_ErrR": neural_true_err_r,
        "neural_unseen": float(neural_result["metrics"]["unseen_rollout_relative_l2"]),
        "symbolic_family_D": diffusion_symbolic.family,
        "symbolic_family_R": reaction_symbolic.family,
        "symbolic_expr_D": diffusion_symbolic.expression,
        "symbolic_expr_R": reaction_symbolic.expression,
        "symbolic_complexity_D": diffusion_symbolic.complexity,
        "symbolic_complexity_R": reaction_symbolic.complexity,
        "symbolic_surrogate_ErrD": relative_l2_error(diffusion_neural, diffusion_symbolic_values),
        "symbolic_surrogate_ErrR": relative_l2_error(reaction_neural, reaction_symbolic_values),
        "symbolic_true_ErrD": symbolic_true_err_d,
        "symbolic_true_ErrR": symbolic_true_err_r,
        "symbolic_unseen": float(symbolic_rollout.relative_l2),
        "bir_D": safe_ratio(symbolic_true_err_d, neural_true_err_d),
        "bir_R": safe_ratio(symbolic_true_err_r, neural_true_err_r),
    }


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
    output_csv = args.output_csv or Path("results") / f"symbolic_compression_{args.case}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    print("symbolic_compression_benchmark")
    print(f"  case={args.case}")
    print(f"  amplitude_range={amplitude_range}")
    print(f"  backbone={args.backbone}")
    if args.backbone == "mlp":
        print(f"  hidden_width={args.hidden_width}")
        print(f"  hidden_depth={args.hidden_depth}")
    print(f"  num_seeds={args.num_seeds}")

    for setting in SETTING_LIBRARY:
        setting_rows: list[dict[str, object]] = []
        for offset in range(args.num_seeds):
            seed = args.base_seed + offset
            raw_dataset = generate_dataset(
                case=case,
                config=simulation_config,
                num_trajectories=args.num_trajectories,
                seed=seed,
                amplitude_range=amplitude_range,
                num_modes=args.num_modes,
            )
            observation_config = ObservationConfig(
                noise_level=setting["noise_level"],
                space_stride=setting["space_stride"],
                time_stride=setting["time_stride"],
            )
            result_row = run_symbolic_compression_once(
                case=case,
                simulation_config=simulation_config,
                num_trajectories=args.num_trajectories,
                amplitude_range=amplitude_range,
                num_modes=args.num_modes,
                epochs=args.epochs,
                backbone=args.backbone,
                hidden_width=args.hidden_width,
                hidden_depth=args.hidden_depth,
                kan_grid_size=args.kan_grid_size,
                observation_config=observation_config,
                seed=seed,
                selection_rel_tol=args.selection_rel_tol,
                selection_abs_tol=args.selection_abs_tol,
                raw_dataset=raw_dataset,
            )
            result_row.update(
                {
                    "case": args.case,
                    "setting": setting["name"],
                    "seed": seed,
                    "noise_level": setting["noise_level"],
                    "space_stride": setting["space_stride"],
                    "time_stride": setting["time_stride"],
                    "backbone": args.backbone,
                }
            )
            rows.append(result_row)
            setting_rows.append(result_row)

        print(
            f"  setting={setting['name']:<12} "
            f"| neural_true_ErrD={summarize([float(row['neural_true_ErrD']) for row in setting_rows])} "
            f"| neural_true_ErrR={summarize([float(row['neural_true_ErrR']) for row in setting_rows])}"
        )
        print(
            f"                 "
            f"| symbolic_surrogate_ErrD={summarize([float(row['symbolic_surrogate_ErrD']) for row in setting_rows])} "
            f"| symbolic_surrogate_ErrR={summarize([float(row['symbolic_surrogate_ErrR']) for row in setting_rows])}"
        )
        print(
            f"                 "
            f"| symbolic_true_ErrD={summarize([float(row['symbolic_true_ErrD']) for row in setting_rows])} "
            f"| symbolic_true_ErrR={summarize([float(row['symbolic_true_ErrR']) for row in setting_rows])}"
        )
        print(
            f"                 "
            f"| neural_unseen={summarize([float(row['neural_unseen']) for row in setting_rows])} "
            f"| symbolic_unseen={summarize([float(row['symbolic_unseen']) for row in setting_rows])} "
            f"| BIR_D={summarize([float(row['bir_D']) for row in setting_rows])} "
            f"| BIR_R={summarize([float(row['bir_R']) for row in setting_rows])}"
        )

    fieldnames = [
        "case",
        "setting",
        "seed",
        "noise_level",
        "space_stride",
        "time_stride",
        "backbone",
        "neural_true_ErrD",
        "neural_true_ErrR",
        "neural_unseen",
        "symbolic_family_D",
        "symbolic_family_R",
        "symbolic_expr_D",
        "symbolic_expr_R",
        "symbolic_complexity_D",
        "symbolic_complexity_R",
        "symbolic_surrogate_ErrD",
        "symbolic_surrogate_ErrR",
        "symbolic_true_ErrD",
        "symbolic_true_ErrR",
        "symbolic_unseen",
        "bir_D",
        "bir_R",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  csv_saved={output_csv}")


if __name__ == "__main__":
    main()
