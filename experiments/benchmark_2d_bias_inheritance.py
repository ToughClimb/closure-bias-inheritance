from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.rd_solver_2d import SimulationConfig2D, generate_dataset
from closure_discovery.evaluation.metrics import relative_l2_error
from closure_discovery.evaluation.rollout_2d import compare_cases_on_shared_initial_conditions_2d
from closure_discovery.pipelines.train_2d_closure import (
    ObservationConfig2D,
    make_paper_training_config_2d,
    run_closure_identification_2d,
)
from closure_discovery.symbolic.restricted_fit import (
    SymbolicClosurePair,
    fit_diffusion_candidates,
    fit_reaction_candidates,
    select_symbolic_expression,
)


DEFAULT_CASES = ["case_a", "case_exp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal 2D benchmark for bias inheritance in closure discovery.")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES, choices=sorted(CASE_BUILDERS))
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--t-final", type=float, default=0.03)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--num-trajectories", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--num-test-modes", type=int, default=1)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--space-stride", type=int, default=1)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--selection-rel-tol", type=float, default=0.05)
    parser.add_argument("--selection-abs-tol", type=float, default=1.0e-8)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/benchmark_2d_bias_inheritance.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/benchmark_2d_bias_inheritance.md"),
    )
    return parser.parse_args()


def summarize(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.3e} +/- {std_value:.3e}"


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1.0e-12:
        return float("nan")
    return float(numerator / denominator)


def setting_name(noise_level: float, space_stride: int, time_stride: int) -> str:
    if noise_level == 0.0 and space_stride == 1 and time_stride == 1:
        return "clean"
    return f"noise_{noise_level:g}_space_{space_stride}_time_{time_stride}"


def run_symbolic_compression_once_2d(
    *,
    case,
    simulation_config: SimulationConfig2D,
    num_trajectories: int,
    amplitude_range: tuple[float, float],
    num_modes: int,
    epochs: int,
    hidden_width: int,
    hidden_depth: int,
    num_test_modes: int,
    observation_config: ObservationConfig2D,
    seed: int,
    selection_rel_tol: float,
    selection_abs_tol: float,
    raw_dataset: dict[str, np.ndarray],
) -> dict[str, object]:
    neural_result = run_closure_identification_2d(
        case=case,
        simulation_config=simulation_config,
        num_trajectories=num_trajectories,
        amplitude_range=amplitude_range,
        num_initial_modes=num_modes,
        observation_config=observation_config,
        training_config=make_paper_training_config_2d(
            epochs=epochs,
            hidden_width=hidden_width,
            hidden_depth=hidden_depth,
            num_test_modes=num_test_modes,
        ),
        initial_clip_range=amplitude_range,
        seed=seed,
        raw_dataset=raw_dataset,
    )

    u_grid = np.asarray(neural_result["evaluation_grid"], dtype=np.float64)
    diffusion_true = np.asarray(neural_result["diffusion_true"], dtype=np.float64)
    reaction_true = np.asarray(neural_result["reaction_true"], dtype=np.float64)
    diffusion_neural = np.asarray(neural_result["diffusion_pred"], dtype=np.float64)
    reaction_neural = np.asarray(neural_result["reaction_pred"], dtype=np.float64)

    diffusion_symbolic = select_symbolic_expression(
        fit_diffusion_candidates(u_grid, diffusion_neural),
        relative_tolerance=selection_rel_tol,
        absolute_tolerance=selection_abs_tol,
    )
    reaction_symbolic = select_symbolic_expression(
        fit_reaction_candidates(u_grid, reaction_neural),
        relative_tolerance=selection_rel_tol,
        absolute_tolerance=selection_abs_tol,
    )

    symbolic_pair = SymbolicClosurePair(
        diffusion_expression=diffusion_symbolic,
        reaction_expression=reaction_symbolic,
        value_range=case.value_range,
        name=f"{case.name}_2d_symbolic_seed_{seed}",
    )
    symbolic_case = symbolic_pair.to_case(description="Restricted symbolic compression of 2D neural surrogate")

    diffusion_symbolic_values = diffusion_symbolic.evaluate(u_grid)
    reaction_symbolic_values = reaction_symbolic.evaluate(u_grid)
    symbolic_rollout = compare_cases_on_shared_initial_conditions_2d(
        true_case=case,
        predicted_case=symbolic_case,
        config=simulation_config,
        num_trajectories=3,
        seed=seed + 1000,
        amplitude_range=amplitude_range,
        initial_clip_range=amplitude_range,
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
        "weak_loss": float(neural_result["metrics"]["final_weak_loss"]),
        "state_bin_coverage": float(neural_result["metrics"]["state_bin_coverage"]),
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


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case"])].append(row)

    lines = [
        "# Minimal 2D Bias-Inheritance Benchmark",
        "",
        "Each row reports mean +/- std across seeds.",
        "",
        "| Case | Neural ErrD | Neural ErrR | Symbolic ErrD | Symbolic ErrR | Neural unseen | Symbolic unseen | BIR_D | BIR_R | Weak loss | Coverage |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case_name in DEFAULT_CASES:
        case_rows = grouped.get(case_name, [])
        if not case_rows:
            continue
        summary = {}
        for key in [
            "neural_true_ErrD",
            "neural_true_ErrR",
            "symbolic_true_ErrD",
            "symbolic_true_ErrR",
            "neural_unseen",
            "symbolic_unseen",
            "bir_D",
            "bir_R",
            "weak_loss",
            "state_bin_coverage",
        ]:
            values = [float(row[key]) for row in case_rows]
            summary[key] = summarize(values)
        lines.append(
            "| "
            + " | ".join(
                [
                    case_name,
                    format_mean_std(*summary["neural_true_ErrD"]),
                    format_mean_std(*summary["neural_true_ErrR"]),
                    format_mean_std(*summary["symbolic_true_ErrD"]),
                    format_mean_std(*summary["symbolic_true_ErrR"]),
                    format_mean_std(*summary["neural_unseen"]),
                    format_mean_std(*summary["symbolic_unseen"]),
                    format_mean_std(*summary["bir_D"]),
                    format_mean_std(*summary["bir_R"]),
                    format_mean_std(*summary["weak_loss"]),
                    format_mean_std(*summary["state_bin_coverage"]),
                ]
            )
            + " |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    simulation_config = SimulationConfig2D(
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        t_final=args.t_final,
        save_every=args.save_every,
        boundary="periodic",
    )
    observation_config = ObservationConfig2D(
        noise_level=args.noise_level,
        space_stride=args.space_stride,
        time_stride=args.time_stride,
    )
    setting = setting_name(args.noise_level, args.space_stride, args.time_stride)
    rows: list[dict[str, object]] = []

    print("benchmark_2d_bias_inheritance")
    print(f"  cases={args.cases}")
    print(f"  setting={setting}")
    print(f"  simulation={simulation_config}")
    print(f"  num_seeds={args.num_seeds}")

    for case_name in args.cases:
        case = CASE_BUILDERS[case_name]()
        amplitude_range = case.value_range
        print(f"  case={case_name} amplitude_range={amplitude_range}")
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
            result_row = run_symbolic_compression_once_2d(
                case=case,
                simulation_config=simulation_config,
                num_trajectories=args.num_trajectories,
                amplitude_range=amplitude_range,
                num_modes=args.num_modes,
                epochs=args.epochs,
                hidden_width=args.hidden_width,
                hidden_depth=args.hidden_depth,
                num_test_modes=args.num_test_modes,
                observation_config=observation_config,
                seed=seed,
                selection_rel_tol=args.selection_rel_tol,
                selection_abs_tol=args.selection_abs_tol,
                raw_dataset=raw_dataset,
            )
            result_row.update(
                {
                    "case": case_name,
                    "setting": setting,
                    "seed": seed,
                    "nx": args.nx,
                    "ny": args.ny,
                    "dt": args.dt,
                    "t_final": args.t_final,
                    "save_every": args.save_every,
                    "epochs": args.epochs,
                    "num_trajectories": args.num_trajectories,
                    "num_modes": args.num_modes,
                    "noise_level": args.noise_level,
                    "space_stride": args.space_stride,
                    "time_stride": args.time_stride,
                }
            )
            rows.append(result_row)
            print(
                f"    seed={seed} neural=({result_row['neural_true_ErrD']:.3e}, {result_row['neural_true_ErrR']:.3e}) "
                f"symbolic=({result_row['symbolic_true_ErrD']:.3e}, {result_row['symbolic_true_ErrR']:.3e}) "
                f"unseen=({result_row['neural_unseen']:.3e}, {result_row['symbolic_unseen']:.3e}) "
                f"BIR=({result_row['bir_D']:.3e}, {result_row['bir_R']:.3e})"
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.output_md, rows)
    print(f"  saved_csv={args.output_csv}")
    print(f"  saved_md={args.output_md}")


if __name__ == "__main__":
    main()
