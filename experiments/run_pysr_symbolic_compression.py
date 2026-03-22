from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

from closure_discovery.data_generation.cases import CASE_BUILDERS, ReactionDiffusionCase
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D
from closure_discovery.evaluation.metrics import relative_l2_error
from closure_discovery.evaluation.rollout import compare_cases_on_shared_initial_conditions
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig,
    make_paper_training_config,
    run_closure_identification,
    summarize_training_objective,
)


SETTING_MAP = {
    (0.0, 1, 1): "clean",
    (0.01, 1, 1): "noise_1",
    (0.05, 1, 1): "noise_5",
    (0.0, 2, 1): "sparse_space",
    (0.0, 1, 2): "sparse_time",
    (0.0, 2, 2): "sparse_both",
}


@dataclass(frozen=True)
class PySRExpression:
    expression: str
    complexity: int
    fit_loss: float
    predict: callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace Stage 2 symbolic compression with PySR for a single neural surrogate."
    )
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_exp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--t-final", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--space-stride", type=int, default=1)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--amplitude-min", type=float, default=None)
    parser.add_argument("--amplitude-max", type=float, default=None)
    parser.add_argument("--pysr-niterations", type=int, default=40)
    parser.add_argument("--pysr-populations", type=int, default=8)
    parser.add_argument("--pysr-population-size", type=int, default=48)
    parser.add_argument("--pysr-maxsize", type=int, default=30)
    parser.add_argument("--pysr-timeout", type=float, default=180.0)
    parser.add_argument(
        "--operator-mode",
        choices=["exp", "exp_div"],
        default="exp",
        help="Operator set for PySR. exp is sufficient to represent case_exp exactly.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("results") / "symbolic_compression_case_exp.csv",
        help="Optional reference CSV from the restricted symbolic stage for side-by-side comparison.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to results/pysr_symbolic_<case>_seed<seed>.csv",
    )
    return parser.parse_args()


def _ensure_vector(values: np.ndarray | float, template: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == ():
        return np.full_like(template, float(array), dtype=np.float64)
    return array


def _make_callable(expr) -> callable:
    symbol = sp.Symbol("x0")
    fn = sp.lambdify(symbol, expr, modules=["numpy"])

    def evaluate(values: np.ndarray) -> np.ndarray:
        support = np.asarray(values, dtype=np.float64)
        prediction = _ensure_vector(fn(support), support)
        if not np.all(np.isfinite(prediction)):
            raise ValueError(f"Non-finite PySR prediction encountered for expression: {expr}")
        return prediction

    return evaluate


def _fit_pysr_expression(
    *,
    name: str,
    u_grid: np.ndarray,
    target_values: np.ndarray,
    seed: int,
    output_dir: Path,
    niterations: int,
    populations: int,
    population_size: int,
    maxsize: int,
    timeout_in_seconds: float,
    operator_mode: str,
) -> PySRExpression:
    try:
        from pysr import PySRRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PySR is not installed. Install it separately to run the optional PySR Stage-2 benchmark."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    binary_operators = ["+", "-", "*"]
    if operator_mode == "exp_div":
        binary_operators.append("/")
    unary_operators = ["exp"]

    model: Any = PySRRegressor(
        model_selection="best",
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        niterations=niterations,
        populations=populations,
        population_size=population_size,
        maxsize=maxsize,
        timeout_in_seconds=timeout_in_seconds,
        parsimony=1.0e-4,
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        delete_tempfiles=False,
        tempdir=str(output_dir),
    )
    model.fit(u_grid.reshape(-1, 1), target_values)
    best = model.get_best()
    expression = best["sympy_format"]
    complexity = int(best["complexity"])
    fit_loss = float(best["loss"])
    return PySRExpression(
        expression=str(expression),
        complexity=complexity,
        fit_loss=fit_loss,
        predict=_make_callable(expression),
    )


def _build_pysr_case(
    *,
    case: ReactionDiffusionCase,
    seed: int,
    diffusion_expression: PySRExpression,
    reaction_expression: PySRExpression,
) -> ReactionDiffusionCase:
    return ReactionDiffusionCase(
        name=f"{case.name}_pysr_seed_{seed}",
        description="PySR compression of the learned neural surrogate",
        diffusion=lambda u: diffusion_expression.predict(u),
        reaction=lambda u: reaction_expression.predict(u),
        value_range=case.value_range,
    )


def _reference_row(
    reference_csv: Path,
    *,
    case: str,
    setting: str,
    seed: int,
) -> dict[str, str] | None:
    if not reference_csv.exists():
        return None

    with reference_csv.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["case"] == case and row["setting"] == setting and int(row["seed"]) == seed:
                return row
    return None


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
            backbone="mlp",
            hidden_width=args.hidden_width,
            hidden_depth=args.hidden_depth,
            num_test_modes=args.num_modes,
        ),
        initial_clip_range=amplitude_range,
        seed=args.seed,
    )

    u_grid = np.asarray(neural_result["evaluation_grid"], dtype=np.float64)
    diffusion_true = np.asarray(neural_result["diffusion_true"], dtype=np.float64)
    reaction_true = np.asarray(neural_result["reaction_true"], dtype=np.float64)
    diffusion_neural = np.asarray(neural_result["diffusion_pred"], dtype=np.float64)
    reaction_neural = np.asarray(neural_result["reaction_pred"], dtype=np.float64)

    setting_name = SETTING_MAP.get(
        (args.noise_level, args.space_stride, args.time_stride),
        f"noise_{args.noise_level:g}_space_{args.space_stride}_time_{args.time_stride}",
    )
    pysr_root = Path("results") / "pysr_runs" / args.case / setting_name / f"seed_{args.seed}"
    diffusion_expr = _fit_pysr_expression(
        name="diffusion",
        u_grid=u_grid,
        target_values=diffusion_neural,
        seed=args.seed,
        output_dir=pysr_root / "diffusion",
        niterations=args.pysr_niterations,
        populations=args.pysr_populations,
        population_size=args.pysr_population_size,
        maxsize=args.pysr_maxsize,
        timeout_in_seconds=args.pysr_timeout,
        operator_mode=args.operator_mode,
    )
    reaction_expr = _fit_pysr_expression(
        name="reaction",
        u_grid=u_grid,
        target_values=reaction_neural,
        seed=args.seed + 101,
        output_dir=pysr_root / "reaction",
        niterations=args.pysr_niterations,
        populations=args.pysr_populations,
        population_size=args.pysr_population_size,
        maxsize=args.pysr_maxsize,
        timeout_in_seconds=args.pysr_timeout,
        operator_mode=args.operator_mode,
    )

    diffusion_pysr = diffusion_expr.predict(u_grid)
    reaction_pysr = reaction_expr.predict(u_grid)
    pysr_case = _build_pysr_case(
        case=case,
        seed=args.seed,
        diffusion_expression=diffusion_expr,
        reaction_expression=reaction_expr,
    )
    rollout = compare_cases_on_shared_initial_conditions(
        true_case=case,
        predicted_case=pysr_case,
        config=simulation_config,
        num_trajectories=4,
        seed=args.seed + 1000,
        amplitude_range=amplitude_range,
        initial_clip_range=amplitude_range,
        num_modes=args.num_modes,
    )

    neural_true_err_d = float(neural_result["metrics"]["relative_error_D"])
    neural_true_err_r = float(neural_result["metrics"]["relative_error_R"])
    pysr_true_err_d = relative_l2_error(diffusion_true, diffusion_pysr)
    pysr_true_err_r = relative_l2_error(reaction_true, reaction_pysr)
    result = {
        "case": args.case,
        "setting": setting_name,
        "seed": args.seed,
        "noise_level": args.noise_level,
        "space_stride": args.space_stride,
        "time_stride": args.time_stride,
        "operator_mode": args.operator_mode,
        "neural_true_ErrD": neural_true_err_d,
        "neural_true_ErrR": neural_true_err_r,
        "neural_unseen": float(neural_result["metrics"]["unseen_rollout_relative_l2"]),
        "pysr_expr_D": diffusion_expr.expression,
        "pysr_expr_R": reaction_expr.expression,
        "pysr_complexity_D": diffusion_expr.complexity,
        "pysr_complexity_R": reaction_expr.complexity,
        "pysr_fit_loss_D": diffusion_expr.fit_loss,
        "pysr_fit_loss_R": reaction_expr.fit_loss,
        "pysr_surrogate_ErrD": relative_l2_error(diffusion_neural, diffusion_pysr),
        "pysr_surrogate_ErrR": relative_l2_error(reaction_neural, reaction_pysr),
        "pysr_true_ErrD": pysr_true_err_d,
        "pysr_true_ErrR": pysr_true_err_r,
        "pysr_unseen": float(rollout.relative_l2),
        "bir_D": pysr_true_err_d / (neural_true_err_d + 1.0e-12),
        "bir_R": pysr_true_err_r / (neural_true_err_r + 1.0e-12),
    }

    output_csv = args.output_csv or Path("results") / f"pysr_symbolic_{args.case}_seed{args.seed}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

    print("pysr_symbolic_compression")
    print(f"  case={args.case}")
    print(f"  setting={setting_name}")
    print(f"  seed={args.seed}")
    print(f"  observation=(noise={args.noise_level}, space_stride={args.space_stride}, time_stride={args.time_stride})")
    print(
        "  training_objective="
        + summarize_training_objective(
            make_paper_training_config(
                epochs=args.epochs,
                backbone="mlp",
                hidden_width=args.hidden_width,
                hidden_depth=args.hidden_depth,
                num_test_modes=args.num_modes,
            )
        )
    )
    print(
        f"  neural_metrics=(ErrD={neural_true_err_d:.6e}, ErrR={neural_true_err_r:.6e}, "
        f"unseen={result['neural_unseen']:.6e})"
    )
    print(
        f"  pysr_diffusion=(complexity={diffusion_expr.complexity}, fit_loss={diffusion_expr.fit_loss:.6e}, "
        f"expr={diffusion_expr.expression})"
    )
    print(
        f"  pysr_reaction=(complexity={reaction_expr.complexity}, fit_loss={reaction_expr.fit_loss:.6e}, "
        f"expr={reaction_expr.expression})"
    )
    print(
        f"  pysr_metrics=(surrogate_ErrD={result['pysr_surrogate_ErrD']:.6e}, "
        f"surrogate_ErrR={result['pysr_surrogate_ErrR']:.6e}, "
        f"true_ErrD={result['pysr_true_ErrD']:.6e}, "
        f"true_ErrR={result['pysr_true_ErrR']:.6e}, "
        f"unseen={result['pysr_unseen']:.6e}, "
        f"BIR_D={result['bir_D']:.6e}, BIR_R={result['bir_R']:.6e})"
    )

    reference = _reference_row(
        args.reference_csv,
        case=args.case,
        setting=setting_name,
        seed=args.seed,
    )
    if reference is not None:
        print("  restricted_reference")
        print(
            f"    symbolic_metrics=(surrogate_ErrD={float(reference['symbolic_surrogate_ErrD']):.6e}, "
            f"surrogate_ErrR={float(reference['symbolic_surrogate_ErrR']):.6e}, "
            f"true_ErrD={float(reference['symbolic_true_ErrD']):.6e}, "
            f"true_ErrR={float(reference['symbolic_true_ErrR']):.6e}, "
            f"unseen={float(reference['symbolic_unseen']):.6e}, "
            f"BIR_D={float(reference['bir_D']):.6e}, BIR_R={float(reference['bir_R']):.6e})"
        )
        print(f"    symbolic_expr_D={reference['symbolic_expr_D']}")
        print(f"    symbolic_expr_R={reference['symbolic_expr_R']}")

    print(f"  output_csv={output_csv}")


if __name__ == "__main__":
    main()
