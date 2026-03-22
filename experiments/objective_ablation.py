from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev

from closure_discovery.data_generation.cases import CASE_BUILDERS
from closure_discovery.data_generation.rd_solver_1d import SimulationConfig1D, generate_dataset
from closure_discovery.pipelines.train_1d_closure import (
    ObservationConfig,
    TrainingConfig,
    make_paper_training_config,
    run_closure_identification,
    summarize_training_objective,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate the Stage-1 objective on a fixed clean dataset. "
            "This is intended as a minimal paper-facing check of whether the current conclusions "
            "depend on the full weak-form-driven hybrid objective."
        )
    )
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS), default="case_exp")
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--t-final", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/objective_ablation_case_exp_clean.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/objective_ablation_case_exp_clean.md"),
    )
    return parser.parse_args()


def summarize(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def build_objectives(args: argparse.Namespace) -> list[TrainingConfig]:
    common = {
        "epochs": args.epochs,
        "backbone": "mlp",
        "hidden_width": args.hidden_width,
        "hidden_depth": args.hidden_depth,
    }
    weak_only = TrainingConfig(
        **common,
        rollout_weight=0.0,
        mass_weight=0.0,
        strong_weight=0.0,
        reaction_anchor_weight=0.0,
        reg_weight=0.0,
        objective_name="weak_only",
    )
    weak_plus_aux_no_strong = TrainingConfig(
        **common,
        rollout_weight=0.1,
        mass_weight=1.0,
        strong_weight=0.0,
        reaction_anchor_weight=1.0,
        reg_weight=1.0e-4,
        objective_name="weak_plus_aux_no_strong",
    )
    full_hybrid = make_paper_training_config(**common)
    return [weak_only, weak_plus_aux_no_strong, full_hybrid]


def write_markdown(path: Path, rows: list[dict[str, float | str]]) -> None:
    lines = [
        "# Stage-1 Objective Ablation",
        "",
        "Fixed clean dataset on `case_exp`; each row reports mean +/- std across seeds.",
        "",
        "| Objective | ErrD | ErrR | Weak loss | Rollout | Strong loss |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["objective_name"]),
                    f"{float(row['ErrD_mean']):.3e} +/- {float(row['ErrD_std']):.3e}",
                    f"{float(row['ErrR_mean']):.3e} +/- {float(row['ErrR_std']):.3e}",
                    f"{float(row['weak_loss_mean']):.3e} +/- {float(row['weak_loss_std']):.3e}",
                    f"{float(row['unseen_mean']):.3e} +/- {float(row['unseen_std']):.3e}",
                    f"{float(row['strong_loss_mean']):.3e} +/- {float(row['strong_loss_std']):.3e}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    case = CASE_BUILDERS[args.case]()
    amplitude_range = case.value_range
    simulation_config = SimulationConfig1D(
        nx=args.nx,
        dt=args.dt,
        t_final=args.t_final,
        save_every=args.save_every,
        boundary="periodic",
    )
    observation_config = ObservationConfig(noise_level=0.0, space_stride=1, time_stride=1)

    master_dataset = generate_dataset(
        case=case,
        config=simulation_config,
        num_trajectories=args.num_trajectories,
        seed=args.dataset_seed,
        amplitude_range=amplitude_range,
        num_modes=args.num_modes,
        initial_clip_range=amplitude_range,
    )
    objectives = build_objectives(args)

    print("objective_ablation")
    print(f"  case={args.case}")
    print(f"  simulation={asdict(simulation_config)}")
    print(f"  amplitude_range={amplitude_range}")
    print(f"  num_trajectories={args.num_trajectories}")
    print(f"  num_seeds={args.num_seeds}")

    rows: list[dict[str, float | str]] = []
    for training_config in objectives:
        print(f"  objective={summarize_training_objective(training_config)}")
        err_d_values: list[float] = []
        err_r_values: list[float] = []
        weak_values: list[float] = []
        unseen_values: list[float] = []
        strong_values: list[float] = []
        total_values: list[float] = []

        for offset in range(args.num_seeds):
            seed = args.base_seed + offset
            result = run_closure_identification(
                case=case,
                simulation_config=simulation_config,
                num_trajectories=args.num_trajectories,
                amplitude_range=amplitude_range,
                num_initial_modes=args.num_modes,
                observation_config=observation_config,
                training_config=training_config,
                initial_clip_range=amplitude_range,
                seed=seed,
                raw_dataset=master_dataset,
            )
            err_d_values.append(float(result["metrics"]["relative_error_D"]))
            err_r_values.append(float(result["metrics"]["relative_error_R"]))
            weak_values.append(float(result["metrics"]["final_weak_loss"]))
            unseen_values.append(float(result["metrics"]["unseen_rollout_relative_l2"]))
            strong_values.append(float(result["metrics"]["final_strong_loss"]))
            total_values.append(float(result["metrics"]["final_loss"]))

        err_d_mean, err_d_std = summarize(err_d_values)
        err_r_mean, err_r_std = summarize(err_r_values)
        weak_mean, weak_std = summarize(weak_values)
        unseen_mean, unseen_std = summarize(unseen_values)
        strong_mean, strong_std = summarize(strong_values)
        total_mean, total_std = summarize(total_values)

        row = {
            "case": args.case,
            "objective_name": training_config.objective_name,
            "objective_summary": summarize_training_objective(training_config),
            "num_seeds": args.num_seeds,
            "dataset_seed": args.dataset_seed,
            "base_seed": args.base_seed,
            "num_trajectories": args.num_trajectories,
            "nx": args.nx,
            "dt": args.dt,
            "t_final": args.t_final,
            "save_every": args.save_every,
            "epochs": args.epochs,
            "ErrD_mean": err_d_mean,
            "ErrD_std": err_d_std,
            "ErrR_mean": err_r_mean,
            "ErrR_std": err_r_std,
            "weak_loss_mean": weak_mean,
            "weak_loss_std": weak_std,
            "unseen_mean": unseen_mean,
            "unseen_std": unseen_std,
            "strong_loss_mean": strong_mean,
            "strong_loss_std": strong_std,
            "total_loss_mean": total_mean,
            "total_loss_std": total_std,
        }
        rows.append(row)
        print(
            f"    -> ErrD={err_d_mean:.3e} +/- {err_d_std:.3e}, "
            f"ErrR={err_r_mean:.3e} +/- {err_r_std:.3e}, "
            f"weak={weak_mean:.3e} +/- {weak_std:.3e}, "
            f"unseen={unseen_mean:.3e} +/- {unseen_std:.3e}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.output_md, rows)
    print(f"csv_saved={args.output_csv}")
    print(f"markdown_saved={args.output_md}")


if __name__ == "__main__":
    main()
