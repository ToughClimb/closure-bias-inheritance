from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


SETTING_ORDER = ["clean", "noise_1", "noise_5", "sparse_space", "sparse_time", "sparse_both"]
SETTING_LABELS = {
    "clean": "clean",
    "noise_1": "noise 1%",
    "noise_5": "noise 5%",
    "sparse_space": "sparse x",
    "sparse_time": "sparse t",
    "sparse_both": "sparse x+t",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate PySR Stage-2 replacement results.")
    parser.add_argument(
        "--pysr-csv",
        action="append",
        required=True,
        help="Path to a per-seed PySR comparison CSV. Can be passed multiple times.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help="Optional restricted symbolic benchmark CSV for side-by-side comparison.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results") / "pysr_case_exp_summary.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results") / "pysr_case_exp_summary.md",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_values(rows: list[dict[str, str]], keys: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return summary


def setting_sort_key(setting: str) -> int:
    try:
        return SETTING_ORDER.index(setting)
    except ValueError:
        return len(SETTING_ORDER)


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.3e} +/- {std_value:.3e}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# PySR Replacement Summary",
        "",
        "Each cell reports mean +/- std across the provided seeds.",
        "",
        "| Setting | Stage 2 | Surrogate ErrD | Surrogate ErrR | True ErrD | True ErrR | Unseen | BIR_D | BIR_R | Complexity D | Complexity R |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    SETTING_LABELS.get(str(row["setting"]), str(row["setting"])),
                    str(row["method"]),
                    format_mean_std(float(row["surrogate_ErrD_mean"]), float(row["surrogate_ErrD_std"])),
                    format_mean_std(float(row["surrogate_ErrR_mean"]), float(row["surrogate_ErrR_std"])),
                    format_mean_std(float(row["true_ErrD_mean"]), float(row["true_ErrD_std"])),
                    format_mean_std(float(row["true_ErrR_mean"]), float(row["true_ErrR_std"])),
                    format_mean_std(float(row["unseen_mean"]), float(row["unseen_std"])),
                    format_mean_std(float(row["bir_D_mean"]), float(row["bir_D_std"])),
                    format_mean_std(float(row["bir_R_mean"]), float(row["bir_R_std"])),
                    format_mean_std(float(row["complexity_D_mean"]), float(row["complexity_D_std"])),
                    format_mean_std(float(row["complexity_R_mean"]), float(row["complexity_R_std"])),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    pysr_rows: list[dict[str, str]] = []
    for csv_path in args.pysr_csv:
        pysr_rows.extend(load_csv(Path(csv_path)))

    grouped_pysr: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pysr_rows:
        grouped_pysr[row["setting"]].append(row)

    summary_rows: list[dict[str, object]] = []
    for setting in sorted(grouped_pysr, key=setting_sort_key):
        rows = grouped_pysr[setting]
        summary = summarize_values(
            rows,
            [
                "pysr_surrogate_ErrD",
                "pysr_surrogate_ErrR",
                "pysr_true_ErrD",
                "pysr_true_ErrR",
                "pysr_unseen",
                "bir_D",
                "bir_R",
                "pysr_complexity_D",
                "pysr_complexity_R",
            ],
        )
        summary_rows.append(
            {
                "setting": setting,
                "method": "PySR",
                "num_seeds": len(rows),
                "surrogate_ErrD_mean": summary["pysr_surrogate_ErrD_mean"],
                "surrogate_ErrD_std": summary["pysr_surrogate_ErrD_std"],
                "surrogate_ErrR_mean": summary["pysr_surrogate_ErrR_mean"],
                "surrogate_ErrR_std": summary["pysr_surrogate_ErrR_std"],
                "true_ErrD_mean": summary["pysr_true_ErrD_mean"],
                "true_ErrD_std": summary["pysr_true_ErrD_std"],
                "true_ErrR_mean": summary["pysr_true_ErrR_mean"],
                "true_ErrR_std": summary["pysr_true_ErrR_std"],
                "unseen_mean": summary["pysr_unseen_mean"],
                "unseen_std": summary["pysr_unseen_std"],
                "bir_D_mean": summary["bir_D_mean"],
                "bir_D_std": summary["bir_D_std"],
                "bir_R_mean": summary["bir_R_mean"],
                "bir_R_std": summary["bir_R_std"],
                "complexity_D_mean": summary["pysr_complexity_D_mean"],
                "complexity_D_std": summary["pysr_complexity_D_std"],
                "complexity_R_mean": summary["pysr_complexity_R_mean"],
                "complexity_R_std": summary["pysr_complexity_R_std"],
            }
        )

    if args.reference_csv is not None:
        reference_rows = load_csv(args.reference_csv)
        grouped_reference: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reference_rows:
            if row["setting"] in grouped_pysr:
                grouped_reference[row["setting"]].append(row)

        for setting in sorted(grouped_reference, key=setting_sort_key):
            rows = grouped_reference[setting]
            summary = summarize_values(
                rows,
                [
                    "symbolic_surrogate_ErrD",
                    "symbolic_surrogate_ErrR",
                    "symbolic_true_ErrD",
                    "symbolic_true_ErrR",
                    "symbolic_unseen",
                    "bir_D",
                    "bir_R",
                    "symbolic_complexity_D",
                    "symbolic_complexity_R",
                ],
            )
            summary_rows.append(
                {
                    "setting": setting,
                    "method": "restricted",
                    "num_seeds": len(rows),
                    "surrogate_ErrD_mean": summary["symbolic_surrogate_ErrD_mean"],
                    "surrogate_ErrD_std": summary["symbolic_surrogate_ErrD_std"],
                    "surrogate_ErrR_mean": summary["symbolic_surrogate_ErrR_mean"],
                    "surrogate_ErrR_std": summary["symbolic_surrogate_ErrR_std"],
                    "true_ErrD_mean": summary["symbolic_true_ErrD_mean"],
                    "true_ErrD_std": summary["symbolic_true_ErrD_std"],
                    "true_ErrR_mean": summary["symbolic_true_ErrR_mean"],
                    "true_ErrR_std": summary["symbolic_true_ErrR_std"],
                    "unseen_mean": summary["symbolic_unseen_mean"],
                    "unseen_std": summary["symbolic_unseen_std"],
                    "bir_D_mean": summary["bir_D_mean"],
                    "bir_D_std": summary["bir_D_std"],
                    "bir_R_mean": summary["bir_R_mean"],
                    "bir_R_std": summary["bir_R_std"],
                    "complexity_D_mean": summary["symbolic_complexity_D_mean"],
                    "complexity_D_std": summary["symbolic_complexity_D_std"],
                    "complexity_R_mean": summary["symbolic_complexity_R_mean"],
                    "complexity_R_std": summary["symbolic_complexity_R_std"],
                }
            )

    summary_rows.sort(key=lambda row: (setting_sort_key(str(row["setting"])), str(row["method"])))
    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["setting", "method", "num_seeds"]
    write_csv(args.output_csv, summary_rows, fieldnames)
    write_markdown(args.output_md, summary_rows)

    print(f"wrote_csv={args.output_csv}")
    print(f"wrote_md={args.output_md}")


if __name__ == "__main__":
    main()
