from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np


CASE_ORDER = ["case_a", "case_b", "case_exp"]
CASE_LABELS = {"case_a": "Case A", "case_b": "Case B", "case_exp": "Case Exp"}
METRICS = [("ErrD", "Diffusion closure error"), ("ErrR", "Reaction closure error")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a multi-case panel for the Stage-1 noise sweep.")
    parser.add_argument(
        "--input-csv",
        action="append",
        required=True,
        help="Noise sweep CSV paths. Pass one per case.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_by_noise(rows: list[dict[str, str]], metric_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        grouped[float(row["noise_percent"])].append(float(row[metric_key]))
    noise_levels = np.asarray(sorted(grouped), dtype=np.float64)
    means = np.asarray([mean(grouped[level]) for level in noise_levels], dtype=np.float64)
    stds = np.asarray([pstdev(grouped[level]) if len(grouped[level]) > 1 else 0.0 for level in noise_levels], dtype=np.float64)
    return noise_levels, means, stds


def main() -> None:
    args = parse_args()
    rows_by_case: dict[str, list[dict[str, str]]] = {}
    for path_text in args.input_csv:
        path = Path(path_text)
        rows = load_rows(path)
        if not rows:
            continue
        case_name = rows[0]["case"]
        rows_by_case[case_name] = rows

    available_cases = [case_name for case_name in CASE_ORDER if case_name in rows_by_case]
    if not available_cases:
        raise ValueError("no usable case data provided")

    fig, axes = plt.subplots(2, len(available_cases), figsize=(4.2 * len(available_cases), 7.0), sharex=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, len(available_cases))

    colors = {"ErrD": "#1f77b4", "ErrR": "#d62728"}
    handles = []
    labels = []

    for col, case_name in enumerate(available_cases):
        case_rows = rows_by_case[case_name]
        for row_idx, (metric_key, title) in enumerate(METRICS):
            axis = axes[row_idx, col]
            x, means, stds = summarize_by_noise(case_rows, metric_key)
            line = axis.plot(x, means, marker="o", linewidth=2, color=colors[metric_key])[0]
            axis.fill_between(x, means - stds, means + stds, color=colors[metric_key], alpha=0.2)
            if col == 0:
                axis.set_ylabel("relative error")
            axis.set_title(f"{CASE_LABELS.get(case_name, case_name)}: {title}")
            axis.grid(axis="y", linestyle=":", alpha=0.4)
            axis.set_xticks(x)
            axis.set_xlabel("noise level (% of std(u))")
            if col == 0:
                handles.append(line)
                labels.append(metric_key)

    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
