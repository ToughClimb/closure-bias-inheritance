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
        "Each cell reports population mean +/- std across unique provided seeds within a case, observation, operator, training and search protocol. Legacy files without identities are explicitly unverified.",
        "",
        "| Case | Setting | Operators | Protocol | Search | Verified | Stage 2 | Surrogate ErrD | Surrogate ErrR | True ErrD | True ErrR | Unseen | BIR_D | BIR_R | Complexity D | Complexity R |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    SETTING_LABELS.get(str(row["setting"]), str(row["setting"])),
                    str(row["operator_mode"]), str(row["protocol_id"])[:12], str(row["search_id"])[:12], str(row["protocol_verified"]),
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


GROUP_KEYS = ('case', 'setting', 'operator_mode', 'protocol_id', 'search_id')


def group_key(row):
    return (row['case'], row['setting'], row.get('operator_mode', 'unspecified'),
            row.get('protocol_id') or 'legacy_unverified', row.get('search_id') or 'legacy_unverified')


def aggregate_comparison(pysr_rows, reference_rows=()):
    grouped = defaultdict(list)
    for row in pysr_rows: grouped[group_key(row)].append(row)
    output = []
    metric_suffixes = dict(surrogate_ErrD='pysr_surrogate_ErrD', surrogate_ErrR='pysr_surrogate_ErrR',
        true_ErrD='pysr_true_ErrD', true_ErrR='pysr_true_ErrR', unseen='pysr_unseen',
        bir_D='bir_D', bir_R='bir_R', complexity_D='pysr_complexity_D', complexity_R='pysr_complexity_R')
    for key, rows in sorted(grouped.items()):
        seeds = [int(row['seed']) for row in rows]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'Duplicate seeds in PySR comparison group {key}')
        summary = summarize_values(rows, list(metric_suffixes.values()))
        output.append(dict(zip(GROUP_KEYS, key), method='PySR', num_seeds=len(rows),
            protocol_verified=key[3]!='legacy_unverified' and key[4]!='legacy_unverified',
            **{name+'_'+suffix:summary[source+'_'+suffix] for name,source in metric_suffixes.items() for suffix in ('mean','std')}))
        matching = []
        by_seed = {int(row['seed']):row for row in rows}
        for reference in reference_rows:
            if reference['case'] != key[0] or reference['setting'] != key[1]: continue
            seed = int(reference['seed'])
            if seed not in by_seed: continue
            if (reference.get('protocol_id') or 'legacy_unverified') != key[3]: continue
            target = by_seed[seed]
            if key[3]!='legacy_unverified' and (not reference.get('surrogate_signature') or
                reference['surrogate_signature']!=target.get('surrogate_signature')):
                raise ValueError(f'Reference frozen target mismatch: {key[0]}/{key[1]}/seed {seed}')
            matching.append(reference)
        if not matching: continue
        if len({int(row['seed']) for row in matching}) != len(matching):
            raise ValueError(f'Duplicate restricted reference seeds for {key}')
        references = dict(surrogate_ErrD='symbolic_surrogate_ErrD', surrogate_ErrR='symbolic_surrogate_ErrR',
            true_ErrD='symbolic_true_ErrD', true_ErrR='symbolic_true_ErrR', unseen='symbolic_unseen',
            bir_D='bir_D', bir_R='bir_R', complexity_D='symbolic_complexity_D', complexity_R='symbolic_complexity_R')
        summary = summarize_values(matching, list(references.values()))
        output.append(dict(zip(GROUP_KEYS,key), method='restricted', num_seeds=len(matching),
            protocol_verified=key[3]!='legacy_unverified' and key[4]!='legacy_unverified',
            **{name+'_'+suffix:summary[source+'_'+suffix] for name,source in references.items() for suffix in ('mean','std')}))
    return output


def main() -> None:
    args = parse_args()
    rows = [row for path in args.pysr_csv for row in load_csv(Path(path))]
    if not rows: raise ValueError('No PySR result rows supplied')
    reference_rows = load_csv(args.reference_csv) if args.reference_csv is not None else []
    summary = aggregate_comparison(rows, reference_rows)
    write_csv(args.output_csv, summary, list(summary[0]))
    write_markdown(args.output_md, summary)
    print(f'wrote_csv={args.output_csv}')
    print(f'wrote_md={args.output_md}')


if __name__ == "__main__":
    main()
