"""Report every predeclared regularization condition without selecting a winner."""
from pathlib import Path
import argparse
import itertools
import json
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import hashlib
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
OUT = REPO / "outputs" / STUDY.name
REPORT = OUT / "reports"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global OUT, REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_root.resolve()
    REPORT = OUT / "reports"
    REPORT.mkdir(parents=True, exist_ok=True)
    rows, hashes, failed = [], {}, []
    for path in sorted((OUT / "conditions").glob("*.json")):
        record = json.loads(path.read_text())
        hashes[str(path.relative_to(OUT))] = sha(path)
        if record["status"] != "passed":
            failed.append(record)
            continue
        solver = record["solver"]
        ls = solver.get("least_squares", solver)
        row = {key: record[key] for key in ("id", "source_input_id", "case", "setting", "data_seed",
            "data_hash", "observation_signature", "variant", "weak_variant", "reg_factor", "constraint",
            "repeated_reference", "status", "objective", "no_reg_reduced_objective",
            "regularizer_fraction", "common_same_variant_default_reg_objective", "common_original_hybrid_objective")}
        row.update(E_D=record["evaluation"]["E_D"], E_R=record["evaluation"]["E_R"],
            minimum_diffusion=record["continuous_minimum_diffusion"]["value"],
            rank=ls["rank"], rank_deficient=ls["rank_deficient"],
            stationarity_relative=ls["stationarity_relative"],
            kkt_stationarity_relative=solver.get("kkt_stationarity_relative"),
            kkt_complementarity_relative=solver.get("kkt_complementarity_relative"),
            primal_dual_gap_relative=solver.get("primal_dual_gap_relative"),
            used_qp=solver.get("used_optimizer", False),
            **{"loss_" + k: v for k, v in record["losses"].items()},
            **{"weighted_loss_" + k: v for k, v in record["weighted_losses"].items()},
            **{"coefficient_" + str(k): v for k, v in enumerate(record["coefficients"])})
        rows.append(row)
    if failed or len(rows) != 320:
        (REPORT / "FAILURE_REPORT.json").write_text(json.dumps(dict(passed=len(rows), failed=failed), indent=2))
        raise RuntimeError("Incomplete conditions preserved; refusing an unqualified complete figure")
    data = pd.DataFrame(rows)
    keys = ["case", "setting", "variant", "constraint", "reg_factor"]
    assert not data.duplicated(keys + ["data_seed"]).any()
    grouped = data.groupby(keys, sort=True)
    expected_groups = set(itertools.product(
        ("case_a", "case_exp"), ("clean", "noise3"),
        ("reduced_space_weak", "space_time_weak"),
        ("unconstrained", "positive_diffusion"), (0., .1, 1., 10.)))
    assert set(grouped.groups) == expected_groups
    assert all(sorted(group.data_seed) == list(range(5)) for _, group in grouped)
    data.to_csv(REPORT / "per_seed.csv", index=False)
    metrics = ["E_D", "E_R", "objective", "no_reg_reduced_objective", "regularizer_fraction",
        "common_same_variant_default_reg_objective", "common_original_hybrid_objective", "minimum_diffusion",
        *["loss_" + k for k in ("weak", "rollout", "mass", "strong", "anchor", "reg")],
        *["weighted_loss_" + k for k in ("weak", "rollout", "mass", "strong", "anchor", "reg")]]
    summary = grouped[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(REPORT / "summary.csv", index=False)
    pairs = []
    for _, group in data.groupby(keys[:-1], sort=True):
        reference = group[group.reg_factor == 1].set_index("data_seed")
        for row in group.to_dict("records"):
            original = reference.loc[row["data_seed"]]
            assert row["data_hash"] == original.data_hash
            for metric in ("E_D", "E_R", "loss_weak", "no_reg_reduced_objective", "regularizer_fraction"):
                pairs.append({**{key: row[key] for key in keys + ["data_seed", "data_hash"]},
                    "metric": metric, "difference": row[metric] - original[metric]})
    paired = pd.DataFrame(pairs)
    paired.to_csv(REPORT / "paired_reference_by_seed.csv", index=False)
    ps = paired.groupby(keys + ["metric"]).difference.agg(["mean", "std", "count"])
    half = student_t.ppf(.975, ps["count"] - 1) * ps["std"] / np.sqrt(ps["count"])
    ps["ci95_low"], ps["ci95_high"] = ps["mean"] - half, ps["mean"] + half
    ps.reset_index().to_csv(REPORT / "paired_reference_summary.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), layout="constrained")
    cells = [(case, setting) for case in ("case_a", "case_exp") for setting in ("clean", "noise3")]
    for i, (case, setting) in enumerate(cells):
        for j, metric in enumerate(("E_D", "E_R")):
            ax = axes[i, j]
            for variant, color, label in (("reduced_space_weak", "#0072B2", "Reduced space weak"),
                                           ("space_time_weak", "#D55E00", "Reduced space-time weak")):
                for constraint, style, marker, clabel in (("unconstrained", "-", "o", "unconstrained"),
                        ("positive_diffusion", "--", "s", "positive D")):
                    subset = data[(data.case == case) & (data.setting == setting) & (data.variant == variant)
                                  & (data.constraint == constraint)]
                    g = subset.groupby("reg_factor")[metric].agg(["mean", "std"]).sort_index()
                    ax.errorbar(g.index, g["mean"], yerr=g["std"], color=color, linestyle=style,
                        marker=marker, markersize=4, markerfacecolor="none" if constraint == "positive_diffusion" else color,
                        capsize=3, linewidth=1.2, label=label + " / " + clabel)
            ax.set_xscale("symlog", linthresh=.1)
            ax.set_xticks([0, .1, 1, 10], ["0", "0.1", "1", "10"])
            ax.set_xlabel("Regularization multiplier (reference weight = 1e-4)")
            ax.set_ylabel("Relative " + ("diffusion" if metric == "E_D" else "reaction") + " error")
            ax.set_title(("A" if case == "case_a" else "Exp") + ": " + ("clean" if setting == "clean" else "3% noise"))
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("Fixed residual matrices: changing only derivative regularization\nMean ± sample SD over five paired data seeds; coincident constraint curves overlap", fontsize=12)
    fig.savefig(REPORT / "regularization_mechanism.png", dpi=190)
    assert not any(artist.get_rasterized() for artist in fig.findobj()
                   if hasattr(artist, "get_rasterized"))
    pdf_path = REPORT / "regularization_mechanism.pdf"
    fig.savefig(pdf_path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    image_check = subprocess.run(["pdfimages", "-list", str(pdf_path)],
        text=True, capture_output=True, check=True)
    image_objects = [line for line in image_check.stdout.splitlines()
        if len(line.split()) > 1 and line.split()[0].isdigit() and line.split()[1].isdigit()]
    image_audit = dict(status="passed" if not image_objects else "failed",
        pdf_sha256=sha(pdf_path), image_object_count=len(image_objects),
        command=["pdfimages", "-list", str(pdf_path)], output=image_check.stdout,
        all_figure_artists_nonrasterized=True,
        renderer="Matplotlib vector PDF from the same figure instance as the PNG")
    (REPORT / "PDF_IMAGE_AUDIT.json").write_text(json.dumps(image_audit, indent=2) + "\n")
    if image_objects:
        raise RuntimeError("Vector PDF audit found raster image objects")

    complete = dict(status="passed", conditions=len(data), repeated_reference_conditions=int(data.repeated_reference.sum()),
        new_interventions=int((~data.repeated_reference).sum()), groups=len(summary), seeds_per_group=5,
        exact_four_factor_two_constraint_group_set=True,
        distinct_observation_inputs=int(data.source_input_id.nunique()),
        constraint_arms_are_not_independent_samples=True,
        maximum_ls_stationarity_relative=float(data.stationarity_relative.max()),
        maximum_kkt_stationarity_relative=float(data.kkt_stationarity_relative.max()),
        maximum_kkt_complementarity_relative=float(data.kkt_complementarity_relative.max()),
        maximum_primal_dual_gap_relative=float(data.primal_dual_gap_relative.max()),
        minimum_positive_diffusion=float(data[data.constraint == "positive_diffusion"].minimum_diffusion.min()),
        rank_deficient_records=int(data.rank_deficient.sum()), used_qp_records=int(data.used_qp.sum()),
        factor_one_reproduction_max=json.loads((OUT / "COMPLETION.json").read_text())["maximum_reference_reproduction_difference"],
        protocol_sha256=sha(STUDY / "PROTOCOL.json"), reporting_source_sha256=sha(Path(__file__)),
        condition_hashes=hashes, figure_sha256=sha(REPORT / "regularization_mechanism.png"),
        vector_figure_sha256=sha(pdf_path), pdf_image_object_count=0,
        pdf_image_audit_sha256=sha(REPORT / "PDF_IMAGE_AUDIT.json"),
        interpretation="All conditions retained; paired t intervals descriptive, no best-factor selection. This was an exploratory follow-up motivated by earlier completed controls.")
    (REPORT / "REPORT_CHECKS.json").write_text(json.dumps(complete, indent=2) + "\n")
    print(json.dumps({k: v for k, v in complete.items() if k != "condition_hashes"}, indent=2))
    print(summary.reset_index()[keys + ["E_D_mean", "E_R_mean", "regularizer_fraction_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
