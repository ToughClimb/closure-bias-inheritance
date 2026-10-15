"""Vector plots of saved common-cohort forward errors and their denominators.

The model selection is fixed to default/drop_strong, both declared epochs and
both native/frozen representations. Polynomial/global references use separate
panels. All four declared ICs and both error metrics are always retained.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
REPORTS = REPO / "outputs" / STUDY.name / "reports"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth(value):
    return value in ("True", "true", "1", True)


def selector(row, method, variant, epoch):
    return row["method"] == method and row["variant"] == (variant or "") and row["epoch"] == (str(epoch) if epoch else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    source = REPORTS / "forward_absolute_by_seed.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    unit_ids = {row["unit_id"] for row in rows}
    assert len(rows) == 12 * len(unit_ids)
    assert len({(row["unit_id"], row["pattern"], row["horizon"]) for row in rows}) == len(rows)
    if args.require_complete and (len(unit_ids) != 500 or len(rows) != 6000):
        raise RuntimeError("Final forward plots require all500 states/6000 horizon rows")
    protocol = json.loads((STUDY / "PROTOCOL.json").read_text())
    output = REPORTS / "figures"
    output.mkdir(exist_ok=True)
    palette = {("default", "native"): "#ba3b2e", ("default", "frozen"): "#e28358",
               ("drop_strong", "native"): "#1d5ba4", ("drop_strong", "frozen"): "#58a4ce"}
    models = [dict(method=method, variant=variant, epoch=epoch,
        label=f"{'Default' if variant == 'default' else 'Drop strong'} {method}, {epoch}",
        color=palette[(variant, method)], linestyle=(":" if epoch == 150 else "-"), marker=("o" if epoch == 150 else "s"))
        for epoch in protocol["training_checkpoints"] for variant in ("default", "drop_strong") for method in ("native", "frozen")]
    references = [dict(method=method, variant=variant, epoch=None, label=label, color=color, linestyle=style, marker="o")
        for method,variant,label,color,style in [
            ("strong_poly", None, "Direct strong polynomial", "#9059ad", "-"),
            ("weak_poly", None, "Direct space-weak polynomial", "#2e9561", "-"),
            ("spacetime_poly", None, "Direct space-time-weak polynomial", "#173e2d", "-"),
            ("global_positive_poly", "default", "Global positive-D default", "#ba3b2e", "--"),
            ("global_positive_poly", "drop_strong", "Global positive-D drop strong", "#1d5ba4", "--")]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    pages, group_rows = [], []
    pdf_path = output / "forward_timegrowth.pdf"
    with PdfPages(pdf_path) as pdf:
        for metric in ("endpoint", "space_time"):
            rel_key = "endpoint_error" if metric == "endpoint" else "space_time_error"
            abs_key = "endpoint_absolute_rms" if metric == "endpoint" else "space_time_absolute_rms"
            for case in protocol["cases"]:
                for setting in protocol["settings"]:
                    figure, axes = plt.subplots(4, 4, figsize=(16.8, 11.9), squeeze=False)
                    subset = [row for row in rows if row["case"] == case and row["setting"] == setting]
                    for pattern_index, pattern in enumerate(protocol["initial_conditions"]):
                        pattern_rows = [row for row in subset if row["pattern"] == pattern["id"]]
                        for column, (collection, value_key, heading) in enumerate([
                            (models, rel_key, "Neural / relative"), (models, abs_key, "Neural / absolute RMS"),
                            (references, rel_key, "Polynomial references / relative"), (references, abs_key, "Polynomial references / absolute RMS")]):
                            axis = axes[pattern_index, column]
                            counts, visible_values = [], []
                            for specification in collection:
                                values, sds, ns, valids = [], [], [], []
                                for horizon in protocol["horizons"]:
                                    group = [row for row in pattern_rows if float(row["horizon"]) == horizon and selector(row, specification["method"], specification["variant"], specification["epoch"])]
                                    assert len({row["data_seed"] for row in group}) == len(group)
                                    completed = [row for row in group if truth(row["reached"])]
                                    errors = [float(row[value_key]) for row in completed]
                                    mean = float(np.mean(errors)) if errors else np.nan
                                    sd = float(np.std(errors, ddof=1)) if len(errors) > 1 else np.nan
                                    values.append(mean); sds.append(sd); ns.append(len(errors))
                                    valids.append(sum(truth(row["accepted_support_valid"]) for row in group))
                                    visible_values.extend(errors)
                                    group_rows.append(dict(case=case, setting=setting, metric=metric, value_key=value_key,
                                        pattern=pattern["id"], method=specification["method"], variant=specification["variant"],
                                        epoch=specification["epoch"], horizon=horizon, planned_seeds=5, evaluated_seeds=len(group),
                                        completed_seeds=len(errors), accepted_support_valid_seeds=valids[-1],
                                        successful_subset_mean=mean if errors else None,
                                        successful_subset_sd=sd if len(errors) > 1 else None))
                                values, sds = np.asarray(values), np.asarray(sds)
                                color = specification["color"]
                                axis.plot(protocol["horizons"], values, color=color, linestyle=specification["linestyle"],
                                    marker=specification["marker"], markersize=3.5, linewidth=1.25)
                                # An SD envelope is shown only when at least two
                                # independent seeds complete that horizon.
                                axis.fill_between(protocol["horizons"], values - sds, values + sds,
                                    where=np.isfinite(sds), color=color, alpha=.10, linewidth=0)
                                if ns != [5, 5, 5] or valids != ns:
                                    counts.append(specification["label"] + ": n=" + "/".join(map(str, ns)) + "; U=" + "/".join(map(str, valids)))
                            if visible_values and min(visible_values) > 0:
                                axis.set_yscale("log")
                            axis.grid(True, which="major", alpha=.20, linewidth=.5)
                            axis.set_xticks(protocol["horizons"])
                            axis.set_xlim(.08, .62)
                            if pattern_index == 0:
                                axis.set_title(heading)
                            if pattern_index == 3:
                                axis.set_xlabel("Horizon T")
                            if column == 0:
                                axis.set_ylabel(f"{pattern['id']} IC\n{'Endpoint' if metric == 'endpoint' else 'Space-time'} error")
                            if counts:
                                axis.text(.03, .02, "\n".join(counts), transform=axis.transAxes, fontsize=5.5,
                                    va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
                    title = f"Scratch-GPU training cohort | {case} / {setting}: {'endpoint field' if metric == 'endpoint' else 'space-time field'} error growth"
                    figure.suptitle(title + ("" if len(unit_ids) == 500 else f" (partial snapshot: {len(unit_ids)}/500 states)"), y=.985, fontsize=14)
                    neural_handles = [Line2D([0], [0], color=row["color"], linestyle=row["linestyle"], marker=row["marker"], markersize=3.5, label=row["label"]) for row in models]
                    reference_handles = [Line2D([0], [0], color=row["color"], linestyle=row["linestyle"], marker=row["marker"], markersize=3.5, label=row["label"]) for row in references]
                    figure.legend(handles=neural_handles, loc="lower left", bbox_to_anchor=(.05, .025), ncol=4, frameon=False, fontsize=7)
                    figure.legend(handles=reference_handles, loc="lower right", bbox_to_anchor=(.97, .025), ncol=2, frameon=False, fontsize=7)
                    figure.text(.05, .005, "Mean with ±1 sample SD across five data seeds where reached; missing/guarded horizons have no score. n = completed seeds at T=.1/.3/.6; U = accepted-support-valid seeds.\n"
                        "Axes use log scale when errors are positive. SD bands crossing zero are truncated by the axis. Native/frozen and both epochs are paired reuse; references are the same 60 direct fits and 40 imported global fits.", fontsize=6.5, va="bottom")
                    figure.subplots_adjust(left=.06, right=.98, top=.94, bottom=.12, hspace=.29, wspace=.28)
                    page_name = f"{metric}_{case}_{setting}"
                    svg_path = output / (page_name + ".svg")
                    figure.savefig(svg_path, format="svg")
                    pdf.savefig(figure)
                    plt.close(figure)
                    pages.append(dict(name=page_name, svg=svg_path.name, sha256=digest(svg_path)))
    # This plot table retains the exact per-horizon denominators printed on
    # panels; the full all-method CSV remains the primary tabular result.
    with (output / "forward_plot_groups.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(group_rows[0]))
        writer.writeheader(); writer.writerows(group_rows)
    manifest = dict(cohort="fresh_GPU_only", snapshot_utc=datetime.now(timezone.utc).isoformat(), complete=len(unit_ids) == 500,
        evaluated_states=len(unit_ids), planned_states=500, source_csv=str(source), source_csv_sha256=digest(source),
        plot_script_sha256=digest(__file__), protocol_file_sha256=digest(STUDY / "PROTOCOL.json"),
        protocol_sha256=json.loads((STUDY / "MANIFEST.json").read_text())["protocol_sha256"],
        pdf=pdf_path.name, pdf_sha256=digest(pdf_path), pages=pages,
        group_csv_sha256=digest(output / "forward_plot_groups.csv"), fits_or_integrations_performed=0,
        selection_rule="All four declared ICs; default/drop_strong, both epochs and native/frozen; direct strong/space-weak/space-time-weak and global positive-D default/drop_strong in separate panels.",
        inference_rule="Mean/SD across data seeds in reached-horizon subsets; no independent-replicate claim for ICs/epochs/representations. Temporal-refinement evidence is separately limited to the declared seed-zero subset.")
    (output / "PLOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key:manifest[key] for key in ("complete", "evaluated_states", "planned_states", "pdf")}, indent=2))


if __name__ == "__main__":
    main()
