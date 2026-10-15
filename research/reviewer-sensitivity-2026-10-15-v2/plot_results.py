"""Read-only, complete-cardinality scientific reports for the frozen V2 study.

This script never imports the fitting driver or restores an optimizer. During a
continuation it can read evaluation/diagnostic metadata from an already saved
trusted local checkpoint. It never changes the numerical implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import textwrap
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
TERMS = ("weak", "rollout", "mass", "strong", "anchor", "reg")
CASE_NAMES = {"case_a": "Case A", "case_exp": "Case Exp"}
SETTING_NAMES = {"clean": "Clean", "noise3": "3% noise"}
METHOD_NAMES = {"neural": "Neural teacher", "frozen": "Frozen symbolic"}
VARIANT_NAMES = {"default": "Default", "drop_strong": "No strong term", "drop_anchor": "No anchor term",
                 "reduced_space_weak": "Reduced space weak", "space_time_weak": "Space-time weak"}
DIRECT_NAMES = {"direct_strong_poly": "Direct strong polynomial", "direct_weak_poly": "Direct space-weak polynomial",
                "direct_spacetime_poly": "Direct space-time polynomial"}
COLORS = dict(zip(TERMS, plt.get_cmap("tab10").colors[:6]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_records(root, manifest, stage, archive=None):
    metrics, gradients, quality, inventory = [], [], [], []
    input_hashes, init_hashes = defaultdict(set), defaultdict(set)
    observed_sources = []
    input_metadata = {}
    for task in manifest["tasks"]:
        path = root / "runs" / task["id"] / "result.json"
        needed = task["evaluation_epochs"] if stage == "full" else [150]
        status, evaluations, diagnostics = "not_run", {}, {}
        completed, read_checkpoint = 0, False
        if path.exists():
            raw = path.read_bytes()
            record = json.loads(raw)
            if archive is not None:
                archive.writestr(path.relative_to(root).as_posix(), raw)
            observed_sources.append(dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest()))
            if record["classification"] != "fresh_followup" or record["config"] != task:
                raise ValueError(f"Nonformal or relabelled record: {task['id']}")
            identity = record["identity"]
            if identity["config_sha256"] != canonical(task) or identity["implementation_signature"] != manifest["implementation_signature"]:
                raise ValueError(f"Identity mismatch: {task['id']}")
            if identity["protocol_sha256"] != manifest["protocol_sha256"]:
                raise ValueError(f"Protocol mismatch: {task['id']}")
            ident = f"{task['case']}_{task['setting']}_s{task['data_seed']}"
            if ident not in input_metadata:
                input_path = root / "inputs" / f"{ident}.json"
                input_bytes = input_path.read_bytes()
                input_metadata[ident] = json.loads(input_bytes)
                if archive is not None:
                    archive.writestr(f"inputs_metadata/{ident}.json", input_bytes)
            meta = input_metadata[ident]
            if (record["data_hash"] != meta["data_hash"] or record["observation_signature"] != meta["observation_signature"]
                    or identity["observation_signature"] != meta["observation_signature"]
                    or identity["initial_model_sha256"] != record["initial_model_sha256"]
                    or identity["source_signature"] != meta["core_source_signature"]):
                raise ValueError(f"Input/source/initialization identity mismatch: {task['id']}")
            status, completed = record["status"], record["completed_epoch"]
            evaluations, diagnostics = record["epochs"], record["gradient_diagnostics"]
            # The running result record is intentionally reset by the driver on
            # continuation. Its self-contained checkpoint still owns epoch 150.
            checkpoint = path.parent / "checkpoint_150.pt"
            if stage == "prefix" and "150" not in evaluations and checkpoint.exists():
                import torch
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if saved["epoch"] != 150 or any(saved.get(k) != v for k, v in identity.items()):
                    raise ValueError(f"Authoritative prefix checkpoint mismatch: {task['id']}")
                evaluations, diagnostics = saved["evaluations"], saved["diagnostics"]
                observed_sources.append(dict(path=str(checkpoint.relative_to(root)), sha256=digest(checkpoint)))
                if archive is not None:
                    metadata = dict(epoch=150, identity=identity, evaluations=evaluations, diagnostics=diagnostics)
                    archive.writestr(f"checkpoint_metadata/{task['id']}_150.json", json.dumps(
                        metadata, allow_nan=False, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item()))
                read_checkpoint = True
            pair = (task["case"], task["setting"], task["data_seed"])
            input_hashes[pair].add(record["data_hash"])
            init_hashes[pair].add(record["initial_model_sha256"])
            common = dict(id=task["id"], case=task["case"], setting=task["setting"], seed=task["data_seed"],
                          variant=task["variant"], weak_variant=task["weak_variant"], data_hash=record["data_hash"],
                          initial_model_sha256=record["initial_model_sha256"])
            for epoch in needed:
                if str(epoch) not in evaluations:
                    continue
                evaluation = evaluations[str(epoch)]
                for method in ("neural", "frozen"):
                    prefix = "teacher" if method == "neural" else "frozen"
                    metrics.append(dict(common, epoch=epoch, method=method, D=evaluation[method]["D"], R=evaluation[method]["R"],
                                        min_D=evaluation[method]["min_D"], R0=evaluation["physical"][prefix + "_R0"],
                                        **{f"{kind}_{branch}": evaluation["compression"][branch][key]
                                           for kind, key in (("BIR", "BIR"), ("bound", "bound"), ("compression_relative", "compression_relative_neural"))
                                           for branch in ("D", "R")}))
            for epoch_string, diagnostic in diagnostics.items():
                epoch = int(epoch_string)
                if stage == "prefix" and epoch > 150:
                    continue
                high = diagnostic["high_precision_validation"]
                if not high["passed"]:
                    raise ValueError(f"Failed FP64 gradient identity: {task['id']} at {epoch}")
                quality.append(dict(common, epoch=epoch, fp64_pass=high["passed"], fp64_error=high["gradient_linearity_error"],
                                    fp64_tolerance=high["tolerance"], f32_error=diagnostic["gradient_linearity_error"],
                                    f32_relative_total=diagnostic["gradient_linearity_relative_to_total"],
                                    f32_relative_component_sum=diagnostic["gradient_linearity_relative_to_sum_component_norms"],
                                    f32_legacy_heuristic_pass=diagnostic["legacy_f32_heuristic_pass"]))
                total_weighted_loss = sum(diagnostic["weighted_losses"].values())
                for block, values in diagnostic["parameter_blocks"].items():
                    norm_sum = sum(values["weighted_gradient_norms"].values())
                    for term in TERMS:
                        gradients.append(dict(common, epoch=epoch, block=block, term=term, weight=task["weights"][term],
                                              raw_loss=diagnostic["raw_losses"][term], weighted_loss=diagnostic["weighted_losses"][term],
                                              weighted_loss_share=diagnostic["weighted_losses"][term] / total_weighted_loss if total_weighted_loss else None,
                                              raw_gradient_norm=values["raw_gradient_norms"][term],
                                              weighted_gradient_norm=values["weighted_gradient_norms"][term],
                                              weighted_gradient_norm_share=values["weighted_gradient_norms"][term] / norm_sum if norm_sum else None,
                                              total_gradient_norm=values["total_gradient_norm"], cosine_with_total=values["cosine_with_total"][term],
                                              actual_update_norm=values["actual_update_norm"],
                                              raw_gradient_dot_actual_update=values["raw_gradient_dot_actual_update"][term],
                                              weighted_gradient_dot_actual_update=values["weighted_gradient_dot_actual_update"][term],
                                              cosine_with_actual_update=values["cosine_with_actual_update"][term]))
        available = sorted(int(e) for e in evaluations if int(e) in needed)
        missing = sorted(set(needed) - set(available))
        actual_evaluations = {int(e) for e in evaluations if stage == "full" or int(e) <= 150}
        unexpected_evaluations = sorted(actual_evaluations - set(needed))
        budget = task["max_epochs"] if stage == "full" else 150
        required_diagnostics = {e for e in manifest["protocol"]["diagnostic_epochs"] if e <= budget} | set(needed)
        actual_diagnostics = {int(e) for e in diagnostics if stage == "full" or int(e) <= 150}
        missing_diagnostics = sorted(required_diagnostics - actual_diagnostics)
        unexpected_diagnostics = sorted(actual_diagnostics - required_diagnostics)
        stage_ready = not (missing or unexpected_evaluations or missing_diagnostics or unexpected_diagnostics) and (
            stage == "prefix" or (status == "complete" and completed == task["max_epochs"]))
        inventory.append(dict(id=task["id"], case=task["case"], setting=task["setting"], seed=task["data_seed"],
                              variant=task["variant"], status=status, completed_epoch=completed, stage_ready=stage_ready,
                              used_checkpoint_150=read_checkpoint, available_epochs=";".join(map(str, available)), missing_epochs=";".join(map(str, missing)),
                              unexpected_evaluation_epochs=";".join(map(str, unexpected_evaluations)),
                              diagnostic_epochs=";".join(map(str, sorted(actual_diagnostics))),
                              missing_diagnostic_epochs=";".join(map(str, missing_diagnostics)),
                              unexpected_diagnostic_epochs=";".join(map(str, unexpected_diagnostics))))
    if any(len(v) != 1 for v in input_hashes.values()) or any(len(v) != 1 for v in init_hashes.values()):
        raise ValueError("Paired comparisons contain different observations or initial states")
    return pd.DataFrame(metrics), pd.DataFrame(gradients), pd.DataFrame(quality), pd.DataFrame(inventory), observed_sources


def summarize(frame, keys, metrics):
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        if group["seed"].duplicated().any():
            raise ValueError(f"Duplicate seed: {dict(zip(keys, key))}")
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) and not np.isfinite(values).all():
                raise ValueError(f"Nonfinite statistic: {metric}")
            n = len(values)
            avg = float(np.mean(values)) if n else None
            sd = float(np.std(values, ddof=1)) if n > 1 else None
            half = float(student_t.ppf(.975, n - 1)) * sd / np.sqrt(n) if n > 1 else None
            identity = dict(zip(keys, key))
            if "metric" in identity:
                identity["statistic"] = metric
            else:
                identity["metric"] = metric
            rows.append(dict(identity, n=n, mean=avg, sample_sd=sd,
                             ci95_low=avg - half if half is not None else None,
                             ci95_high=avg + half if half is not None else None,
                             complete_five_seed_group=n == 5))
    return pd.DataFrame(rows)


def paired_contrasts(metrics):
    keys = ["case", "setting", "seed", "epoch", "method"]
    reference = metrics.loc[metrics.variant == "default", keys + ["data_hash", "initial_model_sha256", "D", "R", "BIR_D", "BIR_R"]]
    merged = metrics.merge(reference, on=keys, suffixes=("", "_default"), validate="many_to_one")
    if not (merged.data_hash == merged.data_hash_default).all() or not (merged.initial_model_sha256 == merged.initial_model_sha256_default).all():
        raise ValueError("Paired reference identities disagree")
    rows = []
    for record in merged.to_dict("records"):
        for metric in ("D", "R", "BIR_D", "BIR_R"):
            rows.append({**{k: record[k] for k in keys + ["variant", "data_hash", "initial_model_sha256"]},
                         "metric": metric, "difference": record[metric] - record[metric + "_default"]
                         if pd.notna(record[metric]) and pd.notna(record[metric + "_default"]) else None})
    return pd.DataFrame(rows)


def budget_contrasts(metrics):
    keys = ["case", "setting", "seed", "variant", "method"]
    reference = metrics.loc[metrics.epoch == 150, keys + ["D", "R", "data_hash", "initial_model_sha256"]]
    merged = metrics.loc[metrics.epoch > 150].merge(reference, on=keys, suffixes=("", "_150"), validate="many_to_one")
    if merged.empty:
        return pd.DataFrame()
    if not (merged.data_hash == merged.data_hash_150).all() or not (merged.initial_model_sha256 == merged.initial_model_sha256_150).all():
        raise ValueError("Budget contrast identities disagree")
    rows = []
    for record in merged.to_dict("records"):
        for metric in ("D", "R"):
            rows.append({**{k: record[k] for k in keys + ["epoch", "data_hash", "initial_model_sha256"]},
                         "metric": metric, "difference": record[metric] - record[metric + "_150"]})
    return pd.DataFrame(rows)


def get_variant(manifest, strong=1, anchor=1):
    wanted = dict(manifest["protocol"]["default_weights"], strong=strong, anchor=anchor)
    variants = {t["variant"] for t in manifest["tasks"] if t["weights"] == wanted and t["weak_variant"] == "space_weak"}
    if len(variants) != 1:
        raise ValueError(f"Nonunique interaction cell: {strong}, {anchor}")
    return variants.pop()


def interactions(metrics, manifest):
    lookup = {(r["case"], r["setting"], r["seed"], r["method"], r["variant"]): r
              for r in metrics.loc[metrics.epoch == 150].to_dict("records")}
    rows = []
    protocol = manifest["protocol"]
    for case in protocol["cases"]:
        for setting in protocol["settings"]:
            for seed in protocol["data_seeds"]:
                for method in ("neural", "frozen"):
                    for strong in (.1, 10.):
                        for anchor in (.1, 10.):
                            variants = [get_variant(manifest, strong, anchor), get_variant(manifest, strong, 1), get_variant(manifest, 1, anchor), "default"]
                            values = [lookup.get((case, setting, seed, method, variant)) for variant in variants]
                            if any(value is None for value in values):
                                continue
                            if len({v["data_hash"] for v in values}) != 1 or len({v["initial_model_sha256"] for v in values}) != 1:
                                raise ValueError("Interaction identities disagree")
                            for metric in ("D", "R", "BIR_D", "BIR_R"):
                                response = [v[metric] for v in values]
                                rows.append(dict(case=case, setting=setting, seed=seed, method=method, strong_factor=strong,
                                                 anchor_factor=anchor, metric=metric, difference=response[0] - response[1] - response[2] + response[3]
                                                 if all(pd.notna(v) for v in response) else None))
    return pd.DataFrame(rows)


def read_controls(path, root, manifest):
    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path, float_precision="round_trip")
    # Explicitly reject clean-field oracle rows from the observation-matched comparison.
    frame = frame.loc[(frame["mode"] == "shared") & (frame["oracle"].astype(str).str.lower() == "false")].copy()
    if len(frame) != 1360 or not frame.status.eq("passed").all():
        raise ValueError("Shared polynomial controls must contain all 1360 passed records")
    tasks = {(t["case"], t["setting"], t["data_seed"], t["variant"]): t for t in manifest["tasks"]}
    metadata, bundles = {}, {}
    for ident in frame.source_input_id.unique():
        metadata[ident] = json.loads((root / "inputs" / f"{ident}.json").read_text())
        bundles[ident] = json.loads((path.parents[1] / "shared" / "bundles" / f"{ident}.json").read_text())
    output = []
    for row in frame.to_dict("records"):
        meta = metadata[row["source_input_id"]]
        setting = meta["setting"]
        task = tasks[(row["case"], setting, row["data_seed"], row["variant"])]
        if any(float(row["weight_" + k]) != float(task["weights"][k]) for k in TERMS) or row["weak_variant"] != task["weak_variant"]:
            raise ValueError("Polynomial control objective mismatch")
        if float(row["noise_level"]) != float(manifest["protocol"]["settings"][setting]):
            raise ValueError("Polynomial control noise mismatch")
        # The bundle is the authoritative source/input link; matching an ID alone is insufficient.
        bundle = bundles[row["source_input_id"]]
        source = bundle["source_metadata"]
        if (source["data_hash"] != meta["data_hash"] or source["observation_signature"] != meta["observation_signature"]
                or bundle["job"]["input_sha256"] != meta["file_sha256"]):
            raise ValueError("Polynomial control observation identity mismatch")
        output.append(dict(case=row["case"], setting=setting, seed=row["data_seed"], variant=row["variant"],
                           method="poly_" + row["constraint"], D=row["E_D"], R=row["E_R"], min_D=row["minimum_diffusion"],
                           data_hash=meta["data_hash"], hypothesis_class="D2/R3 physical-state monomials"))
    result = pd.DataFrame(output)
    if result.duplicated(["case", "setting", "seed", "variant", "method"]).any():
        raise ValueError("Duplicate polynomial control")
    return result


def read_direct_references(forward_root, root, manifest):
    """Evaluate existing coefficients; this function never invokes a fit/solver."""
    if forward_root is None:
        return pd.DataFrame(), []
    sys.path.insert(0, str(REPO / "src"))
    from closure_discovery.data_generation.cases import CASE_BUILDERS
    from closure_discovery.evaluation.metrics import relative_l2_error
    rows, sources = [], []
    protocol = manifest["protocol"]
    for case in protocol["cases"]:
        truth = CASE_BUILDERS[case]()
        for setting in protocol["settings"]:
            for seed in protocol["data_seeds"]:
                ident = f"{case}_{setting}_s{seed}"
                metadata = json.loads((root / "inputs" / f"{ident}.json").read_text())
                grid = np.linspace(*metadata["value_range"], protocol["symbolic"]["evaluation_points"])
                for method in protocol["baselines"]:
                    path = forward_root / "runs" / f"{ident}__{method}" / "result.json"
                    raw = path.read_bytes()
                    record = json.loads(raw)
                    identity, fitted = record["identity"], record["metadata"]
                    unit = identity["unit"]
                    if (record["status"] != "evaluated" or unit["kind"] != "baseline" or unit["input_id"] != ident
                            or unit["method"] != method or unit["case"] != case or unit["setting"] != setting or unit["data_seed"] != seed
                            or identity["input_file_sha256"] != metadata["file_sha256"]
                            or identity["model_metadata_sha256"] != canonical(fitted)):
                        raise ValueError(f"Direct polynomial reference identity mismatch: {path}")
                    d, r = np.asarray(fitted["coefficients_D"]), np.asarray(fitted["coefficients_R"])
                    if d.shape != (3,) or r.shape != (4,) or not np.isfinite(np.r_[d, r]).all():
                        raise ValueError(f"Invalid existing D2/R3 coefficients: {path}")
                    pd_, pr = np.polynomial.polynomial.polyval(grid, d), np.polynomial.polynomial.polyval(grid, r)
                    source_hash = hashlib.sha256(raw).hexdigest()
                    sources.append(dict(path=str(path), sha256=source_hash))
                    rows.append(dict(case=case, setting=setting, seed=seed, method="direct_" + method,
                                     D=relative_l2_error(truth.diffusion(grid), pd_), R=relative_l2_error(truth.reaction(grid), pr),
                                     min_D=float(np.min(pd_)), continuous_min_D=fitted["continuous_polynomial_min_D"]["value"],
                                     R0=float(r[0]), training_mse=fitted["training_mse"], data_hash=metadata["data_hash"],
                                     input_file_sha256=identity["input_file_sha256"], source_result_sha256=source_hash,
                                     source_forward_numerical_failure_patterns=record["numerical_failure_patterns"],
                                     reference_scope="Existing direct D2/R3 residual fits; 60 fits reused by two reports, no new fitting",
                                     **{f"D_coefficient_{i}": float(v) for i, v in enumerate(d)},
                                     **{f"R_coefficient_{i}": float(v) for i, v in enumerate(r)}))
    result = pd.DataFrame(rows)
    if len(result) != 60 or result.duplicated(["case", "setting", "seed", "method"]).any():
        raise ValueError("Expected exactly 60 existing direct reference fits")
    return result, sources


def reference_contrasts(metrics, references, match_variant):
    keys = ["case", "setting", "seed"] + (["variant"] if match_variant else [])
    reference = references[keys + ["method", "D", "R", "data_hash"]].rename(columns={"method": "reference_method"})
    merged = metrics.merge(reference, on=keys, suffixes=("", "_reference"), validate="many_to_many")
    if not merged.data_hash.eq(merged.data_hash_reference).all():
        raise ValueError("Reference contrast uses different observations")
    rows = []
    for row in merged.to_dict("records"):
        for metric in ("D", "R"):
            rows.append({**{k: row[k] for k in ["case", "setting", "seed", "variant", "epoch", "method", "reference_method", "data_hash"]},
                         "metric": metric, "difference": row[metric] - row[metric + "_reference"]})
    return pd.DataFrame(rows)


class Figures:
    def __init__(self, destination, partial, title_suffix):
        self.destination, self.partial, self.title_suffix = destination, partial, title_suffix
        self.paths = []
        destination.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                             "savefig.dpi": 170, "pdf.fonttype": 42, "ps.fonttype": 42})

    def save(self, fig, name, title, caption):
        fig.suptitle(textwrap.fill(title + self.title_suffix, width=int(fig.get_figwidth() * 11),
                                  break_long_words=False, break_on_hyphens=False), fontsize=12)
        if self.partial:
            caption = caption.replace("five", "available (at most five)")
            caption += " Partial group sizes are retained in the CSV tables."
        fig.text(.5, .012, textwrap.fill(caption, width=int(fig.get_figwidth() * 16), break_long_words=False,
                                       break_on_hyphens=False), ha="center", va="bottom", fontsize=7)
        if self.partial:
            fig.text(.5, .5, "PARTIAL REVIEW", fontsize=42, color="crimson", alpha=.16, ha="center", rotation=22)
        fig.tight_layout(rect=(0, .055, 1, .94))
        for suffix in ("pdf", "png"):
            path = self.destination / f"{name}.{suffix}"
            fig.savefig(path)
            self.paths.append(dict(path=path.name, sha256=digest(path)))
        plt.close(fig)


def plotted_statistics(group, field):
    result = group.groupby("factor")[field].agg(["mean", "std", "count"]).sort_index()
    return result.index.to_numpy(), result["mean"].to_numpy(), result["std"].to_numpy()


def legend_if_present(ax, **kwargs):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kwargs)


def vector_heatmap(ax, values, cmap, **kwargs):
    values = np.ma.masked_invalid(np.asarray(values, dtype=float))
    rows, columns = values.shape
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("white")
    mesh = ax.pcolormesh(np.arange(columns + 1) - .5, np.arange(rows + 1) - .5, values,
                         cmap=palette, shading="flat", rasterized=False, **kwargs)
    ax.set_xlim(-.5, columns - .5)
    ax.set_ylim(rows - .5, -.5)
    ax.set_aspect("auto")
    return mesh


def vector_colorbar(fig, mesh, ax, shrink):
    colorbar = fig.colorbar(mesh, ax=ax, shrink=shrink)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    return colorbar


def oat_rows(frame, manifest, term):
    variants = {"default": 1.}
    for task in manifest["tasks"]:
        if task["weak_variant"] == "space_weak" and set(task["factors"]) == {term}:
            variants[task["variant"]] = task["factors"][term]
    result = frame.loc[frame.variant.isin(variants)].copy()
    result["factor"] = result.variant.map(variants)
    return result


def format_oat_axis(ax, rows, branch, term, paired=False):
    factors = sorted(rows.factor.unique())
    positive = [v for v in factors if v > 0]
    ax.set_xscale("symlog", linthresh=min(positive) / 2 if positive else .05)
    ax.set_xticks(factors, labels=[f"{v:g}" for v in factors], rotation=45 if len(factors) > 5 else 0)
    ax.set_title(term)
    ax.set_xlabel("Multiplier of declared default")
    ax.set_ylabel(("Paired change in " if paired else "") + branch)
    ax.grid(alpha=.18)
    if paired:
        ax.axhline(0, color=".5", lw=.7)


def plot_oat(metrics, paired, controls, manifest, figures):
    at150 = metrics.loc[metrics.epoch == 150]
    for case in manifest["protocol"]["cases"]:
        for method in ("neural", "frozen"):
            fig, axs = plt.subplots(2, 6, figsize=(19, 6.4), squeeze=False)
            figp, axsp = plt.subplots(2, 6, figsize=(19, 6.4), squeeze=False)
            for col, term in enumerate(TERMS):
                subset = oat_rows(at150.loc[(at150.case == case) & (at150.method == method)], manifest, term)
                subsetp = oat_rows(paired.loc[(paired.case == case) & (paired.method == method) & (paired.epoch == 150)], manifest, term)
                for row, branch in enumerate(("D", "R")):
                    for setting, color in (("clean", "#1764a0"), ("noise3", "#c14c19")):
                        group = subset.loc[subset.setting == setting]
                        if not group.empty:
                            x, y, sd = plotted_statistics(group, branch)
                            axs[row, col].errorbar(x, y, yerr=sd, fmt="o-", ms=3, lw=1, capsize=2, color=color, label=SETTING_NAMES[setting])
                        gp = subsetp.loc[(subsetp.setting == setting) & (subsetp.metric == branch)]
                        if not gp.empty:
                            x, y, sd = plotted_statistics(gp, "difference")
                            n = gp.groupby("factor").difference.count().sort_index().to_numpy()
                            ci = student_t.ppf(.975, n - 1) * sd / np.sqrt(n)
                            axsp[row, col].errorbar(x, y, yerr=ci, fmt="o-", ms=3, lw=1, capsize=2, color=color, label=SETTING_NAMES[setting])
                    format_oat_axis(axs[row, col], subset, branch, term)
                    format_oat_axis(axsp[row, col], subsetp, branch, term, paired=True)
            legend_if_present(axs[0, 0], fontsize=8)
            legend_if_present(axsp[0, 0], fontsize=8)
            figures.save(fig, f"oat_errors_{case}_{method}", f"{CASE_NAMES[case]}: {METHOD_NAMES[method]} at 150 epochs",
                         "Every declared OAT point. Mean ± sample SD across five paired seeds; symmetric SD bars may cross zero. Panels use independent vertical ranges.")
            figures.save(figp, f"oat_paired_{case}_{method}", f"{CASE_NAMES[case]}: paired change from default, {METHOD_NAMES[method]}",
                         "Seed-matched error difference (variant − default), mean ± descriptive Student-t 95% interval; no multiple-comparison significance claim.")
        fig, axs = plt.subplots(2, 6, figsize=(19, 6.4), squeeze=False)
        for col, term in enumerate(TERMS):
            subset = oat_rows(at150.loc[(at150.case == case) & (at150.method == "neural")], manifest, term)
            for row, branch in enumerate(("D", "R")):
                for setting, color in (("clean", "#1764a0"), ("noise3", "#c14c19")):
                    group = subset.loc[subset.setting == setting]
                    if not group.empty:
                        x, y, sd = plotted_statistics(group, "BIR_" + branch)
                        axs[row, col].errorbar(x, y, yerr=sd, fmt="o-", ms=3, lw=1, capsize=2, color=color, label=SETTING_NAMES[setting])
                format_oat_axis(axs[row, col], subset, "BIR " + branch, term)
                axs[row, col].axhline(1, color=".5", lw=.7, ls="--")
        legend_if_present(axs[0, 0], fontsize=8)
        figures.save(fig, f"oat_bir_{case}", f"{CASE_NAMES[case]}: downstream bias inheritance at 150 epochs",
                     "BIR = symbolic error / teacher error in the same norm; BIR near one does not imply an accurate teacher. Mean ± sample SD, five seeds; undefined denominators remain missing.")
        if controls.empty:
            continue
        for setting in manifest["protocol"]["settings"]:
            fig, axs = plt.subplots(2, 6, figsize=(19, 6.4), squeeze=False)
            selected = at150.loc[(at150.case == case) & (at150.setting == setting) & (at150.method == "neural")]
            poly = controls.loc[(controls.case == case) & (controls.setting == setting)]
            for col, term in enumerate(TERMS):
                subset = oat_rows(pd.concat([selected, poly], ignore_index=True), manifest, term)
                for row, branch in enumerate(("D", "R")):
                    for method, label, color in (("neural", "Neural, 150 epochs", "#1764a0"),
                                                 ("poly_unconstrained", "Global D2/R3, unconstrained", "#b54e12"),
                                                 ("poly_positive_diffusion", "Global D2/R3, positive D", "#318348")):
                        group = subset.loc[subset.method == method]
                        if not group.empty:
                            x, y, sd = plotted_statistics(group, branch)
                            axs[row, col].errorbar(x, y, yerr=sd, fmt="o-", ms=3, lw=1, capsize=2, color=color, label=label)
                    format_oat_axis(axs[row, col], subset, branch, term)
            legend_if_present(axs[0, 0], fontsize=7)
            figures.save(fig, f"oat_objective_controls_{case}_{setting}", f"{CASE_NAMES[case]}, {SETTING_NAMES[setting]}: observation-matched objective controls",
                         "Same six-component objective and saved observations; polynomial D2/R3 is a different hypothesis class. Constraints remain separate. Mean ± sample SD, five seeds.")


def plot_budget(metrics, controls, direct, manifest, figures):
    variants = manifest["protocol"]["longer_variants"]
    cells = [(c, s) for c in manifest["protocol"]["cases"] for s in manifest["protocol"]["settings"]]
    for method in ("neural", "frozen"):
        fig, axs = plt.subplots(4, 2, figsize=(12, 12), squeeze=False)
        for row, (case, setting) in enumerate(cells):
            for col, branch in enumerate(("D", "R")):
                ax = axs[row, col]
                for index, variant in enumerate(variants):
                    subset = metrics.loc[(metrics.case == case) & (metrics.setting == setting) & (metrics.method == method) & (metrics.variant == variant)]
                    result = subset.groupby("epoch")[branch].agg(["mean", "std"]).sort_index()
                    if not result.empty:
                        ax.errorbar(result.index, result["mean"], yerr=result["std"], fmt="o-", ms=3, capsize=2,
                                    color=plt.get_cmap("tab10")(index), label=VARIANT_NAMES[variant])
                for index, reference in enumerate(DIRECT_NAMES):
                    if direct.empty:
                        continue
                    values = direct.loc[(direct.case == case) & (direct.setting == setting) & (direct.method == reference), branch]
                    if len(values):
                        ax.axhline(values.mean(), color=plt.get_cmap("tab10")(index + 5), ls=":", lw=1, label=DIRECT_NAMES[reference])
                ax.set_title(f"{CASE_NAMES[case]}, {SETTING_NAMES[setting]} — {branch}")
                ax.set_xticks([150, 500, 1500]); ax.set_xlabel("Declared Adam epochs"); ax.set_ylabel("Relative error"); ax.grid(alpha=.18)
        legend_if_present(axs[0, 0], fontsize=6, ncol=2)
        figures.save(fig, f"budget_errors_{method}", f"Weight/objective × fixed budget: {METHOD_NAMES[method]}",
                     "Five predeclared continuation arms: mean ± sample SD, five paired seeds. Dotted references reuse three existing direct polynomial fits per input; their SD is retained in separate tables/figure. Fixed budgets do not certify convergence.")
    if controls.empty:
        return
    for case, setting in cells:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
        for col, branch in enumerate(("D", "R")):
            ax = axs[col]
            for offset, (epoch, method, label, color, marker) in enumerate((
                (150, "neural", "Neural 150", "#1764a0", "o"), (500, "neural", "Neural 500", "#5c80b0", "s"),
                (1500, "neural", "Neural 1500", "#15304d", "D"),
                (None, "poly_unconstrained", "Global D2/R3, unconstrained", "#b54e12", "^"),
                (None, "poly_positive_diffusion", "Global D2/R3, positive D", "#318348", "v"))):
                source = metrics if epoch is not None else controls
                subset = source.loc[(source.case == case) & (source.setting == setting) & (source.method == method)]
                if epoch is not None:
                    subset = subset.loc[subset.epoch == epoch]
                summary = subset.groupby("variant")[branch].agg(["mean", "std"]).reindex(variants)
                ax.errorbar(np.arange(len(variants)) + (offset - 2) * .13, summary["mean"], yerr=summary["std"], fmt=marker,
                            ms=4, capsize=2, color=color, label=label)
            ax.set_xticks(range(len(variants)), labels=[VARIANT_NAMES[v] for v in variants], rotation=25, ha="right")
            ax.set_ylabel(branch + " relative error"); ax.grid(axis="y", alpha=.18)
        legend_if_present(axs[0], fontsize=7)
        figures.save(fig, f"budget_objective_controls_{case}_{setting}", f"{CASE_NAMES[case]}, {SETTING_NAMES[setting]}: budget and global objective controls",
                     "Mean ± sample SD, five seeds. Polynomial controls solve a different seven-coefficient hypothesis class; their two constraint arms remain distinct.")


def plot_direct_references(metrics, direct, manifest, figures):
    if direct.empty:
        return
    cells = [(c, s) for c in manifest["protocol"]["cases"] for s in manifest["protocol"]["settings"]]
    epochs = [epoch for epoch in (150, 500, 1500) if metrics.epoch.eq(epoch).any()]
    labels = list(DIRECT_NAMES.values()) + [f"Default neural, {epoch} epochs" for epoch in epochs]
    fig, axs = plt.subplots(4, 2, figsize=(14, 12), squeeze=False)
    for row, (case, setting) in enumerate(cells):
        for col, branch in enumerate(("D", "R")):
            ax = axs[row, col]
            for i, reference in enumerate(DIRECT_NAMES):
                values = direct.loc[(direct.case == case) & (direct.setting == setting) & (direct.method == reference), branch]
                ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt="o", capsize=3, color=plt.get_cmap("tab10")(i + 5))
            for offset, epoch in enumerate(epochs):
                values = metrics.loc[(metrics.case == case) & (metrics.setting == setting) & (metrics.method == "neural")
                                     & (metrics.variant == "default") & (metrics.epoch == epoch), branch]
                if len(values):
                    ax.errorbar(offset + 3, values.mean(), yerr=values.std(ddof=1), fmt="s", capsize=3, color="#1764a0")
            ax.set_xticks(range(len(labels)), labels=labels, rotation=22, ha="right", fontsize=7)
            ax.set_ylabel(branch + " relative error"); ax.grid(axis="y", alpha=.18)
            ax.set_title(f"{CASE_NAMES[case]}, {SETTING_NAMES[setting]}")
    figures.save(fig, "direct_polynomial_references", "Existing direct polynomial residual references and fixed default neural checkpoints",
                 "Mean ± sample SD, five paired seeds, common 256-point norm. Existing 60 strong/space-weak/space-time polynomial fits are reused from forward-v2, without refitting or adding independent samples; these objectives differ from the neural hybrid objective.")


def plot_interactions(metrics, differences, manifest, figures):
    cells = [(c, s) for c in manifest["protocol"]["cases"] for s in manifest["protocol"]["settings"]]
    for method in ("neural", "frozen"):
        for contrast in (False, True):
            factors = [.1, 10.] if contrast else [.1, 1., 10.]
            fig, axs = plt.subplots(4, 2, figsize=(9, 12), squeeze=False)
            for row, (case, setting) in enumerate(cells):
                for col, branch in enumerate(("D", "R")):
                    ax = axs[row, col]
                    avg, sd = np.full((len(factors), len(factors)), np.nan), np.full((len(factors), len(factors)), np.nan)
                    for i, a in enumerate(factors):
                        for j, b in enumerate(factors):
                            if contrast:
                                selected = differences.loc[(differences.case == case) & (differences.setting == setting) & (differences.method == method)
                                                           & (differences.metric == branch) & (differences.strong_factor == a) & (differences.anchor_factor == b), "difference"]
                            else:
                                variant = get_variant(manifest, a, b)
                                selected = metrics.loc[(metrics.case == case) & (metrics.setting == setting) & (metrics.method == method)
                                                       & (metrics.epoch == 150) & (metrics.variant == variant), branch]
                            if len(selected):
                                avg[i, j], sd[i, j] = selected.mean(), selected.std(ddof=1)
                    limit = np.nanmax(np.abs(avg)) if np.isfinite(avg).any() else 1
                    image = vector_heatmap(ax, avg, cmap="coolwarm" if contrast else "viridis",
                                           norm=TwoSlopeNorm(vmin=-max(limit, 1e-15), vcenter=0, vmax=max(limit, 1e-15)) if contrast else None)
                    for i in range(len(factors)):
                        for j in range(len(factors)):
                            if np.isfinite(avg[i, j]):
                                ax.text(j, i, f"{avg[i,j]:.3g}\n±{sd[i,j]:.2g}", ha="center", va="center", fontsize=8,
                                        color="white" if not contrast and avg[i, j] < (np.nanmin(avg) + np.nanmax(avg)) / 2 else "black")
                    ax.set_xticks(range(len(factors)), labels=factors); ax.set_yticks(range(len(factors)), labels=factors)
                    ax.set_xlabel("Anchor multiplier"); ax.set_ylabel("Strong multiplier")
                    ax.set_title(f"{CASE_NAMES[case]}, {SETTING_NAMES[setting]} — {branch}"); vector_colorbar(fig, image, ax=ax, shrink=.7)
            figures.save(fig, f"strong_anchor_{'second_difference' if contrast else 'grid'}_{method}",
                         f"Strong × anchor at 150 epochs: {METHOD_NAMES[method]}" + (", paired second differences" if contrast else ""),
                         "Cell annotation: mean ± sample SD across five seeds. " +
                         ("Second difference y(a,b) − y(a,1) − y(1,b) + y(1,1); response-specific and local to this grid." if contrast else "Every cell in the predeclared 3×3 grid; panels use separate color scales."))


def plot_gradients(gradients, manifest, figures):
    for case in manifest["protocol"]["cases"]:
        for setting in manifest["protocol"]["settings"]:
            subset = gradients.loc[(gradients.case == case) & (gradients.setting == setting) & (gradients.variant == "default")]
            fig, axs = plt.subplots(4, 1, figsize=(10, 10), squeeze=False)
            for row, block in enumerate(("all", "all", "D", "R")):
                field = "weighted_loss_share" if row == 0 else "weighted_gradient_norm_share"
                ax = axs[row, 0]
                for term in TERMS:
                    group = subset.loc[(subset.block == block) & (subset.term == term)].groupby("epoch")[field].agg(["mean", "std"]).sort_index()
                    if not group.empty:
                        ax.errorbar(group.index, group["mean"], yerr=group["std"], fmt="o-", ms=3, capsize=2, color=COLORS[term], label=term)
                ax.set_xscale("symlog", linthresh=1)
                ax.set_yscale("symlog", linthresh=1e-8)
                ax.set_xlabel("Diagnostic epoch"); ax.set_ylabel("Loss share" if row == 0 else f"Gradient-norm share ({block})")
                ax.grid(alpha=.18)
            legend_if_present(axs[0, 0], ncol=6, fontsize=8)
            figures.save(fig, f"gradient_component_scales_{case}_{setting}", f"{CASE_NAMES[case]}, {SETTING_NAMES[setting]}: default component scales",
                         "Per-seed weighted loss / sum losses and weighted gradient norm / sum component norms, then mean ± sample SD. Norm shares are not additive contributions to the total-gradient norm.")
    for block in ("all", "D", "R"):
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), squeeze=False)
        for row, (case, setting) in enumerate((c, s) for c in manifest["protocol"]["cases"] for s in manifest["protocol"]["settings"]):
            subset = gradients.loc[(gradients.case == case) & (gradients.setting == setting) & (gradients.variant == "default") & (gradients.block == block) & (gradients.epoch > 0)]
            pivot = subset.groupby(["term", "epoch"]).cosine_with_actual_update.mean().unstack().reindex(TERMS)
            ax = axs[row, 0]
            if not pivot.empty:
                im = vector_heatmap(ax, pivot.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
                ax.set_yticks(range(6), labels=TERMS)
                for i in range(6):
                    for j in range(len(pivot.columns)):
                        value = pivot.iloc[i, j]
                        if pd.notna(value):
                            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
                vector_colorbar(fig, im, ax=ax, shrink=.8)
            ax.set_title(f"{CASE_NAMES[case]}, {SETTING_NAMES[setting]}"); ax.set_xlabel("Diagnostic epoch")
        figures.save(fig, f"gradient_actual_update_alignment_{block}", f"Default component alignment with actual Adam update: {block} parameters",
                     "Mean cosine across five seeds; zero-norm cases remain missing. Gradients are evaluated after the update and dotted with the most recent actual update: local direction, not exact loss change.")
    variants = manifest["protocol"]["longer_variants"]
    epochs = [epoch for epoch in (150, 1500) if gradients.epoch.eq(epoch).any()]
    cells = [(c, s) for c in manifest["protocol"]["cases"] for s in manifest["protocol"]["settings"]]
    for block in ("all", "D", "R"):
        fig, axs = plt.subplots(4, len(epochs), figsize=(6 * len(epochs), 11), squeeze=False)
        for row, (case, setting) in enumerate(cells):
            for col, epoch in enumerate(epochs):
                ax = axs[row, col]
                subset = gradients.loc[(gradients.case == case) & (gradients.setting == setting) & (gradients.block == block)
                                       & (gradients.epoch == epoch) & gradients.variant.isin(variants) & (gradients.weight > 0)]
                pivot = subset.groupby(["variant", "term"]).cosine_with_actual_update.mean().unstack().reindex(index=variants, columns=TERMS)
                im = vector_heatmap(ax, pivot.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(6), labels=TERMS)
                ax.set_yticks(range(5), labels=[VARIANT_NAMES[v] for v in variants])
                ax.set_title(f"{CASE_NAMES[case]}, {SETTING_NAMES[setting]}, epoch {epoch}")
                for i in range(5):
                    for j in range(6):
                        value = pivot.iloc[i, j]
                        if pd.notna(value):
                            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
                vector_colorbar(fig, im, ax=ax, shrink=.8)
        figures.save(fig, f"gradient_variant_update_alignment_{block}", f"Predeclared objective arms: component/update alignment in {block} parameters",
                     "Mean cosine across five seeds. Inactive zero-weight terms and undefined cosines are blank. Post-update gradients versus recent Adam updates are local direction diagnostics, not exact loss changes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/reviewer-sensitivity-2026-10-15-v2")
    parser.add_argument("--manifest", type=Path, default=STUDY / "MANIFEST.json")
    parser.add_argument("--stage", choices=("prefix", "full"), default="full")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--controls", type=Path, default=REPO / "outputs/reviewer-objective-controls-2026-10-15/reports/per_seed.csv")
    parser.add_argument("--without-controls", action="store_true")
    parser.add_argument("--direct-reference-root", type=Path, default=REPO / "outputs/reviewer-forward-2026-10-15-v2")
    parser.add_argument("--without-direct-references", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    manifest = json.loads(args.manifest.read_text())
    from make_manifest import implementation_signature
    if canonical(manifest["protocol"]) != manifest["protocol_sha256"] or implementation_signature() != manifest["implementation_signature"]:
        raise ValueError("The scientific protocol or numerical implementation changed after the frozen manifest")
    destination = args.destination or root / "reports" / ("prefix_visualization" if args.stage == "prefix" else "full_visualization")
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "REPORT_INPUT_SNAPSHOT.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("MANIFEST.json", args.manifest.read_bytes())
        archive.writestr("REPORTING_PLAN.json", (STUDY / "REPORTING_PLAN.json").read_bytes())
        archive.writestr("plot_results.py", Path(__file__).read_bytes())
        metrics, gradients, quality, inventory, sources = read_records(root, manifest, args.stage, archive)
    inventory.to_csv(destination / "task_inventory.csv", index=False)
    expected_evaluations = sum(len(t["evaluation_epochs"]) if args.stage == "full" else 1 for t in manifest["tasks"])
    expected_diagnostics = 3020 if args.stage == "full" else 2720
    complete = bool(inventory.stage_ready.all()) and len(metrics) == expected_evaluations * 2 and len(quality) == expected_diagnostics
    state = dict(classification="formal_study_report" if complete else ("explicit_partial_report" if args.allow_partial else "incomplete_inventory"), stage=args.stage,
                 report_success=False,
                 generated_utc=datetime.now(timezone.utc).isoformat(), complete=complete, expected_tasks=680,
                 stage_ready_tasks=int(inventory.stage_ready.sum()), expected_evaluation_pairs=expected_evaluations,
                 actual_evaluation_pairs=len(metrics) // 2, expected_diagnostics=expected_diagnostics, actual_diagnostics=len(quality),
                 statuses=dict(Counter(inventory.status)), implementation_signature=manifest["implementation_signature"],
                 manifest_sha256=digest(args.manifest), reporting_script_sha256=digest(__file__), reporting_plan_sha256=digest(STUDY / "REPORTING_PLAN.json"),
                 report_input_snapshot_sha256=digest(archive_path),
                 selection_policy="Every declared configuration/seed/epoch; no truth-based selection", source_records=sources,
                 uncertainty="Sample SD across five seeds; paired descriptive Student-t 95% intervals, no multiple-comparison claim",
                 requested_references=dict(same_objective=not args.without_controls, existing_direct_residual=not args.without_direct_references),
                 reference_scope="Shared non-oracle same-objective D2/R3 controls (two constraints separate), plus 60 existing direct polynomial residual fits from forward-v2; no new fits")
    json_write(destination / "REPORT_STATUS.json", state)
    if not complete and not args.allow_partial:
        raise SystemExit(f"Report is incomplete: {state['stage_ready_tasks']}/680 ready, {len(metrics)//2}/{expected_evaluations} evaluation pairs, {len(quality)}/{expected_diagnostics} diagnostics. Inventory saved. Use --allow-partial explicitly for review only.")
    if metrics.empty:
        print(json.dumps({k: v for k, v in state.items() if k != "source_records"}, indent=2))
        return
    metrics.to_csv(destination / "metrics_by_seed.csv", index=False)
    gradients.to_csv(destination / "gradient_by_seed.csv", index=False)
    quality.to_csv(destination / "diagnostic_quality_by_seed.csv", index=False)
    metric_fields = ["D", "R", "min_D", "R0", "BIR_D", "BIR_R", "bound_D", "bound_R", "compression_relative_D", "compression_relative_R"]
    summarize(metrics, ["case", "setting", "variant", "epoch", "method"], metric_fields).to_csv(destination / "metrics_summary.csv", index=False)
    physical_rows = []
    for key, group in metrics.groupby(["case", "setting", "variant", "epoch", "method"]):
        physical_rows.append(dict(zip(["case", "setting", "variant", "epoch", "method"], key),
                                  n=len(group), sampled_nonpositive_diffusion=int(group.min_D.le(0).sum()),
                                  sampled_minimum_diffusion=float(group.min_D.min()), mean_absolute_R0=float(group.R0.abs().mean()),
                                  scope="Sampled constitutive margins; no forward-trajectory or global-sign certificate"))
    pd.DataFrame(physical_rows).to_csv(destination / "physical_margins_summary.csv", index=False)
    paired = paired_contrasts(metrics)
    paired.to_csv(destination / "paired_default_by_seed.csv", index=False)
    summarize(paired, ["case", "setting", "variant", "epoch", "method", "metric"], ["difference"]).to_csv(destination / "paired_default_summary.csv", index=False)
    budgets = budget_contrasts(metrics)
    if not budgets.empty:
        budgets.to_csv(destination / "paired_budget_by_seed.csv", index=False)
        summarize(budgets, ["case", "setting", "variant", "epoch", "method", "metric"], ["difference"]).to_csv(destination / "paired_budget_summary.csv", index=False)
    interaction = interactions(metrics, manifest)
    interaction.to_csv(destination / "interaction_by_seed.csv", index=False)
    if not interaction.empty:
        summarize(interaction, ["case", "setting", "method", "strong_factor", "anchor_factor", "metric"], ["difference"]).to_csv(destination / "interaction_summary.csv", index=False)
    gradient_fields = ["raw_loss", "weighted_loss", "weighted_loss_share", "raw_gradient_norm", "weighted_gradient_norm", "weighted_gradient_norm_share", "total_gradient_norm",
                       "cosine_with_total", "actual_update_norm", "raw_gradient_dot_actual_update", "weighted_gradient_dot_actual_update", "cosine_with_actual_update"]
    summarize(gradients, ["case", "setting", "variant", "epoch", "block", "term"], gradient_fields).to_csv(destination / "gradient_summary.csv", index=False)
    controls = read_controls(None if args.without_controls else args.controls.resolve(), root, manifest)
    if not controls.empty:
        comparison = metrics.loc[(metrics.epoch == 150) & (metrics.method == "neural")].merge(
            controls, on=["case", "setting", "seed", "variant"], suffixes=("_neural", "_control"), validate="one_to_many")
        if not comparison.data_hash_neural.eq(comparison.data_hash_control).all():
            raise ValueError("Neural and polynomial controls do not use the same observations")
        controls.to_csv(destination / "same_objective_controls_by_seed.csv", index=False)
        summarize(controls, ["case", "setting", "variant", "method"], ["D", "R", "min_D"]).to_csv(destination / "same_objective_controls_summary.csv", index=False)
        differences = reference_contrasts(metrics, controls, match_variant=True)
        differences.to_csv(destination / "same_objective_paired_by_seed.csv", index=False)
        summarize(differences, ["case", "setting", "variant", "epoch", "method", "reference_method", "metric"], ["difference"]).to_csv(destination / "same_objective_paired_summary.csv", index=False)
        state["polynomial_control_source_sha256"] = digest(args.controls)
        state["polynomial_control_rows"] = len(controls)
        with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("controls/per_seed.csv", args.controls.read_bytes())
            for ident in sorted({f"{row.case}_{row.setting}_s{row.seed}" for row in controls.itertuples()}):
                bundle_path = args.controls.resolve().parents[1] / "shared" / "bundles" / f"{ident}.json"
                archive.writestr(f"controls/bundles/{ident}.json", bundle_path.read_bytes())
        state["report_input_snapshot_sha256"] = digest(archive_path)
    direct, direct_sources = read_direct_references(None if args.without_direct_references else args.direct_reference_root.resolve(), root, manifest)
    if not direct.empty:
        direct.to_csv(destination / "direct_references_by_seed.csv", index=False)
        summarize(direct, ["case", "setting", "method"], ["D", "R", "min_D", "R0", "training_mse"]).to_csv(destination / "direct_references_summary.csv", index=False)
        differences = reference_contrasts(metrics, direct, match_variant=False)
        differences.to_csv(destination / "direct_references_paired_by_seed.csv", index=False)
        summarize(differences, ["case", "setting", "variant", "epoch", "method", "reference_method", "metric"], ["difference"]).to_csv(destination / "direct_references_paired_summary.csv", index=False)
        state["direct_reference_sources"] = direct_sources
        state["direct_reference_existing_fit_count"] = 60
        state["new_direct_reference_fits_performed"] = 0
        with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in direct_sources:
                path = Path(source["path"])
                archive.writestr(f"direct_references/{path.parent.name}.json", path.read_bytes())
        state["report_input_snapshot_sha256"] = digest(archive_path)
    state["fp64_gates_passed"] = bool(quality.fp64_pass.all())
    state["undefined_bir_denominators"] = {branch: int(metrics.loc[metrics.method == "neural", "BIR_" + branch].isna().sum()) for branch in ("D", "R")}
    state["f32_legacy_heuristic_failures"] = int((~quality.f32_legacy_heuristic_pass).sum())
    state["maximum_fp64_error_over_tolerance"] = float((quality.fp64_error / quality.fp64_tolerance).max())
    if not args.tables_only:
        figures = Figures(destination / "figures", not complete, "" if complete else f" [partial: {state['stage_ready_tasks']}/680 ready]")
        plot_oat(metrics, paired, controls, manifest, figures)
        plot_direct_references(metrics, direct, manifest, figures)
        if args.stage == "full":
            plot_budget(metrics, controls, direct, manifest, figures)
        if not interaction.empty:
            plot_interactions(metrics, interaction, manifest, figures)
        plot_gradients(gradients, manifest, figures)
        state["figures"] = figures.paths
    state["report_success"] = True
    json_write(destination / "REPORT_STATUS.json", state)
    print(json.dumps({k: v for k, v in state.items() if k not in ("source_records", "figures", "direct_reference_sources")}, indent=2))


if __name__ == "__main__":
    main()
