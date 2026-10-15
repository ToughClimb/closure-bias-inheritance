"""Independent artifact/count/norm/bound checks on selector diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    return list(csv.DictReader(Path(path).open()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/reviewer-selector-2026-10-15")
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = json.loads((out / "PROTOCOL.json").read_text())
    result = json.loads((out / "RESULTS.json").read_text())
    inputs = json.loads((out / "INPUTS.json").read_text())
    assert result["status"] == "complete" and result["failed_teachers"] == 0
    assert all(sha(p["path"]) == p["sha256"] for p in inputs)
    assert all(sha(p["path"]) == p["sha256"] for p in protocol["source_files"])
    candidates = {p.stem: json.loads(p.read_text()) for p in (out / "candidate_records").glob("*.json")}
    assert len(candidates) == 85
    assert Counter(r["config"]["cohort"] for r in candidates.values()) == {"historical_main_150": 75, "sameobs_6000": 10}
    assert all(b["validation"]["passed"] for r in candidates.values() for b in r["branches"].values())
    prior_cells = sum(b["validation"]["old_grid_cells"] for r in candidates.values() for b in r["branches"].values())
    assert prior_cells == 3750
    rows = csv_rows(out / "dense_grid.csv")
    assert len(rows) == 85 * 2 * 15 * 20
    expected_grid = {(r, a) for r in protocol["relative"] for a in protocol["absolute"]}
    by_branch = defaultdict(list)
    by_cell = defaultdict(list)
    violations = []
    maximum_identity = 0.
    maximum_bound = -math.inf
    for row in rows:
        ident, branch = row["id"], row["branch"]
        by_branch[ident, branch].append(row)
        by_cell[tuple(row[k] for k in ["cohort", "case", "setting", "epoch", "branch", "relative", "absolute"])].append(row)
        pool = candidates[ident]["branches"][branch]["candidates"]
        best_mse = min(c["fit_mse"] for c in pool)
        threshold = (1 + float(row["relative"])) * best_mse + float(row["absolute"])
        eligible = [c for c in pool if c["fit_mse"] <= threshold]
        selected = min(eligible, key=lambda c: (c["complexity"], c["fit_mse"]))
        assert selected["index"] == int(row["candidate_index"])
        assert selected["family"] == row["family"] and selected["complexity"] == int(row["complexity"])
        eN, eC, eS = [float(row[k]) for k in ["eN", "eC", "eS"]]
        bir, rho, cos = [float(row[k]) for k in ["BIR", "rho", "cosine_alignment"]]
        assert all(math.isfinite(v) for v in [eN, eC, eS, bir, rho, cos])
        assert eN > 0 and eC >= 0 and eS >= 0 and -1 - 1e-12 <= cos <= 1 + 1e-12
        assert math.isclose(bir, eS / eN, rel_tol=2e-13, abs_tol=2e-13)
        assert math.isclose(rho, eC / eN, rel_tol=2e-13, abs_tol=2e-13)
        identity = abs(bir**2 - 1 - rho**2 - 2 * rho * cos)
        maximum_identity = max(maximum_identity, identity)
        if identity > 1e-9 * max(1., bir**2, rho**2):
            violations.append(dict(id=ident, branch=branch, reason="norm_identity", value=identity))
        if abs(bir - 1) > rho + 1e-10:
            violations.append(dict(id=ident, branch=branch, reason="reverse_triangle_bound"))
        if row["compression_sup_lipschitz_upper"]:
            difference = float(row["sampled_compression_sup"]) - float(row["compression_sup_lipschitz_upper"])
            maximum_bound = max(maximum_bound, difference)
            assert difference <= 1e-12
        assert float(row["sampled_sup_refinement_increment"]) >= -1e-15
    for key, values in by_branch.items():
        coordinates = [(float(r["relative"]), float(r["absolute"])) for r in values]
        assert len(coordinates) == 300 and set(coordinates) == expected_grid
        assert len({r["eN"] for r in values}) == 1  # Teacher stayed fixed.
    assert all(len(values) == 5 and {int(r["seed"]) for r in values} == set(range(5)) for values in by_cell.values())
    assert not violations
    probes = csv_rows(out / "boundary_probes.csv")
    events = csv_rows(out / "boundary_events.csv")
    assert len(events) == result["boundary_events"] and len(probes) == result["boundary_probe_rows"]
    observed = defaultdict(dict)
    for row in probes:
        key = tuple(row[k] for k in ["id", "branch", "boundary_axis", "fixed_other_value", "analytic_boundary"])
        observed[key][row["boundary_side"]] = int(row["candidate_index"])
    for event in events:
        key = tuple(event[k] for k in ["id", "branch", "boundary_axis", "fixed_other_value", "analytic_boundary"])
        assert json.loads(event["probe_selection_indices"]) == observed[key]
        if event["selected_expression_changes"] == "True":
            assert observed[key]["below_separated"] != observed[key]["above_separated"]
    figures = json.loads((out / "FIGURES.json").read_text())
    assert figures["figure_count"] == 3 and figures["status"] == "complete"
    assert figures["source_results_sha256"] == sha(out / "RESULTS.json")
    assert figures["script_sha256"] == sha(figures["script"])
    for figure in figures["figures"]:
        for f in figure["files"]:
            assert sha(f["path"]) == f["sha256"]
    initial = json.loads((out / "attempts/float32_snapshot_fit/VALIDATION.json").read_text())
    assert initial["status"] == "failed" and initial["failed_teachers"] == 10
    assert len(list((out / "attempts/float32_snapshot_fit/failures").glob("*.json"))) == 10
    verification = dict(status="PASS", script_sha256=sha(Path(__file__)),
                        current_run_failed_teachers=0, archived_initial_wrong_input_failures=10,
                        current_nonfinite_cells=result["nonfinite_dense_rows"],
                        teachers=85, branches=170, historical_prior_grid_branch_cells=prior_cells,
                        dense_rows=len(rows), per_teacher_branch_grid=300, per_cell_dataset_seeds=5,
                        boundary_events=len(events), boundary_probes=len(probes),
                        maximum_independent_squared_identity_residual=maximum_identity,
                        maximum_sampled_sup_bound_violation=maximum_bound,
                        source_and_input_hashes_unchanged=True,
                        cohort_separation_verified=True, fixed_teacher_errors_verified=True,
                        figures_hashes_verified=True, failures_retained=True,
                        violations=violations)
    (out / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
