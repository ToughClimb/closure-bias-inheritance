"""Recompute the scientific results into a new directory, preserving frozen data.

Run from a checkout after installing the pinned reproducibility requirements.
Frozen inputs and reference results are included in reproducibility/theory.
"""
from __future__ import annotations
import response_bounds as rb
import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np
import observation_ellipsoid as observation
import mismatched_ellipsoid as mismatch
import metric_audit as metric
import verify_bounds
import build_assets


def verify_inputs():
    """Validate every bundled input before doing numerical work."""
    manifest = json.loads((rb.DATA / "INPUT_MANIFEST.json").read_text())
    for relative, record in manifest["files"].items():
        path = rb.DATA / relative
        if rb.sha(path) != record["sha256"]:
            raise ValueError("Frozen input changed: " + relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    verify_inputs()
    frozen = rb.REFERENCE
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    rb.OUT = target
    # The main scientific driver is imported under its own module name, so pool
    # functions remain pickleable and all workers receive the new output root.
    sys.argv = [str(rb.__file__), "--mode", "formal", "--subdivisions", "100", "--workers", str(args.workers)]
    rb.main()
    sys.argv = [str(rb.__file__), "--mode", "refine", "--subdivisions", "200", "--workers", str(args.workers)]
    rb.main()
    for module, name in ((observation, "observation_ellipsoid"), (mismatch, "mismatched_ellipsoid")):
        folder = target / name; folder.mkdir()
        for seed in range(5):
            got = module.run(seed, folder)
            expected = json.loads((frozen / name / f"seed{seed}.json").read_text())
            keys = ("center_gain", "interval_lower", "interval_upper", "actual_constitutive_gain")
            for key in keys:
                np.testing.assert_allclose(got[key], expected[key], rtol=2e-8, atol=2e-13)
            assert got["robust_sign"] == expected["robust_sign"]
        print(json.dumps(dict(stage=name, status="reproduced")), flush=True)
    folder = target / "metric_verified"; folder.mkdir()
    context = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        completed = list(pool.map(metric.run_seed, range(5)))
    for got in completed:
        expected = json.loads((frozen / "metric_verified" / f"seed{got['seed']}.json").read_text())
        for horizon in ("0.1", "0.3", "0.6"):
            for key in ("maximum_gain", "commutator_norm", "relative_commutator"):
                np.testing.assert_allclose(got["matrix_diagnostics"][horizon][key],
                                           expected["matrix_diagnostics"][horizon][key], rtol=2e-7, atol=2e-10)
        for row in got["constructed_witness_checks"]:
            assert row["actual_dynamical_gain"] > 0
            assert row["diffusion_error_squared_after"] < row["diffusion_error_squared_before"]
            assert row["reaction_error_squared_after"] < row["reaction_error_squared_before"]
    print(json.dumps(dict(stage="metric_verified", status="reproduced")), flush=True)
    verify_bounds.main()
    for name in ("formal_dt100", "refine_dt200"):
        expected = json.loads((frozen / name / "SUMMARY.json").read_text())
        got = json.loads((target / name / "SUMMARY.json").read_text())
        assert got["by_predictor"] == expected["by_predictor"]
    build_assets.main()
    receipt = dict(status="passed", checked_utc=datetime.now(timezone.utc).isoformat(),
        repository=".", output_directory=target.name,
        original_results_preserved="reproducibility/theory/expected",
        response_counts_reproduced=True, sharp_matched_and_mismatched_intervals_reproduced=True,
        all_15_full_metric_diagnostics_reproduced=True,
        all_10_constructed_controls_reproduced=True,
        numerical_ingredient_checks_passed=True, figures_and_tables_regenerated=True,
        driver_sha256=rb.sha(__file__))
    rb.save(target / "REPRODUCTION.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
