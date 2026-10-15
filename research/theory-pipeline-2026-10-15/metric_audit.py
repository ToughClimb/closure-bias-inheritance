"""Compute the complete nine-function PDE response metric and constructive modes.

This is a synthetic-truth mechanism diagnostic. The polynomial projection is an
explicit controlled reference, separate from the actual nonlinear selector.
"""
from __future__ import annotations
import response_bounds as rb
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.integrate import RK45
from function_basis import FunctionBasis


def solve(problem, basis, pattern, direction=None, epsilon=0, tangent=False):
    nx, dx = problem.nx, problem.dx
    initial = problem.initial(pattern)
    y = np.r_[initial, np.zeros(nx * 9)] if tangent else initial.copy()
    minimum_diffusion = float("inf")
    def law(u):
        d, r = .01 + .05 * u, u * (1 - u)
        if direction is not None:
            bd, br = basis.values(u)
            d, r = d + epsilon * (bd @ direction), r + epsilon * (br @ direction)
        return d, r
    def rhs(t, y):
        nonlocal minimum_diffusion
        u = y[:nx]
        d, r = law(u)
        if not np.isfinite(y).all() or np.min(d) <= 0 or np.max(np.abs(u)) > 10:
            raise FloatingPointError("Invalid metric-control state")
        minimum_diffusion = min(minimum_diffusion, float(np.min(d)))
        jump = (np.roll(u, -1) - u) / dx
        flux = .5 * (d + np.roll(d, -1)) * jump
        f = (flux - np.roll(flux, 1)) / dx + r
        if not tangent:
            return f
        z = y[nx:].reshape(nx, 9)
        bd, br = basis.values(u)
        dz_diff = .05 * z + bd
        face = .5 * (dz_diff + np.roll(dz_diff, -1, axis=0)) * jump[:, None]
        face += .5 * (d + np.roll(d, -1))[:, None] * (np.roll(z, -1, axis=0) - z) / dx
        dz = (face - np.roll(face, 1, axis=0)) / dx + (1 - 2 * u)[:, None] * z + br
        return np.r_[f, dz.ravel()]
    d, _ = law(initial)
    solver = RK45(rhs, 0, y, .6, rtol=2e-7, atol=2e-10,
                  max_step=min(.005, .175 * dx**2 / float(d.max())))
    times, states = rb.old.first.TIMES, [y]
    index = 1
    while solver.status == "running":
        d, _ = law(solver.y[:nx])
        solver.max_step = min(.005, .175 * dx**2 / float(d.max()))
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError(message)
        dense = solver.dense_output()
        while index < len(times) and times[index] <= solver.t + 1e-13:
            states.append(dense(float(times[index]))); index += 1
    assert index == len(times)
    return np.asarray(states), minimum_diffusion


def run_seed(seed):
    start = time.perf_counter()
    output = rb.OUT / "metric_verified"
    problem = rb.old.CurvatureProblem(seed, 64)
    basis = FunctionBasis(problem)
    P = np.diag([1.] * 7 + [0.] * 2)
    fields, matrices = {}, {}
    for pattern in rb.old.first.PATTERNS:
        fields[pattern], _ = solve(problem, basis, pattern, tangent=True)
        with np.load(rb.AUDIT / f"formal_nx64/s{seed}_{pattern}_nx64.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(fields[pattern][:, :64], saved["truth"], atol=2e-10, rtol=2e-9)
    for horizon in (.1, .3, .6):
        k = int(round(horizon / .01))
        K = np.concatenate([f[k, 64:].reshape(64, 9) for f in fields.values()]) / np.sqrt(4 * 64)
        M = K.T @ K
        commutator = P @ M - M @ P
        difference = P @ M @ P - M
        eigenvalues, eigenvectors = np.linalg.eigh(difference)
        matrices[str(horizon)] = dict(M=M, K=K, difference=difference, witness=eigenvectors[:, -1],
            maximum_gain=float(eigenvalues[-1]), commutator_norm=float(np.linalg.norm(commutator, 2)),
            relative_commutator=float(np.linalg.norm(commutator, 2) / np.linalg.norm(M, 2)),
            diffusion_reaction_cross_norm=float(np.linalg.norm(M[np.ix_([0, 1, 2, 7], [3, 4, 5, 6, 8])], 2)))
    # The deterministic largest positive mode supplies a constructive witness;
    # no accuracy population claim is made from this mode construction.
    witness = matrices["0.6"]["witness"]
    assert matrices["0.6"]["maximum_gain"] > 0
    sample = np.linspace(-.1, 1.5, 4097)
    bd, _ = basis.values(sample)
    amplitude = min(1e-3, .001 / max(np.max(np.abs(bd @ witness)), 1e-12))
    rows = []
    for epsilon in (amplitude, amplitude / 2):
        original_errors, projected_errors = [], []
        queried_minima = []
        for pattern in rb.old.first.PATTERNS:
            original, minimum_original = solve(problem, basis, pattern, witness, epsilon)
            projected, minimum_projected = solve(problem, basis, pattern, P @ witness, epsilon)
            truth = fields[pattern][:, :64]
            original_errors.append(np.mean((original[-1] - truth[-1])**2))
            projected_errors.append(np.mean((projected[-1] - truth[-1])**2))
            queried_minima.extend((minimum_original, minimum_projected))
        actual = float(np.mean(projected_errors) - np.mean(original_errors))
        predicted = epsilon**2 * matrices["0.6"]["maximum_gain"]
        assert actual > 0 and predicted > 0
        d_indices, r_indices = [0, 1, 2, 7], [3, 4, 5, 6, 8]
        before_d, after_d = np.sum(witness[d_indices]**2), np.sum((P @ witness)[d_indices]**2)
        before_r, after_r = np.sum(witness[r_indices]**2), np.sum((P @ witness)[r_indices]**2)
        assert after_d < before_d and after_r < before_r
        rows.append(dict(epsilon=epsilon, diffusion_error_squared_before=epsilon**2 * before_d,
                         diffusion_error_squared_after=epsilon**2 * after_d,
                         reaction_error_squared_before=epsilon**2 * before_r,
                         reaction_error_squared_after=epsilon**2 * after_r,
                         predicted_dynamical_gain=predicted, actual_dynamical_gain=actual,
                         relative_linearization_error=abs(actual - predicted) / predicted,
                         minimum_queried_diffusion=min(queried_minima)))
    np.savez_compressed(output / f"seed{seed}.npz", **{f"fields_{k}": v for k, v in fields.items()},
                        **{f"M_T{k}": v["M"] for k, v in matrices.items()},
                        **{f"K_T{k}": v["K"] for k, v in matrices.items()}, witness=witness, projection=P)
    record = dict(status="complete", seed=seed, basis=basis.record(),
                  source_sha256={"weights": rb.sha(problem.weights_path), "metadata": rb.sha(problem.metadata_path)},
                  driver_sha256=rb.sha(__file__), basis_sha256=rb.sha(Path(__file__).with_name("function_basis.py")),
                  matrix_diagnostics={k: {key: value for key, value in v.items() if key not in ("M", "K", "difference", "witness")} for k, v in matrices.items()},
                  witness=witness.tolist(), constructed_witness_checks=rows,
                  scope="Five seed-specific function-coordinate diagnostics at synthetic truth; projection is the declared polynomial reference, not the actual nonlinear selected expression.",
                  seconds=time.perf_counter() - start)
    rb.save(output / f"seed{seed}.json", record)
    return record


if __name__ == "__main__":
    output = rb.OUT / "metric_verified"; output.mkdir(exist_ok=False)
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_seed, seed) for seed in range(5)]
        for future in concurrent.futures.as_completed(futures):
            record = future.result(); completed.append(record)
            print(json.dumps(dict(done=len(completed), seed=record["seed"], seconds=record["seconds"])), flush=True)
    rb.save(output / "SUMMARY.json", dict(status="complete", seed_units=5, complete_matrices=15,
        functions_per_matrix=9, new_tangent_solves=20, new_nonlinear_control_solves=80,
        all_commutators_nonzero=all(v["commutator_norm"] > 0 for r in completed for v in r["matrix_diagnostics"].values()),
        all_two_branch_improvements_and_dynamical_increases=True,
        driver_sha256=rb.sha(__file__), completed_utc=datetime.now(timezone.utc).isoformat()))
