"""Declare and run the independent controlled slit-flow campaign."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import concurrent.futures as cf
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from numpy.polynomial.legendre import leggauss
import torch
from slit_flow_core import (exact_velocity, exact_fluidity, exact_flow_rate, central_shear,
                            basis, weighted_projection, polynomial, polynomial_derivative,
                            observation_fit, train_teacher, network_values,
                            velocity_prediction, flow_prediction, geometry)

SOURCE = Path(__file__).resolve().parents[1]
NUMERICAL_FILES = ['experiments/slit_flow_core.py', 'experiments/run_slit_flow_controls.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plan():
    return dict(alpha=0.6, pressure_train=1.0, half_width=1.0,
                intervals=[16, 32, 64], relative_noise=[0.0, 0.01], seeds=list(range(5)),
                noise_standard_deviation='noise fraction times fixed clean maximum velocity, no clipping',
                seed_meaning='At zero noise five network restarts on identical data; at 1% five noise draws with paired network initializations',
                objectives=['differential', 'integral'], epochs=1500, learning_rate=0.005, width=16,
                integration_order_train=32, projection_order=512,
                symbolic_families={'linear':[1], 'odd_cubic':[1, 3]},
                symbolic_fit='Predeclared fixed linear spaces, weighted least squares to the teacher; no structure search or truth-based family selection',
                coefficient_refit='Exact observation least squares in each same fixed family, identical to the direct symbolic estimator; count once',
                diagnostic_controls='Manufactured error direction, finite-difference population bias, derivative-noise variance, nonmonotone small perturbations; not trained network results',
                pressure_test=[0.5, 1.0, 1.25], extrapolation='Pressure1.25 visits stress beyond training interval[0,1]',
                optimization='Fixed final epoch; no truth-based stopping, tuning, or restart selection',
                physical_class='Instantaneous, no-slip, symmetric, strictly monotone generalized-Newtonian fluid; no yield stress or history variables',
                physical_checks='Phi(0)=0, positive Phi(tau) for tau>0, positive incremental derivative on declared stress intervals')


def declare(output):
    p = plan()
    protocol = dict(plan=p, source_sha256={s:digest(SOURCE / s) for s in NUMERICAL_FILES}, datasets=[])
    output.mkdir(parents=True, exist_ok=True)
    for n in p['intervals']:
        y = np.linspace(0, 1, n + 1)
        clean = exact_velocity(y, alpha=p['alpha'])
        for noise in p['relative_noise']:
            for seed in p['seeds']:
                ident = f'n{n}_noise{noise:g}_seed{seed}'
                velocity = clean + noise * float(clean.max()) * np.random.default_rng(seed).normal(size=len(y))
                path = output / 'inputs' / (ident + '.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    with np.load(path) as old:
                        if not np.array_equal(old['y'], y) or not np.array_equal(old['velocity'], velocity):
                            raise ValueError('Existing input differs: ' + ident)
                else:
                    np.savez_compressed(path, y=y, velocity=velocity)
                protocol['datasets'].append(dict(id=ident, intervals=n, noise=noise, seed=seed,
                    file=str(path.relative_to(output)), file_sha256=digest(path),
                    observation_sha256=hashlib.sha256(velocity.tobytes()).hexdigest()))
    path = output / 'PROTOCOL.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol already exists with different inputs or numerical source')
    if not path.exists():
        write_json(path, protocol)
        for relative in NUMERICAL_FILES:
            target = output / 'provenance/source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / relative, target)
        write_json(output / 'provenance/environment.json', dict(command=sys.argv, python=sys.version,
            packages={name:importlib.metadata.version(name) for name in ['numpy', 'scipy', 'torch', 'matplotlib', 'pytest']},
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
            git_status=subprocess.check_output(['git', 'status', '--short'], cwd=SOURCE, text=True),
            existing_core_sha256={str(x.relative_to(SOURCE)):digest(x)
                for x in sorted((SOURCE / 'src').rglob('*.py')) + sorted((SOURCE / 'revision').glob('*.py'))}))
    return protocol


def load_protocol(output):
    protocol = json.loads((output / 'PROTOCOL.json').read_text())
    for relative, value in protocol['source_sha256'].items():
        if digest(SOURCE / relative) != value:
            raise ValueError('Numerical source changed: ' + relative)
    return protocol


def evaluate(function, derivative, y, observed, alpha):
    grid = np.linspace(0, 1, 1025)
    ext = np.linspace(0, 1.25, 1281)
    truth = exact_fluidity(grid, alpha)
    values, slopes = function(grid), derivative(grid)
    relative = float(np.linalg.norm(values - truth) / np.linalg.norm(truth))
    predicted = velocity_prediction(function, y)
    return dict(relative_closure_error=relative, observation_mse=float(np.mean((predicted - observed)**2)),
                sampled_min_positive_response=float(values[1:].min()), sampled_min_derivative=float(slopes.min()),
                sampled_min_derivative_extended=float(derivative(ext).min()),
                zero_response=float(function(np.array([0.]))[0]),
                flow_relative_errors={str(g):abs(flow_prediction(function, g) - exact_flow_rate(g, alpha)) / exact_flow_rate(g, alpha)
                                      for g in (0.5, 1.0, 1.25)})


def task(args):
    output_text, dataset, objective = args
    output = Path(output_text)
    protocol = load_protocol(output)
    p = protocol['plan']
    ident = dataset['id'] + '_' + objective
    path = output / 'fits' / (ident + '.json')
    identity = dict(dataset=dataset, objective=objective, source_sha256=protocol['source_sha256'])
    if path.exists():
        old = json.loads(path.read_text())
        if old['identity'] != identity or old['status'] != 'complete':
            raise ValueError('Existing result cannot be silently replaced: ' + ident)
        return ident, 'cached'
    started = time.perf_counter()
    record = dict(identity=identity, status='running')
    write_json(path, record)
    try:
        input_path = output / dataset['file']
        if digest(input_path) != dataset['file_sha256']:
            raise ValueError('Input digest changed')
        with np.load(input_path) as inputs:
            y, observed = inputs['y'], inputs['velocity']
        model, history = train_teacher(y, observed, objective=objective, init_seed=dataset['seed'],
                                      epochs=p['epochs'], learning_rate=p['learning_rate'],
                                      width=p['width'], order=p['integration_order_train'])
        curve = lambda x:network_values(model, x)
        derivative = lambda x:network_values(model, x, derivative=True)[1]
        nodes, weights = leggauss(p['projection_order'])
        nodes, weights = (nodes + 1) / 2, weights / 2
        target = curve(nodes)
        truth = exact_fluidity(nodes, p['alpha'])
        metrics = {'neural':evaluate(curve, derivative, y, observed, p['alpha'])}
        symbols = {}
        for degree, name in [(1, 'linear'), (3, 'odd_cubic')]:
            coefficients = weighted_projection(nodes, target, weights, degree)
            reference, singular = observation_fit(y, observed, degree)
            symbols[name] = dict(frozen_coefficients=coefficients.tolist(), observation_coefficients=reference.tolist(),
                observation_singular_values=singular.tolist(),
                frozen_geometry=geometry(truth, target, polynomial(nodes, coefficients), weights))
            for kind, c in [('frozen', coefficients), ('observation_fit', reference)]:
                function = lambda x,c=c:polynomial(x, c)
                slope = lambda x,c=c:polynomial_derivative(x, c)
                metrics[name + '_' + kind] = evaluate(function, slope, y, observed, p['alpha'])
        grid = np.linspace(0, 1.25, 1281)
        curves_path = output / 'curves' / (ident + '.npz')
        curves_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(curves_path, stress=grid, neural=curve(grid), truth=exact_fluidity(grid, p['alpha']),
                            **{name + '_' + kind:polynomial(grid, symbols[name][kind + '_coefficients'])
                               for name in symbols for kind in ('frozen', 'observation')})
        model_path = output / 'models' / (ident + '.pt')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict':model.state_dict(), 'width':p['width'], 'dtype':'float64', 'identity':identity}, model_path)
        record.update(status='complete', history=history, metrics=metrics, symbols=symbols,
                      curves_file=str(curves_path.relative_to(output)), curves_sha256=digest(curves_path),
                      model_file=str(model_path.relative_to(output)), model_sha256=digest(model_path),
                      seconds=time.perf_counter() - started, stopping='fixed_epoch_budget')
    except Exception as exc:
        record.update(status='failed', error=str(exc), traceback=traceback.format_exc(), seconds=time.perf_counter() - started)
    write_json(path, record)
    return ident, record['status']


def analytic_controls(output):
    protocol = load_protocol(output)
    alpha = protocol['plan']['alpha']
    nodes, weights = leggauss(512)
    nodes, weights = (nodes + 1) / 2, weights / 2
    truth = exact_fluidity(nodes, alpha)
    norm = lambda v:float(np.sqrt(np.sum(weights * v**2)))
    raw = nodes * np.sin(6 * np.pi * nodes)
    perpendicular = raw - polynomial(nodes, weighted_projection(nodes, raw, weights))
    perpendicular /= norm(perpendicular)
    parallel = nodes / norm(nodes)
    rows = []
    for angle in (0., np.pi / 6, np.pi / 3, np.pi / 2):
        teacher = truth + 0.03 * (np.cos(angle) * parallel + np.sin(angle) * perpendicular)
        for degree in (1, 3):
            coefficients = weighted_projection(nodes, teacher, weights, degree)
            projected_truth = polynomial(nodes, weighted_projection(nodes, truth, weights, degree))
            error = teacher - truth
            removed = error - polynomial(nodes, weighted_projection(nodes, error, weights, degree))
            result = geometry(truth, teacher, polynomial(nodes, coefficients), weights)
            result.update(angle_radians=angle, degree=degree, family_truth_remainder=norm(truth - projected_truth),
                          removed_teacher_error=norm(removed),
                          projection_energy_identity_residual=abs(result['symbolic_error']**2 - result['teacher_error']**2
                              - norm(truth - projected_truth)**2 + norm(removed)**2))
            rows.append(result)
    write_csv(output / 'controls/projection.csv', rows)

    rows = []
    rng = np.random.default_rng(20261013)
    for n in (16, 32, 64):
        y = np.linspace(0, 1, n + 1)
        clean = exact_velocity(y, alpha=alpha)
        fd = central_shear(y, clean)
        coefficients, _, _, _ = np.linalg.lstsq(basis(y[1:-1]), fd, rcond=None)
        sigma = 0.01 * float(clean.max())
        noise = rng.normal(scale=sigma, size=(20000, n + 1))
        differentiated = -(noise[:, 2:] - noise[:, :-2]) * n / 2
        expected = sigma**2 * n**2 / 2
        rows.append(dict(intervals=n, h=1 / n, coefficient1=float(coefficients[0]),
                         coefficient3=float(coefficients[1]), predicted_coefficient1=1 + alpha / n**2,
                         fd_bias_max_error=float(np.max(np.abs(fd - exact_fluidity(y[1:-1], alpha) - alpha * y[1:-1] / n**2))),
                         predicted_noise_variance=expected, empirical_noise_mean_square=float(np.mean(differentiated**2)),
                         monte_carlo_draws=20000, monte_carlo_seed=20261013))
    write_csv(output / 'controls/finite_difference.csv', rows)

    rows = []
    null_flux = nodes**3 - 0.6 * nodes
    for name, direction in [('flux_sensitive', parallel), ('zero_flux_error', null_flux / norm(null_flux))]:
        error = 0.03 * direction
        rows.append(dict(direction=name, closure_error=norm(error), flow_error=float(2 * np.sum(weights * nodes * error)),
                         flow_rate_truth=exact_flow_rate(alpha=alpha)))
    write_csv(output / 'controls/observable_direction.csv', rows)

    rows = []
    for cycles in (4, 16, 64, 256):
        k = 2 * np.pi * cycles
        en, ec = 0.1 / np.sqrt(3), np.sqrt(2) / k
        es = np.sqrt(en**2 + 1.6 / k**2)
        rows.append(dict(cycles=cycles, frequency=k, perturbation_supremum=2 / k, teacher_error=en,
                         compression_error=ec, symbolic_error=es, rho=ec / en, bir=es / en,
                         incremental_derivative_minimum=-0.9, negative_derivative_witness=1 / (2 * cycles),
                         positive_response_for_positive_stress=True, flow_teacher=2 * 1.1 / 3,
                         flow_compressed=2 * 1.1 / 3 - 4 / k**2, flow_truth=2 / 3,
                         construction='Analytical diagnostic expression; not a discovered symbolic fit'))
    write_csv(output / 'controls/monotonicity.csv', rows)
    print('Analytical controls complete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['declare', 'controls', 'train'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.phase == 'declare':
        protocol = declare(args.output)
        print('Declared', len(protocol['datasets']), 'input records; 60 paired teachers', flush=True)
    elif args.phase == 'controls':
        analytic_controls(args.output)
    else:
        protocol = load_protocol(args.output)
        tasks = [(str(args.output), d, objective) for d in protocol['datasets'] for objective in protocol['plan']['objectives']]
        results = []
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(task, tasks):
                results.append(result)
                print(*result, flush=True)
        failed = [result for result in results if result[1] not in ('complete', 'cached')]
        if failed:
            raise RuntimeError('Failed runs retained: ' + repr(failed))


if __name__ == '__main__':
    main()
