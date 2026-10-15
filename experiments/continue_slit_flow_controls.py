"""Observation-only L-BFGS continuation of every declared nx64 teacher."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from numpy.polynomial.legendre import leggauss
import torch
from slit_flow_core import (FluidityNetwork, central_shear, integration_nodes, network_values,
                            exact_fluidity, weighted_projection, polynomial, polynomial_derivative,
                            observation_fit, geometry)
from run_slit_flow_controls import SOURCE, digest, write_json, load_protocol, evaluate


def declare(output):
    protocol = load_protocol(output)
    records = []
    for dataset in protocol['datasets']:
        if dataset['intervals'] != 64:
            continue
        for objective in protocol['plan']['objectives']:
            path = output / 'fits' / (dataset['id'] + '_' + objective + '.json')
            record = json.loads(path.read_text())
            if record['status'] != 'complete':
                raise ValueError('Cannot continue an incomplete teacher')
            records.append(dict(file=str(path.relative_to(output)), sha256=digest(path),
                                model_file=record['model_file'], model_sha256=record['model_sha256']))
    plan = dict(source_sha256=protocol['source_sha256'], continuation_source_sha256=digest(__file__),
                selection='All 20 nx64 teachers: both noise levels, both objectives, all five seeds; no best-seed selection',
                optimizer=dict(name='torch.optim.LBFGS', lr=1.0, max_iter=300, max_eval=450,
                               tolerance_grad=1e-9, tolerance_change=1e-12, history_size=20, line_search_fn='strong_wolfe'),
                target='Saved observations and the unchanged original objective; no truth-based stopping or tuning',
                reason='The first fixed-epoch run leaves a clean integral-objective fitting residual; inspect finite optimization effects',
                inputs=records)
    path = output / 'LBFGS_PROTOCOL.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Existing continuation protocol differs')
    write_json(path, plan)
    target = output / 'provenance/source/experiments/continue_slit_flow_controls.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, target)
    return plan


def task(args):
    output_text, item = args
    output = Path(output_text)
    p = load_protocol(output)['plan']
    cp = json.loads((output / 'LBFGS_PROTOCOL.json').read_text())
    if digest(__file__) != cp['continuation_source_sha256']:
        raise ValueError('Continuation numerical source changed')
    path = output / item['file']
    if digest(path) != item['sha256'] or digest(output / item['model_file']) != item['model_sha256']:
        raise ValueError('Original teacher changed')
    previous = json.loads(path.read_text())
    identity = dict(input=item, continuation_protocol_sha256=digest(output / 'LBFGS_PROTOCOL.json'))
    ident = path.stem
    target_path = output / 'continuation' / (ident + '.json')
    if target_path.exists():
        old = json.loads(target_path.read_text())
        if old['identity'] != identity or old['status'] != 'complete':
            raise ValueError('Existing continuation cannot be silently replaced')
        return ident, 'cached'
    record = dict(identity=identity, dataset=previous['identity']['dataset'],
                  objective=previous['identity']['objective'], status='running')
    write_json(target_path, record)
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        dataset = record['dataset']
        input_path = output / dataset['file']
        if digest(input_path) != dataset['file_sha256']:
            raise ValueError('Observation input changed')
        with np.load(input_path) as inputs:
            y, observed = inputs['y'], inputs['velocity']
        model = FluidityNetwork(p['width']).double()
        model.load_state_dict(torch.load(output / item['model_file'], weights_only=True)['state_dict'])
        if record['objective'] == 'differential':
            argument = torch.tensor(y[1:-1], dtype=torch.float64)
            truth_free_target = torch.tensor(central_shear(y, observed), dtype=torch.float64)
            quadrature = None
        else:
            points, weights = integration_nodes(y, order=p['integration_order_train'])
            argument = torch.tensor(points, dtype=torch.float64)
            quadrature = torch.tensor(weights, dtype=torch.float64)
            truth_free_target = torch.tensor(observed, dtype=torch.float64)
        options = {k:v for k,v in cp['optimizer'].items() if k != 'name'}
        optimizer = torch.optim.LBFGS(model.parameters(), **options)
        evaluations = []

        def closure():
            optimizer.zero_grad()
            prediction = model(argument)
            if quadrature is not None:
                prediction = (prediction * quadrature).sum(dim=1)
            loss = (prediction - truth_free_target).square().mean()
            loss.backward()
            if not torch.isfinite(loss) or not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise FloatingPointError('Nonfinite continuation loss or gradient')
            evaluations.append(float(loss.detach()))
            return loss

        initial_loss = float(closure().detach())
        optimizer.step(closure)
        final_loss = float(closure().detach())
        gradient = float(torch.sqrt(sum(parameter.grad.square().sum() for parameter in model.parameters())))
        state = optimizer.state[next(iter(model.parameters()))]
        curve = lambda x:network_values(model, x)
        derivative = lambda x:network_values(model, x, derivative=True)[1]
        nodes, weights = leggauss(p['projection_order'])
        nodes, weights = (nodes + 1) / 2, weights / 2
        values = curve(nodes)
        true_values = exact_fluidity(nodes, p['alpha'])
        metrics = {'neural':evaluate(curve, derivative, y, observed, p['alpha'])}
        symbols = {}
        for degree, name in [(1, 'linear'), (3, 'odd_cubic')]:
            coefficients = weighted_projection(nodes, values, weights, degree)
            reference, singular = observation_fit(y, observed, degree)
            symbols[name] = dict(frozen_coefficients=coefficients.tolist(), observation_coefficients=reference.tolist(),
                observation_singular_values=singular.tolist(),
                frozen_geometry=geometry(true_values, values, polynomial(nodes, coefficients), weights))
            for kind, c in [('frozen', coefficients), ('observation_fit', reference)]:
                metrics[name + '_' + kind] = evaluate(lambda x,c=c:polynomial(x,c),
                    lambda x,c=c:polynomial_derivative(x,c), y, observed, p['alpha'])
        curve_path = output / 'continuation_curves' / (ident + '.npz')
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        grid = np.linspace(0, 1.25, 1281)
        np.savez_compressed(curve_path, stress=grid, neural=curve(grid), truth=exact_fluidity(grid, p['alpha']),
            **{name + '_' + kind:polynomial(grid, symbols[name][kind + '_coefficients'])
               for name in symbols for kind in ('frozen', 'observation')})
        model_path = output / 'continuation_models' / (ident + '.pt')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict':model.state_dict(), 'optimizer_state':optimizer.state_dict(),
                    'width':p['width'], 'dtype':'float64', 'identity':identity}, model_path)
        record.update(status='complete', metrics=metrics, symbols=symbols,
                      optimizer=dict(initial_loss=initial_loss, final_loss=final_loss,
                                     final_gradient_norm=gradient, iterations=int(state.get('n_iter',0)),
                                     optimizer_evaluations=int(state.get('func_evals',0)), all_closure_evaluations=evaluations,
                                     gradient_tolerance_satisfied=gradient <= options['tolerance_grad']),
                      curves_file=str(curve_path.relative_to(output)), curves_sha256=digest(curve_path),
                      model_file=str(model_path.relative_to(output)), model_sha256=digest(model_path),
                      seconds=time.perf_counter() - started)
    except Exception as exc:
        record.update(status='failed', error=str(exc), traceback=traceback.format_exc(), seconds=time.perf_counter() - started)
    write_json(target_path, record)
    return ident, record['status']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['declare', 'train'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.phase == 'declare':
        cp = declare(args.output)
        print('Declared', len(cp['inputs']), 'same-observation continuations', flush=True)
    else:
        cp = json.loads((args.output / 'LBFGS_PROTOCOL.json').read_text())
        failed = []
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(task, [(str(args.output), item) for item in cp['inputs']]):
                print(*result, flush=True)
                if result[1] not in ('complete', 'cached'):
                    failed.append(result)
        if failed:
            raise RuntimeError('Continuation failures retained: ' + repr(failed))


if __name__ == '__main__':
    main()
