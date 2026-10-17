"""Re-evaluate the 150 published constrained rheology fits without fitting models.

Requires the research environment (NumPy, pandas, SciPy and PyTorch). Reads saved
observations, training splits, teacher weights, coefficients and dual variables.
It neither trains neural networks nor invokes an optimizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from rheology_core import design
from rheology_evaluate import Predictor, evaluate_processing, processing_reference
from rheology_monotone import margin_matrix


def verify(records: Path) -> dict:
    rheology = records / 'rheology'
    fits = json.loads((rheology / 'constraint_fit_records.json').read_text())
    specs = json.loads((rheology / 'inputs/training_specs.json').read_text())
    data = pd.read_csv(rheology / 'inputs/observations.csv')
    expected = pd.read_csv(rheology / 'rheology_monotone_by_model.csv')
    with np.load(rheology / 'inputs/checkpoint_6000.npz', allow_pickle=False) as archive:
        checkpoint = {key: archive[key] for key in archive.files}
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 61), np.linspace(-1, 1, 61),
                                indexing='ij'), axis=-1).reshape(-1, 2)
    extrema = dict(minimum_bernstein_margin=float('inf'),
                   max_recomputed_kkt_stationarity=0., max_recomputed_complementarity=0.,
                   max_recomputed_primal_dual_gap=0., max_heldout_relative_l2_difference=0.,
                   max_window_difference_celsius=0., max_training_log_mse_difference=0.)
    assert len(fits) == 150
    methods = {}
    for fit in fits:
        spec = specs[fit['spec_index']]
        assert spec['arm'] == 'primary_high_shear'
        assert fit['training_row_ids'] == spec['training_row_ids']
        assert fit['heldout_row_ids'] == spec['heldout_row_ids']
        weights = {key: checkpoint[key][spec['index']] for key in ('w0','b0','w2','b2','w4','b4')}
        teacher = Predictor(spec, weights=weights)
        train = data.iloc[spec['training_row_ids']]
        if fit['method'] == 'frozen_d6':
            a, y = design(grid, 6), teacher.predict_x(grid)
        else:
            assert fit['method'] == 'direct_d6'
            a = design(teacher.x(train.temperature.to_numpy(), train.rate.to_numpy()), 6)
            y = np.log(train.viscosity.to_numpy())
        coefficients = np.asarray(fit['coefficients'])
        multipliers = np.asarray(fit['dual_multipliers'])
        q, r = np.linalg.qr(a, mode='reduced')
        transform = solve_triangular(r, np.eye(28), lower=False) * np.sqrt(len(y))
        target = q.T @ y / np.sqrt(len(y))
        v = r @ coefficients / np.sqrt(len(y))
        g = margin_matrix(spec['rate_bounds'])
        raw_h = g @ transform
        scale = np.linalg.norm(raw_h, axis=1)
        h = raw_h / scale[:,None]
        rhs = np.full(36, .05 + 1e-8 - 1) / scale
        slack = h @ v - rhs
        bernstein = 1 + g @ coefficients
        assert np.max(np.abs(bernstein - fit['bernstein_margin'])) < 1e-11
        assert bernstein.min() >= .05 and multipliers.min() >= 0
        assert slack.min() > -1e-10
        stationarity = float(np.linalg.norm(v - target - h.T @ multipliers))
        complementarity = float(np.max(np.abs(multipliers * slack)))
        primal = .5 * np.sum((v - target)**2)
        dual = -.5*np.sum((h.T @ multipliers)**2) - multipliers @ (h @ target - rhs)
        gap = float(primal - dual)
        assert stationarity < 1e-9 and complementarity < 1e-9 and abs(gap) < 1e-9
        training_difference = abs(float(np.mean((a @ coefficients - y)**2)) - fit['diagnostic']['training_log_mse'])
        assert training_difference < 1e-10
        test = data.iloc[spec['heldout_row_ids']]
        high = test[test.rate >= 10]
        predictor = Predictor(spec, coefficients=coefficients, degree=6)
        predicted = np.exp(predictor(high.temperature.to_numpy(), high.rate.to_numpy()))
        relative_l2 = float(np.linalg.norm(predicted-high.viscosity.to_numpy()) / np.linalg.norm(high.viscosity.to_numpy()))
        row = expected[(expected.method == fit['method']) & (expected.material == fit['material'])
                       & (expected.fold == fit['fold']) & (expected.seed == fit['seed'])
                       & (expected.intervention == 'bernstein_constrained')]
        assert len(row) == 1
        relative_difference = abs(relative_l2-float(row.iloc[0].high_shear_relative_l2))
        processing, _, _ = evaluate_processing(predictor, processing_reference(test))
        window_difference = abs(processing['processing_window_symmetric_difference_celsius']
                                - float(row.iloc[0].processing_window_symmetric_difference_celsius))
        assert relative_difference < 1e-10 and window_difference < 1e-9
        extrema['minimum_bernstein_margin'] = min(extrema['minimum_bernstein_margin'], float(bernstein.min()))
        for key,value in [('max_recomputed_kkt_stationarity', stationarity),
                          ('max_recomputed_complementarity', complementarity),
                          ('max_recomputed_primal_dual_gap', abs(gap)),
                          ('max_heldout_relative_l2_difference', relative_difference),
                          ('max_window_difference_celsius', window_difference),
                          ('max_training_log_mse_difference', training_difference)]:
            extrema[key] = max(extrema[key], value)
        methods[fit['method']] = methods.get(fit['method'], 0) + 1
    assert methods == {'direct_d6':50, 'frozen_d6':100}
    return dict(status='passed', saved_fits_verified=len(fits), methods=methods,
                neural_training_runs=0, optimizer_calls=0, **extrema)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, default=Path(__file__).resolve().parent/'paper-records-2026-10-16')
    args = parser.parse_args()
    print(json.dumps(verify(args.records.resolve()), indent=2))
