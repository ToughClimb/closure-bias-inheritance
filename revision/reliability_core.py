"""Portable closure exports and observation-only optimizer diagnostics.

Native MLP exports evaluate its actual weights outside the training interval;
they do not apply the historical constant-endpoint table extension.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
import torch

from artifact_io import json_write, npz_write, source_signature
from revision_core import (DEFAULT_WEIGHTS, objective_terms, make_test_functions_1d,
                          assert_finite_state, torch_write, uniform_spacing)


class NumpyMLPClosure:
    """Float64 evaluation of the trained polynomial-residual MLP weights."""

    def __init__(self, metadata, arrays):
        self.metadata = metadata
        self.arrays = {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
        if metadata['backbone'] != 'mlp':
            raise ValueError('Native export currently supports the polynomial-residual MLP')
        if not all(np.isfinite(value).all() for value in self.arrays.values()):
            raise FloatingPointError('Nonfinite native network export')

    @classmethod
    def from_model(cls, model):
        if model.backbone != 'mlp':
            raise ValueError('Expected an MLP closure')
        arrays = {}
        metadata = dict(backbone='mlp', min_diffusion=float(model.min_diffusion), branches={})
        arrays['log_diffusion_scale'] = model.log_diffusion_scale.detach().cpu().double().numpy().copy()
        for label, branch in [('D', model.diffusion_net), ('R', model.reaction_net)]:
            layers = []
            for module in branch.residual_net.network:
                if isinstance(module, torch.nn.Linear):
                    index = len(layers)
                    arrays[f'{label}_weight_{index}'] = module.weight.detach().cpu().double().numpy().copy()
                    arrays[f'{label}_bias_{index}'] = module.bias.detach().cpu().double().numpy().copy()
                    layers.append(dict(input=module.in_features, output=module.out_features))
                elif not isinstance(module, torch.nn.Tanh):
                    raise ValueError('Unsupported residual activation')
            arrays[f'{label}_coefficients'] = branch.coefficients.detach().cpu().double().numpy().copy()
            metadata['branches'][label] = dict(lower=float(branch.input_lower), upper=float(branch.input_upper),
                degree=branch.degree, residual_scale=float(branch.residual_scale), layers=layers)
        return cls(metadata, arrays)

    def scalar(self, branch, values):
        values = np.asarray(values, dtype=np.float64)
        specification = self.metadata['branches'][branch]
        span = max(specification['upper'] - specification['lower'], 1e-6)
        scaled = 2 * (values - specification['lower']) / span - 1
        polynomial = np.polynomial.polynomial.polyval(scaled, self.arrays[branch + '_coefficients'])
        hidden = scaled[..., None]
        for index in range(len(specification['layers'])):
            hidden = hidden @ self.arrays[f'{branch}_weight_{index}'].T + self.arrays[f'{branch}_bias_{index}']
            if index + 1 < len(specification['layers']):
                hidden = np.tanh(hidden)
        return polynomial + specification['residual_scale'] * hidden[..., 0]

    def diffusion(self, values):
        return self.metadata['min_diffusion'] + np.exp(self.arrays['log_diffusion_scale']) * expit(self.scalar('D', values))

    def reaction(self, values):
        return self.scalar('R', values)

    def save(self, path):
        npz_write(path, metadata=np.array(json.dumps(self.metadata, sort_keys=True)), **self.arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as values:
            metadata = json.loads(str(values['metadata']))
            arrays = {key: values[key].copy() for key in values.files if key != 'metadata'}
        return cls(metadata, arrays)

    def verify(self, model, value_range):
        reference = copy.deepcopy(model).cpu().double()
        probe = np.linspace(value_range[0] - .1, value_range[1] + .1, 257).reshape(1, -1)
        with torch.no_grad():
            tensor = torch.from_numpy(probe)
            expected = (reference.diffusion(tensor).numpy(), reference.reaction(tensor).numpy())
        maxima = {}
        for name, actual, truth in [('D', self.diffusion(probe), expected[0]), ('R', self.reaction(probe), expected[1])]:
            np.testing.assert_allclose(actual, truth, rtol=2e-11, atol=2e-12)
            maxima[name] = float(np.max(np.abs(actual - truth)))
        return dict(evaluation_dtype='float64', original_training_dtype=str(next(model.parameters()).dtype),
            outside_support_probe=True, maximum_absolute_difference=maxima,
            rule='Same trained weights, normalization, polynomial backbone and tanh residual; no state clamping or endpoint extension')


def objective_state(model, observed, value_range):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    u = torch.as_tensor(observed['u'], dtype=dtype, device=device)
    dx = uniform_spacing(observed['x'], 'space')
    dt = uniform_spacing(observed['t'], 'time', minimum=3)
    phi, grad = [torch.as_tensor(value, dtype=dtype, device=device)
                 for value in make_test_functions_1d(observed['x'], num_modes=4)]
    support = torch.linspace(*value_range, 256, dtype=dtype, device=device, requires_grad=True)
    anchor = torch.tensor([value_range[0]], dtype=dtype, device=device)
    return u, dx, dt, phi, grad, support, anchor


def objective_diagnostic(model, state, weights=None):
    weights = DEFAULT_WEIGHTS if weights is None else weights
    model.zero_grad(set_to_none=True)
    state[5].grad = None
    terms = objective_terms(model, *state, variant='space_weak')
    loss = sum(weights[key] * value for key, value in terms.items())
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite diagnostic objective')
    loss.backward()
    assert_finite_state(model, gradients=True)
    norm = float(torch.sqrt(sum(parameter.grad.double().square().sum() for parameter in model.parameters()
                                if parameter.grad is not None)))
    infinity = max(float(parameter.grad.abs().max()) for parameter in model.parameters() if parameter.grad is not None)
    return dict(loss=float(loss.detach()), gradient_norm=norm, gradient_max=infinity,
                terms={key: float(value.detach()) for key, value in terms.items()})


def tail_diagnostic(history, window=1000):
    if len(history) < window:
        raise ValueError('Incomplete tail window')
    width = max(1, window // 10)
    values = np.array([row['loss'] for row in history[-window:]], dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite objective history')
    first, last = float(values[:width].mean()), float(values[-width:].mean())
    return dict(window=window, start_epoch=history[-window]['epoch'], final_epoch=history[-1]['epoch'],
                mean_start=first, mean_end=last, relative_decline=(first-last)/max(abs(first), 1e-12),
                minimum=float(values.min()), maximum=float(values.max()),
                final_pre_step_gradient_norm=history[-1]['grad_norm'])


def lbfgs_crosscheck(model, observed, value_range, output, identity, *, outer_steps=15, inner_iterations=20):
    """Fixed float64 protocol, with safe checkpoints after whole optimizer steps.

    Always report this protocol separately from float32 Adam. Its stopping checks
    use only the training objective and gradient, never true closure error.
    """
    model = copy.deepcopy(model).double()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=inner_iterations,
        max_eval=50, history_size=50, tolerance_grad=1e-7, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    state = objective_state(model, observed, value_range)
    checkpoint = output / 'optimizer_checkpoint.pt'
    history, start_outer, calls = [], 0, 0
    initial_parameters = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).clone()
    initial = objective_diagnostic(model, state)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['identity'] != identity or saved['source_signature'] != source_signature():
            raise ValueError('L-BFGS checkpoint source/input/protocol mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        history = saved['history']; start_outer = saved['outer_steps']; calls = saved['closure_calls']
        initial = saved['initial']; initial_parameters = saved['initial_parameters'].to(next(model.parameters()).device)
        assert_finite_state(model, optimizer)
    stop = 'fixed_outer_budget_exhausted'
    start = time.perf_counter()

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        state[5].grad = None
        terms = objective_terms(model, *state, variant='space_weak')
        loss = sum(DEFAULT_WEIGHTS[key] * value for key, value in terms.items())
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite L-BFGS trial objective')
        loss.backward()
        assert_finite_state(model, gradients=True)
        return loss

    for outer in range(start_outer, outer_steps):
        before = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).clone()
        before_loss = objective_diagnostic(model, state)['loss']
        optimizer.step(closure)
        assert_finite_state(model, optimizer)
        diagnostics = objective_diagnostic(model, state)
        after = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
        delta = float(torch.max(torch.abs(after-before)))
        history.append(dict(outer_step=outer+1, closure_calls=calls, maximum_parameter_change=delta, **diagnostics))
        torch_write(checkpoint, dict(identity=identity, source_signature=source_signature(), model=model.state_dict(),
            optimizer=optimizer.state_dict(), initial=initial, initial_parameters=initial_parameters,
            history=history, outer_steps=outer+1, closure_calls=calls))
        json_write(output / 'history.json', history)
        print(json.dumps(dict(event='lbfgs_step', id=identity['id'], outer_step=outer+1,
                              loss=diagnostics['loss'], gradient_max=diagnostics['gradient_max'], closure_calls=calls)), flush=True)
        if diagnostics['gradient_max'] <= 1e-7:
            stop = 'observed_gradient_tolerance'; break
        if delta <= 1e-10 and abs(diagnostics['loss']-before_loss) <= 1e-12:
            stop = 'observed_parameter_and_objective_tolerances'; break
    final = objective_diagnostic(model, state)
    parameters = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    details = dict(status='complete', stopping=stop, initial=initial, final=final, closure_calls=calls,
        outer_steps=len(history), maximum_outer_steps=outer_steps, maximum_iterations_per_outer_step=inner_iterations,
        relative_parameter_change=float(torch.linalg.vector_norm(parameters-initial_parameters)/
                                        (torch.linalg.vector_norm(initial_parameters)+1e-12)),
        seconds=time.perf_counter()-start, training_dtype='float64',
        interpretation='Separate optimizer-and-precision crosscheck; local stopping or fixed budget, no global convergence claim')
    json_write(output / 'RESULT.json', details)
    return model, details
