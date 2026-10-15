"""Isolated short CUDA timing of original versus coalesced diagnostic reads.

Does not alter the training implementation, campaign checkpoints, precision,
observations, objective or ongoing experiment services. Other jobs may share the
GPU: these are indicative measurements, not an exclusive-device speed benchmark.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'revision'))

import numpy as np
import torch
from artifact_io import ROOT, json_write, source_signature
from reliability_core import objective_state
from revision_core import objective_terms, DEFAULT_WEIGHTS, assert_finite_state
from closure_discovery.models.mlp_closure import ReactionDiffusionClosure


def grouped_checks(model, optimizer=None, *, gradients=False):
    flags = {}
    for parameter in model.parameters():
        values = [parameter]
        if gradients and parameter.grad is not None: values.append(parameter.grad)
        for value in values:
            flags.setdefault(value.device, []).append(torch.isfinite(value).all())
    if optimizer is not None:
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    flags.setdefault(value.device, []).append(torch.isfinite(value).all())
    for values in flags.values():
        if not torch.stack(values).all():
            # Preserve the original detailed error if a finite-state guard fails.
            assert_finite_state(model, optimizer, gradients=gradients)
            raise FloatingPointError('Finite-state guard failed')


def step(model, optimizer, state, coalesced):
    optimizer.zero_grad(set_to_none=True); state[5].grad = None
    terms = objective_terms(model, *state)
    loss = sum(DEFAULT_WEIGHTS[k]*v for k, v in terms.items())
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite benchmark objective')
    loss.backward()
    checker = grouped_checks if coalesced else assert_finite_state
    checker(model, gradients=True)
    grad = torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters() if p.grad is not None))
    if not coalesced: grad = float(grad)
    optimizer.step(); checker(model, optimizer)
    if coalesced:
        values = torch.stack([loss.detach().double(), grad, *[v.detach().double() for v in terms.values()]])
        return values.cpu().tolist()
    return [float(loss.detach()), grad, *[float(v.detach()) for v in terms.values()]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=60)
    args = parser.parse_args()
    if args.steps < 1: parser.error('steps must be positive')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    with np.load(args.observations, allow_pickle=False) as data:
        observed = {key: data[key].copy() for key in data.files}
    torch.set_num_threads(1); torch.manual_seed(4)
    original = ReactionDiffusionClosure(input_range=(0., 1.8)).cuda()
    models = [copy.deepcopy(original) for _ in range(2)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=.001) for model in models]
    states = [objective_state(model, observed, (0., 1.8)) for model in models]
    a = step(models[0], optimizers[0], states[0], False)
    b = step(models[1], optimizers[1], states[1], True)
    if a != b: raise AssertionError('Diagnostic scalar values differ')
    for p, q in zip(models[0].parameters(), models[1].parameters(), strict=True):
        if not torch.equal(p, q): raise AssertionError('One-step parameters differ')
    timings = []
    for coalesced, model, optimizer, state in zip((False, True), models, optimizers, states, strict=True):
        for _ in range(5): step(model, optimizer, state, coalesced)
        torch.cuda.synchronize(); start = time.perf_counter()
        for _ in range(args.steps): step(model, optimizer, state, coalesced)
        torch.cuda.synchronize(); seconds = time.perf_counter()-start
        timings.append(dict(coalesced=coalesced, steps=args.steps, seconds=seconds,
                            milliseconds_per_step=1000*seconds/args.steps))
    parity = all(torch.equal(p, q) for p, q in zip(models[0].parameters(), models[1].parameters(), strict=True))
    if not parity: raise AssertionError('Benchmark parameter trajectories differ')
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profile:
        for _ in range(5): step(models[0], optimizers[0], states[0], False)
        torch.cuda.synchronize()
    events = profile.key_averages()
    table = events.table(sort_by='self_cpu_time_total', row_limit=20)
    (ROOT/'profile.txt').write_text(table)
    synchronization = [dict(name=e.key, count=e.count, self_cpu_time_us=e.self_cpu_time_total,
                            cpu_time_us=e.cpu_time_total)
                       for e in events if 'Synchronize' in e.key or 'Memcpy' in e.key or '_local_scalar' in e.key]
    result = dict(command=sys.argv, source_signature=source_signature(), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), observation_shape=observed['u'].shape,
        finite_guards_retained=True, bitwise_parameter_parity_after_all_timed_steps=parity,
        diagnostic_scalar_parity=True, timings=timings,
        indicative_speed_ratio=timings[0]['seconds']/timings[1]['seconds'],
        profiler_steps=5, synchronization_events=synchronization,
        nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw,utilization.gpu,memory.used', '--format=csv'], text=True),
        limitations='Other formal jobs share GPU. Short indicative timing; no training-source changes or checkpoint writes.')
    json_write(ROOT/'RESULT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
