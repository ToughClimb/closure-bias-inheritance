"""Read-only fresh CUDA initialization and diagnostic state audit; no formal fit."""
from __future__ import annotations

import copy
import json
import random
import time

from run_study import context, diagnose, initial_model_hash, read_manifest, terms, vector
from study_io import DEFAULT_OUTPUT, REPO, STUDY, file_hash, load_core, load_input


def main():
    started = time.perf_counter()
    core = load_core(DEFAULT_OUTPUT, 'cuda')
    torch, np = core.torch, core.np
    torch.set_num_threads(1)
    manifest = read_manifest(STUDY / 'MANIFEST.json')
    protocol = manifest['protocol']
    out = DEFAULT_OUTPUT / 'cuda-state-audit'
    out.mkdir(parents=True, exist_ok=True)

    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {k: clone(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(clone(v) for v in value)
        return copy.deepcopy(value)

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
        if isinstance(left, np.ndarray):
            return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        if isinstance(left, (tuple, list)):
            return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right

    init_rows = []
    for task in manifest['tasks']:
        if task['variant'] != 'default':
            continue
        observed, interval, meta = load_input(core, DEFAULT_OUTPUT, task['case'], task['setting'], task['data_seed'])
        cpu_record = REPO / 'outputs/reviewer-sensitivity-2026-10-15-v2/runs' / task['cpu_reference_task_id'] / 'result.json'
        if file_hash(cpu_record) != task['cpu_reference_metadata_sha256']:
            raise ValueError('CPU pairing metadata changed')
        torch.manual_seed(task['init_seed'])
        model = core.ReactionDiffusionClosure(input_range=interval)
        cpu_hash = initial_model_hash(model)
        model = model.to('cuda')
        torch.cuda.synchronize()
        gpu_hash = initial_model_hash(model)
        row = dict(case=task['case'], setting=task['setting'], seed=task['data_seed'], input_file_sha256=meta['file_sha256'], data_hash=meta['data_hash'], observation_signature=meta['observation_signature'], cpu_initial_sha256=cpu_hash, cuda_initial_sha256=gpu_hash, expected_cpu_initial_sha256=task['paired_cpu_initial_model_sha256'], all_parameters_cuda=all(p.is_cuda for p in model.parameters()))
        row['passed'] = row['all_parameters_cuda'] and cpu_hash == gpu_hash == row['expected_cpu_initial_sha256']
        assert row['passed']
        init_rows.append(row)
        del model
    assert len(init_rows) == 20
    rows = []
    for task in manifest['tasks']:
        if (task['case'], task['setting'], task['data_seed']) != ('case_exp', 'noise3', 0):
            continue
        observed, interval, meta = load_input(core, DEFAULT_OUTPUT, task['case'], task['setting'], task['data_seed'])
        torch.manual_seed(task['init_seed'])
        model = core.ReactionDiffusionClosure(input_range=interval).to('cuda')
        assert initial_model_hash(model) == task['paired_cpu_initial_model_sha256']
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol['optimizer']['learning_rate'])
        ctx = context(core, model, observed, interval, protocol, task['weak_variant'], 'cuda')
        parameters = tuple(model.parameters())

        def snapshot():
            torch.cuda.synchronize()
            return dict(model=clone(model.state_dict()), module_training_flags={name: module.training for name, module in model.named_modules()}, optimizer=clone(optimizer.state_dict()), parameter_gradients=[clone(p.grad) for p in parameters], support_gradient=clone(ctx['support'].grad), context_tensors={k:clone(v) for k,v in ctx.items() if isinstance(v, torch.Tensor)}, temporal=clone(ctx['temporal']), cpu_rng=clone(torch.get_rng_state()), cuda_rng=clone(torch.cuda.get_rng_state_all()), numpy_rng=clone(np.random.get_state()), python_rng=clone(random.getstate()))

        for epoch in range(4):
            update = None
            if epoch:
                optimizer.zero_grad(set_to_none=True)
                ctx['support'].grad = None
                values = terms(core, ctx)
                loss = sum(task['weights'][k]*values[k] for k in task['weights'])
                loss.backward()
                core.assert_finite_state(model, gradients=True, epoch=epoch)
                before_parameters = torch.cat([p.detach().reshape(-1) for p in parameters]).double()
                optimizer.step()
                core.assert_finite_state(model, optimizer, epoch=epoch)
                after_parameters = torch.cat([p.detach().reshape(-1) for p in parameters]).double()
                update = after_parameters-before_parameters
            before = snapshot()
            measured = diagnose(core, ctx, task['weights'], epoch, actual_update=update)
            after = snapshot()
            checks = {key: equal(before[key], after[key]) for key in before}
            checks['fp64_identity_pass'] = measured['high_precision_validation']['passed']
            tensors = [v for v in ctx.values() if isinstance(v, torch.Tensor)] + ([] if ctx['temporal'] is None else list(ctx['temporal']))
            checks['all_objective_tensors_cuda'] = all(v.is_cuda for v in tensors)
            checks['all_parameters_cuda'] = all(p.is_cuda for p in parameters)
            assert all(checks.values()), (task['variant'], epoch, checks)
            rows.append(dict(variant=task['variant'], weak_variant=task['weak_variant'], epoch=epoch, checks=checks, fp64_error=measured['high_precision_validation']['gradient_linearity_error'], fp64_tolerance=measured['high_precision_validation']['tolerance']))
        del model, optimizer, ctx
    assert len(rows) == 20
    report = dict(classification='runtime_pilot', formal_evidence=False, all_checks_passed=all(r['passed'] for r in init_rows) and all(all(r['checks'].values()) for r in rows), device='cuda', gpu_name=torch.cuda.get_device_name(), torch=torch.__version__, implementation_signature=manifest['implementation_signature'], driver_sha256=file_hash(STUDY/'run_study.py'), audit_source_sha256=file_hash(__file__), initial_pair_count=len(init_rows), initial_pairs=init_rows, diagnostics_call_count=len(rows), diagnostic_states=rows, optimizer_steps_per_variant=3, formal_optimizer_steps=0, comparison='Synchronized exact equality of independent CPU clones; original model, buffers, training flags, Adam full state, existing parameter/support gradients, context tensors, and CPU/CUDA/NumPy/Python RNG states before versus after diagnostics.', elapsed_seconds=time.perf_counter()-started)
    core.json_write(out/'CUDA_STATE_AUDIT.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'initial_pairs','diagnostic_states'}}, indent=2))


if __name__ == '__main__':
    main()
