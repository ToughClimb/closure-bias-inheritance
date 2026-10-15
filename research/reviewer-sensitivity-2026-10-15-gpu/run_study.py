"""Fresh, paired loss-weight study with explicit budgets and gradient diagnostics.

The existing repository supplies the model, objective terms, input generation and
symbolic fitter. This driver adds diagnostics and isolated checkpoint identities;
it never imports a historical optimizer/model checkpoint.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from make_manifest import implementation_signature
from study_io import STUDY, REPO, DEFAULT_OUTPUT, canonical_hash, file_hash, input_id, load_core, load_input

INPUT_CACHE = {}
INVOCATION = list(sys.argv)


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest["implementation_signature"] != implementation_signature():
        raise ValueError("Study implementation changed after manifest generation; regenerate the manifest before launching new runs")
    if manifest["protocol_sha256"] != canonical_hash(manifest["protocol"]):
        raise ValueError("Manifest protocol hash mismatch")
    if len({t["id"] for t in manifest["tasks"]}) != len(manifest["tasks"]):
        raise ValueError("Duplicate task identities")
    return manifest


def context(core, model, observed, interval, protocol, weak_variant, device):
    torch, np = core.torch, core.np
    u = torch.as_tensor(observed["u"], dtype=torch.float32, device=device)
    dx = core.uniform_spacing(observed["x"], "space")
    dt = core.uniform_spacing(observed["t"], "time", minimum=3)
    phi, grad_phi = [torch.as_tensor(v, dtype=torch.float32, device=device)
                     for v in core.make_resolved_test_functions_1d(observed["x"], protocol["num_test_modes"])]
    support = torch.linspace(*interval, 256, requires_grad=True, device=device)
    anchor = torch.tensor([interval[0]], dtype=torch.float32, device=device)
    temporal = None
    if weak_variant != "space_weak":
        psi, dpsi = core.make_time_test_functions_1d(observed["t"])
        quadrature = np.ones(u.shape[1]); quadrature[[0, -1]] = 0.5
        temporal = tuple(torch.as_tensor(v, dtype=torch.float32, device=device)
                         for v in (psi, dpsi, quadrature))
    return dict(model=model, u=u, dx=dx, dt=dt, phi=phi, grad_phi=grad_phi,
                support=support, anchor=anchor, variant=weak_variant, temporal=temporal)


def terms(core, ctx):
    return core.objective_terms(**ctx)


def vector(core, grads, parameters):
    return core.torch.cat([(core.torch.zeros_like(p) if g is None else g).reshape(-1).double()
                          for g, p in zip(grads, parameters)])


def parameter_blocks(core, model):
    masks = {"all": [], "D": [], "R": []}
    names = {"D": [], "R": []}
    offset = 0
    for name, parameter in model.named_parameters():
        indexes = list(range(offset, offset + parameter.numel()))
        masks["all"].extend(indexes)
        branch = "D" if name.startswith("diffusion_net.") or name == "log_diffusion_scale" else "R" if name.startswith("reaction_net.") else None
        if branch is None:
            raise ValueError(f"Unclassified model parameter block: {name}")
        masks[branch].extend(indexes); names[branch].append(name)
        offset += parameter.numel()
    assert len(masks["D"]) + len(masks["R"]) == len(masks["all"])
    device = next(model.parameters()).device
    return {k: core.torch.tensor(v, dtype=core.torch.long, device=device) for k, v in masks.items()}, names


def initial_model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def cosine(core, a, b):
    torch = core.torch
    na, nb = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b))
    if na == 0 or nb == 0:
        return None
    return float(torch.clamp(torch.dot(a, b) / (na * nb), -1.0, 1.0))


def _diagnose_components(core, ctx, weights, epoch, actual_update=None):
    """Post-update losses and gradients, without writing optimizer .grad state."""
    torch = core.torch
    values = terms(core, ctx)
    parameters = tuple(ctx["model"].parameters())
    gradients = {}
    for name, value in values.items():
        grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        gradients[name] = vector(core, grads, parameters)
    total_loss = sum(weights[k] * values[k] for k in weights)
    direct = vector(core, torch.autograd.grad(total_loss, parameters, allow_unused=True), parameters)
    summed = sum(weights[k] * gradients[k] for k in weights)
    linearity_error = float(torch.linalg.vector_norm(direct - summed))
    total_norm = float(torch.linalg.vector_norm(direct))
    sum_component_norms = sum(weights[k] * float(torch.linalg.vector_norm(g)) for k, g in gradients.items())
    # Cancellation can make the total norm tiny even with large components.
    # This fixed float32 roundoff rule uses the accumulated component scale.
    tolerance = 64 * torch.finfo(torch.float32).eps * sum_component_norms + 1e-12
    if not all(torch.isfinite(value.detach()).all() for value in values.values()):
        raise FloatingPointError(f"Nonfinite diagnostic objective component at epoch {epoch}")
    if not all(torch.isfinite(g).all() for g in list(gradients.values()) + [direct, summed]):
        raise FloatingPointError(f"Nonfinite diagnostic component gradient at epoch {epoch}")
    masks, names = parameter_blocks(core, ctx["model"])
    blocks = {}
    for block, indexes in masks.items():
        gs = {k: g[indexes] for k, g in gradients.items()}
        total = direct[indexes]
        raw_norms = {k: float(torch.linalg.vector_norm(g)) for k, g in gs.items()}
        weighted_norms = {k: weights[k] * v for k, v in raw_norms.items()}
        update = actual_update[indexes] if actual_update is not None else None
        blocks[block] = dict(parameter_count=int(indexes.numel()),
                             total_gradient_norm=float(torch.linalg.vector_norm(total)),
                             raw_gradient_norms=raw_norms, weighted_gradient_norms=weighted_norms,
                             cosine_with_total={k: cosine(core, weights[k] * g, total) for k, g in gs.items()},
                             actual_update_norm=float(torch.linalg.vector_norm(update)) if update is not None else None,
                             raw_gradient_dot_actual_update={k: float(torch.dot(g, update)) if update is not None else None for k, g in gs.items()},
                             weighted_gradient_dot_actual_update={k: float(weights[k] * torch.dot(g, update)) if update is not None else None for k, g in gs.items()},
                             cosine_with_actual_update={k: cosine(core, g, update) if update is not None else None for k, g in gs.items()},
                             raw_pairwise_cosines={a: {b: cosine(core, gs[a], gs[b]) for b in gs} for a in gs},
                             weighted_pairwise_cosines={a: {b: cosine(core, weights[a] * gs[a], weights[b] * gs[b]) for b in gs} for a in gs})
    return dict(epoch=epoch, state="post_update" if epoch else "initial",
                raw_losses={k: float(v.detach()) for k, v in values.items()},
                weighted_losses={k: weights[k] * float(v.detach()) for k, v in values.items()},
                total_loss=float(total_loss.detach()), gradient_linearity_error=linearity_error,
                gradient_linearity_relative_to_total=linearity_error / total_norm if total_norm else None,
                gradient_linearity_relative_to_sum_component_norms=linearity_error / sum_component_norms if sum_component_norms else None,
                gradient_linearity_tolerance=tolerance,
                sum_weighted_component_gradient_norms=sum_component_norms,
                legacy_f32_heuristic_pass=(linearity_error <= tolerance),
                decomposition_precision=str(next(ctx["model"].parameters()).dtype),
                actual_update_interpretation="Post-update component gradients dotted with the most recent actual Adam parameter update; local directional diagnostics, not an exact finite loss change",
                parameter_blocks=blocks, parameter_names=names,
                zero_norm_cosine_policy="null means undefined; zero-gradient components are never assigned a zero cosine")


def diagnose(core, ctx, weights, epoch, actual_update=None):
    """Preserve actual float32 diagnostics and validate algebra independently in FP64."""
    policy = json.loads((STUDY / "NUMERICAL_POLICY.json").read_text())
    measured = _diagnose_components(core, ctx, weights, epoch, actual_update)
    promoted = dict(ctx, model=copy.deepcopy(ctx["model"]).double())
    for key, value in promoted.items():
        if isinstance(value, core.torch.Tensor):
            promoted[key] = value.detach().double().requires_grad_(key == "support")
        elif key == "temporal" and value is not None:
            promoted[key] = tuple(item.detach().double() for item in value)
    high = _diagnose_components(core, promoted, weights, epoch,
                                actual_update.double() if actual_update is not None else None)
    allowed = policy["validation_atol"] + policy["validation_rtol"] * high["sum_weighted_component_gradient_norms"]
    if high["gradient_linearity_error"] > allowed:
        raise ArithmeticError(f"Independent FP64 diagnostic identity failed at epoch {epoch}: {high['gradient_linearity_error']} > {allowed}")
    measured["float32_decomposition_status"] = "finite numerical estimates; direct-vs-component disagreement reported, not assumed exact"
    measured["high_precision_validation"] = dict(
        passed=True, policy_id=policy["policy_id"], gradient_linearity_error=high["gradient_linearity_error"],
        tolerance=allowed, relative_to_total=high["gradient_linearity_relative_to_total"],
        relative_to_sum_component_norms=high["gradient_linearity_relative_to_sum_component_norms"],
        raw_losses=high["raw_losses"], weighted_losses=high["weighted_losses"],
        sum_weighted_component_gradient_norms=high["sum_weighted_component_gradient_norms"],
        block_total_gradient_norms={name: values["total_gradient_norm"] for name, values in high["parameter_blocks"].items()},
        constants="promoted from the same float32 tensors and model state")
    return measured


def evaluate(core, model, case, observed, interval, epoch, protocol):
    """Truth is confined to this evaluation, not fitting/selection/stopping."""
    torch, np = core.torch, core.np
    device = next(model.parameters()).device
    fit_grid = np.linspace(*interval, protocol["symbolic"]["fit_points"], dtype=np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(fit_grid).to(device)
        d = model.diffusion(tensor).cpu().numpy(); r = model.reaction(tensor).cpu().numpy()
    if not np.isfinite(d).all() or not np.isfinite(r).all():
        raise FloatingPointError("Nonfinite checkpoint constitutive curves")
    expressions = core.fit_symbols(fit_grid, d, r,
                                   relative_tolerance=protocol["symbolic"]["relative_tolerance"],
                                   absolute_tolerance=protocol["symbolic"]["absolute_tolerance"])
    D = lambda z: np.interp(z, fit_grid, d)
    R = lambda z: np.interp(z, fit_grid, r)
    SD, SR = expressions[0].evaluate, expressions[1].evaluate
    grid = np.linspace(*interval, protocol["symbolic"]["evaluation_points"])
    neural = core.curve_metrics(case, grid, D, R, observed)
    frozen = core.curve_metrics(case, grid, SD, SR, observed)
    compression = {}
    for branch, truth, teacher, symbolic in (("D", case.diffusion(grid), D(grid), SD(grid)),
                                              ("R", case.reaction(grid), R(grid), SR(grid))):
        en = float(np.linalg.norm(teacher - truth)); ec = float(np.linalg.norm(symbolic - teacher)); es = float(np.linalg.norm(symbolic - truth))
        tn, truen = float(np.linalg.norm(teacher)), float(np.linalg.norm(truth))
        if not all(np.isfinite(v) for v in (en, ec, es, tn, truen)):
            raise FloatingPointError("Nonfinite symbolic evaluation")
        compression[branch] = dict(teacher_absolute_error=en, compression_absolute_error=ec,
                                   symbolic_absolute_error=es, BIR=es / en if en else None,
                                   bound=ec / en if en else None,
                                   observed_deviation=abs(es / en - 1) if en else None,
                                   compression_relative_neural=ec / tn if tn else None,
                                   zero_denominator=(en == 0))
    small = np.asarray([0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2])
    physical = dict(sampled_teacher_min_D=neural["min_D"], sampled_frozen_min_D=frozen["min_D"],
                    small_states=small, teacher_small_state_R=R(small), frozen_small_state_R=SR(small),
                    teacher_R0=float(R(np.array([0.]))[0]), frozen_R0=float(SR(np.array([0.]))[0]),
                    scope="sampled constitutive margins; no forward-trajectory or global-sign certificate implied")
    record = dict(epoch=epoch, neural=neural, frozen=frozen, compression=compression, physical=physical,
                  symbols=[core.dataclasses.asdict(expression) for expression in expressions],
                  checkpoint_policy="fixed epoch; never selected using true error")
    return record, dict(grid=fit_grid, D=d, R=r)


def record_environment(core, output_root, device, launch=None):
    root = Path(output_root)
    records = root / "environment"
    records.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + str(time.time_ns())
    packages = "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + "\n"
    package_hash = hashlib.sha256(packages.encode()).hexdigest()
    package_file = records / f"installed_packages_{package_hash}.txt"
    if package_file.exists() and file_hash(package_file) != package_hash:
        raise ValueError("Immutable environment snapshot has been changed")
    if not package_file.exists():
        package_file.write_text(packages)
    (records / "installed_packages.txt").write_text(packages)
    git_status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
    (records / "git_status.txt").write_text(git_status)
    (records / f"git_status_{stamp}.txt").write_text(git_status)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    metadata = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                    executable=sys.executable, platform=platform.platform(), git_head=git_head,
                    torch=core.torch.__version__, numpy=core.np.__version__, device=device,
                    command=INVOCATION, implementation_signature=implementation_signature())
    metadata.update(launch or {})
    metadata.update(launch_id=stamp, installed_packages_sha256=package_hash,
                    installed_packages_file=str(package_file.relative_to(root)),
                    git_status_file=f"environment/git_status_{stamp}.txt")
    if device == "cuda":
        metadata["gpu_name"] = core.torch.cuda.get_device_name()
        result = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (records / "nvidia-smi.txt").write_text(result.stdout)
        (records / f"nvidia-smi_{stamp}.txt").write_text(result.stdout)
        metadata["nvidia_smi_returncode"] = result.returncode
    core.json_write(records / "runtime.json", metadata)
    core.json_write(root / "launches" / f"launch_{stamp}.json", metadata)
    core.json_write(root / "LATEST_LAUNCH.json", dict(launch_id=stamp, record=f"launches/launch_{stamp}.json"))
    return metadata


def run_task(task, manifest, output_root, device, stop_epoch=None, retry_failed=False):
    if manifest.get("required_device") != "cuda" or device != "cuda":
        raise ValueError("This fresh GPU cohort requires CUDA; CPU fallback is forbidden")
    core = load_core(output_root, device)
    torch = core.torch
    torch.set_num_threads(1)
    target = task["max_epochs"] if stop_epoch is None else stop_epoch
    if target < 1 or target > task["max_epochs"]:
        raise ValueError(f"Requested prefix {target} is outside task budget {task['max_epochs']}")
    if task["classification"] != "runtime_pilot" and target not in task["evaluation_epochs"]:
        raise ValueError("Formal prefixes must use a declared evaluation epoch")
    out = Path(output_root) / "runs" / task["id"]
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = manifest["protocol"]
    cache_key = (str(output_root), task["case"], task["setting"], task["data_seed"])
    if cache_key not in INPUT_CACHE:
        INPUT_CACHE[cache_key] = load_input(core, output_root, task["case"], task["setting"], task["data_seed"])
    observed, interval, input_metadata = INPUT_CACHE[cache_key]
    if input_metadata["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("Input protocol does not match study manifest")
    torch.manual_seed(task["init_seed"])
    model = core.ReactionDiffusionClosure(input_range=interval).to(core.resolve_device())
    initial_sha = initial_model_hash(model)
    if not all(parameter.is_cuda for parameter in model.parameters()):
        raise ValueError("GPU cohort model parameters are not on CUDA")
    if initial_sha != task["paired_cpu_initial_model_sha256"]:
        raise ValueError("Fresh CUDA initialization does not match the paired CPU initial parameter state")
    identity = dict(study_id=manifest["study_id"], config_sha256=canonical_hash(task),
                    protocol_sha256=manifest["protocol_sha256"], implementation_signature=manifest["implementation_signature"],
                    observation_signature=input_metadata["observation_signature"], source_signature=core.source_signature(),
                    device=device, torch_version=torch.__version__, dtype="float32", initial_model_sha256=initial_sha)
    result_file = out / "result.json"
    if result_file.exists():
        prior = json.loads(result_file.read_text())
        if prior["identity"] != identity:
            raise ValueError(f"Prior identity mismatch for {task['id']}; never overwrite/relabel an older run")
        if prior["status"] == "failed" and not retry_failed:
            raise RuntimeError(f"Prior run failed: {task['id']}; inspect result.json before an explicit --retry-failed")
        if prior["status"] in ("complete", "prefix_complete") and prior["completed_epoch"] >= target:
            saved_path = out / f"checkpoint_{target}.pt"
            if not saved_path.exists() or str(target) not in prior["epochs"] or not (out / f"curves_{target}.npz").exists():
                raise ValueError("Completed cache is missing its requested authoritative checkpoint/exports")
            return task["id"], "cached"
        if prior["status"] == "failed":
            core.json_write(out / f"failure_{time.time_ns()}.json", prior)
    started, cpu_started = time.perf_counter(), time.process_time()
    result = dict(id=task["id"], config=task, identity=identity, classification=task["classification"],
                  status="running", requested_prefix=target, completed_epoch=0,
                  initial_model_sha256=initial_sha,
                  data_hash=input_metadata["data_hash"], observation_signature=input_metadata["observation_signature"],
                  epochs={}, gradient_diagnostics={}, started_utc=datetime.now(timezone.utc).isoformat())
    core.json_write(out / "config.json", task)
    core.json_write(result_file, result)
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol["optimizer"]["learning_rate"])
        ctx = context(core, model, observed, interval, protocol, task["weak_variant"], next(model.parameters()).device)
        tensors = [value for value in ctx.values() if isinstance(value, torch.Tensor)]
        if ctx["temporal"] is not None:
            tensors += list(ctx["temporal"])
        if not all(value.is_cuda for value in tensors):
            raise ValueError("GPU cohort objective tensors are not all on CUDA")
        history, diagnostics, evaluations = [], {}, {}
        epoch = 0; previous_seconds = 0.0
        checkpoints = sorted((p for p in out.glob("checkpoint_*.pt") if int(p.stem.split("_")[-1]) <= target),
                             key=lambda p: int(p.stem.split("_")[-1]))
        if checkpoints:
            saved, history = core.restore_checkpoint(checkpoints[-1], model, optimizer, identity)
            epoch = saved["epoch"]; previous_seconds = saved["elapsed_seconds"]
            diagnostics = saved["diagnostics"]; evaluations = saved["evaluations"]
        else:
            diagnostics["0"] = diagnose(core, ctx, task["weights"], 0)
        evaluation_epochs = set(task["evaluation_epochs"]) | {target}
        diagnostic_epochs = set(protocol["diagnostic_epochs"]) | evaluation_epochs
        parameters = tuple(model.parameters())
        while epoch < target:
            epoch += 1
            optimizer.zero_grad(set_to_none=True)
            ctx["support"].grad = None
            values = terms(core, ctx)
            loss = sum(task["weights"][k] * values[k] for k in task["weights"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite objective at epoch {epoch}")
            loss.backward(); core.assert_finite_state(model, gradients=True, epoch=epoch)
            grad = vector(core, [p.grad for p in parameters], parameters)
            before = torch.cat([p.detach().reshape(-1) for p in parameters]).double()
            optimizer.step(); core.assert_finite_state(model, optimizer, epoch=epoch)
            after = torch.cat([p.detach().reshape(-1) for p in parameters]).double()
            update = after - before
            history.append(dict(epoch=epoch, loss=float(loss.detach()), grad_norm=float(torch.linalg.vector_norm(grad)),
                                parameter_norm_before=float(torch.linalg.vector_norm(before)),
                                update_norm=float(torch.linalg.vector_norm(update)),
                                grad_update_cosine=cosine(core, grad, update),
                                **{k: float(v.detach()) for k, v in values.items()}))
            if epoch in diagnostic_epochs:
                diagnostics[str(epoch)] = diagnose(core, ctx, task["weights"], epoch, actual_update=update)
            if epoch in evaluation_epochs:
                evaluation, curves = evaluate(core, model, core.extra_case(task["case"]), observed, interval, epoch, protocol)
                evaluations[str(epoch)] = evaluation
                core.npz_write(out / f"curves_{epoch}.npz", **curves)
            if epoch % protocol["checkpoint_every"] == 0 or epoch in diagnostic_epochs or epoch == target:
                elapsed = previous_seconds + time.perf_counter() - started
                payload = core.checkpoint_payload(model, optimizer, history, epoch=epoch,
                                                  elapsed_seconds=elapsed, diagnostics=diagnostics,
                                                  evaluations=evaluations, **identity)
                core.torch_write(out / f"checkpoint_{epoch}.pt", payload)
                core.json_write(out / "history.json", history)
                core.json_write(out / "gradient_diagnostics.json", diagnostics)
        result.update(status="complete" if target == task["max_epochs"] else "prefix_complete",
                      completed_epoch=epoch, epochs=evaluations, gradient_diagnostics=diagnostics,
                      seconds=previous_seconds + time.perf_counter() - started,
                      cpu_seconds=time.process_time() - cpu_started,
                      stopped_by="declared fixed budget; no error-based stopping", convergence_claim=False)
        core.validate_history(history, epoch)
        if str(target) not in evaluations or str(target) not in diagnostics:
            raise ValueError("Missing final evaluation/gradient diagnostics")
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc(),
                      completed_epoch=locals().get("epoch", 0), seconds=time.perf_counter() - started,
                      cpu_seconds=time.process_time() - cpu_started)
        core.json_write(result_file, result)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    if device == "cuda":
        result["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    core.json_write(result_file, result)
    return task["id"], result["status"]


def selected_tasks(manifest, phase, ids=None, case=None, setting=None, seeds=None):
    tasks = manifest["tasks"]
    if phase == "longer":
        tasks = [t for t in tasks if t["longer"]]
    elif phase != "all":
        tasks = [t for t in tasks if t["phase"] == phase]
    if ids:
        requested = set(ids)
        unknown = requested - {t["id"] for t in manifest["tasks"]}
        if unknown:
            raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
        tasks = [t for t in tasks if t["id"] in requested]
    if case:
        tasks = [t for t in tasks if t["case"] == case]
    if setting:
        tasks = [t for t in tasks if t["setting"] == setting]
    if seeds is not None:
        tasks = [t for t in tasks if t["data_seed"] in seeds]
    if not tasks:
        raise ValueError("Empty task selection")
    return tasks


def write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to export an empty result table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(output_root, manifest):
    """Keep statuses and paired denominators; never select the best weight."""
    from statistics import mean, stdev
    from scipy.stats import t as student_t
    core = load_core(output_root, "cpu")
    metric_rows, gradient_rows, cosine_rows, status_rows = [], [], [], []
    paired_inputs = defaultdict(set)
    paired_initializations = defaultdict(set)
    expected = {t["id"]: t for t in manifest["tasks"]}
    for ident, task in expected.items():
        path = Path(output_root) / "runs" / ident / "result.json"
        if not path.exists():
            status_rows.append(dict(id=ident, case=task["case"], setting=task["setting"], seed=task["data_seed"], variant=task["variant"], status="not_run", completed_epoch=0))
            continue
        record = json.loads(path.read_text())
        if record["classification"] != "fresh_followup":
            raise ValueError("Pilot/historical record cannot enter formal study summaries")
        if record["identity"]["config_sha256"] != canonical_hash(task) or record["identity"]["implementation_signature"] != manifest["implementation_signature"]:
            raise ValueError(f"Result identity mismatch: {ident}")
        pair_key = (task["case"], task["setting"], task["data_seed"])
        paired_inputs[pair_key].add(record["data_hash"])
        paired_initializations[pair_key].add(record["initial_model_sha256"])
        status_rows.append(dict(id=ident, case=task["case"], setting=task["setting"], seed=task["data_seed"], variant=task["variant"], status=record["status"], completed_epoch=record["completed_epoch"]))
        if record["status"] not in ("complete", "prefix_complete"):
            continue
        common = dict(id=ident, case=task["case"], setting=task["setting"], seed=task["data_seed"],
                      init_seed=task["init_seed"], variant=task["variant"], phase=task["phase"],
                      weak_variant=task["weak_variant"], data_hash=record["data_hash"], classification="fresh_followup")
        for epoch, evaluation in record["epochs"].items():
            for method in ("neural", "frozen"):
                physical_prefix = "teacher" if method == "neural" else "frozen"
                metric_rows.append(dict(common, epoch=int(epoch), method=method,
                                        D=evaluation[method]["D"], R=evaluation[method]["R"],
                                        min_D=evaluation[method]["min_D"],
                                        R0=evaluation["physical"][physical_prefix + "_R0"],
                                        BIR_D=evaluation["compression"]["D"]["BIR"], BIR_R=evaluation["compression"]["R"]["BIR"],
                                        bound_D=evaluation["compression"]["D"]["bound"], bound_R=evaluation["compression"]["R"]["bound"]))
        for epoch, diagnostic in record["gradient_diagnostics"].items():
            for block, values in diagnostic["parameter_blocks"].items():
                for term, weight in task["weights"].items():
                    gradient_rows.append(dict(common, epoch=int(epoch), block=block, term=term, weight=weight,
                                              raw_loss=diagnostic["raw_losses"][term], weighted_loss=diagnostic["weighted_losses"][term],
                                              raw_gradient_norm=values["raw_gradient_norms"][term], weighted_gradient_norm=values["weighted_gradient_norms"][term],
                                              total_gradient_norm=values["total_gradient_norm"], cosine_with_total=values["cosine_with_total"][term],
                                              actual_update_norm=values["actual_update_norm"],
                                              raw_gradient_dot_actual_update=values["raw_gradient_dot_actual_update"][term],
                                              weighted_gradient_dot_actual_update=values["weighted_gradient_dot_actual_update"][term],
                                              cosine_with_actual_update=values["cosine_with_actual_update"][term]))
                for first, second_rows in values["raw_pairwise_cosines"].items():
                    for second, value in second_rows.items():
                        cosine_rows.append(dict(common, epoch=int(epoch), block=block, first_term=first, second_term=second,
                                                raw_cosine=value, weighted_cosine=values["weighted_pairwise_cosines"][first][second]))
    reports = Path(output_root) / "reports"
    write_csv(reports / "status.csv", status_rows)
    if metric_rows:
        write_csv(reports / "metrics_by_seed.csv", metric_rows)
        write_csv(reports / "loss_gradient_by_seed.csv", gradient_rows)
        write_csv(reports / "gradient_cosines_by_seed.csv", cosine_rows)
        groups = defaultdict(list)
        for row in metric_rows:
            groups[(row["case"], row["setting"], row["variant"], row["weak_variant"], row["epoch"], row["method"])].append(row)
        aggregates = []
        for (case, setting, variant, weak_variant, epoch, method), rows in sorted(groups.items()):
            if len({r["seed"] for r in rows}) != len(rows):
                raise ValueError("Duplicate seed in aggregate")
            for metric in ("D", "R", "BIR_D", "BIR_R", "bound_D", "bound_R"):
                vals = [r[metric] for r in rows if r[metric] is not None]
                if not vals:
                    continue
                sd = stdev(vals) if len(vals) > 1 else None
                half = float(student_t.ppf(0.975, len(vals) - 1)) * sd / len(vals)**0.5 if sd is not None else None
                aggregates.append(dict(case=case, setting=setting, variant=variant, weak_variant=weak_variant,
                                       epoch=epoch, method=method, metric=metric, n=len(vals),
                                       mean=mean(vals), sample_sd=sd,
                                       ci95_low=mean(vals) - half if half is not None else None,
                                       ci95_high=mean(vals) + half if half is not None else None,
                                       complete_five_seed_group=(len(vals) == 5)))
        write_csv(reports / "metrics_summary.csv", aggregates)
        paired = []
        lookup = {(r["case"], r["setting"], r["seed"], r["variant"], r["epoch"], r["method"]): r for r in metric_rows}
        for row in metric_rows:
            if row["variant"] == "default":
                continue
            key = (row["case"], row["setting"], row["seed"], "default", row["epoch"], row["method"])
            if key not in lookup:
                continue
            reference = lookup[key]
            if row["data_hash"] != reference["data_hash"]:
                raise ValueError("Paired contrast uses different observations")
            for metric in ("D", "R", "BIR_D", "BIR_R"):
                if row[metric] is not None and reference[metric] is not None:
                    paired.append(dict(case=row["case"], setting=row["setting"], seed=row["seed"], variant=row["variant"],
                                       epoch=row["epoch"], method=row["method"], metric=metric,
                                       difference=row[metric] - reference[metric], data_hash=row["data_hash"]))
        if paired:
            write_csv(reports / "paired_differences.csv", paired)
            paired_groups = defaultdict(list)
            for row in paired:
                paired_groups[(row["case"], row["setting"], row["variant"], row["epoch"], row["method"], row["metric"])].append(row["difference"])
            paired_summary = []
            for (case, setting, variant, epoch, method, metric), vals in sorted(paired_groups.items()):
                sd = stdev(vals) if len(vals) > 1 else None
                half = float(student_t.ppf(0.975, len(vals) - 1)) * sd / len(vals)**0.5 if sd is not None else None
                paired_summary.append(dict(case=case, setting=setting, variant=variant, epoch=epoch,
                                           method=method, metric=metric, n=len(vals),
                                           mean_paired_difference=mean(vals), sample_sd=sd,
                                           ci95_low=mean(vals) - half if half is not None else None,
                                           ci95_high=mean(vals) + half if half is not None else None))
            write_csv(reports / "paired_differences_summary.csv", paired_summary)
        interactions = []
        for row in metric_rows:
            if row["epoch"] != 150 or not row["variant"].startswith("joint_"):
                continue
            task = expected[row["id"]]
            first = task["factors"]["strong"]; second = task["factors"]["anchor"]
            def at(variant):
                return lookup.get((row["case"], row["setting"], row["seed"], variant, 150, row["method"]))
            a, b, default = at(f"strong_x{first:g}"), at(f"anchor_x{second:g}"), at("default")
            if any(value is None for value in (a, b, default)):
                continue
            if len({value["data_hash"] for value in (row, a, b, default)}) != 1:
                raise ValueError("Interaction contrast uses different observations")
            for metric in ("D", "R", "BIR_D", "BIR_R"):
                if all(value[metric] is not None for value in (row, a, b, default)):
                    interactions.append(dict(case=row["case"], setting=row["setting"], seed=row["seed"],
                                             strong_factor=first, anchor_factor=second, epoch=150,
                                             method=row["method"], metric=metric,
                                             interaction_contrast=row[metric] - a[metric] - b[metric] + default[metric],
                                             interpretation="second difference of the reported response, not a global interaction model"))
        if interactions:
            write_csv(reports / "strong_anchor_interactions_by_seed.csv", interactions)
    counts = dict(Counter(row["status"] for row in status_rows))
    if any(len(values) != 1 for values in paired_inputs.values()) or any(len(values) != 1 for values in paired_initializations.values()):
        raise ValueError("Weight/temporal configurations do not share actual observations and initial model states")
    core.json_write(reports / "SUMMARY_STATUS.json", dict(expected_tasks=len(expected), status_counts=counts,
                                                        successful_metric_rows=len(metric_rows),
                                                        input_hashes_paired=True, initial_model_hashes_paired=True,
                                                        selection_policy="All declared rows; no best-weight or best-seed filtering"))
    print(json.dumps(counts, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "summarize", "list"))
    parser.add_argument("--manifest", type=Path, default=STUDY / "MANIFEST.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--phase", choices=("core", "dense", "joint", "weak", "longer", "all"), default="all")
    parser.add_argument("--stop-epoch", type=int)
    parser.add_argument("--case", choices=("case_a", "case_exp"))
    parser.add_argument("--setting", choices=("clean", "noise3"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    if args.command == "summarize":
        summarize(args.output_root.resolve(), manifest)
        return
    if args.device != "cuda":
        raise ValueError("This isolated cohort requires --device cuda; no CPU fallback is allowed")
    tasks = selected_tasks(manifest, args.phase, args.ids, args.case, args.setting, args.seeds)
    if args.command == "list":
        print(json.dumps({"tasks": len(tasks), "phase": args.phase, "ids": [t["id"] for t in tasks]}, indent=2))
        return
    if args.workers < 1:
        raise ValueError("workers must be positive")
    root = args.output_root.resolve()
    if not (root / "INPUT_MANIFEST.json").exists():
        raise FileNotFoundError("Prepare all paired inputs first with prepare_inputs.py")
    core = load_core(root, args.device)
    immutable_manifest = root / "launches" / f"manifest_{canonical_hash(manifest)}.json"
    if immutable_manifest.exists() and json.loads(immutable_manifest.read_text()) != manifest:
        raise ValueError("Immutable launched manifest has been changed")
    if not immutable_manifest.exists():
        core.json_write(immutable_manifest, manifest)
    record_environment(core, root, args.device, launch=dict(
        classification="fresh_followup", workers=args.workers, phase=args.phase,
        stop_epoch=args.stop_epoch, selected_task_count=len(tasks), selected_task_ids=[t["id"] for t in tasks],
        protocol_sha256=manifest["protocol_sha256"], manifest_sha256=canonical_hash(manifest),
        source_manifest_file_sha256=file_hash(args.manifest),
        immutable_manifest_file=str(immutable_manifest.relative_to(root))))
    core.json_write(root / "LAUNCHED_MANIFEST.json", manifest)
    context = multiprocessing.get_context("spawn")
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(run_task, task, manifest, root, args.device, args.stop_epoch, args.retry_failed): task["id"] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                ident, status = future.result()
                print(ident, status, flush=True)
            except Exception:
                failures.append(futures[future]); traceback.print_exc()
    if failures:
        raise RuntimeError(f"{len(failures)} tasks failed; persisted records retain the original errors")


if __name__ == "__main__":
    main()
