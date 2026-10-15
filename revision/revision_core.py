"""Observation-only fitting and explicit, unclipped validation for the EAAI revision.

The implementation is loaded from this repository. Truth functions are used only by data generation
and evaluation; optimization and symbolic selection receive observations/surrogates.
"""
from __future__ import annotations

import dataclasses
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import copy

from project_paths import ROOT, SOURCE_ROOT, INPUT_ROOT, DEVICE_OPTION, input_path, input_glob
sys.path.insert(0, str(SOURCE_ROOT / "src"))
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from closure_discovery.data_generation.cases import CASE_BUILDERS, ReactionDiffusionCase
from closure_discovery.models.mlp_closure import ReactionDiffusionClosure, _flux_divergence_periodic_torch
from closure_discovery.weak_form.test_functions import make_test_functions_1d, make_time_test_functions_1d
from closure_discovery.baselines.polynomial import (
    PolynomialClosure, fit_strong_form_polynomial_closure, fit_weak_form_polynomial_closure,
    _solve_ridge_least_squares,
)
from closure_discovery.symbolic.restricted_fit import (
    SymbolicExpression, fit_diffusion_candidates, fit_reaction_candidates, select_symbolic_expression,
)
from closure_discovery.data_generation.rd_solver_1d import (
    SimulationConfig1D, generate_dataset, rhs as numpy_rhs,
)
from closure_discovery.data_generation.observations import apply_observation_model
from closure_discovery.evaluation.metrics import relative_l2_error, summarize_excitation
from closure_discovery.data_generation.validation import uniform_spacing, positive_stride
from closure_discovery.pipelines.training_state import assert_finite_state, validate_history, checkpoint_payload, restore_checkpoint, atomic_torch_write as torch_write
from artifact_io import load_curves, load_history, json_write, npz_write, report_completed, source_signature, model_signature, validation_identity, cached_validation

DEFAULT_WEIGHTS = dict(weak=1.0, rollout=.1, mass=1.0, strong=1.0, anchor=1.0, reg=1e-4)


def resolve_device():
    requested='cuda' if DEVICE_OPTION=='auto' and torch.cuda.is_available() else ('cpu' if DEVICE_OPTION=='auto' else DEVICE_OPTION)
    if requested=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; install a CUDA-enabled Torch wheel and verify the local NVIDIA driver, or select --device cpu')
    if requested=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    return torch.device('cuda:0' if requested=='cuda' else 'cpu')


def process_pool(workers):
    import concurrent.futures as cf
    import multiprocessing
    # Fresh processes avoid inheriting a CUDA context. Path/device arguments are
    # inherited through explicit EAAI environment variables from project_paths.
    context=multiprocessing.get_context('spawn') if DEVICE_OPTION in ('cuda','auto') else None
    return cf.ProcessPoolExecutor(max_workers=workers,mp_context=context)


def observation_signature(observed, value_range):
    digest = hashlib.sha256()
    for key in ('u', 'x', 't', 'y'):
        if key in observed:
            array = np.ascontiguousarray(observed[key])
            digest.update(key.encode()); digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    digest.update(np.asarray(value_range, dtype=np.float64).tobytes())
    return digest.hexdigest()


def make_resolved_test_functions_1d(xs, num_modes=4):
    """Exclude unresolved/Nyquist sine pairs; preserve source tests at nx64.

    New runs use min(requested, floor((nx-1)/2)); historical unresolved
    controls remain archived inputs and are not rerun with invalid test functions.
    """
    resolved = min(num_modes, (len(xs)-1)//2)
    if resolved < 1:
        raise ValueError('At least three points are needed for resolved Fourier tests')
    return make_test_functions_1d(xs, num_modes=resolved)


def extra_case(name):
    if name in CASE_BUILDERS:
        return CASE_BUILDERS[name]()
    if name == "case_trig":
        return ReactionDiffusionCase(name, "D=.03+.012*sin(2*pi*u/1.8); R=u*(.8*cos(1.7*u)-.25)",
            lambda u: .03 + .012*np.sin(2*np.pi*u/1.8),
            lambda u: u*(.8*np.cos(1.7*u)-.25), (0., 1.8))
    if name == "case_tanh":
        return ReactionDiffusionCase(name, "D=.015+.02*(1+tanh(3*(u-.8)))/2; R=.7*u*(1-u/1.8)/(1+u*u)",
            lambda u: .015+.01*(1+np.tanh(3*(u-.8))),
            lambda u: .7*u*(1-u/1.8)/(1+u*u), (0., 1.8))
    raise ValueError(name)


def objective_terms(model, u, dx, dt, phi, grad_phi, support, anchor, variant="space_weak", temporal=None):
    """Same space-weak objective as source, sharing branch evaluations in autograd."""
    d = model.diffusion(u); r = model.reaction(u)
    mid = u[:, 1:-1]; ut = (u[:, 2:]-u[:, :-2])/(2*dt)
    ux = (torch.roll(mid,-1,-1)-torch.roll(mid,1,-1))/(2*dx)
    if variant == "space_weak":
        residual = (torch.einsum("kx,btx->btk",phi,ut)
            + torch.einsum("kx,btx->btk",grad_phi,d[:,1:-1]*ux)
            - torch.einsum("kx,btx->btk",phi,r[:,1:-1]))*dx
    else:
        psi, dpsi, weights = temporal
        ux_all = (torch.roll(u,-1,-1)-torch.roll(u,1,-1))/(2*dx)
        residual = (-torch.einsum("mt,kx,btx->bmk",dpsi*weights,phi,u)
            +torch.einsum("mt,kx,btx->bmk",psi*weights,grad_phi,d*ux_all)
            -torch.einsum("mt,kx,btx->bmk",psi*weights,phi,r))*dt*dx
        # psi has discrete L2 norm 1: divide by sqrt(T) for comparable units.
        if variant == 'space_time_weak':
            residual = residual / np.sqrt((u.shape[1]-1)*dt)
    full_rhs = _flux_divergence_periodic_torch(u,d,dx)+r
    return dict(weak=residual.square().mean(),
        rollout=(u[:,:-1]+dt*full_rhs[:,:-1]-u[:,1:]).square().mean(),
        mass=(ut.mean(-1)-r[:,1:-1].mean(-1)).square().mean(),
        strong=(ut-full_rhs[:,1:-1]).square().mean(),
        anchor=model.reaction(anchor).square().mean(), reg=model.smoothness_penalty(support))


def train(observed, value_range, *, epochs=150, init_seed=0, weights=None,
          variant="space_weak", checkpoints=(), out_dir=None, num_test_modes=4):
    epochs = positive_stride(epochs, 'epochs')
    requested = set(checkpoints) | {epochs}
    if any(cp < 1 or cp > epochs or int(cp) != cp for cp in requested):
        raise ValueError('Checkpoint epochs must be integers within the training budget')
    if variant not in ('space_weak', 'space_time_weak', 'space_time_rectangle', 'space_time_trapezoid'):
        raise ValueError(f'Unsupported weak variant: {variant}')
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(weights) != set(DEFAULT_WEIGHTS) or not all(np.isfinite(v) for v in weights.values()):
        raise ValueError('Loss weights must specify every objective term with finite values')
    torch.manual_seed(init_seed); torch.set_num_threads(1)
    device = resolve_device()
    model = ReactionDiffusionClosure(input_range=value_range).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    u = torch.as_tensor(observed['u'], dtype=torch.float32, device=device)
    if u.ndim != 3 or not torch.isfinite(u).all():
        raise ValueError('Training observations must be a finite [batch,time,space] array')
    dx = uniform_spacing(observed['x'], 'space')
    dt = uniform_spacing(observed['t'], 'time', minimum=3)
    phi, grad_phi = [torch.as_tensor(v, dtype=torch.float32, device=device)
                    for v in make_test_functions_1d(observed['x'], num_modes=num_test_modes)]
    support = torch.linspace(*value_range, 256, requires_grad=True, device=device)
    anchor = torch.tensor([value_range[0]], dtype=torch.float32, device=device)
    temporal = None
    if variant != 'space_weak':
        psi, dpsi = make_time_test_functions_1d(observed['t'])
        w = np.ones(u.shape[1])
        if variant != 'space_time_rectangle': w[[0, -1]] = .5
        temporal = tuple(torch.as_tensor(v, dtype=torch.float32, device=device) for v in (psi, dpsi, w))
    identity = dict(init_seed=init_seed, weights=weights, variant=variant, num_test_modes=num_test_modes,
                    observation_signature=observation_signature(observed, value_range), source_signature=source_signature())
    history = []; snapshots = {}; start = time.perf_counter(); start_epoch = 1; previous_seconds = 0.
    out = Path(out_dir) if out_dir is not None else None

    def snapshot(current, seconds):
        grid = np.linspace(*value_range, 512, dtype=np.float32)
        with torch.no_grad():
            tensor = torch.from_numpy(grid).to(device)
            d = current.diffusion(tensor).cpu().numpy(); r = current.reaction(tensor).cpu().numpy()
        if not np.isfinite(d).all() or not np.isfinite(r).all():
            raise FloatingPointError('Nonfinite final closure curves')
        return dict(grid=grid, D=d, R=r, seconds=seconds)

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        eligible = sorted((p for p in out.glob('checkpoint_*.pt') if int(p.stem.split('_')[-1]) <= epochs),
                          key=lambda p: int(p.stem.split('_')[-1]))
        if eligible:
            path = eligible[-1]
            saved, history = restore_checkpoint(path, model, optimizer, identity, legacy_history_path=out/'history.json')
            start_epoch = saved['epoch'] + 1; previous_seconds = saved.get('elapsed_seconds', 0.)
            snapshots = {int(k): v for k, v in saved.get('snapshots', {}).items()}
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
            try:
                for cp in sorted(requested):
                    if cp >= start_epoch or cp in snapshots: continue
                    if cp == saved['epoch']:
                        snapshots[cp] = snapshot(model, previous_seconds)
                    else:
                        prior = out/f'checkpoint_{cp}.pt'
                        if not prior.exists():
                            raise FileNotFoundError(f'Cannot reconstruct requested epoch {cp}; its checkpoint is missing')
                        historical = copy.deepcopy(model)
                        historical_optimizer = torch.optim.Adam(historical.parameters(), lr=1e-3)
                        prior_saved, _ = restore_checkpoint(prior, historical, historical_optimizer, identity,
                                                            legacy_history=history[:cp])
                        snapshots[cp] = snapshot(historical, prior_saved.get('elapsed_seconds', 0.))
            finally:
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng, device)
            # Checkpoints are authoritative; exports can be safely reconstructed.
            json_write(out/'history.json', history)
            for cp, values in snapshots.items(): npz_write(out/f'curves_{cp}.npz', **values)
    for epoch in range(start_epoch, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        terms = objective_terms(model, u, dx, dt, phi, grad_phi, support, anchor, variant, temporal)
        loss = sum(weights[k] * v for k, v in terms.items())
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        loss.backward(); assert_finite_state(model, gradients=True, epoch=epoch)
        grad = float(torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters() if p.grad is not None)))
        optimizer.step(); assert_finite_state(model, optimizer, epoch=epoch)
        history.append(dict(epoch=epoch, loss=float(loss.detach()), grad_norm=grad,
                            **{k: float(v.detach()) for k, v in terms.items()}))
        elapsed = previous_seconds + time.perf_counter() - start
        if epoch in requested: snapshots[epoch] = snapshot(model, elapsed)
        if out is not None and (epoch in requested or epoch % 50 == 0):
            payload = checkpoint_payload(model, optimizer, history, snapshots=snapshots,
                                         epoch=epoch, elapsed_seconds=elapsed, device=str(device), **identity)
            torch_write(out/f'checkpoint_{epoch}.pt', payload)
            if epoch in requested: npz_write(out/f'curves_{epoch}.npz', **snapshots[epoch])
            json_write(out/'history.json', history)
    validate_history(history, epochs)
    if not requested.issubset(snapshots):
        raise ValueError(f'Training is incomplete: missing requested snapshots {sorted(requested-set(snapshots))}')
    return model, history, {cp: snapshots[cp] for cp in sorted(requested)}


def fit_symbols(grid, d, r, relative_tolerance=.05, absolute_tolerance=1e-8):
    dc=fit_diffusion_candidates(grid,d); rc=fit_reaction_candidates(grid,r)
    return (select_symbolic_expression(dc,relative_tolerance=relative_tolerance,absolute_tolerance=absolute_tolerance),
            select_symbolic_expression(rc,relative_tolerance=relative_tolerance,absolute_tolerance=absolute_tolerance))


def eval_expr(family, parameters, u, numerator_degree=None, denominator_degree=None):
    return SymbolicExpression(family,tuple(parameters),"",0,0,numerator_degree,denominator_degree).evaluate(u)


def refit_symbols(observed, value_range, symbols, mode="weak", max_nfev=100,num_test_modes=4):
    """Refit only coefficients/rates of frozen selected families to observations.

    No true closure values enter residuals. Includes a sampled positivity penalty;
    final violations are recorded and strict rollouts reject them. No truth selection.
    """
    u=np.asarray(observed["u"],dtype=float); dx=float(observed["x"][1]-observed["x"][0]); dt=float(observed["t"][1]-observed["t"][0])
    phi,gphi=make_test_functions_1d(observed["x"],num_modes=num_test_modes); mid=u[:,1:-1]; ut=(u[:,2:]-u[:,:-2])/(2*dt)
    ux=(np.roll(mid,-1,-1)-np.roll(mid,1,-1))/(2*dx)
    split=len(symbols[0].parameters); p0=np.array(symbols[0].parameters+symbols[1].parameters)
    scale=np.maximum(np.abs(p0),.01); support=np.linspace(*value_range,512)
    def closure(p):
        return (dataclasses.replace(symbols[0],parameters=tuple(p[:split])).evaluate,
                dataclasses.replace(symbols[1],parameters=tuple(p[split:])).evaluate)
    def residual(p):
        D,R=closure(p)
        with np.errstate(all="ignore"):
            if mode == "weak":
                res=(np.einsum("kx,btx->btk",phi,ut)+np.einsum("kx,btx->btk",gphi,D(mid)*ux)
                    -np.einsum("kx,btx->btk",phi,R(mid)))*dx
            elif mode == "data":
                # Central finite-volume equation fit on observations (derivative fit).
                right=np.roll(mid,-1,-1); flux=.5*(D(mid)+D(right))*(right-mid)/dx
                res=ut-(flux-np.roll(flux,1,-1))/dx-R(mid)
            else: raise ValueError(mode)
            # Mean residual objective plus explicit positivity penalty.
            pos=np.minimum(D(support)-1e-4,0)*100
            vec=np.r_[res.ravel()/np.sqrt(res.size),pos/np.sqrt(pos.size)]
        return np.nan_to_num(vec,nan=1e10,posinf=1e10,neginf=-1e10)
    start=time.perf_counter()
    fit=least_squares(residual,p0,x_scale=scale,max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    D,R=closure(fit.x)
    info=dict(mode=mode,nfev=fit.nfev,status=int(fit.status),success=bool(fit.success),
              optimality=float(fit.optimality),cost=float(fit.cost),initial_cost=float(.5*np.sum(residual(p0)**2)),
              seconds=time.perf_counter()-start,family_D=symbols[0].family,family_R=symbols[1].family,
              parameters_D=fit.x[:split].tolist(),parameters_R=fit.x[split:].tolist(),
              numerator_degree_D=symbols[0].numerator_degree,denominator_degree_D=symbols[0].denominator_degree,
              numerator_degree_R=symbols[1].numerator_degree,denominator_degree_R=symbols[1].denominator_degree,
              min_diffusion=float(np.min(D(support))))
    return D,R,info


def spacetime_polynomial(observed, value_range, diffusion_degree=2,reaction_degree=3,num_test_modes=4):
    """Full time weak polynomial baseline with trapezoidal endpoint quadrature."""
    u=np.asarray(observed["u"],dtype=float); dx=float(observed["x"][1]-observed["x"][0]);dt=float(observed["t"][1]-observed["t"][0])
    phi,gphi=make_test_functions_1d(observed["x"],num_modes=num_test_modes);psi,dpsi=make_time_test_functions_1d(observed["t"])
    w=np.ones(u.shape[1]);w[[0,-1]]=.5; psi=psi*w; dpsi=dpsi*w
    ux=(np.roll(u,-1,-1)-np.roll(u,1,-1))/(2*dx)
    time_term=-np.einsum("mt,kx,btx->bmk",dpsi,phi,u)*dt*dx
    cols=[np.einsum("mt,kx,btx->bmk",psi,gphi,(u**i)*ux)*dt*dx for i in range(diffusion_degree+1)]
    cols += [-np.einsum("mt,kx,btx->bmk",psi,phi,u**i)*dt*dx for i in range(reaction_degree+1)]
    design=np.stack(cols,axis=-1).reshape(-1,len(cols))
    co,mse=_solve_ridge_least_squares(design,-time_term.ravel(),ridge=1e-8)
    return PolynomialClosure(co[:diffusion_degree+1],co[diffusion_degree+1:],value_range,"spacetime_poly","Time weak polynomial"),mse


def curve_metrics(case,grid,D,R,observed):
    td=case.diffusion(grid); tr=case.reaction(grid); pd=D(grid);pr=R(grid)
    lo=float(np.min(observed["u"]));hi=float(np.max(observed["u"]));vis=(grid>=lo)&(grid<=hi)
    out=dict(D=relative_l2_error(td,pd),R=relative_l2_error(tr,pr),min_D=float(np.min(pd)),
             visited_min=lo,visited_max=hi,visited_grid_count=int(vis.sum()),support_min=float(grid[0]),support_max=float(grid[-1]),support_points=len(grid))
    for label,mask in [("visited",vis),("extrapolation",~vis)]:
        for branch,t,p in [("D",td,pd),("R",tr,pr)]:
            out[branch+"_"+label]=relative_l2_error(t[mask],p[mask]) if mask.any() else None
    return out


def rollout_targets(t_final, save_dt):
    if not np.isfinite(t_final) or t_final < 0 or not np.isfinite(save_dt) or save_dt <= 0:
        raise ValueError('Rollout horizon must be nonnegative and save_dt positive and finite')
    targets = np.arange(save_dt, t_final + save_dt*.1, save_dt)
    targets = targets[targets <= t_final + 1e-12]
    if t_final > 0:
        if len(targets) and np.isclose(targets[-1], t_final, rtol=0, atol=1e-12):
            targets[-1] = t_final
        else:
            targets = np.r_[targets[targets < t_final], t_final]
    return targets


def strict_rollout(D,R,u0,*,nx=None,t_final=.5,save_dt=.01,max_dt=1e-4,state_clip=None,adaptive=True,allow_nonpositive=False,state_range=None,range_tolerance=1e-10):
    """Periodic RK4; numerical completion and state-domain diagnostics are separate.

    Legacy ``status='stable'`` denotes completion under the numerical guards.
    Range diagnostics never clip or stop an otherwise valid numerical trajectory.
    Accepted states and RK trial stages are tracked separately, including the
    initial state. Legacy state_min/state_max retain their terminal meanings.
    """
    state=np.array(u0,dtype=float)
    if state.ndim < 1 or state.shape[-1] < 2 or not np.isfinite(state).all():
        raise ValueError('Initial states must have a finite spatial axis with at least two cells')
    cells=state.shape[-1]
    if nx is not None and positive_stride(nx,'nx') != cells:
        raise ValueError(f'nx={nx} does not match the initial state spatial size {cells}')
    if not np.isfinite(max_dt) or max_dt<=0:raise ValueError('max_dt must be positive and finite')
    targets=rollout_targets(t_final,save_dt)
    if state_clip is not None and (len(state_clip)!=2 or not np.isfinite(state_clip).all() or state_clip[0]>=state_clip[1]):
        raise ValueError('state_clip must be an ordered finite pair')
    if state_range is not None and (len(state_range)!=2 or not np.isfinite(state_range).all() or state_range[0]>=state_range[1]):
        raise ValueError('state_range must be an ordered finite pair')
    if not np.isfinite(range_tolerance) or range_tolerance<0:raise ValueError('range_tolerance must be nonnegative and finite')
    audit=dict(accepted_min=float(state.min()),accepted_max=float(state.max()),
        stage_min=float(state.min()),stage_max=float(state.max()),first_accepted_violation=None,
        first_stage_violation=None,accepted_nonfinite=False,stage_nonfinite=False)
    def track(values,time,stage,accepted=False):
        key='accepted' if accepted else 'stage'
        if not np.isfinite(values).all():
            audit[key+'_nonfinite']=True
            return
        lo=float(values.min());hi=float(values.max())
        audit[key+'_min']=min(audit[key+'_min'],lo);audit[key+'_max']=max(audit[key+'_max'],hi)
        if state_range is not None and (lo<state_range[0]-range_tolerance or hi>state_range[1]+range_tolerance):
            first='first_'+key+'_violation'
            if audit[first] is None:
                bad=(values<state_range[0]-range_tolerance)|(values>state_range[1]+range_tolerance)
                index=tuple(int(i) for i in np.argwhere(bad)[0])
                audit[first]=dict(time=float(time),stage=stage,index=index,state_min=lo,state_max=hi)
    track(state,0.,'initial',accepted=True);track(state,0.,'initial')
    dx=1/cells;t=0.;snapshots=[state.copy()];times=[0.];steps=0;min_d=float('inf');max_d=0.;clip_count=0;clipped_entries=0
    def constitutive(u):
        d=np.broadcast_to(np.asarray(D(u),dtype=float),u.shape)
        r=np.broadcast_to(np.asarray(R(u),dtype=float),u.shape)
        if not np.isfinite(d).all() or not np.isfinite(r).all() or (np.min(d)<=0 and not allow_nonpositive):
            raise FloatingPointError('invalid constitutive values')
        return d,r
    def rhs(u,time=0.,stage='initial'):
        nonlocal min_d,max_d
        track(u,time,stage)
        d,r=constitutive(u);min_d=min(min_d,float(d.min()));max_d=max(max_d,float(d.max()))
        flux=.5*(d+np.roll(d,-1,-1))*(np.roll(u,-1,-1)-u)/dx
        return (flux-np.roll(flux,1,-1))/dx+r
    status='stable';reason=None
    try:
        rhs(state)  # Validate the initial law even for a zero or sub-save horizon.
        for target in targets:
            while t<target:
                d,_=constitutive(state)
                h=min(max_dt,.35*dx*dx/max(float(np.max(d)),1e-8),target-t) if adaptive else min(max_dt,target-t)
                if t+h==t:raise FloatingPointError('time step underflow')
                k1=rhs(state,t,'k1');k2=rhs(state+h*k1/2,t+h/2,'k2');k3=rhs(state+h*k2/2,t+h/2,'k3');k4=rhs(state+h*k3,t+h,'k4')
                state=state+h*(k1+2*k2+2*k3+k4)/6;t+=h;steps+=1
                track(state,t,'accepted_before_clip',accepted=True)
                if state_clip is not None:
                    changed=(state<state_clip[0])|(state>state_clip[1]);clip_count+=int(changed.any());clipped_entries+=int(changed.sum())
                    state=np.clip(state,*state_clip)
                if not np.isfinite(state).all() or np.max(np.abs(state))>10:raise FloatingPointError('divergent state')
                if steps>200000:raise FloatingPointError('step budget')
            snapshots.append(state.copy());times.append(float(target))
    except (FloatingPointError,OverflowError,ValueError) as error:
        status='failed';reason=str(error)
    reached=bool(status=='stable' and times[-1]==float(t_final))
    return np.array(times),np.array(snapshots),dict(status=status,reason=reason,steps=steps,
        min_D=min_d if np.isfinite(min_d) else None,max_D=max_d,
        state_min=float(np.min(state)) if np.isfinite(state).all() else None,
        state_max=float(np.max(state)) if np.isfinite(state).all() else None,
        clip_steps=clip_count,clipped_cell_steps=clipped_entries,state_clip=state_clip,adaptive=adaptive,
        nx=cells,requested_final_time=float(t_final),reached_final_time=reached,numerical_completion=reached,
        accepted_state_min=None if audit['accepted_nonfinite'] else audit['accepted_min'],
        accepted_state_max=None if audit['accepted_nonfinite'] else audit['accepted_max'],
        rk_stage_state_min=None if audit['stage_nonfinite'] else audit['stage_min'],
        rk_stage_state_max=None if audit['stage_nonfinite'] else audit['stage_max'],
        accepted_state_range_valid=None if state_range is None else audit['first_accepted_violation'] is None and not audit['accepted_nonfinite'],
        rk_stage_state_range_valid=None if state_range is None else audit['first_stage_violation'] is None and not audit['stage_nonfinite'],
        first_accepted_range_violation=audit['first_accepted_violation'],first_rk_stage_range_violation=audit['first_stage_violation'],
        state_range=state_range,range_tolerance=range_tolerance,
        range_tracking='Initial state, every accepted pre-clipping step and every RK RHS stage; legacy extrema remain terminal')


def observation_diagnostics(raw,observed,case,seed,noise):
    clean=raw["u"][:,::int(round((observed['t'][1]-observed['t'][0])/(raw['t'][1]-raw['t'][0]))),
                         ::int(round((observed['x'][1]-observed['x'][0])/(raw['x'][1]-raw['x'][0])))]
    sigma=noise*np.std(clean);rng=np.random.default_rng(seed);pre=clean+rng.normal(0,sigma,clean.shape)
    post=observed["u"];dt=float(observed["t"][1]-observed["t"][0]);err=post-clean
    deriv=(err[:,2:]-err[:,:-2])/(2*dt)
    return dict(sigma=float(sigma),saved_dt=dt,expected_derivative_std=float(sigma/(np.sqrt(2)*dt)),
        observed_derivative_std=float(np.std(deriv)),noise_mean=float(err.mean()),
        clipped_fraction=float(np.mean(pre!=post)) if noise else 0.,
        clipping_mean_shift=float(np.mean(post-pre)),clip_range=case.value_range,
        excitation=dataclasses.asdict(summarize_excitation(post,float(observed['x'][1]-observed['x'][0]),case.value_range)))
