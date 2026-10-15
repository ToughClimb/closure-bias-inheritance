"""Forward-observation coefficient fits with paired boundary-prior interventions.

Truth functions never enter fitting. All arms share a projected initial vector,
parameter scaling, observed snapshots, solver, and stopping rules. This module is
separate from the frozen numerical source of the completed reliability campaign.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from closure_discovery.data_generation.validation import uniform_spacing
from reliability_templates import FixedFamilyTemplate

ARMS = ('unanchored', 'soft_anchor', 'hard_anchor')


def setup(pair):
    templates = [FixedFamilyTemplate(e) for e in pair]
    if pair[1].family not in ('polynomial', 'rational'):
        raise ValueError('This paired protocol expects a reaction constant coefficient')
    if pair[1].family == 'rational' and pair[1].numerator_degree != 2:
        raise ValueError('Expected the selected quadratic numerator')
    split = len(pair[0].parameters)
    initial = np.r_[templates[0].parameters, templates[1].parameters]
    initial[split] = 0.
    scales = np.maximum(np.abs(initial), .01)
    if templates[1].evaluate(initial[split:], np.array([0.]))[0] != 0.:
        raise ValueError('Projected initial reaction is not exactly zero')
    return templates, split, initial, scales


def fit_anchor_control(observed, value_range, pair, arm, *, first_budget=40,
                       continuation_budget=80, soft_weight=1., max_rhs=20000):
    if arm not in ARMS: raise ValueError(arm)
    if min(first_budget, continuation_budget, max_rhs) < 1 or soft_weight <= 0:
        raise ValueError('Budgets and fixed soft penalty must be positive')
    u = np.asarray(observed['u'], dtype=float)
    times = np.asarray(observed['t'], dtype=float)
    if u.ndim != 3 or not np.isfinite(u).all(): raise ValueError('Expected finite observations')
    dx = uniform_spacing(observed['x'], 'space')
    uniform_spacing(times, 'time', minimum=3)
    if u.shape[1] != len(times) or u.shape[-1] != len(observed['x']):
        raise ValueError('Observation shape mismatch')
    templates, split, initial, scales = setup(pair)
    free = np.array([i for i in range(len(initial)) if arm != 'hard_anchor' or i != split])
    def decode(z):
        full = initial.copy(); full[free] = z
        return full
    indices = np.arange(0, len(times), max(1, (len(times)-1)//10))
    if indices[-1] != len(times)-1: indices = np.r_[indices, len(times)-1]
    saved = times[indices]-times[0]; target = u[:,indices]; shape = u[:,0].shape
    support = np.linspace(*value_range, 512)
    counters = dict(residual_evaluations=0, failed_trial_evaluations=0, solver_rhs_evaluations=0)
    def closures(p, z):
        return templates[0].evaluate(p[:split],z), templates[1].evaluate(p[split:],z)
    def prediction(p):
        d,r = closures(p,support)
        if not np.isfinite(d).all() or not np.isfinite(r).all() or d.min() <= 0:
            raise FloatingPointError('Invalid trial law on support')
        calls = 0
        def rhs(t,state):
            nonlocal calls
            calls += 1; counters['solver_rhs_evaluations'] += 1
            if calls > max_rhs: raise FloatingPointError('Per-trial RHS budget exhausted')
            z = state.reshape(shape); diffusion,reaction = closures(p,z)
            if not np.isfinite(diffusion).all() or not np.isfinite(reaction).all() or diffusion.min() <= 0:
                raise FloatingPointError('Invalid law on fitting trajectory')
            flux = .5*(diffusion+np.roll(diffusion,-1,-1))*(np.roll(z,-1,-1)-z)/dx
            result = ((flux-np.roll(flux,1,-1))/dx+reaction).ravel()
            if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite fitting RHS')
            return result
        sol = solve_ivp(rhs,(0,float(saved[-1])),u[:,0].ravel(),t_eval=saved,
                        method='RK45',rtol=2e-6,atol=1e-8,
                        max_step=min(5e-4,.35*dx*dx/float(d.max())))
        if not sol.success or sol.y.shape[1] != len(saved): raise FloatingPointError('Incomplete trial solve')
        result = sol.y.T.reshape(len(saved),*shape).transpose(1,0,2)
        if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite trial prediction')
        return result
    def residual(z):
        counters['residual_evaluations'] += 1; p = decode(z)
        d,r = closures(p,support)
        penalty = np.nan_to_num(np.minimum(d-1e-4,0)*100,nan=1e3,posinf=1e3,neginf=-1e3)/np.sqrt(len(support))
        try: data = (prediction(p)-target).ravel()/np.sqrt(target.size)
        except (FloatingPointError, ValueError, OverflowError):
            counters['failed_trial_evaluations'] += 1
            data = np.full(target.size,1e3/np.sqrt(target.size))
        anchor = np.sqrt(soft_weight)*templates[1].evaluate(p[split:],np.array([0.])) if arm == 'soft_anchor' else np.empty(0)
        return np.r_[data,penalty,anchor]
    start = time.perf_counter(); z = initial[free]; initial_residual = residual(z)
    phases = []
    for budget in (first_budget,continuation_budget):
        before = counters.copy()
        fitted = least_squares(residual,z,x_scale=scales[free],max_nfev=budget,
                               ftol=1e-8,gtol=1e-8,xtol=1e-8)
        phases.append(dict(nfev=int(fitted.nfev),status=int(fitted.status),success=bool(fitted.success),
                           optimality=float(fitted.optimality),message=str(fitted.message),max_nfev=budget,
                           **{key:counters[key]-before[key] for key in counters}))
        z = fitted.x
        if fitted.status != 0: break
    full = decode(z); final = prediction(full); d,r = closures(full,support)
    R0 = float(templates[1].evaluate(full[split:],np.array([0.]))[0])
    components = dict(observation_mse=float(np.mean((final-target)**2)),
                      diffusion_penalty=float(1e4*np.mean(np.minimum(d-1e-4,0)**2)),
                      anchor_penalty=float(soft_weight*R0*R0) if arm=='soft_anchor' else 0.)
    if arm=='hard_anchor' and R0 != 0.: raise AssertionError('Hard boundary constraint failed')
    return full,dict(arm=arm,initial_full_parameters=initial.tolist(),free_indices=free.tolist(),
        fixed_initial_scales=scales.tolist(),parameter_count=len(free),phases=phases,
        optimizer_success=bool(fitted.success),optimizer_status=int(fitted.status),continued=len(phases)>1,
        nfev=sum(p['nfev'] for p in phases),**counters,
        parameters_D=full[:split].tolist(),parameters_R=full[split:].tolist(),R0=R0,
        cost=.5*sum(components.values()),initial_cost=.5*float(np.sum(initial_residual**2)),
        components=components,soft_weight=soft_weight if arm=='soft_anchor' else None,
        seconds=time.perf_counter()-start,fit_snapshot_indices=indices.tolist(),truth_used_for_fitting=False,
        solver='RK45 rtol2e-6 atol1e-8, max_step min(5e-4,.35 dx^2/maxD)',
        rule='Same projected initial vector and fixed scales; continuation only for budget exhaustion')
