"""Seven-coefficient, observation-only global controls for the six-term loss.

No neural fitting or GPU operation. Clean fields enter only explicitly named oracle
arms. All polynomial coefficients refer to monomials of the physical state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import LinearConstraint, minimize, nnls

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
DEFAULT_OUTPUT = REPO / "outputs" / STUDY.name
SHARED_OUTPUT = REPO / "outputs" / "reviewer-sensitivity-2026-10-15"
TERMS = ("weak", "rollout", "mass", "strong", "anchor", "reg")
DEFAULT_WEIGHTS = dict(weak=1., rollout=.1, mass=1., strong=1., anchor=1., reg=1e-4)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_core(output_root=DEFAULT_OUTPUT):
    os.environ["EAAI_WORK_ROOT"] = str(Path(output_root).resolve())
    os.environ["EAAI_INPUT_ROOT"] = str(Path(output_root).resolve())
    os.environ["EAAI_DEVICE"] = "cpu"
    sys.path.insert(0, str(REPO / "revision"))
    core = importlib.import_module("revision_core")
    if core.DEVICE_OPTION != "cpu" or core.ROOT != Path(output_root).resolve():
        raise RuntimeError("Use a fresh CPU process with the requested output root")
    core.torch.set_num_threads(1)
    return core


@dataclass
class Term:
    augmented_qr: np.ndarray
    rows: int
    normalization_count: int

    def loss(self, coefficients):
        residual = self.augmented_qr @ np.r_[coefficients, 1.]
        return float(residual @ residual)


def flux_columns(u, dx):
    """Conservative face-averaged diffusion, one column per D coefficient."""
    right = np.roll(u, -1, axis=-1)
    columns = []
    for degree in range(3):
        flux = .5 * (u**degree + right**degree) * (right-u) / dx
        columns.append((flux-np.roll(flux, 1, axis=-1))/dx)
    return np.stack(columns, axis=-1)


def design_rows(feature_u, target_u, x, t, value_range, core,
                weak_variant="space_weak"):
    """Return A,b,n with each loss exactly ||A theta-b||^2/n.

    Smoothness stacks D' and R' and divides by 256, not 512, because
    the repository adds their two separate means.
    """
    feature_u = np.asarray(feature_u, dtype=float)
    target_u = np.asarray(target_u, dtype=float)
    if feature_u.shape != target_u.shape or feature_u.ndim != 3:
        raise ValueError("Feature/target fields must share [batch,time,space]")
    if not np.isfinite(feature_u).all() or not np.isfinite(target_u).all():
        raise ValueError("Nonfinite field")
    dx = core.uniform_spacing(np.asarray(x), "space")
    dt = core.uniform_spacing(np.asarray(t), "time", minimum=3)
    phi, dphi = core.make_resolved_test_functions_1d(np.asarray(x), 4)
    mid = feature_u[:, 1:-1]
    ut = (target_u[:, 2:]-target_u[:, :-2])/(2*dt)
    ux = (np.roll(mid,-1,axis=-1)-np.roll(mid,1,axis=-1))/(2*dx)
    reaction = np.stack([feature_u**j for j in range(4)], axis=-1)
    full_rhs = np.concatenate([flux_columns(feature_u,dx), reaction], axis=-1)
    if weak_variant == "space_weak":
        dpart = np.stack([-dx*np.einsum("kx,btx->btk",dphi,mid**j*ux)
                          for j in range(3)], axis=-1)
        rpart = dx*np.einsum("kx,btxj->btkj",phi,reaction[:,1:-1])
        weak_a = np.concatenate([dpart,rpart],axis=-1)
        weak_b = dx*np.einsum("kx,btx->btk",phi,ut)
    elif weak_variant == "space_time_weak":
        psi,dpsi = core.make_time_test_functions_1d(np.asarray(t))
        quadrature = np.ones(len(t)); quadrature[[0,-1]] = .5
        fac = dt*dx/np.sqrt((len(t)-1)*dt)
        ux_all = (np.roll(feature_u,-1,axis=-1)-np.roll(feature_u,1,axis=-1))/(2*dx)
        dpart = np.stack([-fac*np.einsum("mt,kx,btx->bmk",psi*quadrature,dphi,
                                       feature_u**j*ux_all) for j in range(3)],axis=-1)
        rpart = fac*np.einsum("mt,kx,btxj->bmkj",psi*quadrature,phi,reaction)
        weak_a = np.concatenate([dpart,rpart],axis=-1)
        weak_b = -fac*np.einsum("mt,kx,btx->bmk",dpsi*quadrature,phi,target_u)
    else:
        raise ValueError(f"Unsupported weak variant {weak_variant}")
    mass_a = np.concatenate([np.zeros((*ut.shape[:2],3)),
                             reaction[:,1:-1].mean(axis=2)],axis=-1)
    support = np.linspace(*value_range,256)
    regularizer = np.zeros((512,7))
    for j in range(1,3): regularizer[:256,j] = j*support**(j-1)
    for j in range(1,4): regularizer[256:,3+j] = j*support**(j-1)
    anchor = np.array([[0.,0.,0.,1.,value_range[0],value_range[0]**2,value_range[0]**3]])
    pairs = dict(weak=(weak_a,weak_b),
                 rollout=(dt*full_rhs[:,:-1],target_u[:,1:]-target_u[:,:-1]),
                 mass=(mass_a,ut.mean(axis=-1)), strong=(full_rhs[:,1:-1],ut),
                 anchor=(anchor,np.zeros(1)), reg=(regularizer,np.zeros(512)))
    return {key:(a.reshape(-1,7),b.ravel(),256 if key=="reg" else b.size)
            for key,(a,b) in pairs.items()}


def compress_rows(rows):
    result = {}
    for key,(a,b,n) in rows.items():
        aug = np.column_stack([a,-b])/np.sqrt(n)
        result[key] = Term(np.linalg.qr(aug,mode="r"),len(b),n)
    return result


def weighted_system(terms,weights):
    if set(weights)!=set(TERMS) or any(not np.isfinite(v) or v<0 for v in weights.values()):
        raise ValueError("Supply all six finite nonnegative weights")
    if not any(weights.values()): raise ValueError("Zero objective")
    aug = np.vstack([np.sqrt(weights[k])*terms[k].augmented_qr for k in TERMS if weights[k]>0])
    return aug[:,:7],-aug[:,7]


def diffusion_constraint(points):
    points=np.asarray(points)
    return np.column_stack([np.ones(len(points)),points,points**2,np.zeros((len(points),4))])


def diffusion_minimum(coefficients,value_range):
    a,b,c=np.asarray(coefficients)[:3]
    candidates=[float(value_range[0]),float(value_range[1])]
    if c>0:
        vertex=-b/(2*c)
        if value_range[0]<vertex<value_range[1]:candidates.append(float(vertex))
    values=[float(a+b*x+c*x*x) for x in candidates]
    i=int(np.argmin(values))
    return dict(value=values[i],state=candidates[i],candidates=candidates,values=values)


def least_squares(terms,weights):
    a,b=weighted_system(terms,weights)
    scale=np.linalg.norm(a,axis=0)
    scale=np.where(scale>0,scale,1.)
    equilibrated=a/scale
    beta,_,rank,singular=np.linalg.lstsq(equilibrated,b,rcond=None)
    theta=beta/scale
    residual=a@theta-b
    gradient=2*a.T@residual
    denom=max(1.,2*np.linalg.norm(a)*np.linalg.norm(residual))
    stationarity_relative=float(np.linalg.norm(gradient)/denom)
    certified=np.isfinite(theta).all() and np.isfinite(stationarity_relative) and stationarity_relative<=1e-7
    info=dict(status="passed" if certified else "failed_certificate",
              coefficients=theta.tolist(),rank=int(rank),
              rank_deficient=bool(rank<7),unique_solution=bool(rank==7),
              stationarity_tolerance=1e-7,
              condition_scaled=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
              condition_unscaled=float(np.linalg.cond(a)),singular_values_scaled=singular.tolist(),
              objective=float(residual@residual),gradient_l2=float(np.linalg.norm(gradient)),
              stationarity_relative=stationarity_relative,
              losses={k:terms[k].loss(theta) for k in TERMS})
    return theta,info,(a,b,scale,equilibrated)


def positive_diffusion(terms,weights,value_range,*,minimum=1e-4,guard=1e-10):
    """Convex semi-infinite QP, certified by quadratic minimum and finite KKT.

    Each finite constraint is necessary for positivity on the interval. A point
    that is feasible on the whole interval and optimal for a finite relaxation
    is also globally optimal for the continuous constrained coefficient problem.
    """
    theta0,ls,system=least_squares(terms,weights)
    a,b,scale,equilibrated=system
    floor=minimum+guard
    min0=diffusion_minimum(theta0,value_range)
    if ls["status"]!="passed":
        return theta0,dict(status="failed_ls_certificate",least_squares=ls,minimum_diffusion=min0)
    if ls["rank"]<7:
        return theta0,dict(status="failed_rank_deficient",least_squares=ls,minimum_diffusion=min0)
    _,r=np.linalg.qr(equilibrated,mode="reduced")
    transform=solve_triangular(r,np.eye(7),lower=False)/scale[:,None]
    points=list(np.linspace(*value_range,33))
    attempts=[]
    z=np.zeros(7)
    solver_success=True
    if min0["value"]<floor:
        feasible=theta0.copy();feasible[:3]=[floor+.01,0.,0.]
        z=r@(scale*(feasible-theta0))
        for cut in range(51):
            c=diffusion_constraint(points)
            g=c@transform;h=floor-c@theta0
            sol=minimize(lambda v:.5*float(v@v),z,jac=lambda v:v,
                         constraints=[LinearConstraint(g,h,np.inf)],method="SLSQP",
                         options=dict(ftol=1e-13,maxiter=1000,disp=False))
            z=np.asarray(sol.x)
            theta=theta0+transform@z
            current=diffusion_minimum(theta,value_range)
            attempts.append(dict(cut=cut,success=bool(sol.success),status=int(sol.status),
                                 message=str(sol.message),iterations=int(sol.nit),
                                 continuous_minimum=current,objective_increase=float(z@z)))
            solver_success=bool(sol.success)
            # Stop on an optimizer failure, preserving its original state.
            if not sol.success or current["value"]>=floor-2e-12:break
            if min(abs(current["state"]-v) for v in points)<1e-13:break
            points.append(current["state"])
    theta=theta0+transform@z
    c=diffusion_constraint(points);g=c@transform;h=floor-c@theta0
    slack=c@theta-floor
    active=np.where(slack<=1e-8)[0]
    multipliers=np.zeros(len(points))
    dual_failure=None
    if len(active):
        try:
            multipliers[active],_=nnls(g[active].T,z,maxiter=10000,atol=1e-13)
        except (RuntimeError,ValueError) as exc:
            dual_failure=f"{type(exc).__name__}: {exc}"
    stationarity=z-g.T@multipliers
    complementarity=multipliers*slack
    dual=float(multipliers@h-.5*np.linalg.norm(g.T@multipliers)**2)
    primal=.5*float(z@z)
    gap=primal-dual
    current=diffusion_minimum(theta,value_range)
    stat_rel=float(np.linalg.norm(stationarity)/max(1.,np.linalg.norm(z)))
    comp_rel=float(np.max(np.abs(complementarity),initial=0)/max(1.,abs(primal)))
    gap_rel=float(abs(gap)/max(1.,abs(primal),abs(dual)))
    passed=(solver_success and dual_failure is None and current["value"]>=minimum-1e-10
            and stat_rel<=1e-7 and comp_rel<=1e-7 and gap_rel<=1e-7)
    residual=a@theta-b
    report=dict(status="passed" if passed else "failed_certificate",coefficients=theta.tolist(),
                objective=float(residual@residual),objective_increase=float(residual@residual)-ls["objective"],
                least_squares=ls,minimum_diffusion=current,required_minimum=minimum,guard=guard,
                used_optimizer=bool(attempts),solver_success=solver_success,attempts=attempts,
                constraint_points=points,slack_min=float(slack.min()),multipliers=multipliers.tolist(),
                active_indices=active.tolist(),dual_failure=dual_failure,
                kkt_stationarity_relative=stat_rel,kkt_complementarity_relative=comp_rel,
                primal_dual_gap=gap,primal_dual_gap_relative=gap_rel,
                losses={k:terms[k].loss(theta) for k in TERMS})
    return theta,report


def evaluation(theta,case,observed_u):
    grid=np.linspace(*case.value_range,256)
    d=np.polynomial.polynomial.polyval(grid,theta[:3])
    r=np.polynomial.polynomial.polyval(grid,theta[3:])
    truth_d=np.asarray(case.diffusion(grid));truth_r=np.asarray(case.reaction(grid))
    # Match the established function metric, which adds 1e-12 to the norm.
    relative=lambda pred,truth:float(np.linalg.norm(pred-truth)/(np.linalg.norm(truth)+1e-12))
    result=dict(E_D=relative(d,truth_d),E_R=relative(r,truth_r),
                minimum_diffusion=diffusion_minimum(theta,case.value_range),
                R0=float(theta[3]),R001=float(np.polynomial.polynomial.polyval(.01,theta[3:])),
                observed_state_min=float(np.min(observed_u)),observed_state_max=float(np.max(observed_u)))
    inside=(grid>=observed_u.min())&(grid<=observed_u.max())
    for name,mask in [("inside",inside),("outside",~inside)]:
        result[name+"_grid_count"]=int(mask.sum())
        result[name+"_E_D"]=relative(d[mask],truth_d[mask]) if mask.any() else None
        result[name+"_E_R"]=relative(r[mask],truth_r[mask]) if mask.any() else None
    return result


def read_input(path):
    path=Path(path)
    metadata=json.loads(path.with_suffix(".json").read_text())
    if file_hash(path)!=metadata["file_sha256"]:raise ValueError(f"Input hash mismatch: {path}")
    with np.load(path,allow_pickle=False) as data:
        arrays={key:data[key].copy() for key in data.files}
    if not np.array_equal(arrays["u0"],arrays["u"][:,0]):raise ValueError("u0 mismatch")
    if hashlib.sha256(arrays["clean_u"].tobytes()).hexdigest()!=metadata["clean_data_hash"]:
        raise ValueError("clean_u hash mismatch")
    return arrays,metadata


def observation_at_noise(arrays,level,seed,clipped=True):
    clean=arrays["clean_u"]
    if level==0:return clean.copy()
    noise=np.random.default_rng(seed).normal(0.,level*np.std(clean),size=clean.shape)
    values=clean+noise
    return np.clip(values,*arrays["value_range"]) if clipped else values
