"""Known-viscosity Burgers flux discovery with strong/weak observation operators.

All fit operators receive sampled state observations. Analytical derivatives
are used only in tests and evaluation, never in a fit target.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.integrate import solve_ivp

NU = .02


def shock(x, t, mean=.25, jump=.5):
    return mean - jump * np.tanh(jump * (np.asarray(x)[None, :] - mean * np.asarray(t)[:, None]) / (2 * NU))


def bump(z):
    z = np.asarray(z)
    q = np.maximum(1 - z*z, 0.)
    return q**4, -8*z*q**3, -8*q**3 + 48*z*z*q**2


def observation_operator(x, t, observed, arm):
    """Return a fixed operator on saved noisy state values, no truth inputs."""
    dx, dt = x[1]-x[0], t[1]-t[0]
    if arm == 'strong':
        state = observed[1:-1, 1:-1].ravel()
        ux = ((observed[1:-1, 2:] - observed[1:-1, :-2])/(2*dx)).ravel()
        uxx = ((observed[1:-1, 2:] - 2*observed[1:-1, 1:-1] + observed[1:-1, :-2])/dx**2).ravel()
        ut = ((observed[2:, 1:-1] - observed[:-2, 1:-1])/(2*dt)).ravel()
        scale = np.sqrt(np.mean(ux*ux))
        return dict(state=state, multiplier=ux/scale, target=(NU*uxx-ut)/scale, arm=arm)
    if arm != 'weak':
        raise ValueError(arm)
    weights = np.ones_like(observed)*dx*dt
    weights[[0,-1], :] *= .5
    weights[:, [0,-1]] *= .5
    rows, targets = [], []
    for width in [.25, .4]:
        for center in np.linspace(-.35, .45, 9):
            px, px1, px2 = bump((x-center)/width)
            for tc in [.08, .15, .22]:
                pt, pt1, _ = bump((t-tc)/.079)
                phix = pt[:,None]*px1[None,:]/width
                phit = pt1[:,None]*px[None,:]/.079
                phixx = pt[:,None]*px2[None,:]/width**2
                row = (weights*phix).ravel()
                scale = np.sum(np.abs(row))
                rows.append(row/scale)
                targets.append(-np.sum(weights*observed*(phit+NU*phixx))/scale)
    return dict(state=observed.ravel(), matrix=np.array(rows), target=np.array(targets), arm=arm)


def polynomial(u, coefficients, derivative=0):
    u = np.asarray(u)
    result = np.zeros_like(u, dtype=float)
    for degree, coefficient in enumerate(coefficients, 1):
        if degree >= derivative:
            factor = float(np.prod(np.arange(degree-derivative+1, degree+1)))
            result += coefficient*factor*u**(degree-derivative)
    return result


def polynomial_design(operator, degree):
    state = operator['state']
    if operator['arm'] == 'strong':
        return np.column_stack([k*state**(k-1)*operator['multiplier'] for k in range(1, degree+1)])
    return operator['matrix'] @ np.column_stack([state**k for k in range(1, degree+1)])


def direct_fit(operator, degree):
    design = polynomial_design(operator, degree)
    scales = np.linalg.norm(design, axis=0)
    normalized = design/scales
    rank = int(np.linalg.matrix_rank(normalized))
    if rank < degree:
        raise ValueError('Polynomial flux design is not identifiable')
    coef = np.linalg.lstsq(normalized, operator['target'], rcond=None)[0]/scales
    return coef, dict(rank=rank, condition=float(np.linalg.cond(normalized)),
                      residual_mse=float(np.mean((design@coef-operator['target'])**2)))


class FluxNet(torch.nn.Module):
    """One hidden tanh layer and exact zero flux gauge. No convexity prior."""
    def __init__(self, seed=0, hidden=16):
        super().__init__()
        torch.manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(hidden, dtype=torch.float64)*.7)
        self.b = torch.nn.Parameter(torch.randn(hidden, dtype=torch.float64)*.2)
        self.v = torch.nn.Parameter(torch.randn(hidden, dtype=torch.float64)*.03)
        self.offset = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, u, derivative=0):
        u = u.reshape(-1)
        z = torch.tanh(u[:,None]*self.w+self.b)
        g = z@self.v+self.offset
        if derivative == 0:
            return u*g
        g1 = ((1-z*z)*self.w)@self.v
        if derivative == 1:
            return g+u*g1
        if derivative == 2:
            g2 = (-2*z*(1-z*z)*self.w**2)@self.v
            return 2*g1+u*g2
        raise ValueError(derivative)

    def numpy(self, u, derivative=0):
        a = np.asarray(u, dtype=float)
        with torch.no_grad():
            return self(torch.from_numpy(a.ravel()), derivative).numpy().reshape(a.shape)

    def numpy_evaluator(self):
        w,b,v,offset = [p.detach().numpy().copy() for p in (self.w,self.b,self.v,self.offset)]
        def evaluate(u, derivative=0):
            original = np.asarray(u)
            u = original.reshape(-1)
            z = np.tanh(u[:,None]*w+b)
            g = z@v+offset
            if derivative==0:
                out=u*g
            elif derivative==1:
                out=g+u*(((1-z*z)*w)@v)
            elif derivative==2:
                out=2*(((1-z*z)*w)@v)+u*((-2*z*(1-z*z)*w*w)@v)
            else:
                raise ValueError(derivative)
            return out.reshape(original.shape)
        return evaluate


def tensor_operator(operator):
    return {k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in operator.items()}


def loss(model, operator):
    if operator['arm']=='strong':
        prediction=model(operator['state'],1)*operator['multiplier']
    else:
        prediction=operator['matrix']@model(operator['state'])
    return torch.mean((prediction-operator['target'])**2)


def gradient_norm(model, operator):
    model.zero_grad(set_to_none=True)
    value=loss(model,operator)
    value.backward()
    gradient=np.sqrt(sum(float((p.grad*p.grad).sum()) for p in model.parameters()))
    return float(value.detach()),float(gradient)


def projection_geometry(grid, teacher, coefficients):
    weights=np.ones(len(grid));weights[[0,-1]]=.5;weights/=weights.sum()
    truth=.5*grid**2
    fitted=polynomial(grid,coefficients)
    n=teacher-truth;c=fitted-teacher;s=fitted-truth
    e_n=float(np.sum(weights*n*n));e_c=float(np.sum(weights*c*c));e_s=float(np.sum(weights*s*s))
    cross=float(np.sum(weights*n*c))
    return dict(teacher_squared_error=e_n,compression_squared_error=e_c,symbolic_squared_error=e_s,
                cross_term=2*cross,decomposition_residual=abs(e_s-e_n-e_c-2*cross),
                bir=float(np.sqrt(e_s/e_n)) if e_n>0 else None)


def frozen_fit(grid, teacher, degree):
    weights=np.ones(len(grid));weights[[0,-1]]=.5
    design=np.column_stack([grid**k for k in range(1,degree+1)])
    return np.linalg.lstsq(design*np.sqrt(weights[:,None]),teacher*np.sqrt(weights),rcond=None)[0]


def periodic_initial(x, kind):
    if kind=='interpolation':
        return .25+.35*np.sin(np.pi*x)+.08*np.sin(2*np.pi*x)
    if kind=='extrapolation':
        return .25+.65*np.sin(np.pi*x)
    raise ValueError(kind)


def conservative_rhs(u, dx, evaluate, viscosity=NU):
    """Centered conservative flux, central diffusion, independently time integrated.

    Viscosity, fine grids, adaptive DOP853 and explicit state/mesh checks are
    essential; this spatial operator is not asserted to be a monotone scheme.
    """
    flux=evaluate(u,0)
    return -(np.roll(flux,-1)-np.roll(flux,1))/(2*dx)+viscosity*(np.roll(u,-1)-2*u+np.roll(u,1))/dx**2


def rollout(evaluate, kind, n=128, horizon=1.5, rtol=2e-7, atol=2e-9, time_points=(.5,1.5)):
    x=np.linspace(-1,1,n,endpoint=False);dx=2/n
    initial=periodic_initial(x,kind)
    times=np.array([0.]+[v for v in time_points if v<=horizon])
    if times[-1] != horizon: times=np.append(times,horizon)
    lo,hi=float(initial.min()),float(initial.max())
    def rhs(t,u):
        if not np.all(np.isfinite(u)) or np.max(np.abs(u))>20:
            raise FloatingPointError('Nonfinite/unbounded rollout stage')
        return conservative_rhs(u,dx,evaluate)
    result=solve_ivp(rhs,(0,horizon),initial,method='DOP853',dense_output=True,rtol=rtol,atol=atol,
                     max_step=min(.01, .35*dx*dx/NU))
    if not result.success or abs(result.t[-1]-horizon)>1e-12 or not np.all(np.isfinite(result.y)):
        raise RuntimeError('Independent rollout failed: '+result.message)
    states=result.sol(times).T
    accepted=result.y.T
    mass=np.mean(accepted,axis=1)
    energy=np.mean(accepted*accepted,axis=1)
    violation=float(max(0.,np.max(accepted)-hi,lo-np.min(accepted)))
    return dict(x=x,t=times,u=states,mass_drift=float(np.max(abs(mass-mass[0]))),
                maximum_principle_violation=violation,energy_increase=float(np.max(np.diff(energy))),
                nfev=int(result.nfev),accepted_steps=len(result.t)-1,diagnostic_scope='all accepted adaptive endpoints',status='completed' if violation<.005 else 'state_bound_failed')


def cole_hopf_periodic(x, time, kind, n_reference=8192):
    """Independent exact heat-semigroup solution for quadratic Burgers flux.

    Positive viscosity and periodic Cole-Hopf transform, constant mean removed.
    Heat Fourier evolution is analytical in time. This is evaluation only.
    """
    if kind=='interpolation':
        mean,A,B=.25,.35,.08
    elif kind=='extrapolation':
        mean,A,B=.25,.65,0.
    else: raise ValueError(kind)
    z=np.arange(n_reference)*2/n_reference
    # Integral of A sin(pi x)+B sin(2pi x); periodic initial transform.
    initial=np.exp(A*np.cos(np.pi*z)/(2*NU*np.pi)+B*np.cos(2*np.pi*z)/(4*NU*np.pi))
    coefficients=np.fft.fft(initial)/n_reference
    frequencies=2*np.pi*np.fft.fftfreq(n_reference,d=2/n_reference)
    active=np.abs(coefficients)>1e-15*np.max(abs(coefficients))
    coefficients=coefficients[active]*np.exp(-NU*frequencies[active]**2*time)
    k=frequencies[active]
    points=np.asarray(x)-mean*time
    phases=np.exp(1j*points[:,None]*k)
    phi=(phases@coefficients).real
    derivative=(phases@(1j*k*coefficients)).real
    if np.min(phi)<=0:raise ValueError('Cole-Hopf transform lost positivity')
    return mean-2*NU*derivative/phi
