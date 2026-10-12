"""Stronger isotropic 2D validation, not an anisotropic/3D generality claim."""
import argparse
import concurrent.futures as cf
import resource
import traceback
from revision_core import *
from closure_discovery.data_generation.rd_solver_2d import SimulationConfig2D, generate_dataset as generate_2d, random_fourier_initial_condition_2d
from closure_discovery.pipelines.train_2d_closure import apply_observation_model_2d, ObservationConfig2D
from closure_discovery.weak_form.test_functions_2d import make_test_functions_2d
from closure_discovery.weak_form.weak_residual_2d import flux_divergence_periodic_2d
from closure_discovery.baselines.polynomial import _solve_ridge_least_squares


def train_2d(observed,value_range,epochs=150,seed=0,out_dir=None,num_test_modes=2):
    epochs=positive_stride(epochs,'epochs')
    device=resolve_device()
    torch.set_num_threads(1);torch.manual_seed(seed);model=ReactionDiffusionClosure(input_range=value_range).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);u=torch.tensor(observed['u'],dtype=torch.float32,device=device)
    dx=uniform_spacing(observed['x'],'x');dy=uniform_spacing(observed['y'],'y');dt=uniform_spacing(observed['t'],'time',minimum=3)
    if not torch.isfinite(u).all():raise ValueError('2D observations must be finite')
    phi,gx,gy=[torch.tensor(v,dtype=torch.float32,device=device) for v in make_test_functions_2d(observed['x'],observed['y'],num_modes=num_test_modes)]
    support=torch.linspace(*value_range,256,requires_grad=True,device=device);anchor=torch.tensor([value_range[0]],device=device)
    mid=u[:,1:-1];ut=(u[:,2:]-u[:,:-2])/(2*dt)
    ux=(torch.roll(mid,-1,-2)-torch.roll(mid,1,-2))/(2*dx);uy=(torch.roll(mid,-1,-1)-torch.roll(mid,1,-1))/(2*dy)
    history=[];start=time.perf_counter();start_epoch=1;previous_seconds=0.
    identity=dict(observation_signature=observation_signature(observed,value_range),seed=seed,
                  num_test_modes=num_test_modes,weights=dict(DEFAULT_WEIGHTS),source_signature=source_signature())
    if out_dir is not None:
        out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
        eligible=sorted((p for p in out.glob('checkpoint_*.pt') if int(p.stem.split('_')[-1])<=epochs),key=lambda p:int(p.stem.split('_')[-1]))
        if eligible:
            saved,history=restore_checkpoint(eligible[-1],model,optimizer,identity,legacy_history_path=out/'history.json')
            start_epoch=saved['epoch']+1;previous_seconds=saved['elapsed_seconds']
    for epoch in range(start_epoch,epochs+1):
        optimizer.zero_grad(set_to_none=True);d=model.diffusion(u);r=model.reaction(u)
        res=(torch.einsum('kxy,btxy->btk',phi,ut)+torch.einsum('kxy,btxy->btk',gx,d[:,1:-1]*ux)
            +torch.einsum('kxy,btxy->btk',gy,d[:,1:-1]*uy)-torch.einsum('kxy,btxy->btk',phi,r[:,1:-1]))*dx*dy
        full_rhs=flux_divergence_periodic_2d(u,d,dx,dy)+r
        terms=dict(weak=res.square().mean(),rollout=(u[:,:-1]+dt*full_rhs[:,:-1]-u[:,1:]).square().mean(),
            mass=(ut.mean((-2,-1))-r[:,1:-1].mean((-2,-1))).square().mean(),strong=(ut-full_rhs[:,1:-1]).square().mean(),
            anchor=model.reaction(anchor).square().mean(),reg=model.smoothness_penalty(support))
        loss=sum(DEFAULT_WEIGHTS[k]*v for k,v in terms.items())
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();assert_finite_state(model,gradients=True,epoch=epoch)
        optimizer.step();assert_finite_state(model,optimizer,epoch=epoch)
        history.append(dict(epoch=epoch,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()}))
        if out_dir is not None and (epoch%50==0 or epoch==epochs):
            torch_write(out/f'checkpoint_{epoch}.pt',checkpoint_payload(model,optimizer,history,
                epoch=epoch,elapsed_seconds=previous_seconds+time.perf_counter()-start,device=str(device),**identity))
            json_write(out/'history.json',history)
    validate_history(history,epochs)
    if out_dir is not None:json_write(out/'history.json',history)
    return model,history,previous_seconds+time.perf_counter()-start


def polynomial_2d(observed,value_range,method='weak',num_test_modes=2):
    u=observed['u'];mid=u[:,1:-1];dt=observed['t'][1]-observed['t'][0];dx=observed['x'][1]-observed['x'][0];dy=observed['y'][1]-observed['y'][0]
    ut=(u[:,2:]-u[:,:-2])/(2*dt);phi,gx,gy=make_test_functions_2d(observed['x'],observed['y'],num_modes=num_test_modes)
    cols=[]
    for p in range(3):
        if method=='weak':
            ux=(np.roll(mid,-1,-2)-np.roll(mid,1,-2))/(2*dx);uy=(np.roll(mid,-1,-1)-np.roll(mid,1,-1))/(2*dy)
            cols.append((np.einsum('kxy,btxy->btk',gx,mid**p*ux)+np.einsum('kxy,btxy->btk',gy,mid**p*uy))*dx*dy)
        else:
            fluxx=.5*(mid**p+np.roll(mid**p,-1,-2))*(np.roll(mid,-1,-2)-mid)/dx
            fluxy=.5*(mid**p+np.roll(mid**p,-1,-1))*(np.roll(mid,-1,-1)-mid)/dy
            cols.append((fluxx-np.roll(fluxx,1,-2))/dx+(fluxy-np.roll(fluxy,1,-1))/dy)
    for p in range(4):cols.append(-np.einsum('kxy,btxy->btk',phi,mid**p)*dx*dy if method=='weak' else mid**p)
    target=-np.einsum('kxy,btxy->btk',phi,ut)*dx*dy if method=='weak' else ut
    design=np.stack(cols,-1).reshape(-1,7);co,mse=_solve_ridge_least_squares(design,target.ravel(),ridge=1e-8)
    return PolynomialClosure(co[:3],co[3:],value_range,method+'_poly_2d','2D polynomial'),mse


def rollout_2d(D,R,u0,t_final=.3,dt=1e-4,save_dt=.01):
    u=np.array(u0,dtype=float)
    if u.ndim<2 or min(u.shape[-2:])<2 or not np.isfinite(u).all():raise ValueError('Invalid 2D initial state')
    if not np.isfinite(dt) or dt<=0:raise ValueError('dt must be finite and positive')
    targets=rollout_targets(t_final,save_dt)
    dx=1/u.shape[-2];dy=1/u.shape[-1];out=[u.copy()];times=[0.];t=0.;steps=0;status='stable';reason=None
    def constitutive(z):
        d=np.broadcast_to(np.asarray(D(z),dtype=float),z.shape);r=np.broadcast_to(np.asarray(R(z),dtype=float),z.shape)
        if np.min(d)<=0 or not np.all(np.isfinite(d)) or not np.all(np.isfinite(r)):raise FloatingPointError('invalid closure')
        return d,r
    def rhs(z):
        d,r=constitutive(z)
        fx=.5*(d+np.roll(d,-1,-2))*(np.roll(z,-1,-2)-z)/dx
        fy=.5*(d+np.roll(d,-1,-1))*(np.roll(z,-1,-1)-z)/dy
        return (fx-np.roll(fx,1,-2))/dx+(fy-np.roll(fy,1,-1))/dy+r
    try:
        constitutive(u)
        for target in targets:
            while t<target:
                d,_=constitutive(u)
                h=min(dt,.35/(float(np.max(d))*(1/dx**2+1/dy**2)),target-t)
                if h<=0 or t+h==t:raise FloatingPointError('time step underflow')
                k1=rhs(u);k2=rhs(u+h*k1/2);k3=rhs(u+h*k2/2);k4=rhs(u+h*k3)
                u+=h*(k1+2*k2+2*k3+k4)/6;t+=h;steps+=1
                if not np.all(np.isfinite(u)) or np.max(np.abs(u))>10:raise FloatingPointError('divergence')
                if steps>200000:raise FloatingPointError('budget')
            out.append(u.copy());times.append(float(target))
    except (FloatingPointError,ValueError) as e:status='failed';reason=str(e)
    return np.array(times),np.array(out),dict(status=status,reason=reason,steps=steps,
        state_min=float(u.min()) if np.isfinite(u).all() else None,state_max=float(u.max()) if np.isfinite(u).all() else None,
        requested_final_time=t_final,reached_final_time=t,nx=u.shape[-2],ny=u.shape[-1])


def work_2d(config):
    ident=f"2d_{config['case']}_{config['noise']:g}_{config['seed']}_e{config['epochs']}";dest=ROOT/'evidence/2d'/f'{ident}.json'
    if dest.exists():
        prior=json.loads(dest.read_text())
        if prior['config']!=config or prior.get('device','cpu')!=str(resolve_device()):raise ValueError(f'2D cached configuration/device mismatch: {ident}; use a distinct work root')
        if prior.get('source_signature')!=source_signature():raise ValueError('2D cached source signature mismatch; use a distinct work root')
        if prior['status']=='complete':
            out=ROOT/'outputs/runs'/ident
            validate_history(json.loads((out/'history.json').read_text()),config['epochs'])
            with np.load(out/'all_curves.npz') as values:
                if not all(np.isfinite(values[key]).all() for key in values):raise ValueError('Invalid cached 2D curves')
            if set(prior.get('metrics',{}))!={'neural','frozen','weak_poly','strong_poly'}:
                raise ValueError('Incomplete cached 2D metrics')
        return ident,'cached' if prior['status']=='complete' else 'cached_'+prior['status']
    start=time.perf_counter();cpu=time.process_time();result=dict(id=ident,config=config,classification='reviewer_experiment' if config['epochs']>=150 else 'runtime_pilot',status='running',source_signature=source_signature())
    out=ROOT/'outputs/runs'/ident;out.mkdir(parents=True,exist_ok=True);json_write(out/'config.json',config)
    try:
        case=extra_case(config['case']);seed=config['seed'];sim=SimulationConfig2D(nx=config['nx'],ny=config['ny'],t_final=config['t_final'],save_every=config['save_every'])
        raw=generate_2d(case,sim,config['num_trajectories'],seed=seed,amplitude_range=case.value_range,num_modes=config['num_initial_modes'],initial_clip_range=case.value_range)
        obs=apply_observation_model_2d(raw,ObservationConfig2D(noise_level=config['noise']),seed=seed,clip_range=case.value_range)
        result['observation_signature']=observation_signature(obs,case.value_range)
        model,history,seconds=train_2d(obs,case.value_range,config['epochs'],seed,out_dir=out,num_test_modes=config['num_test_modes'])
        result['device']=str(next(model.parameters()).device)
        result['torch_version']=torch.__version__
        json_write(out/'history.json',history);torch_write(out/'checkpoint.pt',model.state_dict())
        grid=np.linspace(*case.value_range,512,dtype=np.float32)
        with torch.no_grad():d=model.diffusion(torch.from_numpy(grid).to(next(model.parameters()).device)).cpu().numpy();r=model.reaction(torch.from_numpy(grid).to(next(model.parameters()).device)).cpu().numpy()
        ed,er=fit_symbols(grid,d,r);methods=dict(neural=(lambda z:np.interp(z,grid,d),lambda z:np.interp(z,grid,r)),frozen=(ed.evaluate,er.evaluate))
        result['controls']={}
        for method in ['weak','strong']:
            closure,mse=polynomial_2d(obs,case.value_range,method,num_test_modes=config['num_test_modes'])
            methods[method+'_poly']=(closure.diffusion,closure.reaction)
            result['controls'][method+'_poly']=dict(parameters_D=closure.diffusion_coefficients,parameters_R=closure.reaction_coefficients,fit_mse=mse)
        evaluation=np.linspace(*case.value_range,256);result['metrics']={name:curve_metrics(case,evaluation,D,R,obs) for name,(D,R) in methods.items()}
        result['BIR']={key:result['metrics']['frozen'][key]/result['metrics']['neural'][key] for key in ['D','R']}
        result['symbols']=[dataclasses.asdict(ed),dataclasses.asdict(er)];result['training_seconds']=seconds
        result['rollout']=[];curves=dict(grid=evaluation,true_D=case.diffusion(evaluation),true_R=case.reaction(evaluation))
        for name,(D,R) in methods.items():curves[name+'_D']=D(evaluation);curves[name+'_R']=R(evaluation)
        # Independent 32x32 validation grid, two novel ICs, horizon three times training.
        x=np.arange(32)/32;rng=np.random.default_rng(seed+1000)
        shapes=np.stack([random_fourier_initial_condition_2d(x,x,rng,(0.,1.),4) for _ in range(2)])
        for label,u0 in [('full_support',shapes*case.value_range[1]),('mean0.6_amp0.25',.6+.25*(2*shapes-1))]:
            times,truth,ti=rollout_2d(case.diffusion,case.reaction,u0)
            if ti['status']!='stable':raise FloatingPointError('truth validation')
            for name,(D,R) in methods.items():
                tp,pred,info=rollout_2d(D,R,u0);n=min(len(times),len(tp));err=[relative_l2_error(truth[i],pred[i]) for i in range(n)]
                result['rollout'].append(dict(method=name,pattern=label,times=tp,errors=err,info=info,validation_nx=32,validation_ny=32))
        hist,bins=np.histogram(obs['u'],bins=32,range=case.value_range,density=True);curves.update(state_hist=hist,state_bins=bins)
        npz_write(out/'all_curves.npz',**curves);result['status']='complete'
    except Exception as e:result['status']='failed';result['error']=str(e);result['traceback']=traceback.format_exc()
    result.update(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    json_write(dest,result);return ident,result['status']


def main():
    p=argparse.ArgumentParser(epilog='Common paths: --work-root PATH (output/cache root), --input-root PATH (explicit artifact input root); defaults use the project directory.');p.add_argument('--workers',type=int,default=1);p.add_argument('--epochs',type=int,default=150);p.add_argument('--limit',type=int);a=p.parse_args()
    tasks=[dict(case=c,noise=n,seed=s,epochs=a.epochs,nx=40,ny=40,t_final=.1,save_every=20,num_trajectories=6,num_initial_modes=4,num_test_modes=2)
        for c in ['case_exp','case_trig'] for n in [0.,.03] for s in range(5)]
    tasks += [dict(case='case_a',noise=0.,seed=s,epochs=a.epochs,nx=40,ny=40,t_final=.1,save_every=20,num_trajectories=6,num_initial_modes=4,num_test_modes=2) for s in range(5)]
    if a.limit:tasks=tasks[:a.limit]
    json_write(ROOT/f'evidence/manifest_2d_e{a.epochs}.json',dict(tasks=tasks,scope='isotropic scalar D/R periodic reaction diffusion, no 3D or anisotropy'))
    with process_pool(a.workers) as pool:
        report_completed(f.result() for f in cf.as_completed([pool.submit(work_2d,t) for t in tasks]))


if __name__=='__main__':main()
