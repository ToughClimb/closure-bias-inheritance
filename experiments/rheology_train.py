"""Predeclared whole-replicate rheometry experiment; independent batched NNs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rheology_core import ViscosityNetwork, features


class BatchedNetworks(torch.nn.Module):
    def __init__(self, seeds, width=16):
        super().__init__()
        nets=[]
        for seed in seeds:
            torch.manual_seed(int(seed))
            nets.append(ViscosityNetwork(width).double())
        for layer in (0,2,4):
            self.register_parameter(f'w{layer}',torch.nn.Parameter(torch.stack([n.net[layer].weight.T for n in nets])))
            self.register_parameter(f'b{layer}',torch.nn.Parameter(torch.stack([n.net[layer].bias for n in nets])[:,None,:]))

    def forward(self,x):
        for layer in (0,2,4):
            x=torch.bmm(x,getattr(self,f'w{layer}'))+getattr(self,f'b{layer}')
            if layer!=4:x=torch.tanh(x)
        return x.squeeze(-1)


def prepare(output):
    output=Path(output)
    data=pd.read_csv(output/'observations.csv')
    specs=[]; arrays=[]
    for arm in ('primary_high_shear','full_positive'):
        for material, frame in data.groupby('material',sort=True):
            for fold in range(5):
                train=frame[(frame.replicate!=fold)&(frame.viscosity>0)]
                if arm=='primary_high_shear':train=train[train.rate>=10]
                temperature_bounds=[float(train.temperature.min()),float(train.temperature.max())]
                rate_bounds=[float(train.rate.min()),float(train.rate.max())]
                x=features(train.temperature.to_numpy(),train.rate.to_numpy(),temperature_bounds,rate_bounds)
                y=np.log(train.viscosity.to_numpy()); offset=float(y.mean())
                for seed in (0,1):
                    specs.append(dict(index=len(specs),arm=arm,material=material,fold=fold,seed=seed,
                                      temperature_bounds=temperature_bounds,rate_bounds=rate_bounds,offset=offset,
                                      training_row_ids=train.index.tolist(),heldout_row_ids=frame[frame.replicate==fold].index.tolist(),
                                      training_points=len(train),excluded_nonpositive_training=int(((frame.replicate!=fold)&(frame.viscosity<=0)).sum())))
                    arrays.append((x,y-offset))
    maxn=max(len(y) for x,y in arrays)
    x=np.zeros((len(arrays),maxn,2));y=np.zeros((len(arrays),maxn));mask=np.zeros_like(y)
    for i,(a,b) in enumerate(arrays):x[i,:len(b)]=a;y[i,:len(b)]=b;mask[i,:len(b)]=1
    np.savez_compressed(output/'training_arrays.npz',x=x,y=y,mask=mask)
    (output/'training_specs.json').write_text(json.dumps(specs,indent=2))
    return specs,x,y,mask


def run(output,device='cuda'):
    output=Path(output);(output/'models').mkdir(exist_ok=True)
    if any((output/'models').iterdir()):raise FileExistsError('Existing checkpoints; explicit resume required instead of overwrite')
    protocol=json.loads((output/'PROTOCOL.json').read_text())
    assert protocol['optimizer']['steps']==6000
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    specs,x,y,mask=prepare(output)
    model=BatchedNetworks([s['seed'] for s in specs]).to(device)
    tx=torch.as_tensor(x,device=device);ty=torch.as_tensor(y,device=device);tm=torch.as_tensor(mask,device=device)
    counts=tm.sum(1)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    history=[];start=time.time()
    src=output/'provenance'/'source_at_training';src.mkdir(parents=True,exist_ok=True)
    for name in ('rheology_core.py','rheology_data.py','rheology_train.py'):
        shutil.copy2(Path(__file__).with_name(name),src/name)
    (output/'TRAINING_START.json').write_text(json.dumps(dict(start_unix=start,device=device,
        torch=torch.__version__,protocol_sha256=hashlib.sha256((output/'PROTOCOL.json').read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}),indent=2))
    (output/'ENVIRONMENT.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),
        device=device,cuda_device=torch.cuda.get_device_name() if device.startswith('cuda') else None,
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}),indent=2))
    for step in range(6001):
        optimizer.zero_grad(set_to_none=True)
        loss=(((model(tx)-ty)**2)*tm).sum(1)/counts
        if not bool(torch.isfinite(loss).all()):raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.sum().backward()
        if step%500==0 or step in (2000,6000):
            values=loss.detach().cpu().numpy();grads=sum(p.grad.reshape(len(specs),-1).square().sum(1) for p in model.parameters()).sqrt().cpu().numpy()
            history.append(dict(step=step,seconds=time.time()-start,mean_loss=float(values.mean()),max_loss=float(values.max()),max_gradient=float(grads.max())))
            print(json.dumps(history[-1]),flush=True)
        if step in (2000,6000):
            params={n:p.detach().cpu().numpy() for n,p in model.named_parameters()}
            np.savez_compressed(output/'models'/f'checkpoint_{step}.npz',**params,loss=values,gradient_l2=grads)
            torch.save({'optimizer':optimizer.state_dict(),'step':step},output/'models'/f'optimizer_{step}.pt')
        if step==6000:break
        if step==4000:
            for g in optimizer.param_groups:g['lr']=.001
        if step==5000:
            for g in optimizer.param_groups:g['lr']=.0003
        optimizer.step()
    (output/'training_history.json').write_text(json.dumps(history,indent=2))
    (output/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(status='completed_fixed_budget',independent_teachers=len(specs),
        checkpoints=[2000,6000],seconds=time.time()-start,strict_gradient_l2_below_1e_6=int((grads<=1e-6).sum()),
        no_convergence_claim=True),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();run(args.output,args.device)
