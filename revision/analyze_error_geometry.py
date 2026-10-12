"""Post-fit error direction and empirical five-seed bias/variance diagnostics.

No fitting, settings selection or truth-based tuning occurs. Empirical ensemble
bias is descriptive; five seeds do not identify the population expectation.
"""
from collections import defaultdict
from revision_core import *
from build_report import write_csv


def main():
    geometry=[];groups=defaultdict(list)
    for path in input_glob('evidence/raw', '*.json'):
        r=json.loads(path.read_text())
        if r['status']!='complete':continue
        c=r['config'];case=extra_case(c['case']);g=np.linspace(*case.value_range,256);out=ROOT/'outputs/runs'/r['id']
        for epoch,snap in r['epochs'].items():
            q=load_curves(r['id'],epoch=int(epoch));symbols=[SymbolicExpression(**{**e,'parameters':tuple(e['parameters'])}) for e in snap['symbols']]
            for branch,index in [('D',0),('R',1)]:
                truth=case.diffusion(g) if branch=='D' else case.reaction(g);neural=np.interp(g,q['grid'],q[branch]);frozen=symbols[index].evaluate(g)
                a=neural-truth;b=frozen-truth;delta=frozen-neural;en=np.linalg.norm(a);es=np.linalg.norm(b)
                geometry.append(dict(id=r['id'],group=c['group'],case=c['case'],setting=c['setting'],variant=c.get('variant','default'),seed=c['seed'],epoch=int(epoch),branch=branch,
                    error_cosine=float(np.dot(a,b)/(en*es)) if en*es else None,
                    signed_neural_mean_error=float(np.mean(a)),signed_frozen_mean_error=float(np.mean(b)),
                    BIR=float(es/en),compression_to_neural_error=float(np.linalg.norm(delta)/en)))
                for method,pred in [('neural',neural),('frozen',frozen)]:
                    key=(c['group'],c['case'],c['setting'],c.get('variant','default'),int(epoch),method,branch)
                    groups[key].append((c['seed'],pred,truth))
        curve=out/'all_curves.npz'
        if r.get('controls'):
            q=load_curves(r['id'])
            for method in r.get('controls',{}):
                for branch in ['D','R']:
                    key=(c['group'],c['case'],c['setting'],c.get('variant','default'),c['epochs'],method,branch)
                    groups[key].append((c['seed'],q[method+'_'+branch],q['true_'+branch]))
    decomposition=[]
    for key,items in sorted(groups.items()):
        seeds=[s for s,p,t in items]
        if len(set(seeds))!=len(seeds):raise ValueError('Non-independent duplicated data seed in geometry group')
        pred=np.stack([p for s,p,t in items]);truth=items[0][2];mean=pred.mean(0);norm=float(np.sum(truth**2))
        bias=float(np.sum((mean-truth)**2)/norm);variance=float(np.mean(np.sum((pred-mean)**2,axis=1))/norm);mse=float(np.mean(np.sum((pred-truth)**2,axis=1))/norm)
        assert np.isclose(mse,bias+variance,rtol=1e-12,atol=1e-14)
        decomposition.append(dict(zip(['group','case','setting','variant','epoch','method','branch'],key),n=len(items),
            normalized_empirical_bias_squared=bias,normalized_population_variance=variance,mean_squared_relative_error=mse,
            coherent_error_fraction=bias/mse if mse else None,
            interpretation='Empirical ensemble mean on preset uniform U; population variance convention for exact decomposition; descriptive, not unbiased population-bias estimate'))
    write_csv(ROOT/'evidence/error_direction_raw.csv',geometry);write_csv(ROOT/'evidence/empirical_bias_variance.csv',decomposition)
    print('error direction rows',len(geometry),'ensemble decompositions',len(decomposition))


if __name__=='__main__':main()
