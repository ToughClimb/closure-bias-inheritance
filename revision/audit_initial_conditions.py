"""Report exact IC means/ranges; legacy 'mean' names denote midpoints.

This corrects labels/interpretation only. Initial states, model fits, rollout
errors, rankings and BIR are unchanged. Random Fourier min/max normalization
does not generally make its spatial average equal to its range midpoint.
"""
from revision_core import *
from build_report import write_csv
from closure_discovery.data_generation.rd_solver_1d import random_fourier_initial_condition
from closure_discovery.data_generation.rd_solver_2d import random_fourier_initial_condition_2d


def main():
    rows=[]
    for directory in ['rollout','2d','convergence_validation']:
        for f in input_glob(Path('evidence')/directory, '*.json'):
            r=json.loads(f.read_text())
            if r['status']!='complete' or r['classification']=='runtime_pilot':continue
            c=r['config'];case=extra_case(c['case']);rng=np.random.default_rng(c['seed']+1000)
            if directory in ['rollout','convergence_validation']:
                x=np.arange(64)/64;shape=np.stack([random_fourier_initial_condition(x,rng,(0.,1.),4) for _ in range(4)])
                patterns=[('full_support',sum(case.value_range)/2,(case.value_range[1]-case.value_range[0])/2)]
                if directory=='convergence_validation':patterns += [('midpoint1.1_amplitude0.25',1.1,.25)]
                elif case.name=='case_exp':patterns += [(f'mean{m:g}_amplitude{a:g}',m,a) for m,a in [(.25,.1),(.6,.2),(1.1,.25),(1.5,.1)]]
                elif c['group']=='families':patterns += [('mean0.6_amplitude0.2',.6,.2)]
            else:
                x=np.arange(32)/32;shape=np.stack([random_fourier_initial_condition_2d(x,x,rng,(0.,1.),4) for _ in range(2)])
                patterns=[('full_support',sum(case.value_range)/2,(case.value_range[1]-case.value_range[0])/2),('mean0.6_amp0.25',.6,.25)]
            for pattern,midpoint,amplitude in patterns:
                u0=shape*(case.value_range[1]-case.value_range[0])+case.value_range[0] if pattern=='full_support' else midpoint+amplitude*(2*shape-1)
                for i,z in enumerate(u0):rows.append(dict(id=r['id'],dimension=2 if directory=='2d' else 1,
                    case=c['case'],seed=c['seed'],pattern=pattern,trajectory=i,range_midpoint=midpoint,half_range=amplitude,
                    actual_spatial_mean=float(z.mean()),mean_minus_midpoint=float(z.mean()-midpoint),
                    initial_min=float(z.min()),initial_max=float(z.max()),
                    convention='Legacy pattern name mean denotes midpoint; actual spatial mean explicitly reported'))
    write_csv(ROOT/'evidence/initial_condition_means.csv',rows);print('initial-state descriptions',len(rows))


if __name__=='__main__':main()
