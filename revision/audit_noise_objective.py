"""Evaluate the fixed objective at true/initial/fitted laws; never fit to truth.

This is a numerical diagnostic of loss magnitudes and errors-in-variables, not
an additional trained estimator. Truth is used only for evaluation. Its tiny
smoothness contribution uses finite grid derivatives, explicitly recorded.
"""
from revision_core import *
from run_suite import SETTINGS
from build_report import write_csv


class EvaluationTruth:
    def __init__(self,case):self.case=case
    def diffusion(self,u):return torch.from_numpy(self.case.diffusion(u.detach().numpy())).to(u.dtype)
    def reaction(self,u):return torch.from_numpy(self.case.reaction(u.detach().numpy())).to(u.dtype)
    def smoothness_penalty(self,g):
        x=g.detach().numpy().astype(float)
        d=np.gradient(self.case.diffusion(x),x,edge_order=2);r=np.gradient(self.case.reaction(x),x,edge_order=2)
        return torch.tensor(np.mean(d*d+r*r),dtype=g.dtype)


def main():
    torch.set_num_threads(1);case=extra_case('case_exp');rows=[];start=time.perf_counter();cpu=time.process_time()
    for seed in range(5):
        raw=generate_dataset(case,SimulationConfig1D(nx=64,t_final=.1,save_every=10),8,seed=seed,
                             amplitude_range=case.value_range,initial_clip_range=case.value_range)
        for setting in ['clean','noise1','noise3','noise5']:
            noise,ss,ts=SETTINGS[setting];observed=apply_observation_model(raw,noise_level=noise,space_stride=ss,time_stride=ts,seed=seed,clip_range=case.value_range)
            ident=f'main_case_exp_{setting}_{seed}_default';u=torch.tensor(observed['u'],dtype=torch.float32)
            phi,gphi=[torch.tensor(v,dtype=torch.float32) for v in make_test_functions_1d(observed['x'])]
            support=torch.linspace(*case.value_range,256,requires_grad=True);anchor=torch.tensor([0.])
            torch.manual_seed(seed);initial=ReactionDiffusionClosure(input_range=case.value_range)
            trained=ReactionDiffusionClosure(input_range=case.value_range)
            trained.load_state_dict(torch.load(input_path(Path('outputs/runs')/ident/'checkpoint_150.pt'),weights_only=False,map_location='cpu')['model'])
            for name,model in [('true_evaluation_only',EvaluationTruth(case)),('initial',initial),('trained',trained)]:
                terms=objective_terms(model,u,1/64,.001,phi,gphi,support,anchor)
                total=sum(DEFAULT_WEIGHTS[k]*v for k,v in terms.items())
                for term,value in terms.items():rows.append(dict(id=ident,seed=seed,setting=setting,model=name,term=term,
                    raw=float(value.detach()),weight=DEFAULT_WEIGHTS[term],weighted=float(value.detach())*DEFAULT_WEIGHTS[term],total=float(total.detach())))
    write_csv(ROOT/'evidence/noise_objective_diagnostic.csv',rows)
    json_write(ROOT/'evidence/noise_objective_diagnostic.json',dict(status='complete',classification='evaluation_only_numerical_audit',
        id='noise_objective_diagnostic',seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        cases=['case_exp'],seeds=list(range(5)),settings=['clean','noise1','noise3','noise5'],
        rule='True closures used only for diagnostic objective evaluation, never for fitting, selection or hyperparameter tuning',
        caveat='Oracle smoothness term approximated by 256-point grid gradients; all data residuals use the exact implemented objective'))
    print('diagnostic rows',len(rows))


if __name__=='__main__':main()
