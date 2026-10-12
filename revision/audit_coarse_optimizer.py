"""Evaluation-only localization of optimizer overflow in unresolved nx8 tests."""
from revision_core import *
from build_report import write_csv


def main():
    rows=[]
    for seed in range(5):
        for group,setting in [('main','clean'),('coarse','coarse16'),('coarse','coarse8'),('coarse_corrected','coarse8')]:
            ident=f'{group}_case_exp_{setting}_{seed}_default'
            p=input_path(Path('outputs/runs')/ident/'checkpoint_150.pt')
            saved=torch.load(p,weights_only=False,map_location='cpu')
            values=[v for state in saved['optimizer']['state'].values() for k,v in state.items()
                    if isinstance(v,torch.Tensor) and k in ('exp_avg','exp_avg_sq')]
            count=sum(int((~torch.isfinite(v)).sum()) for v in values)
            rows.append(dict(id=ident,seed=seed,group=group,setting=setting,optimizer_state_elements=sum(v.numel() for v in values),
                nonfinite_elements=count,model_all_finite=all(bool(torch.isfinite(v).all()) for v in saved['model'].values()),
                interpretation='Numerically invalid original nx8 test construction; diagnostic comparison only' if count else 'Finite optimizer state'))
    write_csv(ROOT/'evidence/coarse_optimizer_diagnostic.csv',rows)
    print('checkpoints',len(rows),'with_nonfinite_optimizer',sum(r['nonfinite_elements']>0 for r in rows))


if __name__=='__main__':main()
