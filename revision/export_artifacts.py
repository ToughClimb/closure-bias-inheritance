"""Export computed arrays to portable decimal CSV; does not run experiments."""
from revision_core import *
from build_report import write_csv


def main():
    count=0
    for directory in ['raw','2d','pysr']:
        for path in input_glob(Path('evidence')/directory):
            r=json.loads(path.read_text())
            if r['status']!='complete':continue
            ident=r['id'];out=input_path(Path('outputs/runs')/ident)
            epochs=r.get('epochs',{})
            for epoch in epochs:
                q=load_curves(ident,epoch=int(epoch))
                write_csv(ROOT/'evidence/surrogate_curves'/f'{ident}_e{epoch}.csv',
                    [dict(u=float(z),D=float(q['D'][i]),R=float(q['R'][i])) for i,z in enumerate(q['grid'])])
                count+=1
            leaf='pysr/curves.npz' if directory=='pysr' else 'all_curves.npz'
            curve=out/leaf
            if curve.exists():
                q=load_curves(ident,pysr=directory=='pysr');prefix='pysr_' if directory=='pysr' else ''
                write_csv(ROOT/'evidence/curves'/f'{prefix}{ident}.csv',
                    [dict(u=float(z),**{k:float(q[k][i]) for k in q if k not in ['grid','state_hist','state_bins']}) for i,z in enumerate(q['grid'])])
                if 'state_hist' in q:
                    write_csv(ROOT/'evidence/state_histograms'/f'{ident}.csv',
                        [dict(lower=a,upper=b,density=v) for a,b,v in zip(q['state_bins'][:-1],q['state_bins'][1:],q['state_hist'])])
            if (out/'history.json').exists():
                write_csv(ROOT/'evidence/histories'/f'{ident}.csv',load_history(ident))
    print('portable surrogate snapshots',count)


if __name__=='__main__':main()
