"""Record the actual checkout, environment, inputs and local checkpoint hashes."""
import importlib.metadata
import platform
import subprocess
from project_paths import INVOCATION
from revision_core import *
from build_report import write_csv


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    checkout=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE_ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE_ROOT,text=True)
    files=sorted((SOURCE_ROOT/'src').rglob('*.py'))+sorted((SOURCE_ROOT/'revision').glob('*.py'))
    code=[dict(path=str(p.relative_to(SOURCE_ROOT)),sha256=digest(p)) for p in files]
    records=[];inputs=[]
    for directory in ('raw','2d','pysr','rollout','short_validation','paired','convergence_validation'):
        for f in input_glob(Path('evidence')/directory):
            r=json.loads(f.read_text())
            inputs.append(dict(id=r['id'],kind=directory,status=r['status'],sha256=digest(f),
                               source_signature=r.get('source_signature'),model_signature=r.get('model_signature')))
            if directory not in ('raw','2d'):continue
            for p in sorted((ROOT/'outputs/runs'/r['id']).glob('checkpoint*.pt')):
                records.append(dict(id=r['id'],path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p),
                    data_hash=r.get('data_hash'),observation_signature=r.get('observation_signature'),
                    configuration_sha256=hashlib.sha256(json.dumps(r['config'],sort_keys=True).encode()).hexdigest()))
    write_csv(ROOT/'evidence/checkpoint_manifest.csv',records)
    json_write(ROOT/'evidence/final_provenance.json',dict(checkout_commit=checkout,git_status=status,
        source_signature=source_signature(),work_root=str(ROOT),input_root=str(INPUT_ROOT),command=INVOCATION,
        platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        code=code,checkpoint_files=len(records),inputs=inputs,
        note='Current code identity is distinct from historical input identities. A dirty checkout is identified by file hashes.'))
    print('code hashes',len(code),'checkpoint hashes',len(records),'input hashes',len(inputs))


if __name__=='__main__':main()
