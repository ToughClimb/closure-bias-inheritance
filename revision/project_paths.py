"""Shared CLI paths; no machine-specific filesystem assumptions.

Every entry point accepts --work-root and --input-root before its own options.
Relative paths are resolved against the caller's current directory. The source
and implementation always come from this checkout. Input reads prefer
new local artifacts, then the explicitly selected input project.
"""
from pathlib import Path
import argparse
import os
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
INVOCATION = list(sys.argv)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--work-root', default=os.environ.get('EAAI_WORK_ROOT'))
_parser.add_argument('--input-root', default=os.environ.get('EAAI_INPUT_ROOT'))
_parser.add_argument('--device',choices=['cpu','cuda','auto'],default=os.environ.get('EAAI_DEVICE','cpu'))
_args, _remaining = _parser.parse_known_args()
sys.argv[1:] = _remaining
ROOT = Path(_args.work_root).expanduser().resolve() if _args.work_root else SOURCE_ROOT / 'outputs/revision'
INPUT_ROOT = Path(_args.input_root).expanduser().resolve() if _args.input_root else ROOT
DEVICE_OPTION = _args.device
ROOT.mkdir(parents=True, exist_ok=True)
os.environ['EAAI_WORK_ROOT']=str(ROOT)
os.environ['EAAI_INPUT_ROOT']=str(INPUT_ROOT)
os.environ['EAAI_DEVICE']=DEVICE_OPTION
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'outputs/cache'))


def input_path(relative):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError('Artifact inputs must be relative to an explicit project root')
    local = ROOT / relative
    return local if local.exists() else INPUT_ROOT / relative


def input_glob(relative, pattern='*.json'):
    files = {p.name: p for p in (INPUT_ROOT / relative).glob(pattern)}
    files.update({p.name: p for p in (ROOT / relative).glob(pattern)})
    return [files[k] for k in sorted(files)]
