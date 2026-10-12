"""Artifact IO and cache identity without importing Torch or optional Julia."""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from project_paths import ROOT, SOURCE_ROOT, input_path


def source_signature():
    digest = hashlib.sha256()
    files = sorted((SOURCE_ROOT / 'src').rglob('*.py')) + sorted((SOURCE_ROOT / 'revision').glob('*.py'))
    for path in files:
        digest.update(str(path.relative_to(SOURCE_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def json_write(path, value):
    def default(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        if dataclasses.is_dataclass(item): return dataclasses.asdict(item)
        raise TypeError(f'Cannot serialize {type(item).__name__}')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(json.dumps(value, indent=2, default=default, allow_nan=False))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def npz_write(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **values)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class CurveValues(dict):
    @property
    def files(self): return list(self)


def load_curves(ident, epoch=None, pysr=False):
    leaf = f'curves_{epoch}.npz' if epoch is not None else ('pysr/curves.npz' if pysr else 'all_curves.npz')
    path = input_path(Path('outputs/runs') / ident / leaf)
    if path.exists():
        with np.load(path) as values:
            return CurveValues({key: values[key].copy() for key in values.files})
    name = f'{ident}_e{epoch}' if epoch is not None else ('pysr_' + ident if pysr else ident)
    directory = 'surrogate_curves' if epoch is not None else 'curves'
    path = input_path(Path('evidence') / directory / f'{name}.csv')
    if not path.exists(): raise FileNotFoundError(f'Missing declared curve input: {path}')
    with path.open(newline='') as stream: rows = list(csv.DictReader(stream))
    if not rows: raise ValueError(f'Empty curve input: {path}')
    values = {('grid' if key == 'u' else key): np.array([float(row[key]) for row in rows]) for key in rows[0]}
    histogram = input_path(Path('evidence/state_histograms') / f'{ident}.csv')
    if histogram.exists() and epoch is None and not pysr:
        with histogram.open(newline='') as stream: rows = list(csv.DictReader(stream))
        values['state_hist'] = np.array([float(row['density']) for row in rows])
        values['state_bins'] = np.array([float(row['lower']) for row in rows] + [float(rows[-1]['upper'])])
    return CurveValues(values)


def load_history(ident):
    path = input_path(Path('outputs/runs') / ident / 'history.json')
    if path.exists(): return json.loads(path.read_text())
    path = input_path(Path('evidence/histories') / f'{ident}.csv')
    with path.open(newline='') as stream:
        return [{key: int(value) if key == 'epoch' else float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def model_signature(row):
    values = load_curves(row['id'], epoch=row['config']['epochs'])
    digest = hashlib.sha256(json.dumps(dict(config=row['config'], data_hash=row.get('data_hash')), sort_keys=True).encode())
    for key in ('grid', 'D', 'R'):
        array = np.ascontiguousarray(values[key], dtype=np.float64)
        if not np.isfinite(array).all(): raise FloatingPointError(f'Nonfinite surrogate {key}: {row["id"]}')
        digest.update(key.encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def report_completed(results):
    failed = []
    for ident, status in results:
        print(ident, status, flush=True)
        if status not in ('complete', 'cached'): failed.append(ident)
    if failed: raise RuntimeError(f'{len(failed)} tasks did not complete; inspect their persisted raw records: {failed[:5]}')


def validation_identity(row, protocol):
    if row.get('status') != 'complete':
        raise ValueError(f'Validation requires a complete input: {row["id"]}')
    return dict(source_signature=source_signature(), model_signature=model_signature(row),
                input_signature=hashlib.sha256(json.dumps(row,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                validation_protocol=protocol)


def cached_validation(path, identity):
    if not Path(path).exists(): return None
    prior=json.loads(Path(path).read_text())
    if any(prior.get(key) != value for key,value in identity.items()):
        raise ValueError(f'Validation cache source/input/protocol mismatch: {path}; use a distinct work root')
    return 'cached' if prior['status']=='complete' else 'cached_'+prior['status']
