"""Lossless numeric adapter for Zenodo 19699588 RheoCompass exports."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def parse_rheocompass(path):
    path = Path(path)
    lines = path.read_text(encoding='utf-16').splitlines()
    match = re.fullmatch(r'(.+)_Resin Viscosity\s*([1-4]?)\.csv', path.name)
    if not match:
        raise ValueError(f'Unrecognized replicate filename: {path.name}')
    material, replicate = match.group(1), int(match.group(2) or 0)
    block, units, result = -1, False, []
    test_ids = [line.split('\t')[1] for line in lines if line.startswith('Test ID:\t')]
    if len(test_ids) != 1:
        raise ValueError(f'Need one whole-run test ID: {path}')
    for lineno, line in enumerate(lines, 1):
        fields = line.split('\t')
        if line.startswith('Interval data:\t'):
            if fields[1:] != ['Point No.', 'Temperature', 'Viscosity', 'Status', 'Shear Rate', 'Rotational Speed']:
                raise ValueError(f'Unexpected numeric header: {path}:{lineno}')
            block += 1
            units = False
        if fields == ['', '', '[°C]', '[cP]', '', '[1/s]', '[rpm]']:
            units = True
        if len(fields) == 7 and fields[0] == '' and fields[1].isdigit():
            if not units or block < 0:
                raise ValueError('Data before verified units/header')
            point = int(fields[1])
            result.append(dict(material=material, replicate=replicate, file=path.name,
                               test_id=test_ids[0], raw_line=lineno, block=block, point=point,
                               nominal_temperature=35+10*round((float(fields[2])-35)/10), temperature=float(fields[2]),
                               viscosity_cp=float(fields[3]), viscosity=float(fields[3])/1000,
                               status=fields[4], rate=float(fields[5]), rpm=float(fields[6])))
    df = pd.DataFrame(result)
    if len(df) != 250 or block != 9 or not np.isfinite(df[['temperature','viscosity','rate','rpm']]).all().all():
        raise ValueError(f'Unexpected run dimensions or nonfinite data: {path}')
    if np.max(np.abs(df.temperature-df.nominal_temperature)) > .2:
        raise ValueError('Temperature block does not match the declared experimental grid')
    for _, frame in df.groupby('block'):
        if frame.point.tolist() != list(range(1,26)) or np.any(np.diff(frame.rate) <= 0):
            raise ValueError('Unexpected measurement order; no silent sorting or deletion')
    if sorted(df.nominal_temperature.unique()) != list(range(35,126,10)):
        raise ValueError('Missing or duplicate nominal temperature conditions')
    return df


def load_dataset(root):
    files = sorted(Path(root).rglob('*.csv'))
    if len(files) != 50:
        raise ValueError(f'Expected exactly 50 published CSV files; found {len(files)}')
    df = pd.concat([parse_rheocompass(p) for p in files], ignore_index=True)
    if df.material.nunique() != 10 or df.test_id.nunique() != 50:
        raise ValueError('Unexpected material or independent test-ID count')
    for material, frame in df.groupby('material'):
        if sorted(frame.replicate.unique()) != [0,1,2,3,4]:
            raise ValueError(f'Incomplete repeat identifiers: {material}')
    return df


def audit_dataset(raw, output):
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_dataset(raw/'extracted')
    data.to_csv(output/'observations.csv', index=False)
    rows = []
    for (material, replicate), frame in data.groupby(['material','replicate']):
        rows.append(dict(material=material, replicate=int(replicate), records=len(frame),
                         nonpositive=int((frame.viscosity<=0).sum()),
                         primary_records=int((frame.rate>=10).sum()),
                         primary_nonpositive=int(((frame.rate>=10)&(frame.viscosity<=0)).sum()),
                         test_id=frame.test_id.iloc[0]))
    pd.DataFrame(rows).to_csv(output/'raw_run_audit.csv', index=False)
    hashes = {str(p.relative_to(raw)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((raw/'extracted').rglob('*')) if p.is_file()}
    raw_monotonicity=[]
    for (material, replicate, block), f in data.groupby(['material','replicate','block']):
        for arm, f2 in [('full',f),('primary',f[f.rate>=10])]:
            stress=f2.rate.to_numpy()*f2.viscosity.to_numpy()
            raw_monotonicity.append(dict(material=material,replicate=int(replicate),block=int(block),arm=arm,
                                         nonpositive_viscosity=int((f2.viscosity<=0).sum()),
                                         stress_decreasing_intervals=int((np.diff(stress)<=0).sum())))
    pd.DataFrame(raw_monotonicity).to_csv(output/'raw_monotonicity.csv',index=False)
    audit=dict(rows=len(data),files=50,materials=10,independent_test_ids=50,
               units={'viscosity_raw':'cP','viscosity_analysis':'Pa s','temperature':'C','rate':'1/s'},
               conversion_cp_to_pa_s=.001,nonpositive_rows=int((data.viscosity<=0).sum()),
               high_shear_nonpositive_rows=int(((data.rate>=10)&(data.viscosity<=0)).sum()),
               actual_temperature_range=[float(data.temperature.min()),float(data.temperature.max())],
               actual_rate_range=[float(data.rate.min()),float(data.rate.max())],
               nominal_temperature_grid=list(range(35,126,10)),
               original_file_sha256=hashes,unique_file_hashes=len(set(hashes.values())),
               no_rows_removed_from_observations_csv=True,
               source_measurement='Steady shear rotation; each CSV is a whole test run; all temperature blocks remain together',
               license='CC-BY-4.0',doi='10.5281/zenodo.19699588')
    (output/'DATA_AUDIT.json').write_text(json.dumps(audit,indent=2))
    return data,audit


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if (args.output/'DATA_AUDIT.json').exists():
        raise FileExistsError('Audit already exists; choose a new output directory')
    audit_dataset(args.raw,args.output)
