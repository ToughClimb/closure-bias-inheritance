"""Verify the frozen numerical records cited by the revised manuscript."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tomllib
import zipfile


DEFAULT_RECORDS = Path(__file__).resolve().parent / "paper-records-2026-10-16"


def verify(records: Path) -> dict:
    manifest = json.loads((records / "MANIFEST.json").read_text())
    entries = manifest["files"]
    assert len(entries) == manifest["record_files"]
    assert len({entry["path"] for entry in entries}) == len(entries)
    assert {entry['path'] for entry in entries} == {
        str(path.relative_to(records)) for path in records.rglob('*')
        if path.is_file() and path.name not in {'MANIFEST.json','README.md'}
    }
    total_bytes = 0
    for entry in entries:
        path = records / entry["path"]
        data = path.read_bytes()
        assert len(data) == entry["size"], entry["path"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
        total_bytes += len(data)
        if path.suffix == ".csv":
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            assert reader.fieldnames == entry["columns"], entry["path"]
            assert len(rows) == entry["rows"], entry["path"]
            assert all(None not in row for row in rows), entry["path"]
        elif path.suffix == '.json':
            value = json.loads(data)
            assert type(value).__name__ == entry["json_type"], entry["path"]
            assert len(value) == entry["records"], entry["path"]
        elif path.suffix == '.npz':
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None
                assert sorted(archive.namelist()) == entry['npz_members']
        elif path.suffix == '.toml':
            value=tomllib.loads(data.decode('utf-8'))
            assert len(value)==entry['toml_tables'],entry['path']
        else:
            raise AssertionError(entry['path'])
    assert total_bytes == manifest["total_bytes"]

    models = json.loads((records / "sameobs_model_descriptors.json").read_text())
    with (records / "sameobs_closure_by_seed.csv").open(newline="") as stream:
        closure_rows = list(csv.DictReader(stream))
    assert len(models) == 10 and len(closure_rows) == 100
    assert {item["id"] for item in models} == {row["id"] for row in closure_rows}
    for item in models:
        methods = {row["method"] for row in closure_rows if row["id"] == item["id"]}
        assert set(item["descriptors"]) == methods, item["id"]

    with (records / "campaign_bir_summary.csv").open(newline="") as stream:
        bir = list(csv.DictReader(stream))
    assert len(bir) == 576 and all(row["n"] == "5" for row in bir)
    critical = 2.7764451051977987
    max_difference = 0.0
    for row in bir:
        mean, sd, low, high = (float(row[key]) for key in ("mean", "sd", "ci95_low", "ci95_high"))
        assert all(math.isfinite(value) for value in (mean, sd, low, high))
        delta = critical * sd / math.sqrt(5)
        max_difference = max(max_difference, abs(low - (mean - delta)), abs(high - (mean + delta)))
    assert max_difference < 1e-9, max_difference

    # Recompute the endpoint summaries, including the failed-run denominator.
    with (records/'twod/forward_by_seed.csv').open(newline='') as stream:
        forward = list(csv.DictReader(stream))
    with (records/'twod/forward_summary.csv').open(newline='') as stream:
        summaries = list(csv.DictReader(stream))
    assert len(forward) == 200 and len(summaries) == 40
    keys = ['case','setting','method','pattern']
    for summary in summaries:
        group = [row for row in forward if all(row[key]==summary[key] for key in keys)]
        completed = [row for row in group if row['completed']=='1']
        assert len(group)==int(summary['total'])==5
        assert len(completed)==int(summary['completed'])
        assert all(row['endpoint_relative_l2']=='' for row in group if row['completed']=='0')
        values = [float(row['endpoint_relative_l2']) for row in completed]
        assert abs(statistics.mean(values)-float(summary['mean'])) < 1e-12
        assert abs(statistics.stdev(values)-float(summary['sample_sd'])) < 1e-12
    completion = {method:sum(int(row['completed']) for row in forward if row['method']==method)
                  for method in ['neural','frozen','weak_poly','strong_poly']}
    assert completion == {'neural':50,'frozen':50,'weak_poly':50,'strong_poly':46}
    task_records = json.loads((records/'twod/task_records.json').read_text())
    assert len(task_records)==25 and {r['config']['epochs'] for r in task_records}=={150}
    fits = json.loads((records/'rheology/constraint_fit_records.json').read_text())
    assert len(fits)==150 and all(row['diagnostic']['passed'] for row in fits)
    assert min(min(row['bernstein_margin']) for row in fits) >= .05
    return dict(status="passed", record_files=len(entries), total_bytes=total_bytes,
                same_observation_datasets=len(models), model_records=len(closure_rows),
                bir_summary_rows=len(bir), ci_max_absolute_difference=max_difference,
                twod_tasks=25, twod_forward_records=200, twod_completion=completion,
                constrained_rheology_records=len(fits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    args = parser.parse_args()
    print(json.dumps(verify(args.records.resolve()), indent=2))


if __name__ == "__main__":
    main()
