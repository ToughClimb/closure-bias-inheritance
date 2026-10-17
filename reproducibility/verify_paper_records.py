"""Verify the frozen numerical records cited by the revised manuscript."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


DEFAULT_RECORDS = Path(__file__).resolve().parent / "paper-records-2026-10-16"


def verify(records: Path) -> dict:
    manifest = json.loads((records / "MANIFEST.json").read_text())
    entries = manifest["files"]
    assert len(entries) == manifest["record_files"] == 41
    assert len({entry["path"] for entry in entries}) == len(entries)
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
        else:
            value = json.loads(data)
            assert type(value).__name__ == entry["json_type"], entry["path"]
            assert len(value) == entry["records"], entry["path"]
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
    return dict(status="passed", record_files=len(entries), total_bytes=total_bytes,
                same_observation_datasets=len(models), model_records=len(closure_rows),
                bir_summary_rows=len(bir), ci_max_absolute_difference=max_difference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    args = parser.parse_args()
    print(json.dumps(verify(args.records.resolve()), indent=2))


if __name__ == "__main__":
    main()
