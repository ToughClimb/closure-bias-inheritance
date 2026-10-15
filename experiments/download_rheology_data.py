"""Download and verify the public rheology dataset used by the experiments."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile


RECORD = "19699588"
FILENAME = "Rheological_Data_For_Syntactic_Foam.zip"
EXPECTED_MD5 = "ef9df53250cb9984fa1697b87d781db1"
EXPECTED_BYTES = 343805


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory to preserve existing inputs")
    url = f"https://zenodo.org/api/records/{RECORD}/files/{FILENAME}/content"
    with urlopen(Request(url, headers={"User-Agent": "closure-reproducibility"}), timeout=60) as response:
        payload = response.read()
    # MD5 is the archive checksum published by the data host, not an authentication mechanism.
    if len(payload) != EXPECTED_BYTES or hashlib.md5(payload).hexdigest() != EXPECTED_MD5:
        raise ValueError("The downloaded archive differs from the recorded dataset version")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive member: " + name)
        args.output.mkdir(parents=True)
        (args.output / FILENAME).write_bytes(payload)
        archive.extractall(args.output / "extracted")
    record = dict(doi="10.5281/zenodo.19699588", title="Rheological Data For Syntactic Foam",
                  creators=["Caleb Beckwith", "Nikhil Gupta"], license="CC-BY-4.0",
                  url=url, archive=FILENAME, bytes=len(payload), md5=EXPECTED_MD5,
                  sha256=hashlib.sha256(payload).hexdigest())
    (args.output / "DATA_SOURCE.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(status="verified", archive_bytes=len(payload))))


if __name__ == "__main__":
    main()
