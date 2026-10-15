"""Check the exact Git index for private paths, credentials and unsafe artifacts.

Run after staging the intended public files. This is a conservative publication
check, supplemented by review of the staged diff. It prints file names and rule
names rather than any matching secret or private value.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
from pathlib import Path
import re
import subprocess
import tokenize

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "personal_home_or_mount_path": re.compile(rb"/(?:home|Users|mnt|media)/[A-Za-z0-9_.-]+/"),
    "windows_profile_path": re.compile(rb"[A-Za-z]:\\(?:Users|Documents and Settings)\\"),
    "private_key_material": re.compile(rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "github_access_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "email_address": re.compile(rb"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}
DISALLOWED_PARTS = {".venv", ".ssh", ".aws", "__pycache__", "manuscript", "local_preview", "research-sandbox"}
DISALLOWED_NAMES = {".env", "id_rsa", "id_ed25519", "credentials", "credentials.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    findings, count, npz_count = [], 0, 0
    for name in filter(None, names):
        path = Path(name)
        if DISALLOWED_PARTS.intersection(path.parts) or path.name in DISALLOWED_NAMES:
            findings.append(dict(file=name, rule="private_or_generated_artifact"))
        data = subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
        payloads = [data]
        email_payloads = payloads
        if path.suffix == ".py":
            source = data.decode()
            strings = [node.value for node in ast.walk(ast.parse(source))
                       if isinstance(node, ast.Constant) and isinstance(node.value, str)]
            strings.extend(token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
                           if token.type == tokenize.COMMENT)
            # A @ B.attribute is a Python matrix product, not an email address.
            email_payloads = [value.encode() for value in strings]
        if path.suffix == ".npz":
            with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                for key in arrays.files:
                    array = arrays[key]
                    if array.dtype.kind in "SU":
                        payloads.append(str(array.tolist()).encode())
            npz_count += 1
        for rule, pattern in RULES.items():
            candidates = email_payloads if rule == "email_address" else payloads
            if any(pattern.search(payload) for payload in candidates):
                findings.append(dict(file=name, rule=rule))
        if len(data) > 95 * 1024 * 1024:
            findings.append(dict(file=name, rule="oversized_git_blob"))
        count += 1
    result = dict(status="passed" if not findings else "failed", scope="exact Git index",
                  files=count, npz_metadata_checked=npz_count, findings=findings)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
