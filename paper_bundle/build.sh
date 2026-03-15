#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TECTONIC_BIN="${TECTONIC_BIN:-tectonic}"
TEX_ENTRY="${1:-main.tex}"

if [[ -x "${TECTONIC_BIN}" ]]; then
  : "TECTONIC_BIN is an executable path."
elif command -v "${TECTONIC_BIN}" >/dev/null 2>&1; then
  TECTONIC_BIN="$(command -v "${TECTONIC_BIN}")"
else
  echo "tectonic not found." >&2
  echo "Install tectonic (recommended) or set TECTONIC_BIN to a local binary path." >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/build"
"${TECTONIC_BIN}" --outdir "${SCRIPT_DIR}/build" "${SCRIPT_DIR}/${TEX_ENTRY}"
