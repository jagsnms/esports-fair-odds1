#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

python_bin=""
if command -v python3.11 >/dev/null 2>&1; then
  python_bin="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
fi

if [[ -z "${python_bin}" ]]; then
  echo "ERROR: python3.11 or python3 is required on PATH." >&2
  exit 1
fi

if [[ ! -f "requirements-dev.txt" ]]; then
  echo "ERROR: requirements-dev.txt was not found in ${repo_root}." >&2
  exit 1
fi

"${python_bin}" -m pip install --upgrade pip
"${python_bin}" -m pip install -r requirements-dev.txt

echo ""
echo "Bootstrap complete."
echo "Run tests with: ${python_bin} -m pytest -q"
