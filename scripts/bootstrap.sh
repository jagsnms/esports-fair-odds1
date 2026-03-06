#!/usr/bin/env bash
# Bootstrap script for Linux/macOS: create venv and install dev/test deps.
# Run from repo root: ./scripts/bootstrap.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if command -v python3.11 >/dev/null 2>&1; then
  PY_BIN="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
else
  echo "ERROR: Python 3 not found in PATH." >&2
  exit 1
fi

VENV_DIR=".venv311"
if [[ -d "$VENV_DIR" && ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Existing $VENV_DIR is incomplete; recreating." >&2
  rm -rf "$VENV_DIR"
fi

if [[ -d "$VENV_DIR" ]]; then
  echo "Using existing $VENV_DIR at $REPO_ROOT/$VENV_DIR"
else
  echo "Creating venv at $VENV_DIR (Python: $PY_BIN)"
  if ! "$PY_BIN" -m venv "$VENV_DIR"; then
    echo "python -m venv failed; falling back to virtualenv bootstrap." >&2
    rm -rf "$VENV_DIR"
    "$PY_BIN" -m pip install --user --upgrade virtualenv
    "$PY_BIN" -m virtualenv "$VENV_DIR"
  fi
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: venv activation script not found at $VENV_DIR/bin/activate" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ ! -f "requirements-dev.txt" ]]; then
  echo "ERROR: requirements-dev.txt not found at $REPO_ROOT/requirements-dev.txt" >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

if ! python -m playwright install; then
  echo "WARNING: playwright install failed (some environments may not need it)." >&2
fi

echo
echo "Done."
echo "Run tests with: ./.venv311/bin/python -m pytest -q"
