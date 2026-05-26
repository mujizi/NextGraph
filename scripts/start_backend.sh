#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

BACKEND_HOST="${NEXTGRAPH_BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${NEXTGRAPH_BACKEND_PORT:-5190}"
CONDA_ENV_NAME="${NEXTGRAPH_CONDA_ENV:-nextgraph}"
PYTHON_BIN="${NEXTGRAPH_PYTHON_BIN:-python}"

cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  exec conda run --no-capture-output -n "$CONDA_ENV_NAME" \
    python -m uvicorn backend.app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
fi

exec "$PYTHON_BIN" -m uvicorn backend.app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
