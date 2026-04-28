#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv-mineru}"

if [ ! -d "$VENV_DIR" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 "$VENV_DIR"
  elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv "$VENV_DIR"
  else
    echo "Python 3.11 is required. Install python3.11 or uv first." >&2
    exit 1
  fi
fi

if [ -f backend/.env ]; then
  set -a
  source backend/.env
  set +a
fi

"$VENV_DIR/bin/python" -m pip install -r backend/requirements.txt
exec "$VENV_DIR/bin/uvicorn" backend.app.main:app --host 127.0.0.1 --port "${PORT:-8001}"
