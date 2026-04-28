#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8004}"
GPUS="${GPUS:-0,1,2,3}"
VENV_DIR="${VENV_DIR:-.venv-mineru}"
MINERU_ROUTER_BIN="${MINERU_ROUTER_BIN:-}"

if [ -z "$MINERU_ROUTER_BIN" ] && [ -x "$VENV_DIR/bin/mineru-router" ]; then
  MINERU_ROUTER_BIN="$VENV_DIR/bin/mineru-router"
fi

if [ -z "$MINERU_ROUTER_BIN" ] && command -v mineru-router >/dev/null 2>&1; then
  MINERU_ROUTER_BIN="$(command -v mineru-router)"
fi

if [ -z "$MINERU_ROUTER_BIN" ]; then
  echo "mineru-router command not found. Please install MinerU in this environment first." >&2
  exit 1
fi

echo "Starting MinerU Router on ${HOST}:${PORT} with GPUs: ${GPUS}"
exec "$MINERU_ROUTER_BIN" \
  --host "$HOST" \
  --port "$PORT" \
  --local-gpus "$GPUS"
