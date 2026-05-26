#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

PORT="${NEXTGRAPH_FRONTEND_PORT:-${PORT:-5173}}"

PIDS="$(lsof -ti tcp:"$PORT" || true)"
if [ -n "$PIDS" ]; then
  kill $PIDS
fi

exec "$ROOT_DIR/scripts/start_frontend.sh"
