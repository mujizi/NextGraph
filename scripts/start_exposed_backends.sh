#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

CONDA_ENV_NAME="${NEXTGRAPH_CONDA_ENV:-nextgraph}"
PYTHON_BIN="${NEXTGRAPH_PYTHON_BIN:-python}"
QUERY_API_SCRIPT="$ROOT_DIR/backend/scripts/exposed_query_api.py"
ENTITY_GRAPH_API_SCRIPT="$ROOT_DIR/backend/scripts/exposed_entity_graph_api.py"
QUERY_API_PORT="${NEXTGRAPH_QUERY_API_PORT:-8710}"
ENTITY_GRAPH_PORT="${NEXTGRAPH_ENTITY_GRAPH_PORT:-8711}"

cd "$ROOT_DIR"

start_query_api() {
  if command -v conda >/dev/null 2>&1; then
    NEXTGRAPH_BACKEND_PORT="$QUERY_API_PORT" \
      conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$QUERY_API_SCRIPT"
  else
    NEXTGRAPH_BACKEND_PORT="$QUERY_API_PORT" "$PYTHON_BIN" "$QUERY_API_SCRIPT"
  fi
}

start_entity_graph_api() {
  if command -v conda >/dev/null 2>&1; then
    NEXTGRAPH_BACKEND_PORT="$ENTITY_GRAPH_PORT" \
      conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$ENTITY_GRAPH_API_SCRIPT"
  else
    NEXTGRAPH_BACKEND_PORT="$ENTITY_GRAPH_PORT" "$PYTHON_BIN" "$ENTITY_GRAPH_API_SCRIPT"
  fi
}

start_query_api &
QUERY_PID=$!

start_entity_graph_api &
ENTITY_PID=$!

cleanup() {
  kill "$QUERY_PID" "$ENTITY_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

wait "$QUERY_PID" "$ENTITY_PID"
