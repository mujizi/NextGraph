#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR/frontend"

# 新前端默认复用当前 frontend/，但允许单独指定端口。
export VITE_API_BASE_URL="${VITE_API_BASE_URL:-}"
export VITE_QUERY_API_BASE_URL="${VITE_QUERY_API_BASE_URL:-http://127.0.0.1:${NEXTGRAPH_QUERY_API_PORT:-8710}}"
export VITE_ENTITY_GRAPH_API_BASE_URL="${VITE_ENTITY_GRAPH_API_BASE_URL:-http://127.0.0.1:${NEXTGRAPH_ENTITY_GRAPH_PORT:-8711}}"
export VITE_DEFAULT_KB_ID="${VITE_DEFAULT_KB_ID:-${NEXTGRAPH_DEFAULT_KB_ID:-0616}}"
export NEXTGRAPH_FRONTEND_PORT="${NEXTGRAPH_NEW_FRONTEND_PORT:-${NEXTGRAPH_FRONTEND_PORT:-5173}}"
export NEXTGRAPH_BACKEND_HOST="${NEXTGRAPH_BACKEND_HOST:-127.0.0.1}"
export NEXTGRAPH_BACKEND_PORT="${NEXTGRAPH_BACKEND_PORT:-5190}"

if [ ! -d node_modules ]; then
  npm install
fi

exec npm run dev -- --port "$NEXTGRAPH_FRONTEND_PORT"
