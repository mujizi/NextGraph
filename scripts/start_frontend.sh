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

# 允许 VITE_API_BASE_URL 为空，从而使用 Vite 代理
export VITE_API_BASE_URL="${VITE_API_BASE_URL:-}"
export NEXTGRAPH_FRONTEND_PORT="${NEXTGRAPH_FRONTEND_PORT:-5173}"
export NEXTGRAPH_BACKEND_HOST="${NEXTGRAPH_BACKEND_HOST:-127.0.0.1}"
export NEXTGRAPH_BACKEND_PORT="${NEXTGRAPH_BACKEND_PORT:-5190}"

if [ ! -d node_modules ]; then
  npm install
fi

exec npm run dev -- --port "$NEXTGRAPH_FRONTEND_PORT"
