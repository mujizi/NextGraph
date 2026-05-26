#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
mkdir -p "$RUN_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

BACKEND_PORT="${NEXTGRAPH_BACKEND_PORT:-5190}"
FRONTEND_PORT="${NEXTGRAPH_FRONTEND_PORT:-5173}"
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"
FRONTEND_URL="http://127.0.0.1:${FRONTEND_PORT}"

ensure_port_free() {
  local port="$1"
  local name="$2"
  local lines
  lines="$(lsof -nP -i tcp:"$port" 2>/dev/null || true)"
  if [ -n "$lines" ]; then
    echo "$name 端口 $port 已被占用，请先修改 /opt/Workspace/CRX/NextGraph/.env 或停止占用进程。"
    echo "$lines"
    exit 1
  fi
}

ensure_port_free "$BACKEND_PORT" "后端"
ensure_port_free "$FRONTEND_PORT" "前端"

nohup "$ROOT_DIR/scripts/start_backend.sh" >"$RUN_DIR/backend-${BACKEND_PORT}.log" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" >"$RUN_DIR/backend-${BACKEND_PORT}.pid"

nohup "$ROOT_DIR/scripts/start_frontend.sh" >"$RUN_DIR/frontend-${FRONTEND_PORT}.log" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" >"$RUN_DIR/frontend-${FRONTEND_PORT}.pid"

echo "NextGraph 测试环境已启动"
echo "后端: $BACKEND_URL"
echo "前端: $FRONTEND_URL"
echo "后端日志: $RUN_DIR/backend-${BACKEND_PORT}.log"
echo "前端日志: $RUN_DIR/frontend-${FRONTEND_PORT}.log"
echo "后端 PID: $BACKEND_PID"
echo "前端 PID: $FRONTEND_PID"
