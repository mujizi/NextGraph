#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

BACKEND_PORT="${NEXTGRAPH_BACKEND_PORT:-5190}"
FRONTEND_PORT="${NEXTGRAPH_FRONTEND_PORT:-5173}"

stop_pid_file() {
  local pid_file="$1"
  local name="$2"
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file")"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid"
      echo "已停止$name进程 PID=$pid"
    fi
    rm -f "$pid_file"
  fi
}

stop_port() {
  local port="$1"
  local name="$2"
  local pids
  pids="$(lsof -ti tcp:"$port" || true)"
  if [ -n "$pids" ]; then
    kill $pids
    echo "已停止$name端口 $port 上的进程"
  fi
}

stop_pid_file "$RUN_DIR/backend-${BACKEND_PORT}.pid" "后端"
stop_pid_file "$RUN_DIR/frontend-${FRONTEND_PORT}.pid" "前端"

stop_port "$BACKEND_PORT" "后端"
stop_port "$FRONTEND_PORT" "前端"
