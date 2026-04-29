#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/tmp/logs"

FRONTEND_PORT="${FRONTEND_PORT:-${PORT:-5173}}"
BACKEND_PORT="${BACKEND_PORT:-8001}"

mkdir -p "$LOG_DIR"

stop_port() {
  local name="$1"
  local port="$2"
  local pids

  pids="$(lsof -ti tcp:"$port" || true)"
  if [ -n "$pids" ]; then
    echo "Stopping ${name} on port ${port}: ${pids}"
    kill $pids
  else
    echo "${name} is not running on port ${port}"
  fi
}

start_service() {
  local name="$1"
  local port="$2"
  local script="$3"
  local log_file="$4"

  echo "Starting ${name} on port ${port}"
  PORT="$port" nohup "$script" >"$log_file" 2>&1 &
  echo "${name} pid: $! logs: ${log_file}"
}

stop_port "frontend" "$FRONTEND_PORT"
stop_port "backend" "$BACKEND_PORT"

sleep 1

start_service "backend" "$BACKEND_PORT" "$ROOT_DIR/scripts/start_backend.sh" "$LOG_DIR/backend.log"
start_service "frontend" "$FRONTEND_PORT" "$ROOT_DIR/scripts/start_frontend.sh" "$LOG_DIR/frontend.log"

echo "Done."
