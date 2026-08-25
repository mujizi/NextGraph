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
QUERY_API_PORT="${NEXTGRAPH_QUERY_API_PORT:-8710}"
ENTITY_GRAPH_PORT="${NEXTGRAPH_ENTITY_GRAPH_PORT:-8711}"
FRONTEND_PORT="${NEXTGRAPH_NEW_FRONTEND_PORT:-${NEXTGRAPH_FRONTEND_PORT:-5173}}"

kill_descendants() {
  local parent_pid="$1"
  local children
  children="$(pgrep -P "$parent_pid" || true)"
  if [ -n "$children" ]; then
    local child
    for child in $children; do
      kill_descendants "$child"
    done
    kill $children 2>/dev/null || true
  fi
}

wait_for_port_release() {
  local port="$1"
  local name="$2"
  local attempt
  for attempt in $(seq 1 20); do
    if ! lsof -ti tcp:"$port" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "$name端口 $port 仍未释放"
  lsof -nP -i tcp:"$port" 2>/dev/null || true
  return 1
}

force_stop_port() {
  local port="$1"
  local name="$2"
  local pids
  pids="$(lsof -ti tcp:"$port" || true)"
  if [ -n "$pids" ]; then
    kill -9 $pids 2>/dev/null || true
    echo "已强制停止$name端口 $port 上的进程"
  fi
}

stop_pid_file() {
  local pid_file="$1"
  local name="$2"
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file")"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      kill_descendants "$pid"
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

stop_pid_file "$RUN_DIR/new-backend-${BACKEND_PORT}.pid" "旧后端"
stop_pid_file "$RUN_DIR/new-backend-query-${QUERY_API_PORT}-entity-${ENTITY_GRAPH_PORT}.pid" "暴露后端"
stop_pid_file "$RUN_DIR/new-frontend-${FRONTEND_PORT}.pid" "新前端"

stop_port "$BACKEND_PORT" "旧后端"
stop_port "$QUERY_API_PORT" "查询后端"
stop_port "$ENTITY_GRAPH_PORT" "实体图后端"
stop_port "$FRONTEND_PORT" "新前端"

wait_for_port_release "$BACKEND_PORT" "旧后端" || {
  force_stop_port "$BACKEND_PORT" "旧后端"
  wait_for_port_release "$BACKEND_PORT" "旧后端"
}
wait_for_port_release "$QUERY_API_PORT" "查询后端" || {
  force_stop_port "$QUERY_API_PORT" "查询后端"
  wait_for_port_release "$QUERY_API_PORT" "查询后端"
}
wait_for_port_release "$ENTITY_GRAPH_PORT" "实体图后端" || {
  force_stop_port "$ENTITY_GRAPH_PORT" "实体图后端"
  wait_for_port_release "$ENTITY_GRAPH_PORT" "实体图后端"
}
wait_for_port_release "$FRONTEND_PORT" "新前端" || {
  force_stop_port "$FRONTEND_PORT" "新前端"
  wait_for_port_release "$FRONTEND_PORT" "新前端"
}
