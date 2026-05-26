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
BACKEND_PID_FILE="$RUN_DIR/backend-${BACKEND_PORT}.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend-${FRONTEND_PORT}.pid"

print_pid_status() {
  local pid_file="$1"
  local name="$2"
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file")"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "$name PID 文件: $pid_file"
      echo "$name PID 状态: 运行中 (PID=$pid)"
      return
    fi
    echo "$name PID 文件: $pid_file"
    echo "$name PID 状态: 文件存在，但进程不在运行"
    return
  fi
  echo "$name PID 文件: 未找到 ($pid_file)"
}

print_port_status() {
  local port="$1"
  local name="$2"
  local lines
  lines="$(lsof -nP -i tcp:"$port" 2>/dev/null || true)"
  if [ -n "$lines" ]; then
    echo "$name 端口状态: $port 正在监听"
    echo "$lines"
    return
  fi
  echo "$name 端口状态: $port 未监听"
}

echo "NextGraph Dev Status"
echo "项目目录: $ROOT_DIR"
echo "配置文件: $ROOT_DIR/.env"
echo "后端端口: $BACKEND_PORT"
echo "前端端口: $FRONTEND_PORT"
echo "Conda 环境: ${NEXTGRAPH_CONDA_ENV:-nextgraph}"
echo
print_pid_status "$BACKEND_PID_FILE" "后端"
echo
print_pid_status "$FRONTEND_PID_FILE" "前端"
echo
print_port_status "$BACKEND_PORT" "后端"
echo
print_port_status "$FRONTEND_PORT" "前端"
