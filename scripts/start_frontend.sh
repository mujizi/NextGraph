#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/frontend"

PORT="${PORT:-5173}"

if [ ! -d node_modules ]; then
  npm install
fi

npm run dev -- --port "$PORT"
