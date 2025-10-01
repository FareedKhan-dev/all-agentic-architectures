#!/usr/bin/env bash
set -euo pipefail

# Config
PROJECT="${PROJECT:-$HOME/workspace-agents/all-agentic}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
LOGFILE="$PROJECT/uvicorn_all_agentic.log"
PIDFILE="$PROJECT/uvicorn.pid"
VENV="$PROJECT/.venv"
PY="$VENV/bin/python"
FRONT="$PROJECT/frontend"
DIST="$FRONT/dist"

port_listening() {
  if command -v ss >/dev/null 2>&1; then ss -ltnp | grep -q ":$PORT"; return $?; fi
  if command -v lsof >/dev/null 2>&1; then lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; return $?; fi
  return 1
}

ensure_venv() {
  if [ ! -d "$VENV" ]; then echo "[ERR] venv missing: $VENV" >&2; exit 1; fi
}

ensure_node() {
  if ! command -v npm >/dev/null 2>&1; then
    echo "[ERR] npm not found. Please install NodeJS (>=18) and re-run." >&2; exit 1
  fi
}

build() {
  ensure_node
  mkdir -p "$FRONT"
  cd "$FRONT"
  # Install deps and build
  if [ -f package-lock.json ]; then npm ci; else npm install; fi
  npm run build
}

start() {
  ensure_venv
  # Optional build if dist missing
  if [ ! -d "$DIST" ]; then echo "[INFO] frontend/dist missing → build"; build; fi
  # Stop existing on port
  stop || true
  echo "[INFO] starting uvicorn on $HOST:$PORT"
  cd "$PROJECT"
  nohup env PYTHONPATH="$PROJECT" "$PY" -m uvicorn backend.app:app \
    --host "$HOST" --port "$PORT" --workers 1 --log-level info \
    > "$LOGFILE" 2>&1 & echo $! > "$PIDFILE"
  # Wait port
  for i in $(seq 1 40); do
    if port_listening; then echo "[OK] listening on $HOST:$PORT"; break; fi
    sleep 0.25
  done
  if ! port_listening; then echo "[ERR] port not listening" >&2; tail -n 200 "$LOGFILE" || true; exit 1; fi
  echo "[INFO] URL: http://$HOST:$PORT/"
}

stop() {
  local rc=0
  if [ -f "$PIDFILE" ]; then
    local pid; pid=$(cat "$PIDFILE" || true)
    if [ -n "${pid:-}" ] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "[INFO] killing pid $pid"; kill "$pid" || true; sleep 1
    fi
    rm -f "$PIDFILE"
  fi
  # Ensure port freed
  if port_listening; then
    echo "[INFO] freeing port $PORT"
    if command -v fuser >/dev/null 2>&1; then fuser -k ${PORT}/tcp || true; else rc=1; fi
  fi
  return $rc
}

status() {
  echo "[STATUS] PROJECT=$PROJECT HOST=$HOST PORT=$PORT"
  if port_listening; then echo "[STATUS] port listening"; else echo "[STATUS] port closed"; fi
  if [ -f "$PIDFILE" ]; then echo "[STATUS] pid=$(cat "$PIDFILE")"; fi
}

logs() {
  tail -n 200 "$LOGFILE" || true
}

open_browser() {
  python - <<PY
import webbrowser
print(webbrowser.open("http://$HOST:$PORT/"))
PY
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>
  build        Build frontend (npm install + npm run build)
  start        Start API + serve frontend (prod-like)
  stop         Stop server and free port
  restart      Stop then start
  status       Show server status
  logs         Tail recent logs
  open         Open browser to /
EOF
}

cmd="${1:-}"
case "$cmd" in
  build) build ;;
  start) start ;;
  stop) stop ;;
  restart) stop || true; start ;;
  status) status ;;
  logs) logs ;;
  open) open_browser ;;
  *) usage; exit 1 ;;
esac
