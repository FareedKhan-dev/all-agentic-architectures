#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DEV="$ROOT/backend/ops/env_templates/.env.example"
SRC_PROD="$ROOT/backend/ops/env_templates/.env.production.example"

usage() {
  echo "Usage:"
  echo "  DEV  : $0 dev    # copie .env.example -> .env (si .env absent, à la racine)"
  echo "  PROD : $0 prod   # copie .env.production.example -> /etc/all-agentic/.env (si absent)"
  exit 1
}

case "${1:-}" in
  dev)
    if [ -f "$ROOT/.env" ]; then
      echo "[INFO] .env existe déjà, rien à faire."
    else
      cp "$SRC_DEV" "$ROOT/.env"
      echo "[OK] Copié $SRC_DEV -> $ROOT/.env (à personnaliser)."
    fi
    ;;
  prod)
    sudo mkdir -p /etc/all-agentic
    if [ -f /etc/all-agentic/.env ]; then
      echo "[INFO] /etc/all-agentic/.env existe déjà, rien à faire."
    else
      sudo cp "$SRC_PROD" /etc/all-agentic/.env
      sudo chmod 600 /etc/all-agentic/.env
      echo "[OK] Copié $SRC_PROD -> /etc/all-agentic/.env (à personnaliser)."
    fi
    ;;
  *)
    usage
    ;;
fi
