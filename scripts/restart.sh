#!/bin/bash
# restart.sh - Redémarre l'environnement de développement
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PORT="${PORT:-8010}"

echo "🔄 Redémarrage de l'environnement de développement..."

# 1. Arrêter le process existant
echo "🛑 Arrêt du process existant..."
"$SCRIPT_DIR/port_free.sh" "$PORT" || true

# 2. Nettoyer les fichiers temporaires
rm -f "$PROJECT_ROOT/uvicorn.pid"

# 3. Relancer
echo "🚀 Relancement..."
"$SCRIPT_DIR/dev_run.sh"
