#!/bin/bash
# port_free.sh - Libère un port donné en tuant le process uvicorn correspondant
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PORT="${1:-8010}"

echo "🔍 Vérification du port $PORT..."

# Vérifier si le port est occupé
if ! lsof -i ":$PORT" -P -n > /dev/null 2>&1; then
    echo "✅ Port $PORT libre"
    exit 0
fi

# Récupérer les informations sur le process
PROCESS_INFO=$(lsof -i ":$PORT" -P -n | tail -n +2)
if [[ -z "$PROCESS_INFO" ]]; then
    echo "✅ Port $PORT libre"
    exit 0
fi

PID=$(echo "$PROCESS_INFO" | awk '{print $2}')
CMD=$(echo "$PROCESS_INFO" | awk '{print $1}')

echo "⚠️  Port $PORT occupé par $CMD (PID: $PID)"

# Vérifier si c'est un uvicorn de ce repo
FULL_CMD=$(ps -p "$PID" -o args= 2>/dev/null || echo "")
if echo "$FULL_CMD" | grep -q "uvicorn.*backend.app:app"; then
    echo "🔄 Arrêt du process uvicorn existant..."
    kill "$PID"
    sleep 2
    
    # Vérifier que le process est bien arrêté
    if ! lsof -i ":$PORT" -P -n > /dev/null 2>&1; then
        echo "✅ Port $PORT libéré"
    else
        echo "❌ Échec de l'arrêt du process"
        exit 1
    fi
else
    echo "⚠️  Port $PORT occupé par un autre service ($CMD)"
    echo "💡 Utilisez un autre port avec: PORT=8011 $0"
    exit 1
fi
