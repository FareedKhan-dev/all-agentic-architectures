#!/bin/bash
# dev_run.sh - Script de développement complet: venv + front build + backend launch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PORT="${PORT:-8010}"

echo "🚀 Démarrage de l'environnement de développement..."

# 1. Vérifier l'environnement Python
source "$SCRIPT_DIR/ensure_venv.sh"

# 2. Vérifier si uvicorn est déjà en cours
if [[ -f "$PROJECT_ROOT/uvicorn.pid" ]]; then
    PID=$(cat "$PROJECT_ROOT/uvicorn.pid")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "⚠️  uvicorn déjà en cours (PID: $PID)"
        echo "💡 Utilisez scripts/restart.sh pour redémarrer"
        exit 1
    else
        echo "🧹 Nettoyage du pidfile obsolète..."
        rm -f "$PROJECT_ROOT/uvicorn.pid"
    fi
fi

# 3. Libérer le port si nécessaire
if ! "$SCRIPT_DIR/port_free.sh" "$PORT"; then
    # Si le port est occupé par autre chose, essayer 8011
    if [[ "$PORT" == "8010" ]]; then
        echo "🔄 Basculement vers le port 8011..."
        PORT=8011
        if ! "$SCRIPT_DIR/port_free.sh" "$PORT"; then
            echo "❌ Aucun port libre trouvé (8010, 8011)"
            exit 1
        fi
    else
        echo "❌ Port $PORT non disponible"
        exit 1
    fi
fi

# 4. Configuration Node et build frontend
echo "📦 Configuration Node..."
if [[ -f "$PROJECT_ROOT/.nvmrc" ]]; then
    # Charger nvm si disponible
    if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
        source "$HOME/.nvm/nvm.sh"
        nvm use
    else
        echo "⚠️  nvm non trouvé, utilisation de la version Node actuelle"
    fi
else
    echo "⚠️  Fichier .nvmrc non trouvé, utilisation de la version Node actuelle"
fi

echo "🏗️  Build du frontend..."
cd "$PROJECT_ROOT/frontend"
npm ci > /dev/null
npm run build > /dev/null
cd "$PROJECT_ROOT"

# 5. Lancement du backend
echo "🚀 Lancement du backend sur le port $PORT..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Lancer uvicorn en arrière-plan
uvicorn backend.app:app \
    --host 127.0.0.1 \
    --port "$PORT" \
    --workers 1 \
    --log-level info &

# Sauvegarder le PID
echo $! > "$PROJECT_ROOT/uvicorn.pid"

# Attendre que le serveur démarre
echo "⏳ Attente du démarrage du serveur..."
sleep 3

# 6. Test de santé
echo "🏥 Test de santé..."
if curl -sf "http://127.0.0.1:$PORT/healthz" > /dev/null; then
    echo "✅ Backend opérationnel sur http://127.0.0.1:$PORT"
    echo "📊 Health check: $(curl -s "http://127.0.0.1:$PORT/healthz" | grep -o '"ok":[^,]*' | cut -d: -f2 || echo "unknown")"
else
    echo "❌ Échec du test de santé"
    exit 1
fi

echo ""
echo "🎉 Environnement de développement prêt !"
echo "🌐 Frontend: http://127.0.0.1:$PORT"
echo "🔧 API: http://127.0.0.1:$PORT/docs"
echo "🛑 Arrêt: scripts/port_free.sh $PORT"