#!/bin/bash
# ensure_venv.sh - Crée/valide le venv, upgrade pip, installe backend/requirements.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧 Vérification de l'environnement Python..."

# Vérifier si le venv existe et est valide
if [[ ! -d "$PROJECT_ROOT/.venv" ]] || [[ ! -x "$PROJECT_ROOT/.venv/bin/pip" ]]; then
    echo "📦 Création d'un nouveau venv..."
    rm -rf "$PROJECT_ROOT/.venv"
    python3 -m venv "$PROJECT_ROOT/.venv"
fi

# Activer le venv
source "$PROJECT_ROOT/.venv/bin/activate"

# Upgrade pip
echo "⬆️  Mise à jour de pip..."
python -m pip install -U pip > /dev/null

# Installer les dépendances
echo "📚 Installation des dépendances backend..."
pip install -r "$PROJECT_ROOT/backend/requirements.txt" > /dev/null

echo "✅ Environnement Python prêt (pip $(pip --version | cut -d' ' -f2))"
