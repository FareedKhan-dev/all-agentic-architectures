#!/usr/bin/env bash
set -euo pipefail
# Détection robuste de la racine du projet — évaluée à l'exécution, pas à l'écriture
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

# venv
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate

# deps backend
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

# build frontend si nécessaire
if [ -d frontend ] && [ ! -d frontend/dist ]; then
  ( cd frontend && npm i && npm run build )
fi

# run uvicorn (dev)
export PYTHONPATH="$ROOT"
exec python -m uvicorn backend.app:app --host 127.0.0.1 --port 8010 --workers 1 --log-level info
