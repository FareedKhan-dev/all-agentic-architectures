#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

# activer venv si présent
if [ -d .venv ]; then source .venv/bin/activate; fi

export PYTHONPATH="$ROOT"
python -m backend.ingestion.build_faiss --rebuild
