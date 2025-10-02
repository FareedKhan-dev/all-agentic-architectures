#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"

echo "== PROBE /healthz =="
curl -sS "http://$HOST:$PORT/healthz" | python - <<'PY'
import sys, json
print(json.dumps(json.load(sys.stdin), indent=2, ensure_ascii=False))
PY

echo -e "\n== PROBE A01 Reflection =="
curl -sS "http://$HOST:$PORT/api/a01/run" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explique la loi des grands nombres simplement."}' \
| python - <<'PY'
import sys, json
print(json.dumps(json.load(sys.stdin), indent=2, ensure_ascii=False))
PY

echo -e "\n== PROBE RAG QA =="
curl -sS "http://$HOST:$PORT/api/rag/qa" \
  -H 'Content-Type: application/json' \
  -d '{"question":"Quels thèmes sont évoqués dans la démo RAG ?","k":5}' \
| python - <<'PY'
import sys, json
print(json.dumps(json.load(sys.stdin), indent=2, ensure_ascii=False))
PY

echo -e "\n== PROBE A02 Tool Use (si TAVILY_API_KEY présent) =="
curl -sS "http://$HOST:$PORT/api/a02/run" \
  -H 'Content-Type: application/json' \
  -d '{"question":"Quelles ont été les annonces majeures de la dernière WWDC ?"}' \
| python - <<'PY'
import sys, json
raw = sys.stdin.read().strip()
if not raw:
    print("[WARN] vide (peut-être Tavily absent)"); sys.exit(0)
try:
    print(json.dumps(json.loads(raw), indent=2, ensure_ascii=False))
except Exception:
    print("[WARN] réponse non JSON:")
    print(raw)
PY
