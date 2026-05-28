# Docker

Optional. The library runs fine on a plain Python ≥3.10 venv; Docker is here for two cases:

1. **Reproducible dev environment** — same image across contributors, no host-Python conflicts.
2. **Optional stateful services** — Neo4j (for graph notebooks), Qdrant (alt vector store), Ollama (fully-local LLM).

## TL;DR

```bash
# From the repo root
cp .env.example .env
# Fill in NEBIUS_API_KEY, TAVILY_API_KEY, etc.

docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec app bash
```

Inside the `app` container:

```bash
pytest -q                                    # 283 tests, no LLM cost
python -c "from agentic_architectures.architectures import Reflexion; Reflexion().run('hi')"
python -m mkdocs serve --dev-addr 0.0.0.0:8000   # then visit http://localhost:8000/all-agentic-architectures/
```

## What gets built

| Service | Image | Port(s) | Used by |
|---|---|---|---|
| `app` | Built from `docker/Dockerfile` (Python 3.11-slim + the package + Playwright Chromium) | 8000, 8888 | All notebooks. Source is bind-mounted from the host so edits are live. |
| `neo4j` | `neo4j:5-community` | 7474 (UI), 7687 (Bolt) | Notebooks 08, 12, 27 when `GRAPH_BACKEND=neo4j` in `.env`. |
| `qdrant` | `qdrant/qdrant:latest` | 6333 (HTTP), 6334 (gRPC) | Optional vector backend swap for FAISS. |
| `ollama` | `ollama/ollama:latest` | 11434 | Optional fully-local LLM. Set `LLM_PROVIDER=ollama` in `.env` + `ollama pull llama3.3:70b` once. |

## Use only the `app` service

If you don't need Neo4j / Qdrant / Ollama:

```bash
docker compose -f docker/docker-compose.yml up -d app
```

The other services won't start, but `app` won't fail because they're depended on with `condition: service_started`/`service_healthy` — you can skip starting them by editing the `depends_on` block out.

## GitHub Codespaces

This repo ships a `.devcontainer/devcontainer.json` that points at the same `docker/Dockerfile`. Click "Code → Codespaces → New codespace" on GitHub and you get the full environment in ~3 minutes — no local Docker needed.

## Image size

~1.8 GB after `playwright install --with-deps chromium`. The bulk is the Chromium binary + system libraries; the Python deps are ~400 MB.

If you don't need the BrowserAgent (nb 34) you can remove the `playwright install --with-deps chromium` line from the Dockerfile and save ~1 GB.

## Stop / clean up

```bash
docker compose -f docker/docker-compose.yml down              # stop containers, keep volumes
docker compose -f docker/docker-compose.yml down -v           # also wipe Neo4j/Qdrant/Ollama data
docker rmi agentic-arch:dev                                   # remove the image
```
