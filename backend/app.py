import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, Body
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse

# Charge .env depuis la racine du projet (all-agentic/.env), même si uvicorn est lancé ailleurs.
# __file__ = all-agentic/backend/app.py  → parents[1] = all-agentic/
_DOTENV_PATH = (Path(__file__).resolve().parents[1] / ".env")
if _DOTENV_PATH.exists():
    # override=False : ne pas écraser les variables déjà présentes dans l'environnement du process
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)
else:
    # Fallback : tenter un load_dotenv() simple sur le CWD
    load_dotenv(override=False)

from backend.core.schemas import (
    ReflectionIn, ReflectionOut,
    ToolUseIn, ToolUseOut,
    RAGQAIn, RAGQAOut,
)
from backend.agents import a01_reflection, a02_tool_use
from backend.agents import a09_tree_of_thoughts
from backend.agents import a05_pev
from backend.agents import a06_blackboard
from backend.agents import a08_graph_memory
from backend.agents import a10_ensemble_decision
from backend.agents import a11_meta_controller
from backend.agents import a12_self_ask
from backend.agents import a13_chain_of_verification
from backend.agents import a14_dry_run
from backend.agents import a15_rlhf_loop
from backend.agents import a16_simulator_mitl
from backend.agents import a17_reflexive_metacognitive
from backend.core.rag import retriever
from backend.core.llm import make_llm
from backend.core.patterns_data import PATTERNS
from backend.core.schemas import PatternListOut, PatternMeta, PatternExecIn, PatternExecOut
import time

app = FastAPI(title="All Agentic Architectures API (Local)")


@app.middleware("http")
async def add_headers(request, call_next):
    resp: Response = await call_next(request)
    resp.headers["X-LLM-Provider"] = os.getenv("DEFAULT_LLM_PROVIDER","")
    resp.headers["X-Embeddings-Provider"] = os.getenv("DEFAULT_EMBEDDING_PROVIDER","")
    return resp


def run_graph(build_fn, state: dict):
    try:
        graph = build_fn()
        return graph.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/a01/run", response_model=ReflectionOut)
def a01_run(body: ReflectionIn):
    state = {"prompt": body.prompt, "criteria": body.criteria or [], "draft": "", "review": "", "answer": ""}
    out = run_graph(a01_reflection.build_app, state)
    return {"answer": out.get("answer", ""), "critic": out.get("review")}


@app.post("/api/a02/run", response_model=ToolUseOut)
def a02_run(body: ToolUseIn):
    state = {"question": body.question, "hits": [], "sources": [], "answer": ""}
    out = run_graph(a02_tool_use.build_app, state)
    return {"answer": out.get("answer", ""), "sources": out.get("sources", [])}


@app.post("/api/rag/qa", response_model=RAGQAOut)
def rag_qa(body: RAGQAIn):
    try:
        ret = retriever(k=body.k)
        docs = ret.get_relevant_documents(body.question)
        context = "\n\n".join([f"- {d.metadata.get('source','?')}\n{d.page_content}" for d in docs])
        llm = make_llm()
        ans = llm.invoke(
            f"Question: {body.question}\nContexte (extraits):\n{context}\n\n"
            "Réponds en français et cite les chemins de fichiers en fin de réponse."
        ).content
        sources = [d.metadata.get("source", "") for d in docs]
        return {"answer": ans, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
def healthz():
    # NOTE: en PROD, ne renvoyer que {"ok": true, "ts": ...}
    return {"ok": True, "env": {
        "DEFAULT_LLM_PROVIDER": os.getenv("DEFAULT_LLM_PROVIDER", ""),
        "DEFAULT_LLM_MODEL": os.getenv("DEFAULT_LLM_MODEL", ""),
        "DEFAULT_EMBEDDING_PROVIDER": os.getenv("DEFAULT_EMBEDDING_PROVIDER", ""),
        "DEFAULT_EMBEDDING_MODEL": os.getenv("DEFAULT_EMBEDDING_MODEL", ""),
    }}

# ---------- Patterns Catalogue ----------
@app.get("/api/patterns", response_model=PatternListOut)
def list_patterns():
    return {"items": PATTERNS}

@app.get("/api/patterns/{pid}", response_model=PatternMeta)
def get_pattern(pid: str):
    for p in PATTERNS:
        if p.id_pattern == pid:
            return p
    raise HTTPException(status_code=404, detail=f"Pattern '{pid}' introuvable")

# ---------- Exécution (démo) ----------
_LAST_METRICS = {}

@app.post("/api/execute/{pid}", response_model=PatternExecOut)
def execute_pattern(pid: str, body: PatternExecIn):
    t0 = time.time()
    if pid == "tool_use":
        # démo: réutilise a02
        from backend.agents import a02_tool_use
        state = {"question": body.input.get("question","Quelles ont été les annonces majeures de la dernière WWDC ?"),
                 "hits": [], "sources": [], "answer": ""}
        graph = a02_tool_use.build_app()
        out = graph.invoke(state)
        dt = time.time() - t0
        metrics = {"latency_s": round(dt,3), "calls": {"llm":1,"tools":1}, "est_cost_usd":"~"}
        global _LAST_METRICS; _LAST_METRICS = {"id": pid, **metrics}
        return {"ok": True, "id_pattern": pid, "output": out, "metrics": metrics}
    if pid == "reflection":
        from backend.agents import a01_reflection
        state = {"prompt": body.input.get("prompt","Explique la loi des grands nombres simplement."),
                 "criteria": body.input.get("criteria", []),
                 "draft":"", "review":"", "answer":""}
        graph = a01_reflection.build_app()
        out = graph.invoke(state)
        dt = time.time() - t0
        metrics = {"latency_s": round(dt,3), "calls": {"llm":3}, "est_cost_usd":"~"}
        _LAST_METRICS = {"id": pid, **metrics}
        return {"ok": True, "id_pattern": pid, "output": out, "metrics": metrics}
    # par défaut: non implémenté côté démo
    raise HTTPException(status_code=501, detail=f"Demo non implémentée pour '{pid}'")

@app.get("/api/metrics/last")
def last_metrics():
    return _LAST_METRICS or {}

# ---------- Config provider (in-memory) ----------
@app.get("/api/config/provider")
def get_provider_config():
    return {
        "llm": {
            "provider": os.getenv("DEFAULT_LLM_PROVIDER", ""),
            "model": os.getenv("DEFAULT_LLM_MODEL", ""),
        },
        "embeddings": {
            "provider": os.getenv("DEFAULT_EMBEDDING_PROVIDER", ""),
            "model": os.getenv("DEFAULT_EMBEDDING_MODEL", ""),
        }
    }

@app.post("/api/config/provider")
def set_provider_config(
    llm: dict = Body(default={}),
    embeddings: dict = Body(default={})
):
    """Met à jour os.environ pour les appels suivants (ne modifie pas .env sur disque)."""
    if "provider" in llm:
        os.environ["DEFAULT_LLM_PROVIDER"] = str(llm["provider"]).lower()
    if "model" in llm:
        os.environ["DEFAULT_LLM_MODEL"] = str(llm["model"])
    if "provider" in embeddings:
        os.environ["DEFAULT_EMBEDDING_PROVIDER"] = str(embeddings["provider"]).lower()
    if "model" in embeddings:
        os.environ["DEFAULT_EMBEDDING_MODEL"] = str(embeddings["model"])
    return {"ok": True, "applied": get_provider_config()}

# ---- Static frontend (prod-like) ----
# If ./frontend/dist exists, mount it at root / so the SPA is served by the same origin as the API.
_DIST = (Path(__file__).resolve().parents[1] / "frontend" / "dist")
if _DIST.exists():
    # Serve built assets and index
    assets_dir = _DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir, html=False), name="frontend_assets")

    @app.get("/", include_in_schema=False)
    async def _index():
        return FileResponse(_DIST / "index.html")

    @app.get("/vite.svg", include_in_schema=False)
    async def _vite_svg():
        p = _DIST / "vite.svg"
        if p.exists():
            return FileResponse(p)
        raise HTTPException(status_code=404, detail="vite.svg non trouvé")

    @app.get("/favicon.ico", include_in_schema=False)
    async def _favicon():
        # Sert favicon.ico si présent, sinon fallback sur vite.svg
        ico = _DIST / "favicon.ico"
        if ico.exists():
            return FileResponse(ico)
        svg = _DIST / "vite.svg"
        if svg.exists():
            return FileResponse(svg)
        raise HTTPException(status_code=404, detail="favicon non trouvé")

@app.post("/api/a09/run")
def a09_run(body: dict):
    b = int(body.get("breadth", 3))
    d = int(body.get("depth", 2))
    s = {"task": body.get("task",""), "breadth": b, "depth": d, "frontier": [], "best": {}, "trace": [], "answer": ""}
    out = run_graph(a09_tree_of_thoughts.build_app, s)
    return {"answer": out.get("answer",""), "best": out.get("best",{}), "trace": out.get("trace",[]), "frontier": out.get("frontier",[])}

@app.post("/api/a10/run")
def a10_run(body: dict):
    s = {"prompt": body.get("prompt",""), "candidates": [], "votes": {}, "best": "", "answer": ""}
    out = run_graph(a10_ensemble_decision.build_app, s)
    return {"answer": out.get("answer",""), "candidates": out.get("candidates",[]), "votes": out.get("votes",{}), "best": out.get("best","")}

@app.post("/api/a11/run")
def a11_run(body: dict):
    q = body.get("query", "")
    out = a11_meta_controller.run(q)
    return out

@app.post("/api/a12/run")
def a12_run(body: dict):
    s = {"query": body.get("query",""), "subs": [], "partials": [], "answer": ""}
    out = run_graph(a12_self_ask.build_app, s)
    return {"answer": out.get("answer",""), "subs": out.get("subs",[]), "partials": out.get("partials",[])}

@app.post("/api/a13/run")
def a13_run(body: dict):
    s = {"prompt": body.get("prompt",""), "draft": "", "checklist": [], "report": [], "answer": ""}
    out = run_graph(a13_chain_of_verification.build_app, s)
    return {"answer": out.get("answer",""), "checklist": out.get("checklist",[]), "report": out.get("report",[])}

@app.get("/api/a14/run")
def a14_run():
    return a14_dry_run.run()

@app.post("/api/a15/run")
def a15_run(body: dict):
    s = {"prompt": body.get("prompt",""), "draft": "", "critique": "", "reward": 0.0, "answer": ""}
    out = run_graph(a15_rlhf_loop.build_app, s)
    return {"answer": out.get("answer",""), "draft": out.get("draft",""), "critique": out.get("critique",""), "reward": out.get("reward", 0)}

@app.post("/api/a16/run")
def a16_run(body: dict):
    s = {"task": body.get("task",""), "world": "", "actions": [], "feedbacks": [], "answer": ""}
    out = run_graph(a16_simulator_mitl.build_app, s)
    return {"answer": out.get("answer",""), "world": out.get("world",""), "actions": out.get("actions",[]), "feedbacks": out.get("feedbacks",[])}

@app.post("/api/a17/run")
def a17_run(body: dict):
    s = {"prompt": body.get("prompt",""), "answer": "", "meta": {}}
    out = run_graph(a17_reflexive_metacognitive.build_app, s)
    return {"answer": out.get("answer",""), "meta": out.get("meta",{})}

@app.post("/api/a05/run")
def a05_run(body: dict):
    s = {"task": body.get("task",""), "plan":"", "result":"", "verdict":"", "answer":""}
    out = run_graph(a05_pev.build_app, s)
    return {"answer": out.get("answer",""), "plan": out.get("plan",""), "result": out.get("result",""), "verdict": out.get("verdict","")}

@app.post("/api/a06/run")
def a06_run(body: dict):
    s = {"task": body.get("task",""), "board": [], "answer": ""}
    out = run_graph(a06_blackboard.build_app, s)
    return {"answer": out.get("answer",""), "board": out.get("board",[])}

@app.post("/api/a08/run")
def a08_run(body: dict):
    s = {"text": body.get("text",""), "graph": {}, "question": body.get("question",""), "answer": ""}
    out = run_graph(a08_graph_memory.build_app, s)
    return {"answer": out.get("answer",""), "graph": out.get("graph",{})}
