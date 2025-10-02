from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm
from backend.core.rag import retriever


class S(TypedDict):
    question: str
    episodic: List[str]
    semantic: List[str]
    answer: str


def n_retrieve_semantic(s: S):
    """Récupère des extraits sémantiques via RAG et les met au format texte."""
    try:
        ret = retriever(k=5)
        docs = ret.get_relevant_documents(s["question"])
        sem = []
        for d in docs:
            src = d.metadata.get("source", "?")
            sem.append(f"[SRC] {src}\n{d.page_content}")
        return {"semantic": sem}
    except Exception:
        return {"semantic": []}


def n_generate_episodic(s: S):
    """Simule des souvenirs/observations épisodiques pertinents (3 lignes)."""
    llm = make_llm()
    prompt = (
        "Tu disposes d'une mémoire épisodique: 3 observations/souvenirs pertinents (courts).\n"
        f"Question: {s['question']}\n"
        "Donne 3 items (une ligne chacun)."
    )
    out = llm.invoke(prompt).content
    epi = [line.strip() for line in out.splitlines() if line.strip()]
    return {"episodic": epi[:3]}


def n_answer(s: S):
    llm = make_llm()
    ctx = "\n\n".join((s.get("episodic", []) or []) + (s.get("semantic", []) or []))
    ans = llm.invoke(
        f"Question: {s['question']}\nContexte (épisodique + sémantique):\n{ctx}\n"
        "Réponds de façon structurée et cite les sources sémantiques si présentes."
    ).content
    return {"answer": ans}


def build_app():
    g = StateGraph(S)
    g.add_node("retrieve_semantic", n_retrieve_semantic)
    g.add_node("generate_episodic", n_generate_episodic)
    g.add_node("answer", n_answer)
    g.set_entry_point("retrieve_semantic")
    g.add_edge("retrieve_semantic", "generate_episodic")
    g.add_edge("generate_episodic", "answer")
    g.add_edge("answer", END)
    return g.compile()
