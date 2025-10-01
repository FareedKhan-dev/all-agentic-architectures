from typing import TypedDict
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm


class S(TypedDict):
    prompt: str
    draft: str
    critique: str
    reward: float
    answer: str


def n_draft(s: S):
    llm = make_llm()
    d = llm.invoke(f"Réponds au mieux à:\n{s['prompt']}").content
    return {"draft": d}


def n_critique(s: S):
    llm = make_llm()
    c = llm.invoke(f"Critique sévère et propose corrections:\n{s['draft']}").content
    r = llm.invoke("Note (1..10) la qualité de la réponse précédente (juste le nombre).").content.strip()
    try:
        rw = float(r.replace(",", "."))
    except Exception:
        rw = 6.0
    return {"critique": c, "reward": rw}


def n_refine(s: S):
    llm = make_llm()
    a = llm.invoke(
        f"Réécris la réponse selon cette critique (viser score>=8):\n{s['critique']}\n\nRéponse initiale:\n{s['draft']}"
    ).content
    return {"answer": a}


def build_app():
    g = StateGraph(S)
    g.add_node("draft", n_draft)
    g.add_node("crit", n_critique)
    g.add_node("refine", n_refine)
    g.set_entry_point("draft")
    g.add_edge("draft", "crit")
    g.add_edge("crit", "refine")
    g.add_edge("refine", END)
    return g.compile()
