from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm


class S(TypedDict):
    task: str
    plan: List[str]
    answer: str


def n_plan(s: S):
    llm = make_llm()
    out = llm.invoke(
        f"Planifie en étapes concrètes et ordonnées la tâche suivante (liste numérotée):\n{s['task']}"
    ).content
    steps = [x.strip() for x in out.splitlines() if x.strip()]
    return {"plan": steps}


def n_summarize(s: S):
    llm = make_llm()
    ans = llm.invoke(
        f"Résume le plan suivant en un paragraphe clair, avec risques et prérequis:\n{s['plan']}"
    ).content
    return {"answer": ans}


def build_app():
    g = StateGraph(S)
    g.add_node("plan", n_plan)
    g.add_node("summarize", n_summarize)
    g.set_entry_point("plan")
    g.add_edge("plan", "summarize")
    g.add_edge("summarize", END)
    return g.compile()
