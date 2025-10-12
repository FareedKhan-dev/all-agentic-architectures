from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm


class S(TypedDict):
    task: str
    board: List[str]
    answer: str


def n_generate(s: S):
    llm = make_llm()
    out = llm.invoke(
        "Génère 2 contributions distinctes et complémentaires (lignes séparées) pour répondre à la tâche suivante:\n"
        f"{s['task']}"
    ).content
    ideas = [line.strip() for line in out.splitlines() if line.strip()]
    return {"board": ideas[:2]}


def n_summarize(s: S):
    llm = make_llm()
    ans = llm.invoke(
        f"Fusionne ces contributions en réponse claire:\n{s['board']}"
    ).content
    return {"answer": ans}


def build_app():
    g = StateGraph(S)
    g.add_node("generate", n_generate)
    g.add_node("summarize", n_summarize)
    g.set_entry_point("generate")
    g.add_edge("generate", "summarize")
    g.add_edge("summarize", END)
    return g.compile()
