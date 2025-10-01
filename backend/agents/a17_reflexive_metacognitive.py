from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm


class S(TypedDict):
    prompt: str
    answer: str
    meta: Dict[str, str]


def n_answer(s: S):
    llm = make_llm()
    a = llm.invoke(f"Réponds rigoureusement à:\n{s['prompt']}").content
    return {"answer": a}


def n_meta(s: S):
    llm = make_llm()
    m = llm.invoke(
        "Auto-évalue-toi en JSON avec clés {uncertainty:%, limitations:list, unknowns:list} sur ta dernière réponse:\n"
        f"{s['answer']}"
    ).content
    return {"meta": {"raw": m}}


def build_app():
    g = StateGraph(S)
    g.add_node("answer", n_answer)
    g.add_node("meta", n_meta)
    g.set_entry_point("answer")
    g.add_edge("answer", "meta")
    g.add_edge("meta", END)
    return g.compile()
