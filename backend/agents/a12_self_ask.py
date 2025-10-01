from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm


class S(TypedDict):
    query: str
    subs: List[str]
    partials: List[str]
    answer: str


def n_subs(s: S):
    llm = make_llm()
    out = llm.invoke(
        f"Décompose la question en 2-3 sous-questions numérotées:\n{s['query']}"
    ).content
    subs = [l.strip(" -") for l in out.splitlines() if l.strip()][:3]
    return {"subs": subs}


def n_answer_parts(s: S):
    llm = make_llm()
    parts: List[str] = []
    for sub in s["subs"]:
        parts.append(llm.invoke(f"Réponds brièvement à: {sub}").content)
    return {"partials": parts}


def n_combine(s: S):
    llm = make_llm()
    ctx = "\n".join(f"- {p}" for p in s["partials"])
    ans = llm.invoke(
        f"Question: {s['query']}\nÉléments:\n{ctx}\nDonne une réponse finale structurée."
    ).content
    return {"answer": ans}


def build_app():
    g = StateGraph(S)
    g.add_node("subs", n_subs)
    g.add_node("parts", n_answer_parts)
    g.add_node("combine", n_combine)
    g.set_entry_point("subs")
    g.add_edge("subs", "parts")
    g.add_edge("parts", "combine")
    g.add_edge("combine", END)
    return g.compile()
