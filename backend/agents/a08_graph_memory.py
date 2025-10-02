from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm


class S(TypedDict):
    text: str
    graph: Dict[str, List[str]]
    question: str
    answer: str


def n_extract(s: S):
    llm = make_llm()
    prompt = (
        "Extrait un petit graphe entités->relations sous forme de couples 'A -> B' ligne par ligne.\n"
        f"Texte:\n{s['text']}\n"
    )
    out = llm.invoke(prompt).content
    g: Dict[str, List[str]] = {}
    for line in out.splitlines():
        if "->" in line:
            a, b = [x.strip(" -:") for x in line.split("->", 1)]
            g.setdefault(a, []).append(b)
    return {"graph": g}


def n_answer(s: S):
    llm = make_llm()
    ctx = "\n".join(f"{k} -> {', '.join(v)}" for k, v in s.get("graph", {}).items())
    ans = llm.invoke(
        f"Question: {s['question']}\nGraphe:\n{ctx}\nDonne une réponse concise, en t'appuyant sur les relations."
    ).content
    return {"answer": ans}


def build_app():
    g = StateGraph(S)
    g.add_node("extract", n_extract)
    g.add_node("answer", n_answer)
    g.set_entry_point("extract")
    g.add_edge("extract", "answer")
    g.add_edge("answer", END)
    return g.compile()
