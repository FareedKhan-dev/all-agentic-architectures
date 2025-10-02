from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm
from backend.core.tools import web_search_tool


class S(TypedDict):
    question: str
    hits: List[dict]
    sources: List[str]
    answer: str


def n_search(s: S):
    tool = web_search_tool(4)
    hits = tool.run(s["question"])  # liste de dicts {title,url,content,...}
    return {"hits": hits, "sources": [h.get("url", "") for h in hits]}


def n_answer(s: S):
    llm = make_llm()
    prompt = (
        f"Question: {s['question']}\n"
        f"Résultats web (JSON): {s['hits']}\n\n"
        "Rédige une réponse structurée en français et cite les URLs en fin de réponse."
    )
    ans = llm.invoke(prompt).content
    return {"answer": ans}


def build_app():
    g = StateGraph(S)
    g.add_node("search", n_search)
    g.add_node("answer", n_answer)
    g.set_entry_point("search")
    g.add_edge("search", "answer")
    g.add_edge("answer", END)
    return g.compile()
