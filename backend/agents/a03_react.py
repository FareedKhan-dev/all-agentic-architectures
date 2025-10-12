from typing import TypedDict, Any, List
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm
from backend.core.tools import web_search_tool


class S(TypedDict):
    task: str
    trace: List[str]
    obs: Any
    answer: str
    steps: int


def n_think(s: S):
    llm = make_llm()
    out = llm.invoke(
        "Tu es un agent ReAct. Propose l'action suivante (TOOL: web_search | FIN) "
        f"avec une requête concise si TOOL.\nTâche: {s['task']}\nObservations: {s.get('obs')}"
    ).content
    trace = s.get("trace", []) + [out]
    return {"trace": trace}


def n_act(s: S):
    tool = web_search_tool(3)
    query = s["trace"][-1]
    hits = tool.run(query)
    return {"obs": hits}


def n_answer(s: S):
    llm = make_llm()
    ans = llm.invoke(
        f"Réponds en t'appuyant sur la trace et les observations.\nTRACE:\n{s['trace']}\nOBS:\n{s.get('obs')}"
    ).content
    return {"answer": ans}


def build_app():
    g = StateGraph(S)
    g.add_node("think", n_think)
    g.add_node("act", n_act)
    g.add_node("answer", n_answer)
    g.set_entry_point("think")
    g.add_edge("think", "act")
    g.add_edge("act", "think")
    g.add_edge("think", "answer")
    g.add_edge("answer", END)
    return g.compile()
