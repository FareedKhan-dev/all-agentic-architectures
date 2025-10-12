from typing import TypedDict, List, Any, Dict
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm


class S(TypedDict):
    task: str
    breadth: int
    depth: int
    frontier: List[Dict[str, Any]]
    best: Dict[str, Any]
    trace: List[Dict[str, Any]]
    answer: str


def _propose_children(task: str, node_text: str, breadth: int) -> List[str]:
    llm = make_llm()
    prompt = (
        "Tu es un agent qui explore plusieurs pistes de raisonnement.\n"
        f"Tâche: {task}\n"
        f"Contexte courant:\n{node_text}\n\n"
        f"Propose {breadth} prolongements (courts, clairs, numérotés 1..{breadth})."
    )
    out = llm.invoke(prompt).content
    children = [line.strip(" -") for line in out.splitlines() if line.strip()]
    if len(children) > breadth:
        children = children[:breadth]
    return children


def _score(node_text: str) -> float:
    llm = make_llm()
    prompt = (
        "Note la promesse de cette piste sur 0..10 (juste le nombre).\n"
        f"Piste:\n{node_text}\n"
    )
    out = llm.invoke(prompt).content.strip()
    try:
        return float(out.split()[0].replace(",", "."))
    except Exception:
        return 5.0


def n_init(s: S):
    start = {"path": [], "text": "Départ", "score": 0.0}
    return {"frontier": [start], "trace": [{"event": "init", "node": start}]}


def n_expand_layer(s: S):
    task = s["task"]
    breadth = s["breadth"]
    new_frontier: List[Dict[str, Any]] = []
    trace = s.get("trace", [])
    for node in s["frontier"]:
        children = _propose_children(task, node["text"], breadth)
        for idx, ch in enumerate(children, start=1):
            t = {"path": node["path"] + [idx], "text": ch, "score": _score(ch)}
            new_frontier.append(t)
            trace.append({"event": "child", "parent": node, "child": t})
    new_frontier.sort(key=lambda x: x["score"], reverse=True)
    k = min(breadth, len(new_frontier))
    return {"frontier": new_frontier[:k], "trace": trace}


def n_last_answer(s: S):
    llm = make_llm()
    best = s["frontier"][0] if s["frontier"] else {"text": "(vide)", "score": 0}
    task = s["task"]
    ans = llm.invoke(
        f"Tâche: {task}\nPiste finale (meilleur score {best.get('score', 0)}):\n{best.get('text', '')}\n\n"
        "Propose une réponse claire et structurée, en expliquant brièvement le raisonnement."
    ).content
    return {"answer": ans, "best": best}


def build_app():
    g = StateGraph(S)
    g.add_node("init", n_init)
    g.add_node("expand", n_expand_layer)
    g.add_node("final", n_last_answer)
    g.set_entry_point("init")
    g.add_edge("init", "expand")
    g.add_edge("expand", "expand")
    g.add_edge("expand", "final")
    g.add_edge("final", END)
    return g.compile()
