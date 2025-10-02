from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm


class S(TypedDict):
    task: str
    world: str
    actions: List[str]
    feedbacks: List[str]
    answer: str


def n_world(s: S):
    llm = make_llm()
    w = llm.invoke(f"Imagine un environnement succinct pour:\n{s['task']}").content
    return {"world": w, "actions": [], "feedbacks": []}


def n_act1(s: S):
    llm = make_llm()
    a = llm.invoke(
        f"Dans ce monde:\n{s['world']}\nPropose une première action."
    ).content
    f = llm.invoke(f"Feedback sur l'action:\n{a}").content
    return {"actions": [a], "feedbacks": [f]}


def n_act2(s: S):
    llm = make_llm()
    a2 = llm.invoke(
        f"Compte tenu du feedback précédent {s['feedbacks'][-1]}, propose une seconde action."
    ).content
    f2 = llm.invoke(f"Feedback sur l'action:\n{a2}").content
    acts = s["actions"] + [a2]
    fbs = s["feedbacks"] + [f2]
    return {"actions": acts, "feedbacks": fbs}


def n_final(s: S):
    llm = make_llm()
    ans = llm.invoke(
        f"Tâche: {s['task']}\nMonde:\n{s['world']}\nActions:{s['actions']}\nFeedback:{s['feedbacks']}\nConclusion et recommandations."
    ).content
    return {"answer": ans}


def build_app():
    g = StateGraph(S)
    g.add_node("world", n_world)
    g.add_node("act1", n_act1)
    g.add_node("act2", n_act2)
    g.add_node("final", n_final)
    g.set_entry_point("world")
    g.add_edge("world", "act1")
    g.add_edge("act1", "act2")
    g.add_edge("act2", "final")
    g.add_edge("final", END)
    return g.compile()
