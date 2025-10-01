from typing import TypedDict
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm

class S(TypedDict):
    task: str
    plan: str
    result: str
    verdict: str
    answer: str

def n_plan(s:S):
    llm = make_llm()
    plan = llm.invoke(
        f"Élabore un plan numéroté pour réaliser la tâche suivante:\n{s['task']}"
    ).content
    return {"plan": plan}

def n_execute(s:S):
    llm = make_llm()
    result = llm.invoke(
        f"En suivant le plan ci-dessous, exécute mentalement la tâche et propose un résultat concis.\nPLAN:\n{s['plan']}"
    ).content
    return {"result": result}

def n_verify(s:S):
    llm = make_llm()
    verdict = llm.invoke(
        "Vérifie de façon critique le résultat. Énumère erreurs potentielles et corrige si nécessaire.\n"
        f"PLAN:\n{s['plan']}\n\nRESULTAT:\n{s['result']}"
    ).content
    return {"verdict": verdict, "answer": verdict}

def build_app():
    g = StateGraph(S)
    g.add_node("plan", n_plan)
    g.add_node("execute", n_execute)
    g.add_node("verify", n_verify)
    g.set_entry_point("plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "verify")
    g.add_edge("verify", END)
    return g.compile()
