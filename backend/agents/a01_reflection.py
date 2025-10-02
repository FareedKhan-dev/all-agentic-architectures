from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm


class S(TypedDict):
    prompt: str
    criteria: List[str]
    draft: str
    review: str
    answer: str


def n_generate(s: S):
    llm = make_llm()
    draft = llm.invoke(
        f"Génère un premier brouillon pédagogique:\n\n{s['prompt']}"
    ).content
    return {"draft": draft}


def n_review(s: S):
    llm = make_llm()
    crit = "\n".join(s.get("criteria") or ["justesse", "clarté", "rigueur"])
    review = llm.invoke(
        f"Critique le texte selon ces critères:\n{crit}\n\nTEXTE:\n{s['draft']}\n\n"
        "Signale erreurs, manques, et propose des corrections ciblées."
    ).content
    return {"review": review}


def n_rewrite(s: S):
    llm = make_llm()
    ans = llm.invoke(
        f"Réécris le texte en corrigeant d'après la critique suivante:\n{s['review']}\n\nTEXTE:\n{s['draft']}\n\n"
        "Donne une version finale claire et structurée."
    ).content
    return {"answer": ans}


def build_app():
    g = StateGraph(S)
    g.add_node("generate", n_generate)
    g.add_node("review", n_review)
    g.add_node("rewrite", n_rewrite)
    g.set_entry_point("generate")
    g.add_edge("generate", "review")
    g.add_edge("review", "rewrite")
    g.add_edge("rewrite", END)
    return g.compile()
