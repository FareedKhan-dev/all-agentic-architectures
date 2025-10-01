from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm


class S(TypedDict):
    prompt: str
    draft: str
    checklist: List[str]
    report: List[str]
    answer: str


def n_draft(s: S):
    llm = make_llm()
    draft = llm.invoke(f"Rédige un premier brouillon:\n{s['prompt']}").content
    return {"draft": draft}


def n_checklist(s: S):
    llm = make_llm()
    ck = llm.invoke(
        "Dresse une checklist de vérification (3-5 points) pour ce brouillon:\n" + s["draft"]
    ).content
    lines = [l.strip(" -") for l in ck.splitlines() if l.strip()][:5]
    return {"checklist": lines}


def n_verify(s: S):
    llm = make_llm()
    rep: List[str] = []
    for item in s["checklist"]:
        rep.append(
            llm.invoke(
                f"Vérifie le point: {item}\nTexte:\n{s['draft']}\nSignale erreurs et propose fix."
            ).content
        )
    return {"report": rep}


def n_rewrite(s: S):
    llm = make_llm()
    report = "\n\n".join(s["report"])
    ans = llm.invoke(
        f"Corrige/réécris le brouillon selon ce rapport:\n{report}\n\nBrouillon:\n{s['draft']}"
    ).content
    return {"answer": ans}


def build_app():
    g = StateGraph(S)
    g.add_node("draft", n_draft)
    g.add_node("check", n_checklist)
    g.add_node("verify", n_verify)
    g.add_node("rewrite", n_rewrite)
    g.set_entry_point("draft")
    g.add_edge("draft", "check")
    g.add_edge("check", "verify")
    g.add_edge("verify", "rewrite")
    g.add_edge("rewrite", END)
    return g.compile()
