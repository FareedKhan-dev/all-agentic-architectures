from typing import TypedDict, List, Dict
from backend.agents import a01_reflection


class Out(TypedDict):
    cases: List[Dict[str, str]]


def run() -> Out:
    prompts = [
        "Explique la loi des grands nombres simplement.",
        "Décris l’algorithme de Dijkstra.",
        "Qu’est-ce qu’un graphe biparti ?",
    ]
    app = a01_reflection.build_app()
    cases: List[Dict[str, str]] = []
    for p in prompts:
        s = {"prompt": p, "criteria": [], "draft": "", "review": "", "answer": ""}
        out = app.invoke(s)
        cases.append({"prompt": p, "answer": out.get("answer", "")})
    return {"cases": cases}
