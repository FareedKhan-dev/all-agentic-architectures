from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from backend.core.llm import make_llm

class S(TypedDict):
    prompt: str
    candidates: List[str]
    votes: Dict[str,int]
    best: str
    answer: str

def n_generate(s:S):
    llm = make_llm()
    cand: List[str] = []
    for i in range(3):
        cand.append(llm.invoke(f"Propose une réponse candidate {i+1} pour:\n{s['prompt']}").content)
    return {"candidates": cand}

def n_judge(s:S):
    llm = make_llm()
    scores: List[float] = []
    for c in s["candidates"]:
        sc = llm.invoke(
            "Note sur 0..10 cette réponse (renvoie uniquement le nombre).\n" + c
        ).content.strip()
        try:
            scores.append(float(sc.replace(",", ".")))
        except Exception:
            scores.append(5.0)
    if not scores:
        return {"votes": {}, "best": "", "answer": ""}
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best = s["candidates"][best_idx]
    votes = {f"cand{i+1}": int(scores[i]) for i in range(len(scores))}
    ans = llm.invoke(
        f"Fusionne/Améliore la meilleure réponse (score {scores[best_idx]}):\n{best}"
    ).content
    return {"votes": votes, "best": best, "answer": ans}

def build_app():
    g = StateGraph(S)
    g.add_node("gen", n_generate)
    g.add_node("judge", n_judge)
    g.set_entry_point("gen")
    g.add_edge("gen", "judge")
    g.add_edge("judge", END)
    return g.compile()
