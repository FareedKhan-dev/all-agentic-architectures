from typing import TypedDict
from backend.core.llm import make_llm
from backend.core.rag import retriever
from backend.core.tools import web_search_tool


class In(TypedDict):
    query: str


class Out(TypedDict):
    route: str
    answer: str


def run(query: str) -> Out:
    q = query.lower()
    if any(
        k in q
        for k in [
            "cherche",
            "news",
            "wwdc",
            "en temps réel",
            "aujourd'hui",
            "dernières",
        ]
    ):
        tool = web_search_tool(3)
        hits = tool.run(query)
        llm = make_llm()
        ans = llm.invoke(
            f"Question: {query}\nRésultats web: {hits}\nSynthèse brève."
        ).content
        return {"route": "tool", "answer": ans}
    if any(k in q for k in ["document", "pdf", "corpus", "rag", "source"]):
        docs = retriever(k=4).get_relevant_documents(query)
        ctx = "\n\n".join(d.page_content[:600] for d in docs)
        llm = make_llm()
        ans = llm.invoke(
            f"Question: {query}\nContexte extrait:\n{ctx}\nRéponds brièvement."
        ).content
        return {"route": "rag", "answer": ans}
    # défaut: reflection
    llm = make_llm()
    ans = llm.invoke(f"Rédige une réponse claire et rigoureuse à: {query}").content
    return {"route": "reflection", "answer": ans}
